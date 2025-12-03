import copy
import logging
import os
import torch
from torch import nn, Tensor
from torch.nn import Sequential, Linear, ReLU
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from quadprog import solve_qp
import numpy as np

from data import NNDataset, split_data_to_training_and_validation
from utils import get_device


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for classification tasks.

    The model layers consist of:
        - An input layer of size `input_dim`
        - A hidden layer with 512 neurons and ReLU activation
        - An output layer with `output_dim` neurons

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output classes.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super(MLP, self).__init__()
        self.layers = Sequential(
            Linear(input_dim, 512),
            ReLU(),
            Linear(512, output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the model layers.

        Args:
            inputs (Tensor): Input tensor of shape `(batch_size, input_dim)`.

        Returns:
            Tensor: Output tensor of shape `(batch_size, output_dim)`.
        """
        outputs = self.layers(inputs)
        return outputs


class BiasLayer(nn.Module):
    """
    A learnable bias layer that applies a linear transformation (scaling and shifting)
    to a specific subset of the input tensor's features.

    The transformation applied is:
        `output[:, lower_bound:upper_bound] = alpha * input + beta`

    where:
        - `alpha` is a learnable scaling parameter (initialized to 1.0)
        - `beta` is a learnable shift parameter (initialized to 0.0)
    """

    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, inputs: Tensor, lower_bound: int, upper_bound: int) -> Tensor:
        """
        Applies the bias transformation to the specified slice of the input tensor.

        Args:
            inputs (Tensor): Input tensor of shape (batch_size, num_classes).
            lower_bound (int): Starting index (inclusive) of the class logits to transform.
            upper_bound (int): Ending index (exclusive) of the class logits to transform.

        Returns:
            Tensor: Transformed output tensor of the same shape as `inputs`, with
            the selected range scaled and shifted by `alpha` and `beta`.
        """
        ret_x = inputs.clone()
        alpha = self.alpha.to(inputs.device)
        beta = self.beta.to(inputs.device)
        ret_x[:, lower_bound:upper_bound] = alpha * inputs[:, lower_bound:upper_bound] + beta
        return ret_x

    def get_params(self):
        """
        Gets the bias layer parameter values.

        Returns:
            A tuple containing bias layer parameter values (alpha, beta)
        """
        return self.alpha.item(), self.beta.item()


class BaseNetwork(nn.Module):
    """
    Represents a trainable neural network for classification.

    This class manages model training, inference, serialization, and deserialization.

    Args:
        n_classes (int): The number of output classes.
        dimensionality (int): Input feature dimension.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
    """

    def __init__(self, n_classes: int, dimensionality: int, epochs: int, *args, **kwargs):
        super(BaseNetwork, self).__init__(*args, **kwargs)

        self.device = get_device()
        self.dimensionality = dimensionality

        # Model and its hyperparameters
        self.model = None
        if n_classes > 0:
            self.model = MLP(input_dim=dimensionality, output_dim=n_classes).to(self.device)
        self.epochs = epochs

    def expand_classifier(self, total_classes: int) -> bool:
        return False

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Performs a forward pass through the internal MLP model.

        Args:
            inputs (Tensor): Input tensor of shape `(batch_size, input_dim)`.

        Returns:
            Tensor: Output logits of shape `(batch_size, output_dim)`.
        """
        return self.model(inputs)


class ExpandableNetwork(BaseNetwork):
    def expand_classifier(self, total_classes: int) -> bool:
        """
        Expands the output (classifier) layer of the NN model to accommodate additional classes
        while preserving previously learned weights.

        Args:
            total_classes (int): The total number of output classes after expansion.
        """
        if self.model is None:
            self.model = MLP(input_dim=self.dimensionality, output_dim=total_classes).to(self.device)
            return True

        old_classifier: Linear = self.model.layers[-1]
        current_classes = old_classifier.out_features

        if total_classes <= current_classes:
            logging.info(f"No expansion needed. Current output classes: {current_classes}, Requested: {total_classes}")
            return False
        logging.info(f"Expanding output layer from {current_classes} to {total_classes} classes.")

        new_classifier = Linear(old_classifier.in_features, total_classes).to(self.device)
        with torch.no_grad():
            new_classifier.weight[:current_classes] = old_classifier.weight[:current_classes]
            new_classifier.bias[:current_classes] = old_classifier.bias[:current_classes]
        self.model.layers[-1] = new_classifier
        return True


class ExpandableNetworkWithBiasLayer(ExpandableNetwork):
    """
    Extends the ExpandableNetwork by adding task-specific bias layers.

    Bias correction layers allow for adjusting logits for specific class ranges, useful in methods
    like BiC (Bias Correction) where old and new class logits may become imbalanced over time.

    Args:
        bias_correction (bool): Whether to enable bias correction during forward passes.
    """

    def __init__(
            self,
            n_classes: int,
            dimensionality: int,
            epochs: int,
            bias_correction=False,
            *args,
            **kwargs
    ):
        super().__init__(n_classes, dimensionality, epochs, *args, **kwargs)
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Performs a forward pass through the network, applying bias correction layers
        to class-specific logit ranges if enabled.

        Args:
            inputs (Tensor): Input tensor of shape `(batch_size, input_dim)`.

        Returns:
            Tensor: Output logits with optional bias correction applied.
        """
        outputs = super().forward(inputs)
        if self.bias_correction:
            for i, layer in enumerate(self.bias_layers):
                outputs = layer(
                    outputs, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
        return outputs

    def expand_classifier(self, total_classes: int) -> bool:
        """
        Expands the classifier and appends a new bias correction layer corresponding
        to the newly added class range.

        Args:
            total_classes (int): The new total number of output classes.
        """
        if not super().expand_classifier(total_classes):
            return False
        new_task_size = total_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())
        return True

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            assert isinstance(layer, BiasLayer)
            params.append(layer.get_params())
        return params


BASE_MOMENTUM = 0.9
BASE_WEIGHT_DECAY = 0


# The baseline - with no class expansion, supports data replay
class BaseLearner:
    """
    A general learner class for training and evaluating neural networks.

    This class provides shared training, evaluation, and prediction logic for all learner types.
    It can be subclassed for specific continual learning strategies such as
    incremental learning or bias correction.

    Args:
        dimensionality (int): The input feature dimensionality.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        network_type (str): Type of network architecture to use. Default is "basic".
        n_classes (int): Total number of output classes. Default is 1000.
    """

    def __init__(
            self,
            dimensionality,
            epochs,
            lr,
            show_loss,
            network_type="basic",
            n_classes=1000,
            exponential_lr_decay=False,
            gamma=0.0
    ):
        self._current_task = -1
        self._known_classes = 0
        self._total_classes = 0

        # Replay buffer
        self._memory_X = None
        self._memory_y = None

        # Create neurons for all future classes at once
        self.network = create_nn(dimensionality, epochs, network_type, n_classes)
        self._validate_network_type()

        # Hyperparameters for the learner
        self.lr = lr
        self._configure_optimizer()
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self.show_loss = show_loss

        self.exponential_lr_decay = exponential_lr_decay
        self.gamma = gamma
        self._seen_examples = 0

    def get_num_known_classes(self) -> int:
        return self._known_classes

    def get_replay_buffer(self) -> tuple[Tensor, Tensor]:
        return self._memory_X, self._memory_y

    def set_replay_buffer(self, X: Tensor, y: Tensor) -> None:
        self._memory_X = X
        self._memory_y = y

    def _configure_optimizer(self) -> None:
        """
        Configures an optimizer for the network. Usually called in init and after
        expanding the network classifier.
        """
        assert self.network is not None
        self._optimizer = torch.optim.SGD(
            self.network.model.parameters(),
            lr=self.lr,
            momentum=BASE_MOMENTUM,
            weight_decay=BASE_WEIGHT_DECAY
        )

    def _after_task(self):
        self._known_classes = self._total_classes

    def _validate_network_type(self) -> None:
        """
        Validates that the current learner is using a compatible BaseNetwork subclass.

        This method should be overridden by subclasses to enforce specific types.
        """
        assert isinstance(self.network, BaseNetwork)

    def expand_network_classifier(self, total_classes: int) -> None:
        pass

    def train_network(self, X: Tensor, y: Tensor) -> None:
        """
        Trains the neural network on a given dataset (task).

        Args:
            X (Tensor): Training feature embeddings of shape `(N, D)`.
            y (Tensor): Training labels of shape `(N,)`.
        """
        assert self.network.dimensionality == X.shape[1]
        X, y = X.to(self.network.device), y.to(self.network.device)

        task_size = int(torch.unique(y).numel())
        self._current_task += 1
        self._total_classes = self._known_classes + task_size

        self.expand_network_classifier(self._total_classes)

        if self._memory_X is not None and self._memory_y is not None:
            mem_X, mem_y = self._memory_X.to(self.network.device), self._memory_y.to(self.network.device)
            X = torch.cat([X, mem_X])
            y = torch.cat([y, mem_y])

        train_loader = DataLoader(
            dataset=NNDataset(X, y), batch_size=256, shuffle=True
        )
        self.train()

        step = max(1, self.network.epochs // 10)
        logging.info(f"Epochs: {self.network.epochs}, step: {step}")

        for epoch in range(self.network.epochs):
            correct, total = 0, 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.network.device), y_batch.to(self.network.device)

                self._optimizer.zero_grad()
                outputs = self.network(X_batch)
                loss = self._loss_fn(outputs, y_batch)
                loss.backward()
                self._optimizer.step()

                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == y_batch).sum()
                total += y_batch.size(0)

                if self.show_loss and epoch % step == 0 and epoch != 0:
                    logging.info(f"Epoch {epoch + 1} | Loss {loss.item():.5f}")

            accuracy = correct / total
            logging.info(f"Epoch {epoch + 1} | Train Accuracy: {accuracy:.2%}")

        self._seen_examples += X.shape[0]
        if self.exponential_lr_decay:
            decayed_lr = self.lr * (self.gamma ** self._seen_examples)
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = decayed_lr
            logging.info(f"Updated LR after task {self._current_task + 1}: {decayed_lr:.6e}")

        self._after_task()

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def compute_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute per-sample CrossEntropy loss (no reduction)."""
        assert self.network is not None, "Network is not initialized."

        self.eval()
        with torch.no_grad():
            logits = self.network(X)
            criterion = nn.CrossEntropyLoss(reduction="none")
            loss = criterion(logits, y)

        return loss

    def evaluate_network(
            self,
            X: Tensor,
            y: Tensor,
    ) -> float:
        """
        Evaluates a trained neural network on a test dataset.

        Args:
            X (Tensor): Test feature embeddings of shape `(N, D)`.
            y (Tensor): True test labels of shape `(N,)`.

        Returns:
            float: The classification accuracy of the network.
        """
        device = get_device()
        X, y = X.to(device), y.to(device)

        correct, total = 0, X.shape[0]
        for i, query in enumerate(X):
            correct += 1 if self._search(query) == y[i] else 0
        accuracy = correct / total
        logging.info(f"Network | Test Accuracy: {accuracy:.2%}")

        return accuracy

    def _search(self, query: Tensor) -> int:
        """
        Predicts a class label for a given query.

        Args:
            query (Tensor): A single input tensor of shape `(D,)`.

        Returns:
            int: The predicted class label.
        """
        return int(self._predict(query, 1)[1].item())

    def _predict(self, X: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
        """
        Predicts the top-k class labels for a set of input samples.

        Args:
            X (Tensor): Input tensor of shape `(N, D)`.
            top_k (int): Number of top predictions to return.

        Returns:
            tuple (tuple[Tensor, Tensor]): Probabilities of the top-k predictions and the corresponding class labels of the top-k predictions.
        """
        assert self.network is not None, "Network is not initialized."

        self.eval()
        with torch.no_grad():
            X = X.to(self.network.device)
            if X.dim() == 1:
                X = X.unsqueeze(0)
            logits = self.network(X)

        probs = softmax(logits, dim=1)
        probabilities, classes = probs.topk(top_k, dim=1)

        return probabilities.squeeze(0), classes.squeeze(0)

    def serialize(self, filename: str) -> None:
        """
        Saves the current network state to a file.

        Args:
            filename (str): The path to save the network.
        """
        serialized = {
            "model_state": self.network.model.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
            "loss_fn": self._loss_fn.state_dict(),
        }
        torch.save(serialized, f"./networks/{filename}")
        logging.info(f"Network saved to ./networks/{filename}.")

    def deserialize(self, filename: str) -> None:
        """
        Loads a saved network state from a file.

        Args:
            filename (str): The path to the saved network file.
        """
        if not os.path.exists(f"./networks/{filename}"):
            logging.warning(f"Loading failed. No saved network {filename} found.")
            return

        serialized = torch.load(filename, map_location=self.network.device, weights_only=True)
        self.network.model.load_state_dict(serialized["model_state"])
        self._optimizer.load_state_dict(serialized["optimizer_state"])
        self._loss_fn.load_state_dict(serialized["loss_fn"])
        logging.info(f"Network loaded from ./networks/{filename}.")


# Supports class expansion
class ExpandableLearner(BaseLearner):
    """
    A learner class that supports classifier expansion for continual learning.

    This class uses ExpandableNetwork as its underlying network, which allows
    dynamically growing the classifier to support new classes over time.
    """

    def __init__(self, dimensionality, epochs, lr, show_loss, exponential_lr_decay, lr_decay_gamma):
        super().__init__(dimensionality, epochs, lr, show_loss, "expandable", 0, exponential_lr_decay, lr_decay_gamma)

    def _configure_optimizer(self) -> None:
        assert self.network is not None
        if self._total_classes > 0:
            self._optimizer = torch.optim.SGD(
                self.network.model.parameters(),
                lr=self.lr,
                momentum=BASE_MOMENTUM,
                weight_decay=BASE_WEIGHT_DECAY
            )

    def expand_network_classifier(self, total_classes: int) -> None:
        """
        Expands the network classifier layer to accommodate new classes.

        Args:
            total_classes (int): The total number of output classes after expansion.
        """
        expanded = self.network.expand_classifier(total_classes)

        # Reconfigure the optimizer for the updated network model
        if expanded:
            self._configure_optimizer()

    def _validate_network_type(self) -> None:
        assert isinstance(self.network, ExpandableNetwork)


# Like, incremental, but also supports additional bias layer
class ExpandableLearnerWithBiasLayer(BaseLearner):
    """
    A learner class that supports classifier expansion and bias correction.

    This class uses ExpandableNetworkWithBiasLayer, which maintains task-specific bias
    layers for correcting logits after each task in a class-incremental setup.
    """

    def __init__(self, dimensionality, epochs, lr, show_loss, exponential_lr_decay, lr_decay_gamma):
        super().__init__(dimensionality, epochs, lr, show_loss, "expandable-bias", 0, exponential_lr_decay, lr_decay_gamma)
        self.decayed_lr = lr

    def expand_network_classifier(self, total_classes: int) -> None:
        """
        Expands the network classifier layer to accommodate new classes.

        Args:
            total_classes (int): The total number of output classes after expansion.
        """
        expanded = self.network.expand_classifier(total_classes)
        assert expanded

        network = self._get_typed_network()
        ignored_params = list(map(id, network.bias_layers.parameters()))
        base_params = filter(
            lambda p: id(p) not in ignored_params, network.parameters()
        )
        network_params = [
            {"params": base_params, "lr": self.decayed_lr, "weight_decay": BASE_WEIGHT_DECAY},
            {
                "params": network.bias_layers.parameters(),
                "lr": 0,
                "weight_decay": 0,
            },
        ]

        self._configure_optimizer(network_params)

    def _configure_optimizer(self, *args) -> None:
        if len(args) > 0:
            self._optimizer = torch.optim.SGD(*args, lr=self.lr, momentum=BASE_MOMENTUM, weight_decay=BASE_WEIGHT_DECAY)

    def _validate_network_type(self) -> None:
        assert isinstance(self.network, ExpandableNetworkWithBiasLayer)

    def _get_typed_network(self) -> ExpandableNetworkWithBiasLayer:
        network = self.network
        assert isinstance(network, ExpandableNetworkWithBiasLayer)
        return network

    def train(self):
        network = self._get_typed_network()
        network.train()
        network.bias_correction = False

    def eval(self):
        network = self._get_typed_network()
        network.eval()
        network.bias_correction = True

    def bias_correction(self):
        network = self._get_typed_network()
        network.bias_correction = True

    def train_network(self, X: Tensor, y: Tensor) -> None:
        assert self.network.dimensionality == X.shape[1]
        X, y = X.to(self.network.device), y.to(self.network.device)

        task_size = int(torch.unique(y).numel())
        self._current_task += 1
        self._total_classes = self._known_classes + task_size

        self.expand_network_classifier(self._total_classes)

        if self._memory_X is not None and self._memory_y is not None:
            mem_X, mem_y = self._memory_X.to(self.network.device), self._memory_y.to(self.network.device)
            X = torch.cat([X, mem_X])
            y = torch.cat([y, mem_y])

        val_loader: DataLoader | None = None
        if self._current_task >= 1:
            X_train, y_train, X_val, y_val = split_data_to_training_and_validation(X, y)
            train_loader = DataLoader(
                dataset=NNDataset(X_train, y_train), batch_size=256, shuffle=True
            )
            val_loader = DataLoader(
                dataset=NNDataset(X_val, y_val), batch_size=256, shuffle=True
            )
        else:
            train_loader = DataLoader(
                dataset=NNDataset(X, y), batch_size=256, shuffle=True
            )

        self.train()
        self._training(train_loader)

        if self._current_task >= 1:
            assert val_loader is not None
            self.bias_correction()
            self._bias_correction(val_loader)

        self._seen_examples += X.shape[0]
        if self.exponential_lr_decay:
            self.decayed_lr = self.lr * (self.gamma ** self._seen_examples)
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = self.decayed_lr
            logging.info(f"Updated LR after task {self._current_task + 1}: {self.decayed_lr:.6e}")

        self._log_bias_params()
        self._after_task()

    def _run(self, loader: DataLoader, stage: str) -> None:
        step = max(1, self.network.epochs // 10)
        logging.info(f"Epochs: {self.network.epochs}, step: {step}, stage: {stage}")

        for epoch in range(self.network.epochs):
            correct, total = 0, 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.network.device), y_batch.to(self.network.device)
                outputs = self.network(X_batch)

                if stage == "training":
                    loss = self._loss_fn(outputs, y_batch)
                    predictions = torch.argmax(outputs, dim=1)
                    correct += (predictions == y_batch).sum()
                    total += y_batch.size(0)

                    if self.show_loss and epoch % step == 0 and epoch != 0:
                        logging.info(f"Epoch {epoch + 1} | Loss {loss.item():.5f}")
                elif stage == "bias_correction":
                    loss = self._loss_fn(outputs, y_batch)
                else:
                    raise ValueError("Invalid BiC stage")

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            if stage == "training":
                accuracy = correct / total
                logging.info(f"Epoch {epoch + 1} | Train Accuracy: {accuracy:.2%}")

    def _training(self, train_loader: DataLoader) -> None:
        network = self._get_typed_network()
        ignored_params = list(map(id, network.bias_layers.parameters()))
        base_params = filter(
            lambda p: id(p) not in ignored_params, network.parameters()
        )
        network_params = [
            {"params": base_params, "lr": self.decayed_lr, "weight_decay": BASE_WEIGHT_DECAY},
            {
                "params": network.bias_layers.parameters(),
                "lr": 0,
                "weight_decay": 0,
            },
        ]
        self._configure_optimizer(network_params)
        self._run(train_loader, "training")

    def _bias_correction(self, val_loader: DataLoader) -> None:
        network = self._get_typed_network()
        network_params = [
            {
                "params": network.bias_layers[-1].parameters(),
                "lr": 0.01,
                "weight_decay": BASE_WEIGHT_DECAY,
            }
        ]
        self._configure_optimizer(network_params)
        self._run(val_loader, "bias_correction")

    def _log_bias_params(self):
        logging.info(f"Parameters of bias layer after task {self._current_task + 1}:")
        network = self._get_typed_network()
        params = network.get_bias_params()
        for i, param in enumerate(params):
            logging.info(f"{i} => alpha: {param[0]}, beta: {param[1]}")


# Hyperparameters from the original paper tweaked to our experiment
# https://arxiv.org/pdf/2302.03648
BiC_MOMENTUM = 0.9
BiC_WEIGHT_DECAY = 0  # 1e-5
BiC_LR_DECAY = 0.8
BiC_MILESTONES = [10, 20, 27]
BiC_T = 2


# Knowledge distillation (logit distillation) + model rectify
class BiCLearner(ExpandableLearnerWithBiasLayer):
    def __init__(self, dimensionality, epochs, lr, show_loss, exponential_lr_decay, lr_decay_gamma):
        super().__init__(dimensionality, epochs, lr, show_loss, exponential_lr_decay, lr_decay_gamma)
        self.lamda = 0
        self._old_network = None

    def expand_network_classifier(self, total_classes: int) -> None:
        expanded = self.network.expand_classifier(total_classes)
        assert expanded

        network = self._get_typed_network()
        ignored_params = list(map(id, network.bias_layers.parameters()))
        base_params = filter(
            lambda p: id(p) not in ignored_params, network.parameters()
        )
        network_params = [
            {"params": base_params, "lr": self.decayed_lr, "weight_decay": BiC_WEIGHT_DECAY},
            {
                "params": network.bias_layers.parameters(),
                "lr": 0,
                "weight_decay": 0,
            },
        ]

        self._configure_optimizer(network_params)
        self._configure_scheduler()

    def _after_task(self):
        self._old_network = self.network.copy().freeze()
        self._known_classes = self._total_classes

    def _configure_optimizer(self, *args) -> None:
        if len(args) > 0:
            self._optimizer = torch.optim.SGD(*args, lr=self.lr, momentum=BiC_MOMENTUM, weight_decay=BiC_WEIGHT_DECAY)

    def _configure_scheduler(self) -> None:
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._optimizer, milestones=BiC_MILESTONES, gamma=BiC_LR_DECAY
        )

    def train_network(self, X: Tensor, y: Tensor) -> None:
        assert self.network.dimensionality == X.shape[1]
        X, y = X.to(self.network.device), y.to(self.network.device)

        task_size = int(torch.unique(y).numel())
        self._current_task += 1
        self._total_classes = self._known_classes + task_size

        self.expand_network_classifier(self._total_classes)

        if self._memory_X is not None and self._memory_y is not None:
            mem_X, mem_y = self._memory_X.to(self.network.device), self._memory_y.to(self.network.device)
            X = torch.cat([X, mem_X])
            y = torch.cat([y, mem_y])

        val_loader: DataLoader | None = None
        if self._current_task >= 1:
            X_train, y_train, X_val, y_val = split_data_to_training_and_validation(X, y)
            train_loader = DataLoader(
                dataset=NNDataset(X_train, y_train), batch_size=256, shuffle=True
            )
            val_loader = DataLoader(
                dataset=NNDataset(X_val, y_val), batch_size=256, shuffle=True
            )
            self.lamda = self._known_classes / self._total_classes
        else:
            train_loader = DataLoader(
                dataset=NNDataset(X, y), batch_size=256, shuffle=True
            )

        self.train()
        self._training(train_loader)

        if self._current_task >= 1:
            assert val_loader is not None
            self.bias_correction()
            self._bias_correction(val_loader)

        self._log_bias_params()
        self._after_task()

    def _run(self, loader: DataLoader, stage: str) -> None:
        step = max(1, self.network.epochs // 10)
        logging.info(f"Epochs: {self.network.epochs}, step: {step}, stage: {stage}")

        for epoch in range(self.network.epochs):
            correct, total = 0, 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.network.device), y_batch.to(self.network.device)
                logits = self.network(X_batch)

                if stage == "training":
                    clf_loss = self._loss_fn(logits, y_batch)
                    if self._old_network is not None:
                        old_logits = self._old_network(X_batch).detach()
                        hat_pai_k = torch.softmax(old_logits / BiC_T, dim=1)
                        log_pai_k = torch.log_softmax(
                            logits[:, : self._known_classes] / BiC_T, dim=1
                        )
                        distill_loss = -torch.mean(
                            torch.sum(hat_pai_k * log_pai_k, dim=1)
                        )
                        loss = distill_loss * self.lamda + clf_loss * (1 - self.lamda)
                    else:
                        loss = clf_loss

                    if self.show_loss and epoch % step == 0 and epoch != 0:
                        logging.info(f"Epoch {epoch + 1} | Loss {loss.item():.5f}")

                    predictions = torch.argmax(logits, dim=1)
                    correct += (predictions == y_batch).sum()
                    total += y_batch.size(0)
                elif stage == "bias_correction":
                    loss = self._loss_fn(logits, y_batch)
                else:
                    raise ValueError("Invalid BiC stage")

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            if stage == "training":
                accuracy = correct / total
                logging.info(f"Epoch {epoch + 1} | Train Accuracy: {accuracy:.2%}")

            self._scheduler.step()

    def _training(self, train_loader: DataLoader) -> None:
        network = self._get_typed_network()
        ignored_params = list(map(id, network.bias_layers.parameters()))
        base_params = filter(
            lambda p: id(p) not in ignored_params, network.parameters()
        )
        network_params = [
            {"params": base_params, "lr": self.decayed_lr, "weight_decay": BiC_WEIGHT_DECAY},
            {
                "params": network.bias_layers.parameters(),
                "lr": 0,
                "weight_decay": 0,
            },
        ]
        self._configure_optimizer(network_params)
        self._configure_scheduler()
        self._run(train_loader, "training")

    def _bias_correction(self, val_loader: DataLoader) -> None:
        network = self._get_typed_network()
        network_params = [
            {
                "params": network.bias_layers[-1].parameters(),
                "lr": 0.01,
                "weight_decay": BiC_WEIGHT_DECAY,
            }
        ]
        self._configure_optimizer(network_params)
        self._configure_scheduler()
        self._run(val_loader, "bias_correction")


# Hyperparameters from the original paper tweaked to our experiment
# https://arxiv.org/pdf/2302.03648
iCaRL_MOMENTUM = 0.9  # Same as init
iCaRL_INIT_WEIGHT_DECAY = 0
iCaRL_WEIGHT_DECAY = 0
iCaRL_LR_DECAY = 0.8  # Same as init
iCaRL_INIT_MILESTONES = [10, 20, 27]
iCaRL_MILESTONES = [15, 20]
iCaRL_T = 2


def _KD_loss(predictions, soft, T):
    predictions = torch.log_softmax(predictions / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, predictions).sum() / predictions.shape[0]


# Knowledge distillation (logit distillation) + template based
class iCaRLLearner(ExpandableLearner):
    def __init__(self, dimensionality, epochs, lr, show_loss, exponential_lr_decay, lr_decay_gamma):
        super().__init__(dimensionality, epochs, lr, show_loss, exponential_lr_decay, lr_decay_gamma)
        self._old_network = None

    def _after_task(self):
        self._old_network = self.network.copy().freeze()
        self._known_classes = self._total_classes

    def _configure_init_optimizer(self) -> None:
        assert self.network is not None
        self._optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=iCaRL_MOMENTUM,
                                          weight_decay=iCaRL_INIT_WEIGHT_DECAY)

    def _configure_optimizer(self) -> None:
        assert self.network is not None
        if self._total_classes > 0:
            self._optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=iCaRL_MOMENTUM,
                                              weight_decay=iCaRL_WEIGHT_DECAY)

    def _configure_init_scheduler(self) -> None:
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._optimizer, milestones=iCaRL_INIT_MILESTONES, gamma=iCaRL_LR_DECAY
        )

    def _configure_scheduler(self) -> None:
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._optimizer, milestones=iCaRL_MILESTONES, gamma=iCaRL_LR_DECAY
        )

    def train_network(self, X: Tensor, y: Tensor) -> None:
        assert self.network.dimensionality == X.shape[1]
        X, y = X.to(self.network.device), y.to(self.network.device)

        task_size = int(torch.unique(y).numel())
        self._current_task += 1
        self._total_classes = self._known_classes + task_size

        self.expand_network_classifier(self._total_classes)

        if self._memory_X is not None and self._memory_y is not None:
            mem_X, mem_y = self._memory_X.to(self.network.device), self._memory_y.to(self.network.device)
            X = torch.cat([X, mem_X])
            y = torch.cat([y, mem_y])

        train_loader = DataLoader(
            dataset=NNDataset(X, y), batch_size=256, shuffle=True
        )
        self.network.train()
        self._train(train_loader)
        self._after_task()

    def _train(self, train_loader: DataLoader) -> None:
        if self._current_task == 0:
            self._configure_init_optimizer()
            self._configure_init_scheduler()
            self._init_train(train_loader)
        else:
            self._configure_optimizer()
            self._configure_scheduler()
            self._update_representation(train_loader)

    def _init_train(self, train_loader: DataLoader) -> None:
        init_epochs = int(self.network.epochs * 1.5)
        step = max(1, init_epochs // 10)
        logging.info(f"Epochs: {init_epochs}, step: {step}, method: _init_train")

        for epoch in range(init_epochs):
            correct, total = 0, 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.network.device), y_batch.to(self.network.device)
                logits = self.network(X_batch)

                loss = self._loss_fn(logits, y_batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self.show_loss and epoch % step == 0 and epoch != 0:
                    logging.info(f"Epoch {epoch + 1} | Loss {loss.item():.5f}")

                _, predictions = torch.max(logits, dim=1)
                correct += predictions.eq(y_batch.expand_as(predictions)).cpu().sum()
                total += y_batch.size(0)

            accuracy = correct / total
            logging.info(f"Epoch {epoch + 1} | Train Accuracy: {accuracy:.2%}")

            self._scheduler.step()

    def _update_representation(self, train_loader: DataLoader) -> None:
        step = max(1, self.network.epochs // 10)
        logging.info(f"Epochs: {self.network.epochs}, step: {step}, method: _update_representation")

        for epoch in range(self.network.epochs):
            correct, total = 0, 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.network.device), y_batch.to(self.network.device)
                logits = self.network(X_batch)

                loss_clf = self._loss_fn(logits, y_batch)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(X_batch),
                    iCaRL_T,
                )

                loss = loss_clf + loss_kd

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self.show_loss and epoch % step == 0 and epoch != 0:
                    logging.info(f"Epoch {epoch + 1} | Loss {loss.item():.5f}")

                _, predictions = torch.max(logits, dim=1)
                correct += predictions.eq(y_batch.expand_as(predictions)).cpu().sum()
                total += y_batch.size(0)

            accuracy = correct / total
            logging.info(f"Epoch {epoch + 1} | Train Accuracy: {accuracy:.2%}")

            self._scheduler.step()


# Hyperparameters from the original paper tweaked to our experiment
# https://arxiv.org/pdf/2302.03648
GEM_MOMENTUM = 0.9  # Same as init
GEM_INIT_WEIGHT_DECAY = 0
GEM_WEIGHT_DECAY = 0
GEM_LR_DECAY = 0.8  # Same as init
GEM_INIT_MILESTONES = [10, 20, 27]
GEM_MILESTONES = [15, 20]


# Data regularization
class GEMLearner(ExpandableLearner):
    def __init__(self, dimensionality, epochs, lr, show_loss, exponential_lr_decay, lr_decay_gamma):
        super().__init__(dimensionality, epochs, lr, show_loss, exponential_lr_decay, lr_decay_gamma)
        self.prev_X = None
        self.prev_y = None

    def set_previous_data(self, X: Tensor, y: Tensor) -> None:
        self.prev_X, self.prev_y = X, y

    def _configure_init_optimizer(self) -> None:
        assert self.network is not None
        self._optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=GEM_MOMENTUM,
                                          weight_decay=GEM_INIT_WEIGHT_DECAY)

    def _configure_optimizer(self) -> None:
        assert self.network is not None
        if self._total_classes > 0:
            self._optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=GEM_MOMENTUM,
                                              weight_decay=GEM_WEIGHT_DECAY)

    def _configure_init_scheduler(self) -> None:
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._optimizer, milestones=GEM_INIT_MILESTONES, gamma=GEM_LR_DECAY
        )

    def _configure_scheduler(self) -> None:
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._optimizer, milestones=GEM_MILESTONES, gamma=GEM_LR_DECAY
        )

    def train_network(self, X: Tensor, y: Tensor) -> None:
        assert self.network.dimensionality == X.shape[1]
        X, y = X.to(self.network.device), y.to(self.network.device)

        task_size = int(torch.unique(y).numel())
        self._current_task += 1
        self._total_classes = self._known_classes + task_size

        self.expand_network_classifier(self._total_classes)

        if self._memory_X is not None and self._memory_y is not None:
            mem_X, mem_y = self._memory_X.to(self.network.device), self._memory_y.to(self.network.device)
            X = torch.cat([X, mem_X])
            y = torch.cat([y, mem_y])
            self.set_previous_data(mem_X, mem_y)

        train_loader = DataLoader(
            dataset=NNDataset(X, y), batch_size=256, shuffle=True
        )
        self.network.train()
        self._train(train_loader)
        self._after_task()

    def _train(self, train_loader: DataLoader) -> None:
        if self._current_task == 0:
            assert self.prev_X is None and self.prev_y is None, "Previous data should be None"
            self._configure_init_optimizer()
            self._configure_init_scheduler()
            self._init_train(train_loader)
        else:
            assert self.prev_X is not None and self.prev_y is not None, "Previous data should not be None"
            self._configure_optimizer()
            self._configure_scheduler()
            self._update_representation(train_loader)

    def _init_train(self, train_loader: DataLoader) -> None:
        init_epochs = self.network.epochs * 2
        step = max(1, init_epochs // 10)
        logging.info(f"Epochs: {init_epochs}, step: {step}, method: _init_train")

        for epoch in range(init_epochs):
            correct, total = 0, 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.network.device), y_batch.to(self.network.device)
                logits = self.network(X_batch)

                loss = self._loss_fn(logits, y_batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self.show_loss and epoch % step == 0 and epoch != 0:
                    logging.info(f"Epoch {epoch + 1} | Loss {loss.item():.5f}")

                _, predictions = torch.max(logits, dim=1)
                correct += predictions.eq(y_batch.expand_as(predictions)).cpu().sum()
                total += y_batch.size(0)

            accuracy = correct / total
            logging.info(f"Epoch {epoch + 1} | Train Accuracy: {accuracy:.2%}")

            self._scheduler.step()

    def _update_representation(self, train_loader: DataLoader) -> None:
        grad_numels = []
        for params in self.network.parameters():
            grad_numels.append(params.data.numel())
        G = torch.zeros((sum(grad_numels), self._current_task + 1)).to(self.network.device)

        step = max(1, self.network.epochs // 10)
        logging.info(f"Epochs: {self.network.epochs}, step: {step}, method: _update_representation")

        for epoch in range(self.network.epochs):
            correct, total = 0, 0
            for X_batch, y_batch in train_loader:
                incremental_step = self._total_classes - self._known_classes
                for k in range(0, self._current_task):
                    self._optimizer.zero_grad()
                    mask = torch.where(
                        (self.prev_y >= k * incremental_step)
                        & (self.prev_y < (k + 1) * incremental_step)
                    )[0]
                    data_ = self.prev_X[mask].to(self.network.device)
                    label_ = self.prev_y[mask].to(self.network.device)
                    pred_ = self.network(data_)
                    pred_[:, : k * incremental_step].data.fill_(-10e10)
                    pred_[:, (k + 1) * incremental_step :].data.fill_(-10e10)
                    loss_ = self._loss_fn(pred_, label_)
                    loss_.backward()

                    j = 0
                    for params in self.network.parameters():
                        if params is not None:
                            if j == 0:
                                stpt = 0
                            else:
                                stpt = sum(grad_numels[:j])

                            endpt = sum(grad_numels[: j + 1])
                            G[stpt:endpt, k].data.copy_(params.grad.data.view(-1))
                            j += 1

                    self._optimizer.zero_grad()

                X_batch, y_batch = X_batch.to(self.network.device), y_batch.to(self.network.device)
                logits = self.network(X_batch)
                logits[:, : self._known_classes].data.fill_(-10e10)
                loss_clf = self._loss_fn(logits, y_batch)

                loss = loss_clf

                self._optimizer.zero_grad()
                loss.backward()

                j = 0
                for params in self.network.parameters():
                    if params is not None:
                        if j == 0:
                            stpt = 0
                        else:
                            stpt = sum(grad_numels[:j])

                        endpt = sum(grad_numels[: j + 1])
                        G[stpt:endpt, self._current_task].data.copy_(
                            params.grad.data.view(-1)
                        )
                        j += 1

                dotprod = torch.mm(
                    G[:, self._current_task].unsqueeze(0), G[:, : self._current_task]
                )

                if (dotprod < 0).sum() > 0:
                    old_grad = G[:, : self._current_task].cpu().t().double().numpy()
                    cur_grad = G[:, self._current_task].cpu().contiguous().double().numpy()

                    C = old_grad @ old_grad.T
                    p = old_grad @ cur_grad
                    A = np.eye(old_grad.shape[0])
                    b = np.zeros(old_grad.shape[0])

                    v = solve_qp(C, -p, A, b)[0]

                    new_grad = old_grad.T @ v + cur_grad
                    new_grad = torch.tensor(new_grad).float().to(self.network.device)

                    new_dotprod = torch.mm(
                        new_grad.unsqueeze(0), G[:, : self._current_task]
                    )
                    if (new_dotprod < -0.01).sum() > 0:
                        assert 0
                    j = 0
                    for params in self.network.parameters():
                        if params is not None:
                            if j == 0:
                                stpt = 0
                            else:
                                stpt = sum(grad_numels[:j])

                            endpt = sum(grad_numels[: j + 1])
                            params.grad.data.copy_(
                                new_grad[stpt:endpt]
                                .contiguous()
                                .view(params.grad.data.size())
                            )
                            j += 1

                self._optimizer.step()

                if self.show_loss and epoch % step == 0 and epoch != 0:
                    logging.info(f"Epoch {epoch + 1} | Loss {loss.item():.5f}")

                _, predictions = torch.max(logits, dim=1)
                correct += predictions.eq(y_batch.expand_as(predictions)).cpu().sum()
                total += y_batch.size(0)

            accuracy = correct / total
            logging.info(f"Epoch {epoch + 1} | Train Accuracy: {accuracy:.2%}")

            self._scheduler.step()


# Hyperparameters from the original paper tweaked to our experiment
# https://arxiv.org/pdf/2302.03648
EWC_MOMENTUM = 0.9  # Same as init
EWC_INIT_WEIGHT_DECAY = 0
EWC_WEIGHT_DECAY = 0
EWC_LR_DECAY = 0.8  # Same as init
EWC_INIT_MILESTONES = [10, 20, 27]
EWC_MILESTONES = [15, 20]
EWC_FISHERMAX = 0.0001
EWC_LAMBDA = 1000


# Parameter regularization
class EWCLearner(ExpandableLearner):
    def __init__(self, dimensionality, epochs, lr, show_loss, exponential_lr_decay, lr_decay_gamma):
        super().__init__(dimensionality, epochs, lr, show_loss, exponential_lr_decay, lr_decay_gamma)
        self.mean = {}
        self._fisher = None

    def _configure_init_optimizer(self) -> None:
        assert self.network is not None
        self._optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=EWC_MOMENTUM,
                                          weight_decay=EWC_INIT_WEIGHT_DECAY)

    def _configure_optimizer(self) -> None:
        assert self.network is not None
        if self._total_classes > 0:
            self._optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=EWC_MOMENTUM,
                                              weight_decay=EWC_WEIGHT_DECAY)

    def _configure_init_scheduler(self) -> None:
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._optimizer, milestones=EWC_INIT_MILESTONES, gamma=EWC_LR_DECAY
        )

    def _configure_scheduler(self) -> None:
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self._optimizer, milestones=EWC_MILESTONES, gamma=EWC_LR_DECAY
        )

    def train_network(self, X: Tensor, y: Tensor) -> None:
        assert self.network.dimensionality == X.shape[1]
        X, y = X.to(self.network.device), y.to(self.network.device)

        task_size = int(torch.unique(y).numel())
        self._current_task += 1
        self._total_classes = self._known_classes + task_size

        self.expand_network_classifier(self._total_classes)

        if self._memory_X is not None and self._memory_y is not None:
            mem_X, mem_y = self._memory_X.to(self.network.device), self._memory_y.to(self.network.device)
            X = torch.cat([X, mem_X])
            y = torch.cat([y, mem_y])

        train_loader = DataLoader(
            dataset=NNDataset(X, y), batch_size=256, shuffle=True
        )
        self.network.train()
        self._train(train_loader)

        if self._fisher is None:
            self._fisher = self.getFisherDiagonal(train_loader)
        else:
            alpha = self._known_classes / self._total_classes
            new_finsher = self.getFisherDiagonal(train_loader)
            for n, p in new_finsher.items():
                new_finsher[n][: len(self._fisher[n])] = (
                        alpha * self._fisher[n]
                        + (1 - alpha) * new_finsher[n][: len(self._fisher[n])]
                )
            self._fisher = new_finsher
        self.mean = {
            n: p.clone().detach()
            for n, p in self.network.named_parameters()
            if p.requires_grad
        }
        self._after_task()

    def _train(self, train_loader: DataLoader) -> None:
        if self._current_task == 0:
            self._configure_init_optimizer()
            self._configure_init_scheduler()
            self._init_train(train_loader)
        else:
            self._configure_optimizer()
            self._configure_scheduler()
            self._update_representation(train_loader)

    def _init_train(self, train_loader: DataLoader) -> None:
        init_epochs = self.network.epochs * 2
        step = max(1, init_epochs // 10)
        logging.info(f"Epochs: {init_epochs}, step: {step}, method: _init_train")

        for epoch in range(init_epochs):
            correct, total = 0, 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.network.device), y_batch.to(self.network.device)
                logits = self.network(X_batch)

                loss = self._loss_fn(logits, y_batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self.show_loss and epoch % step == 0 and epoch != 0:
                    logging.info(f"Epoch {epoch + 1} | Loss {loss.item():.5f}")

                _, predictions = torch.max(logits, dim=1)
                correct += predictions.eq(y_batch.expand_as(predictions)).cpu().sum()
                total += y_batch.size(0)

            accuracy = correct / total
            logging.info(f"Epoch {epoch + 1} | Train Accuracy: {accuracy:.2%}")

            self._scheduler.step()

    def _update_representation(self, train_loader: DataLoader) -> None:
        step = max(1, self.network.epochs // 10)
        logging.info(f"Epochs: {self.network.epochs}, step: {step}, method: _update_representation")

        for epoch in range(self.network.epochs):
            correct, total = 0, 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.network.device), y_batch.to(self.network.device)
                logits = self.network(X_batch)

                loss_clf = self._loss_fn(logits, y_batch)
                loss_ewc = self.compute_ewc()
                loss = loss_clf + EWC_LAMBDA * loss_ewc

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self.show_loss and epoch % step == 0 and epoch != 0:
                    logging.info(f"Epoch {epoch + 1} | Loss {loss.item():.5f}")

                _, predictions = torch.max(logits, dim=1)
                correct += predictions.eq(y_batch.expand_as(predictions)).cpu().sum()
                total += y_batch.size(0)

            accuracy = correct / total
            logging.info(f"Epoch {epoch + 1} | Train Accuracy: {accuracy:.2%}")

            self._scheduler.step()

    def compute_ewc(self):
        loss = 0
        for n, p in self.network.named_parameters():
            if n in self._fisher.keys():
                loss += (
                        torch.sum(
                            (self._fisher[n])
                            * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                        )
                        / 2
                )
        return loss

    def getFisherDiagonal(self, train_loader):
        fisher = {
            n: torch.zeros(p.shape).to(self.network.device)
            for n, p in self.network.named_parameters()
            if p.requires_grad
        }
        self.network.train()
        optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr)
        for X_batch, y_batch in train_loader:
            inputs, targets = X_batch.to(self.network.device), y_batch.to(self.network.device)
            logits = self.network(X_batch)
            loss = self._loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            for n, p in self.network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(EWC_FISHERMAX))
        return fisher


def create_learner(
        learner_type,
        dimensionality,
        epochs,
        lr,
        show_loss,
        exponential_lr_decay,
        lr_decay_gamma
) -> BaseLearner:
    """
    Creates and returns a learner instance based on the specified network type.

    Args:
        learner_type (str): The type of network architecture to use.
        dimensionality (int): Input feature dimension.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        show_loss (bool): Learner configuration whether to show loss in epochs or not.
        exponential_lr_decay (bool): Learner configuration whether to exponentially decay LR or not.
        lr_decay_gamma (float): A value affecting the decayed LR (by default should be 1/6th of the original LR after all tasks are processed)

    Returns:
        BaseLearner: A learner configured with the chosen network type.
    """

    kwargs = {
        "dimensionality": dimensionality,
        "epochs": epochs,
        "lr": lr,
        "show_loss": show_loss,
        "exponential_lr_decay": exponential_lr_decay,
        "lr_decay_gamma": lr_decay_gamma
    }

    match learner_type:
        case "basic":  # no classifier expansion, support for replay buffer
            learner = BaseLearner(**kwargs)
        case "expandable":  # classifier expansion to new classes
            learner = ExpandableLearner(**kwargs)
        case "expandable-bias":  # classifier expansion to new classes with additional bias layers
            learner = ExpandableLearnerWithBiasLayer(**kwargs)  # training the same as in incremental learner
        case "bic":  # classifier expansion to new classes with additional bias layer
            learner = BiCLearner(**kwargs)  # training in stages (model rectify + knowledge distillation)
        case "icarl":  # classifier expansion to new classes
            learner = iCaRLLearner(**kwargs)  # knowledge distillation + template based classification
        case "ewc":
            learner = EWCLearner(**kwargs)  # parameter regularization
        case "gem":
            learner = GEMLearner(**kwargs)  # data regularization
        case _:
            raise ValueError("Invalid Learner Type")

    return learner


def create_nn(
        dimensionality: int,
        epochs: int,
        network_type: str,
        n_classes: int,
) -> BaseNetwork:
    """
    Creates and initializes a neural network for training.

    Args:
        dimensionality (int): Input feature dimension.
        epochs (int): Number of training epochs.
        network_type: A type of neural network to use.
        n_classes: number of classes to classify

    Returns:
        BaseNetwork: A newly created neural network instance.
    """
    kwargs = {
        "n_classes": n_classes,
        "dimensionality": dimensionality,
        "epochs": epochs,
    }

    match network_type:
        case "basic":  # no classifier expansion
            network = BaseNetwork(**kwargs)
        case "expandable":  # classifier expansion to accommodate new classes
            network = ExpandableNetwork(**kwargs)
        case "expandable-bias":  # classifier expansion to accommodate new classes with additional bias layer
            network = ExpandableNetworkWithBiasLayer(**kwargs)
        case _:
            raise ValueError("Invalid Network Type")

    return network
