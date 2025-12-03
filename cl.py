import logging
import random
from typing import Generator
import torch
import torch.nn.functional as F
from torch import Tensor

from model import BaseNetwork, create_learner, BaseLearner


def cl_with_no_rehearsal_memory(
        datasets: list[Tensor],
        labels: list[Tensor],
        epochs: int,
        lr: float,
        show_loss: bool,
        exponential_lr_decay: bool,
        lr_decay_gamma: float,
        timestamp: str,
        learner_type: str,
) -> tuple[BaseLearner, int]:
    """
    Performs continual learning (CL) without using a replay buffer.

    This simulates catastrophic forgetting: after training on one dataset, 
    the network is immediately trained on the second dataset without retaining 
    knowledge from the first.

    Training sequence:
    1. Train sequentially on dataset and labels from input.
    2. After end of each step do not append anything to replay buffer.

    Args:
        datasets (list[Tensor]): A list of feature embeddings of shape (N1, D).
        labels (list[Tensor]): A list of labels of shape (N1,).
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the neural network.
        show_loss (bool): Learner configuration whether to show loss in epochs or not.
        exponential_lr_decay (bool): Learner configuration whether to exponentially decay LR or not.
        lr_decay_gamma (float): A value affecting the decayed LR (by default should be 1/6th of the original LR after all tasks are processed)
        timestamp (str): A timestamp identifier for saving the network.
        learner_type (str): A network type the learner should use.

    Returns:
        tuple (tuple[BaseLearner, int]): A learner with trained neural network and the number
        of samples stored in the replay buffer.

    Note:
        - The function saves the trained network as "nn_no_replay_{timestamp}.pth".
        - Expect catastrophic forgetting, as no data from (X1, y1) is retained.
    """
    learner = create_learner(
        dimensionality=datasets[0].shape[1],
        epochs=epochs,
        lr=lr,
        learner_type=learner_type,
        show_loss=show_loss,
        exponential_lr_decay=exponential_lr_decay,
        lr_decay_gamma=lr_decay_gamma,
    )

    for data, label in zip(datasets, labels):
        data, label = data.to(learner.network.device), label.to(learner.network.device)
        learner.train_network(data, label)

    learner.serialize(f"nn_no_replay_{timestamp}.pth")
    return learner, 0


def cl_with_full_rehearsal_memory(
        datasets: list[Tensor],
        labels: list[Tensor],
        epochs: int,
        lr: float,
        show_loss: bool,
        exponential_lr_decay: bool,
        lr_decay_gamma: float,
        timestamp: str,
        learner_type: str,
) -> tuple[BaseLearner, int]:
    """
    Performs continual learning (CL) with full rehearsal memory.

    This simulates a classic machine learning setting where previous data 
    is fully retained. Instead of catastrophic forgetting, the network is trained
    on both new and previous datasets together.

    Training sequence:
    1. Train sequentially on dataset and labels from input.
    2. After end of each step append the entirety of previous data and labels to replay buffer.

    Args:
        datasets (list[Tensor]): A list of feature embeddings of shape (N1, D).
        labels (list[Tensor]): A list of labels of shape (N1,).
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the neural network.
        show_loss (bool): Learner configuration whether to show loss in epochs or not.
        exponential_lr_decay (bool): Learner configuration whether to exponentially decay LR or not.
        lr_decay_gamma (float): A value affecting the decayed LR (by default should be 1/6th of the original LR after all tasks are processed)
        timestamp (str): A timestamp identifier for saving the network.
        learner_type (str): A network type the learner should use.

    Returns:
        tuple (tuple[BaseLearner, int]): A learner with trained neural network and the number
        of samples stored in the replay buffer.

    Note:
        - The function saves the trained network as "nn_full_replay_{timestamp}.pth".
        - This avoids catastrophic forgetting alltogether since all old data is replayed.
    """
    learner = create_learner(
        dimensionality=datasets[0].shape[1],
        epochs=epochs,
        lr=lr,
        learner_type=learner_type,
        show_loss=show_loss,
        exponential_lr_decay=exponential_lr_decay,
        lr_decay_gamma=lr_decay_gamma
    )

    replay_X, replay_y = datasets[0].to(learner.network.device), labels[0].to(learner.network.device)
    for n_task, (data, label) in enumerate(zip(datasets, labels)):
        data, label = data.to(learner.network.device), label.to(learner.network.device)
        learner.train_network(data, label)

        # No need to create buffer because there's no more data to append it to
        if n_task == len(datasets) - 1:
            break

        learner.set_replay_buffer(replay_X, replay_y)
        if 0 < n_task:
            replay_X, replay_y = torch.cat((replay_X, data), dim=0), torch.cat((replay_y, label), dim=0)

    learner.serialize(f"nn_full_replay_{timestamp}.pth")
    return learner, int(replay_X.shape[0])


def build_random_rehearsal_memory(X: Tensor, y: Tensor, samples_per_class: int) -> tuple[Tensor, Tensor]:
    """
    Creates a randomized rehearsal buffer by selecting a fixed number of 
    exemplars per class at random.

    Unlike herding-based selection, this method randomly selects a subset of 
    the dataset, making it simpler but less optimal.

    Steps:
    1. Iterate over all unique classes.
    2. Select `samples_per_class` random exemplars for each class.
    3. Concatenate selected exemplars to form the replay buffer.

    Args:
        X (Tensor): Feature embeddings of shape (N, D).
        y (Tensor): Corresponding labels of shape (N,).
        samples_per_class (int): Number of exemplars to store per class.

    Returns:
        tuple (tuple[Tensor, Tensor]): Selected samples and corresponding class labels.

    Example:
        >>> X = torch.randn(100, 512)  # 100 samples, 512 features
        >>> y = torch.randint(0, 10, (100,))  # 10 classes
        >>> X_replay, y_replay = build_random_rehearsal_memory(X, y, 5)
        >>> print(X_replay.shape)  # (50, 512) if 10 classes × 5 samples each
        >>> print(y_replay.shape)  # (50,)

    Note:
        - This method does not optimize selection for representativeness.
        - Results may vary due to randomness in sampling.
    """
    unique_classes = torch.unique(y)
    X_replay, y_replay = [], []

    for cls in unique_classes:
        class_indices = (y == cls).nonzero(as_tuple=True)[0]
        selected_indices = class_indices[torch.randperm(len(class_indices))[:samples_per_class]]
        X_replay.append(X[selected_indices])
        y_replay.append(y[selected_indices])

    if len(X_replay) > 0:
        X_replay = torch.cat(X_replay, dim=0)
        y_replay = torch.cat(y_replay, dim=0)
    else:
        X_replay = torch.empty((0, X.shape[1]), device=X.device)
        y_replay = torch.empty((0,), dtype=y.dtype, device=y.device)

    return X_replay, y_replay


# An implementation of class mean computation, described in exemplar selection 
# in the article Class-Incremental Learning: A Survey (https://arxiv.org/pdf/2302.03648)
def compute_class_means(X: Tensor, y: Tensor) -> dict[int, Tensor]:
    """
    Compute the mean feature vector (`μ_y`) for each class.
    Equation: `μ_y ← 1/n ∑ φ(x_i)`.
    
    Args:
        X (Tensor): Tensor of feature embeddings of shape (N, D).
        y (Tensor): Tensor of corresponding class labels of shape (N,).

    Returns:
        class_means (dict[int, Tensor]): Dictionary mapping class labels to their mean feature vector.
    """
    unique_classes = torch.unique(y)
    class_means = {}

    for cls in unique_classes:
        class_indices = (y == cls).nonzero(as_tuple=True)[0]
        class_samples = X[class_indices]

        # μ_y = (1/n) ∑ φ(x_i)
        class_means[cls.item()] = class_samples.mean(dim=0)

    return class_means


# An implementation of herding algorithm described in Eq.4 in the article
# Class-Incremental Learning: A Survey (https://arxiv.org/pdf/2302.03648)
def build_herding_rehearsal_memory(X: Tensor, y: Tensor, samples_per_class: int) -> tuple[Tensor, Tensor]:
    """
    Implements herding exemplar selection for continual learning.
    Equation: `p_k ← argmin_x ∥ μ_y - (1/k) [ φ(x) + ∑_{j=1}^{k-1} φ(p_j) ] ∥`

    Args:
        X (Tensor): Tensor of feature embeddings of shape (N, D).
        y (Tensor): Tensor of corresponding class labels of shape (N,).
        samples_per_class (int): Number of exemplars to store per class.

    Returns:
        tuple (tuple[Tensor, Tensor]): Tensor of selected exemplars and corresponding class labels of exemplars.
    """
    class_means = compute_class_means(X, y)  # Compute μ_y for each class
    unique_classes = torch.unique(y)
    X_replay_list, y_replay_list = [], []

    for cls in unique_classes:
        class_indices = (y == cls).nonzero(as_tuple=True)[0]
        class_samples = X[class_indices]
        class_mean = class_means[cls.item()]  # Get μ_y for class cls

        selected_samples = []
        running_sum = torch.zeros_like(class_mean)  # Tracks ∑_{j=1}^{k-1} φ(p_j)

        # All class samples are available for selection at first
        available_mask = torch.ones(len(class_samples), dtype=torch.bool, device=class_samples.device)

        for k in range(1, min(samples_per_class, len(class_samples)) + 1):
            # Get available samples (those not already selected)
            available_samples = class_samples[available_mask]

            # (1/k) [ φ(x) + ∑_{j=1}^{k-1} φ(p_j) ]
            candidate_means = (available_samples + running_sum) / k

            # ∥ μ_y - (1/k) [ φ(x) + ∑_{j=1}^{k-1} φ(p_j) ] ∥
            distances = torch.linalg.vector_norm(class_mean - candidate_means, dim=1)

            # Select the sample that minimizes this distance (argmin)
            closest_idx = torch.argmin(distances).item()
            selected_sample = available_samples[closest_idx]
            selected_samples.append(selected_sample)

            # Update running sum: running_sum += φ(p_k)
            running_sum += selected_sample

            # Mark sample as unavailable, because it was already selected
            original_idx = torch.where(available_mask)[0][closest_idx]
            available_mask[original_idx] = False

        if samples_per_class > 0:
            X_selected = torch.stack(selected_samples)
            y_selected = torch.full((X_selected.shape[0],), cls, dtype=y.dtype, device=y.device)

            X_replay_list.append(X_selected)
            y_replay_list.append(y_selected)

    if X_replay_list:
        X_replay = torch.cat(X_replay_list, dim=0)
        y_replay = torch.cat(y_replay_list, dim=0)
    else:
        X_replay = torch.empty((0, X.shape[1]), device=X.device)
        y_replay = torch.empty((0,), dtype=y.dtype, device=y.device)

    return X_replay, y_replay


# An implementation of entropy based sampling algorithm described in chapter 4 in the article
# Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence (https://arxiv.org/pdf/1801.10112)
def compute_entropy(logits: Tensor) -> Tensor:
    """
    Compute the entropy of the model's output probability distribution.

    Args:
        logits (Tensor): Model logits of shape (N, C), where C is the number of classes.

    Returns:
        entropy (Tensor): Entropy values for each sample of shape (N,).
    """
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-10)  # Avoid log(0) to prevent NaNs
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy


# An implementation of entropy based sampling algorithm described in chapter 4 in the article
# Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence (https://arxiv.org/pdf/1801.10112)
def build_entropy_rehearsal_memory(
        network: BaseNetwork,
        X: Tensor, y: Tensor,
        samples_per_class: int
) -> tuple[Tensor, Tensor]:
    """
    Selects high-entropy (uncertain) samples to build a rehearsal buffer for continual learning.

    Args:
        network (BaseNetwork): Trained neural network.
        X (Tensor): Feature embeddings of shape (N, D).
        y (Tensor): Corresponding labels of shape (N,).
        samples_per_class (int): Number of exemplars to store per class.

    Returns:
        tuple (tuple[Tensor, Tensor]): Selected high-entropy samples and the corresponding labels.
    """
    with torch.no_grad():
        logits = network.model(X)
        entropy = compute_entropy(logits)

    unique_classes = torch.unique(y)
    X_replay, y_replay = [], []

    for cls in unique_classes:
        class_indices = (y == cls).nonzero(as_tuple=True)[0]
        class_entropy = entropy[class_indices]

        num_samples = min(samples_per_class, len(class_indices))
        if num_samples > 0:
            _, top_indices = torch.topk(class_entropy, num_samples)
            selected_indices = class_indices[top_indices]

            X_replay.append(X[selected_indices])
            y_replay.append(y[selected_indices])

    if len(X_replay) > 0:
        X_replay = torch.cat(X_replay, dim=0)
        y_replay = torch.cat(y_replay, dim=0)
    else:
        X_replay = torch.empty((0, X.shape[1]), device=X.device)
        y_replay = torch.empty((0,), dtype=y.dtype, device=y.device)

    return X_replay, y_replay


def cl_with_rehearsal_memory(
        datasets: list[Tensor],
        labels: list[Tensor],
        epochs: int,
        lr: float,
        show_loss: bool,
        exponential_lr_decay: bool,
        lr_decay_gamma: float,
        memory_build_strategy: str,
        timestamp: str,
        learner_type: str,
        fixed_samples_per_class: bool,
        memory_size: int | None = None,
        samples_per_class: int | None = None,
        filename_suffix: int | None = None,
) -> tuple[BaseLearner, int]:
    """
    Performs continual learning (CL) using a replay buffer.

    This function trains a neural network sequentially in two phases:
    1. First, it trains on a dataset with labels.
    2. Then, it builds a replay buffer using a specified memory build strategy.
    3. Finally, it trains on a new dataset (from next task) augmented with replayed samples (from previous task).

    Args:
        datasets (list[Tensor]): A list of feature embeddings of shape (N1, D).
        labels (list[Tensor]): A list of labels of shape (N1,).
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the neural network.
        show_loss: Learner configuration whether to show loss in epochs or not.
        exponential_lr_decay (bool): Learner configuration whether to exponentially decay LR or not.
        lr_decay_gamma (float): A value affecting the decayed LR (by default should be 1/6th of the original LR after all tasks are processed)
        fixed_samples_per_class (bool): Flag determining whether the number of samples per class is fixed or no.
        memory_size (int | None): Max size of memory permitted in the buffer, if fixed buffer, None otherwise.
        samples_per_class (int | None): Number of exemplars to store per class, if gradual buffer, None otherwise.
        memory_build_strategy (str): Strategy to build the replay buffer. 
            Options:
                - "random"
                - "herding"
                - "entropy"
        timestamp (str): A timestamp identifier for saving the model.
        learner_type (str): A network type the learner should use.
        filename_suffix (int, optional): A numerical suffix to differentiate saved models.

    Returns:
        tuple (tuple[BaseLearner, int]): A learner with trained neural network and the number
        of samples stored in the replay buffer.

    Raises:
        ValueError: If an invalid memory build strategy is provided.

    Note:
        - The function saves the trained network as "nn_{memory_build_strategy}_replay_{suffix}_{timestamp}.pth".
        - This avoids catastrophic forgetting depending on the strategy as well as number of samples in the buffer
          (specified in suffix).
    """
    learner = create_learner(
        dimensionality=datasets[0].shape[1],
        epochs=epochs,
        lr=lr,
        learner_type=learner_type,
        show_loss=show_loss,
        exponential_lr_decay=exponential_lr_decay,
        lr_decay_gamma=lr_decay_gamma
    )

    if fixed_samples_per_class:
        assert samples_per_class is not None and memory_size is None
    else:
        assert samples_per_class is None and memory_size is not None

    X_replay: Tensor | None = None

    for n_task, (data, label) in enumerate(zip(datasets, labels)):
        # Get known classes before processing the task
        old_known_classes = learner.get_num_known_classes()

        data, label = data.to(learner.network.device), label.to(learner.network.device)
        learner.train_network(data, label)

        # No need to create buffer because there's no more data to append it to
        if n_task == len(datasets) - 1:
            break

        if fixed_samples_per_class:
            per_class = samples_per_class
        else:
            per_class = memory_size // learner.get_num_known_classes()

        message = f"Building memory with samples per class: {per_class} | Known classes: {learner.get_num_known_classes()}"
        if not fixed_samples_per_class:
            message = f"{message} | Memory size: {memory_size}"
        logging.info(message)

        # Processing old exemplars
        X_replay_old, y_replay_old = learner.get_replay_buffer()
        if X_replay_old is not None and y_replay_old is not None:
            X_replay_old, y_replay_old = X_replay_old.to(learner.network.device), y_replay_old.to(learner.network.device)

            if fixed_samples_per_class:
                X_replay_reduced, y_replay_reduced = X_replay_old, y_replay_old
            else:
                X_replay_reduced = torch.empty((0, X_replay_old.shape[1]), device=X_replay_old.device)
                y_replay_reduced = torch.empty((0,), dtype=y_replay_old.dtype, device=y_replay_old.device)
                for class_idx in range(old_known_classes):
                    mask = (y_replay_old == class_idx).nonzero(as_tuple=True)[0]
                    X, y = X_replay_old[mask][:per_class], y_replay_old[mask][:per_class]
                    X_replay_reduced = torch.cat((X_replay_reduced, X), dim=0)
                    y_replay_reduced = torch.cat((y_replay_reduced, y))
        else:
            X_replay_reduced, y_replay_reduced = None, None

        # Making new exemplars
        match memory_build_strategy:
            case "random":
                X_replay_new, y_replay_new = build_random_rehearsal_memory(data, label, per_class)
            case "herding":
                X_replay_new, y_replay_new = build_herding_rehearsal_memory(data, label, per_class)
            case "entropy":
                X_replay_new, y_replay_new = build_entropy_rehearsal_memory(learner.network, data, label, per_class)
            case _:
                raise ValueError("Invalid Memory Build Strategy")

        # Setting the final replay buffer (old exemplars + new exemplars)
        if X_replay_reduced is not None and y_replay_reduced is not None:
            X_replay, y_replay = torch.cat((X_replay_reduced, X_replay_new), dim=0), torch.cat((y_replay_reduced, y_replay_new), dim=0)
        else:
            X_replay, y_replay = X_replay_new, y_replay_new

        logging.info(f"Setting final replay buffer, with size: {X_replay.shape[0]}")
        learner.set_replay_buffer(X_replay, y_replay)

    filename = f"nn_{memory_build_strategy}_replay"
    if filename_suffix is not None:
        filename = f"{filename}_{filename_suffix}"
    filename = f"{filename}_{timestamp}.pth"
    learner.serialize(filename)

    return learner, int(X_replay.shape[0])


def cl_with_reservoir_sampling(
        datasets: list[Tensor],
        labels: list[Tensor],
        epochs: int,
        lr: float,
        show_loss: bool,
        exponential_lr_decay: bool,
        lr_decay_gamma: float,
        memory_build_strategy: str,
        timestamp: str,
        learner_type: str,
        memory_size: int,
        filename_suffix: int | None = None,
) -> tuple[BaseLearner, int]:
    learner = create_learner(
        dimensionality=datasets[0].shape[1],
        epochs=epochs,
        lr=lr,
        learner_type=learner_type,
        show_loss=show_loss,
        exponential_lr_decay=exponential_lr_decay,
        lr_decay_gamma=lr_decay_gamma
    )

    # Initialize buffers
    fst_X, fst_y = datasets[0].to(learner.network.device), labels[0].to(learner.network.device)
    X_replay = torch.empty((0, fst_X.shape[1]), device=fst_X.device)
    y_replay = torch.empty((0,), dtype=fst_y.dtype, device=fst_y.device)
    loss_replay = torch.empty((0,), dtype=torch.float32, device=fst_X.device)

    seen_samples = 0
    for n_task, (data, label) in enumerate(zip(datasets, labels)):
        data, label = data.to(learner.network.device), label.to(learner.network.device)
        learner.train_network(data, label)

        # Update the loss of current items stored in the buffer
        if memory_build_strategy == "loss_aware_reservoir" and len(X_replay) > 0:
            loss_replay = learner.compute_loss(X_replay, y_replay).clone()

        if n_task == len(datasets) - 1:
            break  # No need to build buffer after last task

        for sample_X, sample_y in zip(data, label):
            sample_X = sample_X.unsqueeze(0)
            sample_y_batch = sample_y.unsqueeze(0)
            sample_loss = learner.compute_loss(sample_X, sample_y_batch)[0]

            if memory_size > seen_samples:
                X_replay = torch.cat([X_replay, sample_X], dim=0)
                y_replay = torch.cat([y_replay, sample_y_batch], dim=0)
                loss_replay = torch.cat([loss_replay, sample_loss.unsqueeze(0)], dim=0)
            else:
                j = random.randint(0, seen_samples)
                if j < memory_size:
                    match memory_build_strategy:
                        case "reservoir":
                            X_replay[j] = sample_X.squeeze(0)
                            y_replay[j] = sample_y
                        case "balanced_reservoir":
                            unique, counts = torch.unique(y_replay, return_counts=True)
                            class_counts = dict(zip(unique.tolist(), counts.tolist()))
                            most_frequent_class = max(class_counts, key=class_counts.get)
                            candidate_indices = (y_replay == most_frequent_class).nonzero(as_tuple=True)[0]
                            k = random.choice(candidate_indices.tolist())
                            X_replay[k] = sample_X.squeeze(0)
                            y_replay[k] = sample_y
                        case "loss_aware_reservoir":
                            unique, counts = torch.unique(y_replay, return_counts=True)
                            max_class = y_replay.max()
                            class_count_tensor = torch.zeros(
                                int(max_class) + 1, dtype=torch.float32, device=y_replay.device
                            )
                            class_count_tensor[unique.long()] = counts.float()

                            S_balance = class_count_tensor[y_replay.long()]
                            S_loss = -loss_replay

                            mean_S_balance = S_balance.mean()
                            mean_S_loss = S_loss.mean()
                            alpha = mean_S_balance / mean_S_loss if mean_S_loss != 0 else 1.0

                            S = alpha * S_loss + S_balance
                            probs = S / S.sum()

                            k = torch.multinomial(probs, num_samples=1).item()
                            X_replay[k] = sample_X.squeeze(0)
                            y_replay[k] = sample_y
                            loss_replay[k] = sample_loss
                        case _:
                            raise ValueError("Invalid Reservoir Sampling Type")

            seen_samples += 1

        learner.set_replay_buffer(X_replay, y_replay)

    # Save model
    filename = f"nn_{memory_build_strategy}_replay"
    if filename_suffix is not None:
        filename = f"{filename}_{filename_suffix}"
    filename = f"{filename}_{timestamp}.pth"
    learner.serialize(filename)

    return learner, int(X_replay.shape[0])


def perform_cl(buffer_type: str, fixed_samples_per_class: bool, **kwargs) -> tuple[BaseLearner, int]:
    """
    Performs continual learning (CL) based on the specified buffer type.

    Args:
        buffer_type (str): The type of rehearsal memory strategy.
            Options:
                - "none" (No replay buffer, leads to catastrophic forgetting).
                - "full" (Full dataset replay).
                - "random" (Randomly sampled replay buffer).
                - "herding" (Herding-based replay buffer).
                - "entropy" (Uncertainty-based replay buffer).
        fixed_samples_per_class (bool): Flag determining whether the number of samples per class is fixed or no.
        **kwargs: Additional arguments passed to the corresponding CL function.

    Returns:
        tuple (tuple[BaseLearner, int]): A learner with trained neural network and the number of samples stored in the replay buffer.

    Raises:
        ValueError: If an invalid buffer type is provided.
    """
    match buffer_type:
        case "none":
            learner, objects_in_replay = cl_with_no_rehearsal_memory(**kwargs)
        case "full":
            learner, objects_in_replay = cl_with_full_rehearsal_memory(**kwargs)
        case "random" | "herding" | "entropy":
            kwargs["fixed_samples_per_class"] = fixed_samples_per_class
            learner, objects_in_replay = cl_with_rehearsal_memory(**kwargs)
        case "reservoir" | "balanced_reservoir" | "loss_aware_reservoir":
            learner, objects_in_replay = cl_with_reservoir_sampling(**kwargs)
        case _:
            raise ValueError("Invalid Memory Build Strategy")

    return learner, objects_in_replay


def perform_cl_gradual(buffer_type: str, fixed_samples_per_class: bool, increments: list[int], **kwargs) -> Generator[
    tuple[BaseLearner, int], None, None]:
    """
    Performs continual learning (CL) with a gradually increasing replay buffer.
    The replay buffer can either have fixed samples per class (and thus they are the increments),
    or can have a total memory size (and pick samples per class dynamically to satisfy the memory size).
    If the latter is the case the increments are total memory sizes.

    This method starts with an empty replay buffer and incrementally increases 
    the number of stored samples per class over multiple iterations.

    Args:
        buffer_type (str): The type of rehearsal memory strategy to use.
        fixed_samples_per_class (bool): Flag determining whether the number of samples per class is fixed or no.
        increments (list[int]): List of increments - to gradually increase the buffer size.
        **kwargs: Additional arguments for CL run.

    Returns:
        tuple (tuple[BaseLearner, int]): A learner with trained neural network and the number of
        samples stored in the replay buffer at each step.

    Example:
        >>> sample_sizes = [0, 5, 10, 20]
        >>> for model, buffer_size in perform_cl_gradual("entropy", True, sample_sizes, **kwargs):
        >>>     print(f"Buffer size: {buffer_size}, Model trained successfully.")
    """
    if fixed_samples_per_class:
        kwargs["samples_per_class"] = 0
    else:
        kwargs["memory_size"] = 0
    kwargs["filename_suffix"] = f"mem_{0}"
    yield perform_cl(buffer_type=buffer_type, fixed_samples_per_class=fixed_samples_per_class, **kwargs)

    for _, increment in enumerate(increments, start=1):
        if fixed_samples_per_class:
            kwargs["samples_per_class"] = increment
        else:
            kwargs["memory_size"] = increment
        kwargs["filename_suffix"] = f"mem_{increment}"
        yield perform_cl(buffer_type=buffer_type, fixed_samples_per_class=fixed_samples_per_class, **kwargs)
