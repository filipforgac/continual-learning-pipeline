import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import Tensor


class NNDataset(Dataset):
    """
    Encapsulates a dataset for neural networks, including feature embeddings (X) 
    and corresponding labels (y). This is used for training and evaluation.

    Args:
        X (Tensor): Feature embeddings, shape `(N, D)`, where N is the number of samples.
        y (Tensor): Labels, shape `(N,)`.
    """

    def __init__(self, X: Tensor, y: Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        Retrieves a sample and its label by index.

        Args:
            index (int): The index of the sample.

        Returns:
            tuple (tuple[Tensor, Tensor]): The feature embedding and its label.
        """
        return self.X[index], self.y[index]


def get_data(X_path: str, y_path: str) -> tuple[Tensor, Tensor]:
    """
    Loads data from file paths, retrieving feature embeddings and labels.

    Args:
        X_path (str): Path to the HDF5 file containing the feature embeddings.
        y_path (str): Path to the NumPy (.npy) file containing the labels.

    Returns:
        tuple (tuple[Tensor, Tensor]): Feature embeddings of shape (N, D) and the corresponding labels of shape (N,).
    """
    X = torch.from_numpy(h5py.File(X_path, "r")["emb"][:]).to(torch.float32)
    y = torch.from_numpy(np.load(y_path))
    return X, y


def split_data(X: Tensor, y: Tensor, num_tasks: int) -> tuple[list[Tensor], list[Tensor]]:
    """
    Splits a dataset into num_tasks parts.

    Args:
        X (Tensor): Feature embeddings of shape `(N, D)`.
        y (Tensor): Corresponding labels of shape `(N,)`.
        num_tasks (int): Number of tasks to split the data into.

    Returns:
        tuple (tuple[list[Tensor], list[Tensor]]):
            - datasets (list[Tensor]): A list containing num_tasks datasets.
            - labels (list[Tensor]): A list containing num_tasks labels for the datasets.
    """
    num_classes = torch.unique(y).shape[0]
    per_task = num_classes // num_tasks

    datasets: list[Tensor] = list()
    labels: list[Tensor] = list()

    appended_classes = 0
    for _ in range(num_tasks):
        append_range = (y >= appended_classes) & (y < appended_classes + per_task)
        append_X, append_y = X[append_range], y[append_range]
        datasets.append(append_X)
        labels.append(append_y)
        appended_classes += per_task

    remnant = num_classes % num_tasks
    if remnant > 0:
        append_range = (y >= appended_classes) & (y < appended_classes + remnant)
        append_X, append_y = X[append_range], y[append_range]

        # If there's extra samples/labels to be appended, append them to the last task in order
        # not to exceed the required number of tasks
        datasets[-1] = torch.cat((datasets[-1], append_X), dim=0)
        labels[-1] = torch.cat((labels[-1], append_y), dim=0)

    return datasets, labels

def split_data_to_training_and_validation(X: Tensor, y: Tensor, val_ratio: float = 0.1) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Splits input tensors into training and validation sets.

    Args:
        X (Tensor): Feature tensor of shape (N, D).
        y (Tensor): Label tensor of shape (N,).
        val_ratio (float): Proportion of the data to use for validation (e.g., 0.1 for 10%).

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: (X_train, y_train, X_val, y_val)
    """
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
    num_samples = X.shape[0]
    num_val = int(num_samples * val_ratio)

    indices = torch.randperm(num_samples)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    return X_train, y_train, X_val, y_val
