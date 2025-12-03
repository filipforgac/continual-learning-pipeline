import csv
import logging
import os
import threading
import time
import psutil
import torch
from argparse import ArgumentParser
from torch import Tensor


class Settings:
    def __init__(self, args):
        self.num_tasks = args["num_tasks"]
        self.buffer_type = args["replay_buffer_type"]
        self.cl_configuration_type = args["cl_configuration_type"]  # whether to perform a single cl, or multiple (for comparison)
        self.num_of_buffer_increments = args["replay_buffer_increments"]
        self.buffer_size = args["replay_buffer_size"]
        self.fixed_samples_per_class = args["fixed_samples_per_class"]
        self.samples_per_class = args["samples_per_class"]
        self.learner_type = args["learner_type"]
        self.epochs, self.lr = args["epochs"], args["lr"]
        self.exponential_lr_decay = args["exponential_lr_decay"]
        self.show_loss = args["show_loss"]
        self.dataset_name = args["dataset"]
        self.queries_name = args["queries"]

    @classmethod
    def init_from(cls, args):
        return cls(args)


def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility in PyTorch.

    This ensures that PyTorch operations produce the same results across runs.

    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Retrieves the appropriate device for computation.

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    parser = create_parser()
    args = vars(parser.parse_args())

    if args["force_cpu"]:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_parser() -> ArgumentParser:
    """
    Creates an argument parser with default settings for training.

    Returns:
        ArgumentParser: A parser object with predefined arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="laion2B-en-clip768v2-n=300K.h5")
    parser.add_argument("--dataset-labels", default="X_labels.npy")
    parser.add_argument("--num-tasks", default=2, type=int)
    parser.add_argument("--queries", default="public-queries-2024-laion2B-en-clip768v2-n=10k.h5")
    parser.add_argument("--query-labels", default="Q_labels.npy")
    parser.add_argument("--learner-type", choices=["basic", "expandable", "expandable-bias", "bic", "icarl", "ewc", "gem"], default="basic")
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--show-loss", default=False, action="store_true")
    parser.add_argument("--exponential-lr-decay", default=False, action="store_true")
    parser.add_argument(
        "--fixed-samples-per-class",
        default=False,
        action="store_true",
    )
    parser.add_argument("--samples-per-class", default=10, type=int)
    parser.add_argument(
        "--replay-buffer-type",
        choices=["none", "full", "random", "herding", "entropy", "reservoir", "balanced_reservoir", "loss_aware_reservoir"],
        default="none",
        type=str,
    )
    parser.add_argument(
        "--cl-configuration-type",
        choices=["singular", "gradual"],
        default="singular",
        type=str,
    )
    parser.add_argument(
        "--replay-buffer-increments",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--replay-buffer-size",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--force-cpu",
        default=False,
        action="store_true",
    )
    return parser


def get_most_frequent_class_size(y: Tensor) -> int:
    """
    Determines the size (number of samples) of the most frequent class in a dataset.

    Args:
        y (Tensor): Class labels of shape `(N,)`.

    Returns:
        int: The size of the most frequent class.
    """
    _, counts = torch.unique(y, return_counts=True)
    return counts.max().item()


def get_logarithmic_increments(max_value: int, steps: int, min_value: int = 1) -> list[int]:
    """
    Generates a list of logarithmically spaced integers between min_value and max_value.

    Args:
        min_value (int): Minimum value in the output (inclusive).
        max_value (int): Maximum value in the output (inclusive).
        steps (int): Number of steps to generate.

    Returns:
        list[int]: A sorted list of unique sample sizes.
    """
    sample_sizes = torch.logspace(
        start=torch.log10(torch.tensor(min_value, dtype=torch.float)),
        end=torch.log10(torch.tensor(max_value, dtype=torch.float)),
        steps=steps-1
    ).int().tolist()

    # Remove duplicates and ensure limits are respected
    sample_sizes = sorted(set(sample_sizes))
    return sample_sizes


def get_linear_increments(max_value: int, num_steps: int) -> list[int]:
    """
    Returns a list of cumulative linear increments up to max_value.

    Args:
        max_value (int): The total value to reach.
        num_steps (int): Number of increments.

    Returns:
        list[int]: List of cumulative values increasing linearly.
    """
    step_size = max_value // num_steps
    return [step_size * i for i in range(1, num_steps)]


def make_buffer_settings_log(typ: str, subtype: str, steps: int) -> str:
    """
    Generates a formatted string describing replay buffer settings.

    Args:
        typ (str): Type of replay buffer (e.g., "random", "herding", "entropy").
        subtype (str): Subtype of replay buffer ("singular" or "gradual").
        steps (int): Number of steps (only relevant for "gradual" subtype).

    Returns:
        str: A formatted log message.
    """
    buffer_settings = [f"Type: {typ}"]
    if typ in {"random", "herding", "entropy"}:
        buffer_settings.append(f"Subtype: {subtype}")
        if subtype == "gradual":
            buffer_settings.append(f"Steps: {str(steps)}")
    return f"Replay Buffer Settings | {', '.join(buffer_settings)}"


def make_csv_filename(parts: list[str]) -> str:
    """
    Generates a CSV filename from a list of string components.

    Args:
        parts (list[str]): List of components to include in the filename.

    Returns:
        str: The generated CSV filename.
    """
    name = "_".join(parts)
    return f"{name}.csv"


def save_accuracy_to_csv(objects_in_replay: int, accuracy: float, filename: str) -> None:
    """
    Saves accuracy results to a CSV file.

    If the file does not exist, it creates it with appropriate headers.

    Args:
        objects_in_replay (int): Number of objects in the replay buffer.
        accuracy (float): Accuracy percentage.
        filename (str): Name of the CSV file.
    """
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        if not file_exists:
            writer.writerow(["Objects in Buffer", "Accuracy"])
        writer.writerow([objects_in_replay, accuracy])

    logging.info(f"Saved accuracy to {filename}")


class MemoryMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.ram_samples = []
        self.gpu_mem_samples = []
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread = threading.Thread(target=self._record)

    def _record(self):
        process = psutil.Process()
        while not self._stop_event.is_set():
            if not self._pause_event.is_set():
                # RAM
                ram = process.memory_info().rss / (1024 ** 2)  # MB
                self.ram_samples.append(ram)

                # GPU memory
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                else:
                    gpu_mem = 0.0
                self.gpu_mem_samples.append(gpu_mem)

            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    def summarize(self):
        return {
            "avg_ram": sum(self.ram_samples) / len(self.ram_samples),
            "avg_gpu": sum(self.gpu_mem_samples) / len(self.gpu_mem_samples),
            "peak_ram": max(self.ram_samples),
            "peak_gpu": max(self.gpu_mem_samples),
        }
