import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker


class PlotSetting:
    def __init__(self, marker, label, color, linestyle):
        self.marker = marker
        self.label = label
        self.color = color
        self.linestyle = linestyle


def visualize_accuracy_graph(
    title: str,
    min_accuracy_csv: str,
    max_accuracy_csv: str,
    mlp_accuracies_csvs: list[str],
    mlp_accuracies_plot_settings: list[PlotSetting],
) -> None:
    df = pd.read_csv(min_accuracy_csv, delimiter=",")
    min_acc = df["Accuracy"][0]

    df = pd.read_csv(max_accuracy_csv, delimiter=",")
    max_acc = df["Accuracy"][0]

    plt.figure(figsize=(8, 5))
    plt.axhline(y=max_acc, color="green", linestyle="dashed", linewidth=1.5, label=f"All past samples: {max_acc:.2f}%")
    plt.axhline(y=min_acc, color="red", linestyle="dashed", linewidth=1.5, label=f"No past samples: {min_acc:.2f}%")

    assert len(mlp_accuracies_csvs) == len(mlp_accuracies_plot_settings)
    for csv, settings in zip(mlp_accuracies_csvs, mlp_accuracies_plot_settings):
        df = pd.read_csv(csv, delimiter=",")
        plt.plot(
            df["Objects in Buffer"],
            df["Accuracy"],
            marker=settings.marker,
            linestyle=settings.linestyle,
            color=settings.color,
            label=settings.label,
        )

        # for x, y in zip(df["Objects in Buffer"], df["Accuracy"]):
        #     plt.text(x, y + 1, f"{y:.2f}%", ha="center", fontsize=8, fontweight="bold", color=settings.color)
    
    plt.xlabel("Objects in Buffer")
    plt.ylabel("Accuracy (%)")
    plt.title(title)

    def format_k(x, _):
        return f"{int(x/1000)}k" if x >= 1000 else str(int(x))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_k))

    plt.gca().set_facecolor("#f0f0f0")  # Light grey background   
    plt.grid(color="#ffffff", linestyle="-", linewidth=1.5)  # White grid lines

    plt.ylim(0, 100)
    plt.legend()

    plt.show()


def visualize_cost_graph() -> None:
    # Folder containing all method CSVs
    data_dir = "./costs"  # Update this if needed

    metrics = ["Execution Time", "Average RAM", "Average GPU", "Peak RAM", "Peak GPU"]
    blacklist = ["tricks.csv"]  # Blacklist other costs if needed

    methods = []
    accuracies = []
    raw_costs = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".csv") or filename in blacklist:
            continue

        filepath = os.path.join(data_dir, filename)
        df = pd.read_csv(filepath)

        if df.empty:
            continue

        row = df.iloc[0]
        method_name = os.path.splitext(filename)[0]

        # Compute weighted cost
        cost = sum(row[metric] for metric in metrics)
        accuracy = row["Accuracy"]

        methods.append(method_name)
        raw_costs.append(cost)
        accuracies.append(accuracy)

    # Normalize raw costs to [0, 100]
    costs_np = np.array(raw_costs, dtype=np.float64)
    min_cost, max_cost = costs_np.min(), costs_np.max()
    if max_cost == min_cost:
        norm_costs = np.full_like(costs_np, 50.0)
    else:
        norm_costs = 100 * (costs_np - min_cost) / (max_cost - min_cost)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.grid(True, zorder=0)
    plt.gca().set_facecolor("#f0f0f0")  # Light grey background
    plt.grid(color="#ffffff", linestyle="-", linewidth=1.5)  # White grid lines
    for i in range(len(methods)):
        plt.scatter(norm_costs[i], accuracies[i], s=75, label=methods[i], zorder=3)
        plt.text(norm_costs[i] + 1, accuracies[i], methods[i], fontsize=9, zorder=4)

    plt.xlim(-5, 105)
    plt.ylim(0, 100)
    plt.xlabel("Normalized Cost Score (0 = best, 100 = worst)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Accuracy vs Cost Score (Buffer Size - 10k)", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_accuracy_graph(
        title = "Accuracy by Objects in Replay Buffer (Expandable Learner, Different Strategies, 2 Tasks)",
        min_accuracy_csv="./accuracies/accuracy_2_tasks_no_buffer.csv",
        max_accuracy_csv="./accuracies/accuracy_2_tasks_full_buffer.csv",
        mlp_accuracies_csvs=[
            "./accuracies/accuracy_2_tasks_random_buffer.csv",
            "./accuracies/accuracy_2_tasks_entropy_buffer.csv",
            "./accuracies/accuracy_2_tasks_herding_buffer.csv",
            "./accuracies/accuracy_2_tasks_reservoir_buffer.csv",
            "./accuracies/accuracy_2_tasks_balanced_reservoir_buffer.csv",
            "./accuracies/accuracy_2_tasks_loss_aware_reservoir_buffer.csv",
        ],
        mlp_accuracies_plot_settings=[
            PlotSetting(".", "Random Buffer", "blue", "-"),
            PlotSetting(".", "Entropy Buffer", "black", "-"),
            PlotSetting(".", "Herding Buffer", "orange", "-"),
            PlotSetting(".", "Reservoir Buffer", "purple", "-"),
            PlotSetting(".", "Balanced Reservoir Buffer", "red", "-"),
            PlotSetting(".", "Loss-aware Reservoir Buffer", "brown", "-"),
        ]
    )

    visualize_accuracy_graph(
        title="Accuracy by Objects in Replay Buffer (Expandable Learner, Different Strategies, 10 Tasks)",
        min_accuracy_csv="./accuracies/accuracy_10_tasks_no_buffer.csv",
        max_accuracy_csv="./accuracies/accuracy_10_tasks_full_buffer.csv",
        mlp_accuracies_csvs=[
            "./accuracies/accuracy_10_tasks_random_buffer.csv",
            "./accuracies/accuracy_10_tasks_entropy_buffer.csv",
            "./accuracies/accuracy_10_tasks_herding_buffer.csv",
        ],
        mlp_accuracies_plot_settings=[
            PlotSetting(".", "Random Buffer", "blue", "-"),
            PlotSetting(".", "Entropy Buffer", "black", "-"),
            PlotSetting(".", "Herding Buffer", "orange", "-"),
        ]
    )

    visualize_accuracy_graph(
        title="Accuracy by Objects in Replay Buffer (Herding, 2 Tasks)",
        min_accuracy_csv="./accuracies/accuracy_2_tasks_no_buffer.csv",
        max_accuracy_csv="./accuracies/accuracy_2_tasks_full_buffer.csv",
        mlp_accuracies_csvs=[
            "./accuracies/accuracy_2_tasks_incremental.csv",
            "./accuracies/accuracy_2_tasks_icarl.csv",
            "./accuracies/accuracy_2_tasks_ewc.csv",
            "./accuracies/accuracy_2_tasks_bic.csv",
        ],
        mlp_accuracies_plot_settings=[
            PlotSetting(".", "Expandable Learner", "blue", "-"),
            PlotSetting(".", "iCaRL Learner", "orange", "-"),
            PlotSetting(".", "EWC Learner", "purple", "-"),
            PlotSetting(".", "BiC Learner", "black", "-"),
        ]
    )

    visualize_accuracy_graph(
        title="Accuracy by Objects in Replay Buffer (Herding, 10 Tasks)",
        min_accuracy_csv="./accuracies/accuracy_10_tasks_no_buffer.csv",
        max_accuracy_csv="./accuracies/accuracy_10_tasks_full_buffer.csv",
        mlp_accuracies_csvs=[
            "./accuracies/accuracy_10_tasks_incremental.csv",
            "./accuracies/accuracy_10_tasks_icarl.csv",
            "./accuracies/accuracy_10_tasks_ewc.csv",
            "./accuracies/accuracy_10_tasks_bic.csv",
        ],
        mlp_accuracies_plot_settings=[
            PlotSetting(".", "Expandable Learner", "blue", "-"),
            PlotSetting(".", "iCaRL Learner", "orange", "-"),
            PlotSetting(".", "EWC Learner", "purple", "-"),
            PlotSetting(".", "BiC Learner", "black", "-"),
        ]
    )

    visualize_cost_graph()
