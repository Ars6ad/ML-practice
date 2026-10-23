import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set publication style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_class_distribution(
    manifest_df: pd.DataFrame,
    save_path: str = "outputs/plots/class_distribution.png",
) -> None:
    """Plots stratified class distribution across train, val, and test splits."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cross_tab = pd.crosstab(manifest_df["label"], manifest_df["split"])
    # Reorder if present
    order = [c for c in ["Good", "Bad", "Mixed"] if c in cross_tab.index]
    cross_tab = cross_tab.loc[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    cross_tab.plot(kind="bar", stacked=False, ax=ax, colormap="viridis", edgecolor="black", width=0.7)

    ax.set_title("Postharvest Fruit Quality Dataset: Class Distribution by Split", pad=15, fontweight="bold")
    ax.set_xlabel("Quality Class", fontweight="bold")
    ax.set_ylabel("Number of Images", fontweight="bold")
    ax.set_xticklabels(cross_tab.index, rotation=0)
    ax.legend(title="Split", frameon=True)

    # Annotate bar counts
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{int(h)}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved class distribution plot to {save_path}")


def plot_training_curves(
    histories: Dict[str, Dict[str, Any]],
    save_path: str = "outputs/plots/training_curves.png",
) -> None:
    """Plots comparative training & validation curves across all completed experiments."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"baseline": "#1f77b4", "augmentation": "#2ca02c", "class_weighted": "#ff7f0e"}

    for exp_name, hist in histories.items():
        color = colors.get(exp_name, "#333333")
        epochs = range(1, len(hist["train_loss"]) + 1)
        label = exp_name.replace("_", " ").title()

        # 1. Losses
        axes[0, 0].plot(epochs, hist["train_loss"], linestyle="--", color=color, alpha=0.6, label=f"{label} Train")
        axes[0, 0].plot(epochs, hist["val_loss"], linestyle="-", color=color, linewidth=2, label=f"{label} Val")

        # 2. Accuracies
        axes[0, 1].plot(epochs, [a * 100 for a in hist["train_acc"]], linestyle="--", color=color, alpha=0.6, label=f"{label} Train")
        axes[0, 1].plot(epochs, [a * 100 for a in hist["val_acc"]], linestyle="-", color=color, linewidth=2, label=f"{label} Val")

        # 3. Macro F1
        axes[1, 0].plot(epochs, hist["val_macro_f1"], linestyle="-", marker="o", markersize=4, color=color, linewidth=2, label=label)

        # 4. Mixed Recall
        axes[1, 1].plot(epochs, [r * 100 for r in hist["val_mixed_recall"]], linestyle="-", marker="s", markersize=4, color=color, linewidth=2, label=label)

    axes[0, 0].set_title("Cross-Entropy Loss vs. Epoch", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].set_title("Accuracy (%) vs. Epoch", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()

    axes[1, 0].set_title("Validation Macro F1-Score", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Macro F1")
    axes[1, 0].legend()

    axes[1, 1].set_title("Validation Mixed-Class Recall (%)", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Mixed Recall (%)")
    axes[1, 1].legend()

    plt.suptitle("Comparative Training Dynamics Across Imbalance Mitigation Strategies", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_confusion_matrix_single(
    cm: List[List[int]],
    class_names: List[str],
    experiment_name: str,
    save_path: str,
) -> None:
    """Plots an individual normalized confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_arr = np.array(cm)
    cm_norm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    annot_matrix = np.empty_like(cm_arr, dtype=object)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            annot_matrix[i, j] = f"{cm_arr[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {experiment_name.replace('_', ' ').title()}", fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("Ground Truth Label", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_metrics_comparison(
    comparison_df: pd.DataFrame,
    save_path: str = "outputs/plots/metrics_comparison.png",
) -> None:
    """Plots comparative bar chart across overall metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metrics_to_plot = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    metric_labels = ["Accuracy", "Balanced Acc", "Macro Prec", "Macro Recall", "Macro F1", "Weighted F1"]

    plot_df = comparison_df.set_index("experiment")[metrics_to_plot]
    plot_df.columns = metric_labels

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.T.plot(kind="bar", ax=ax, colormap="tab10", width=0.7, edgecolor="black")

    ax.set_title("Overall Test Metrics Comparison Across Experiments", fontweight="bold", pad=15)
    ax.set_ylabel("Score (0.0 to 1.0)", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticklabels(metric_labels, rotation=15, ha="right")
    ax.legend(title="Experiment", frameon=True)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=7.5,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics comparison plot to {save_path}")


def plot_mixed_class_analysis(
    comparison_df: pd.DataFrame,
    save_path: str = "outputs/plots/mixed_class_analysis.png",
) -> None:
    """Plots focused analysis on the minority 'Mixed' class precision, recall, and F1 trade-offs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    mixed_cols = ["mixed_precision", "mixed_recall", "mixed_f1"]
    labels = ["Mixed Precision", "Mixed Recall", "Mixed F1-Score"]

    plot_df = comparison_df.set_index("experiment")[mixed_cols]
    plot_df.columns = labels

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.T.plot(kind="bar", ax=ax, colormap="Set2", width=0.65, edgecolor="black")

    ax.set_title("Minority Class ('Mixed') Performance Trade-Off Analysis", fontweight="bold", pad=15)
    ax.set_ylabel("Metric Value", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend(title="Experiment", frameon=True)

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved mixed class analysis plot to {save_path}")


def plot_robustness_comparison(
    rob_df: pd.DataFrame,
    save_path: str = "outputs/plots/robustness_comparison.png",
) -> None:
    """Plots model Macro F1 and Mixed Recall performance across synthetic environmental corruptions."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)

    sns.lineplot(
        data=rob_df,
        x="corruption",
        y="macro_f1",
        hue="experiment",
        marker="o",
        markersize=7,
        linewidth=2,
        ax=axes[0],
    )
    axes[0].set_title("Macro F1-Score Under Environmental Corruptions", fontweight="bold")
    axes[0].set_ylabel("Macro F1-Score")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=40, ha="right")

    sns.lineplot(
        data=rob_df,
        x="corruption",
        y="mixed_recall",
        hue="experiment",
        marker="s",
        markersize=7,
        linewidth=2,
        ax=axes[1],
    )
    axes[1].set_title("Mixed-Class Recall Under Environmental Corruptions", fontweight="bold")
    axes[1].set_ylabel("Mixed Recall")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=40, ha="right")

    plt.suptitle("Environmental Robustness & Generalization Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved robustness comparison plot to {save_path}")
