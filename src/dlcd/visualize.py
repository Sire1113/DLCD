from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .data import IMAGENET_MEAN, IMAGENET_STD
from .engine import EvaluationResult, History, PredictionRecord


MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def _prepare_path(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)


def _unnormalize(image: torch.Tensor) -> torch.Tensor:
    return (image * STD + MEAN).clamp(0.0, 1.0)


def plot_training_history(history: History, output_path: Path) -> None:
    _prepare_path(output_path)
    epochs = range(1, len(history.train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    axes[0].plot(epochs, history.train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history.val_loss, label="Val Loss", linewidth=2)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history.train_acc, label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history.val_acc, label="Val Acc", linewidth=2)
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    matrix: list[list[int]],
    class_names: list[str],
    output_path: Path,
) -> None:
    _prepare_path(output_path)
    values = torch.tensor(matrix, dtype=torch.float32)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, int(values[row, col].item()), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_example_grid(
    records: list[PredictionRecord],
    class_names: list[str],
    output_path: Path,
    title: str,
    max_items: int = 8,
) -> None:
    _prepare_path(output_path)
    selected = records[:max_items]
    if not selected:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
        ax.axis("off")
        ax.set_title(title)
        ax.text(0.5, 0.5, "No examples available", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    columns = min(4, len(selected))
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), dpi=150)
    axes_array = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for axis, record in zip(axes_array, selected):
        axis.imshow(_unnormalize(record.image).permute(1, 2, 0))
        axis.set_title(
            f"T:{class_names[record.target]} P:{class_names[record.pred]}",
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes_array[len(selected) :]:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_prediction_examples(
    result: EvaluationResult,
    class_names: list[str],
    output_dir: Path,
    *,
    max_items: int = 8,
) -> None:
    correct = [record for record in result.records if record.target == record.pred]
    wrong = [record for record in result.records if record.target != record.pred]
    _plot_example_grid(correct, class_names, output_dir / "correct_examples.png", "Correct Predictions", max_items=max_items)
    _plot_example_grid(wrong, class_names, output_dir / "wrong_examples.png", "Wrong Predictions", max_items=max_items)
