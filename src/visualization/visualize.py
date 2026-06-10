"""Visualization utilities for results."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json


plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

OUTPUT_DIR = Path("./output")


def plot_training_curves(history: dict, save_path: str = None):
    """Plot training and validation loss + accuracy curves."""
    if save_path is None:
        save_path = str(OUTPUT_DIR / "training_curves.png")

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    ax.plot(epochs, history["val_loss"], "r-s", label="Val Loss", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["val_acc"], "g-^", label="Val Accuracy", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Zero-shot Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_accuracy_comparison(
    results: dict,
    rgb_baseline: dict = None,
    save_path: str = None,
):
    """Bar chart comparing depth zero-shot vs RGB baseline."""
    if save_path is None:
        save_path = str(OUTPUT_DIR / "accuracy_comparison.png")

    labels_list = []
    values = []

    # Depth adapter result
    labels_list.append("Depth (Ours)")
    values.append(results["top1_accuracy"] * 100)

    # Random baseline
    n_classes = len(results.get("per_class_accuracy", {}))
    labels_list.append("Random")
    values.append(100.0 / n_classes if n_classes > 0 else 0)

    # RGB baseline (CLIP upper bound)
    if rgb_baseline is not None:
        labels_list.append("RGB (CLIP Upper Bound)")
        values.append(rgb_baseline["top1_accuracy"] * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(labels_list, values, color=colors[:len(labels_list)], width=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Zero-shot Classification on NYU Depth V2")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Accuracy comparison saved to {save_path}")


def plot_per_class_accuracy(per_class: dict, save_path: str = None):
    """Horizontal bar chart of per-class accuracy."""
    if save_path is None:
        save_path = str(OUTPUT_DIR / "per_class_accuracy.png")

    # Filter NaN
    classes = [k for k, v in per_class.items() if not np.isnan(v)]
    accs = [per_class[c] * 100 for c in classes]

    # Sort by accuracy
    sorted_idx = np.argsort(accs)
    classes = [classes[i] for i in sorted_idx]
    accs = [accs[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(classes, accs, color="#4C72B0")

    for bar, val in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center")

    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Per-Class Zero-shot Accuracy")
    ax.set_xlim(0, max(accs) * 1.2)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Per-class accuracy saved to {save_path}")


def save_results_json(results: dict, save_path: str = None):
    """Save evaluation results as JSON."""
    if save_path is None:
        save_path = str(OUTPUT_DIR / "results.json")

    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, dict):
            serializable[k] = {
                kk: (float(vv) if not isinstance(vv, (int, float)) else vv)
                for kk, vv in v.items()
            }
        else:
            serializable[k] = v

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {save_path}")
