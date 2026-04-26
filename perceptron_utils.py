"""
utils/perceptron_utils.py
Shared helper functions for Soft Computing Lab – Module 1.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(
    ax,
    weights: np.ndarray,
    bias: float,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Decision Boundary",
    x_range: tuple = (-0.5, 1.5),
    y_range: tuple = (-0.5, 1.5),
    feature_names: tuple[str, str] = ("x₁", "x₂"),
    class_names: dict | None = None,
) -> None:
    """Draw scatter plot + linear decision boundary on *ax*."""
    if class_names is None:
        class_names = {0: "Class 0", 1: "Class 1"}

    palette = {0: "#e74c3c", 1: "#2980b9"}

    for xi, yi in zip(X, y):
        ax.scatter(xi[0], xi[1], c=palette[yi], s=250, zorder=5,
                   edgecolors="black", lw=1.5)

    if weights[1] != 0:
        x1 = np.linspace(*x_range, 300)
        x2 = -(weights[0] * x1 + bias) / weights[1]
        ax.plot(x1, x2, "g-", lw=2, label="Decision Boundary")

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor=palette[k], markersize=10, label=v)
        for k, v in class_names.items()
    ] + [Line2D([0],[0], color="g", lw=2, label="Decision Boundary")]

    ax.legend(handles=legend_els, fontsize=9)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel(feature_names[0], fontsize=11)
    ax.set_ylabel(feature_names[1], fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)


def plot_convergence(
    ax,
    errors_per_epoch: list[int],
    title: str = "Convergence Curve",
) -> None:
    """Plot misclassification count per epoch on *ax*."""
    ax.plot(range(1, len(errors_per_epoch) + 1), errors_per_epoch,
            "b-o", markersize=5, lw=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Misclassifications", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    conv = next((i + 1 for i, e in enumerate(errors_per_epoch) if e == 0), None)
    if conv:
        ax.axvline(x=conv, color="green", linestyle="--", alpha=0.8,
                   label=f"Converged @ epoch {conv}")
        ax.legend()


def confusion_matrix_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print a simple 2×2 confusion matrix."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    print("\n  Confusion Matrix:")
    print(f"          Pred 0  Pred 1")
    print(f"  True 0    {tn:>4}    {fp:>4}")
    print(f"  True 1    {fn:>4}    {tp:>4}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    print(f"\n  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1-Score  : {f1:.3f}")
    print(f"  Accuracy  : {(tp + tn) / len(y_true) * 100:.1f}%")
