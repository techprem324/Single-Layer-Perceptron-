"""
Experiment 1.3 – Convergence Analysis & Final Weight Extraction

Aim: Plot the convergence curve (error vs epoch) and extract the final
     learned weights and decision boundary.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; change to "TkAgg" for pop-up
import matplotlib.pyplot as plt


class PerceptronAnalyzer:
    """Perceptron with detailed convergence tracking."""

    def __init__(self, lr: float = 0.1, epochs: int = 50):
        self.lr = lr
        self.epochs = epochs
        self.errors_per_epoch: list[int] = []
        self.w: np.ndarray = None
        self.b: float = 0.0

    def _step(self, net: float) -> int:
        return 1 if net >= 0 else 0

    def _update(self, xi: np.ndarray, yi: int) -> bool:
        """Apply one weight update. Returns True if an update was made."""
        out = self._step(np.dot(xi, self.w) + self.b)
        err = yi - out
        if err != 0:
            self.w += self.lr * err * xi
            self.b += self.lr * err
            return True
        return False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train and record per-epoch misclassification counts."""
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.errors_per_epoch = []

        for _ in range(self.epochs):
            errs = sum(
                1 for xi, yi in zip(X, y)
                if self._update(xi, yi)
            )
            self.errors_per_epoch.append(errs)
            if errs == 0:
                break

    def convergence_epoch(self) -> int | str:
        """Return the first epoch index (1-based) where errors hit zero."""
        for i, e in enumerate(self.errors_per_epoch):
            if e == 0:
                return i + 1
        return "Not converged"

    def decision_boundary_equation(self) -> str:
        return (
            f"{self.w[0]:.4f}·x₁ + {self.w[1]:.4f}·x₂ + {self.b:.4f} = 0"
        )


# ── Dataset ───────────────────────────────────────────────────────────────────

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 0, 1])

pa = PerceptronAnalyzer(lr=0.1, epochs=30)
pa.fit(X, y)

# ── Console report ────────────────────────────────────────────────────────────

print("=" * 50)
print("   Experiment 1.3 – Convergence Analysis")
print("=" * 50)
print(f"Final Weights    : {np.round(pa.w, 4)}")
print(f"Final Bias       : {round(pa.b, 4)}")
print(f"Converged at     : epoch {pa.convergence_epoch()}")
print(f"Decision Boundary: {pa.decision_boundary_equation()}")
print(f"Errors per epoch : {pa.errors_per_epoch}")

# ── Plots ─────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- Convergence curve -------------------------------------------------------
ax1.plot(
    range(1, len(pa.errors_per_epoch) + 1),
    pa.errors_per_epoch,
    "b-o", markersize=6, linewidth=2,
)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Misclassifications", fontsize=12)
ax1.set_title("Convergence Curve – AND Gate Perceptron", fontsize=12, fontweight="bold")
ax1.set_ylim(-0.2, max(pa.errors_per_epoch) + 1)
ax1.grid(True, linestyle="--", alpha=0.6)

# Annotate convergence point
conv = pa.convergence_epoch()
if isinstance(conv, int):
    ax1.axvline(x=conv, color="green", linestyle="--", alpha=0.7, label=f"Converged @ epoch {conv}")
    ax1.legend()

# --- Decision boundary -------------------------------------------------------
class_colors = {0: "red", 1: "blue"}
class_labels = {0: "Class 0 (Non-diabetic / 0)", 1: "Class 1 (Positive / 1)"}

for xi, yi in zip(X, y):
    ax2.scatter(
        xi[0], xi[1],
        c=class_colors[yi], s=250, zorder=5,
        edgecolors="black", linewidth=1.5,
        label=class_labels[yi],
    )
    ax2.annotate(
        f"({int(xi[0])},{int(xi[1])})={yi}",
        (xi[0] + 0.05, xi[1] + 0.05),
        fontsize=10,
    )

if pa.w[1] != 0:
    x1_range = np.linspace(-0.5, 1.5, 300)
    x2_range = -(pa.w[0] * x1_range + pa.b) / pa.w[1]
    ax2.plot(x1_range, x2_range, "g-", lw=2, label="Decision Boundary")

ax2.set_xlim(-0.5, 1.5)
ax2.set_ylim(-0.5, 1.5)
ax2.set_xlabel("x₁", fontsize=12)
ax2.set_ylabel("x₂", fontsize=12)
ax2.set_title("Decision Boundary – AND Gate", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)

# De-duplicate legend handles
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=9)

plt.suptitle(
    "Experiment 1.3 – Convergence Analysis & Decision Boundary",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()

output_path = "experiments/exp1_3_convergence_analysis.png"
plt.savefig(output_path, dpi=100, bbox_inches="tight")
print(f"\nPlot saved → {output_path}")
