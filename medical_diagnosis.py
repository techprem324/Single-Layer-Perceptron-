"""
Case Study – Medical Diagnosis using Single Layer Perceptron

Problem Statement:
    Classify patients as Diabetic (1) or Non-Diabetic (0) based on two
    normalised features: Glucose Level and BMI (both in range [0, 1]).

Dataset:
    6 synthetic patient records (normalised values).
    Features: [Glucose (0–1), BMI (0–1)]
    Label   : 1 = Diabetic, 0 = Non-Diabetic
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Dataset ───────────────────────────────────────────────────────────────────

PATIENT_RECORDS = [
    # Glucose  BMI    Label  Description
    (0.7,     0.8,   1,     "High glucose, High BMI"),
    (0.2,     0.3,   0,     "Low glucose, Low BMI"),
    (0.8,     0.6,   1,     "High glucose, Medium BMI"),
    (0.1,     0.2,   0,     "Very low glucose, Very low BMI"),
    (0.6,     0.9,   1,     "Medium glucose, High BMI"),
    (0.3,     0.4,   0,     "Low glucose, Medium BMI"),
]

X_medical = np.array([[g, b] for g, b, _, _ in PATIENT_RECORDS])
y_medical  = np.array([lbl for _, _, lbl, _ in PATIENT_RECORDS])
descriptions = [desc for _, _, _, desc in PATIENT_RECORDS]


# ── Model ─────────────────────────────────────────────────────────────────────

class Perceptron:
    """Minimal Single Layer Perceptron."""

    def __init__(self, lr: float = 0.1, epochs: int = 200):
        self.lr = lr
        self.epochs = epochs
        self.w: np.ndarray = None
        self.b: float = 0.0
        self.errors_per_epoch: list[int] = []

    def _step(self, net: float) -> int:
        return 1 if net >= 0 else 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.errors_per_epoch = []

        for _ in range(self.epochs):
            errs = 0
            for xi, yi in zip(X, y):
                out = self._step(np.dot(xi, self.w) + self.b)
                e   = yi - out
                if e != 0:
                    self.w += self.lr * e * xi
                    self.b += self.lr * e
                    errs   += 1
            self.errors_per_epoch.append(errs)
            if errs == 0:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._step(np.dot(xi, self.w) + self.b) for xi in X])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y) * 100


# ── Train ─────────────────────────────────────────────────────────────────────

slp = Perceptron(lr=0.1, epochs=200)
slp.fit(X_medical, y_medical)
predictions = slp.predict(X_medical)

# ── Results ───────────────────────────────────────────────────────────────────

print("=" * 65)
print("   Case Study – Medical Diagnosis (Diabetes Classification)")
print("=" * 65)
print(f"\nLearned Weights : Glucose = {slp.w[0]:.3f},  BMI = {slp.w[1]:.3f}")
print(f"Learned Bias    : {slp.b:.3f}")
print(f"Decision Rule   : {slp.w[0]:.3f}·Glucose + {slp.w[1]:.3f}·BMI + {slp.b:.3f} >= 0 → Diabetic\n")

print(f"{'#':<4} {'Glucose':>8} {'BMI':>6} {'Target':>8} {'Pred':>8} {'Status':>6}  Description")
print("-" * 75)

for i, (xi, yi, pi, desc) in enumerate(
    zip(X_medical, y_medical, predictions, descriptions), start=1
):
    label  = "Diabetic" if pi == 1 else "Non-Diabetic"
    status = "✔ OK" if yi == pi else "✘ WRONG"
    print(f"{i:<4} {xi[0]:>8.1f} {xi[1]:>6.1f} {yi:>8} {pi:>8}  {status:<7}  {desc}")

print(f"\nOverall Accuracy: {slp.accuracy(X_medical, y_medical):.1f}%")

# ── Visualisation ─────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Scatter: patient feature space ------------------------------------------
colors = ["#e74c3c" if yi == 0 else "#2980b9" for yi in y_medical]
for i, (xi, yi, col) in enumerate(zip(X_medical, y_medical, colors), start=1):
    ax1.scatter(xi[0], xi[1], c=col, s=300, zorder=5, edgecolors="black", lw=1.5)
    ax1.annotate(f"P{i}", (xi[0] + 0.012, xi[1] + 0.012), fontsize=9, fontweight="bold")

# Decision boundary
if slp.w[1] != 0:
    g_range = np.linspace(0, 1, 300)
    b_range = -(slp.w[0] * g_range + slp.b) / slp.w[1]
    ax1.plot(g_range, b_range, "g-", lw=2.5, label="Decision Boundary")

from matplotlib.lines import Line2D
legend_els = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#2980b9", markersize=11, label="Diabetic (1)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=11, label="Non-Diabetic (0)"),
    Line2D([0],[0], color="g", lw=2, label="Decision Boundary"),
]
ax1.legend(handles=legend_els, loc="upper left", fontsize=9)
ax1.set_xlabel("Glucose Level (normalised)", fontsize=11)
ax1.set_ylabel("BMI (normalised)", fontsize=11)
ax1.set_title("Patient Feature Space & Decision Boundary", fontsize=11, fontweight="bold")
ax1.set_xlim(-0.05, 1.1)
ax1.set_ylim(-0.05, 1.1)
ax1.grid(True, linestyle="--", alpha=0.5)

# --- Convergence curve -------------------------------------------------------
ax2.plot(slp.errors_per_epoch, "b-o", markersize=5, lw=2)
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Misclassifications", fontsize=11)
ax2.set_title("Training Convergence", fontsize=11, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)

conv_ep = next((i + 1 for i, e in enumerate(slp.errors_per_epoch) if e == 0), None)
if conv_ep:
    ax2.axvline(x=conv_ep, color="green", linestyle="--", alpha=0.8, label=f"Converged @ ep {conv_ep}")
    ax2.legend()

plt.suptitle(
    "Case Study – Medical Diagnosis with Single Layer Perceptron",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig("case_study/medical_diagnosis.png", dpi=100, bbox_inches="tight")
print("\nPlot saved → case_study/medical_diagnosis.png")
