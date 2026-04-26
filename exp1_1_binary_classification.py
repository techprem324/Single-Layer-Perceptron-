"""
Experiment 1.1 – Single Layer Perceptron for Binary Classification
Course  : Soft Computing Lab (PEC-CSG691A)
Sem     : B.Tech CSE-2023, Semester VI
Faculty : Dr. Vandana Yadav & Dr. Sandip Das

Aim: Implement a Single Layer Perceptron (SLP) to perform binary
     classification and verify it on logic gate datasets (AND, OR).

Theory:
    A Single Layer Perceptron is the simplest ANN with one layer of weights
    connecting inputs to output. It uses a step activation function and can
    classify linearly separable problems.
    Output y = 1 if net >= theta, else y = 0.
"""

import numpy as np
import matplotlib.pyplot as plt


class SingleLayerPerceptron:
    """Single Layer Perceptron with step activation function."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors = []

    def step_function(self, net: float) -> int:
        """Step (Heaviside) activation: returns 1 if net >= 0, else 0."""
        return 1 if net >= 0 else 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the perceptron using the weight update rule."""
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)   # initialise weights to zero
        self.bias = 0
        self.errors = []

        for epoch in range(self.epochs):
            total_error = 0
            for xi, yi in zip(X, y):
                net = np.dot(xi, self.weights) + self.bias
                output = self.step_function(net)
                error = yi - output

                # Perceptron weight update rule: Δw = lr * error * x
                self.weights += self.lr * error * xi
                self.bias    += self.lr * error
                total_error  += int(error != 0)

            self.errors.append(total_error)

            if total_error == 0:
                print(f"Converged at epoch {epoch + 1}")
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for input matrix X."""
        net = np.dot(X, self.weights) + self.bias
        return np.array([self.step_function(n) for n in net])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy in percent."""
        preds = self.predict(X)
        return np.mean(preds == y) * 100


# ── Dataset definitions ──────────────────────────────────────────────────────

X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or  = np.array([0, 1, 1, 1])

# ── Training & evaluation ────────────────────────────────────────────────────

results = {}

for name, X, y in [("AND", X_and, y_and), ("OR", X_or, y_or)]:
    slp = SingleLayerPerceptron(learning_rate=0.1, epochs=100)
    slp.fit(X, y)
    preds = slp.predict(X)

    results[name] = slp

    print(f"\n=== {name} Gate ===")
    print(f"Weights  : {slp.weights}")
    print(f"Bias     : {slp.bias}")
    print(f"Accuracy : {slp.accuracy(X, y):.1f}%")
    for xi, yi, pi in zip(X, y, preds):
        print(f"  Input:{xi}  Target:{yi}  Predicted:{pi}")

# ── Visualisation ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, X, y) in zip(axes, [("AND", X_and, y_and), ("OR", X_or, y_or)]):
    slp = results[name]
    colors = ["red" if yi == 0 else "blue" for yi in y]

    for xi, yi, col in zip(X, y, colors):
        ax.scatter(xi[0], xi[1], c=col, s=250, zorder=5, edgecolors="black")
        ax.annotate(
            f"({int(xi[0])},{int(xi[1])})={yi}",
            (xi[0] + 0.05, xi[1] + 0.05),
            fontsize=9,
        )

    # Decision boundary: w0*x1 + w1*x2 + b = 0  →  x2 = -(w0*x1 + b) / w1
    if slp.weights[1] != 0:
        x1_vals = np.linspace(-0.5, 1.5, 200)
        x2_vals = -(slp.weights[0] * x1_vals + slp.bias) / slp.weights[1]
        ax.plot(x1_vals, x2_vals, "g-", lw=2, label="Decision Boundary")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"{name} Gate – Decision Boundary")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Custom legend for class colours
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",  markersize=10, label="Class 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Class 1"),
    ]
    ax.legend(handles=legend_elements + ax.lines[:1], loc="upper left")

plt.suptitle("Experiment 1.1 – SLP Decision Boundaries", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("experiments/exp1_1_decision_boundaries.png", dpi=100)
print("\nPlot saved → experiments/exp1_1_decision_boundaries.png")
plt.show()
