"""
Experiment 1.2 – Training Perceptron using Fixed Increment Learning Algorithm
Course  : Soft Computing Lab (PEC-CSG691A)
Sem     : B.Tech CSE-2023, Semester VI
Faculty : Dr. Vandana Yadav & Dr. Sandip Das

Aim: Train a perceptron using the Fixed Increment Learning Rule and track
     convergence over epochs.

Theory:
    The Fixed Increment Rule always applies a correction of fixed magnitude
    (learning rate) regardless of the magnitude of error.
        Δw_i = α * (target − output) * x_i
    This is guaranteed to converge for linearly separable data (Perceptron
    Convergence Theorem).
"""

import numpy as np


class FixedIncrementPerceptron:
    """Perceptron trained with the Fixed Increment Learning Rule (lr = 1)."""

    def __init__(self, lr: int = 1):
        self.lr = lr
        self.history: list[tuple] = []   # (epoch, weights, bias, errors)

    @staticmethod
    def step(net: float) -> int:
        """Binary step activation function."""
        return 1 if net >= 0 else 0

    def train(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 20):
        """
        Train using fixed-increment updates and print a convergence trace table.

        Returns
        -------
        w : np.ndarray  – final weight vector
        b : float       – final bias
        """
        n_features = X.shape[1]
        w = np.zeros(n_features)
        b = 0.0

        header = f"{'Epoch':>6}  {'Weights':>25}  {'Bias':>6}  {'Errors':>7}"
        print(header)
        print("-" * len(header))

        for epoch in range(1, max_epochs + 1):
            errors = 0

            for xi, yi in zip(X, y):
                net = np.dot(xi, w) + b
                out = self.step(net)
                diff = yi - out

                if diff != 0:
                    w += self.lr * diff * xi   # fixed-increment update
                    b += self.lr * diff
                    errors += 1

            self.history.append((epoch, w.copy(), b, errors))
            print(f"{epoch:>6}  w={np.round(w, 2)}  b={b:>4.1f}  err={errors}")

            if errors == 0:
                print(f"\n✔ Converged!  Final weights: w={w},  bias: b={b}")
                return w, b

        print("\n✘ Did not converge within max_epochs.")
        return w, b

    def print_weight_history(self) -> None:
        """Pretty-print the full weight update history."""
        print("\n=== Weight Update History ===")
        for epoch, w, b, errs in self.history:
            print(f"  Epoch {epoch:02d}: w={np.round(w, 3)}, b={b:.2f}, errors={errs}")


# ── Dataset: AND gate ─────────────────────────────────────────────────────────

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 0, 1])

print("=" * 55)
print("   Fixed Increment Perceptron – AND Gate Training")
print("=" * 55)

fip = FixedIncrementPerceptron(lr=1)
w_final, b_final = fip.train(X, y)

fip.print_weight_history()

# ── Verify predictions ────────────────────────────────────────────────────────

print("\n=== Verification on AND Gate ===")
for xi, yi in zip(X, y):
    net = np.dot(xi, w_final) + b_final
    pred = 1 if net >= 0 else 0
    status = "✔" if pred == yi else "✘"
    print(f"  Input: {xi.astype(int)}  Target: {yi}  Predicted: {pred}  {status}")

accuracy = np.mean(
    np.array([1 if np.dot(xi, w_final) + b_final >= 0 else 0 for xi in X]) == y
) * 100
print(f"\nFinal Accuracy: {accuracy:.1f}%")
