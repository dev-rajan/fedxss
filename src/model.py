"""
The detector: a two-layer MLP, implemented in plain NumPy.

Why not PyTorch
---------------
Three reasons, all of which belong in Section 3.4:

1. Section 3.2 requires local training INSIDE the browser. ONNX Runtime Web
   is an inference runtime; it does not train. A hand-written forward and
   backward pass in NumPy transliterates almost line-for-line into the
   JavaScript the extension needs, so the simulation and the deployed
   client run provably the same arithmetic.
2. DP-SGD needs PER-EXAMPLE gradient clipping. For a two-layer MLP the
   per-example gradients have a closed form (see per_example_grads), so
   this is a few lines rather than a dependency on Opacus hooks.
3. It removes torch and opacus from the client entirely, which matters for
   the model-size and memory numbers reported in Table 4.3 — and sidesteps
   the fact that torch has no wheels for Python 3.14 yet.

Architecture: DIM -> HIDDEN (ReLU) -> 1 (sigmoid), binary cross-entropy.
"""

from __future__ import annotations

import numpy as np

HIDDEN = 64


class MLP:
    def __init__(self, dim: int, hidden: int = HIDDEN, seed: int = 0):
        rng = np.random.default_rng(seed)
        # He initialisation for the ReLU layer
        self.W1 = (rng.standard_normal((dim, hidden)) * np.sqrt(2.0 / dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, 1)) * np.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    # -- parameter plumbing (Flower exchanges plain ndarrays) ---------------

    def get_parameters(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def set_parameters(self, params: list[np.ndarray]) -> None:
        self.W1, self.b1, self.W2, self.b2 = [p.astype(np.float32) for p in params]

    def n_params(self) -> int:
        return sum(p.size for p in self.get_parameters())

    # -- forward / backward -------------------------------------------------

    def forward(self, X: np.ndarray):
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(Z1, 0.0)
        Z2 = A1 @ self.W2 + self.b2
        P = 1.0 / (1.0 + np.exp(-np.clip(Z2, -30, 30)))
        return P.ravel(), (X, Z1, A1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    def loss(self, P: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-7
        return float(-np.mean(y * np.log(P + eps) + (1 - y) * np.log(1 - P + eps)))

    def backward(self, cache, P: np.ndarray, y: np.ndarray):
        """Mean gradients over the batch."""
        X, Z1, A1 = cache
        n = X.shape[0]
        dZ2 = ((P - y) / n).reshape(-1, 1)
        gW2 = A1.T @ dZ2
        gb2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        gW1 = X.T @ dZ1
        gb1 = dZ1.sum(axis=0)
        return [gW1, gb1, gW2, gb2]

    def per_example_grads(self, cache, P: np.ndarray, y: np.ndarray):
        """
        Gradients for each example separately — required by DP-SGD, which
        clips each example's contribution before averaging.

        Returned shapes carry a leading batch dimension:
          gW1 (n, dim, hidden)   gb1 (n, hidden)
          gW2 (n, hidden, 1)     gb2 (n, 1)
        """
        X, Z1, A1 = cache
        dZ2 = (P - y).reshape(-1, 1)                       # (n, 1)
        gW2 = A1[:, :, None] * dZ2[:, None, :]             # (n, hidden, 1)
        gb2 = dZ2                                          # (n, 1)
        dZ1 = (dZ2 @ self.W2.T) * (Z1 > 0)                 # (n, hidden)
        gW1 = X[:, :, None] * dZ1[:, None, :]              # (n, dim, hidden)
        gb1 = dZ1                                          # (n, hidden)
        return [gW1, gb1, gW2, gb2]

    def apply_grads(self, grads, lr: float) -> None:
        self.W1 -= lr * grads[0]
        self.b1 -= lr * grads[1].reshape(self.b1.shape)
        self.W2 -= lr * grads[2]
        self.b2 -= lr * grads[3].reshape(self.b2.shape)

    # -- plain SGD (no privacy) --------------------------------------------

    def fit_epoch(self, X, y, lr=0.1, batch_size=64, rng=None) -> float:
        rng = rng or np.random.default_rng(0)
        idx = rng.permutation(len(X))
        total = 0.0
        for start in range(0, len(idx), batch_size):
            b = idx[start:start + batch_size]
            P, cache = self.forward(X[b])
            total += self.loss(P, y[b]) * len(b)
            self.apply_grads(self.backward(cache, P, y[b]), lr)
        return total / max(len(idx), 1)
