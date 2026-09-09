"""Linear probes on residual-stream vectors.

``LinearProbe`` is a standardized logistic regression inside one sklearn
pipeline, so the scaler only ever sees training rows.

``MassMeanProbe`` is the difference of class means (Marks and Tegmark style).
Scores are projections on that direction; probabilities come from a one-
dimensional logistic fit on the training projections so the output is usable
by a calibrated gate. The direction doubles as a steering vector.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LinearProbe:
    name = "linear"

    def __init__(self, C: float = 1.0, max_iter: int = 5000, seed: int = 0):
        self.pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=seed)),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearProbe:
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(X)[:, 1]

    def decision(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.decision_function(X)

    @property
    def direction(self) -> np.ndarray:
        clf: LogisticRegression = self.pipe.named_steps["clf"]
        scale: StandardScaler = self.pipe.named_steps["scale"]
        return clf.coef_[0] / (scale.scale_ + 1e-8)


class MassMeanProbe:
    name = "massmean"

    def __init__(self, seed: int = 0):
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.direction: np.ndarray | None = None
        self.calib = LogisticRegression(max_iter=1000, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> MassMeanProbe:
        y = np.asarray(y)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-8
        Z = (X - self.mu) / self.sigma
        d = Z[y == 1].mean(axis=0) - Z[y == 0].mean(axis=0)
        self.direction = d / (np.linalg.norm(d) + 1e-12)
        self.calib.fit(self.decision(X).reshape(-1, 1), y)
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        assert self.direction is not None
        Z = (X - self.mu) / self.sigma
        return Z @ self.direction

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calib.predict_proba(self.decision(X).reshape(-1, 1))[:, 1]

    def raw_direction(self) -> np.ndarray:
        """Direction in the model's own coordinates, for steering."""
        assert self.direction is not None and self.sigma is not None
        d = self.direction * self.sigma
        return d / (np.linalg.norm(d) + 1e-12)
