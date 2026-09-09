"""Small nonlinear probe as an upper bound on what the layer encodes."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MLPProbe:
    name = "mlp"

    def __init__(self, hidden: int = 256, seed: int = 0, max_iter: int = 400):
        self.pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(hidden,),
                        alpha=1e-2,
                        early_stopping=False,
                        max_iter=max_iter,
                        random_state=seed,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPProbe:
        self.pipe.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(X)[:, 1]

    def decision(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(self.predict_proba(X), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
