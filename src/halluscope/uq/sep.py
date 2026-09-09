"""Semantic Entropy Probe (Kossen et al. 2024, arXiv:2406.15927).

A linear model trained to predict semantic entropy from a single hidden state,
so the expensive sampling happens only at training time. We regress the
semantic entropy computed on the training split and use the predicted value
as the uncertainty score at test time.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SEProbe:
    name = "sep"

    def __init__(self, alpha: float = 10.0):
        self.pipe = Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=alpha))])

    def fit(self, X: np.ndarray, se: np.ndarray) -> SEProbe:
        self.pipe.fit(X, se)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipe.predict(X)
