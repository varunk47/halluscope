"""Split conformal threshold for the clarify gate.

We want: among requests that are actually fully specified, the gate fires
(falsely asks a question) at most ``alpha`` of the time, with a finite-sample
guarantee. Treat the probe score on specified calibration items as the
nonconformity score and take the ceil((n+1)(1-alpha))/n empirical quantile.
Any new specified item then exceeds the threshold with probability at most
alpha, marginally over calibration draws.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_threshold(scores_specified_cal: np.ndarray, alpha: float = 0.10) -> float:
    s = np.sort(np.asarray(scores_specified_cal, dtype=float))
    n = len(s)
    if n == 0:
        raise ValueError("need calibration scores")
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")  # too few calibration points to certify alpha
    return float(s[k - 1])


def gate_decisions(scores: np.ndarray, threshold: float) -> np.ndarray:
    """1 = ask a clarifying question."""
    return (np.asarray(scores, dtype=float) > threshold).astype(int)


def false_flag_rate(scores_specified: np.ndarray, threshold: float) -> float:
    return float(np.mean(gate_decisions(scores_specified, threshold)))
