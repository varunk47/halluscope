"""Inter-judge and human-judge agreement."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cohens_kappa(a: Sequence, b: Sequence) -> float:
    a = list(a)
    b = list(b)
    if len(a) != len(b) or not a:
        raise ValueError("sequences must be non-empty and equal length")
    labels = sorted(set(a) | set(b))
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    M = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        M[idx[x], idx[y]] += 1
    n = M.sum()
    po = np.trace(M) / n
    pe = float((M.sum(axis=1) @ M.sum(axis=0)) / (n * n))
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))


def agreement_rate(a: Sequence, b: Sequence) -> float:
    a, b = list(a), list(b)
    return float(np.mean([x == y for x, y in zip(a, b, strict=True)]))
