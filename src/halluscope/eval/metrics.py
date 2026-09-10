"""Metrics with bootstrap confidence intervals.

Convention everywhere: ``y_true`` is 1 for the positive class (the item should
be flagged) and ``score`` is higher when the method thinks it should be
flagged. Calibration metrics take probabilities in [0, 1].
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, score))


def accuracy(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((np.asarray(prob) >= threshold).astype(int) == np.asarray(y_true)))


def ece(y_true: np.ndarray, prob: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width bins."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = len(prob)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (prob >= lo) & (prob < hi) if hi < 1.0 else (prob >= lo) & (prob <= hi)
        if mask.sum() == 0:
            continue
        conf = prob[mask].mean()
        acc = y_true[mask].mean()
        total += (mask.sum() / n) * abs(acc - conf)
    return float(total)


def reliability_curve(
    y_true: np.ndarray, prob: np.ndarray, bins: int = 10
) -> dict[str, list[float]]:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    conf, acc, count = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (prob >= lo) & (prob < hi) if hi < 1.0 else (prob >= lo) & (prob <= hi)
        if mask.sum() == 0:
            continue
        conf.append(float(prob[mask].mean()))
        acc.append(float(y_true[mask].mean()))
        count.append(int(mask.sum()))
    return {"confidence": conf, "accuracy": acc, "count": count}


def bootstrap_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    score: np.ndarray,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (point, lower, upper) percentile bootstrap over items."""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    point = fn(y_true, score)
    rng = np.random.default_rng(seed)
    N = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        v = fn(y_true[idx], score[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def paired_bootstrap_delta(
    fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap the difference fn(a) - fn(b) resampling both on the same items.

    Two separate intervals overlapping does not mean the difference is
    indistinguishable from zero: the scores are computed on the same test items
    and are correlated, so the comparison has to be paired to say anything. The
    p value is the two-sided fraction of resamples on the wrong side of zero.
    """
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    point = fn(y_true, score_a) - fn(y_true, score_b)
    rng = np.random.default_rng(seed)
    N = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        va, vb = fn(y_true[idx], score_a[idx]), fn(y_true[idx], score_b[idx])
        if not (np.isnan(va) or np.isnan(vb)):
            vals.append(va - vb)
    if not vals:
        return {"delta": point, "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    arr = np.array(vals)
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    frac = float(np.mean(arr <= 0)) if point > 0 else float(np.mean(arr >= 0))
    return {
        "delta": float(point),
        "lo": float(lo),
        "hi": float(hi),
        "p": float(min(1.0, 2 * frac)),
        "n": int(N),
    }


@dataclass
class MetricsWithCI:
    n: int
    auroc: float
    auroc_lo: float
    auroc_hi: float
    auprc: float
    auprc_lo: float
    auprc_hi: float
    accuracy: float | None = None
    ece: float | None = None
    reliability: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_scores(
    y_true: np.ndarray,
    score: np.ndarray,
    prob: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
    ece_bins: int = 15,
) -> MetricsWithCI:
    a, a_lo, a_hi = bootstrap_ci(auroc, y_true, score, n=n_bootstrap, seed=seed)
    p, p_lo, p_hi = bootstrap_ci(auprc, y_true, score, n=n_bootstrap, seed=seed)
    m = MetricsWithCI(
        n=int(len(y_true)),
        auroc=a,
        auroc_lo=a_lo,
        auroc_hi=a_hi,
        auprc=p,
        auprc_lo=p_lo,
        auprc_hi=p_hi,
    )
    if prob is not None:
        m.accuracy = accuracy(y_true, prob)
        m.ece = ece(y_true, prob, bins=ece_bins)
        m.reliability = reliability_curve(y_true, prob)
    return m
