import numpy as np

from halluscope.eval.metrics import (
    auprc,
    auroc,
    bootstrap_ci,
    ece,
    evaluate_scores,
    reliability_curve,
)


def test_auroc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert np.isnan(auroc(np.array([1, 1]), np.array([0.1, 0.2])))


def test_ece_perfect_calibration_is_zero():
    prob = np.array([0.0] * 50 + [1.0] * 50)
    y = np.array([0] * 50 + [1] * 50)
    assert ece(y, prob) == 0.0


def test_ece_overconfident_is_high():
    prob = np.full(100, 0.95)
    y = np.array([0] * 50 + [1] * 50)
    assert abs(ece(y, prob) - 0.45) < 1e-6


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = y + rng.normal(0, 1.0, 200)
    point, lo, hi = bootstrap_ci(auroc, y, s, n=200, seed=1)
    assert lo <= point <= hi
    assert 0.5 < point < 1.0


def test_evaluate_scores_fields():
    y = np.array([0, 1] * 20)
    s = np.array([0.2, 0.8] * 20)
    m = evaluate_scores(y, s, prob=s, n_bootstrap=50)
    assert m.n == 40 and m.auroc == 1.0 and m.accuracy == 1.0
    assert m.ece is not None and m.ece < 0.25
    assert set(m.reliability) == {"confidence", "accuracy", "count"}
    assert auprc(y, s) == 1.0
    rc = reliability_curve(y, s, bins=5)
    assert len(rc["count"]) == 2
