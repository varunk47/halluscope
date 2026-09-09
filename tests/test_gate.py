import numpy as np

from halluscope.gate.conformal import conformal_threshold, false_flag_rate, gate_decisions


def test_conformal_threshold_controls_false_flag_rate():
    rng = np.random.default_rng(0)
    alpha = 0.10
    rates = []
    for _ in range(200):
        cal = rng.normal(0, 1, 300)
        thr = conformal_threshold(cal, alpha=alpha)
        fresh = rng.normal(0, 1, 500)
        rates.append(false_flag_rate(fresh, thr))
    assert np.mean(rates) <= alpha + 0.01
    assert np.mean(rates) >= alpha - 0.03


def test_conformal_threshold_infinite_when_too_few_points():
    assert conformal_threshold(np.array([0.1, 0.2]), alpha=0.05) == float("inf")


def test_gate_decisions_shape():
    d = gate_decisions(np.array([0.1, 0.9, 0.5]), 0.5)
    assert d.tolist() == [0, 1, 0]
