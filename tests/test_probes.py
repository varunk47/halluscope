import numpy as np
import pytest

from halluscope.probes.linear import LinearProbe, MassMeanProbe
from halluscope.probes.mlp import MLPProbe
from halluscope.probes.sweep import layer_sweep


def _separable(n=120, d=16, seed=0, shift=6.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, d))
    X[:, 0] += shift * y
    return X.astype(np.float32), y


@pytest.mark.parametrize("probe_cls", [LinearProbe, MassMeanProbe, MLPProbe])
def test_probes_separate_synthetic_data(probe_cls):
    X, y = _separable()
    p = probe_cls().fit(X, y)
    prob = p.predict_proba(X)
    assert prob.shape == (len(y),)
    assert np.mean((prob > 0.5) == y) > 0.95
    assert np.corrcoef(p.decision(X), prob)[0, 1] > 0.9


def test_massmean_direction_points_along_signal():
    X, y = _separable(d=8)
    p = MassMeanProbe().fit(X, y)
    assert abs(p.direction[0]) > 0.8
    assert abs(np.linalg.norm(p.raw_direction()) - 1.0) < 1e-6


def test_layer_sweep_selects_on_validation_only():
    """Layer 2 is best on validation, layer 0 is best on test; selection must pick 2."""
    rng = np.random.default_rng(1)
    n_tr, n_va, n_te, d = 200, 80, 80, 8
    y_tr, y_va, y_te = (rng.integers(0, 2, n) for n in (n_tr, n_va, n_te))
    test_reads: dict[int, int] = {}

    def source(layer):
        def feats(y, strength, seed):
            r = np.random.default_rng(seed)
            X = r.normal(0, 1, (len(y), d))
            X[:, 0] += strength * y
            return X

        strength_tr = {0: 1.0, 1: 1.0, 2: 1.0}[layer]
        strength_va = {0: 0.0, 1: 0.5, 2: 3.0}[layer]
        strength_te = {0: 3.0, 1: 0.5, 2: 0.5}[layer]
        test_reads[layer] = test_reads.get(layer, 0) + 1
        return (
            feats(y_tr, strength_tr, 10 + layer),
            feats(y_va, strength_va, 20 + layer),
            feats(y_te, strength_te, 30 + layer),
        )

    res = layer_sweep(
        source,
        y_tr,
        y_va,
        y_te,
        layers=[0, 1, 2],
        probe_factory=LinearProbe,
        n_bootstrap=20,
        record_test_curve=False,
    )
    assert res.best_layer == 2
    assert res.test.auroc_lo <= res.test.auroc <= res.test.auroc_hi
    assert set(res.per_layer_val_auroc) == {0, 1, 2}
    assert res.per_layer_test_auroc == {}
    assert len(res.test_prob) == n_te
