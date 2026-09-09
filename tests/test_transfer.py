import numpy as np

from halluscope.probes.transfer import linear_cka, matched_layer


def test_matched_layer_scales_relative_depth():
    assert matched_layer(16, 33, 43) == 21
    assert matched_layer(0, 33, 43) == 0
    assert matched_layer(32, 33, 43) == 42


def test_linear_cka_identity_and_rotation_invariance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    assert abs(linear_cka(X, X) - 1.0) < 1e-6
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert abs(linear_cka(X, X @ Q) - 1.0) < 1e-6
    Y = rng.normal(size=(50, 8))
    assert linear_cka(X, Y) < 0.5
