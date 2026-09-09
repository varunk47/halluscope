import numpy as np

from halluscope.models.chat import Generation
from halluscope.uq.eigenscore import eigen_spectrum, eigenscore
from halluscope.uq.entropy import predictive_entropy
from halluscope.uq.semantic_entropy import (
    cluster_by_equivalence,
    exact_match_equivalent,
    semantic_entropy,
)


def test_eigenscore_identical_lower_than_orthogonal():
    rng = np.random.default_rng(0)
    v = rng.normal(size=64)
    same = np.stack([v + 1e-3 * rng.normal(size=64) for _ in range(6)])
    ortho = np.linalg.qr(rng.normal(size=(64, 6)))[0].T * 10
    assert eigenscore(same) < eigenscore(ortho)
    spec = eigen_spectrum(ortho)
    assert spec.shape == (6,) and np.all(np.diff(spec) <= 1e-9)


def test_predictive_entropy_of_certain_generation_is_zero():
    g = Generation(text="x", token_ids=[1, 2, 3], logprobs=[0.0, 0.0, 0.0])
    assert predictive_entropy(g) == 0.0
    g2 = Generation(text="x", token_ids=[1, 2], logprobs=[-1.0, -3.0])
    assert predictive_entropy(g2) == 2.0
    assert np.isnan(predictive_entropy(Generation(text="", token_ids=[], logprobs=[])))


def test_semantic_entropy_clusters():
    texts = ["Use k=5", "use K=5", "Use k=8", "Use k=5 "]
    assign = cluster_by_equivalence(texts, exact_match_equivalent)
    assert assign == [0, 0, 1, 0]
    h = semantic_entropy(texts, exact_match_equivalent)
    p = np.array([0.75, 0.25])
    assert abs(h - float(-(p * np.log(p)).sum())) < 1e-9
    assert semantic_entropy(["a", "a", "a"], exact_match_equivalent) == 0.0
