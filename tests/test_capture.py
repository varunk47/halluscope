import numpy as np
import pytest

from halluscope.capture.activations import Capture
from halluscope.capture.cache import ActivationCache, CaptureKey, model_slug


def _cap(L=5, H=8, seed=0):
    rng = np.random.default_rng(seed)
    return Capture(
        last=rng.normal(size=(L, H)).astype(np.float16),
        mean_user=rng.normal(size=(L, H)).astype(np.float16),
        n_tokens=12,
        prompt_sha="abc123",
    )


def test_cache_roundtrip_and_matrix_order(tmp_path):
    cache = ActivationCache(tmp_path)
    caps = {f"it-{i}": _cap(seed=i) for i in range(3)}
    for iid, cap in caps.items():
        cache.put(CaptureKey("Qwen/Qwen3.5-0.8B", iid, 1), cap, extra={"topic": "rag"})
    got = cache.get(CaptureKey("Qwen/Qwen3.5-0.8B", "it-1", 1))
    np.testing.assert_array_equal(got.last, caps["it-1"].last)
    assert got.n_tokens == 12 and got.prompt_sha == "abc123"
    assert cache.metadata(CaptureKey("Qwen/Qwen3.5-0.8B", "it-1", 1))["topic"] == "rag"

    M = cache.matrix("Qwen/Qwen3.5-0.8B", ["it-2", "it-0"], 1, layer=3, pooling="mean_user")
    assert M.shape == (2, 8) and M.dtype == np.float32
    np.testing.assert_allclose(M[0], caps["it-2"].mean_user[3].astype(np.float32))
    assert cache.n_layers("Qwen/Qwen3.5-0.8B") == 5
    assert cache.list_items("Qwen/Qwen3.5-0.8B") == [("it-0", 1), ("it-1", 1), ("it-2", 1)]


def test_missing_key_raises(tmp_path):
    cache = ActivationCache(tmp_path)
    assert not cache.has(CaptureKey("m", "x", 1))
    with pytest.raises(KeyError):
        cache.get(CaptureKey("m", "x", 1))


def test_model_slug():
    assert model_slug("Qwen/Qwen3.5-4B") == "Qwen_Qwen3.5-4B"


@pytest.mark.network
def test_capture_prefix_shapes_on_tiny_model():
    from halluscope.capture.activations import capture_prefix
    from halluscope.config import Settings
    from halluscope.data.schema import Turn
    from halluscope.models.loader import load_model

    tiny = load_model(Settings().models["tiny"], device="cpu")
    turns = [
        Turn(role="user", content="We index docs with bge-m3."),
        Turn(role="assistant", content="Noted."),
        Turn(role="user", content="Write the retrieval code."),
    ]
    cap = capture_prefix(tiny, turns)
    assert cap.last.shape == (tiny.n_layers + 1, tiny.hidden)
    assert cap.mean_user.shape == (tiny.n_layers + 1, tiny.hidden)
    assert cap.last.dtype == np.float16 and cap.n_tokens > 0
    assert not np.isnan(cap.mean_user.astype(np.float32)).any()
