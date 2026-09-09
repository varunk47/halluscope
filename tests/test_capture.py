import numpy as np
import pytest

from halluscope.capture.activations import Capture
from halluscope.capture.cache import (
    ActivationCache,
    AmbiguousCaptureError,
    CaptureKey,
    content_sha,
    model_slug,
    shas_for,
)
from halluscope.data.schema import Turn


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


def test_two_builds_under_one_item_id_do_not_collide(tmp_path):
    """The bug this guards: a rebuilt dataset reuses item ids for different text.

    Keyed on the id alone, the cache called the new item already captured and then
    served the old build's activations to the probe, silently.
    """
    cache = ActivationCache(tmp_path)
    free = [Turn(role="user", content="Chunk at 512 tokens with 64 overlap.")]
    minimal = [Turn(role="user", content="Chunk at a reasonable size with some overlap.")]
    a, b = content_sha(free), content_sha(minimal)
    assert a != b

    cache.put(CaptureKey("m", "rag-0001-a", 0, a), _cap(seed=1))
    # Same id, different text: not captured yet, and capturing it must not overwrite.
    assert not cache.has(CaptureKey("m", "rag-0001-a", 0, b))
    cache.put(CaptureKey("m", "rag-0001-a", 0, b), _cap(seed=2))

    np.testing.assert_array_equal(
        cache.get(CaptureKey("m", "rag-0001-a", 0, a)).last, _cap(seed=1).last
    )
    np.testing.assert_array_equal(
        cache.get(CaptureKey("m", "rag-0001-a", 0, b)).last, _cap(seed=2).last
    )
    # With both builds present, a hash-free lookup must refuse to guess.
    with pytest.raises(AmbiguousCaptureError):
        cache.get(CaptureKey("m", "rag-0001-a", 0))
    assert not cache.has(CaptureKey("m", "rag-0001-a", 0))


def test_shas_for_hashes_the_final_user_turn(tmp_path):
    from halluscope.data.schema import Item

    it = Item(
        id="rag-0001-a",
        family="rag-0001",
        topic="rag",
        variant="a",
        label="specified",
        turns=[Turn(role="user", content="Chunk at 512.")],
    )
    assert shas_for([it]) == {it.id: content_sha(it.prefix(it.n_user_turns))}


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
