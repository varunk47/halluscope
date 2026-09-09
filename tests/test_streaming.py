import pytest
import torch

from halluscope.models.streaming import rename_key


def test_rename_key_maps_multimodal_layout_to_text_only():
    assert (
        rename_key("model.language_model.layers.0.mlp.up_proj.weight")
        == "model.layers.0.mlp.up_proj.weight"
    )
    assert rename_key("model.language_model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert rename_key("lm_head.weight") == "lm_head.weight"
    assert rename_key("model.layers.3.norm.weight") == "model.layers.3.norm.weight"
    assert rename_key("mtp.layers.0.self_attn.v_proj.weight") is None
    assert rename_key("model.visual.blocks.0.attn.qkv.weight") is None


@pytest.mark.network
def test_streaming_load_matches_transformers_on_cpu():
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

    from halluscope.models.streaming import load_streaming, shard_files

    mid = "Qwen/Qwen3.5-0.8B"
    snap = Path(snapshot_download(mid, allow_patterns=["*.safetensors", "*.json"]))
    cfg = AutoConfig.from_pretrained(mid)
    tcfg = getattr(cfg, "text_config", cfg)
    streamed = load_streaming(
        Qwen3_5ForCausalLM, tcfg, shard_files(snap), device="cpu", dtype=torch.float32
    )
    # The reference path is the one that cannot cope with a busy Windows box: it
    # opens every shard on CPU at once, which is the whole reason load_streaming
    # exists. When it hits that wall the machine is at fault, not the loader, so
    # skip rather than report a false failure.
    try:
        ref = Qwen3_5ForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    except OSError as e:
        if "paging file" in str(e) or getattr(e, "winerror", None) == 1455:
            pytest.skip(f"stock from_pretrained ran out of commit charge: {e}")
        raise
    tok = AutoTokenizer.from_pretrained(mid)
    enc = tok("Chunk the docs at 512 tokens.", return_tensors="pt")
    with torch.no_grad():
        a = streamed(**enc).logits
        b = ref(**enc).logits
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-3, rtol=1e-3)
