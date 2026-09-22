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
    import gc
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

    from halluscope.models.streaming import load_streaming, shard_files

    mid = "Qwen/Qwen3.5-0.8B"
    snap = Path(snapshot_download(mid, allow_patterns=["*.safetensors", "*.json"]))
    cfg = AutoConfig.from_pretrained(mid)
    tcfg = getattr(cfg, "text_config", cfg)
    tok = AutoTokenizer.from_pretrained(mid)
    enc = tok("Chunk the docs at 512 tokens.", return_tensors="pt")

    # Hold one model at a time. Two float32 copies of a 0.8B checkpoint is 6.4 GB
    # of weights before transformers opens a single shard, which is more commit
    # charge than this machine has, and holding both is not what the test is
    # about. Score each model, keep the logits, drop the weights.
    streamed = load_streaming(
        Qwen3_5ForCausalLM, tcfg, shard_files(snap), device="cpu", dtype=torch.float32
    )
    with torch.no_grad():
        a = streamed(**enc).logits.clone()
    del streamed
    gc.collect()

    # The reference path opens every shard on CPU at once, which is the whole
    # reason load_streaming exists. If it still runs out of room with nothing
    # else resident, the machine is at fault rather than the loader, so skip
    # rather than report a false failure.
    try:
        ref = Qwen3_5ForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
    except OSError as e:
        if "paging file" in str(e) or getattr(e, "winerror", None) == 1455:
            pytest.skip(f"stock from_pretrained ran out of commit charge: {e}")
        raise
    with torch.no_grad():
        b = ref(**enc).logits
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-3, rtol=1e-3)
