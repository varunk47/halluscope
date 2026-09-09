"""Open-weight model loading tuned for one 8 GB GPU.

Qwen3.5 and Gemma 4 ship as multimodal checkpoints. We load their text-only
causal LM classes so the vision towers never touch memory. 4B-class models use
nf4 through bitsandbytes; 3B and smaller load in bf16.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv

from halluscope.config import ModelSpec

load_dotenv()

_LOADED: dict[str, LoadedModel] = {}


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    spec: ModelSpec
    n_layers: int
    hidden: int

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


def _text_config(config):
    return getattr(config, "text_config", None) or config


def _causal_lm_class(model_id: str):
    """Pick the text-only class when the checkpoint is multimodal."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    model_type = getattr(cfg, "model_type", "")
    if model_type in ("qwen3_5", "qwen3_5_moe"):
        from transformers import Qwen3_5ForCausalLM  # type: ignore[attr-defined]

        return Qwen3_5ForCausalLM, cfg
    if model_type in ("gemma4",):
        try:
            from transformers import Gemma4ForCausalLM  # type: ignore[attr-defined]

            return Gemma4ForCausalLM, cfg
        except ImportError:
            pass
    return AutoModelForCausalLM, cfg


def quantized_dir(spec: ModelSpec) -> Path:
    """Where a one-time nf4 export of this model lives (small shards, cheap to load)."""
    root = Path(os.environ.get("HALLUSCOPE_QUANT_DIR", "D:/dev-cache/quant"))
    return root / (spec.id.replace("/", "_") + "-nf4")


def export_quantized(spec: ModelSpec, max_shard_size: str = "500MB") -> Path:
    """Load once with on-the-fly nf4 and save the quantized weights in small shards.

    Loading the original bf16 checkpoint reads every shard into RAM on Windows;
    after this export, loads need only the ~3 GB quantized files."""
    loaded = load_model(spec, prefer_quantized_dir=False)
    out = quantized_dir(spec)
    out.mkdir(parents=True, exist_ok=True)
    loaded.model.save_pretrained(str(out), max_shard_size=max_shard_size, safe_serialization=True)
    loaded.tokenizer.save_pretrained(str(out))
    return out


def load_model(
    spec: ModelSpec, device: str = "auto", prefer_quantized_dir: bool = True
) -> LoadedModel:
    if spec.id in _LOADED:
        return _LOADED[spec.id]

    from transformers import AutoTokenizer, BitsAndBytesConfig

    token = os.environ.get("HF_TOKEN") or None
    source = spec.id
    qdir = quantized_dir(spec)
    use_cuda = torch.cuda.is_available() and device != "cpu"
    from_quantized_dir = (
        prefer_quantized_dir
        and spec.quant == "nf4"
        and use_cuda
        and (qdir / "config.json").exists()
    )
    if from_quantized_dir:
        source = str(qdir)

    tokenizer = AutoTokenizer.from_pretrained(source, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    cls, cfg = _causal_lm_class(source)

    # Windows: avoid transformers' whole-shard reads by streaming tensors ourselves.
    stream = (
        os.environ.get("HALLUSCOPE_STREAM_LOAD", "1" if sys.platform == "win32" else "0") == "1"
    )
    if stream and not from_quantized_dir and use_cuda:
        from huggingface_hub import snapshot_download

        from halluscope.models.streaming import load_streaming, shard_files

        snap = Path(
            snapshot_download(spec.id, token=token, allow_patterns=["*.safetensors", "*.json"])
        )
        dtype = (
            torch.bfloat16
            if spec.quant in ("bf16", "nf4")
            else torch.float16
            if spec.quant == "fp16"
            else torch.float32
        )
        model = load_streaming(
            cls, _text_config(cfg), shard_files(snap), device="cuda", dtype=dtype, quant=spec.quant
        )
        tcfg = _text_config(model.config)
        loaded = LoadedModel(
            model=model,
            tokenizer=tokenizer,
            spec=spec,
            n_layers=int(tcfg.num_hidden_layers),
            hidden=int(tcfg.hidden_size),
        )
        _LOADED[spec.id] = loaded
        return loaded

    kwargs: dict = {"token": token}
    if from_quantized_dir:
        kwargs["device_map"] = "auto"
    elif spec.quant == "nf4" and use_cuda:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = "auto"
    else:
        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "nf4": torch.bfloat16,
        }[spec.quant]
        if not use_cuda and dtype != torch.float32:
            dtype = torch.float32
        kwargs["dtype"] = dtype
        kwargs["device_map"] = "auto" if use_cuda else None

    model = cls.from_pretrained(source, **kwargs)
    if not use_cuda:
        model = model.to("cpu")
    model.eval()

    tcfg = _text_config(model.config)
    loaded = LoadedModel(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        n_layers=int(tcfg.num_hidden_layers),
        hidden=int(tcfg.hidden_size),
    )
    _LOADED[spec.id] = loaded
    return loaded


def unload_all() -> None:
    _LOADED.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
