"""Open-weight model loading tuned for one 8 GB GPU.

Qwen3.5 and Gemma 4 ship as multimodal checkpoints. We load their text-only
causal LM classes so the vision towers never touch memory. 4B-class models use
nf4 through bitsandbytes; 3B and smaller load in bf16.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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


def load_model(spec: ModelSpec, device: str = "auto") -> LoadedModel:
    if spec.id in _LOADED:
        return _LOADED[spec.id]

    from transformers import AutoTokenizer, BitsAndBytesConfig

    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(spec.id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    cls, cfg = _causal_lm_class(spec.id)
    kwargs: dict = {"token": token}
    use_cuda = torch.cuda.is_available() and device != "cpu"
    if spec.quant == "nf4" and use_cuda:
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

    model = cls.from_pretrained(spec.id, **kwargs)
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
