"""Tensor-by-tensor checkpoint loading.

On Windows, transformers reads every safetensors shard fully into memory when
it opens them, so a 9 GB checkpoint needs 9 GB of commit charge even when the
weights are headed for the GPU. This loader parses the safetensors header
itself and reads one tensor at a time, so the peak is one tensor plus the
model on the GPU. Optional nf4 quantization happens per tensor on the fly.

Handles the Qwen3.5 and Gemma 4 multimodal layouts by keeping only the
language model weights and renaming them to the text-only class layout.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# checkpoint prefix -> model prefix; anything not matched is dropped
_RENAMES = (
    (re.compile(r"^model\.language_model\."), "model."),
    (re.compile(r"^language_model\.model\."), "model."),
    (re.compile(r"^language_model\.lm_head\."), "lm_head."),
    (
        re.compile(
            r"^model\.(?!visual|vision_tower|audio_tower|mtp|multi_modal_projector|embed_vision|embed_audio)"
        ),
        "model.",
    ),
    (re.compile(r"^lm_head\."), "lm_head."),
)
_DROP = re.compile(
    r"^(mtp\.|model\.visual|model\.vision_tower|model\.audio_tower|model\.multi_modal_projector|model\.embed_vision|model\.embed_audio|vision_tower|audio_tower|multi_modal_projector)"
)


def rename_key(key: str) -> str | None:
    if _DROP.match(key):
        return None
    for pat, repl in _RENAMES:
        if pat.match(key):
            return pat.sub(repl, key, count=1)
    return None


def iter_safetensors(path: Path | str) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (name, tensor) reading one tensor at a time from a safetensors file."""
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
        base = 8 + n
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            fh.seek(base + start)
            buf = bytearray(fh.read(end - start))
            dtype = _DTYPES[info["dtype"]]
            shape = info["shape"]
            if end - start == 0:
                t = torch.empty(shape, dtype=dtype)
            else:
                t = torch.frombuffer(buf, dtype=dtype).reshape(shape)
            yield name, t
            del buf


def shard_files(snapshot_dir: Path) -> list[Path]:
    files = sorted(snapshot_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {snapshot_dir}")
    return files


def _quantize_linear_modules(
    model: torch.nn.Module, compute_dtype: torch.dtype, skip: tuple[str, ...]
) -> set[str]:
    """Swap nn.Linear for bnb Linear4bit (on meta) and return their weight names."""
    import bitsandbytes as bnb

    names: set[str] = set()
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, torch.nn.Linear) or any(s in name for s in skip):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        with init_empty_weights():
            new = bnb.nn.Linear4bit(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
                compute_dtype=compute_dtype,
                quant_type="nf4",
                compress_statistics=True,
            )
        setattr(parent, child, new)
        names.add(f"{name}.weight")
    return names


def load_streaming(
    cls,
    text_config,
    files: list[Path],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    quant: str = "bf16",
) -> torch.nn.Module:
    with init_empty_weights(include_buffers=False):
        model = cls(text_config)
    keep_fp32 = tuple(getattr(cls, "_keep_in_fp32_modules", None) or ())
    quant_names: set[str] = set()
    if quant == "nf4":
        quant_names = _quantize_linear_modules(model, dtype, skip=("lm_head",))

    expected = set(model.state_dict().keys())
    loaded: set[str] = set()
    for f in files:
        for raw_name, t in iter_safetensors(f):
            name = rename_key(raw_name)
            if name is None or name not in expected:
                continue
            if name in quant_names:
                import bitsandbytes as bnb

                mod_name = name[: -len(".weight")]
                mod = model.get_submodule(mod_name)
                mod.weight = bnb.nn.Params4bit(
                    t.to(dtype), requires_grad=False, quant_type="nf4", compress_statistics=True
                ).to(device)
            else:
                target = dtype
                if not t.is_floating_point():
                    target = t.dtype
                elif any(k in name for k in keep_fp32):
                    target = torch.float32
                set_module_tensor_to_device(model, name, device, value=t, dtype=target)
            loaded.add(name)
            del t
    missing = expected - loaded
    if "lm_head.weight" in missing and getattr(text_config, "tie_word_embeddings", False):
        model.lm_head.weight = model.model.embed_tokens.weight
        missing.discard("lm_head.weight")
    if missing:
        raise RuntimeError(
            f"streaming load left {len(missing)} tensors unloaded, e.g. {sorted(missing)[:5]}"
        )
    # buffers (rotary tables etc.) were created on CPU; move them
    for mod in model.modules():
        for bname, buf in list(mod.named_buffers(recurse=False)):
            if buf.device.type != torch.device(device).type:
                mod._buffers[bname] = buf.to(device)
    if quant == "nf4":
        model.is_loaded_in_4bit = True
        model.is_quantized = True
    model.eval()
    return model
