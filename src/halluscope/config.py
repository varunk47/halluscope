"""Typed configuration.

Precedence, highest first: explicit overrides dict, environment variables
prefixed with ``HALLUSCOPE_`` (nested keys use ``__``), a YAML file, defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env before any Hugging Face import so HF_HOME points at the big disk.
load_dotenv()

Quant = Literal["nf4", "bf16", "fp16", "fp32"]
Pooling = Literal["last", "mean_user"]


class ModelSpec(BaseModel):
    id: str
    family: str
    quant: Quant = "nf4"
    text_only: bool = True
    max_context: int = 4096


class Paths(BaseModel):
    root: Path = Path(".")
    cache_dir: Path = Path("cache")
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    seeds_dir: Path = Path("src/halluscope/data/seeds")
    figures_dir: Path = Path("docs/figures")
    cost_log: Path = Path("logs/llm_cost.jsonl")


class CaptureCfg(BaseModel):
    poolings: list[Pooling] = ["last", "mean_user"]
    dtype: Literal["float16", "float32"] = "float16"
    batch_size: int = 1


class ProbeCfg(BaseModel):
    C: float = 1.0
    n_bootstrap: int = 1000
    seed: int = 0
    mlp_hidden: int = 256
    ece_bins: int = 15


class GenCfg(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95


class UQCfg(BaseModel):
    K: int = 8
    seed: int = 0
    gen: GenCfg = GenCfg()
    methods: list[str] = [
        "predictive_entropy",
        "semantic_entropy",
        "eigenscore",
        "ptrue",
        "verbalized",
        "logit_lens_entropy",
        "sep",
    ]


class AliasCfg(BaseModel):
    """One logical role mapped to an ordered list of LiteLLM model strings."""

    models: list[str]
    temperature: float = 0.0
    max_tokens: int = 1024


class JudgeCfg(BaseModel):
    aliases: dict[str, AliasCfg] = {
        # Primary judge: OpenAI. Secondary judge must be a different family for
        # cross-family agreement; it falls back to an older OpenAI model only when
        # no Anthropic or Gemini key is present, and the report flags that.
        "judge_primary": AliasCfg(models=["openai/gpt-5.1", "openai/gpt-5"]),
        "judge_secondary": AliasCfg(
            models=["anthropic/claude-sonnet-5", "gemini/gemini-2.5-pro", "openai/gpt-4.1"]
        ),
        "simulated_user": AliasCfg(
            models=["openai/gpt-5-mini", "openai/gpt-4.1-mini"], temperature=0.3
        ),
        "augmenter": AliasCfg(
            models=["anthropic/claude-sonnet-5", "openai/gpt-5.1", "openai/gpt-5"],
            temperature=0.9,
            max_tokens=4096,
        ),
    }
    max_retries: int = 3
    human_subset: int = 120


class GateCfg(BaseModel):
    alpha: float = 0.10
    max_rounds: int = 2


class SplitCfg(BaseModel):
    fractions: tuple[float, float, float] = (0.6, 0.15, 0.25)
    seed: int = 0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HALLUSCOPE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    models: dict[str, ModelSpec] = {
        "qwen": ModelSpec(id="Qwen/Qwen3.5-4B", family="qwen", quant="nf4"),
        "qwen2b": ModelSpec(id="Qwen/Qwen3.5-2B", family="qwen", quant="bf16"),
        "gemma": ModelSpec(id="google/gemma-4-E4B-it", family="gemma", quant="nf4"),
        "llama": ModelSpec(id="meta-llama/Llama-3.2-3B-Instruct", family="llama", quant="bf16"),
        "tiny": ModelSpec(id="Qwen/Qwen3.5-0.8B", family="qwen", quant="bf16"),
    }
    primary_model: str = "qwen"
    paths: Paths = Paths()
    capture: CaptureCfg = CaptureCfg()
    probe: ProbeCfg = ProbeCfg()
    uq: UQCfg = UQCfg()
    judge: JudgeCfg = JudgeCfg()
    gate: GateCfg = GateCfg()
    split: SplitCfg = SplitCfg()
    cache_dir: Path | None = Field(default=None, description="Overrides paths.cache_dir")

    def resolved_cache_dir(self) -> Path:
        return self.cache_dir if self.cache_dir is not None else self.paths.cache_dir

    def model_spec(self, key: str | None = None) -> ModelSpec:
        return self.models[key or self.primary_model]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(
    overrides: dict[str, Any] | None = None,
    yaml_path: Path | str | None = None,
) -> Settings:
    """Build settings from YAML, environment, then explicit overrides."""
    data: dict[str, Any] = {}
    if yaml_path is not None:
        with open(yaml_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return Settings(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    default_yaml = Path("configs/default.yaml")
    return load_settings(yaml_path=default_yaml if default_yaml.exists() else None)
