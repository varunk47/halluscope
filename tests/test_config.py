from pathlib import Path

from halluscope.config import Settings, load_settings


def test_defaults_have_primary_qwen():
    s = Settings()
    assert s.primary_model == "qwen"
    assert s.model_spec().id == "Qwen/Qwen3.5-4B"
    assert s.model_spec().quant == "nf4"
    assert s.uq.K == 6


def test_yaml_override(tmp_path: Path):
    y = tmp_path / "c.yaml"
    y.write_text("uq:\n  K: 3\ngate:\n  alpha: 0.2\n", encoding="utf-8")
    s = load_settings(yaml_path=y)
    assert s.uq.K == 3
    assert s.gate.alpha == 0.2
    assert s.probe.n_bootstrap == 1000


def test_explicit_override_beats_yaml(tmp_path: Path):
    y = tmp_path / "c.yaml"
    y.write_text("uq:\n  K: 3\n", encoding="utf-8")
    s = load_settings(overrides={"uq": {"K": 11}}, yaml_path=y)
    assert s.uq.K == 11


def test_env_override(monkeypatch):
    monkeypatch.setenv("HALLUSCOPE_CACHE_DIR", "somewhere/else")
    s = Settings()
    assert s.resolved_cache_dir() == Path("somewhere/else")


def test_env_beats_yaml_for_nested_field(tmp_path: Path, monkeypatch):
    y = tmp_path / "c.yaml"
    y.write_text("paths:\n  results_dir: from_yaml\nuq:\n  K: 3\n", encoding="utf-8")
    monkeypatch.setenv("HALLUSCOPE_PATHS__RESULTS_DIR", "from_env")
    s = load_settings(yaml_path=y)
    assert s.paths.results_dir == Path("from_env")
    assert s.uq.K == 3  # untouched yaml value survives
