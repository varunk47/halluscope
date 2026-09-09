# HalluScope v2: working notes for agents

- Python env: `.venv` (uv, Python 3.12). Use `C:\Users\varun\.local\bin\uv.exe` on this machine; the `uv` on PATH is stale. Run tools as `.venv/Scripts/python.exe -m ...` or `uv run ...`.
- Caches live on D: (`.env` sets `HF_HOME=D:/dev-cache/hf`, `HALLUSCOPE_CACHE_DIR=D:/dev-cache/halluscope`). Do not move them to C:, which is nearly full.
- Model loading on Windows goes through `src/halluscope/models/streaming.py` (tensor-by-tensor). Do not switch back to plain `from_pretrained` for 2B+ models; it fails with a paging-file error on this laptop.
- Tests: `pytest -m "not network and not gpu"` is the CI set and must stay green. `network` tests download Qwen3.5-0.8B and run on CPU.
- Splits are grouped by family root. Never split by row. Layer selection uses validation only. Every reported metric carries a bootstrap interval.
- Judges must not share a family with the model under test. Pairwise judgments run in both orders.
- Text style for docs and UI copy: no em dashes, plain sentences.
- Results files are the contract between CLI and UI: `results/probe_*.json`, `uq_*.json`, `behavior_*.json`, `loop_*.json`. Keep their shapes stable or update `server/routes/results.py` and the UI together.
- Git hook blocks commit messages containing "no-verify"; also avoid the bare word "verify" in messages, the hook's pattern is loose.
