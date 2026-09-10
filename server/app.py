"""FastAPI server for the HalluScope UI.

Routes:
  GET  /api/health                      status, loaded model, gate info
  GET  /api/models                      configured models and which are cached
  GET  /api/items?status=&topic=&limit= dataset items for review and browsing
  PATCH /api/items/{id}                 review decision {status, turns?, gap?, ...}
  GET  /api/results                     every results/*.json summarized
  GET  /api/results/{name}              one results file, full
  GET  /api/provenance                  versions, seeds, cost log totals
  POST /api/score                       {turns, model_key?} -> per-layer probe, gate, uq grid
  POST /api/chat                        SSE stream: gate decision, then clarifying question or answer

The scorer loads the model lazily on first use and keeps it in memory.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from halluscope import __version__
from halluscope.config import get_settings
from server.routes import items as items_routes
from server.routes import results as results_routes
from server.routes import score as score_routes

app = FastAPI(title="HalluScope", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(items_routes.router, prefix="/api")
app.include_router(results_routes.router, prefix="/api")
app.include_router(score_routes.router, prefix="/api")


class Health(BaseModel):
    ok: bool
    version: str
    model_loaded: str | None
    gate: dict


@app.get("/api/health", response_model=Health)
def health() -> Health:
    st = score_routes.state()
    return Health(ok=True, version=__version__, model_loaded=st.model_id, gate=st.gate_info)


@app.get("/api/models")
def models() -> dict:
    cfg = get_settings()
    from halluscope.capture.cache import ActivationCache

    cache = ActivationCache(cfg.resolved_cache_dir() / "activations")
    out = {}
    for key, spec in cfg.models.items():
        try:
            n = len(cache.list_items(spec.id))
        except Exception:  # noqa: BLE001
            n = 0
        out[key] = {
            "id": spec.id,
            "family": spec.family,
            "quant": spec.quant,
            "captured_item_turns": n,
        }
    return {"primary": cfg.primary_model, "models": out}


@app.get("/api/provenance")
def provenance() -> dict:
    cfg = get_settings()
    import torch
    import transformers

    cost_rows = []
    if Path(cfg.paths.cost_log).exists():
        with open(cfg.paths.cost_log, encoding="utf-8") as fh:
            cost_rows = [json.loads(line) for line in fh if line.strip()]
    by_alias: dict[str, dict] = {}
    for r in cost_rows:
        a = by_alias.setdefault(r["alias"], {"calls": 0, "ok": 0, "cost_usd": 0.0, "models": set()})
        a["calls"] += 1
        a["ok"] += int(r.get("ok", False))
        a["cost_usd"] += float(r.get("cost_usd", 0.0))
        a["models"].add(r.get("model"))
    for a in by_alias.values():
        a["models"] = sorted(m for m in a["models"] if m)
        a["cost_usd"] = round(a["cost_usd"], 4)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": cfg.split.seed,
        "split_fractions": list(cfg.split.fractions),
        "n_bootstrap": cfg.probe.n_bootstrap,
        "gate_alpha": cfg.gate.alpha,
        "judge_aliases": {k: v.models for k, v in cfg.judge.aliases.items()},
        "llm_calls": by_alias,
        "total_cost_usd": round(sum(a["cost_usd"] for a in by_alias.values()), 4),
    }


# Serve the built UI when present (ui/dist), API routes take precedence.
_dist = Path(__file__).resolve().parents[1] / "ui" / "dist"
if _dist.exists():
    from fastapi.responses import FileResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException

    class SpaFiles(StaticFiles):
        """Static files, with the app shell for any path the router owns.

        The UI is a single page whose routes live in the browser. Without this,
        a refresh on /results, or a pasted link to one, answers 404.
        """

        async def get_response(self, path: str, scope):
            try:
                return await super().get_response(path, scope)
            except StarletteHTTPException as e:
                if e.status_code == 404 and "." not in Path(path).name:
                    return FileResponse(_dist / "index.html")
                raise

    app.mount("/", SpaFiles(directory=str(_dist), html=True), name="ui")
else:

    @app.get("/")
    def root() -> dict:
        raise HTTPException(
            status_code=404,
            detail="UI not built; run `cd ui && npm run build`, or use the Vite dev server on :5173",
        )
