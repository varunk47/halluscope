"""Live scoring: probe over every layer, conformal gate, uncertainty grid, and a
streamed reply that either asks a clarifying question or answers.

State is a module singleton: one loaded model, one fitted probe per layer
(fitted from the activation cache on first request), one gate threshold.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from halluscope.config import get_settings
from halluscope.data.schema import Turn

router = APIRouter(tags=["score"])
_lock = Lock()


@dataclass
class ScorerState:
    model_key: str | None = None
    model_id: str | None = None
    loaded: object | None = None
    probes: dict[int, object] = field(default_factory=dict)
    best_layer: int | None = None
    threshold: float | None = None
    pooling: str = "last"
    gate_info: dict = field(default_factory=dict)


_STATE = ScorerState()


def state() -> ScorerState:
    return _STATE


def _ensure(model_key: str | None = None, pooling: str = "last") -> ScorerState:
    cfg = get_settings()
    key = model_key or cfg.primary_model
    with _lock:
        if _STATE.model_key == key and _STATE.pooling == pooling and _STATE.loaded is not None:
            return _STATE
        from halluscope.capture.cache import ActivationCache
        from halluscope.data.io import approved, load_items
        from halluscope.data.splits import grouped_split
        from halluscope.eval.metrics import auroc
        from halluscope.gate.conformal import conformal_threshold
        from halluscope.models.loader import load_model
        from halluscope.probes.linear import LinearProbe
        from halluscope.probes.run import gap_label

        spec = cfg.model_spec(key)
        loaded = load_model(spec)
        cache = ActivationCache(cfg.resolved_cache_dir() / "activations")
        items_path = cfg.paths.data_dir / "augmented" / "items.jsonl"
        items = approved(load_items(items_path if items_path.exists() else cfg.paths.seeds_dir))
        try:
            have = {iid for iid, _ in cache.list_items(spec.id)}
        except (KeyError, FileNotFoundError):
            have = set()
        items = [i for i in items if i.id in have]
        probes: dict[int, LinearProbe] = {}
        best_layer, thr, gate_info = (
            None,
            None,
            {"fitted": False, "reason": "fewer than 40 cached items"},
        )
        if len(items) >= 40:
            splits = grouped_split(items, seed=cfg.split.seed, fractions=cfg.split.fractions)
            tr, va = splits["train"], splits["val"]
            t_idx = {i.id: i.n_user_turns for i in items}
            y_tr = np.array([gap_label(i) for i in tr])
            y_va = np.array([gap_label(i) for i in va])
            n_layers = cache.n_layers(spec.id)
            res_path = Path(cfg.paths.results_dir) / f"probe_{key}_gap_{pooling}.json"
            if res_path.exists():
                with open(res_path, encoding="utf-8") as fh:
                    best_layer = int(json.load(fh)["probes"]["linear"]["best_layer"])
            val_auc: dict[int, float] = {}
            for layer in range(n_layers):
                Xtr = cache.matrix(spec.id, [i.id for i in tr], t_idx, layer, pooling)
                p = LinearProbe(seed=cfg.probe.seed).fit(Xtr, y_tr)
                probes[layer] = p
                Xva = cache.matrix(spec.id, [i.id for i in va], t_idx, layer, pooling)
                val_auc[layer] = auroc(y_va, p.decision(Xva))
            if best_layer is None:
                best_layer = max(val_auc, key=lambda k: (val_auc[k], -k))
            Xva = cache.matrix(spec.id, [i.id for i in va], t_idx, best_layer, pooling)
            spec_mask = np.array([i.label == "specified" for i in va])
            s_spec = probes[best_layer].decision(Xva)[spec_mask]
            thr = conformal_threshold(s_spec, alpha=cfg.gate.alpha)
            gate_info = {
                "fitted": True,
                "layer": best_layer,
                "threshold": float(thr),
                "alpha": cfg.gate.alpha,
                "n_train": len(tr),
                "n_val": len(va),
                "val_auroc_best": float(val_auc[best_layer]),
                "val_auroc_per_layer": {str(k): float(v) for k, v in val_auc.items()},
            }
        _STATE.model_key, _STATE.model_id, _STATE.loaded = key, spec.id, loaded
        _STATE.probes, _STATE.best_layer, _STATE.threshold = probes, best_layer, thr
        _STATE.pooling, _STATE.gate_info = pooling, gate_info
        return _STATE


class ScoreRequest(BaseModel):
    turns: list[Turn] = Field(min_length=1)
    model_key: str | None = None
    pooling: str = "last"
    uq: bool = True


class LayerScore(BaseModel):
    layer: int
    prob: float
    decision: float


class ScoreResponse(BaseModel):
    model: str
    best_layer: int | None
    gate_fired: bool | None
    threshold: float | None
    prob_best: float | None
    per_layer: list[LayerScore]
    per_turn_prob_best: list[float]
    logit_lens_entropy: list[float]
    uq: dict[str, float]
    seconds: float


def _score_turns(st: ScorerState, turns: list[Turn]) -> list[LayerScore]:
    from halluscope.capture.activations import capture_prefix

    cap = capture_prefix(st.loaded, turns)
    vec = cap.pooled(st.pooling).astype(np.float32)
    out = []
    for layer, p in sorted(st.probes.items()):
        x = vec[layer][None, :]
        out.append(
            LayerScore(
                layer=layer, prob=float(p.predict_proba(x)[0]), decision=float(p.decision(x)[0])
            )
        )
    return out


def _prefix(turns: list[Turn], k: int) -> list[Turn]:
    out: list[Turn] = []
    seen = 0
    for t in turns:
        out.append(t)
        if t.role == "user":
            seen += 1
            if seen == k:
                break
    return out


@router.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if req.turns[-1].role != "user":
        raise HTTPException(400, "dialogue must end on a user turn")
    t0 = time.time()
    st = _ensure(req.model_key, req.pooling)
    per_layer = _score_turns(st, req.turns)
    best = st.best_layer
    prob_best = next((ls.prob for ls in per_layer if ls.layer == best), None)
    dec_best = next((ls.decision for ls in per_layer if ls.layer == best), None)
    fired = None if st.threshold is None or dec_best is None else bool(dec_best > st.threshold)

    per_turn: list[float] = []
    if best is not None:
        n_user = sum(1 for t in req.turns if t.role == "user")
        for k in range(1, n_user + 1):
            pl = _score_turns(st, _prefix(req.turns, k))
            per_turn.append(next(ls.prob for ls in pl if ls.layer == best))

    ll: list[float] = []
    uq: dict[str, float] = {}
    if req.uq:
        from halluscope.uq.entropy import logit_lens_entropy

        ents = logit_lens_entropy(st.loaded, req.turns)
        ll = [float(x) for x in ents]
        uq["logit_lens_entropy_last8"] = float(np.nanmean(ents[-8:]))
        try:
            from halluscope.uq.prompted import verbalized_confidence

            uq["verbalized_uncertainty"] = float(verbalized_confidence(st.loaded, req.turns))
        except Exception:  # noqa: BLE001
            pass
    return ScoreResponse(
        model=st.model_id or "",
        best_layer=best,
        gate_fired=fired,
        threshold=st.threshold,
        prob_best=prob_best,
        per_layer=per_layer,
        per_turn_prob_best=per_turn,
        logit_lens_entropy=ll,
        uq=uq,
        seconds=round(time.time() - t0, 3),
    )


class ChatRequest(BaseModel):
    turns: list[Turn] = Field(min_length=1)
    model_key: str | None = None
    force: str | None = Field(None, description="'ask' or 'answer' to override the gate")
    max_new_tokens: int = 256


@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    """SSE events: gate {...}, token {text}, done {mode, text}."""
    if req.turns[-1].role != "user":
        raise HTTPException(400, "dialogue must end on a user turn")
    st = _ensure(req.model_key)

    async def gen():
        from halluscope.gate.loop import ASK_INSTRUCTION
        from halluscope.models.chat import generate

        per_layer = await asyncio.to_thread(_score_turns, st, req.turns)
        dec_best = next((ls.decision for ls in per_layer if ls.layer == st.best_layer), None)
        prob_best = next((ls.prob for ls in per_layer if ls.layer == st.best_layer), None)
        fired = bool(dec_best is not None and st.threshold is not None and dec_best > st.threshold)
        mode = req.force or ("ask" if fired else "answer")
        yield {
            "event": "gate",
            "data": json.dumps(
                {"fired": fired, "prob": prob_best, "threshold": st.threshold, "mode": mode}
            ),
        }
        turns = list(req.turns)
        if mode == "ask":
            turns = turns + [Turn(role="user", content=ASK_INSTRUCTION)]
        gens = await asyncio.to_thread(
            generate,
            st.loaded,
            turns,
            max_new_tokens=(96 if mode == "ask" else req.max_new_tokens),
            temperature=0.0,
            n=1,
        )
        text = gens[0].text
        words = text.split(" ")
        for i in range(0, len(words), 4):
            yield {"event": "token", "data": json.dumps({"text": " ".join(words[i : i + 4]) + " "})}
            await asyncio.sleep(0.01)
        yield {"event": "done", "data": json.dumps({"mode": mode, "text": text})}

    return EventSourceResponse(gen())
