"""Multi-turn simulated-user evaluation of the clarify gate.

Conditions:
- ``off``     assistant answers immediately
- ``gate``    assistant asks one clarifying question when the probe gate fires
- ``always``  oracle that always asks first
- ``prompt``  no probe; the system prompt tells the model to ask if unsure

A simulated user (API model) holds the fully specified reference and answers
clarifying questions truthfully. Final answers are graded by two cross-family
judges against the reference. Metrics: correctness, assumption rate,
questions per task.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from rich.progress import track

from halluscope.config import get_settings
from halluscope.data.io import approved, load_items
from halluscope.data.schema import Item, Turn
from halluscope.judge.client import JudgeClient, JudgeError
from halluscope.judge.rubrics import CorrectnessVerdict, correctness_prompt
from halluscope.models.chat import SYSTEM_PROMPT, generate
from halluscope.models.loader import LoadedModel, load_model

GateFn = Callable[[Item, list[Turn]], bool]

# Whole-dialogue retries before an item is given up on. The judge client
# already retries the JSON repair internally; this covers the case where
# every one of those repairs fails and the request needs starting over.
_ITEM_ATTEMPTS = 3

ASK_INSTRUCTION = (
    "The request may be missing information or contain conflicting requirements. "
    "Ask the user exactly one concise clarifying question that targets the most "
    "important gap. Do not answer the task yet."
)
PROMPT_ONLY_SYSTEM = SYSTEM_PROMPT + (
    " If the request is missing a detail you need or contains conflicting "
    "requirements, ask one clarifying question instead of answering."
)


@dataclass
class LoopRecord:
    item_id: str
    label: str
    condition: str
    asked: int
    questions: list[str] = field(default_factory=list)
    user_replies: list[str] = field(default_factory=list)
    final_answer: str = ""
    grade: str = ""
    assumption_made: bool | None = None
    grade2: str | None = None
    assumption_made2: bool | None = None
    # Non-empty when the dialogue could not be completed, almost always because
    # a judge returned unparseable JSON on every retry. Such a record carries no
    # grade, so summarize() drops it rather than scoring it as incorrect.
    error: str = ""


def reference_request(item: Item, by_id: dict[str, Item]) -> str:
    ref = by_id.get(item.reference_specified_variant or "") if item.label != "specified" else item
    if ref is None:
        ref = item
    return "\n".join(f"{t.role}: {t.content}" for t in ref.turns)


def simulated_user_reply(client: JudgeClient, item: Item, reference: str, question: str) -> str:
    msgs = [
        {
            "role": "system",
            "content": (
                "You are the engineer who made a request to an AI assistant. You know the full "
                "specification below. The assistant asked a clarifying question. Answer it "
                "truthfully and briefly using only the specification. If the assistant asks "
                "about a conflict, say which instruction should win: the earlier specification."
            ),
        },
        {
            "role": "user",
            "content": f"Full specification:\n{reference}\n\nAssistant's question:\n{question}\n\nYour reply:",
        },
    ]
    return client.text("simulated_user", msgs, tag=f"simuser:{item.id}").strip()


def run_dialogue(
    loaded: LoadedModel,
    client: JudgeClient,
    item: Item,
    reference: str,
    condition: str,
    gate: GateFn | None,
    max_rounds: int = 2,
    max_new_tokens: int = 256,
) -> LoopRecord:
    turns = list(item.turns)
    rec = LoopRecord(item_id=item.id, label=item.label, condition=condition, asked=0)
    system = PROMPT_ONLY_SYSTEM if condition == "prompt" else SYSTEM_PROMPT

    for _ in range(max_rounds):
        should_ask = condition == "always" or (
            condition == "gate" and gate is not None and gate(item, turns)
        )
        if not should_ask:
            break
        ask_turns = turns + [Turn(role="user", content=ASK_INSTRUCTION)]
        q = generate(loaded, ask_turns, max_new_tokens=96, temperature=0.0, n=1, system=system)[
            0
        ].text
        reply = simulated_user_reply(client, item, reference, q)
        rec.asked += 1
        rec.questions.append(q)
        rec.user_replies.append(reply)
        turns = turns + [Turn(role="assistant", content=q), Turn(role="user", content=reply)]
        if condition == "always":
            break

    answer = generate(
        loaded, turns, max_new_tokens=max_new_tokens, temperature=0.0, n=1, system=system
    )[0].text
    if (
        condition == "prompt"
        and rec.asked == 0
        and answer.rstrip().endswith("?")
        and len(answer) < 400
    ):
        # the prompt-only model chose to ask; let the simulated user answer once
        reply = simulated_user_reply(client, item, reference, answer)
        rec.asked = 1
        rec.questions.append(answer)
        rec.user_replies.append(reply)
        turns = turns + [Turn(role="assistant", content=answer), Turn(role="user", content=reply)]
        answer = generate(
            loaded, turns, max_new_tokens=max_new_tokens, temperature=0.0, n=1, system=system
        )[0].text
    rec.final_answer = answer

    judged_item = Item(**{**item.model_dump(), "turns": [t.model_dump() for t in turns]})
    v = client.complete(
        "judge_primary",
        correctness_prompt(judged_item, reference, answer),
        CorrectnessVerdict,
        tag=f"loop:{item.id}",
    )
    rec.grade, rec.assumption_made = v.grade, v.assumption_made
    if "judge_secondary" in client.cfg.aliases:
        try:
            second = client.complete(
                "judge_secondary",
                correctness_prompt(judged_item, reference, answer),
                CorrectnessVerdict,
                tag=f"loop2:{item.id}",
            )
            rec.grade2, rec.assumption_made2 = second.grade, second.assumption_made
        except Exception:  # noqa: BLE001
            pass
    return rec


def load_checkpoint(path: Path) -> dict[str, LoopRecord]:
    """Finished records from a previous, interrupted run of the same condition."""
    done: dict[str, LoopRecord] = {}
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = LoopRecord(**json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue  # truncated final line from a hard kill
            done[rec.item_id] = rec
    return done


def append_checkpoint(path: Path, rec: LoopRecord) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(rec)) + "\n")


def run_item(
    loaded: LoadedModel,
    client: JudgeClient,
    item: Item,
    reference: str,
    condition: str,
    gate: GateFn | None,
    max_rounds: int = 2,
    max_new_tokens: int = 256,
    attempts: int = _ITEM_ATTEMPTS,
) -> LoopRecord:
    """One dialogue, retried on judge failure, never raising.

    A run is hours of GPU and API time. Before this existed a single judge
    returning unparseable JSON aborted the whole thing, which is how the
    2026-09-19 ``always`` run lost 159 minutes at 57% complete.
    """
    last_err: Exception | None = None
    for _ in range(attempts):
        try:
            return run_dialogue(
                loaded,
                client,
                item,
                reference,
                condition,
                gate,
                max_rounds=max_rounds,
                max_new_tokens=max_new_tokens,
            )
        except JudgeError as e:
            last_err = e
    # Record the failure rather than dropping the item silently, so the output
    # says how many tasks it could not grade.
    print(f"item {item.id} failed after {attempts} attempts: {last_err}")
    return LoopRecord(
        item_id=item.id,
        label=item.label,
        condition=condition,
        asked=0,
        error=f"{type(last_err).__name__}: {last_err}",
    )


def summarize(records: list[LoopRecord]) -> dict:
    out: dict = {}
    graded = [r for r in records if not r.error]
    for label in ("all", "specified", "underspecified", "inconsistent"):
        sub = [r for r in graded if label == "all" or r.label == label]
        if not sub:
            continue
        out[label] = {
            "n": len(sub),
            "correct": float(np.mean([r.grade == "correct" for r in sub])),
            "partial_or_better": float(np.mean([r.grade in ("correct", "partial") for r in sub])),
            "assumption_rate": float(np.mean([bool(r.assumption_made) for r in sub])),
            "questions_per_task": float(np.mean([r.asked for r in sub])),
        }
    return out


def build_probe_gate(
    model_key: str,
    results_dir: Path,
    pooling: str,
    alpha: float,
    items_path: Path | None = None,
) -> tuple[GateFn, dict]:
    """Gate from the saved linear probe result: refit at the chosen layer, conformal threshold on validation."""
    import pickle

    from halluscope.capture.activations import capture_prefix
    from halluscope.capture.cache import ActivationCache, CaptureKey, shas_for
    from halluscope.data.splits import grouped_split
    from halluscope.gate.conformal import conformal_threshold
    from halluscope.probes.linear import LinearProbe
    from halluscope.probes.run import gap_label, probe_result_path

    cfg = get_settings()
    spec = cfg.model_spec(model_key)
    items_path = items_path or cfg.paths.data_dir / "augmented" / "items.jsonl"
    with open(
        probe_result_path(results_dir, model_key, "gap", pooling, items_path), encoding="utf-8"
    ) as fh:
        res = json.load(fh)
    layer = int(res["probes"]["linear"]["best_layer"])
    cache = ActivationCache(cfg.resolved_cache_dir() / "activations")
    # Calibrate on the same build the loop judges, so layer, threshold and
    # tasks all describe one experiment.
    items = approved(load_items(items_path))
    shas = shas_for(items)
    items = [i for i in items if cache.has(CaptureKey(spec.id, i.id, i.n_user_turns, shas[i.id]))]
    splits = grouped_split(items, seed=cfg.split.seed, fractions=cfg.split.fractions)
    tr, va = splits["train"], splits["val"]
    t_idx = {i.id: i.n_user_turns for i in items}
    Xtr = cache.matrix(spec.id, [i.id for i in tr], t_idx, layer, pooling, shas=shas)
    ytr = np.array([gap_label(i) for i in tr])
    probe = LinearProbe(seed=cfg.probe.seed).fit(Xtr, ytr)
    Xva = cache.matrix(spec.id, [i.id for i in va], t_idx, layer, pooling, shas=shas)
    va_spec = np.array([i.label == "specified" for i in va])
    thr = conformal_threshold(probe.decision(Xva)[va_spec], alpha=alpha)
    loaded = load_model(spec)

    def gate(item: Item, turns: list[Turn]) -> bool:
        cap = capture_prefix(loaded, turns)
        x = cap.pooled(pooling)[layer].astype(np.float32)[None, :]
        return bool(probe.decision(x)[0] > thr)

    gate_path = results_dir / f"gate_{model_key}_{pooling}.pkl"
    with open(gate_path, "wb") as fh:
        pickle.dump({"layer": layer, "threshold": thr, "probe": probe, "pooling": pooling}, fh)
    return gate, {"layer": layer, "threshold": float(thr), "alpha": alpha}


def run_loop_cli(
    model_key: str,
    items_path: Path,
    condition: str = "gate",
    seed: int = 0,
    out_dir: Path = Path("results"),
    split: str = "test",
    limit: int | None = None,
    pooling: str = "last",
) -> Path:
    from halluscope.data.splits import grouped_split

    cfg = get_settings()
    spec = cfg.model_spec(model_key)
    all_items = approved(load_items(items_path))
    by_id = {i.id: i for i in all_items}
    items = grouped_split(all_items, seed=cfg.split.seed, fractions=cfg.split.fractions)[split]
    if limit:
        items = items[:limit]
    client = JudgeClient(cfg.judge, cost_log=cfg.paths.cost_log)
    gate, gate_info = (None, {})
    if condition == "gate":
        gate, gate_info = build_probe_gate(model_key, out_dir, pooling, cfg.gate.alpha, items_path)
    loaded = load_model(spec)

    out_dir.mkdir(parents=True, exist_ok=True)

    # A run is six hours of API calls and GPU time. Losing all of it to one
    # malformed judge response is not acceptable, so every finished dialogue is
    # appended to a checkpoint immediately and a restart picks up where it
    # stopped. Greedy decoding makes a resumed run identical to an uninterrupted
    # one, so this changes cost and not results.
    ckpt = out_dir / f".loop_{model_key}_{condition}_s{seed}.partial.jsonl"
    done = load_checkpoint(ckpt)
    if done:
        print(f"resuming from {ckpt.name}: {len(done)} of {len(items)} already done")

    records: list[LoopRecord] = []
    for it in track(items, description=f"loop {condition} seed {seed}"):
        if it.id in done:
            records.append(done[it.id])
            continue
        rec = run_item(
            loaded,
            client,
            it,
            reference_request(it, by_id),
            condition,
            gate,
            max_rounds=cfg.gate.max_rounds,
            max_new_tokens=cfg.uq.gen.max_new_tokens,
        )
        records.append(rec)
        append_checkpoint(ckpt, rec)

    failed = [r.item_id for r in records if r.error]
    path = out_dir / f"loop_{model_key}_{condition}_s{seed}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": spec.id,
                # Record the build. The gate layer is read from whichever probe
                # result matches this path, so a file that does not say which
                # build it ran on cannot be checked later without re-deriving it
                # from which families happen to appear in the split.
                "dataset": items_path.name,
                "split": split,
                "n_items": len(records),
                "n_failed": len(failed),
                "failed_ids": failed,
                "condition": condition,
                "seed": seed,
                "gate": gate_info,
                "summary": summarize(records),
                "cost": client.cost_summary(),
                "rows": [asdict(r) for r in records],
            },
            fh,
            indent=2,
        )
    ckpt.unlink(missing_ok=True)
    return path
