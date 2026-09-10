"""Turn results JSON into markdown tables and matplotlib figures.

Reads everything under ``results/`` and writes ``docs/figures/*.png`` plus
``docs/results.md``. Every table cell that carries a metric shows the point
estimate with its 95 percent bootstrap interval.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {
    "linear": "#2563eb",
    "massmean": "#7c3aed",
    "mlp": "#0891b2",
    "baseline": "#9ca3af",
    "accent": "#f97316",
}


def _fmt(m: dict, key: str = "auroc") -> str:
    v, lo, hi = m.get(key), m.get(f"{key}_lo"), m.get(f"{key}_hi")
    if v is None or v != v:
        return "n/a"
    if lo is None or lo != lo:
        return f"{v:.3f}"
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def _load_all(results_dir: Path, prefix: str) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob(f"{prefix}_*.json")):
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        d["_file"] = p.name
        out.append(d)
    return out


def dataset_label(r: dict) -> str:
    """Which dataset build a probe file came from.

    Newer files record it; older ones only carry it in the filename suffix, and
    a file with neither was run on the default build.
    """
    if r.get("tag"):
        return str(r["tag"])
    if r.get("dataset"):
        return str(r["dataset"]).removeprefix("items_") or "items"
    stem = Path(r.get("_file", "")).stem
    prefix = f"probe_{r.get('model_key')}_{r.get('target')}_{r.get('pooling')}_"
    return stem.removeprefix(prefix) if stem.startswith(prefix) else "items"


PAIR_NAMES = {"single_turn_ab": "single turn, a vs b", "multi_turn_cd": "multi turn, c vs d"}


def headline_table(probe_results: list[dict]) -> str:
    """Probe against the best surface baseline on identical test items, per pair.

    This is the table to read first. A bare AUROC says how separable the labels
    are; only the paired delta says whether reading the hidden state adds
    anything over reading the words.
    """
    lines = [
        "| dataset | pair | probe | probe AUROC | words AUROC | delta [95% CI] | p | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in probe_results:
        pvs = r.get("probe_vs_surface")
        if not pvs:
            continue
        surface = r["baselines"].get(pvs["baseline"], {}).get("by_pair", {})
        for probe, pairs in pvs["by_pair"].items():
            for pair, d in pairs.items():
                pa = r["probes"][probe].get("by_pair", {}).get(pair, {}).get("auroc")
                sa = surface.get(pair, {}).get("auroc")
                lines.append(
                    f"| {dataset_label(r)} | {PAIR_NAMES.get(pair, pair)} | {probe} | "
                    f"{pa:.3f} | {sa:.3f} | {d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | "
                    f"{d['p']:.3f} | {d['n']} |"
                )
    return "\n".join(lines)


def probe_table(probe_results: list[dict]) -> str:
    lines = [
        "| dataset | target | probe | layer | AUROC [95% CI] | AUPRC [95% CI] | acc | ECE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in probe_results:
        ds = dataset_label(r)
        for name, p in r["probes"].items():
            t = p["test"]
            lines.append(
                f"| {ds} | {r['target']} | {name} | {p['best_layer']}/{r['n_layers'] - 1} | "
                f"{_fmt(t)} | {_fmt(t, 'auprc')} | {t.get('accuracy', float('nan')):.3f} | {t.get('ece', float('nan')):.3f} |"
            )
        for name, b in r["baselines"].items():
            lines.append(
                f"| {ds} | {r['target']} | baseline: {name} | - | {_fmt(b)} | {_fmt(b, 'auprc')} | "
                f"{b.get('accuracy') if b.get('accuracy') is not None else float('nan'):.3f} | {b.get('ece') if b.get('ece') is not None else float('nan'):.3f} |"
            )
    return "\n".join(lines)


def loto_table(probe_results: list[dict]) -> str:
    lines = [
        "| dataset | target | held-out topic | linear probe AUROC [95% CI] | n |",
        "|---|---|---|---|---|",
    ]
    for r in probe_results:
        for topic, m in sorted(r.get("loto", {}).items()):
            lines.append(f"| {dataset_label(r)} | {r['target']} | {topic} | {_fmt(m)} | {m['n']} |")
    return "\n".join(lines)


def uq_table(uq_results: list[dict]) -> str:
    lines = [
        "| model | split | method | AUROC [95% CI] | AUPRC [95% CI] | generations | mean seconds/item |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in uq_results:
        for m, s in sorted(r["summary"].items(), key=lambda kv: -(kv[1].get("auroc") or 0)):
            lines.append(
                f"| {r['model']} | {r['split']} | {m} | {_fmt(s)} | {_fmt(s, 'auprc')} | {s.get('n_generations', 0)} | {s.get('mean_seconds', 0):.2f} |"
            )
    return "\n".join(lines)


def behavior_table(behavior_results: list[dict]) -> str:
    lines = [
        "| model | label | n | asked | flagged | silently assumed |",
        "|---|---|---|---|---|---|",
    ]
    for r in behavior_results:
        for label, s in r["summary"].items():
            if label == "judge_agreement":
                continue
            lines.append(
                f"| {r['model']} | {label} | {s['n']} | {s['asked']:.2f} | {s['flagged']:.2f} | {s['silently_assumed']:.2f} |"
            )
        ja = r["summary"].get("judge_agreement")
        if ja:
            lines.append(
                f"| {r['model']} | judge agreement (kappa, silently assumed) | {ja['n']} | | | {ja['kappa_silently_assumed']:.2f} |"
            )
    return "\n".join(lines)


def loop_table(loop_results: list[dict]) -> str:
    lines = [
        "| model | condition | seed | label | n | correct | partial or better | assumption rate | questions/task |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in loop_results:
        for label, s in r["summary"].items():
            lines.append(
                f"| {r['model']} | {r['condition']} | {r['seed']} | {label} | {s['n']} | {s['correct']:.2f} | "
                f"{s['partial_or_better']:.2f} | {s['assumption_rate']:.2f} | {s['questions_per_task']:.2f} |"
            )
    return "\n".join(lines)


def transfer_table(rows: list[dict]) -> str:
    lines = [
        "| dataset | source | target | reading | layer | AUROC [95% CI] | single turn | multi turn | CKA |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ds = r.get("tag") or str(r.get("dataset", "")).removeprefix("items_") or "items"
        for name, key, layer, cka in (
            ("source at its best layer", "src_at_best", r["src_layer"], ""),
            (
                "target at matched depth",
                "dst_at_matched",
                r["dst_matched_layer"],
                f"{r['cka_matched']:.3f}",
            ),
            (
                "target at its own best",
                "dst_at_own_best",
                r["dst_best_layer"],
                f"{r['cka_best']:.3f}",
            ),
        ):
            m = r[key]
            bp = m.get("by_pair", {})
            lines.append(
                f"| {ds} | {r['src_key']} | {r['dst_key']} | {name} | {layer} | {_fmt(m)} | "
                f"{_fmt(bp.get('single_turn_ab', {}))} | {_fmt(bp.get('multi_turn_cd', {}))} | {cka} |"
            )
    return "\n".join(lines)


def steer_table(rows: list[dict]) -> str:
    lines = []
    for r in rows:
        ds = r.get("tag") or str(r.get("dataset", "")).removeprefix("items_") or "items"
        alphas = [f"{a:+.1f}" for a in r["alphas"]]
        lines += [
            f"Model {r['model']}, {ds} build, layer {r['layer']}, {r['n_items']} test items "
            f"({r['n_specified']} specified, {r['n_gap']} with a gap). Alpha is in train-set standard "
            "deviations along the mass-mean direction; cells are the fraction of steered replies that "
            "ask a clarifying question.\n",
            "| label | " + " | ".join(alphas) + " |",
            "|---|" + "---|" * len(alphas),
        ]
        for label, curve in r["asking_rate"].items():
            lines.append(
                f"| {label} | "
                + " | ".join(f"{curve.get(a, float('nan')):.2f}" for a in alphas)
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def fig_layer_sweep(r: dict, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for name, p in r["probes"].items():
        layers = sorted(int(k) for k in p["per_layer_val_auroc"])
        ax.plot(
            layers,
            [p["per_layer_val_auroc"][str(k)] for k in layers],
            color=PALETTE.get(name, "k"),
            lw=2,
            label=f"{name} (val)",
        )
        if p.get("per_layer_test_auroc"):
            ax.plot(
                layers,
                [p["per_layer_test_auroc"][str(k)] for k in layers],
                color=PALETTE.get(name, "k"),
                lw=1,
                ls="--",
                alpha=0.6,
                label=f"{name} (test, post hoc)",
            )
        ax.axvline(p["best_layer"], color=PALETTE.get(name, "k"), lw=0.8, alpha=0.4)
    for name, b in r["baselines"].items():
        ax.axhline(b["auroc"], color=PALETTE["baseline"], lw=1, ls=":", alpha=0.9)
        ax.text(0.2, b["auroc"] + 0.005, name, fontsize=7, color=PALETTE["baseline"])
    ax.axhline(0.5, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(f"{r['model']}  {dataset_label(r)}  target={r['target']}  pooling={r['pooling']}")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = out / f"layer_sweep_{r['model_key']}_{r['target']}_{r['pooling']}_{dataset_label(r)}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_reliability(r: dict, out: Path) -> Path | None:
    p = r["probes"].get("linear")
    if not p or not p["test"].get("reliability"):
        return None
    rc = p["test"]["reliability"]
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=150)
    ax.plot([0, 1], [0, 1], color="k", lw=0.8, alpha=0.4)
    ax.plot(rc["confidence"], rc["accuracy"], marker="o", color=PALETTE["linear"])
    for c, a, n in zip(rc["confidence"], rc["accuracy"], rc["count"], strict=True):
        ax.annotate(str(n), (c, a), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"reliability, ECE={p['test']['ece']:.3f}", fontsize=9)
    fig.tight_layout()
    path = out / f"reliability_{r['model_key']}_{r['target']}_{r['pooling']}_{dataset_label(r)}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_uq_bars(r: dict, out: Path) -> Path:
    names = sorted(r["summary"], key=lambda m: -(r["summary"][m].get("auroc") or 0))
    vals = [r["summary"][m]["auroc"] for m in names]
    los = [r["summary"][m]["auroc"] - r["summary"][m]["auroc_lo"] for m in names]
    his = [r["summary"][m]["auroc_hi"] - r["summary"][m]["auroc"] for m in names]
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
    colors = [
        PALETTE["accent"] if m in ("sep", "logit_lens_entropy") else PALETTE["linear"]
        for m in names
    ]
    ax.bar(names, vals, yerr=[los, his], color=colors, capsize=3)
    ax.axhline(0.5, color="k", lw=0.5, alpha=0.3)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUROC (flag underspecified)")
    ax.set_title(f"uncertainty baselines, {r['model']} {r['split']}", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    path = out / f"uq_{Path(r['_file']).stem}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def build_report(
    results_dir: Path = Path("results"), figures_dir: Path = Path("docs/figures")
) -> Path:
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    probes = _load_all(results_dir, "probe")
    uqs = _load_all(results_dir, "uq")
    behaviors = _load_all(results_dir, "behavior")
    loops = _load_all(results_dir, "loop")
    transfers = _load_all(results_dir, "transfer")
    steers = _load_all(results_dir, "steer")

    figs: list[Path] = []
    for r in probes:
        figs.append(fig_layer_sweep(r, figures_dir))
        f = fig_reliability(r, figures_dir)
        if f:
            figs.append(f)
    for r in uqs:
        if r["summary"]:
            figs.append(fig_uq_bars(r, figures_dir))

    md = [
        "# Results\n",
        "Generated by `halluscope report`. Intervals are 1000-sample bootstrap 95 percent.\n",
    ]
    if any(r.get("probe_vs_surface") for r in probes):
        md += [
            "## Does the hidden state beat the words?\n",
            "Linear, mass-mean and MLP probes against the strongest bag-of-words baseline "
            "(TF-IDF on the final turn), resampled together on the same test items. "
            "The single-turn pair is decidable from the words by construction, so it is a "
            "sanity check; the multi-turn pair is the question.\n",
            headline_table(probes),
            "\n",
        ]
    if probes:
        md += [
            "## Probes\n",
            probe_table(probes),
            "\n",
            "### Leave-one-topic-out\n",
            loto_table(probes),
            "\n",
        ]
    if behaviors:
        md += [
            "## Model behavior (asked, flagged, silently assumed)\n",
            behavior_table(behaviors),
            "\n",
        ]
    if uqs:
        md += ["## Uncertainty baselines\n", uq_table(uqs), "\n"]
    if loops:
        md += ["## Clarify gate, simulated-user loop\n", loop_table(loops), "\n"]
    if transfers:
        md += ["## Cross-model transfer\n", transfer_table(transfers), "\n"]
    if steers:
        md += ["## Activation steering\n", steer_table(steers), "\n"]
    if figs:
        md += ["## Figures\n"] + [f"![{f.stem}](figures/{f.name})\n" for f in figs]
    out = figures_dir.parent / "results.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return out
