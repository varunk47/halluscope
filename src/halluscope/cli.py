"""Command line entry point. Subcommands import their modules lazily so the
CLI stays fast and CPU-only commands never import torch."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from halluscope import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False, help="HalluScope")
console = Console()


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"halluscope {__version__}")


@app.command("build-data")
def build_data(
    seeds: Path = typer.Option(Path("src/halluscope/data/seeds"), help="Seed families directory"),
    out: Path = typer.Option(Path("data/augmented/items.jsonl"), help="Output JSONL"),
    paraphrases: int = typer.Option(3, help="Paraphrase families per seed family"),
    limit: int | None = typer.Option(None, help="Only process the first N seed families"),
    mode: str = typer.Option(
        "free",
        help="free | minimal; minimal forces near-identical pairs that differ only in the gap",
    ),
) -> None:
    """Augment hand-written seed families with LLM paraphrases."""
    from halluscope.data.augment import build_dataset

    n = build_dataset(
        seeds_dir=seeds, out_path=out, n_paraphrases=paraphrases, limit=limit, mode=mode
    )
    console.print(f"wrote {n} items to {out}")


@app.command()
def verify(
    items: Path = typer.Option(Path("data/augmented/items.jsonl")),
    report: Path = typer.Option(Path("results/verify_report.json")),
    reapply: bool = typer.Option(
        False, help="Recompute rejections from the saved report, no API calls"
    ),
    workers: int = typer.Option(4, help="Parallel judge calls; drop to 1 on a throttled free tier"),
    resume: bool = typer.Option(
        False, help="Keep verdicts already in the report and judge only what is missing"
    ),
    apply: bool = typer.Option(
        True,
        help="Write rejections into the dataset; off for a second judge whose verdicts are compared, not applied",
    ),
) -> None:
    """LLM verification pass: reject augmented families whose variants do not match their labels."""
    from halluscope.data.verify import reapply as _reapply
    from halluscope.data.verify import run_verify

    if reapply:
        r = _reapply(items, report)
        console.print(f"reapplied: {r}")
        return
    r = run_verify(items, out_report=report, workers=workers, resume=resume, apply=apply)
    console.print(
        f"checked {r['families_checked']} families, rejected {r['families_rejected']}; {r['cost']}"
    )


@app.command()
def capture(
    model: str = typer.Option("qwen", help="Model key from config"),
    items: Path = typer.Option(Path("data/augmented/items.jsonl")),
    approved_only: bool = typer.Option(True),
) -> None:
    """Run forward passes and cache per-layer activations for every item and turn."""
    from halluscope.capture.run import run_capture

    n = run_capture(model_key=model, items_path=items, approved_only=approved_only)
    console.print(f"captured {n} item-turns for {model}")


@app.command()
def probe(
    model: str = typer.Option("qwen"),
    items: Path = typer.Option(Path("data/augmented/items.jsonl")),
    target: str = typer.Option("gap", help="gap | will_assume | will_ask"),
    pooling: str = typer.Option("last"),
    out: Path = typer.Option(Path("results")),
    exclude_seeds: bool = typer.Option(False, help="Use augmented (length-matched) items only"),
    tag: str = typer.Option("", help="Suffix for the results file name"),
) -> None:
    """Layer sweep with grouped splits, nested layer selection, bootstrap intervals."""
    from halluscope.probes.run import run_probe

    path = run_probe(
        model_key=model,
        items_path=items,
        target=target,
        pooling=pooling,
        out_dir=out,
        exclude_seeds=exclude_seeds,
        tag=tag,
    )
    console.print(f"wrote {path}")


@app.command()
def residualize(
    model: str = typer.Option("qwen"),
    items: Path = typer.Option(Path("data/augmented/items_minimal.jsonl")),
    target: str = typer.Option("gap"),
    pooling: str = typer.Option("last"),
    out: Path = typer.Option(Path("results")),
    lexical_components: int = typer.Option(
        50, help="Latent bag-of-words dimensions to remove as well; 0 for handcrafted features only"
    ),
    tag: str = typer.Option("", help="Suffix for the results file name"),
) -> None:
    """Re-fit the probe after regressing length, digit count and wording out of the activations."""
    from halluscope.probes.residualize import run_residual_probe

    path = run_residual_probe(
        model_key=model,
        items_path=items,
        target=target,
        pooling=pooling,
        out_dir=out,
        lexical_components=lexical_components,
        tag=tag,
    )
    console.print(f"wrote {path}")


@app.command()
def uq(
    model: str = typer.Option("qwen"),
    items: Path = typer.Option(Path("data/augmented/items.jsonl")),
    split: str = typer.Option("test"),
    out: Path = typer.Option(Path("results")),
    limit: int | None = typer.Option(None, help="Only the first N items of the split"),
    skip: str = typer.Option(
        "",
        help="Comma-separated methods to leave out, e.g. semantic_entropy when no judge is reachable",
    ),
    only: str = typer.Option(
        "", help="Comma-separated methods to run, adding them to rows already scored"
    ),
) -> None:
    """Score items with every uncertainty baseline."""
    from halluscope.config import get_settings
    from halluscope.uq.runner import run_uq

    left_out = {m.strip() for m in skip.split(",") if m.strip()}
    wanted = {m.strip() for m in only.split(",") if m.strip()}
    methods = [
        m for m in get_settings().uq.methods if m not in left_out and (not wanted or m in wanted)
    ]
    path = run_uq(
        model_key=model, items_path=items, split=split, out_dir=out, limit=limit, methods=methods
    )
    console.print(f"wrote {path}")


@app.command()
def behavior(
    model: str = typer.Option("qwen"),
    items: Path = typer.Option(Path("data/augmented/items.jsonl")),
    out: Path = typer.Option(Path("results")),
    tag: str = typer.Option(
        "", help="Suffix for the results file name; defaults to the dataset build"
    ),
) -> None:
    """Generate the model's own answers and judge whether it asked or silently assumed."""
    from halluscope.gate.behavior import run_behavior

    path = run_behavior(model_key=model, items_path=items, out_dir=out, tag=tag)
    console.print(f"wrote {path}")


@app.command()
def transfer(
    src: str = typer.Option("qwen", help="Model whose gap-probe run fixes the layer"),
    dst: str = typer.Option("qwen2b", help="Model to test the same recipe on; capture it first"),
    items: Path = typer.Option(Path("data/augmented/items_minimal.jsonl")),
    pooling: str = typer.Option("last"),
    out: Path = typer.Option(Path("results")),
    tag: str = typer.Option(
        "", help="Suffix for the results file name; defaults to the dataset build"
    ),
) -> None:
    """Same probe recipe on a second model at the matched depth, with CKA between the two."""
    from halluscope.probes.experiments import run_transfer

    path = run_transfer(src, dst, items, out_dir=out, pooling=pooling, tag=tag)
    console.print(f"wrote {path}")


@app.command()
def steer(
    model: str = typer.Option("qwen"),
    items: Path = typer.Option(Path("data/augmented/items_minimal.jsonl")),
    pooling: str = typer.Option("last"),
    out: Path = typer.Option(Path("results")),
    limit: int = typer.Option(32, help="Test items, half specified and half with a gap"),
    alphas: str = typer.Option("-2,-1,0,1,2", help="Push sizes in train-set standard deviations"),
    tag: str = typer.Option(
        "", help="Suffix for the results file name; defaults to the dataset build"
    ),
    random_controls: int = typer.Option(
        0, help="Repeat the sweep along this many random unit directions at the same push norm"
    ),
    control_seed: int | None = typer.Option(None, help="Seed for the random control directions"),
    resume: bool = typer.Option(
        False, help="Keep generations already in the results file and only run what is missing"
    ),
) -> None:
    """Push the residual stream along the gap direction while generating and count clarifying questions."""
    from halluscope.probes.experiments import run_steer

    path = run_steer(
        model,
        items,
        out_dir=out,
        pooling=pooling,
        tag=tag,
        limit=limit,
        alphas=tuple(float(a) for a in alphas.split(",")),
        random_controls=random_controls,
        control_seed=control_seed,
        resume=resume,
    )
    console.print(f"wrote {path}")


@app.command()
def loop(
    model: str = typer.Option("qwen"),
    # Same build as probe and steer. The loop used to default to items.jsonl,
    # which quietly ran the headline experiment on the leaky build while every
    # other result was on the minimal one.
    items: Path = typer.Option(Path("data/augmented/items_minimal.jsonl")),
    condition: str = typer.Option("gate", help="off | gate | always | prompt"),
    seed: int = typer.Option(0),
    out: Path = typer.Option(Path("results")),
    limit: int | None = typer.Option(None, help="Only the first N items of the test split"),
) -> None:
    """Multi-turn simulated-user evaluation of the clarify gate."""
    from halluscope.gate.loop import run_loop_cli

    path = run_loop_cli(
        model_key=model, items_path=items, condition=condition, seed=seed, out_dir=out, limit=limit
    )
    console.print(f"wrote {path}")


@app.command()
def report(
    results: Path = typer.Option(Path("results")),
    figures: Path = typer.Option(Path("docs/figures")),
) -> None:
    """Turn results JSON into tables and figures."""
    from halluscope.eval.report import build_report

    build_report(results_dir=results, figures_dir=figures)
    console.print(f"figures in {figures}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the FastAPI server for the UI."""
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
