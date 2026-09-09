"""Command line entry point. Subcommands import their modules lazily so the
CLI stays fast and CPU-only commands never import torch."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from halluscope import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False, help="HalluScope v2")
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
) -> None:
    """Augment hand-written seed families with LLM paraphrases."""
    from halluscope.data.augment import build_dataset

    n = build_dataset(seeds_dir=seeds, out_path=out, n_paraphrases=paraphrases, limit=limit)
    console.print(f"wrote {n} items to {out}")


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
) -> None:
    """Layer sweep with grouped splits, nested layer selection, bootstrap intervals."""
    from halluscope.probes.run import run_probe

    path = run_probe(model_key=model, items_path=items, target=target, pooling=pooling, out_dir=out)
    console.print(f"wrote {path}")


@app.command()
def uq(
    model: str = typer.Option("qwen"),
    items: Path = typer.Option(Path("data/augmented/items.jsonl")),
    split: str = typer.Option("test"),
    out: Path = typer.Option(Path("results")),
) -> None:
    """Score items with every uncertainty baseline."""
    from halluscope.uq.runner import run_uq

    path = run_uq(model_key=model, items_path=items, split=split, out_dir=out)
    console.print(f"wrote {path}")


@app.command()
def behavior(
    model: str = typer.Option("qwen"),
    items: Path = typer.Option(Path("data/augmented/items.jsonl")),
    out: Path = typer.Option(Path("results")),
) -> None:
    """Generate the model's own answers and judge whether it asked or silently assumed."""
    from halluscope.gate.behavior import run_behavior

    path = run_behavior(model_key=model, items_path=items, out_dir=out)
    console.print(f"wrote {path}")


@app.command()
def loop(
    model: str = typer.Option("qwen"),
    items: Path = typer.Option(Path("data/augmented/items.jsonl")),
    condition: str = typer.Option("gate", help="off | gate | always | prompt"),
    seed: int = typer.Option(0),
    out: Path = typer.Option(Path("results")),
) -> None:
    """Multi-turn simulated-user evaluation of the clarify gate."""
    from halluscope.gate.loop import run_loop_cli

    path = run_loop_cli(
        model_key=model, items_path=items, condition=condition, seed=seed, out_dir=out
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
