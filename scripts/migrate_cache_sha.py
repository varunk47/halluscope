"""Rename legacy activation-cache entries to content-addressed filenames.

Entries written before the content hash existed are named ``{id}__t{turn}``,
which collides across dataset builds that reuse item ids. Recapturing them costs
an hour of GPU time, and it is not necessary: each entry stores the sha of the
rendered chat prompt it was produced from, so a candidate item can be matched
against it exactly. Only entries whose prompt sha checks out are renamed. The
rest are reported and left alone, because a mismatch means the file came from
text we no longer have.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from halluscope.capture.activations import prompt_hash, render_chat
from halluscope.capture.cache import CaptureKey, content_sha, model_slug
from halluscope.config import get_settings
from halluscope.data.io import load_items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--items", default="data/augmented/items.jsonl")
    ap.add_argument("--apply", action="store_true", help="without this, only report")
    args = ap.parse_args()

    cfg = get_settings()
    spec = cfg.model_spec(args.model)
    root = cfg.resolved_cache_dir() / "activations" / model_slug(spec.id)
    legacy = [p for p in sorted(root.glob("*.safetensors")) if p.stem.count("__") == 1]
    if not legacy:
        print(f"no legacy entries under {root}")
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.id)
    items = {i.id: i for i in load_items(args.items)}

    renamed = unmatched = missing = 0
    for p in legacy:
        iid, _, turn = p.stem.rpartition("__t")
        it = items.get(iid)
        if it is None:
            missing += 1
            continue
        turns = it.prefix(int(turn))
        if prompt_hash(render_chat(tok, turns, add_generation_prompt=True)) != _stored(p):
            unmatched += 1
            continue
        sha = content_sha(turns)
        dest = p.with_name(CaptureKey(spec.id, iid, int(turn), sha).filename())
        if args.apply:
            _rewrite(p, dest, sha)
        renamed += 1

    verb = "renamed" if args.apply else "would rename"
    print(
        f"{len(legacy)} legacy entries: {verb} {renamed}, {unmatched} prompt-sha mismatch, "
        f"{missing} item id not in {args.items}"
    )
    if not args.apply:
        print("re-run with --apply to write the changes")


def _stored(p: Path) -> str:
    with safe_open(str(p), framework="np") as f:
        return (f.metadata() or {}).get("prompt_sha", "")


def _rewrite(src: Path, dest: Path, sha: str) -> None:
    """Rewrite with the hash in both the name and the metadata, then drop the old file."""
    with safe_open(str(src), framework="np") as f:
        meta = dict(f.metadata() or {})
    meta["content_sha"] = sha
    save_file(load_file(str(src)), str(dest), metadata=meta)
    if dest.resolve() != src.resolve():
        src.unlink()


if __name__ == "__main__":
    main()
