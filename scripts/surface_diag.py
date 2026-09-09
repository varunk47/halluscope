"""Surface-text separability diagnostic.

Answers one question before any GPU time is spent: how much of the label can a
bag of words recover? Reports TF-IDF within each variant pair and, crucially,
across pairs. Cross-pair transfer near chance means the two pairs are separable
only through unrelated shortcuts, not through one shared notion of a gap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from halluscope.data.io import load_items
from halluscope.data.splits import grouped_split


def text(item, final_only: bool) -> str:
    turns = item.turns[-1:] if final_only else item.turns
    return "\n".join(f"{t.role}: {t.content}" for t in turns)


def auroc(tr, te, final_only: bool) -> float:
    ytr = np.array([int(i.label != "specified") for i in tr])
    yte = np.array([int(i.label != "specified") for i in te])
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    xtr = vec.fit_transform([text(i, final_only) for i in tr])
    xte = vec.transform([text(i, final_only) for i in te])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(xte)[:, 1]))


def measure(path: str, augmented_only: bool = True) -> dict:
    items = [i for i in load_items(path) if i.review_status != "rejected"]
    if augmented_only:
        items = [i for i in items if i.source != "seed"]
    sp = grouped_split(items, seed=0, fractions=(0.7, 0.15, 0.15))
    tr, te = sp["train"], sp["test"]

    def sel(xs, vs):
        return [i for i in xs if i.variant in vs]

    out: dict = {"path": path, "n": len(items), "n_train": len(tr), "n_test": len(te)}
    for name, final_only in (("dialogue", False), ("final_turn", True)):
        block = {
            f"{p}_to_{p}": auroc(sel(tr, vs), sel(te, vs), final_only)
            for p, vs in (("all", "abcd"), ("ab", "ab"), ("cd", "cd"))
        }
        block["ab_to_cd"] = auroc(sel(tr, "ab"), sel(te, "cd"), final_only)
        block["cd_to_ab"] = auroc(sel(tr, "cd"), sel(te, "ab"), final_only)
        out[name] = block
    return out


def show(r: dict) -> None:
    print(f"{r['path']}  n={r['n']}  train={r['n_train']} test={r['n_test']}")
    for name in ("dialogue", "final_turn"):
        print(f"  -- {name} --")
        for k, v in r[name].items():
            print(f"   {k:>10} {v:.3f}")


if __name__ == "__main__":
    rows = []
    for a in sys.argv[1:] or ["data/augmented/items.jsonl"]:
        r = measure(a)
        show(r)
        print()
        rows.append(r)
    Path("results").mkdir(exist_ok=True)
    Path("results/surface_ablation.json").write_text(
        json.dumps({"datasets": rows}, indent=2), encoding="utf-8"
    )
    print("wrote results/surface_ablation.json")
