# Loop runs on the free build

These four files are the 256-task loop runs from 2026-09-17. They ran on
`data/augmented/items.jsonl`, the build that keeps ten borderline families the
minimal build rejects, and they are kept here rather than in `results/` because
every other result in this project is on `items_minimal.jsonl`.

## How the build was identified

The files predate the `dataset` field, so neither one states which build it
ran on. Two independent checks settle it:

1. `evaluation-0003-p1` and `finetuning-0005-p1` appear in the test split.
   `items_minimal` marks both `rejected`, `items.jsonl` marks both `pending`,
   and `run_loop_cli` filters through `approved()` before splitting. They can
   only be present if the items file was `items.jsonl`.
2. Every file records `"layer": 11`. Layer 11 is the best linear-probe layer on
   the free build. The minimal build's best layer is 16, which is what the
   48-task pilot at seed 0 records.

The cause was a default: `halluscope loop` defaulted to `items.jsonl` while
`probe` and `steer` defaulted to `items_minimal.jsonl`, so a run that passed no
`--items` flag silently crossed builds. The default is now the minimal build and
the output records `dataset`, `split` and `n_items`.

## Why they should not be quoted

On the free build TF-IDF over the final turn alone reaches 0.997 AUROC and the
linear probe reaches 0.99989, so gap detection is close to trivial there and the
gate's selectivity is partly reading surface wording rather than internal state.

In particular the McNemar result that these files support, gate over always on
underspecified tasks at b=13, c=4, exact p=0.049, does not reproduce on the
clean build. Treat it as retired.

## What replaced them

`results/loop_qwen_*_s1.json`, rerun on 2026-09-19 on `items_minimal.jsonl` at
gate layer 16 over the clean test split of 248 tasks.
