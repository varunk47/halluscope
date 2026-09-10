#!/usr/bin/env bash
# Everything that still needs GPU or judge time, in dependency order, one
# job at a time on one 8 GB GPU. Stops at the first failure so a broken step
# is not buried under the ones after it. Log: logs/overnight.log
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
MIN=data/augmented/items_minimal.jsonl
step() { echo; echo "[$(date '+%F %T')] $*"; }
run() { "$PY" -m halluscope.cli "$@" || { echo "[$(date '+%F %T')] FAILED: $*"; exit 1; }; }

step "behavior labels on the minimal build"
run behavior --model qwen --items $MIN
step "behavior probes on the minimal build"
run probe --model qwen --items $MIN --target will_assume --tag minimal
run probe --model qwen --items $MIN --target will_ask --tag minimal
step "capture Qwen3.5-2B on the minimal build"
run capture --model qwen2b --items $MIN
step "gap probe on Qwen3.5-2B"
run probe --model qwen2b --items $MIN --target gap --exclude-seeds --tag minimal
step "cross-model transfer qwen -> qwen2b"
run transfer --src qwen --dst qwen2b --items $MIN
step "activation steering on qwen"
run steer --model qwen --items $MIN --limit 32
step "semantic entropy on the train split, for the SEP probe"
run uq --model qwen --items $MIN --split train --limit 96 --only semantic_entropy
step "semantic entropy and SEP on the test split"
run uq --model qwen --items $MIN --split test --limit 96 --only semantic_entropy,sep
step "report"
run report
echo "[$(date '+%F %T')] overnight chain done"
