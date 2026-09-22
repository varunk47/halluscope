#!/usr/bin/env bash
# Rerun all four loop conditions on the minimal build at seed 1.
#
# Strictly serial and in one process. The corrupted file in results/quarantine
# came from several queue_loops.sh pollers launching the same job at once, so
# there is no backgrounding and no polling here.
#
# Order is gate, always, off, prompt: the first two carry the headline
# comparison, so if the machine dies overnight the important half is on disk.
set -u

cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
ITEMS=data/augmented/items_minimal.jsonl
LOG=logs/rerun_minimal.log

mkdir -p logs
echo "=== rerun on $ITEMS started $(date '+%F %T') ===" | tee -a "$LOG"

for cond in gate always off prompt; do
  start=$(date +%s)
  echo "--- $cond start $(date '+%F %T') ---" | tee -a "$LOG"
  "$PY" -X utf8 -m halluscope.cli loop \
      --model qwen --condition "$cond" --seed 1 --items "$ITEMS" >>"$LOG" 2>&1
  rc=$?
  mins=$(( ($(date +%s) - start) / 60 ))
  if [ $rc -ne 0 ]; then
    echo "--- $cond FAILED rc=$rc after ${mins}m ---" | tee -a "$LOG"
    exit $rc
  fi
  echo "--- $cond done in ${mins}m ---" | tee -a "$LOG"
done

echo "=== all four done $(date '+%F %T') ===" | tee -a "$LOG"
