# Quarantined results

## loop_qwen_off_s1.CORRUPT.json

Written 2026-09-16 16:46:23, in under a minute, while several duplicate
`queue_loops.sh` pollers were alive at once and all launched the same job.
A genuine loop run on this machine takes 30 to 60 minutes.

The tell is the wall clock, not the row count. It claims 256 rows written in
under a minute, which is not a rate this hardware can produce: the legitimate
full-split runs each took roughly four hours. Read the timestamps before the
summary block.

Note that 256 rows is now the *correct* shape for a run over the full test
split, so row count alone no longer separates a good file from a bad one. When
this note was first written the only comparison available was the 48-task
pilot, which made 256 look anomalous by itself. It no longer is.

It is kept rather than deleted so the artifact can be inspected, but it must
not be read by `report` or by the deck: `loop_rows()` averages rates across
runs and prints `runs[0]["n"]`, so mixing a fabricated file into a real one
would silently present a mean over measurements that were never taken.

Already replaced. `halluscope loop --model qwen --condition off --seed 1` was
rerun on 2026-09-17 and wrote a genuine `results/loop_qwen_off_s1.json`.
