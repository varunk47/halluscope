# Run the loop conditions that are still outstanding, resiliently.
#
# Replaces scripts/rerun_loops_minimal.sh, which had two flaws that cost a
# night: it used `exit $rc` so one condition failing skipped every condition
# after it, and the loop itself kept no checkpoint, so a single malformed judge
# response threw away 159 minutes of finished dialogues.
#
# halluscope.gate.loop now appends every finished dialogue to a checkpoint and
# resumes from it, so re-invoking the same condition continues rather than
# restarting. This script therefore just retries a condition a few times and
# moves on regardless, instead of aborting the queue.

$ErrorActionPreference = 'Continue'
Set-Location 'D:\Projects\halluscope'

$Conditions = @('always', 'off', 'prompt')
$Attempts   = 3
$Items      = 'data\augmented\items_minimal.jsonl'
$Log        = 'logs\remaining_loops.log'

function Note($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    $line | Out-File -Append -Encoding utf8 $Log
}

Note "=== queue start: $($Conditions -join ', ') ==="

foreach ($cond in $Conditions) {
    $out = "results\loop_qwen_${cond}_s1.json"

    if (Test-Path $out) {
        Note "$cond already has output, skipping"
        continue
    }

    $ok = $false
    for ($i = 1; $i -le $Attempts; $i++) {
        Note "$cond attempt $i of $Attempts"
        $start = Get-Date

        & '.venv\Scripts\python.exe' -X utf8 -m halluscope.cli loop `
            --model qwen --condition $cond --seed 1 --items $Items *>> $Log
        $rc = $LASTEXITCODE

        $mins = [math]::Round(((Get-Date) - $start).TotalMinutes)

        if ($rc -eq 0 -and (Test-Path $out)) {
            Note "$cond done in ${mins}m"
            $ok = $true
            break
        }
        Note "$cond attempt $i failed rc=$rc after ${mins}m (checkpoint kept, next attempt resumes)"
        Start-Sleep -Seconds 60
    }

    if (-not $ok) {
        Note "$cond GAVE UP after $Attempts attempts, moving to next condition"
    }
}

Note "=== queue finished ==="
