# Where are the loop re-runs up to?
#
# The run prints nothing once stdout is redirected (rich.track writes to a tty
# only), so the checkpoint file is the progress signal: one JSON record per
# finished dialogue, so line count is tasks done.

Set-Location 'D:\Projects\halluscope'

$Total = 248
$Order = @('gate', 'always', 'off', 'prompt')

foreach ($cond in $Order) {
    $done = "results\loop_qwen_${cond}_s1.json"
    $ckpt = "results\.loop_qwen_${cond}_s1.partial.jsonl"

    if (Test-Path $done) {
        $when = (Get-Item $done).LastWriteTime.ToString('HH:mm')
        Write-Host ("{0,-7} done      {1}/{1}  finished {2}" -f $cond, $Total, $when)
        continue
    }

    if (-not (Test-Path $ckpt)) {
        Write-Host ("{0,-7} queued" -f $cond)
        continue
    }

    $lines   = @(Get-Content $ckpt)
    $n       = $lines.Count
    $errored = @($lines | Where-Object { $_ -match '"error": "\w' }).Count
    $started = (Get-Item $ckpt).CreationTime
    $mins    = ((Get-Date) - $started).TotalMinutes
    $pace    = if ($n -gt 0) { $mins / $n } else { 0 }
    $eta     = if ($pace -gt 0) { (Get-Date).AddMinutes($pace * ($Total - $n)).ToString('HH:mm') } else { '?' }
    $pct     = [math]::Round(100 * $n / $Total)

    Write-Host ("{0,-7} RUNNING   {1}/{2} ({3}%)  {4:N2} min/task  {5} errors  ETA {6}" -f `
        $cond, $n, $Total, $pct, $pace, $errored, $eta)
}

$driver = Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='pwsh.exe'" |
          Where-Object { $_.CommandLine -like '*run_remaining_loops*' }
Write-Host ''
Write-Host $(if ($driver) { "driver alive (PID $($driver.ProcessId))" } else { "DRIVER NOT RUNNING" })
# The log interleaves the driver's own notes with redirected python output that
# lands in a different encoding, so show only the dated driver lines.
Get-Content 'logs\remaining_loops.log' |
    Where-Object { $_ -match '^\d{4}-\d{2}-\d{2} ' } |
    Select-Object -Last 3
