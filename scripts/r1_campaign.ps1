# R1 campaign driver (p21 §9 runbook sequencing; APPROVED p21, 2026-09-06).
#
# Phase 1 runs the engineering calibration (step 0: the preregistered
# s/eval measurement behind the p21 §5 S-fallback rule) and stops — the
# S=512-vs-256 decision is taken from the calibration evidence BEFORE any
# formal config run.  Phase 2 (-RunConfigs) runs the three runnable
# configs (r1_medium_a, r1_medium_ab, r1_short_a) strictly sequentially,
# each as its own evaluation invocation (fresh run_id, one full-window
# load per config), with marker files so an interrupted driver can be
# relaunched: completed configs are skipped, incomplete ones re-run
# (killed partial runs stay in the append-only ledger as attempts; reruns
# use new run ids — AGENTS §4.3).
#
# The driver never edits configs, never touches production defaults, and
# performs no data writes: every leg is a read-only-input evaluation whose
# outputs land under data/r1/ (worktree-local, git-ignored).
#
# Usage:  powershell -File scripts\r1_campaign.ps1               (phase 1)
#         powershell -File scripts\r1_campaign.ps1 -RunConfigs   (phase 2)

param(
    [switch]$RunConfigs
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = "D:\minequant\.venv\Scripts\python.exe"

function Write-Marker([string]$name, [string]$state, [string]$detail) {
    $line = "{0} {1} {2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $state, $detail
    Add-Content -Path "data\r1\markers.log" -Value $line
    Write-Host $line
}

New-Item -ItemType Directory -Force -Path "data\r1" | Out-Null

# ---- step 0: engineering calibration (budget fallback gate, p21 §5) ----
if (-not (Test-Path "data\r1\calibration.done")) {
    Write-Marker "calibration" "START" "S=512 preview: fold 0, medium, seed 42"
    & $py -m ashare_model.evaluation --config configs/r1_calibration.yaml `
        --tier screening --folds 1 --seeds 42 `
        --output data/r1/calibration/protocol_result.json `
        --ledger data/r1/calibration/ledger.jsonl `
        *> "logs\r1_calibration.log"
    if ($LASTEXITCODE -eq 0) {
        New-Item -ItemType File -Force -Path "data\r1\calibration.done" | Out-Null
        Write-Marker "calibration" "DONE" "read logs/r1_calibration.log + data/r1/calibration/ledger.jsonl; the preregistered p21 §5 rule decides S=512 vs S=256 BEFORE any config run"
    } else {
        Write-Marker "calibration" "FAILED" "exit $LASTEXITCODE — campaign stopped, evidence in logs/r1_calibration.log"
        exit 1
    }
} else {
    Write-Marker "calibration" "SKIP" "marker present"
}

if (-not $RunConfigs) {
    Write-Marker "phase1" "COMPLETE" "calibration only; the p21 §5 s/eval decision gates phase 2"
    exit 0
}

# ---- configs 1/2/3/4 (p21 §4; config 3 unblocked 2026-09-08 via t73
# denominator fix 85cf416; config 5 not_run until t81 clears industry_index) ----
$configs = @(
    @{ id = "r1_medium_a";  cfg = "configs/r1_medium_a.yaml" },
    @{ id = "r1_medium_ab"; cfg = "configs/r1_medium_ab.yaml" },
    @{ id = "r1_slow_ab";   cfg = "configs/r1_slow_ab.yaml" },
    @{ id = "r1_short_a";   cfg = "configs/r1_short_a.yaml" }
)

foreach ($c in $configs) {
    $id = $c.id
    if (Test-Path "data\r1\$id.done") {
        Write-Marker $id "SKIP" "marker present"
        continue
    }
    New-Item -ItemType Directory -Force -Path "data\r1\$id" | Out-Null
    Write-Marker $id "START" "formal run, screening S=512, 5 folds x seeds 42/7/2024"
    & $py -m ashare_model.evaluation --config $c.cfg `
        --tier screening `
        --output "data/r1/$id/protocol_result.json" `
        --ledger "data/r1/ledger.jsonl" `
        *> "logs\r1_$id.log"
    if ($LASTEXITCODE -eq 0) {
        New-Item -ItemType File -Force -Path "data\r1\$id.done" | Out-Null
        Write-Marker $id "DONE" "protocol_result.json written"
    } else {
        # fail-closed per config: keep the evidence, record not_run/failed,
        # continue with the remaining configs (p21 §6: not_run is recorded,
        # never silently skipped).
        Write-Marker $id "FAILED" "exit $LASTEXITCODE — evidence in logs\r1_$id.log; continuing with remaining configs"
    }
}

Write-Marker "campaign" "COMPLETE" "driver finished; summaries/adjudication follow in the measurement-log step"
exit 0
