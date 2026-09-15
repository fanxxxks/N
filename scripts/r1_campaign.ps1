# R1 campaign driver entry (p21 §9 runbook semantics preserved; B03/p32
# rewiring 2026-09-15).
#
# The execution authority now lives in `scripts/r1_campaign.py`
# (ashare_model.campaign_executor): every config leg runs as a
# TaskExecutor attempt (independent worker subprocess, ledger-bound
# trial, stage telemetry, row budgets), completed-leg reuse is decided
# by the leg identity (config/snapshot/code/seed/budget, p32 §3.2) and
# the legacy `.done` files are derived display only.  Partial failures
# are campaign failures (exit 1); user abort exits 130; operational
# refusals exit 2.
#
# Phase semantics (p21, unchanged): default runs the engineering
# calibration leg only and stops — the S=512-vs-256 decision is a
# RESEARCH decision taken from the calibration evidence BEFORE any
# formal config run.  -RunConfigs proceeds to the five config legs
# (each as an explicit engineering-mode probe; formal identity wiring is
# R01's preregistered scope).
#
# Usage:  powershell -File scripts\r1_campaign.ps1               (phase 1)
#         powershell -File scripts\r1_campaign.ps1 -RunConfigs   (phase 2)
#         Extra args are forwarded to scripts\r1_campaign.py
#         (--retry-failed, --clear-stale-lock, --state-dir).

param(
    [switch]$RunConfigs,
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$ForwardedArgs
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = "D:\minequant\.venv\Scripts\python.exe"

$args = @("scripts\r1_campaign.py")
if ($RunConfigs) { $args += "--run-configs" }
if ($ForwardedArgs) { $args += $ForwardedArgs }

& $py @args
exit $LASTEXITCODE
