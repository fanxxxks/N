#!/usr/bin/env bash
# Four-searcher full run: one searcher leg = full-budget search (screening
# tier 150 x 256) in engineering mode + shared-engine backtest of the
# selected strategy. Artifacts are snapshotted per searcher so later legs
# never overwrite earlier evidence.
set -euo pipefail
cd /d/minequant/AlphaGPT

SEARCHER="$1"
PY=/d/minequant/.venv/Scripts/python.exe
OUT=data/foursearch
mkdir -p "$OUT"

echo "[$(date +%T)] leg=$SEARCHER train start"
$PY -m ashare_model.train \
  --config "config/ashare_config_${SEARCHER}.yaml" \
  --lifecycle-mode engineering \
  > "$OUT/${SEARCHER}_train.log" 2>&1
echo "[$(date +%T)] leg=$SEARCHER train done (exit=$?)"

cp data/best_ashare_strategy.json "$OUT/${SEARCHER}_strategy.json"

echo "[$(date +%T)] leg=$SEARCHER backtest start"
$PY -m ashare_model.backtest \
  --config "config/ashare_config_${SEARCHER}.yaml" \
  --formula-file "$OUT/${SEARCHER}_strategy.json" \
  --output "$OUT/${SEARCHER}_backtest.json" \
  --lifecycle-mode engineering \
  > "$OUT/${SEARCHER}_backtest.log" 2>&1
echo "[$(date +%T)] leg=$SEARCHER backtest done (exit=$?)"
