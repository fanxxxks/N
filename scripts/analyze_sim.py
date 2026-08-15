"""Analyze AlphaGPT sim trades: fee drag, gross P&L, and anomalies.

Run from anywhere: python scripts/analyze_sim.py
"""
import json, glob, os, collections
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
trades_dir = str(REPO_ROOT / "data" / "sim_trades")
tot = collections.Counter()
buys = sells = 0.0
ntrades = 0
days = set()
per_year_fees = collections.defaultdict(float)

files = sorted(glob.glob(os.path.join(trades_dir, "*.json")))
for f in files:
    d = os.path.basename(f)[:8]
    try:
        rows = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        print("ERR", f, e)
        continue
    if not rows:
        continue
    days.add(d)
    for t in rows:
        ntrades += 1
        tot["amount"] += t["amount"]
        tot["commission"] += t["commission"]
        tot["stamp_tax"] += t["stamp_tax"]
        tot["transfer_fee"] += t["transfer_fee"]
        tot["slippage"] += t["slippage"]
        per_year_fees[d[:4]] += (t["commission"] + t["stamp_tax"] + t["transfer_fee"] + t["slippage"])
        if t["side"] == "buy":
            buys += t["amount"]
        else:
            sells += t["amount"]

fees = tot["commission"] + tot["stamp_tax"] + tot["transfer_fee"] + tot["slippage"]
print("days with trades:", len(days))
print("total trades:", ntrades, " avg trades/trading-day:", round(ntrades / len(days), 1))
print(f"buy amount:  {buys:>15,.0f}")
print(f"sell amount: {sells:>15,.0f}")
print(f"total amount:{tot['amount']:>15,.0f}")
print(f"fees: commission={tot['commission']:,.0f} stamp={tot['stamp_tax']:,.0f} "
      f"transfer={tot['transfer_fee']:,.0f} slippage={tot['slippage']:,.0f} TOTAL={fees:,.0f}")
print(f"initial 100,000 -> final cash 5,946.37; implied gross pnl (no fees) = "
      f"{100000 + sells - buys:,.0f}; with fees = {100000 + sells - buys - fees:,.0f}")
print(f"cumulative fee drag = {fees / 100000 * 100:.1f}% of initial capital")
print("fees by year:")
for y in sorted(per_year_fees):
    print(f"  {y}: {per_year_fees[y]:>10,.0f}")

# P&L attribution over the whole run using cash accounting
print()
print("cash accounting: initial - buy_amount + sell_amount - fees =",
      round(100000 - buys + sells - fees, 2))
