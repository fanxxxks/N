"""Read-only probe: limit-up/down event incidence for the t3 contract.

Measures how often sealed limit-ups (close at the limit band), one-word
limit-ups, intraday breaks and sealed limit-downs occur per session, so the
family-3 (limit-event conditional) contract can commit to event definitions
with known cross-sectional incidence.  Run:

    python docs/factor_inventory_audit_20260831/limit_incidence_probe.py
"""

from __future__ import annotations

import duckdb

RATE_CASE = (
    "CASE WHEN substring(ts_code,1,3) IN ('300','301','688','689') "
    "THEN 0.20 ELSE 0.10 END"
)


def main() -> None:
    con = duckdb.connect("data/ashare.duckdb", read_only=True)
    base = f"""
        SELECT ts_code, trade_date, open, high, low, close, pre_close,
               close/pre_close - 1.0 AS chg, {RATE_CASE} AS rate
        FROM daily_bar
        WHERE pre_close > 0
    """
    stats = con.execute(
        f"""
        WITH b AS ({base})
        SELECT
          COUNT(*) AS cells,
          SUM(CASE WHEN chg >= rate - 0.005 THEN 1 ELSE 0 END) AS sealed_up,
          SUM(CASE WHEN abs(close-high) < 1e-9 AND abs(close-low) < 1e-9
                    AND chg >= rate - 0.005 THEN 1 ELSE 0 END) AS oneword_up,
          SUM(CASE WHEN high >= pre_close * (1.0 + rate) - 1e-9
                    AND chg < rate - 0.005 THEN 1 ELSE 0 END) AS break_up,
          SUM(CASE WHEN chg <= -rate + 0.005 THEN 1 ELSE 0 END) AS sealed_down
        FROM b
        """
    ).fetchone()
    per_day = con.execute(
        f"""
        WITH b AS ({base})
        SELECT
          COUNT(DISTINCT trade_date) AS days,
          AVG(n_stock) AS avg_stocks,
          AVG(n_sealed) AS avg_sealed,
          MIN(n_sealed) AS min_sealed,
          MAX(n_sealed) AS max_sealed
        FROM (
          SELECT trade_date, COUNT(*) AS n_stock,
                 SUM(CASE WHEN chg >= rate - 0.005 THEN 1 ELSE 0 END) AS n_sealed
          FROM b GROUP BY trade_date
        )
        """
    ).fetchone()
    yearly = con.execute(
        f"""
        WITH b AS ({base})
        SELECT substring(trade_date,1,4) AS yr,
               COUNT(*) AS cells,
               SUM(CASE WHEN chg >= rate - 0.005 THEN 1 ELSE 0 END) AS sealed_up,
               SUM(CASE WHEN chg <= -rate + 0.005 THEN 1 ELSE 0 END) AS sealed_down
        FROM b GROUP BY yr ORDER BY yr
        """
    ).fetchall()
    trailing5 = con.execute(
        f"""
        WITH b AS ({base}),
        ev AS (
          SELECT ts_code, trade_date,
                 CASE WHEN chg >= rate - 0.005 THEN 1 ELSE 0 END AS sealed
          FROM b
        ),
        roll AS (
          SELECT trade_date,
                 SUM(CASE WHEN had THEN 1 ELSE 0 END) AS n_recent
          FROM (
            SELECT ts_code, trade_date,
                   MAX(sealed) OVER (
                     PARTITION BY ts_code ORDER BY trade_date
                     ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                   ) = 1 AS had
            FROM ev
          )
          GROUP BY trade_date
        ),
        cnt AS (
          SELECT trade_date, COUNT(*) AS n FROM ({base}) GROUP BY trade_date
        )
        SELECT AVG(1.0 * n_recent / n), MIN(1.0 * n_recent / n), MAX(1.0 * n_recent / n)
        FROM roll JOIN cnt USING (trade_date)
        """
    ).fetchone()
    con.close()
    print("cells/sealed_up/oneword_up/break_up/sealed_down:", stats)
    print("per-day: days, avg_stocks, avg_sealed, min, max:", per_day)
    print("share of stocks with a sealed limit-up in trailing 5d (avg/min/max):", trailing5)
    for row in yearly:
        print("year:", row)


if __name__ == "__main__":
    main()
