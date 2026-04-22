#!/usr/bin/env python3
"""Forward evaluation of historical run logs.

Reads the ``_final.csv`` snapshots written by ``main.py --run-log`` and,
for each trade whose expiry date has already passed, fetches the closing
price on that date from Yahoo Finance and computes the realised P&L.

Outputs a per-trade summary table and — when two or more completed trades
are available — reports the Spearman rank correlation between the
pre-trade composite score and the actual outcome.  A healthy signal
would show ρ ≥ 0.3; anything near 0 suggests the scoring formula needs
calibration against real results.

Usage
-----
  # Evaluate all run logs in the default directory (``runs/``)
  python forward_eval.py

  # Specify a directory and save the per-trade results to CSV
  python forward_eval.py --run-dir runs/ --output eval_results.csv
"""

import logging
import math
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Data helpers ──────────────────────────────────────────────────────────────


def _fetch_close_on_or_after(ticker: str, target_date: str) -> Optional[float]:
    """Return the closing price for *ticker* on *target_date* (YYYY-MM-DD).

    If the market was closed on that date (weekend / holiday), returns the
    first available close price in the following five calendar days.
    Returns ``None`` when no price can be found.
    """
    try:
        start_dt = datetime.strptime(target_date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=6)
        t = yf.Ticker(ticker)
        hist = t.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
        )
        if hist.empty:
            return None
        return float(hist["Close"].iloc[0])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch %s close for %s: %s", ticker, target_date, exc)
        return None


# ── P&L calculation ───────────────────────────────────────────────────────────


def _realised_pnl_per_share(row: pd.Series, expiry_close: float) -> float:
    """Compute realised P&L *per share* for a short spread at expiry.

    **Short put (bull put spread)**:
    - Stock closed **above** strike → full credit collected (bid).
    - Stock closed **below** strike → loss = strike − expiry_close − bid,
      capped at the spread's max loss per share.

    **Short call (bear call spread)**:
    - Stock closed **below** strike → full credit collected (bid).
    - Stock closed **above** strike → loss = expiry_close − strike − bid,
      capped at the spread's max loss per share.
    """
    bid = float(row.get("bid", 0.0) or 0.0)
    strike = float(row.get("strike", 0.0) or 0.0)
    # max_spread_loss is stored in dollars per contract (100 shares)
    max_loss_per_share = float(row.get("max_spread_loss", 0.0) or 0.0) / 100.0
    option_type = str(row.get("option_type", "")).lower()

    if option_type == "put":
        if expiry_close >= strike:
            return bid  # expired worthless — full credit
        raw_loss = strike - expiry_close - bid
        return -min(raw_loss, max_loss_per_share)

    if option_type == "call":
        if expiry_close <= strike:
            return bid  # expired worthless — full credit
        raw_loss = expiry_close - strike - bid
        return -min(raw_loss, max_loss_per_share)

    return 0.0


# ── Run evaluation ────────────────────────────────────────────────────────────


def evaluate_run(final_csv: Path) -> pd.DataFrame:
    """Evaluate a single run's ``_final.csv`` file.

    Returns a DataFrame with columns::

        run_file, ticker, option_type, expiry, strike, bid, score,
        expiry_close, realised_pnl_per_share, outcome

    Only rows whose expiry date is strictly in the past are included.
    Rows for which no historical close price can be retrieved are skipped.
    """
    try:
        df = pd.read_csv(final_csv)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", final_csv, exc)
        return pd.DataFrame()

    if df.empty or "expiry" not in df.columns:
        return pd.DataFrame()

    today = date.today()
    df["_expiry_dt"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    past = df[df["_expiry_dt"] < today].copy()

    if past.empty:
        logger.info("%s: no expired trades yet.", final_csv.name)
        return pd.DataFrame()

    records = []
    for _, row in past.iterrows():
        ticker = str(row.get("ticker", ""))
        expiry = str(row.get("expiry", ""))
        if not ticker or not expiry:
            continue

        expiry_close = _fetch_close_on_or_after(ticker, expiry)
        if expiry_close is None:
            logger.warning(
                "No close price found for %s on %s — skipping.", ticker, expiry
            )
            continue

        pnl = _realised_pnl_per_share(row, expiry_close)
        outcome = "profit" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
        records.append(
            {
                "run_file": final_csv.name,
                "ticker": ticker,
                "option_type": row.get("option_type", ""),
                "expiry": expiry,
                "strike": row.get("strike"),
                "bid": row.get("bid"),
                "score": row.get("score"),
                "expiry_close": expiry_close,
                "realised_pnl_per_share": round(pnl, 4),
                "outcome": outcome,
            }
        )

    return pd.DataFrame(records)


def evaluate_all_runs(runs_dir: Path) -> pd.DataFrame:
    """Aggregate evaluations across all ``_final.csv`` files in *runs_dir*."""
    csv_files = sorted(runs_dir.glob("*_final.csv"))
    if not csv_files:
        logger.warning("No _final.csv files found in %s", runs_dir)
        return pd.DataFrame()

    frames = [evaluate_run(f) for f in csv_files]
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame()

    return pd.concat(non_empty, ignore_index=True)


# ── Statistics ────────────────────────────────────────────────────────────────


def _spearman_correlation(x: pd.Series, y: pd.Series) -> float:
    """Return the Spearman rank correlation between *x* and *y*.

    Computed without external dependencies using rank sums.
    Returns ``nan`` when fewer than two data points are available.
    """
    rx = x.rank()
    ry = y.rank()
    n = len(rx)
    if n < 2:
        return float("nan")
    mean_rx = rx.mean()
    mean_ry = ry.mean()
    num = ((rx - mean_rx) * (ry - mean_ry)).sum()
    denom = math.sqrt(
        ((rx - mean_rx) ** 2).sum() * ((ry - mean_ry) ** 2).sum()
    )
    return float(num / denom) if denom > 0 else float("nan")


# ── Output ────────────────────────────────────────────────────────────────────


def print_summary(results: pd.DataFrame) -> None:
    """Print a human-readable evaluation summary to stdout."""
    if results.empty:
        print("\nNo completed trades to evaluate.")
        return

    n = len(results)
    n_profit = (results["outcome"] == "profit").sum()
    n_loss = (results["outcome"] == "loss").sum()
    avg_pnl = results["realised_pnl_per_share"].mean()

    sep = "=" * 72
    print(f"\n{sep}")
    print("  FORWARD EVALUATION SUMMARY")
    print(sep)
    print(f"  Evaluated trades   : {n}")
    print(f"  Profitable         : {n_profit}  ({n_profit / n * 100:.0f} %)")
    print(f"  Losses             : {n_loss}  ({n_loss / n * 100:.0f} %)")
    print(f"  Average P&L/share  : ${avg_pnl:+.4f}")

    valid = results.dropna(subset=["score", "realised_pnl_per_share"])
    if len(valid) >= 2:
        corr = _spearman_correlation(
            valid["score"], valid["realised_pnl_per_share"]
        )
        print(f"\n  Score vs P&L (Spearman ρ) : {corr:+.3f}")
        if math.isnan(corr):
            print("  → Could not compute correlation.")
        elif abs(corr) < 0.20:
            print(
                "  → Weak/no correlation — score may not yet predict outcomes.\n"
                "     Continue accumulating run logs before drawing conclusions."
            )
        elif corr >= 0.40:
            print("  → Moderate positive correlation — score has predictive value.")
        else:
            print(
                "  → Some positive correlation — accumulate more data to confirm."
            )
    else:
        print("\n  (Need ≥ 2 trades with scores to compute correlation.)")

    print(f"{sep}\n")

    display_cols = [
        c for c in (
            "ticker", "option_type", "expiry", "strike", "bid", "score",
            "expiry_close", "realised_pnl_per_share", "outcome",
        )
        if c in results.columns
    ]
    try:
        from tabulate import tabulate  # noqa: PLC0415

        print(
            tabulate(
                results[display_cols],
                headers="keys",
                tablefmt="simple",
                showindex=False,
                floatfmt=".3f",
            )
        )
    except ImportError:
        print(results[display_cols].to_string(index=False))


# ── CLI entry point ───────────────────────────────────────────────────────────


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate realised P&L for past run-log suggestions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        default=os.environ.get("RUN_LOG_DIR", "runs"),
        help="Directory containing run-log CSVs (written by main.py --run-log).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Optional path to save per-trade results as CSV.",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error("Run-log directory '%s' does not exist.", run_dir)
        return 1

    results = evaluate_all_runs(run_dir)
    print_summary(results)

    if args.output and not results.empty:
        results.to_csv(args.output, index=False)
        logger.info("Results saved to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
