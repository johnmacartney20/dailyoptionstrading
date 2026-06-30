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
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from scanner.config import PORTFOLIO_STATE_FILE

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

# Round-trip commission assumed per share: $0.65 to open + $0.65 to close = $1.30
# per contract ÷ 100 shares = $0.013 per share.  Spreads that expire worthless
# require only an open leg (no close); the full round-trip is used as a
# conservative upper bound.
COMMISSION_PER_SHARE: float = 1.30 / 100.0


def realised_pnl_per_share(row: pd.Series, expiry_close: float) -> float:
    """Compute realised P&L *per share* for a short spread at expiry.

    Uses the **mid-price** ``(bid + ask) / 2`` as the assumed fill price, which
    is more realistic than the bid alone for liquid options.  Falls back to the
    bid when ``ask`` is absent or invalid (e.g. old run-log CSVs that pre-date
    the ``ask`` column).

    A fixed round-trip commission of :data:`COMMISSION_PER_SHARE` is subtracted
    from every outcome to reflect real transaction costs.

    **Short put (bull put spread)**:
    - Stock closed **above** strike → full credit collected (mid) minus commission.
    - Stock closed **below** strike → loss = strike − expiry_close − mid,
      capped at the spread's max loss per share, minus commission.

    **Short call (bear call spread)**:
    - Stock closed **below** strike → full credit collected (mid) minus commission.
    - Stock closed **above** strike → loss = expiry_close − strike − mid,
      capped at the spread's max loss per share, minus commission.
    """
    bid = float(row.get("bid", 0.0) or 0.0)
    # Use mid-price as fill assumption; fall back to bid if ask is unavailable.
    ask_raw = row.get("ask", None)
    ask = float(ask_raw) if ask_raw is not None and float(ask_raw) > 0 else bid
    if ask < bid:
        logger.debug(
            "ask (%.4f) < bid (%.4f) for %s — correcting to bid for mid-price calculation.",
            ask,
            bid,
            row.get("ticker", "unknown"),
        )
        ask = bid
    premium = (bid + ask) / 2.0

    strike = float(row.get("strike", 0.0) or 0.0)
    # max_spread_loss is stored in dollars per contract (100 shares)
    max_loss_per_share = float(row.get("max_spread_loss", 0.0) or 0.0) / 100.0
    option_type = str(row.get("option_type", "")).lower()

    if option_type == "put":
        if expiry_close >= strike:
            return premium - COMMISSION_PER_SHARE  # expired worthless — full credit
        raw_loss = strike - expiry_close - premium
        return -min(raw_loss, max_loss_per_share) - COMMISSION_PER_SHARE

    if option_type == "call":
        if expiry_close <= strike:
            return premium - COMMISSION_PER_SHARE  # expired worthless — full credit
        raw_loss = expiry_close - strike - premium
        return -min(raw_loss, max_loss_per_share) - COMMISSION_PER_SHARE

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

        pnl = realised_pnl_per_share(row, expiry_close)
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


def evaluate_closed_positions(state_file: Path) -> pd.DataFrame:
    """Evaluate closed positions captured in persistent portfolio state.

    Returns columns including entry/exit score, score drift and (when
    derivable) realised P&L.
    """
    if not state_file.exists():
        logger.info("State file %s not found; skipping closed-position evaluation.", state_file)
        return pd.DataFrame()

    try:
        state = json.loads(state_file.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse state file %s: %s", state_file, exc)
        return pd.DataFrame()

    closed = state.get("closed_positions", [])
    if not closed:
        return pd.DataFrame()

    rows = []
    for pos in closed:
        ticker = str(pos.get("ticker", ""))
        sub = str(pos.get("sub_portfolio", "")).lower()
        entry_score = float(pos.get("entry_composite_score", 0.0) or 0.0)
        exit_score = float(pos.get("exit_score", pos.get("last_review_score", 0.0)) or 0.0)
        score_drift = exit_score - entry_score

        entry_date = str(pos.get("entry_date", ""))
        exit_date = str(pos.get("exit_date", ""))
        quantity = int(pos.get("quantity", 0) or 0)
        entry_price = float(pos.get("entry_price", 0.0) or 0.0)

        days_held = 0
        try:
            if entry_date and exit_date:
                days_held = max((datetime.strptime(exit_date, "%Y-%m-%d") - datetime.strptime(entry_date, "%Y-%m-%d")).days, 0)
        except ValueError:
            days_held = 0

        realised_pnl = float("nan")
        exit_price = float("nan")
        if quantity > 0 and entry_price > 0 and sub in {"growth", "stability"} and exit_date:
            maybe_close = _fetch_close_on_or_after(ticker, exit_date)
            if maybe_close is not None:
                exit_price = float(maybe_close)
                realised_pnl = (exit_price - entry_price) * quantity

        rows.append(
            {
                "source": "state_closed",
                "ticker": ticker,
                "account_type": pos.get("account_type", ""),
                "sub_portfolio": pos.get("sub_portfolio", ""),
                "entry_date": entry_date,
                "exit_date": exit_date,
                "days_held": days_held,
                "entry_score": round(entry_score, 4),
                "exit_score": round(exit_score, 4),
                "score_drift": round(score_drift, 4),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "realised_pnl": realised_pnl,
            }
        )

    return pd.DataFrame(rows)


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


def print_closed_positions_summary(results: pd.DataFrame) -> None:
    """Print summary for state-driven closed-position analysis."""
    if results.empty:
        print("\nNo closed positions found in portfolio state.")
        return

    sep = "=" * 72
    print(f"\n{sep}")
    print("  CLOSED POSITION REVIEW (STATE JSON)")
    print(sep)
    print(f"  Closed positions   : {len(results)}")

    valid_drift = results.dropna(subset=["entry_score", "exit_score"])
    if not valid_drift.empty:
        avg_drift = valid_drift["score_drift"].mean()
        print(f"  Avg score drift    : {avg_drift:+.3f}")

    pnl_valid = results.dropna(subset=["realised_pnl"])
    if not pnl_valid.empty:
        avg_pnl = pnl_valid["realised_pnl"].mean()
        print(f"  Avg realised P&L   : ${avg_pnl:+.2f}")
        if len(pnl_valid) >= 2:
            corr = _spearman_correlation(pnl_valid["exit_score"], pnl_valid["realised_pnl"])
            print(f"  Exit score vs P&L  : {corr:+.3f}")

    print(f"{sep}\n")
    cols = [
        c
        for c in (
            "ticker",
            "account_type",
            "sub_portfolio",
            "entry_date",
            "exit_date",
            "days_held",
            "entry_score",
            "exit_score",
            "score_drift",
            "realised_pnl",
        )
        if c in results.columns
    ]
    try:
        from tabulate import tabulate  # noqa: PLC0415

        print(
            tabulate(
                results[cols],
                headers="keys",
                tablefmt="simple",
                showindex=False,
                floatfmt=".3f",
            )
        )
    except ImportError:
        print(results[cols].to_string(index=False))


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
        "--state-file",
        default=PORTFOLIO_STATE_FILE,
        help="Portfolio state JSON file used to evaluate closed-position score drift.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Optional path to save per-trade results as CSV.",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if run_dir.exists():
        results = evaluate_all_runs(run_dir)
    else:
        logger.warning("Run-log directory '%s' does not exist. Skipping run-log evaluation.", run_dir)
        results = pd.DataFrame()

    print_summary(results)

    closed_results = evaluate_closed_positions(Path(args.state_file))
    print_closed_positions_summary(closed_results)

    if args.output and not results.empty:
        out = results.copy()
        if not closed_results.empty:
            aligned = closed_results.rename(columns={"realised_pnl": "realised_pnl_per_share"})
            out = pd.concat([out, aligned], ignore_index=True, sort=False)
        out.to_csv(args.output, index=False)
        logger.info("Results saved to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
