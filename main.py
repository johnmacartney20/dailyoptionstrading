#!/usr/bin/env python3
"""Daily Options Trading Scanner – main entry point.

Scans public free data from Yahoo Finance for options trading opportunities
on the TSX (Toronto Stock Exchange) and NASDAQ.

Usage examples
--------------
  # Scan all exchanges (default)
  python main.py

  # NASDAQ only, show top 10
  python main.py --exchange nasdaq --top 10

  # TSX only, show only put suggestions, save to CSV
  python main.py --exchange tsx --strategy put --output suggestions.csv

  # Scan and email results (credentials via env vars SMTP_USER / SMTP_PASSWORD)
  python main.py --email you@example.com

  # Scan and email with explicit SMTP settings
  python main.py --email you@example.com --smtp-host smtp.gmail.com \\
                 --smtp-user sender@gmail.com --smtp-password "app-password"
"""

import argparse
import logging
import os
import sys
import time
from datetime import date
from typing import List, Optional

import pandas as pd

from scanner.config import NASDAQ_TICKERS, SCREENING_PARAMS, TSX_TICKERS
from scanner.data_fetcher import (
    get_expiration_dates,
    get_options_chain,
    get_stock_price,
)
from scanner.emailer import build_html_email, send_email
from scanner.portfolio_allocator import PortfolioAllocation, allocate_portfolio
from scanner.suggester import generate_suggestions, screen_options

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Pause between full-ticker scans to stay well within Yahoo Finance rate limits.
_TICKER_DELAY = 0.5


def scan_ticker(ticker: str) -> List[pd.DataFrame]:
    """Fetch and screen all qualifying options for *ticker*.

    Returns a (possibly empty) list of screened DataFrames, one per
    expiry / option-type combination that produced at least one candidate.
    """
    results: List[pd.DataFrame] = []

    price = get_stock_price(ticker)
    if price is None:
        logger.debug("Skipping %s – no price data.", ticker)
        return results

    expiries = get_expiration_dates(ticker)
    if not expiries:
        logger.debug("Skipping %s – no options listed.", ticker)
        return results

    for expiry in expiries:
        chain = get_options_chain(ticker, expiry)
        if chain is None:
            continue
        calls_df, puts_df = chain

        for opt_type, opt_df in (("call", calls_df), ("put", puts_df)):
            screened = screen_options(opt_df, price, opt_type, expiry, ticker)
            if not screened.empty:
                results.append(screened)

    return results


def _print_portfolio_allocation(portfolio: PortfolioAllocation) -> None:
    """Pretty-print the portfolio allocation summary to stdout."""
    sep = "=" * 88
    dash = "-" * 88

    print(f"\n{sep}")
    print("  PORTFOLIO ALLOCATION  —  $1,000 Total Capital Deployment")
    print(sep)
    print(f"  Total capital deployed : ${portfolio.total_deployed:,.2f}")
    print(f"  Open trades suggested  : {portfolio.num_open_trades}")
    print()

    if portfolio.selected:
        headers = [
            "Ticker", "Sector", "Strategy", "Short", "Long", "Expiry",
            "Score", "Max Profit", "Max Loss", "Allocation", "% Port",
        ]
        col_w = [7, 14, 16, 7, 7, 12, 7, 11, 9, 12, 7]

        header_row = "  " + "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
        print(header_row)
        print("  " + dash[:len(header_row) - 2])

        for t in portfolio.selected:
            row = [
                t.ticker,
                t.sector,
                t.strategy_type,
                f"${t.short_strike:.2f}",
                f"${t.long_strike:.2f}",
                t.expiration,
                f"{t.score:.1f}",
                f"${t.max_profit:.2f}",
                f"${t.max_loss:.2f}",
                f"${t.allocation:,.2f}",
                f"{t.pct_of_portfolio:.1f}%",
            ]
            print("  " + "  ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row)))

    else:
        print("  No qualifying put spreads found for portfolio allocation.")

    if portfolio.rejected:
        print(f"\n  Rejected top candidates:")
        for r in portfolio.rejected:
            print(f"    • {r.ticker:<10}  score {r.score:5.1f}  —  {r.reason}")

    print(f"{sep}\n")


def _print_suggestions(suggestions: pd.DataFrame, top: int) -> None:
    """Pretty-print *suggestions* to stdout, split by strategy."""
    today = date.today().strftime("%Y-%m-%d")
    sep = "=" * 88

    print(f"\n{sep}")
    print(f"  DAILY OPTIONS TRADING SUGGESTIONS  —  {today}")
    print(sep)
    print(f"  Total qualifying opportunities: {len(suggestions)}")

    strategy_labels = {
        "call": "COVERED CALLS  (sell call, own shares)",
        "put": "CASH-SECURED PUTS  (sell put, set aside cash)",
    }

    # Columns to display and their friendly headers
    display_map = {
        "ticker": "Ticker",
        "option_type": "Type",
        "expiry": "Expiry",
        "dte": "DTE",
        "strike": "Strike",
        "stock_price": "Stock $",
        "bid": "Bid",
        "ask": "Ask",
        "openInterest": "OI",
        "otm_pct": "OTM %",
        "impliedVolatility": "IV %",
        "risk_adjusted_return": "Risk-Adj",
        "spread_structure": "Spread Structure",
        "score": "Score",
    }

    avail_cols = {k: v for k, v in display_map.items() if k in suggestions.columns}

    for opt_type, label in strategy_labels.items():
        sub = suggestions[suggestions["option_type"] == opt_type].head(top)
        if sub.empty:
            continue

        print(f"\n  ── {label} ──")
        print(f"  Showing top {len(sub)}")
        print()

        display = sub[list(avail_cols.keys())].copy()
        display.columns = list(avail_cols.values())

        # Format numeric columns for readability
        for col, fmt in [
            ("Risk-Adj", 3),
            ("Score", 1),
            ("Stock $", 2),
            ("Strike", 2),
            ("Bid", 2),
            ("Ask", 2),
        ]:
            if col in display.columns:
                display[col] = display[col].round(fmt)

        if "OTM %" in display.columns:
            display["OTM %"] = (display["OTM %"] * 100).round(1)
        if "IV %" in display.columns:
            display["IV %"] = (display["IV %"] * 100).round(1)

        try:
            from tabulate import tabulate

            print(
                tabulate(
                    display,
                    headers="keys",
                    tablefmt="simple",
                    showindex=False,
                )
            )
        except ImportError:
            print(display.to_string(index=False))

    print(f"\n{sep}")
    print("  Data source : Yahoo Finance (yfinance) – free public data")
    print("  DISCLAIMER  : These suggestions are NOT financial advice.")
    print("                Always do your own research before trading options.")
    print(f"{sep}\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan TSX & NASDAQ for daily options trading suggestions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exchange",
        choices=["tsx", "nasdaq", "all"],
        default="all",
        help="Exchange(s) to scan.",
    )
    parser.add_argument(
        "--strategy",
        choices=["call", "put", "all"],
        default="all",
        help="Options strategy filter.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top suggestions per strategy to display.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Optional path to save full results as CSV.",
    )
    # ── Email options ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--email",
        metavar="ADDRESS",
        default=None,
        help=(
            "Send the HTML summary to this email address. "
            "Multiple addresses can be separated by commas. "
            "Requires SMTP credentials (see --smtp-* flags or env vars)."
        ),
    )
    parser.add_argument(
        "--smtp-host",
        default=os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        help="SMTP server hostname (default: smtp.gmail.com).",
    )
    parser.add_argument(
        "--smtp-port",
        type=int,
        default=int(os.environ.get("SMTP_PORT", "587")),
        help="SMTP server port (default: 587 / STARTTLS).",
    )
    parser.add_argument(
        "--smtp-user",
        default=os.environ.get("SMTP_USER"),
        help="SMTP login username / sender address (env: SMTP_USER).",
    )
    parser.add_argument(
        "--smtp-password",
        default=os.environ.get("SMTP_PASSWORD"),
        help="SMTP password or App Password (env: SMTP_PASSWORD).",
    )
    args = parser.parse_args(argv)

    # ── Build ticker list ──────────────────────────────────────────────────────
    tickers: List[str] = []
    if args.exchange in ("tsx", "all"):
        tickers.extend(TSX_TICKERS)
    if args.exchange in ("nasdaq", "all"):
        tickers.extend(NASDAQ_TICKERS)

    logger.info(
        "Scanning %d tickers on %s  |  params: %s",
        len(tickers),
        args.exchange.upper(),
        SCREENING_PARAMS,
    )

    # ── Scan ──────────────────────────────────────────────────────────────────
    all_frames: List[pd.DataFrame] = []
    for idx, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] %s", idx, len(tickers), ticker)
        frames = scan_ticker(ticker)
        all_frames.extend(frames)
        time.sleep(_TICKER_DELAY)

    suggestions = generate_suggestions(all_frames)

    if suggestions.empty:
        logger.warning("No qualifying options found with the current parameters.")
        return 0

    # ── Strategy filter ────────────────────────────────────────────────────────
    if args.strategy != "all":
        suggestions = suggestions[suggestions["option_type"] == args.strategy]

    # ── Display ────────────────────────────────────────────────────────────────
    portfolio = allocate_portfolio(suggestions)
    _print_portfolio_allocation(portfolio)
    _print_suggestions(suggestions, top=args.top)

    # ── Optional CSV export ────────────────────────────────────────────────────
    if args.output:
        suggestions.to_csv(args.output, index=False)
        logger.info("Full results saved to %s", args.output)

    # ── Optional email delivery ────────────────────────────────────────────────
    if args.email:
        recipients = [addr.strip() for addr in args.email.split(",") if addr.strip()]
        html = build_html_email(suggestions, exchange=args.exchange, top=args.top, portfolio=portfolio)
        try:
            send_email(
                html,
                recipients=recipients,
                smtp_host=args.smtp_host,
                smtp_port=args.smtp_port,
                smtp_user=args.smtp_user,
                smtp_password=args.smtp_password,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send email: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
