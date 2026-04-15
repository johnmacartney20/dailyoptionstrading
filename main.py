#!/usr/bin/env python3
"""Daily Options Trading Scanner – main entry point.

Scans public free data from Yahoo Finance for options trading opportunities
on the TSX (Toronto Stock Exchange) and NASDAQ.

Both the non-registered portfolio allocation (bull put spreads) and the
TFSA-compatible allocation (long calls, ~30 DTE) are always computed in
parallel and included in every run and email summary.

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

  # Also send the monthly $20,000 TFSA + RRSP portfolio review email
  python main.py --email you@example.com --monthly-email you@example.com
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import List, Optional

import pandas as pd

from scanner.config import NASDAQ_TICKERS, RRSP_TICKERS, SCREENING_PARAMS, TSX_TICKERS
from scanner.data_fetcher import (
    get_expiration_dates,
    get_market_return,
    get_options_chain,
    get_price_history,
    get_stock_price,
)
from scanner.emailer import build_html_email, build_monthly_portfolio_email, send_email
from scanner.portfolio_allocator import (
    PortfolioAllocation,
    RrspPortfolio,
    TfsaAllocation,
    TfsaStockPortfolio,
    allocate_portfolio,
    allocate_rrsp_portfolio,
    allocate_tfsa_portfolio,
    allocate_tfsa_stock_portfolio,
)
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


def _print_tfsa_stock_allocation(tfsa_stock: TfsaStockPortfolio) -> None:
    """Pretty-print the TFSA stock portfolio summary to stdout."""
    sep = "=" * 88
    dash = "-" * 88

    print(f"\n{sep}")
    print("  TFSA ALLOCATION  —  Stock Portfolio (Growth Focus)")
    print(sep)
    print("  Strategy    : Direct stock ownership  (growth-oriented, TFSA-compatible)")
    print("  Scoring     : Trend(30%) + RS(20%) + Vol(15%) + Liquidity(15%) + Drawdown(20%)")
    print(f"  Positions   : {tfsa_stock.num_positions} high-conviction stock(s)")
    print(f"  Deployed    : ${tfsa_stock.total_deployed:,.2f}")
    print()

    if tfsa_stock.selected:
        headers = [
            "Ticker", "Sector", "Price", "Score", "Allocation", "% Port", "Reasoning",
        ]
        col_w = [7, 14, 8, 7, 12, 7, 30]

        header_row = "  " + "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
        print(header_row)
        print("  " + dash[:len(header_row) - 2])

        for t in tfsa_stock.selected:
            row = [
                t.ticker,
                t.sector,
                f"${t.current_price:.2f}",
                f"{t.composite_score:.1f}",
                f"${t.allocation:,.2f}",
                f"{t.pct_of_portfolio:.1f}%",
                t.reasoning[:30],
            ]
            print("  " + "  ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row)))

        print()
        print(f"  {tfsa_stock.exit_guidance}")

    else:
        print("  No qualifying stocks found for TFSA stock allocation.")

    if tfsa_stock.rejected:
        print(f"\n  Rejected candidates:")
        for r in tfsa_stock.rejected:
            print(f"    • {r.ticker:<10}  score {r.score:5.1f}  —  {r.reason}")

    print(f"{sep}\n")


def _print_rrsp_allocation(rrsp: RrspPortfolio) -> None:
    """Pretty-print the RRSP portfolio summary to stdout."""
    sep = "=" * 88
    dash = "-" * 88

    print(f"\n{sep}")
    print("  RRSP ALLOCATION  —  Stability Focus (Long-Term Holdings)")
    print(sep)
    print("  Strategy    : Large-cap stocks & ETFs  (consistency over growth)")
    print(f"  Positions   : {rrsp.num_positions} stable position(s)")
    print(f"  Deployed    : ${rrsp.total_deployed:,.2f}")
    print()

    if rrsp.selected:
        headers = [
            "Ticker", "Sector", "Price", "Score", "Allocation", "% Port", "Long-Term Thesis",
        ]
        col_w = [7, 14, 8, 7, 12, 7, 30]

        header_row = "  " + "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
        print(header_row)
        print("  " + dash[:len(header_row) - 2])

        for t in rrsp.selected:
            row = [
                t.ticker,
                t.sector,
                f"${t.current_price:.2f}",
                f"{t.composite_score:.1f}",
                f"${t.allocation:,.2f}",
                f"{t.pct_of_portfolio:.1f}%",
                t.long_term_thesis[:30],
            ]
            print("  " + "  ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row)))

    else:
        print("  No qualifying positions found for RRSP allocation.")

    if rrsp.rejected:
        print(f"\n  Rejected candidates:")
        for r in rrsp.rejected:
            print(f"    • {r.ticker:<10}  score {r.score:5.1f}  —  {r.reason}")

    print(f"{sep}\n")


def _fetch_price_histories(tickers: List[str]) -> dict:
    """Fetch price histories for a list of tickers in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    def _fetch_one(ticker: str):
        return ticker, get_price_history(ticker, period="3mo")

    histories = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        for ticker, hist in executor.map(_fetch_one, tickers):
            if hist is not None and not hist.empty:
                histories[ticker] = hist
    return histories


def _print_tfsa_allocation(tfsa: TfsaAllocation) -> None:
    """Pretty-print the TFSA allocation summary to stdout."""
    sep = "=" * 88
    dash = "-" * 88

    print(f"\n{sep}")
    print("  TFSA ALLOCATION  —  Long Calls (Single-Leg Options)")
    print(sep)
    print("  Strategy    : Long Calls  (single-leg, no short premium — TFSA-compatible)")
    print("  Ranking     : Expected upside potential  (not probability of profit)")
    print(f"  Trades      : {tfsa.num_open_trades} high-conviction trade(s)")
    print(f"  Deployed    : ${tfsa.total_deployed:,.2f}")
    print()

    if tfsa.selected:
        headers = [
            "Ticker", "Sector", "Strategy", "Buy", "Expiry",
            "Score", "Max Loss", "Allocation", "% Port",
        ]
        col_w = [7, 14, 12, 7, 12, 7, 9, 12, 7]

        header_row = "  " + "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
        print(header_row)
        print("  " + dash[:len(header_row) - 2])

        for t in tfsa.selected:
            row = [
                t.ticker,
                t.sector,
                t.strategy_type,
                f"${t.buy_strike:.2f}",
                t.expiration,
                f"{t.tfsa_score:.1f}",
                f"${t.max_loss:.2f}",
                f"${t.allocation:,.2f}",
                f"{t.pct_of_portfolio:.1f}%",
            ]
            print("  " + "  ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row)))

    else:
        print("  No qualifying long calls found for TFSA allocation.")

    if tfsa.rejected:
        print(f"\n  Rejected top candidates:")
        for r in tfsa.rejected:
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
    parser.add_argument(
        "--monthly-email",
        metavar="ADDRESS",
        default=None,
        help=(
            "Send a separate monthly $20,000 TFSA + RRSP portfolio review email "
            "to this address (comma-separated for multiple recipients). "
            "Uses the same SMTP credentials as --email. "
            "Intended to be triggered once per month."
        ),
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

    # ── Options portfolio allocation ───────────────────────────────────────────
    # Both options allocations are independent — run them in parallel.
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_portfolio = executor.submit(allocate_portfolio, suggestions)
        fut_tfsa_opts = executor.submit(allocate_tfsa_portfolio, suggestions)
        portfolio = fut_portfolio.result()
        tfsa_opts = fut_tfsa_opts.result()

    _print_portfolio_allocation(portfolio)
    _print_tfsa_allocation(tfsa_opts)

    # ── Stock-based TFSA (growth) & RRSP (stability) allocations ──────────────
    logger.info("Fetching price histories for TFSA/RRSP stock scoring …")
    tfsa_stock_histories = _fetch_price_histories(tickers)
    rrsp_histories = _fetch_price_histories(RRSP_TICKERS)
    market_ret = get_market_return("SPY", period_days=20)
    logger.debug("SPY 20-day return used as market benchmark: %.2f%%", market_ret * 100)

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_tfsa_stock = executor.submit(
            allocate_tfsa_stock_portfolio,
            tfsa_stock_histories,
            1000.0,
            3,
            0.50,
            0.50,
            market_ret,
        )
        fut_rrsp = executor.submit(
            allocate_rrsp_portfolio,
            rrsp_histories,
            1000.0,
            3,
            0.50,
        )
        tfsa_stock = fut_tfsa_stock.result()
        rrsp = fut_rrsp.result()

    _print_tfsa_stock_allocation(tfsa_stock)
    _print_rrsp_allocation(rrsp)

    _print_suggestions(suggestions, top=args.top)

    # ── Optional CSV export ────────────────────────────────────────────────────
    if args.output:
        suggestions.to_csv(args.output, index=False)
        logger.info("Full results saved to %s", args.output)

    # ── Optional email delivery ────────────────────────────────────────────────
    if args.email:
        recipients = [addr.strip() for addr in args.email.split(",") if addr.strip()]
        html = build_html_email(
            suggestions,
            exchange=args.exchange,
            top=args.top,
            portfolio=portfolio,
            tfsa_allocation=tfsa_opts,
            tfsa_stock=tfsa_stock,
            rrsp=rrsp,
        )
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

    # ── Optional monthly portfolio email ──────────────────────────────────────
    if args.monthly_email:
        monthly_recipients = [
            addr.strip() for addr in args.monthly_email.split(",") if addr.strip()
        ]

        # $20,000 total portfolio: $10K TFSA (stocks + long calls) + $10K RRSP.
        # TFSA split: 20% long calls ($2K), 80% growth stocks ($8K).
        _MONTHLY_TFSA_CAPITAL: float = 10_000.0
        _MONTHLY_RRSP_CAPITAL: float = 10_000.0
        _MONTHLY_TFSA_CALLS_PCT: float = 0.20  # 20% of TFSA in long calls
        tfsa_calls_capital = _MONTHLY_TFSA_CAPITAL * _MONTHLY_TFSA_CALLS_PCT
        tfsa_stock_capital = _MONTHLY_TFSA_CAPITAL * (1.0 - _MONTHLY_TFSA_CALLS_PCT)

        logger.info("Computing monthly $20,000 TFSA + RRSP portfolio …")
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_m_tfsa_opts = executor.submit(
                allocate_tfsa_portfolio,
                suggestions,
                tfsa_calls_capital,
                3,
                0.50,
            )
            fut_m_tfsa_stock = executor.submit(
                allocate_tfsa_stock_portfolio,
                tfsa_stock_histories,
                tfsa_stock_capital,
                5,
                0.40,
                0.40,
                market_ret,
            )
            fut_m_rrsp = executor.submit(
                allocate_rrsp_portfolio,
                rrsp_histories,
                _MONTHLY_RRSP_CAPITAL,
                5,
                0.40,
            )
            monthly_tfsa_opts = fut_m_tfsa_opts.result()
            monthly_tfsa_stock = fut_m_tfsa_stock.result()
            monthly_rrsp = fut_m_rrsp.result()

        from datetime import date as _date

        monthly_html = build_monthly_portfolio_email(
            tfsa_stock=monthly_tfsa_stock,
            tfsa_opts=monthly_tfsa_opts,
            rrsp=monthly_rrsp,
            tfsa_capital=_MONTHLY_TFSA_CAPITAL,
            rrsp_capital=_MONTHLY_RRSP_CAPITAL,
        )
        monthly_subject = (
            f"Monthly Portfolio Review — {_date.today().strftime('%B %Y')}"
        )
        try:
            send_email(
                monthly_html,
                recipients=monthly_recipients,
                smtp_host=args.smtp_host,
                smtp_port=args.smtp_port,
                smtp_user=args.smtp_user,
                smtp_password=args.smtp_password,
                subject=monthly_subject,
            )
            logger.info("Monthly portfolio email sent to %s", monthly_recipients)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send monthly portfolio email: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
