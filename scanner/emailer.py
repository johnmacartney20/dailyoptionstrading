"""Email delivery module.

Composes an HTML email summarising the day's options suggestions and
sends it via SMTP.  Supports Gmail (with an App Password) and any
other SMTP relay that accepts STARTTLS on port 587.

Environment variables (all optional; can also be passed directly):
    SMTP_HOST     – defaults to smtp.gmail.com
    SMTP_PORT     – defaults to 587
    SMTP_USER     – sender email address
    SMTP_PASSWORD – SMTP / App Password
    EMAIL_TO      – comma-separated recipient addresses
"""

import logging
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

import pandas as pd

from .portfolio_allocator import PortfolioAllocation, RrspPortfolio, TfsaAllocation, TfsaStockPortfolio

logger = logging.getLogger(__name__)

# ── HTML template helpers ─────────────────────────────────────────────────────

_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  body  {{ font-family: Arial, sans-serif; font-size: 13px; color: #222; }}
  h1    {{ color: #1a5276; font-size: 20px; margin-bottom: 4px; }}
  h2    {{ color: #1a5276; font-size: 15px; margin-top: 24px; border-bottom: 2px solid #1a5276; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
  th    {{ background: #1a5276; color: #fff; padding: 6px 10px; text-align: left; font-size: 12px; }}
  td    {{ padding: 5px 10px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) td {{ background: #f2f2f2; }}
  .meta {{ color: #555; font-size: 12px; }}
  .disc {{ font-size: 11px; color: #888; margin-top: 20px; border-top: 1px solid #ccc; padding-top: 8px; }}
  .port-box {{ background: #eaf4fb; border: 1px solid #1a5276; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; }}
  .port-summary {{ font-size: 12px; color: #333; margin-bottom: 8px; }}
  .port-summary strong {{ color: #1a5276; }}
  .port-rejected {{ font-size: 11px; color: #666; margin-top: 10px; }}
  .port-rejected li {{ margin: 2px 0; }}
  .tfsa-box {{ background: #eafaf1; border: 1px solid #1e8449; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; }}
  .tfsa-box h2 {{ color: #1e8449; border-bottom-color: #1e8449; }}
  .tfsa-summary {{ font-size: 12px; color: #333; margin-bottom: 8px; }}
  .tfsa-summary strong {{ color: #1e8449; }}
  .tfsa-meta {{ font-size: 11px; color: #555; margin-bottom: 6px; }}
  .tfsa-rejected {{ font-size: 11px; color: #666; margin-top: 10px; }}
  .tfsa-rejected li {{ margin: 2px 0; }}
  .tfsa-stock-box {{ background: #fef9e7; border: 1px solid #b7950b; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; }}
  .tfsa-stock-box h2 {{ color: #7d6608; border-bottom-color: #b7950b; }}
  .tfsa-stock-exit {{ font-size: 11px; color: #555; margin-top: 10px; font-style: italic; }}
  .rrsp-box {{ background: #f4ecf7; border: 1px solid #7d3c98; border-radius: 4px; padding: 12px 16px; margin-bottom: 20px; }}
  .rrsp-box h2 {{ color: #6c3483; border-bottom-color: #7d3c98; }}
  .rrsp-meta {{ font-size: 11px; color: #555; margin-bottom: 6px; }}
  .rrsp-summary {{ font-size: 12px; color: #333; margin-bottom: 8px; }}
  .rrsp-summary strong {{ color: #6c3483; }}
</style>
</head>
<body>
<h1>📈 Daily Options Suggestions — {date}</h1>
<p class="meta">Exchange(s): <strong>{exchange}</strong> &nbsp;|&nbsp;
Total qualifying opportunities: <strong>{total}</strong></p>
"""

_HTML_FOOT = """
<p class="disc">
  Data source: Yahoo Finance (yfinance) – free public data.<br>
  <strong>DISCLAIMER:</strong> These suggestions are <em>not</em> financial advice.
  Always do your own research and consult a licensed adviser before trading options.
</p>
</body></html>"""

_STRATEGY_LABELS = {
    "call": "Covered Calls (sell call, own shares)",
    "put": "Cash-Secured Puts (sell put, set aside cash)",
}

# Columns to include in the email table and their display names.
_EMAIL_COLUMNS = {
    "ticker": "Ticker",
    "expiry": "Expiry",
    "dte": "DTE",
    "strike": "Strike",
    "stock_price": "Stock $",
    "bid": "Bid",
    "ask": "Ask",
    "openInterest": "OI",
    "otm_pct": "OTM %",
    "impliedVolatility": "IV %",
    "risk_adjusted_return": "Risk-Adj Ret",
    "spread_structure": "Spread",
    "score": "Score",
}


def _df_to_html_table(df: pd.DataFrame, top: int) -> str:
    """Return an HTML ``<table>`` string from the top *n* rows of *df*."""
    avail = {k: v for k, v in _EMAIL_COLUMNS.items() if k in df.columns}
    display = df[list(avail.keys())].head(top).copy()
    display.columns = list(avail.values())

    # Format numbers
    for col, decimals in [
        ("Ann. Ret %", 1), ("Score", 1), ("Risk-Adj Ret", 3),
        ("Stock $", 2), ("Strike", 2), ("Bid", 2), ("Ask", 2),
    ]:
        if col in display.columns:
            display[col] = display[col].round(decimals)

    if "OTM %" in display.columns:
        display["OTM %"] = (display["OTM %"] * 100).round(1)
    if "IV %" in display.columns:
        display["IV %"] = (display["IV %"] * 100).round(1)

    # Build HTML manually for consistent styling
    headers = "".join(f"<th>{h}</th>" for h in display.columns)
    rows = ""
    for _, row in display.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        rows += f"<tr>{cells}</tr>"

    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"


def _portfolio_to_html(portfolio: PortfolioAllocation) -> str:
    """Return an HTML snippet summarising the portfolio allocation."""
    html = '<div class="port-box">'
    html += "<h2>💼 Portfolio Allocation — $1,000 Capital Deployment</h2>"
    html += (
        f'<p class="port-summary">'
        f"Total capital deployed: <strong>${portfolio.total_deployed:,.2f}</strong>"
        f" &nbsp;|&nbsp; Open trades suggested: <strong>{portfolio.num_open_trades}</strong>"
        f"</p>"
    )

    if portfolio.selected:
        cols = [
            "Ticker", "Sector", "Strategy", "Short Strike", "Long Strike",
            "Expiry", "Score", "Max Profit", "Max Loss", "Allocation", "% Portfolio",
        ]
        headers = "".join(f"<th>{h}</th>" for h in cols)
        rows_html = ""
        for t in portfolio.selected:
            cells = "".join(
                f"<td>{v}</td>"
                for v in [
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
            )
            rows_html += f"<tr>{cells}</tr>"
        html += f"<table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>"
    else:
        html += "<p>No qualifying put spreads found for portfolio allocation.</p>"

    if portfolio.rejected:
        html += '<div class="port-rejected"><strong>Rejected top candidates:</strong><ul>'
        for r in portfolio.rejected:
            html += f"<li><strong>{r.ticker}</strong> (score {r.score:.1f}) — {r.reason}</li>"
        html += "</ul></div>"

    html += "</div>"
    return html


def _tfsa_allocation_to_html(tfsa: TfsaAllocation) -> str:
    """Return an HTML snippet summarising the TFSA allocation."""
    html = '<div class="tfsa-box">'
    html += "<h2>🇨🇦 TFSA Allocation — Long Calls (Single-Leg, Defined Risk)</h2>"
    html += (
        '<p class="tfsa-meta">'
        "Strategy: <strong>Long Calls</strong> &nbsp;|&nbsp; "
        "Ranking: <strong>Expected upside potential</strong> (not probability of profit)"
        "</p>"
    )
    html += (
        f'<p class="tfsa-summary">'
        f"Total deployed: <strong>${tfsa.total_deployed:,.2f}</strong>"
        f" &nbsp;|&nbsp; High-conviction trades: <strong>{tfsa.num_open_trades}</strong>"
        f"</p>"
    )

    if tfsa.selected:
        cols = [
            "Ticker", "Sector", "Strategy", "Buy Strike",
            "Expiry", "Score", "Max Loss", "Allocation", "% Portfolio",
        ]
        headers = "".join(f"<th>{h}</th>" for h in cols)
        rows_html = ""
        for t in tfsa.selected:
            cells = "".join(
                f"<td>{v}</td>"
                for v in [
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
            )
            rows_html += f"<tr>{cells}</tr>"
        html += f"<table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>"
    else:
        html += "<p>No qualifying long calls found for TFSA allocation.</p>"

    if tfsa.rejected:
        html += '<div class="tfsa-rejected"><strong>Rejected top candidates:</strong><ul>'
        for r in tfsa.rejected:
            html += f"<li><strong>{r.ticker}</strong> (score {r.score:.1f}) — {r.reason}</li>"
        html += "</ul></div>"

    html += "</div>"
    return html


def _tfsa_stock_to_html(tfsa_stock: TfsaStockPortfolio) -> str:
    """Return an HTML snippet for the TFSA stock (growth) allocation."""
    html = '<div class="tfsa-stock-box">'
    html += "<h2>📈 TFSA Allocation — Stock Portfolio (Growth Focus)</h2>"
    html += (
        '<p class="tfsa-meta">'
        "Strategy: <strong>Direct stock ownership</strong> &nbsp;|&nbsp; "
        "Scoring: Trend(30%) + RS(20%) + Vol(15%) + Liquidity(15%) + Drawdown(20%)"
        "</p>"
    )
    html += (
        f'<p class="tfsa-summary">'
        f"Total deployed: <strong>${tfsa_stock.total_deployed:,.2f}</strong>"
        f" &nbsp;|&nbsp; High-conviction positions: <strong>{tfsa_stock.num_positions}</strong>"
        f"</p>"
    )

    if tfsa_stock.selected:
        cols = [
            "Ticker", "Sector", "Price", "Score",
            "Allocation", "% Portfolio", "Reasoning",
        ]
        headers = "".join(f"<th>{h}</th>" for h in cols)
        rows_html = ""
        for t in tfsa_stock.selected:
            cells = "".join(
                f"<td>{v}</td>"
                for v in [
                    t.ticker,
                    t.sector,
                    f"${t.current_price:.2f}",
                    f"{t.composite_score:.1f}",
                    f"${t.allocation:,.2f}",
                    f"{t.pct_of_portfolio:.1f}%",
                    t.reasoning,
                ]
            )
            rows_html += f"<tr>{cells}</tr>"
        html += f"<table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>"
        html += (
            f'<p class="tfsa-stock-exit">{tfsa_stock.exit_guidance}</p>'
        )
    else:
        html += "<p>No qualifying stocks found for TFSA stock allocation.</p>"

    if tfsa_stock.rejected:
        html += '<div class="tfsa-rejected"><strong>Rejected candidates:</strong><ul>'
        for r in tfsa_stock.rejected:
            html += f"<li><strong>{r.ticker}</strong> (score {r.score:.1f}) — {r.reason}</li>"
        html += "</ul></div>"

    html += "</div>"
    return html


def _rrsp_to_html(rrsp: RrspPortfolio) -> str:
    """Return an HTML snippet for the RRSP stability allocation."""
    html = '<div class="rrsp-box">'
    html += "<h2>🏦 RRSP Allocation — Stability Focus (Long-Term Holdings)</h2>"
    html += (
        '<p class="rrsp-meta">'
        "Strategy: <strong>Large-cap stocks &amp; ETFs</strong> &nbsp;|&nbsp; "
        "Emphasis: consistency over growth, low volatility"
        "</p>"
    )
    html += (
        f'<p class="rrsp-summary">'
        f"Total deployed: <strong>${rrsp.total_deployed:,.2f}</strong>"
        f" &nbsp;|&nbsp; Positions: <strong>{rrsp.num_positions}</strong>"
        f"</p>"
    )

    if rrsp.selected:
        cols = [
            "Ticker", "Sector", "Price", "Score",
            "Allocation", "% Portfolio", "Long-Term Thesis",
        ]
        headers = "".join(f"<th>{h}</th>" for h in cols)
        rows_html = ""
        for t in rrsp.selected:
            cells = "".join(
                f"<td>{v}</td>"
                for v in [
                    t.ticker,
                    t.sector,
                    f"${t.current_price:.2f}",
                    f"{t.composite_score:.1f}",
                    f"${t.allocation:,.2f}",
                    f"{t.pct_of_portfolio:.1f}%",
                    t.long_term_thesis,
                ]
            )
            rows_html += f"<tr>{cells}</tr>"
        html += f"<table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>"
    else:
        html += "<p>No qualifying positions found for RRSP allocation.</p>"

    if rrsp.rejected:
        html += '<div class="tfsa-rejected"><strong>Rejected candidates:</strong><ul>'
        for r in rrsp.rejected:
            html += f"<li><strong>{r.ticker}</strong> (score {r.score:.1f}) — {r.reason}</li>"
        html += "</ul></div>"

    html += "</div>"
    return html


# ── Monthly portfolio email ───────────────────────────────────────────────────

_HTML_MONTHLY_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  body  {{ font-family: Arial, sans-serif; font-size: 13px; color: #222; }}
  h1    {{ color: #1a5276; font-size: 20px; margin-bottom: 4px; }}
  h2    {{ color: #1a5276; font-size: 16px; margin-top: 28px; border-bottom: 2px solid #1a5276; padding-bottom: 4px; }}
  h3    {{ color: #2e4057; font-size: 14px; margin-top: 16px; margin-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
  th    {{ background: #1a5276; color: #fff; padding: 6px 10px; text-align: left; font-size: 12px; }}
  td    {{ padding: 5px 10px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) td {{ background: #f2f2f2; }}
  .meta {{ color: #555; font-size: 12px; }}
  .disc {{ font-size: 11px; color: #888; margin-top: 20px; border-top: 1px solid #ccc; padding-top: 8px; }}
  .section-box {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;
                  padding: 14px 18px; margin-bottom: 24px; }}
  .section-box h2 {{ color: #1a5276; border-bottom-color: #1a5276; }}
  .perf-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
  .perf-table th {{ background: #2e4057; color: #fff; padding: 6px 10px; text-align: left; }}
  .perf-table td {{ padding: 6px 10px; border-bottom: 1px solid #ddd; }}
  .insight-block {{ margin: 8px 0 4px 0; padding: 8px 12px; border-left: 3px solid #1a5276;
                    background: #eaf4fb; font-size: 12px; color: #333; }}
  .tfsa-box {{ background: #eafaf1; border: 1px solid #1e8449; border-radius: 4px; padding: 14px 18px; margin-bottom: 24px; }}
  .tfsa-box h2 {{ color: #1e8449; border-bottom-color: #1e8449; }}
  .tfsa-box h3 {{ color: #1e8449; }}
  .tfsa-meta {{ font-size: 11px; color: #555; margin-bottom: 6px; }}
  .rrsp-box {{ background: #f4ecf7; border: 1px solid #7d3c98; border-radius: 4px; padding: 14px 18px; margin-bottom: 24px; }}
  .rrsp-box h2 {{ color: #6c3483; border-bottom-color: #7d3c98; }}
  .rrsp-box h3 {{ color: #6c3483; }}
  .rrsp-meta {{ font-size: 11px; color: #555; margin-bottom: 6px; }}
  .strategy-note {{ font-size: 12px; color: #555; font-style: italic; margin: 6px 0; }}
  .alloc-reasoning {{ font-size: 11px; color: #444; }}
</style>
</head>
<body>
<h1>📊 Monthly Portfolio Review — {month_year}</h1>
<p class="meta">
  TFSA (Growth): <strong>${tfsa_capital:,.0f}</strong>
  &nbsp;|&nbsp; RRSP (Stability): <strong>${rrsp_capital:,.0f}</strong>
  &nbsp;|&nbsp; Total: <strong>${total_capital:,.0f}</strong>
</p>
"""


def _estimate_monthly_return_growth(score: float) -> float:
    """Estimate a simulated monthly % return for a TFSA growth stock from its composite score.

    Mapping (linear interpolation):
    * score ≤ 30  → −2.0 %
    * score = 60  →  3.0 %
    * score = 80  →  6.5 %
    * score = 100 → 10.0 %
    """
    if score <= 30:
        return -2.0
    if score <= 60:
        return -2.0 + (score - 30) / 30.0 * 5.0
    if score <= 80:
        return 3.0 + (score - 60) / 20.0 * 3.5
    return 6.5 + (score - 80) / 20.0 * 3.5


def _estimate_monthly_return_stability(score: float) -> float:
    """Estimate a simulated monthly % return for an RRSP stability holding from its composite score.

    Mapping:
    * score ≤ 30  → −1.0 %
    * score = 60  →  1.5 %
    * score = 80  →  3.0 %
    * score = 100 →  5.0 %
    """
    if score <= 30:
        return -1.0
    if score <= 60:
        return -1.0 + (score - 30) / 30.0 * 2.5
    if score <= 80:
        return 1.5 + (score - 60) / 20.0 * 1.5
    return 3.0 + (score - 80) / 20.0 * 2.0


def _monthly_perf_section_html(
    tfsa_stock: "TfsaStockPortfolio",
    rrsp: "RrspPortfolio",
    tfsa_capital: float,
    rrsp_capital: float,
) -> str:
    """Return HTML for Section 1 — Performance Summary."""
    html = '<div class="section-box">'
    html += "<h2>📈 Section 1 — Performance Summary</h2>"
    html += (
        '<p class="strategy-note">'
        "Returns below are <em>simulated estimates</em> derived from each position's "
        "composite momentum and stability score. They reflect expected behaviour given "
        "current trend, volatility, and liquidity conditions — not guaranteed results."
        "</p>"
    )

    for label, portfolio_obj, capital, estimator, box_class in [
        ("TFSA (Growth)", tfsa_stock, tfsa_capital, _estimate_monthly_return_growth, "tfsa"),
        ("RRSP (Stability)", rrsp, rrsp_capital, _estimate_monthly_return_stability, "rrsp"),
    ]:
        positions = portfolio_obj.selected  # type: ignore[union-attr]
        if not positions:
            html += f"<p>No positions selected for {label}.</p>"
            continue

        # Weighted average return (weight = allocation fraction)
        total_alloc = sum(p.allocation for p in positions)
        if total_alloc > 0:
            weighted_return = sum(
                estimator(p.composite_score) * p.allocation / total_alloc
                for p in positions
            )
        else:
            weighted_return = 0.0

        ending_value = capital * (1 + weighted_return / 100)
        dollar_gain = ending_value - capital

        # Sort by estimated return for strongest/weakest
        scored = sorted(
            [(p, estimator(p.composite_score)) for p in positions],
            key=lambda x: x[1],
            reverse=True,
        )
        strongest = scored[:2]
        weakest = scored[-1:]

        html += f"<h3>{label}</h3>"
        html += (
            "<table class='perf-table'>"
            "<thead><tr>"
            "<th>Metric</th><th>Value</th>"
            "</tr></thead><tbody>"
            f"<tr><td>Starting Value</td><td><strong>${capital:,.0f}</strong></td></tr>"
            f"<tr><td>Estimated Ending Value</td><td><strong>${ending_value:,.0f}</strong></td></tr>"
            f"<tr><td>Estimated Monthly Return</td><td><strong>{weighted_return:+.1f}%</strong> (${dollar_gain:+,.0f})</td></tr>"
        )
        strong_str = ", ".join(
            f"{p.ticker} ({ret:+.1f}%, {p.sector})" for p, ret in strongest
        )
        weak_str = ", ".join(
            f"{p.ticker} ({ret:+.1f}%, {p.sector})" for p, ret in weakest
        )
        html += (
            f"<tr><td>Strongest Positions</td><td>{strong_str}</td></tr>"
            f"<tr><td>Weakest Positions</td><td>{weak_str}</td></tr>"
        )
        html += "</tbody></table>"

    html += "</div>"
    return html


def _monthly_insights_section_html(
    tfsa_stock: "TfsaStockPortfolio",
    rrsp: "RrspPortfolio",
) -> str:
    """Return HTML for Section 2 — Insights and Learnings."""
    html = '<div class="section-box">'
    html += "<h2>🔍 Section 2 — Insights and Learnings</h2>"

    all_positions = list(tfsa_stock.selected) + list(rrsp.selected)  # type: ignore[operator]
    if not all_positions:
        html += "<p>No positions to analyse.</p></div>"
        return html

    # Sector performance: average composite score per sector
    sector_scores: dict = {}
    for p in all_positions:
        sector_scores.setdefault(p.sector, []).append(p.composite_score)
    sector_avg = {s: sum(scores) / len(scores) for s, scores in sector_scores.items()}
    top_sectors = sorted(sector_avg.items(), key=lambda x: x[1], reverse=True)
    winning_sectors = top_sectors[:3]
    lagging_sectors = top_sectors[-2:] if len(top_sectors) > 2 else []

    html += "<h3>Winning Patterns</h3>"
    html += '<div class="insight-block"><ul>'
    for sector, avg in winning_sectors:
        html += f"<li><strong>{sector}</strong> — avg composite score {avg:.0f}: strong momentum and trend consistency.</li>"
    html += "</ul></div>"

    if lagging_sectors:
        html += "<h3>Lagging Patterns</h3>"
        html += '<div class="insight-block"><ul>'
        for sector, avg in lagging_sectors:
            html += (
                f"<li><strong>{sector}</strong> — avg composite score {avg:.0f}: "
                "weaker trend or elevated drawdown risk; underweight or monitor closely.</li>"
            )
        html += "</ul></div>"

    # Volatility and correlation commentary
    high_score = [p for p in all_positions if p.composite_score >= 70]
    low_score = [p for p in all_positions if p.composite_score < 50]

    html += "<h3>Macro and Volatility Commentary</h3>"
    html += '<div class="insight-block"><ul>'
    if high_score:
        high_str = ", ".join(p.ticker for p in high_score)
        html += (
            f"<li>High-conviction positions ({high_str}) show strong trend alignment "
            "and controlled volatility — suitable to hold through minor pullbacks.</li>"
        )
    if low_score:
        low_str = ", ".join(p.ticker for p in low_score)
        html += (
            f"<li>Lower-scoring positions ({low_str}) carry elevated volatility or "
            "weaker momentum — consider tighter position sizing or exit on breakdown.</li>"
        )
    if not high_score and not low_score:
        html += "<li>All positions fall in the moderate range — maintain current sizing and review again next month.</li>"
    html += (
        "<li>Sector concentration is capped at &lt;50% per account to limit "
        "correlated drawdown risk across both TFSA and RRSP.</li>"
    )
    html += "</ul></div>"

    html += "</div>"
    return html


def _monthly_rebalance_section_html(
    tfsa_stock: "TfsaStockPortfolio",
    rrsp: "RrspPortfolio",
) -> str:
    """Return HTML for Section 3 — Portfolio Rebalance Strategy."""
    html = '<div class="section-box">'
    html += "<h2>🔄 Section 3 — Portfolio Rebalance Strategy</h2>"

    # TFSA rebalance
    html += '<div class="tfsa-box">'
    html += "<h3>📈 TFSA (Growth Focus)</h3>"
    html += (
        '<p class="tfsa-meta">'
        "Priority: <strong>High-upside, trend-driven positions</strong> &nbsp;|&nbsp; "
        "Scoring: Trend(30%) + Relative Strength(20%) + Vol(15%) + Liquidity(15%) + Drawdown(20%)"
        "</p>"
    )
    if tfsa_stock.selected:
        html += "<ul>"
        for p in tfsa_stock.selected:
            html += (
                f"<li><strong>{p.ticker}</strong> ({p.sector}) — "
                f"Score: {p.composite_score:.0f} | "
                f"<span class='alloc-reasoning'>{p.reasoning}</span></li>"
            )
        html += "</ul>"
        html += (
            f'<p class="strategy-note">'
            f"Maintain {len(tfsa_stock.selected)} concentrated positions. "
            "Take partial profits at +15–25%; allow winners to run if trend holds. "
            "Exit if price breaks the 20-day moving average.</p>"
        )
    else:
        html += "<p>No qualifying growth positions identified this month.</p>"
    html += "</div>"

    # RRSP rebalance
    html += '<div class="rrsp-box">'
    html += "<h3>🏦 RRSP (Stability Focus)</h3>"
    html += (
        '<p class="rrsp-meta">'
        "Priority: <strong>Diversification and low volatility</strong> &nbsp;|&nbsp; "
        "Scoring: Consistency(30%) + Low Volatility(30%) + Liquidity(20%) + Trend Protection(20%)"
        "</p>"
    )
    if rrsp.selected:
        html += "<ul>"
        for p in rrsp.selected:
            html += (
                f"<li><strong>{p.ticker}</strong> ({p.sector}) — "
                f"Score: {p.composite_score:.0f} | "
                f"<span class='alloc-reasoning'>{p.long_term_thesis}</span></li>"
            )
        html += "</ul>"
        html += (
            f'<p class="strategy-note">'
            f"Hold {len(rrsp.selected)} diversified positions across sectors. "
            "Rebalance only if a position drops below its 50-day MA or a better-scoring "
            "alternative emerges. Prioritise income and capital preservation.</p>"
        )
    else:
        html += "<p>No qualifying stability positions identified this month.</p>"
    html += "</div>"

    html += "</div>"
    return html


def _monthly_allocation_section_html(
    tfsa_stock: "TfsaStockPortfolio",
    rrsp: "RrspPortfolio",
    tfsa_capital: float,
    rrsp_capital: float,
) -> str:
    """Return HTML for Section 4 — Suggested Allocation for Next Month (two tables)."""
    html = '<div class="section-box">'
    html += "<h2>💰 Section 4 — Suggested Allocation for Next Month</h2>"

    # TFSA table
    html += '<div class="tfsa-box">'
    html += f"<h3>📈 TFSA (Growth) — ${tfsa_capital:,.0f} Total</h3>"
    html += (
        '<p class="tfsa-meta">'
        "Strategy: <strong>Direct stock ownership</strong> &nbsp;|&nbsp; "
        "Focus: trend momentum, relative strength, controlled drawdown"
        "</p>"
    )
    if tfsa_stock.selected:
        cols = ["Ticker", "Sector", "Allocation ($)", "% of Portfolio", "Reasoning"]
        headers_html = "".join(f"<th>{h}</th>" for h in cols)
        rows_html = ""
        for p in tfsa_stock.selected:
            rows_html += "<tr>" + "".join(
                f"<td>{v}</td>"
                for v in [
                    p.ticker,
                    p.sector,
                    f"${p.allocation:,.2f}",
                    f"{p.pct_of_portfolio:.1f}%",
                    p.reasoning,
                ]
            ) + "</tr>"
        html += f"<table><thead><tr>{headers_html}</tr></thead><tbody>{rows_html}</tbody></table>"
    else:
        html += "<p>No qualifying growth positions for this period.</p>"
    html += "</div>"

    # RRSP table
    html += '<div class="rrsp-box">'
    html += f"<h3>🏦 RRSP (Stability) — ${rrsp_capital:,.0f} Total</h3>"
    html += (
        '<p class="rrsp-meta">'
        "Strategy: <strong>Large-cap stocks &amp; ETFs</strong> &nbsp;|&nbsp; "
        "Focus: consistency, low volatility, long-term thesis"
        "</p>"
    )
    if rrsp.selected:
        cols = ["Ticker", "Sector", "Allocation ($)", "% of Portfolio", "Reasoning"]
        headers_html = "".join(f"<th>{h}</th>" for h in cols)
        rows_html = ""
        for p in rrsp.selected:
            rows_html += "<tr>" + "".join(
                f"<td>{v}</td>"
                for v in [
                    p.ticker,
                    p.sector,
                    f"${p.allocation:,.2f}",
                    f"{p.pct_of_portfolio:.1f}%",
                    p.long_term_thesis,
                ]
            ) + "</tr>"
        html += f"<table><thead><tr>{headers_html}</tr></thead><tbody>{rows_html}</tbody></table>"
    else:
        html += "<p>No qualifying stability positions for this period.</p>"
    html += "</div>"

    html += "</div>"
    return html


def build_monthly_portfolio_email(
    tfsa_stock: TfsaStockPortfolio,
    rrsp: RrspPortfolio,
    tfsa_capital: float = 25_000.0,
    rrsp_capital: float = 25_000.0,
) -> str:
    """Return a complete HTML email for the monthly TFSA + RRSP portfolio review.

    The email is structured into four sections:

    1. **Performance Summary** — simulated monthly performance estimates for
       each account based on composite scoring.
    2. **Insights and Learnings** — sector patterns, winning vs losing positions,
       volatility and correlation commentary.
    3. **Portfolio Rebalance Strategy** — recommended adjustments and conviction
       narrative for TFSA (growth) and RRSP (stability) separately.
    4. **Suggested Allocation for Next Month** — two tables (TFSA and RRSP) with
       ticker, sector, dollar allocation, portfolio %, and reasoning.

    Parameters
    ----------
    tfsa_stock:
        TFSA stock growth allocation ($25,000 growth focus, 3–5 positions).
    rrsp:
        RRSP stability allocation ($25,000 stability focus, 3–6 positions).
    tfsa_capital:
        Total TFSA capital (default $25,000).
    rrsp_capital:
        Total RRSP capital (default $25,000).
    """
    today = date.today()
    month_year = today.strftime("%B %Y")
    total_capital = tfsa_capital + rrsp_capital

    html = _HTML_MONTHLY_HEAD.format(
        month_year=month_year,
        total_capital=total_capital,
        tfsa_capital=tfsa_capital,
        rrsp_capital=rrsp_capital,
    )

    html += _monthly_perf_section_html(tfsa_stock, rrsp, tfsa_capital, rrsp_capital)
    html += _monthly_insights_section_html(tfsa_stock, rrsp)
    html += _monthly_rebalance_section_html(tfsa_stock, rrsp)
    html += _monthly_allocation_section_html(tfsa_stock, rrsp, tfsa_capital, rrsp_capital)

    html += _HTML_FOOT
    return html


def build_html_email(
    suggestions: pd.DataFrame,
    exchange: str,
    top: int = 10,
    portfolio: Optional[PortfolioAllocation] = None,
    tfsa_allocation: Optional[TfsaAllocation] = None,
    tfsa_stock: Optional[TfsaStockPortfolio] = None,
    rrsp: Optional[RrspPortfolio] = None,
) -> str:
    """Return a complete HTML email string for the given *suggestions* DataFrame."""
    today = date.today().strftime("%Y-%m-%d")
    html = _HTML_HEAD.format(date=today, exchange=exchange.upper(), total=len(suggestions))

    if portfolio is not None:
        html += _portfolio_to_html(portfolio)

    if tfsa_allocation is not None:
        html += _tfsa_allocation_to_html(tfsa_allocation)

    if tfsa_stock is not None:
        html += _tfsa_stock_to_html(tfsa_stock)

    if rrsp is not None:
        html += _rrsp_to_html(rrsp)

    for opt_type, label in _STRATEGY_LABELS.items():
        sub = suggestions[suggestions["option_type"] == opt_type]
        if sub.empty:
            continue
        html += f"<h2>{label}</h2>"
        html += f"<p class='meta'>Showing top {min(len(sub), top)}</p>"
        html += _df_to_html_table(sub, top)

    html += _HTML_FOOT
    return html


def send_email(
    html_body: str,
    recipients: List[str],
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None,
    subject: Optional[str] = None,
) -> None:
    """Send *html_body* as an HTML email to *recipients* via STARTTLS SMTP.

    Credentials fall back to the ``SMTP_USER`` / ``SMTP_PASSWORD`` environment
    variables when not supplied directly.

    Parameters
    ----------
    html_body:
        Full HTML content of the email.
    recipients:
        List of recipient email addresses.
    smtp_host:
        SMTP server hostname.
    smtp_port:
        SMTP server port (default 587 / STARTTLS).
    smtp_user:
        Sender login username; falls back to ``SMTP_USER`` env var.
    smtp_password:
        SMTP password; falls back to ``SMTP_PASSWORD`` env var.
    subject:
        Optional custom email subject.  Defaults to
        ``"Daily Options Suggestions — YYYY-MM-DD"``.

    Raises
    ------
    ValueError
        When required SMTP credentials are missing.
    smtplib.SMTPException
        On any SMTP-level error.
    """
    user = smtp_user or os.environ.get("SMTP_USER", "")
    password = smtp_password or os.environ.get("SMTP_PASSWORD", "")

    if not user or not password:
        raise ValueError(
            "SMTP credentials are required. "
            "Set SMTP_USER and SMTP_PASSWORD environment variables, "
            "or pass smtp_user/smtp_password arguments."
        )

    today = date.today().strftime("%Y-%m-%d")
    email_subject = subject if subject is not None else f"Daily Options Suggestions — {today}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = email_subject
    msg["From"] = user
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html_body, "html"))

    logger.info("Sending email to %s via %s:%s", recipients, smtp_host, smtp_port)
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.ehlo()
        server.starttls()
        server.login(user, password)
        server.sendmail(user, recipients, msg.as_string())
    logger.info("Email sent successfully.")
