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
  h2    {{ color: #1a5276; font-size: 15px; margin-top: 24px; border-bottom: 2px solid #1a5276; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
  th    {{ background: #1a5276; color: #fff; padding: 6px 10px; text-align: left; font-size: 12px; }}
  td    {{ padding: 5px 10px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) td {{ background: #f2f2f2; }}
  .meta {{ color: #555; font-size: 12px; }}
  .disc {{ font-size: 11px; color: #888; margin-top: 20px; border-top: 1px solid #ccc; padding-top: 8px; }}
  .summary-box {{ background: #eaf4fb; border: 1px solid #1a5276; border-radius: 4px;
                  padding: 12px 16px; margin-bottom: 20px; }}
  .summary-box h2 {{ color: #1a5276; border-bottom-color: #1a5276; }}
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
  .balance-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  .balance-table th {{ background: #1a5276; color: #fff; padding: 6px 10px; text-align: left; }}
  .balance-table td {{ padding: 6px 10px; border-bottom: 1px solid #ddd; }}
</style>
</head>
<body>
<h1>📊 Monthly Portfolio Review — {month_year}</h1>
<p class="meta">Total portfolio value: <strong>${total_capital:,.0f}</strong>
&nbsp;|&nbsp; TFSA: <strong>${tfsa_capital:,.0f}</strong>
&nbsp;|&nbsp; RRSP: <strong>${rrsp_capital:,.0f}</strong></p>
"""


def _monthly_portfolio_summary_html(
    tfsa_capital: float,
    rrsp_capital: float,
    tfsa_opts_deployed: float,
    tfsa_stock_deployed: float,
    rrsp_deployed: float,
) -> str:
    """Return an HTML overview table showing the high-level capital split."""
    total = tfsa_capital + rrsp_capital
    html = '<div class="summary-box">'
    html += "<h2>💼 Portfolio Balance Overview</h2>"
    html += (
        "<table class='balance-table'>"
        "<thead><tr>"
        "<th>Account</th><th>Strategy</th><th>Allocated</th><th>% of Total</th>"
        "</tr></thead><tbody>"
    )
    rows = [
        ("TFSA", "Long Calls (options)", tfsa_opts_deployed),
        ("TFSA", "Growth Stocks", tfsa_stock_deployed),
        ("RRSP", "Stability Stocks &amp; ETFs", rrsp_deployed),
    ]
    for account, strategy, deployed in rows:
        pct = deployed / total * 100 if total > 0 else 0.0
        html += (
            f"<tr><td>{account}</td><td>{strategy}</td>"
            f"<td>${deployed:,.2f}</td><td>{pct:.1f}%</td></tr>"
        )
    total_deployed = tfsa_opts_deployed + tfsa_stock_deployed + rrsp_deployed
    html += (
        f"<tr><td><strong>Total</strong></td><td></td>"
        f"<td><strong>${total_deployed:,.2f}</strong></td>"
        f"<td><strong>{total_deployed / total * 100:.1f}%</strong></td></tr>"
        if total > 0 else ""
    )
    html += "</tbody></table></div>"
    return html


def build_monthly_portfolio_email(
    tfsa_stock: TfsaStockPortfolio,
    tfsa_opts: TfsaAllocation,
    rrsp: RrspPortfolio,
    tfsa_capital: float = 10_000.0,
    rrsp_capital: float = 10_000.0,
) -> str:
    """Return a complete HTML email for the monthly $20,000 TFSA + RRSP portfolio review.

    The email shows an adequately balanced portfolio:

    * **TFSA** (*tfsa_capital*, default $10,000): growth stocks + long calls.
    * **RRSP** (*rrsp_capital*, default $10,000): stability stocks & ETFs.

    Parameters
    ----------
    tfsa_stock:
        TFSA stock growth allocation (from :func:`~scanner.portfolio_allocator.allocate_tfsa_stock_portfolio`).
    tfsa_opts:
        TFSA long call options allocation (from :func:`~scanner.portfolio_allocator.allocate_tfsa_portfolio`).
    rrsp:
        RRSP stability allocation (from :func:`~scanner.portfolio_allocator.allocate_rrsp_portfolio`).
    tfsa_capital:
        Total TFSA capital (default $10,000).
    rrsp_capital:
        Total RRSP capital (default $10,000).
    """
    from datetime import date as _date

    today = _date.today()
    month_year = today.strftime("%B %Y")
    total_capital = tfsa_capital + rrsp_capital

    html = _HTML_MONTHLY_HEAD.format(
        month_year=month_year,
        total_capital=total_capital,
        tfsa_capital=tfsa_capital,
        rrsp_capital=rrsp_capital,
    )

    html += _monthly_portfolio_summary_html(
        tfsa_capital=tfsa_capital,
        rrsp_capital=rrsp_capital,
        tfsa_opts_deployed=tfsa_opts.total_deployed,
        tfsa_stock_deployed=tfsa_stock.total_deployed,
        rrsp_deployed=rrsp.total_deployed,
    )

    # TFSA long calls section
    html += '<div class="tfsa-box">'
    html += "<h2>🇨🇦 TFSA — Long Calls (~30 DTE, Defined Risk)</h2>"
    html += (
        '<p class="tfsa-meta">'
        "Strategy: <strong>Long Calls</strong> &nbsp;|&nbsp; "
        "Target DTE: <strong>~30 days</strong> &nbsp;|&nbsp; "
        f"Deployed: <strong>${tfsa_opts.total_deployed:,.2f}</strong>"
        "</p>"
    )
    if tfsa_opts.selected:
        cols = ["Ticker", "Sector", "Buy Strike", "Expiry", "Score", "Max Loss", "Allocation", "% TFSA"]
        headers_html = "".join(f"<th>{h}</th>" for h in cols)
        rows_html = ""
        for t in tfsa_opts.selected:
            pct_tfsa = t.allocation / tfsa_capital * 100 if tfsa_capital > 0 else 0.0
            cells = "".join(
                f"<td>{v}</td>"
                for v in [
                    t.ticker, t.sector,
                    f"${t.buy_strike:.2f}", t.expiration,
                    f"{t.tfsa_score:.1f}", f"${t.max_loss:.2f}",
                    f"${t.allocation:,.2f}", f"{pct_tfsa:.1f}%",
                ]
            )
            rows_html += f"<tr>{cells}</tr>"
        html += f"<table><thead><tr>{headers_html}</tr></thead><tbody>{rows_html}</tbody></table>"
    else:
        html += "<p>No qualifying long calls found.</p>"
    html += "</div>"

    # TFSA growth stocks section
    html += _tfsa_stock_to_html(tfsa_stock)

    # RRSP stability section
    html += _rrsp_to_html(rrsp)

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
