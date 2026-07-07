"""Tests for weekly email options performance section."""

from types import SimpleNamespace

from scanner.emailer import build_weekly_portfolio_email


def test_weekly_email_includes_high_conviction_options_section_when_provided():
    tfsa_stock = SimpleNamespace(selected=[])
    rrsp = SimpleNamespace(selected=[])

    options_summary = {
        "lookback_days": 7,
        "min_entry_score": 8.0,
        "tracked_positions": 1,
        "total_weekly_pnl_change": 120.5,
        "total_unrealized_pnl": 310.0,
        "rows": [
            {
                "ticker": "AAPL",
                "account": "TFSA",
                "option_type": "CALL",
                "expiry": "2026-08-21",
                "qty": 2,
                "entry_score": 9.2,
                "start_mark": 1.8,
                "end_mark": 2.4,
                "weekly_pnl_change": 120.5,
                "unrealized_pnl": 310.0,
                "weekly_return_pct": 15.1,
                "days_captured": 5,
            }
        ],
    }

    html = build_weekly_portfolio_email(
        tfsa_stock=tfsa_stock,
        rrsp=rrsp,
        options_weekly_summary=options_summary,
    )

    assert "High-Conviction Options Performance" in html
    assert "AAPL" in html
    assert "120.5" in html
