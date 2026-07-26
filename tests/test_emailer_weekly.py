"""Tests for weekly email options performance section."""

from types import SimpleNamespace

from scanner.emailer import build_weekly_portfolio_email


def test_weekly_email_excludes_options_performance_section():
    tfsa_stock = SimpleNamespace(selected=[])
    rrsp = SimpleNamespace(selected=[])

    html = build_weekly_portfolio_email(
        tfsa_stock=tfsa_stock,
        rrsp=rrsp,
    )

    assert "High-Conviction Options Performance" not in html
