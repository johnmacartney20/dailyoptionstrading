"""Tests for the daily email layout."""

from types import SimpleNamespace

import pandas as pd

from scanner.emailer import build_html_email


def _make_portfolio_item(**overrides):
    base = {
        "ticker": "AAPL",
        "sector": "Technology",
        "strategy_type": "Sell put spread",
        "short_strike": 95.0,
        "long_strike": 90.0,
        "expiration": "2026-08-21",
        "score": 87.5,
        "max_profit": 150.0,
        "max_loss": 350.0,
        "allocation": 500.0,
        "pct_of_portfolio": 50.0,
        "buy_strike": 105.0,
        "tfsa_score": 91.0,
        "current_price": 345.0,
        "composite_score": 88.0,
        "reasoning": "strong trend",
        "long_term_thesis": "stable thesis",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_daily_email_prioritizes_actions_and_groups_rejections():
    suggestions = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "option_type": "put",
                "expiry": "2026-08-21",
                "score": 92.1,
            }
        ]
    )
    holdings_review = pd.DataFrame(
        [
            {
                "ticker": "RY.TO",
                "account": "RRSP",
                "sub_portfolio": "stability",
                "entry_score": 80.0,
                "current_score": 81.0,
                "score_delta": 1.0,
                "days_held": 12,
                "verdict": "HOLD",
                "reason": "score 81.00 above hold floor 6.00",
            },
            {
                "ticker": "NVDA",
                "account": "TFSA",
                "sub_portfolio": "growth",
                "entry_score": 82.0,
                "current_score": 60.0,
                "score_delta": -22.0,
                "days_held": 8,
                "verdict": "EXIT",
                "reason": "score 60.00 below hard exit 4.50",
            },
            {
                "ticker": "MSFT",
                "account": "TFSA",
                "sub_portfolio": "growth",
                "entry_score": 88.0,
                "current_score": 70.0,
                "score_delta": -18.0,
                "days_held": 6,
                "verdict": "HOLD",
                "reason": "score decay 20.5% exceeds 30.0%",
            },
        ]
    )
    portfolio = SimpleNamespace(selected=[_make_portfolio_item(ticker="SPY")])
    options_stock = SimpleNamespace(selected=[_make_portfolio_item(ticker="SHOP", account="OPTIONS", action="Buy stock", current_price=150.0, composite_score=86.0, allocation=250.0, pct_of_portfolio=25.0)])
    tfsa_allocation = SimpleNamespace(selected=[_make_portfolio_item(ticker="AAPL", strategy_type="Buy long call", buy_strike=105.0, tfsa_score=93.5, allocation=300.0, pct_of_portfolio=30.0)])
    tfsa_stock = SimpleNamespace(selected=[_make_portfolio_item(ticker="MFC.TO", account="TFSA", action="Buy stock", current_price=57.05, composite_score=87.5, allocation=450.0, pct_of_portfolio=45.0)])
    rrsp = SimpleNamespace(selected=[_make_portfolio_item(ticker="RY.TO", account="RRSP", action="Buy stock", current_price=288.41, composite_score=95.0, allocation=500.0, pct_of_portfolio=50.0)])
    fhsa_stock = SimpleNamespace(selected=[_make_portfolio_item(ticker="ATD.TO", account="FHSA", action="Buy stock", current_price=82.0, composite_score=84.0, allocation=400.0, pct_of_portfolio=40.0)])
    rejected_candidates = [
        {"ticker": "NVDA", "score": 90.0, "reason": "duplicate ticker"},
        {"ticker": "NVDA", "score": 89.0, "reason": "duplicate ticker"},
        {"ticker": "NVDA", "score": 88.0, "reason": "no available slots"},
        {"ticker": "AVGO", "score": 87.0, "reason": "duplicate ticker"},
    ]
    rebalance_plan = [
        {
            "account": "TFSA",
            "sub_portfolio": "growth",
            "actions": [
                {
                    "action": "SWAP",
                    "sell_ticker": "NVDA",
                    "buy_ticker": "MFC.TO",
                    "capital_required": 450.0,
                    "reason": "capital-constrained buy linked to funding sell",
                }
            ],
        }
    ]

    html = build_html_email(
        suggestions=suggestions,
        exchange="tsx",
        top=5,
        portfolio=portfolio,
        options_stock=options_stock,
        tfsa_allocation=tfsa_allocation,
        tfsa_stock=tfsa_stock,
        rrsp=rrsp,
        fhsa_stock=fhsa_stock,
        holdings_review=holdings_review,
        portfolio_state_summary={"total_positions": 3, "by_status": {"HOLD": 2, "FLAG": 0, "EXIT": 1}},
        rejected_candidates=rejected_candidates,
        rebalance_plan=rebalance_plan,
    )

    assert "Action Summary" in html
    assert html.index("Action Summary") < html.index("Holdings Review")
    assert "<details><summary><strong>Holdings Review</strong></summary>" in html
    assert "1 actionable review(s) from 3 holdings" in html
    assert "NVDA" in html and "EXIT" in html
    assert "New trades entered: <strong>6</strong>" in html
    assert "Model Holdings Snapshot" not in html
    assert "NVDA" in html and "2" in html
    assert "Portfolio Actions" in html
    assert "Rebalance Actions" in html
    assert "NVDA → MFC.TO" in html
    assert "FHSA 1" in html
    assert "OPTIONS 2" in html
    assert "Top Options Watchlist" in html
    assert "<details><summary><strong>Top Options Watchlist</strong></summary>" in html


def test_daily_email_all_holdings_collapses_to_single_summary_line():
    suggestions = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "option_type": "put",
                "expiry": "2026-08-21",
                "score": 92.1,
            }
        ]
    )
    holdings_review = pd.DataFrame(
        [
            {
                "ticker": "RY.TO",
                "account": "RRSP",
                "sub_portfolio": "stability",
                "entry_score": 80.0,
                "current_score": 81.0,
                "score_delta": 1.0,
                "days_held": 12,
                "verdict": "HOLD",
                "reason": "score 81.00 above hold floor 6.00",
            },
            {
                "ticker": "MFC.TO",
                "account": "TFSA",
                "sub_portfolio": "growth",
                "entry_score": 85.0,
                "current_score": 82.0,
                "score_delta": -3.0,
                "days_held": 6,
                "verdict": "HOLD",
                "reason": "score 82.00 above hold floor 6.00",
            },
        ]
    )

    html = build_html_email(
        suggestions=suggestions,
        exchange="tsx",
        holdings_review=holdings_review,
        portfolio_state_summary={"total_positions": 2, "by_status": {"HOLD": 2, "FLAG": 0, "EXIT": 0}},
    )

    assert "2/2 HOLD, no action needed" in html
    assert "EXIT" not in html
    assert "FLAG" not in html
    assert "No rows exceeded the action threshold." in html


def test_daily_email_reports_degraded_options_feed_when_no_suggestions():
    html = build_html_email(
        suggestions=pd.DataFrame(),
        exchange="all",
        scan_diagnostics={"stale_chain_ratio": 0.75, "stale_chains": 18, "chains_checked": 24},
    )

    assert "Top Options Watchlist" in html
    assert "No qualifying options met filters today, and the feed looked degraded" in html
    assert "18/24 eligible option chains had mostly zero bids" in html


def test_daily_email_rebalance_actions_handles_sell_keep_and_empty_states():
    sell_html = build_html_email(
        suggestions=pd.DataFrame(),
        exchange="all",
        rebalance_plan=[
            {
                "account": "TFSA",
                "sub_portfolio": "growth",
                "actions": [
                    {"action": "SELL", "ticker": "NVDA", "delta": -500.0, "reason": "not in target portfolio"}
                ],
            }
        ],
    )
    keep_only_html = build_html_email(
        suggestions=pd.DataFrame(),
        exchange="all",
        rebalance_plan=[
            {
                "account": "TFSA",
                "sub_portfolio": "growth",
                "actions": [
                    {"action": "KEEP", "ticker": "NVDA", "delta": 0.0, "reason": "within rebalance band"}
                ],
            }
        ],
    )
    empty_html = build_html_email(
        suggestions=pd.DataFrame(),
        exchange="all",
        rebalance_plan=[],
    )

    assert "SELL" in sell_html
    assert "NVDA" in sell_html
    assert "$-500.00" in sell_html
    assert "No sell, trim, or buy changes needed versus current holdings." in keep_only_html
    assert "No rebalance actions generated today." in empty_html
