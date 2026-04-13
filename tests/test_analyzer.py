"""Unit tests for scanner.analyzer."""

import math
from datetime import date, timedelta

import pytest

from scanner.analyzer import (
    calculate_annualized_return,
    calculate_dte,
    calculate_otm_pct,
    calculate_risk_adjusted_return,
    enrich_options,
    score_option,
    suggest_spread_structure,
    _spread_width,
)


# ── calculate_dte ──────────────────────────────────────────────────────────────


def test_calculate_dte_future():
    future = date.today() + timedelta(days=30)
    assert calculate_dte(future.strftime("%Y-%m-%d")) == 30


def test_calculate_dte_today():
    today = date.today().strftime("%Y-%m-%d")
    assert calculate_dte(today) == 0


def test_calculate_dte_past_returns_zero():
    past = date.today() - timedelta(days=5)
    assert calculate_dte(past.strftime("%Y-%m-%d")) == 0


# ── calculate_otm_pct ─────────────────────────────────────────────────────────


def test_otm_pct_put_otm():
    # Strike 95, stock 100 → 5 % OTM put
    assert calculate_otm_pct(95, 100, "put") == pytest.approx(0.05)


def test_otm_pct_put_itm():
    # Strike 105, stock 100 → -5 % (ITM put)
    assert calculate_otm_pct(105, 100, "put") == pytest.approx(-0.05)


def test_otm_pct_call_otm():
    # Strike 105, stock 100 → 5 % OTM call
    assert calculate_otm_pct(105, 100, "call") == pytest.approx(0.05)


def test_otm_pct_call_itm():
    # Strike 95, stock 100 → -5 % (ITM call)
    assert calculate_otm_pct(95, 100, "call") == pytest.approx(-0.05)


def test_otm_pct_at_the_money():
    assert calculate_otm_pct(100, 100, "put") == pytest.approx(0.0)
    assert calculate_otm_pct(100, 100, "call") == pytest.approx(0.0)


def test_otm_pct_zero_stock_price():
    result = calculate_otm_pct(100, 0, "put")
    assert math.isnan(result)


# ── calculate_annualized_return ───────────────────────────────────────────────


def test_annualized_return_basic():
    # bid=1, strike=100, dte=30 → (1/100)*(365/30)*100 ≈ 12.17 %
    result = calculate_annualized_return(1.0, 100.0, 30)
    expected = (1.0 / 100.0) * (365.0 / 30) * 100.0
    assert result == pytest.approx(expected)


def test_annualized_return_zero_dte():
    assert calculate_annualized_return(1.0, 100.0, 0) == 0.0


def test_annualized_return_zero_strike():
    assert calculate_annualized_return(1.0, 0.0, 30) == 0.0


def test_annualized_return_zero_bid():
    assert calculate_annualized_return(0.0, 100.0, 30) == 0.0


def test_annualized_return_positive():
    assert calculate_annualized_return(2.0, 50.0, 21) > 0


# ── calculate_risk_adjusted_return ────────────────────────────────────────────


def test_risk_adjusted_return_basic():
    # bid=1, strike=100 → width=5, max_loss=4, risk_adj=1/4=0.25
    result = calculate_risk_adjusted_return(1.0, 100.0)
    assert result == pytest.approx(0.25)


def test_risk_adjusted_return_zero_bid():
    assert calculate_risk_adjusted_return(0.0, 100.0) == 0.0


def test_risk_adjusted_return_bid_exceeds_width():
    # bid=6 > width=5 → max_loss negative → return 0
    assert calculate_risk_adjusted_return(6.0, 100.0) == 0.0


def test_risk_adjusted_return_small_strike():
    # strike=30 → width=2.5, bid=0.5 → max_loss=2.0, risk_adj=0.25
    result = calculate_risk_adjusted_return(0.5, 30.0)
    assert result == pytest.approx(0.25)


def test_risk_adjusted_return_large_strike():
    # strike=250 → width=10, bid=2 → max_loss=8, risk_adj=0.25
    result = calculate_risk_adjusted_return(2.0, 250.0)
    assert result == pytest.approx(0.25)


# ── _spread_width ─────────────────────────────────────────────────────────────


def test_spread_width_small_strike():
    assert _spread_width(30.0) == 2.5


def test_spread_width_medium_strike():
    assert _spread_width(100.0) == 5.0


def test_spread_width_large_strike():
    assert _spread_width(300.0) == 10.0


# ── suggest_spread_structure ──────────────────────────────────────────────────


def test_suggest_spread_structure_put():
    assert suggest_spread_structure(95.0, "put") == "Sell 95P / Buy 90P"


def test_suggest_spread_structure_call():
    assert suggest_spread_structure(105.0, "call") == "Sell 105C / Buy 110C"


def test_suggest_spread_structure_small_strike_put():
    # strike=30, width=2.5 → long=27.5
    assert suggest_spread_structure(30.0, "put") == "Sell 30P / Buy 28P"


# ── score_option ──────────────────────────────────────────────────────────────


def test_score_option_higher_oi_wins():
    """Same premium but more open interest → higher score."""
    s_low_oi = score_option(
        bid=1.0, ask=1.1, strike=100.0, stock_price=105.0,
        open_interest=200, implied_volatility=0.3, otm_pct=0.05,
    )
    s_high_oi = score_option(
        bid=1.0, ask=1.1, strike=100.0, stock_price=105.0,
        open_interest=5000, implied_volatility=0.3, otm_pct=0.05,
    )
    assert s_high_oi > s_low_oi


def test_score_option_higher_return_wins():
    """Higher premium with same OI → higher risk-adjusted score."""
    s_low = score_option(
        bid=0.5, ask=0.6, strike=100.0, stock_price=105.0,
        open_interest=1000, implied_volatility=0.3, otm_pct=0.05,
    )
    s_high = score_option(
        bid=3.0, ask=3.1, strike=100.0, stock_price=105.0,
        open_interest=1000, implied_volatility=0.3, otm_pct=0.05,
    )
    assert s_high > s_low


def test_score_option_zero_bid_returns_zero():
    assert score_option(
        bid=0.0, ask=0.1, strike=100.0, stock_price=105.0,
        open_interest=1000, implied_volatility=0.3, otm_pct=0.05,
    ) == 0.0


def test_score_option_atm_lower_than_otm():
    """A strike right at the stock price (0 % OTM) should score below a 5 % OTM strike."""
    s_atm = score_option(
        bid=1.0, ask=1.1, strike=100.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.3, otm_pct=0.0,
    )
    s_otm = score_option(
        bid=1.0, ask=1.1, strike=95.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.3, otm_pct=0.05,
    )
    assert s_otm > s_atm


def test_score_option_high_iv_tight_strike_penalised():
    """High IV paired with tight strike distance should score below high IV with safer distance."""
    s_risky = score_option(
        bid=1.0, ask=1.1, strike=99.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.55, otm_pct=0.01,
    )
    s_safe = score_option(
        bid=1.0, ask=1.1, strike=95.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.55, otm_pct=0.05,
    )
    assert s_safe > s_risky


# ── enrich_options ────────────────────────────────────────────────────────────


def test_enrich_options_adds_columns():
    import pandas as pd

    expiry = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    df = pd.DataFrame(
        [
            {
                "strike": 95.0,
                "bid": 1.5,
                "ask": 1.6,
                "openInterest": 1000,
                "volume": 50,
                "impliedVolatility": 0.3,
                "inTheMoney": False,
                "contractSymbol": "AAPL95P",
            }
        ]
    )
    enriched = enrich_options(df, stock_price=100.0, option_type="put", expiry=expiry, ticker="AAPL")

    for col in (
        "ticker", "option_type", "expiry", "dte", "stock_price",
        "otm_pct", "annualized_return", "bid_ask_spread_pct",
        "risk_adjusted_return", "max_spread_loss", "spread_structure", "score",
    ):
        assert col in enriched.columns

    assert enriched["ticker"].iloc[0] == "AAPL"
    assert enriched["option_type"].iloc[0] == "put"
    assert enriched["dte"].iloc[0] == 30
    assert enriched["otm_pct"].iloc[0] == pytest.approx(0.05)
    assert enriched["annualized_return"].iloc[0] > 0
    assert enriched["bid_ask_spread_pct"].iloc[0] == pytest.approx(0.1 / 1.5)
    assert enriched["risk_adjusted_return"].iloc[0] > 0
    assert enriched["max_spread_loss"].iloc[0] > 0
    assert enriched["spread_structure"].iloc[0] == "Sell 95P / Buy 90P"
    assert enriched["score"].iloc[0] > 0


def test_enrich_options_empty_df():
    import pandas as pd

    expiry = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    result = enrich_options(pd.DataFrame(), 100.0, "put", expiry, "AAPL")
    assert result.empty

