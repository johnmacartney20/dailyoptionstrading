"""Unit tests for scanner.analyzer."""

import math
from datetime import date, timedelta

import pytest

from scanner.analyzer import (
    calculate_annualized_return,
    calculate_dte,
    calculate_otm_pct,
    enrich_options,
    score_option,
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


# ── score_option ──────────────────────────────────────────────────────────────


def test_score_option_higher_oi_wins():
    """Same premium but more open interest → higher score."""
    s_low_oi = score_option(
        bid=1.0, strike=100.0, dte=30, open_interest=10, implied_volatility=0.3
    )
    s_high_oi = score_option(
        bid=1.0, strike=100.0, dte=30, open_interest=5000, implied_volatility=0.3
    )
    assert s_high_oi > s_low_oi


def test_score_option_higher_return_wins():
    """Higher premium with same OI → higher score."""
    s_low = score_option(
        bid=0.5, strike=100.0, dte=30, open_interest=100, implied_volatility=0.3
    )
    s_high = score_option(
        bid=3.0, strike=100.0, dte=30, open_interest=100, implied_volatility=0.3
    )
    assert s_high > s_low


def test_score_option_invalid_dte():
    assert score_option(bid=1.0, strike=100.0, dte=0, open_interest=100, implied_volatility=0.3) == 0.0


# ── enrich_options ────────────────────────────────────────────────────────────


def test_enrich_options_adds_columns():
    import pandas as pd

    expiry = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    df = pd.DataFrame(
        [
            {
                "strike": 95.0,
                "bid": 1.5,
                "ask": 1.7,
                "openInterest": 200,
                "volume": 50,
                "impliedVolatility": 0.3,
                "inTheMoney": False,
                "contractSymbol": "AAPL95P",
            }
        ]
    )
    enriched = enrich_options(df, stock_price=100.0, option_type="put", expiry=expiry, ticker="AAPL")

    for col in ("ticker", "option_type", "expiry", "dte", "stock_price", "otm_pct", "annualized_return", "score"):
        assert col in enriched.columns

    assert enriched["ticker"].iloc[0] == "AAPL"
    assert enriched["option_type"].iloc[0] == "put"
    assert enriched["dte"].iloc[0] == 30
    assert enriched["otm_pct"].iloc[0] == pytest.approx(0.05)
    assert enriched["annualized_return"].iloc[0] > 0
    assert enriched["score"].iloc[0] > 0


def test_enrich_options_empty_df():
    import pandas as pd

    expiry = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    result = enrich_options(pd.DataFrame(), 100.0, "put", expiry, "AAPL")
    assert result.empty
