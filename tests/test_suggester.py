"""Unit tests for scanner.suggester."""

from datetime import date, timedelta

import pandas as pd
import pytest

from scanner.suggester import generate_suggestions, screen_options


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_options_df(*rows) -> pd.DataFrame:
    """Return a DataFrame from a list of option-row dicts."""
    defaults = {
        "strike": 95.0,
        "bid": 1.5,
        "ask": 1.7,
        "openInterest": 200,
        "volume": 50,
        "impliedVolatility": 0.3,
        "inTheMoney": False,
        "contractSymbol": "AAPL95P",
        "lastPrice": 1.5,
        "change": 0.0,
        "percentChange": 0.0,
        "lastTradeDate": date.today().isoformat(),
    }
    records = [{**defaults, **r} for r in rows]
    return pd.DataFrame(records)


def _expiry(days: int = 30) -> str:
    return (date.today() + timedelta(days=days)).strftime("%Y-%m-%d")


# ── screen_options ─────────────────────────────────────────────────────────────


def test_screen_options_returns_valid_candidates():
    df = _make_options_df({"strike": 95.0, "bid": 1.5, "openInterest": 200})
    result = screen_options(df, stock_price=100.0, option_type="put", expiry=_expiry(30), ticker="AAPL")
    assert not result.empty
    assert "annualized_return" in result.columns
    assert "score" in result.columns
    assert "otm_pct" in result.columns


def test_screen_options_filters_low_bid():
    df = _make_options_df({"strike": 95.0, "bid": 0.01, "openInterest": 200})
    result = screen_options(df, stock_price=100.0, option_type="put", expiry=_expiry(30), ticker="AAPL")
    assert result.empty


def test_screen_options_filters_low_open_interest():
    df = _make_options_df({"strike": 95.0, "bid": 1.5, "openInterest": 2})
    result = screen_options(df, stock_price=100.0, option_type="put", expiry=_expiry(30), ticker="AAPL")
    assert result.empty


def test_screen_options_filters_itm_put():
    """A put with strike above stock price is ITM and should be rejected."""
    df = _make_options_df({"strike": 110.0, "bid": 1.5, "openInterest": 200})
    result = screen_options(df, stock_price=100.0, option_type="put", expiry=_expiry(30), ticker="AAPL")
    assert result.empty


def test_screen_options_filters_too_far_otm():
    """A put strike 15 % below stock price exceeds max_otm_pct (10 %) – reject."""
    df = _make_options_df({"strike": 85.0, "bid": 0.5, "openInterest": 200})
    result = screen_options(df, stock_price=100.0, option_type="put", expiry=_expiry(30), ticker="AAPL")
    assert result.empty


def test_screen_options_filters_expired():
    """Options that have already expired (DTE < min_dte) should be rejected."""
    df = _make_options_df({"strike": 95.0, "bid": 1.5, "openInterest": 200})
    result = screen_options(df, stock_price=100.0, option_type="put", expiry=_expiry(1), ticker="AAPL")
    # min_dte is 7 – 1 day should be filtered out
    assert result.empty


def test_screen_options_filters_too_far_out():
    """Expiry beyond max_dte (60 days) should be rejected."""
    df = _make_options_df({"strike": 95.0, "bid": 1.5, "openInterest": 200})
    result = screen_options(df, stock_price=100.0, option_type="put", expiry=_expiry(90), ticker="AAPL")
    assert result.empty


def test_screen_options_call():
    """Covered call: OTM call with strike 5 % above stock price should qualify."""
    df = _make_options_df({"strike": 105.0, "bid": 1.5, "openInterest": 200})
    result = screen_options(df, stock_price=100.0, option_type="call", expiry=_expiry(30), ticker="AAPL")
    assert not result.empty


def test_screen_options_empty_df():
    result = screen_options(pd.DataFrame(), stock_price=100.0, option_type="put", expiry=_expiry(30), ticker="AAPL")
    assert result.empty


def test_screen_options_sorted_by_score():
    """Higher-quality options should appear first."""
    df = _make_options_df(
        {"strike": 95.0, "bid": 0.5, "openInterest": 50},   # lower quality
        {"strike": 96.0, "bid": 2.5, "openInterest": 500},  # higher quality
    )
    result = screen_options(df, stock_price=100.0, option_type="put", expiry=_expiry(30), ticker="AAPL")
    assert not result.empty
    scores = result["score"].tolist()
    assert scores == sorted(scores, reverse=True)


# ── generate_suggestions ──────────────────────────────────────────────────────


def test_generate_suggestions_empty_list():
    result = generate_suggestions([])
    assert result.empty


def test_generate_suggestions_empty_dataframes():
    result = generate_suggestions([pd.DataFrame(), pd.DataFrame()])
    assert result.empty


def test_generate_suggestions_sorted_by_score():
    expiry = _expiry(30)
    low_df = _make_options_df({"strike": 95.0, "bid": 0.5, "openInterest": 50})
    high_df = _make_options_df({"strike": 96.0, "bid": 3.0, "openInterest": 1000})

    screened_low = screen_options(low_df, 100.0, "put", expiry, "LOW")
    screened_high = screen_options(high_df, 100.0, "put", expiry, "HIGH")

    suggestions = generate_suggestions([screened_low, screened_high])
    assert not suggestions.empty
    scores = suggestions["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_generate_suggestions_combines_calls_and_puts():
    expiry = _expiry(30)
    put_df = _make_options_df({"strike": 95.0, "bid": 1.5, "openInterest": 200})
    call_df = _make_options_df({"strike": 105.0, "bid": 1.5, "openInterest": 200})

    screened_put = screen_options(put_df, 100.0, "put", expiry, "AAPL")
    screened_call = screen_options(call_df, 100.0, "call", expiry, "AAPL")

    suggestions = generate_suggestions([screened_put, screened_call])
    assert not suggestions.empty
    option_types = set(suggestions["option_type"].unique())
    assert "put" in option_types
    assert "call" in option_types


def test_generate_suggestions_output_columns():
    expiry = _expiry(30)
    df = _make_options_df({"strike": 95.0, "bid": 1.5, "openInterest": 200})
    screened = screen_options(df, 100.0, "put", expiry, "AAPL")
    suggestions = generate_suggestions([screened])

    for col in ("ticker", "option_type", "expiry", "dte", "strike", "score"):
        assert col in suggestions.columns
