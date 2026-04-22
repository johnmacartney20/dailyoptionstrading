"""Unit tests for scanner.analyzer."""

import math
from datetime import date, timedelta

import pytest

from scanner.analyzer import (
    calculate_annualized_return,
    calculate_dte,
    calculate_otm_pct,
    calculate_pop,
    calculate_risk_adjusted_return,
    enrich_options,
    score_option,
    score_option_tfsa,
    suggest_call_debit_spread,
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
        "risk_adjusted_return", "max_spread_loss", "spread_structure",
        "pop", "score",
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
    # PoP for a 5 % OTM put at 30 % IV / 30 DTE:
    # d2 = (ln(100/95) - 0.5*0.09*(30/365)) / (0.30*sqrt(30/365)) ≈ 0.31
    # N(0.31) ≈ 0.622 — should be comfortably between 0.55 and 0.75
    assert 0.55 < enriched["pop"].iloc[0] < 0.75
    assert enriched["score"].iloc[0] > 0


def test_enrich_options_empty_df():
    import pandas as pd

    expiry = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    result = enrich_options(pd.DataFrame(), 100.0, "put", expiry, "AAPL")
    assert result.empty


# ── score differentiation ─────────────────────────────────────────────────────


def _base_kwargs(**overrides):
    """Return a full set of valid score_option kwargs with optional overrides."""
    defaults = dict(
        bid=1.5,
        ask=1.6,
        strike=100.0,
        stock_price=100.0,
        open_interest=3000,
        implied_volatility=0.40,
        otm_pct=0.07,
    )
    defaults.update(overrides)
    return defaults


def test_score_range_weak_vs_excellent():
    """Weak and excellent trades should have a wide score gap (≥ 40 pts)."""
    weak = score_option(
        bid=0.10, ask=0.30, strike=100.0, stock_price=100.0,
        open_interest=80, implied_volatility=0.20, otm_pct=0.025,
    )
    excellent = score_option(
        bid=3.00, ask=3.05, strike=100.0, stock_price=100.0,
        open_interest=15000, implied_volatility=0.50, otm_pct=0.12,
    )
    assert excellent - weak >= 40.0


def test_score_excellent_below_ceiling():
    """Even an excellent trade should not hit the theoretical 100-pt ceiling."""
    excellent = score_option(
        bid=3.00, ask=3.05, strike=100.0, stock_price=100.0,
        open_interest=15000, implied_volatility=0.50, otm_pct=0.12,
    )
    assert excellent < 100.0


def test_score_low_oi_strongly_penalised():
    """Low OI (< 100) should score significantly less than moderate OI (≥ 2 000)."""
    s_low = score_option(**_base_kwargs(open_interest=50))
    s_mod = score_option(**_base_kwargs(open_interest=2000))
    assert s_mod - s_low >= 10.0


def test_score_wide_spread_penalised():
    """A very wide bid-ask spread (> 25 % of bid) scores below a tight market."""
    s_tight = score_option(**_base_kwargs(ask=1.55))   # ~3 % spread
    s_wide = score_option(**_base_kwargs(ask=2.20))    # ~47 % spread
    assert s_tight > s_wide


def test_score_tight_strike_strongly_penalised():
    """OTM < 2 % should score far below a comfortable 7 % OTM strike."""
    s_tight = score_option(**_base_kwargs(otm_pct=0.015))
    s_safe = score_option(**_base_kwargs(otm_pct=0.07))
    assert s_safe - s_tight >= 10.0


def test_score_low_credit_penalised():
    """Very low credit relative to risk (ratio < 0.15) scores below decent credit."""
    # bid=0.05 → risk_adj ≈ 0.01, well below 0.15 threshold
    s_low = score_option(**_base_kwargs(bid=0.05, ask=0.15))
    # bid=1.50 → risk_adj ≈ 0.43
    s_ok = score_option(**_base_kwargs(bid=1.50, ask=1.60))
    assert s_ok > s_low


def test_score_iv_tight_strike_penalty_extended_to_4pct():
    """High IV with OTM between 3–4 % should still be penalised (new threshold)."""
    s_penalised = score_option(
        bid=1.0, ask=1.1, strike=100.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.55, otm_pct=0.035,
    )
    s_safe = score_option(
        bid=1.0, ask=1.1, strike=100.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.55, otm_pct=0.06,
    )
    assert s_safe > s_penalised


def test_score_iv_env_bonus_requires_35pct():
    """Environmental IV bonus should only apply when IV ≥ 0.35 (raised from 0.30)."""
    s_below = score_option(**_base_kwargs(implied_volatility=0.30, otm_pct=0.07))
    s_above = score_option(**_base_kwargs(implied_volatility=0.35, otm_pct=0.07))
    # The 5-pt env bonus should make IV=0.35 score higher than IV=0.30
    assert s_above > s_below


# ── suggest_call_debit_spread ─────────────────────────────────────────────────


def test_suggest_call_debit_spread_medium_strike():
    # strike=105 → width=5 → sell=110
    assert suggest_call_debit_spread(105.0) == "Buy 105C / Sell 110C"


def test_suggest_call_debit_spread_small_strike():
    # strike=30 → width=2.5 → sell=32.5 → rounded to 32
    assert suggest_call_debit_spread(30.0) == "Buy 30C / Sell 32C"


def test_suggest_call_debit_spread_large_strike():
    # strike=250 → width=10 → sell=260
    assert suggest_call_debit_spread(250.0) == "Buy 250C / Sell 260C"


def test_suggest_call_debit_spread_direction():
    """The buy strike must be lower than the sell strike."""
    result = suggest_call_debit_spread(100.0)
    assert result.startswith("Buy ")
    assert "/ Sell " in result


# ── score_option_tfsa ─────────────────────────────────────────────────────────


def test_score_tfsa_zero_ask_returns_zero():
    assert score_option_tfsa(
        ask=0.0, strike=105.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.3, otm_pct=0.05,
    ) == 0.0


def test_score_tfsa_ask_exceeds_spread_width_returns_zero():
    # ask=6 > width=5 for strike=100 → no positive upside
    assert score_option_tfsa(
        ask=6.0, strike=100.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.3, otm_pct=0.05,
    ) == 0.0


def test_score_tfsa_higher_upside_ratio_wins():
    """Cheaper ask relative to spread width → higher TFSA score."""
    s_cheap = score_option_tfsa(
        ask=0.5, strike=105.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.3, otm_pct=0.05,
    )
    s_expensive = score_option_tfsa(
        ask=4.0, strike=105.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.3, otm_pct=0.05,
    )
    assert s_cheap > s_expensive


def test_score_tfsa_sweet_spot_otm():
    """Calls in the 3–10 % OTM range should score higher than deep OTM (20 %)."""
    s_sweet = score_option_tfsa(
        ask=1.0, strike=106.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.35, otm_pct=0.06,
    )
    s_deep = score_option_tfsa(
        ask=1.0, strike=120.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.35, otm_pct=0.20,
    )
    assert s_sweet > s_deep


def test_score_tfsa_higher_oi_wins():
    """More open interest → higher TFSA score (liquidity component)."""
    s_low_oi = score_option_tfsa(
        ask=1.0, strike=105.0, stock_price=100.0,
        open_interest=100, implied_volatility=0.3, otm_pct=0.05,
    )
    s_high_oi = score_option_tfsa(
        ask=1.0, strike=105.0, stock_price=100.0,
        open_interest=8000, implied_volatility=0.3, otm_pct=0.05,
    )
    assert s_high_oi > s_low_oi


def test_score_tfsa_moderate_iv_beats_very_low_iv():
    """Moderate IV (0.35) should score higher than very low IV (0.15) for TFSA."""
    s_low_iv = score_option_tfsa(
        ask=1.0, strike=105.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.15, otm_pct=0.05,
    )
    s_mod_iv = score_option_tfsa(
        ask=1.0, strike=105.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.35, otm_pct=0.05,
    )
    assert s_mod_iv > s_low_iv


def test_score_tfsa_itm_call_scores_zero():
    """ITM calls (otm_pct < 0) should receive no OTM sweet-spot score."""
    s_itm = score_option_tfsa(
        ask=1.0, strike=95.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.3, otm_pct=-0.05,
    )
    s_otm = score_option_tfsa(
        ask=1.0, strike=106.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.3, otm_pct=0.06,
    )
    assert s_otm > s_itm


def test_enrich_options_call_adds_tfsa_columns():
    """enrich_options should add tfsa_score and tfsa_spread for calls."""
    import pandas as pd

    expiry = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    df = pd.DataFrame([{
        "strike": 105.0, "bid": 1.0, "ask": 1.1,
        "openInterest": 1000, "volume": 50,
        "impliedVolatility": 0.35, "inTheMoney": False,
        "contractSymbol": "AAPL105C",
    }])
    enriched = enrich_options(df, stock_price=100.0, option_type="call", expiry=expiry, ticker="AAPL")

    assert "tfsa_score" in enriched.columns
    assert "tfsa_spread" in enriched.columns
    assert enriched["tfsa_score"].iloc[0] > 0
    assert enriched["tfsa_spread"].iloc[0] == "Buy 105C / Sell 110C"


def test_enrich_options_put_tfsa_columns_zero():
    """enrich_options should set tfsa_score=0 and tfsa_spread='' for puts."""
    import pandas as pd

    expiry = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    df = pd.DataFrame([{
        "strike": 95.0, "bid": 1.5, "ask": 1.6,
        "openInterest": 1000, "volume": 50,
        "impliedVolatility": 0.3, "inTheMoney": False,
        "contractSymbol": "AAPL95P",
    }])
    enriched = enrich_options(df, stock_price=100.0, option_type="put", expiry=expiry, ticker="AAPL")

    assert "tfsa_score" in enriched.columns
    assert "tfsa_spread" in enriched.columns
    assert enriched["tfsa_score"].iloc[0] == 0.0
    assert enriched["tfsa_spread"].iloc[0] == ""


# ── calculate_pop ─────────────────────────────────────────────────────────────


def test_calculate_pop_atm_put_near_half():
    """ATM short-put PoP should be close to 0.5 (slightly below due to -0.5σ²T term)."""
    pop = calculate_pop(100.0, 100.0, 0.30, 30, "put")
    assert 0.40 <= pop <= 0.60


def test_calculate_pop_deep_otm_put_high_pop():
    """A 20 % OTM short put with moderate IV should have PoP > 0.95."""
    pop = calculate_pop(80.0, 100.0, 0.25, 30, "put")
    assert pop > 0.95


def test_calculate_pop_near_atm_put_moderate_pop():
    """A 3 % OTM short put should have a moderate PoP (0.55–0.80)."""
    pop = calculate_pop(97.0, 100.0, 0.30, 30, "put")
    assert 0.55 <= pop <= 0.80


def test_calculate_pop_call_vs_put_roughly_symmetric():
    """Symmetric OTM call and put (equal % from ATM) should have approximately equal PoP."""
    pop_put = calculate_pop(97.0, 100.0, 0.30, 30, "put")
    pop_call = calculate_pop(103.0, 100.0, 0.30, 30, "call")
    assert abs(pop_put - pop_call) < 0.06


def test_calculate_pop_zero_iv_returns_neutral():
    """Zero IV is invalid; function should return the neutral sentinel 0.5."""
    assert calculate_pop(95.0, 100.0, 0.0, 30, "put") == pytest.approx(0.5)


def test_calculate_pop_zero_dte_returns_neutral():
    """Zero DTE is invalid; function should return the neutral sentinel 0.5."""
    assert calculate_pop(95.0, 100.0, 0.30, 0, "put") == pytest.approx(0.5)


def test_calculate_pop_range():
    """PoP should always be in (0, 1] for typical option parameters."""
    for iv in (0.15, 0.30, 0.60):
        for dte in (7, 30, 60):
            for strike_pct in (0.90, 0.95, 1.00, 1.05, 1.10):
                pop = calculate_pop(100.0 * strike_pct, 100.0, iv, dte, "put")
                assert 0.0 < pop <= 1.0


# ── score_option with PoP ────────────────────────────────────────────────────


def test_score_option_higher_pop_wins():
    """Same OTM% / same other params but higher PoP → higher score."""
    s_high_pop = score_option(
        bid=1.0, ask=1.1, strike=95.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.30, otm_pct=0.05, pop=0.90,
    )
    s_low_pop = score_option(
        bid=1.0, ask=1.1, strike=95.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.30, otm_pct=0.05, pop=0.55,
    )
    assert s_high_pop > s_low_pop


def test_score_option_pop_none_unchanged():
    """Omitting pop (default None) must leave scores identical to the pre-PoP behaviour."""
    kwargs = dict(
        bid=1.0, ask=1.1, strike=95.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.30, otm_pct=0.05,
    )
    assert score_option(**kwargs) == score_option(**kwargs, pop=None)


def test_score_option_pop_one_unchanged():
    """pop=1.0 (certainty) must leave the distance score unchanged."""
    kwargs = dict(
        bid=1.0, ask=1.1, strike=95.0, stock_price=100.0,
        open_interest=1000, implied_volatility=0.30, otm_pct=0.05,
    )
    assert score_option(**kwargs, pop=None) == pytest.approx(
        score_option(**kwargs, pop=1.0)
    )

