"""Unit tests for forward_eval P&L calculations."""

import pandas as pd
import pytest

from forward_eval import _realised_pnl_per_share, _COMMISSION_PER_SHARE


# ── _realised_pnl_per_share ───────────────────────────────────────────────────


def _put_row(**overrides) -> pd.Series:
    """Return a minimal short-put row for testing."""
    defaults = {
        "bid": 1.50,
        "ask": 1.60,
        "strike": 95.0,
        "max_spread_loss": 350.0,  # ($5 - $1.55mid) × 100
        "option_type": "put",
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def _call_row(**overrides) -> pd.Series:
    """Return a minimal short-call row for testing."""
    defaults = {
        "bid": 1.50,
        "ask": 1.60,
        "strike": 105.0,
        "max_spread_loss": 350.0,
        "option_type": "call",
    }
    defaults.update(overrides)
    return pd.Series(defaults)


# ── Short put ─────────────────────────────────────────────────────────────────


def test_put_expires_worthless_returns_mid_minus_commission():
    """Put expires OTM → full mid-price credit minus round-trip commission."""
    row = _put_row(bid=1.50, ask=1.60)
    mid = (1.50 + 1.60) / 2.0          # 1.55
    expected = mid - _COMMISSION_PER_SHARE
    assert _realised_pnl_per_share(row, expiry_close=100.0) == pytest.approx(expected)


def test_put_expires_at_strike_returns_mid_minus_commission():
    """Put expires exactly at the strike price → still treated as worthless."""
    row = _put_row(bid=1.50, ask=1.60)
    mid = (1.50 + 1.60) / 2.0
    pnl = _realised_pnl_per_share(row, expiry_close=95.0)  # close == strike
    assert pnl == pytest.approx(mid - _COMMISSION_PER_SHARE)


def test_put_partial_loss_capped_at_max_loss():
    """When assigned, loss should be capped at max_loss_per_share from max_spread_loss."""
    row = _put_row(bid=1.50, ask=1.60, strike=95.0, max_spread_loss=350.0)
    mid = (1.50 + 1.60) / 2.0          # 1.55
    # Deep ITM: raw_loss = 95 - 60 - 1.55 = 33.45 >> max_loss_per_share (3.50)
    pnl = _realised_pnl_per_share(row, expiry_close=60.0)
    expected = -3.50 - _COMMISSION_PER_SHARE
    assert pnl == pytest.approx(expected)


def test_put_small_loss_not_capped():
    """When the stock drops just past the strike, loss is not capped."""
    row = _put_row(bid=1.50, ask=1.60, strike=95.0, max_spread_loss=350.0)
    mid = (1.50 + 1.60) / 2.0          # 1.55
    # Close = 93.0 → raw_loss = 95 - 93 - 1.55 = 0.45 (< max_loss_per_share 3.50)
    pnl = _realised_pnl_per_share(row, expiry_close=93.0)
    expected = -(95.0 - 93.0 - mid) - _COMMISSION_PER_SHARE
    assert pnl == pytest.approx(expected, abs=1e-9)


# ── Short call ────────────────────────────────────────────────────────────────


def test_call_expires_worthless_returns_mid_minus_commission():
    """Call expires OTM → full mid-price credit minus round-trip commission."""
    row = _call_row(bid=1.50, ask=1.60)
    mid = (1.50 + 1.60) / 2.0
    pnl = _realised_pnl_per_share(row, expiry_close=100.0)  # below strike 105
    assert pnl == pytest.approx(mid - _COMMISSION_PER_SHARE)


def test_call_partial_loss_capped():
    """When assigned on a call, loss capped at max_loss_per_share."""
    row = _call_row(bid=1.50, ask=1.60, strike=105.0, max_spread_loss=350.0)
    mid = (1.50 + 1.60) / 2.0
    # Deep ITM: raw_loss = 150 - 105 - 1.55 = 43.45 >> max_loss_per_share (3.50)
    pnl = _realised_pnl_per_share(row, expiry_close=150.0)
    expected = -3.50 - _COMMISSION_PER_SHARE
    assert pnl == pytest.approx(expected)


# ── Backward-compat: missing ask falls back to bid ───────────────────────────


def test_missing_ask_falls_back_to_bid():
    """When ask is not in the row, mid-price = bid (no ask uplift)."""
    row = pd.Series({
        "bid": 1.50,
        "strike": 95.0,
        "max_spread_loss": 350.0,
        "option_type": "put",
    })
    # No "ask" key → premium should equal bid
    pnl = _realised_pnl_per_share(row, expiry_close=100.0)
    expected = 1.50 - _COMMISSION_PER_SHARE
    assert pnl == pytest.approx(expected)


def test_invalid_ask_falls_back_to_bid():
    """When ask is 0 or negative, mid-price falls back to bid."""
    row = _put_row(bid=1.50, ask=0.0)
    pnl = _realised_pnl_per_share(row, expiry_close=100.0)
    expected = 1.50 - _COMMISSION_PER_SHARE
    assert pnl == pytest.approx(expected)


# ── Unknown option type returns 0 ────────────────────────────────────────────


def test_unknown_option_type_returns_zero():
    row = pd.Series({
        "bid": 1.0, "ask": 1.1, "strike": 95.0,
        "max_spread_loss": 350.0, "option_type": "exotic",
    })
    assert _realised_pnl_per_share(row, expiry_close=100.0) == 0.0


# ── Commission constant sanity check ─────────────────────────────────────────


def test_commission_constant_positive():
    """Commission must be a small positive number."""
    assert 0 < _COMMISSION_PER_SHARE < 0.10
