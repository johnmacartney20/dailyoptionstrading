"""Unit tests for scanner.data_fetcher.check_data_health."""

from unittest.mock import patch

import pytest

from scanner.data_fetcher import check_data_health


# ── check_data_health ─────────────────────────────────────────────────────────


def test_health_ok_when_all_prices_valid():
    """All tickers returning valid prices → status 'ok' and valid_pct == 1.0."""
    tickers = ["AAPL", "MSFT", "GOOGL"]
    with patch("scanner.data_fetcher.get_stock_price", return_value=150.0):
        result = check_data_health(tickers, sample_size=3)

    assert result["checked"] == 3
    assert result["valid"] == 3
    assert result["valid_pct"] == pytest.approx(1.0)
    assert result["status"] == "ok"
    assert result["failed_tickers"] == []


def test_health_degraded_when_some_prices_missing():
    """~75 % valid prices → status 'degraded'."""
    tickers = ["A", "B", "C", "D"]
    prices = {"A": 100.0, "B": None, "C": 50.0, "D": 75.0}

    with patch("scanner.data_fetcher.get_stock_price", side_effect=lambda t: prices[t]):
        result = check_data_health(tickers, sample_size=4)

    assert result["valid"] == 3
    assert result["status"] == "degraded"
    assert "B" in result["failed_tickers"]


def test_health_down_when_most_prices_missing():
    """< 70 % valid → status 'down'."""
    tickers = ["A", "B", "C", "D", "E"]
    prices = {"A": 100.0, "B": None, "C": None, "D": None, "E": None}

    with patch("scanner.data_fetcher.get_stock_price", side_effect=lambda t: prices[t]):
        result = check_data_health(tickers, sample_size=5)

    assert result["valid"] == 1
    assert result["status"] == "down"
    assert len(result["failed_tickers"]) == 4


def test_health_zero_price_treated_as_invalid():
    """A price of 0.0 should count as a failure."""
    with patch("scanner.data_fetcher.get_stock_price", return_value=0.0):
        result = check_data_health(["XYZ"], sample_size=1)

    assert result["valid"] == 0
    assert result["failed_tickers"] == ["XYZ"]


def test_health_sample_size_limits_checked_tickers():
    """sample_size should cap how many tickers are actually checked."""
    tickers = [f"T{i}" for i in range(20)]
    with patch("scanner.data_fetcher.get_stock_price", return_value=10.0) as mock_price:
        result = check_data_health(tickers, sample_size=5)

    assert result["checked"] == 5
    assert mock_price.call_count == 5


def test_health_empty_ticker_list():
    """Empty ticker list → checked=0, valid=0, status='down' (no data is treated as down)."""
    with patch("scanner.data_fetcher.get_stock_price") as mock_price:
        result = check_data_health([], sample_size=10)

    mock_price.assert_not_called()
    assert result["checked"] == 0
    assert result["valid"] == 0
    assert result["valid_pct"] == pytest.approx(0.0)
    assert result["status"] == "down"


def test_health_returns_dict_keys():
    """Return value must contain all expected keys."""
    with patch("scanner.data_fetcher.get_stock_price", return_value=50.0):
        result = check_data_health(["AAPL"])

    for key in ("checked", "valid", "valid_pct", "status", "failed_tickers"):
        assert key in result
