"""Data fetcher module.

Retrieves stock prices and options chains from Yahoo Finance,
a free public data source accessible via the ``yfinance`` library.
"""

import logging
import time
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# How long to pause between consecutive Yahoo Finance requests (seconds).
_REQUEST_DELAY = 0.25


def get_stock_price(ticker: str) -> Optional[float]:
    """Return the latest closing price for *ticker*, or ``None`` on failure.

    Tries ``fast_info.last_price`` first (lightweight), then falls back to a
    one-day history download so that after-hours or delayed quotes are handled
    gracefully.
    """
    try:
        t = yf.Ticker(ticker)
        price = t.fast_info.last_price
        if price and price > 0:
            return float(price)
        # Fallback: pull last closing price from history
        hist = t.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        logger.warning("No price data available for %s", ticker)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch price for %s: %s", ticker, exc)
        return None


def get_expiration_dates(ticker: str) -> List[str]:
    """Return available options expiration dates for *ticker*.

    Returns an empty list when options are not available or on error.
    """
    try:
        t = yf.Ticker(ticker)
        return list(t.options)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch expiration dates for %s: %s", ticker, exc)
        return []


def get_options_chain(
    ticker: str, expiry: str
) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return ``(calls_df, puts_df)`` for *ticker* at *expiry*, or ``None`` on error.

    Both DataFrames use the standard columns provided by yfinance:
    ``strike``, ``bid``, ``ask``, ``volume``, ``openInterest``,
    ``impliedVolatility``, ``inTheMoney``, ``contractSymbol``, etc.
    """
    try:
        t = yf.Ticker(ticker)
        chain = t.option_chain(expiry)
        time.sleep(_REQUEST_DELAY)
        return chain.calls, chain.puts
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not fetch options chain for %s %s: %s", ticker, expiry, exc
        )
        return None
