"""Data fetcher module.

Retrieves stock prices and options chains from Yahoo Finance,
a free public data source accessible via the ``yfinance`` library.
"""

import logging
import time
from datetime import date, datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# How long to pause between consecutive Yahoo Finance requests (seconds).
_REQUEST_DELAY = 0.25

# Retry settings for transient network failures.
_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY: float = 1.0  # seconds; doubles on each attempt


def get_stock_price(ticker: str) -> Optional[float]:
    """Return the latest closing price for *ticker*, or ``None`` on failure.

    Tries ``fast_info.last_price`` first (lightweight), then falls back to a
    one-day history download so that after-hours or delayed quotes are handled
    gracefully.  Retries up to :data:`_MAX_RETRIES` times on transient errors.
    """
    for attempt in range(_MAX_RETRIES):
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
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.debug(
                    "Retrying price fetch for %s (attempt %d/%d, wait %.1fs): %s",
                    ticker, attempt + 1, _MAX_RETRIES, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Could not fetch price for %s after %d attempts: %s",
                    ticker, _MAX_RETRIES, exc,
                )
    return None


def get_expiration_dates(ticker: str) -> List[str]:
    """Return available options expiration dates for *ticker*.

    Returns an empty list when options are not available or on error.
    Retries up to :data:`_MAX_RETRIES` times on transient errors.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            t = yf.Ticker(ticker)
            return list(t.options)
        except Exception as exc:  # noqa: BLE001
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.debug(
                    "Retrying expiration dates for %s (attempt %d/%d, wait %.1fs): %s",
                    ticker, attempt + 1, _MAX_RETRIES, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Could not fetch expiration dates for %s after %d attempts: %s",
                    ticker, _MAX_RETRIES, exc,
                )
    return []


def get_price_history(ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
    """Return OHLCV daily price history for *ticker* over *period*, or ``None`` on failure.

    The returned DataFrame has the standard yfinance columns:
    ``Open``, ``High``, ``Low``, ``Close``, ``Volume``, etc.
    Uses a ``period`` string accepted by yfinance (e.g. ``"3mo"``, ``"6mo"``).
    Retries up to :data:`_MAX_RETRIES` times on transient errors.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period)
            time.sleep(_REQUEST_DELAY)
            if hist.empty:
                logger.warning("No price history available for %s", ticker)
                return None
            return hist
        except Exception as exc:  # noqa: BLE001
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.debug(
                    "Retrying price history for %s (attempt %d/%d, wait %.1fs): %s",
                    ticker, attempt + 1, _MAX_RETRIES, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Could not fetch price history for %s after %d attempts: %s",
                    ticker, _MAX_RETRIES, exc,
                )
    return None


def get_market_return(benchmark: str = "SPY", period_days: int = 20) -> float:
    """Return the trailing *period_days*-day simple return for *benchmark*.

    Uses daily close prices from ``get_price_history``.  Returns ``0.0`` when
    insufficient history is available so callers receive a neutral baseline.

    Logs at ``ERROR`` level when the benchmark cannot be fetched so that stale
    or inflated relative-strength scores are visible in the run log.
    """
    hist = get_price_history(benchmark, period="3mo")
    if hist is None:
        logger.error(
            "Failed to fetch %s history; market benchmark defaulting to 0.0 "
            "— relative-strength scores may be unreliable this run.",
            benchmark,
        )
        return 0.0
    if len(hist) < period_days + 1:
        logger.error(
            "Insufficient %s history (%d rows, need ≥ %d); "
            "market benchmark defaulting to 0.0.",
            benchmark, len(hist), period_days + 1,
        )
        return 0.0
    close = hist["Close"].astype(float)
    return float(close.iloc[-1]) / float(close.iloc[-(period_days + 1)]) - 1.0


def get_earnings_date(ticker: str) -> Optional[date]:
    """Return the next confirmed earnings date for *ticker*, or ``None``.

    Uses ``yf.Ticker.calendar`` which returns a dict with an
    ``"Earnings Date"`` key whose value is a list of ``datetime``-like
    objects.  Only the first (soonest) future entry is returned.

    Returns ``None`` when the calendar is unavailable, empty, or the date
    cannot be parsed, so callers receive a safe fallback without crashing.
    """
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None:
            return None

        # yfinance may return a dict or a DataFrame depending on version.
        if isinstance(cal, pd.DataFrame):
            if "Earnings Date" not in cal.index:
                return None
            raw = cal.loc["Earnings Date"]
            dates = list(raw) if hasattr(raw, "__iter__") else [raw]
        elif isinstance(cal, dict):
            dates = cal.get("Earnings Date") or []
        else:
            return None

        today = date.today()
        for entry in dates:
            try:
                if isinstance(entry, (datetime, pd.Timestamp)):
                    d = entry.date() if hasattr(entry, "date") else entry
                else:
                    d = datetime.strptime(str(entry)[:10], "%Y-%m-%d").date()
                if d >= today:
                    return d
            except (ValueError, TypeError):
                continue
        return None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not fetch earnings date for %s: %s", ticker, exc)
        return None


def get_premarket_gap(ticker: str) -> Optional[float]:
    """Return the pre-market / overnight gap as a fraction of the prior close.

    Computed as ``(today_open - prior_close) / prior_close`` using the last
    two trading sessions from a 5-day daily history download.  A positive
    value means the stock opened higher; negative means it opened lower.

    Returns ``None`` when the data is unavailable (e.g. before market open,
    weekend, or a failed fetch), so callers can treat missing as neutral.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d", interval="1d")
        if hist is None or len(hist) < 2:
            return None
        prior_close = float(hist["Close"].iloc[-2])
        today_open = float(hist["Open"].iloc[-1])
        if prior_close <= 0:
            return None
        return (today_open - prior_close) / prior_close
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not compute pre-market gap for %s: %s", ticker, exc)
        return None


def get_options_chain(
    ticker: str, expiry: str
) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return ``(calls_df, puts_df)`` for *ticker* at *expiry*, or ``None`` on error.

    Both DataFrames use the standard columns provided by yfinance:
    ``strike``, ``bid``, ``ask``, ``volume``, ``openInterest``,
    ``impliedVolatility``, ``inTheMoney``, ``contractSymbol``, etc.
    Retries up to :data:`_MAX_RETRIES` times on transient errors.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            t = yf.Ticker(ticker)
            chain = t.option_chain(expiry)
            time.sleep(_REQUEST_DELAY)
            return chain.calls, chain.puts
        except Exception as exc:  # noqa: BLE001
            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.debug(
                    "Retrying options chain for %s %s (attempt %d/%d, wait %.1fs): %s",
                    ticker, expiry, attempt + 1, _MAX_RETRIES, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "Could not fetch options chain for %s %s after %d attempts: %s",
                    ticker, expiry, _MAX_RETRIES, exc,
                )
    return None
