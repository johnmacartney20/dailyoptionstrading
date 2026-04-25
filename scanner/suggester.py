"""Options screening and suggestion module.

Applies the configurable screening criteria to enriched options DataFrames and
produces a ranked list of trade suggestions.
"""

import logging
from datetime import date
from typing import List, Optional

import pandas as pd

from .analyzer import enrich_options
from .config import SCREENING_PARAMS

logger = logging.getLogger(__name__)

# Columns to include in the final suggestions output (in display order).
OUTPUT_COLUMNS = [
    "ticker",
    "option_type",
    "expiry",
    "dte",
    "strike",
    "stock_price",
    "bid",
    "ask",
    "mid",
    "spread_pct",
    "openInterest",
    "volume",
    "impliedVolatility",
    "otm_pct",
    "pop",
    "annualized_return",
    "bid_ask_spread_pct",
    "risk_adjusted_return",
    "max_spread_loss",
    "spread_structure",
    "score",
    "tfsa_score",
    "tfsa_spread",
    "premarket_gap_pct",
    "earnings_within_expiry",
    "inTheMoney",
    "contractSymbol",
]


def screen_options(
    options_df: pd.DataFrame,
    stock_price: float,
    option_type: str,
    expiry: str,
    ticker: str,
    premarket_gap: Optional[float] = None,
    earnings_date: Optional[date] = None,
) -> pd.DataFrame:
    """Enrich and filter *options_df* returning only qualifying candidates.

    Filtering rules (all from :data:`~scanner.config.SCREENING_PARAMS`):

    * DTE within the configured window.
    * Bid ≥ ``min_bid``.
    * Open interest ≥ ``min_open_interest`` (500 – enforces liquidity).
    * Bid-ask spread ≤ ``max_bid_ask_spread_pct`` of the bid price.
    * OTM % between ``min_otm_pct`` (3 %) and ``max_otm_pct`` (15 %).
      Trades within 3 % of the stock price are excluded as too close to ATM.
    * Annualised return ≥ ``min_annualized_return_pct`` (floor only).
    * Max spread loss ≤ ``max_spread_loss`` (small-account compatibility).
    * **Earnings filter**: options whose expiry falls on or after the next
      confirmed earnings date are excluded.  Holding a short spread through
      a binary earnings event exposes the position to undefined IV crush or
      gap risk that the scoring model does not account for.

    The ``premarket_gap_pct`` column is informational only (not a filter).
    A pre-market gap of ≥ 3 % (up or down) is a regime-change signal —
    the column is preserved so the end user can review it before trading.

    Parameters
    ----------
    premarket_gap:
        Today's open-vs-prior-close gap as a fraction.  ``None`` when
        unavailable.
    earnings_date:
        Next confirmed earnings date for the underlying.  Options expiring
        on or after this date are removed from the results.

    Returns an empty :class:`~pandas.DataFrame` when no options qualify.
    """
    if options_df is None or options_df.empty:
        return pd.DataFrame()

    params = SCREENING_PARAMS

    # ── Data quality gate: detect stale / bad options chains ─────────────────
    # When >80 % of bids in a chain are zero the feed is almost certainly
    # stale or malformed.  Returning early avoids polluting the run with
    # phantom candidates that survive subsequent filters on a single row.
    zero_bid_ratio = (
        options_df["bid"].fillna(0).eq(0).sum() / max(len(options_df), 1)
    )
    if zero_bid_ratio > 0.80:
        logger.warning(
            "Skipping %s %s %s — %.0f%% of bids are zero (stale/bad chain).",
            ticker, option_type, expiry, zero_bid_ratio * 100,
        )
        return pd.DataFrame()

    # ── Pre-market gap direction filter ──────────────────────────────────────
    # A large downside gap (stock opened ≥ 3 % lower) signals negative near-
    # term sentiment; short puts on such names carry elevated assignment risk.
    # Symmetrically, a large upside gap suppresses short calls.
    # The threshold is 3 % to match the OTM inner band (min_otm_pct = 0.03).
    _GAP_THRESHOLD = 0.03
    if premarket_gap is not None:
        if option_type == "put" and premarket_gap <= -_GAP_THRESHOLD:
            logger.info(
                "Pre-market gap filter: suppressing %s puts for %s "
                "(gap = %+.1f%% ≤ −%.0f%%).",
                expiry, ticker, premarket_gap * 100, _GAP_THRESHOLD * 100,
            )
            return pd.DataFrame()
        if option_type == "call" and premarket_gap >= _GAP_THRESHOLD:
            logger.info(
                "Pre-market gap filter: suppressing %s calls for %s "
                "(gap = %+.1f%% ≥ +%.0f%%).",
                expiry, ticker, premarket_gap * 100, _GAP_THRESHOLD * 100,
            )
            return pd.DataFrame()

    # Enrich with computed metrics
    df = enrich_options(
        options_df, stock_price, option_type, expiry, ticker,
        premarket_gap=premarket_gap,
        earnings_date=earnings_date,
    )

    # ── DTE filter ────────────────────────────────────────────────────────────
    df = df[
        (df["dte"] >= params["min_dte"]) & (df["dte"] <= params["max_dte"])
    ]
    if df.empty:
        return pd.DataFrame()

    # ── Numeric filters ───────────────────────────────────────────────────────
    df = df[df["bid"] >= params["min_bid"]]
    df = df[
        df["openInterest"].fillna(0).astype(int) >= params["min_open_interest"]
    ]
    df = df[df["annualized_return"] >= params["min_annualized_return_pct"]]

    # ── Liquidity: bid-ask spread filter ─────────────────────────────────────
    df = df[
        df["bid_ask_spread_pct"].fillna(float("inf")) <= params["max_bid_ask_spread_pct"]
    ]

    # ── Moneyness filter: require min_otm_pct ≤ OTM % ≤ max_otm_pct ─────────
    df = df[
        (df["otm_pct"] >= params["min_otm_pct"])
        & (df["otm_pct"] <= params["max_otm_pct"])
    ]

    # ── Small-account: max spread loss filter ─────────────────────────────────
    df = df[df["max_spread_loss"] <= params["max_spread_loss"]]

    # ── Earnings filter: exclude options expiring through earnings ─────────────
    # Short spreads should not be held through a binary earnings event.
    # When an earnings date is known, any expiry on or after that date is dropped.
    if "earnings_within_expiry" in df.columns:
        before = len(df)
        df = df[~df["earnings_within_expiry"]]
        removed = before - len(df)
        if removed > 0:
            logger.info(
                "Earnings filter: removed %d %s %s option(s) for %s "
                "(earnings on or before expiry %s).",
                removed, option_type, expiry, ticker, earnings_date,
            )

    if df.empty:
        return pd.DataFrame()

    return df.sort_values("score", ascending=False).reset_index(drop=True)


def generate_suggestions(screened_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge all per-ticker/expiry screened DataFrames into a ranked table.

    Parameters
    ----------
    screened_frames:
        List of DataFrames produced by :func:`screen_options`.  May contain
        empty DataFrames – they are silently ignored.

    Returns
    -------
    A single DataFrame sorted by ``score`` descending, with only the standard
    :data:`OUTPUT_COLUMNS` (any missing columns are skipped gracefully).
    """
    non_empty = [f for f in screened_frames if not f.empty]
    if not non_empty:
        return pd.DataFrame()

    combined = pd.concat(non_empty, ignore_index=True)

    # Keep only output columns that are actually present
    cols = [c for c in OUTPUT_COLUMNS if c in combined.columns]
    combined = combined[cols].sort_values("score", ascending=False).reset_index(drop=True)
    return combined
