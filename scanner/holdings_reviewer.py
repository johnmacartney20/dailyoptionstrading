"""Daily holdings review for persistent portfolio state.

Re-scores every live position using existing analyzer logic and applies
HOLD/FLAG/EXIT verdicts from configurable thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .analyzer import enrich_options, score_stock_growth, score_stock_stability
from .data_fetcher import get_options_chain, get_price_history, get_stock_price
from .portfolio_state import STATUS_EXIT, STATUS_FLAG, STATUS_HOLD, record_review

logger = logging.getLogger(__name__)

_OPTION_SUB_PORTFOLIOS = {"put-spread", "long-call"}


@dataclass
class HoldingReview:
    """Structured output for one reviewed holding."""

    ticker: str
    account_type: str
    sub_portfolio: str
    entry_score: float
    current_score: float
    score_delta: float
    days_held: int
    verdict: str
    reason: str


@dataclass
class ReviewSummary:
    """Aggregate review counts for reporting."""

    total: int
    holds: int
    flags: int
    exits: int


def _days_between(start_date: str, end_date: Optional[str] = None) -> int:
    if not start_date:
        return 0
    try:
        d0 = datetime.strptime(start_date, "%Y-%m-%d").date()
    except ValueError:
        return 0
    d1 = date.today() if end_date is None else datetime.strptime(end_date, "%Y-%m-%d").date()
    return max((d1 - d0).days, 0)


def _score_stock_position(
    ticker: str,
    sub_portfolio: str,
    market_return_20d: float,
) -> Tuple[float, str]:
    hist = get_price_history(ticker, period="3mo")
    if hist is None or hist.empty:
        return 0.0, "no recent price history"

    if sub_portfolio.lower() == "stability":
        score = score_stock_stability(hist)
    else:
        score = score_stock_growth(hist, market_return_20d)

    return float(score.composite), score.reasoning


def _score_option_position(
    position: Dict[str, Any],
) -> Tuple[float, str]:
    ticker = str(position.get("ticker", ""))
    sub_portfolio = str(position.get("sub_portfolio", "")).lower()
    metadata = position.get("metadata", {}) or {}

    option_type = str(metadata.get("option_type", "")).lower()
    expiry = str(metadata.get("expiry", ""))
    strike = metadata.get("strike", None)

    if not option_type or not expiry or strike is None:
        return 0.0, "missing option metadata (option_type/expiry/strike)"

    try:
        strike_float = float(strike)
    except (TypeError, ValueError):
        return 0.0, "invalid option strike metadata"

    stock_price = get_stock_price(ticker)
    if stock_price is None:
        return 0.0, "no stock price available"

    chain = get_options_chain(ticker, expiry)
    if chain is None:
        return 0.0, "no options chain available"

    calls_df, puts_df = chain
    options_df = calls_df if option_type == "call" else puts_df
    if options_df is None or options_df.empty:
        return 0.0, "empty options side for expiry"

    enriched = enrich_options(
        options_df,
        stock_price=float(stock_price),
        option_type=option_type,
        expiry=expiry,
        ticker=ticker,
    )
    if enriched.empty or "strike" not in enriched.columns:
        return 0.0, "unable to enrich options chain"

    idx = (enriched["strike"].astype(float) - strike_float).abs().idxmin()
    row = enriched.loc[idx]

    if sub_portfolio == "long-call":
        score = float(row.get("tfsa_score", 0.0) or 0.0)
    else:
        score = float(row.get("score", 0.0) or 0.0)

    return score, f"rescored {option_type} @ {float(row.get('strike', strike_float)):.2f}"


def _score_position(position: Dict[str, Any], market_return_20d: float) -> Tuple[float, str]:
    sub_portfolio = str(position.get("sub_portfolio", "")).lower()
    ticker = str(position.get("ticker", ""))

    if sub_portfolio in {"put-spread", "long-call"}:
        score, reason = _score_option_position(position)
        if score > 0:
            return score, reason
        # Safe fallback for positions missing option metadata.
        fallback, fallback_reason = _score_stock_position(
            ticker=ticker,
            sub_portfolio="growth",
            market_return_20d=market_return_20d,
        )
        return fallback, f"{reason}; fallback stock score"

    return _score_stock_position(ticker=ticker, sub_portfolio=sub_portfolio, market_return_20d=market_return_20d)


def _safe_float(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(val):
        return None
    return val


def _mid_from_row(row: pd.Series) -> Optional[float]:
    """Return a robust option mark using bid/ask midpoint fallback logic."""
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    last = _safe_float(row.get("lastPrice"))

    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if bid is not None and bid > 0:
        return bid
    if ask is not None and ask > 0:
        return ask
    if last is not None and last > 0:
        return last
    return None


def _nearest_strike_row(options_df: pd.DataFrame, strike: float) -> Optional[pd.Series]:
    if options_df is None or options_df.empty or "strike" not in options_df.columns:
        return None
    strike_series = pd.to_numeric(options_df["strike"], errors="coerce")
    diffs = (strike_series - float(strike)).abs()
    if diffs.isna().all():
        return None
    idx = diffs.idxmin()
    if pd.isna(idx):
        return None
    return options_df.loc[idx]


def _days_to_expiry(expiry: str, as_of: date) -> Optional[int]:
    if not expiry:
        return None
    try:
        exp_dt = datetime.strptime(str(expiry), "%Y-%m-%d").date()
    except ValueError:
        return None
    return max((exp_dt - as_of).days, 0)


def track_options_performance(
    positions: Sequence[Dict[str, Any]],
    as_of: Optional[str] = None,
) -> pd.DataFrame:
    """Mark options positions to market and append one daily performance snapshot.

    Mutates each position's ``metadata`` by updating ``performance_history`` and
    last-known mark/P&L fields. Returns a report-friendly DataFrame.
    """
    as_of_date = date.today() if as_of is None else datetime.strptime(as_of, "%Y-%m-%d").date()
    as_of_str = as_of_date.isoformat()

    rows: List[Dict[str, Any]] = []
    chain_cache: Dict[Tuple[str, str], Optional[Tuple[pd.DataFrame, pd.DataFrame]]] = {}
    price_cache: Dict[str, Optional[float]] = {}

    for pos in positions:
        sub_portfolio = str(pos.get("sub_portfolio", "")).lower()
        if sub_portfolio not in _OPTION_SUB_PORTFOLIOS:
            continue

        ticker = str(pos.get("ticker", "")).upper()
        account = str(pos.get("account_type", ""))
        status = str(pos.get("status", ""))
        qty = int(max(int(pos.get("quantity", 1) or 1), 1))
        entry_price = float(pos.get("entry_price", 0.0) or 0.0)

        metadata = pos.setdefault("metadata", {})
        option_type = str(metadata.get("option_type", "")).lower()
        expiry = str(metadata.get("expiry", ""))
        short_strike = _safe_float(metadata.get("strike"))
        long_strike = _safe_float(metadata.get("long_strike"))

        if not ticker or not option_type or not expiry or short_strike is None:
            rows.append(
                {
                    "ticker": ticker,
                    "account": account,
                    "sub_portfolio": sub_portfolio,
                    "option_type": option_type,
                    "expiry": expiry,
                    "strike": short_strike,
                    "long_strike": long_strike,
                    "qty": qty,
                    "entry": entry_price,
                    "mark": None,
                    "daily_change": None,
                    "unrealized_pnl": None,
                    "return_pct": None,
                    "underlying": None,
                    "dte": None,
                    "status": status,
                    "note": "missing option metadata",
                }
            )
            continue

        if ticker not in price_cache:
            price_cache[ticker] = get_stock_price(ticker)
        underlying = price_cache[ticker]

        cache_key = (ticker, expiry)
        if cache_key not in chain_cache:
            chain_cache[cache_key] = get_options_chain(ticker, expiry)
        chain = chain_cache[cache_key]
        if chain is None:
            rows.append(
                {
                    "ticker": ticker,
                    "account": account,
                    "sub_portfolio": sub_portfolio,
                    "option_type": option_type,
                    "expiry": expiry,
                    "strike": short_strike,
                    "long_strike": long_strike,
                    "qty": qty,
                    "entry": round(entry_price, 4),
                    "mark": None,
                    "daily_change": None,
                    "unrealized_pnl": None,
                    "return_pct": None,
                    "underlying": underlying,
                    "dte": _days_to_expiry(expiry, as_of_date),
                    "status": status,
                    "note": "no options chain available",
                }
            )
            continue

        calls_df, puts_df = chain
        options_df = calls_df if option_type == "call" else puts_df
        short_row = _nearest_strike_row(options_df, short_strike)
        if short_row is None:
            rows.append(
                {
                    "ticker": ticker,
                    "account": account,
                    "sub_portfolio": sub_portfolio,
                    "option_type": option_type,
                    "expiry": expiry,
                    "strike": short_strike,
                    "long_strike": long_strike,
                    "qty": qty,
                    "entry": round(entry_price, 4),
                    "mark": None,
                    "daily_change": None,
                    "unrealized_pnl": None,
                    "return_pct": None,
                    "underlying": underlying,
                    "dte": _days_to_expiry(expiry, as_of_date),
                    "status": status,
                    "note": "could not locate strike in options chain",
                }
            )
            continue

        short_mid = _mid_from_row(short_row)
        if short_mid is None:
            rows.append(
                {
                    "ticker": ticker,
                    "account": account,
                    "sub_portfolio": sub_portfolio,
                    "option_type": option_type,
                    "expiry": expiry,
                    "strike": short_strike,
                    "long_strike": long_strike,
                    "qty": qty,
                    "entry": round(entry_price, 4),
                    "mark": None,
                    "daily_change": None,
                    "unrealized_pnl": None,
                    "return_pct": None,
                    "underlying": underlying,
                    "dte": _days_to_expiry(expiry, as_of_date),
                    "status": status,
                    "note": "missing bid/ask for strike",
                }
            )
            continue

        mark = float(short_mid)
        note = "ok"
        if sub_portfolio == "put-spread" and long_strike is not None:
            long_row = _nearest_strike_row(options_df, long_strike)
            long_mid = _mid_from_row(long_row) if long_row is not None else None
            if long_mid is not None:
                mark = max(short_mid - long_mid, 0.0)
            else:
                note = "long leg quote missing; using short-leg mark"

        if sub_portfolio == "long-call":
            pnl_per_contract = (mark - entry_price) * 100.0
            basis = max(entry_price * 100.0, 1.0)
        else:
            pnl_per_contract = (entry_price - mark) * 100.0
            basis = max(entry_price * 100.0, 1.0)

        pnl_total = pnl_per_contract * qty
        return_pct = (pnl_per_contract / basis) * 100.0

        perf_hist = metadata.setdefault("performance_history", [])
        prev_mark: Optional[float] = None
        if perf_hist:
            last = perf_hist[-1]
            if str(last.get("date", "")) != as_of_str:
                prev_mark = _safe_float(last.get("mark"))
            else:
                if len(perf_hist) >= 2:
                    prev_mark = _safe_float(perf_hist[-2].get("mark"))
                perf_hist.pop()
        daily_change = None if prev_mark is None else (mark - prev_mark)

        snapshot = {
            "date": as_of_str,
            "mark": round(mark, 4),
            "daily_change": None if daily_change is None else round(daily_change, 4),
            "pnl_per_contract": round(pnl_per_contract, 2),
            "pnl_total": round(pnl_total, 2),
            "return_pct": round(return_pct, 2),
            "underlying": None if underlying is None else round(float(underlying), 4),
            "dte": _days_to_expiry(expiry, as_of_date),
            "status": status,
        }
        perf_hist.append(snapshot)
        metadata["last_mark_date"] = as_of_str
        metadata["last_mark"] = snapshot["mark"]
        metadata["last_underlying_price"] = snapshot["underlying"]
        metadata["last_option_pnl_total"] = snapshot["pnl_total"]
        metadata["last_option_pnl_per_contract"] = snapshot["pnl_per_contract"]
        metadata["last_option_return_pct"] = snapshot["return_pct"]

        rows.append(
            {
                "ticker": ticker,
                "account": account,
                "sub_portfolio": sub_portfolio,
                "option_type": option_type,
                "expiry": expiry,
                "strike": short_strike,
                "long_strike": long_strike,
                "qty": qty,
                "entry": round(entry_price, 4),
                "mark": snapshot["mark"],
                "daily_change": snapshot["daily_change"],
                "unrealized_pnl": snapshot["pnl_total"],
                "return_pct": snapshot["return_pct"],
                "underlying": snapshot["underlying"],
                "dte": snapshot["dte"],
                "status": status,
                "note": note,
            }
        )

    return pd.DataFrame(rows)


def _apply_thresholds(
    entry_score: float,
    current_score: float,
    hold_floor: float,
    hard_exit: float,
    score_decay_warn_pct: float,
) -> Tuple[str, str]:
    if current_score < hard_exit:
        return STATUS_EXIT, f"score {current_score:.2f} below hard exit {hard_exit:.2f}"

    if current_score < hold_floor:
        return STATUS_FLAG, f"score {current_score:.2f} below hold floor {hold_floor:.2f}"

    if entry_score > 0:
        decay = (entry_score - current_score) / entry_score
        if decay >= score_decay_warn_pct:
            return (
                STATUS_FLAG,
                f"score decay {decay * 100:.1f}% exceeds {score_decay_warn_pct * 100:.1f}%",
            )

    return STATUS_HOLD, f"score {current_score:.2f} above hold floor {hold_floor:.2f}"


def review_holdings(
    positions: Sequence[Dict[str, Any]],
    thresholds: Dict[str, float],
    market_return_20d: float = 0.0,
) -> List[HoldingReview]:
    """Re-score positions and return HOLD/FLAG/EXIT reviews."""
    hold_floor = float(thresholds.get("hold_floor", 6.0))
    hard_exit = float(thresholds.get("hard_exit", 4.5))
    score_decay_warn_pct = float(thresholds.get("score_decay_warn_pct", 0.30))

    reviews: List[HoldingReview] = []
    for pos in positions:
        metadata = pos.get("metadata", {}) or {}
        if bool(metadata.get("is_cash", False)):
            continue

        ticker = str(pos.get("ticker", "")).upper()
        account = str(pos.get("account_type", ""))
        sub_portfolio = str(pos.get("sub_portfolio", ""))
        entry_score = float(pos.get("entry_composite_score", 0.0) or 0.0)

        current_score, score_reason = _score_position(pos, market_return_20d=market_return_20d)
        score_delta = current_score - entry_score
        verdict, threshold_reason = _apply_thresholds(
            entry_score=entry_score,
            current_score=current_score,
            hold_floor=hold_floor,
            hard_exit=hard_exit,
            score_decay_warn_pct=score_decay_warn_pct,
        )

        reason = f"{threshold_reason}; {score_reason}"
        reviews.append(
            HoldingReview(
                ticker=ticker,
                account_type=account,
                sub_portfolio=sub_portfolio,
                entry_score=round(entry_score, 2),
                current_score=round(current_score, 2),
                score_delta=round(score_delta, 2),
                days_held=_days_between(str(pos.get("entry_date", ""))),
                verdict=verdict,
                reason=reason,
            )
        )

    return reviews


def apply_reviews_to_positions(
    positions: Sequence[Dict[str, Any]],
    reviews: Sequence[HoldingReview],
    review_date: Optional[str] = None,
) -> None:
    """Write review outputs back into mutable position dicts."""
    review_dt = review_date or date.today().isoformat()
    review_map: Dict[Tuple[str, str, str], HoldingReview] = {
        (r.ticker.upper(), r.account_type.upper(), r.sub_portfolio.lower()): r
        for r in reviews
    }

    for pos in positions:
        key = (
            str(pos.get("ticker", "")).upper(),
            str(pos.get("account_type", "")).upper(),
            str(pos.get("sub_portfolio", "")).lower(),
        )
        rev = review_map.get(key)
        if rev is None:
            continue
        record_review(
            position=pos,
            review_date=review_dt,
            current_score=rev.current_score,
            verdict=rev.verdict,
            reason=rev.reason,
        )


def review_summary(reviews: Sequence[HoldingReview]) -> ReviewSummary:
    """Return count summary for HOLD/FLAG/EXIT verdicts."""
    holds = sum(1 for r in reviews if r.verdict == STATUS_HOLD)
    flags = sum(1 for r in reviews if r.verdict == STATUS_FLAG)
    exits = sum(1 for r in reviews if r.verdict == STATUS_EXIT)
    return ReviewSummary(total=len(reviews), holds=holds, flags=flags, exits=exits)


def reviews_to_frame(reviews: Sequence[HoldingReview]) -> pd.DataFrame:
    """Convert reviews into a report-friendly DataFrame."""
    if not reviews:
        return pd.DataFrame(
            columns=[
                "ticker",
                "account",
                "sub_portfolio",
                "entry_score",
                "current_score",
                "score_delta",
                "days_held",
                "verdict",
                "reason",
            ]
        )

    return pd.DataFrame(
        [
            {
                "ticker": r.ticker,
                "account": r.account_type,
                "sub_portfolio": r.sub_portfolio,
                "entry_score": r.entry_score,
                "current_score": r.current_score,
                "score_delta": r.score_delta,
                "days_held": r.days_held,
                "verdict": r.verdict,
                "reason": r.reason,
            }
            for r in reviews
        ]
    )
