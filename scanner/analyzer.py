"""Options analytics module.

Pure-function helpers for computing metrics used to rank and filter options:
days-to-expiration, out-of-the-money percentage, annualized return on capital,
risk-adjusted return, suggested spread structure, and a composite score that
prioritises high-probability, executable trades over raw annualised return.

Also contains composite stock scoring functions used for TFSA (growth) and
RRSP (stability) portfolio allocation models.
"""

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional

import pandas as pd


def _safe_int(value, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def calculate_dte(expiry_str: str) -> int:
    """Return calendar days from today to *expiry_str* (``"YYYY-MM-DD"`` format).

    Returns 0 if the expiration date is today or in the past.
    """
    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    delta = (expiry - date.today()).days
    return max(delta, 0)


def calculate_otm_pct(strike: float, stock_price: float, option_type: str) -> float:
    """Return the out-of-the-money percentage for an option.

    Positive values mean the option is OTM; negative values mean ITM.

    * **put**:  ``(stock_price - strike) / stock_price``
      – a put with strike < stock_price is OTM.
    * **call**: ``(strike - stock_price) / stock_price``
      – a call with strike > stock_price is OTM.
    """
    if stock_price <= 0:
        return float("nan")
    if option_type == "put":
        return (stock_price - strike) / stock_price
    # call
    return (strike - stock_price) / stock_price


def calculate_annualized_return(bid: float, strike: float, dte: int) -> float:
    """Return the annualised return on capital (as a percentage) for a short option.

    Formula: ``(bid / strike) × (365 / dte) × 100``

    Returns ``0.0`` when inputs are invalid.
    """
    if dte <= 0 or strike <= 0 or bid <= 0:
        return 0.0
    return (bid / strike) * (365.0 / dte) * 100.0


def _spread_width(strike: float) -> float:
    """Return an appropriate spread width (dollars) for a defined-risk spread.

    Keeps the per-contract max loss within small-account limits:
    * strike < $50  → $2.50 wide  (max loss ≈ $250)
    * strike < $200 → $5.00 wide  (max loss ≈ $500)
    * strike ≥ $200 → $10.00 wide (max loss ≈ $1,000)
    """
    if strike < 50:
        return 2.5
    if strike < 200:
        return 5.0
    return 10.0


def calculate_risk_adjusted_return(bid: float, strike: float) -> float:
    """Return the risk-adjusted return ratio for a short spread.

    Defined as ``premium_received / max_loss_per_share`` where max loss is
    ``spread_width - bid`` (a standard defined-risk spread).

    Returns ``0.0`` when the inputs are invalid or the bid exceeds the spread
    width (which would indicate a data anomaly).
    """
    width = _spread_width(strike)
    max_loss = width - bid
    if bid <= 0 or max_loss <= 0:
        return 0.0
    return bid / max_loss


def suggest_spread_structure(strike: float, option_type: str) -> str:
    """Return a human-readable spread structure string for a defined-risk trade.

    Examples
    --------
    * put  at strike 95  → ``"Sell 95P / Buy 90P"``
    * call at strike 105 → ``"Sell 105C / Buy 110C"``
    """
    width = _spread_width(strike)
    if option_type == "put":
        long_strike = strike - width
        return f"Sell {strike:.0f}P / Buy {long_strike:.0f}P"
    long_strike = strike + width
    return f"Sell {strike:.0f}C / Buy {long_strike:.0f}C"


def suggest_call_debit_spread(strike: float) -> str:
    """Return a reference spread string showing the long call and a higher strike.

    Used internally to populate the ``tfsa_spread`` column for informational
    purposes only.  The TFSA strategy is a **single-leg long call** – the sell
    leg shown here is *not* executed.

    Example
    -------
    * strike 105 → ``"Buy 105C / Sell 110C"``
    """
    width = _spread_width(strike)
    sell_strike = strike + width
    return f"Buy {strike:.0f}C / Sell {sell_strike:.0f}C"


def score_option_tfsa(
    ask: float,
    strike: float,
    stock_price: float,
    open_interest: int,
    implied_volatility: float,
    otm_pct: float,
) -> float:
    """Return a composite TFSA score for ranking long-call candidates (higher is better).

    Unlike :func:`score_option`, which rewards premium *sellers*, this score
    rewards trades with **asymmetric upside and defined risk** – suitable for a
    TFSA where only long (single-leg) calls are permitted.  Components:

    1. **Upside ratio** (0–35 pts): potential upside relative to premium paid.
       A higher ratio means more potential profit per dollar risked.
    2. **OTM sweet-spot** (0–30 pts): rewards strikes 3–10 % OTM, where long
       calls offer the best leverage-vs-probability balance.
    3. **Liquidity** (0–20 pts): log-scaled open-interest bonus (same as the
       standard scorer) rewarding depth and participation.
    4. **IV momentum** (0–15 pts): moderate-to-high IV signals potential for a
       large directional move; very high IV is penalised slightly because it
       inflates the premium cost.

    Returns ``0.0`` when inputs are invalid or the ask exceeds the spread width
    (no positive upside).
    """
    if ask <= 0 or strike <= 0:
        return 0.0

    width = _spread_width(strike)
    max_profit = width - ask
    if max_profit <= 0:
        return 0.0

    # ── 1. Upside ratio (0–35 pts) ────────────────────────────────────────────
    upside_ratio = max_profit / ask
    if upside_ratio < 0.5:
        upside_score = upside_ratio * 20.0
    elif upside_ratio < 2.0:
        upside_score = 10.0 + (upside_ratio - 0.5) / 1.5 * 15.0
    else:
        upside_score = min(25.0 + (upside_ratio - 2.0) * 5.0, 35.0)

    # ── 2. OTM sweet-spot (0–30 pts) ─────────────────────────────────────────
    if otm_pct < 0:
        otm_score = 0.0
    elif otm_pct < 0.02:
        otm_score = 5.0 * otm_pct / 0.02
    elif otm_pct < 0.05:
        otm_score = 5.0 + 20.0 * (otm_pct - 0.02) / 0.03
    elif otm_pct < 0.10:
        otm_score = 25.0 + 5.0 * (otm_pct - 0.05) / 0.05
    elif otm_pct <= 0.15:
        otm_score = 30.0 - 10.0 * (otm_pct - 0.10) / 0.05
    else:
        otm_score = 20.0

    # ── 3. Liquidity (0–20 pts) ───────────────────────────────────────────────
    liquidity_score = min(
        max(math.log10(max(open_interest, 1)) - 1.5, 0.0) * 9.0, 20.0
    )

    # ── 4. IV momentum (0–15 pts) ─────────────────────────────────────────────
    if implied_volatility < 0.20:
        iv_score = 0.0
    elif implied_volatility < 0.35:
        iv_score = (implied_volatility - 0.20) / 0.15 * 8.0
    elif implied_volatility < 0.55:
        iv_score = 8.0 + (implied_volatility - 0.35) / 0.20 * 7.0
    else:
        iv_score = max(15.0 - (implied_volatility - 0.55) * 10.0, 10.0)

    return upside_score + otm_score + liquidity_score + iv_score


def score_option(
    bid: float,
    ask: float,
    strike: float,
    stock_price: float,
    open_interest: int,
    implied_volatility: float,
    otm_pct: float,
) -> float:
    """Return a composite score for ranking options (higher is better).

    The score is built from four components that collectively prioritise
    high-probability, liquid, and risk-adjusted trades.  The thresholds are
    calibrated so that typical trades span roughly 15–95 pts, avoiding
    compression at the top end:

    1. **Distance score** (0–40 pts): rewards strikes further OTM.
       ATM or within 2 % OTM receive no credit.  The reward ramp is
       intentionally shallow in the 2–5 % zone and steepens past 5 %,
       requiring 20 %+ OTM to reach the ceiling.
    2. **Liquidity score** (up to 25 pts, can be slightly negative for very
       wide markets): log-scaled open-interest bonus — requires OI ≥ ~8 000
       to hit the 20-pt ceiling (vs. OI ~316 previously) — plus a bid-ask
       spread bonus for tight markets and a −3 pt penalty for very wide
       markets (spread > 25 % of bid).
    3. **IV-edge score** (0–20 pts): rewards high implied-volatility
       environments but penalises the combination of elevated IV *and* tight
       strike distance (elevated gap-risk).  The penalty now applies when
       OTM < 4 % (was 3 %) and has three tiers; the environmental bonus
       requires IV ≥ 0.35 (was 0.30).
    4. **Risk-adjusted return** (0–15 pts, capped): premium / max-loss on a
       standard defined-risk spread.  Low-credit trades (ratio < 0.15)
       receive a steeper penalty than before; the cap is only reached at a
       ratio ≥ ~0.55.
    """
    if bid <= 0 or strike <= 0:
        return 0.0

    # ── 1. Distance score (0–40 pts) ─────────────────────────────────────────
    # Shallow ramp in the 2–5 % zone enforces stronger penalties for tight
    # strikes; the reward accelerates past 5 % and plateaus only at 20 % OTM.
    if otm_pct < 0.02:
        distance_score = 0.0
    elif otm_pct < 0.03:
        distance_score = 5.0 * (otm_pct - 0.02) / 0.01
    elif otm_pct < 0.05:
        distance_score = 5.0 + 15.0 * (otm_pct - 0.03) / 0.02
    elif otm_pct < 0.10:
        distance_score = 20.0 + 13.0 * (otm_pct - 0.05) / 0.05
    elif otm_pct <= 0.20:
        distance_score = 33.0 + 7.0 * (otm_pct - 0.10) / 0.10
    else:
        distance_score = 40.0

    # ── 2. Liquidity score (up to 25 pts) ────────────────────────────────────
    # OI bar raised: OI < ~32 → 0 pts; OI ~8 000 → 20 pts (old ceiling at 316).
    oi_score = min(max(math.log10(max(open_interest, 1)) - 1.5, 0.0) * 9.0, 20.0)

    spread_score = 0.0
    if bid > 0 and ask > bid:
        spread_pct = (ask - bid) / bid
        if spread_pct < 0.05:
            spread_score = 5.0
        elif spread_pct < 0.10:
            spread_score = 2.5
        elif spread_pct > 0.25:
            spread_score = -3.0

    liquidity_score = oi_score + spread_score

    # ── 3. IV-edge score (0–20 pts) ──────────────────────────────────────────
    iv_base = min(implied_volatility * 40.0, 15.0)

    iv_penalty = 0.0
    if implied_volatility > 0.50 and otm_pct < 0.04:
        iv_penalty = 12.0
    elif implied_volatility > 0.40 and otm_pct < 0.04:
        iv_penalty = 8.0
    elif implied_volatility > 0.30 and otm_pct < 0.04:
        iv_penalty = 4.0

    iv_env_bonus = 5.0 if implied_volatility >= 0.35 else 0.0
    iv_edge_score = min(max(iv_base - iv_penalty, 0.0) + iv_env_bonus, 20.0)

    # ── 4. Risk-adjusted return (0–15 pts, capped) ───────────────────────────
    # Steeper ramp-up below 0.15 ratio; cap reached only at ratio ≥ ~0.55.
    risk_adj = calculate_risk_adjusted_return(bid, strike)
    if risk_adj < 0.15:
        risk_adj_score = risk_adj * 20.0
    else:
        risk_adj_score = min(3.0 + (risk_adj - 0.15) * 30.0, 15.0)

    return distance_score + liquidity_score + iv_edge_score + risk_adj_score


def enrich_options(
    df: pd.DataFrame,
    stock_price: float,
    option_type: str,
    expiry: str,
    ticker: str,
) -> pd.DataFrame:
    """Add computed columns to a raw options DataFrame.

    Added columns: ``ticker``, ``option_type``, ``expiry``, ``dte``,
    ``stock_price``, ``otm_pct``, ``annualized_return``,
    ``bid_ask_spread_pct``, ``risk_adjusted_return``, ``max_spread_loss``,
    ``spread_structure``, ``score``.
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    dte = calculate_dte(expiry)

    out["ticker"] = ticker
    out["option_type"] = option_type
    out["expiry"] = expiry
    out["dte"] = dte
    out["stock_price"] = stock_price

    out["otm_pct"] = out["strike"].apply(
        lambda s: calculate_otm_pct(s, stock_price, option_type)
    )
    out["annualized_return"] = out.apply(
        lambda row: calculate_annualized_return(row["bid"], row["strike"], dte),
        axis=1,
    )

    # Bid-ask spread as a fraction of the bid price
    out["bid_ask_spread_pct"] = out.apply(
        lambda row: (row["ask"] - row["bid"]) / row["bid"]
        if row.get("ask", 0) > 0 and row["bid"] > 0
        else float("nan"),
        axis=1,
    )

    out["risk_adjusted_return"] = out.apply(
        lambda row: calculate_risk_adjusted_return(row["bid"], row["strike"]),
        axis=1,
    )

    # Max loss per contract (in dollars) for the defined-risk spread
    out["max_spread_loss"] = out.apply(
        lambda row: max((_spread_width(row["strike"]) - row["bid"]) * 100.0, 0.0),
        axis=1,
    )

    out["spread_structure"] = out.apply(
        lambda row: suggest_spread_structure(row["strike"], option_type),
        axis=1,
    )

    out["score"] = out.apply(
        lambda row: score_option(
            row["bid"],
            float(row.get("ask", 0.0) or 0.0),
            row["strike"],
            stock_price,
            _safe_int(row.get("openInterest"), 0),
            float(row.get("impliedVolatility", 0.0) or 0.0),
            row["otm_pct"],
        ),
        axis=1,
    )

    # ── TFSA-specific columns (populated for calls only) ──────────────────────
    if option_type == "call":
        out["tfsa_score"] = out.apply(
            lambda row: score_option_tfsa(
                float(row.get("ask", 0.0) or 0.0),
                row["strike"],
                stock_price,
                _safe_int(row.get("openInterest"), 0),
                float(row.get("impliedVolatility", 0.0) or 0.0),
                row["otm_pct"],
            ),
            axis=1,
        )
        out["tfsa_spread"] = out["strike"].apply(suggest_call_debit_spread)
    else:
        out["tfsa_score"] = 0.0
        out["tfsa_spread"] = ""

    return out


# ── Stock scoring for TFSA / RRSP allocation ──────────────────────────────────

# Minimum trading days of price history required for reliable scoring.
_MIN_HISTORY_DAYS: int = 20


@dataclass
class StockScoreComponents:
    """Composite stock score with individual component breakdown.

    Used by both the TFSA growth model and the RRSP stability model.
    All components are non-negative; the composite is their sum.
    """

    trend_strength: float       # 0–30 (TFSA) / 0–30 (RRSP: consistency)
    relative_strength: float    # 0–20 (TFSA only; 0.0 for RRSP)
    volatility_control: float   # 0–15 (TFSA) / 0–30 (RRSP: low-vol reward)
    liquidity: float            # 0–15 (TFSA) / 0–20 (RRSP)
    drawdown_risk: float        # 0–20 (both)
    composite: float            # sum of all components (0–100)
    reasoning: str              # human-readable explanation of key drivers


def score_stock_growth(
    price_history: pd.DataFrame,
    market_return_20d: float = 0.0,
) -> StockScoreComponents:
    """Score a stock for TFSA growth allocation (higher = better, max ≈ 100).

    Five weighted components:

    1. **Trend Strength** (0–30 pts): price above 20/50-day MAs, plus
       short-term (5-day) and medium-term (20-day) momentum bonuses.
    2. **Relative Strength** (0–20 pts): 20-day return in excess of the
       broader market (e.g. SPY).  Underperforming stocks score 0.
    3. **Volatility Control** (0–15 pts): annualised vol < 15 % earns full
       points; penalises excessively volatile stocks progressively.
    4. **Liquidity** (0–15 pts): log-scaled 20-day average daily volume.
       ~100 K shares/day earns ~5 pts; 10 M+ earns the full 15 pts.
    5. **Drawdown Risk** (0–20 pts): penalises stocks that are heavily
       extended above their 20-day MA or sitting at a 20-day high
       (reversal risk zone).

    Parameters
    ----------
    price_history:
        OHLCV DataFrame from ``get_price_history`` (yfinance format).
        Must contain ``Close`` and ``Volume`` columns.
    market_return_20d:
        Trailing 20-day return of the benchmark (e.g. SPY).  Pass ``0.0``
        for a neutral baseline when the benchmark is unavailable.

    Returns
    -------
    :class:`StockScoreComponents`
        All scores are 0 when ``price_history`` has fewer than
        :data:`_MIN_HISTORY_DAYS` rows.
    """
    _zero = StockScoreComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Insufficient price history")
    if price_history is None or len(price_history) < _MIN_HISTORY_DAYS:
        return _zero

    close = price_history["Close"].astype(float)
    volume = price_history["Volume"].astype(float)
    current_price = float(close.iloc[-1])
    if current_price <= 0:
        return _zero

    # ── 1. Trend Strength (0–30 pts) ─────────────────────────────────────────
    ma20 = float(close.tail(20).mean())
    above_20ma = current_price > ma20
    trend_pts = 10.0 if above_20ma else 0.0

    above_50ma = False
    if len(close) >= 50:
        ma50 = float(close.tail(50).mean())
        above_50ma = current_price > ma50
        trend_pts += 10.0 if above_50ma else 0.0

    # 5-day momentum (0–5 pts)
    if len(close) >= 6:
        mom_5d = current_price / float(close.iloc[-6]) - 1.0
        if mom_5d > 0:
            trend_pts += min(mom_5d / 0.05 * 5.0, 5.0)

    # 20-day momentum (0–5 pts)
    if len(close) >= 21:
        mom_20d = current_price / float(close.iloc[-21]) - 1.0
        if mom_20d > 0:
            trend_pts += min(mom_20d / 0.10 * 5.0, 5.0)

    trend_pts = min(trend_pts, 30.0)

    # ── 2. Relative Strength (0–20 pts) ──────────────────────────────────────
    rs_pts = 0.0
    stock_return_20d = 0.0
    if len(close) >= 21:
        stock_return_20d = current_price / float(close.iloc[-21]) - 1.0
        excess = stock_return_20d - market_return_20d
        if excess > 0:
            rs_pts = min(excess / 0.10 * 20.0, 20.0)   # linear: each 1% excess → 2 pts, capped at 20 pts (10% excess)
        # underperformance floors at 0 (no negative)

    # ── 3. Volatility Control (0–15 pts) ─────────────────────────────────────
    daily_returns = close.pct_change().dropna()
    ann_vol = 0.0
    vol_pts = 7.5  # neutral default when insufficient history
    if len(daily_returns) >= 10:
        ann_vol = float(daily_returns.tail(20).std()) * math.sqrt(252.0)
        if ann_vol <= 0.15:
            vol_pts = 15.0
        elif ann_vol <= 0.30:
            vol_pts = 15.0 - (ann_vol - 0.15) / 0.15 * 5.0
        elif ann_vol <= 0.50:
            vol_pts = 10.0 - (ann_vol - 0.30) / 0.20 * 10.0
        else:
            vol_pts = 0.0
        vol_pts = max(vol_pts, 0.0)

    # ── 4. Liquidity (0–15 pts) ───────────────────────────────────────────────
    avg_vol_20d = float(volume.tail(20).mean())
    liq_pts = min(max(math.log10(max(avg_vol_20d, 1.0)) - 4.0, 0.0) * 5.0, 15.0)

    # ── 5. Drawdown Risk (0–20 pts) ───────────────────────────────────────────
    high_20d = float(close.tail(20).max())
    extension = (current_price - ma20) / ma20 if ma20 > 0 else 0.0
    pct_from_high = (high_20d - current_price) / high_20d if high_20d > 0 else 0.0

    # Penalise overextension above 20-day MA
    if extension > 0.20:
        ext_penalty = 15.0
    elif extension > 0.10:
        ext_penalty = 8.0
    elif extension > 0.05:
        ext_penalty = 3.0
    else:
        ext_penalty = 0.0

    # Penalise stocks at their 20-day high (reversal risk) or deep drawdown
    if pct_from_high < 0.02:
        high_penalty = 5.0   # very close to top – reversal zone
    elif pct_from_high > 0.25:
        high_penalty = 5.0   # large pullback – avoid catching a falling knife
    else:
        high_penalty = 0.0

    drawdown_pts = max(20.0 - ext_penalty - high_penalty, 0.0)

    # ── Composite ─────────────────────────────────────────────────────────────
    composite = trend_pts + rs_pts + vol_pts + liq_pts + drawdown_pts

    # ── Reasoning ─────────────────────────────────────────────────────────────
    reasons: List[str] = []
    if above_20ma:
        reasons.append("above 20MA")
    else:
        reasons.append("below 20MA")
    if above_50ma:
        reasons.append("above 50MA")
    elif len(close) >= 50:
        reasons.append("below 50MA")
    if rs_pts >= 10.0:
        reasons.append("outperforming market")
    elif stock_return_20d < market_return_20d and len(close) >= 21:
        reasons.append("lagging market")
    if ann_vol > 0 and ann_vol <= 0.25:
        reasons.append(f"low vol ({ann_vol:.0%})")
    elif ann_vol > 0.45:
        reasons.append(f"high vol ({ann_vol:.0%}) penalised")
    if extension > 0.10:
        reasons.append(f"extended {extension:.0%} above MA")
    if liq_pts >= 10.0:
        reasons.append("strong liquidity")
    reasoning = "; ".join(reasons) if reasons else "moderate trend and momentum"

    return StockScoreComponents(
        trend_strength=round(trend_pts, 2),
        relative_strength=round(rs_pts, 2),
        volatility_control=round(vol_pts, 2),
        liquidity=round(liq_pts, 2),
        drawdown_risk=round(drawdown_pts, 2),
        composite=round(composite, 2),
        reasoning=reasoning,
    )


def score_stock_stability(
    price_history: pd.DataFrame,
) -> StockScoreComponents:
    """Score a stock for RRSP stability allocation (higher = better, max ≈ 100).

    Four components optimised for long-term, low-turnover holdings:

    1. **Consistency** (0–30 pts): above 20/50-day MAs and positive
       20-day return signal a stable upward trend.
    2. **Low Volatility** (0–30 pts): directly rewards low annualised vol;
       large-cap defensives with < 12 % vol earn full points.
    3. **Liquidity** (0–20 pts): log-scaled 20-day average daily volume,
       favouring the highly liquid large-caps appropriate for buy-and-hold.
    4. **Trend Protection** (0–20 pts): rewards stocks near recent highs
       and penalises those in a confirmed downtrend below their 20-day MA.

    Parameters
    ----------
    price_history:
        OHLCV DataFrame from ``get_price_history`` (yfinance format).

    Returns
    -------
    :class:`StockScoreComponents`
        ``relative_strength`` is always ``0.0`` for the stability model.
    """
    _zero = StockScoreComponents(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Insufficient price history")
    if price_history is None or len(price_history) < _MIN_HISTORY_DAYS:
        return _zero

    close = price_history["Close"].astype(float)
    volume = price_history["Volume"].astype(float)
    current_price = float(close.iloc[-1])
    if current_price <= 0:
        return _zero

    # ── 1. Consistency / Trend (0–30 pts) ────────────────────────────────────
    ma20 = float(close.tail(20).mean())
    above_20ma = current_price > ma20
    consistency_pts = 10.0 if above_20ma else 0.0

    above_50ma = False
    if len(close) >= 50:
        ma50 = float(close.tail(50).mean())
        above_50ma = current_price > ma50
        consistency_pts += 10.0 if above_50ma else 0.0

    # Positive 20-day return (0–10 pts)
    if len(close) >= 21:
        ret_20d = current_price / float(close.iloc[-21]) - 1.0
        consistency_pts += min(max(ret_20d / 0.05 * 5.0, 0.0), 10.0)

    consistency_pts = min(consistency_pts, 30.0)

    # ── 2. Low Volatility (0–30 pts) ─────────────────────────────────────────
    daily_returns = close.pct_change().dropna()
    ann_vol = 0.0
    low_vol_pts = 15.0  # neutral default
    if len(daily_returns) >= 10:
        ann_vol = float(daily_returns.tail(20).std()) * math.sqrt(252.0)
        if ann_vol <= 0.12:
            low_vol_pts = 30.0
        elif ann_vol <= 0.20:
            low_vol_pts = 30.0 - (ann_vol - 0.12) / 0.08 * 10.0
        elif ann_vol <= 0.30:
            low_vol_pts = 20.0 - (ann_vol - 0.20) / 0.10 * 10.0
        elif ann_vol <= 0.50:
            low_vol_pts = 10.0 - (ann_vol - 0.30) / 0.20 * 10.0
        else:
            low_vol_pts = 0.0
        low_vol_pts = max(low_vol_pts, 0.0)

    # ── 3. Liquidity (0–20 pts) ───────────────────────────────────────────────
    avg_vol_20d = float(volume.tail(20).mean())
    liq_pts = min(max(math.log10(max(avg_vol_20d, 1.0)) - 4.0, 0.0) * 6.67, 20.0)

    # ── 4. Trend Protection (0–20 pts) ────────────────────────────────────────
    high_20d = float(close.tail(20).max())
    pct_from_high = (high_20d - current_price) / high_20d if high_20d > 0 else 0.0

    if pct_from_high <= 0.05:
        trend_prot_pts = 20.0
    elif pct_from_high <= 0.15:
        trend_prot_pts = 20.0 - (pct_from_high - 0.05) / 0.10 * 10.0
    else:
        trend_prot_pts = 10.0 - (pct_from_high - 0.15) / 0.15 * 10.0

    if not above_20ma:
        trend_prot_pts = max(trend_prot_pts - 5.0, 0.0)   # downtrend penalty

    trend_prot_pts = max(min(trend_prot_pts, 20.0), 0.0)

    # ── Composite ─────────────────────────────────────────────────────────────
    composite = consistency_pts + low_vol_pts + liq_pts + trend_prot_pts

    # ── Long-term thesis ──────────────────────────────────────────────────────
    reasons: List[str] = []
    if above_50ma:
        reasons.append("above 50-day MA (strong trend)")
    elif above_20ma:
        reasons.append("above 20-day MA")
    else:
        reasons.append("below 20-day MA (caution)")
    if ann_vol > 0 and ann_vol <= 0.20:
        reasons.append(f"low volatility ({ann_vol:.0%})")
    elif ann_vol > 0 and ann_vol <= 0.35:
        reasons.append(f"moderate volatility ({ann_vol:.0%})")
    elif ann_vol > 0:
        reasons.append(f"elevated volatility ({ann_vol:.0%})")
    if liq_pts >= 13.0:
        reasons.append("highly liquid large-cap")
    elif liq_pts >= 6.0:
        reasons.append("adequate liquidity")
    if pct_from_high <= 0.10:
        reasons.append("near recent highs (strength)")
    long_term_thesis = "; ".join(reasons) if reasons else "stable large-cap holding"

    return StockScoreComponents(
        trend_strength=round(consistency_pts, 2),
        relative_strength=0.0,        # not used for RRSP
        volatility_control=round(low_vol_pts, 2),
        liquidity=round(liq_pts, 2),
        drawdown_risk=round(trend_prot_pts, 2),
        composite=round(composite, 2),
        reasoning=long_term_thesis,
    )

