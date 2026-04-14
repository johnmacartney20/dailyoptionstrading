"""Options analytics module.

Pure-function helpers for computing metrics used to rank and filter options:
days-to-expiration, out-of-the-money percentage, annualized return on capital,
risk-adjusted return, suggested spread structure, and a composite score that
prioritises high-probability, executable trades over raw annualised return.
"""

import math
from datetime import date, datetime

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
    """Return a human-readable call debit spread string (long the lower strike).

    This is the TFSA-appropriate structure: buy the call at *strike*, sell a
    higher-strike call to cap cost and define maximum risk.

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
    TFSA where short premium is not permitted.  Components:

    1. **Upside ratio** (0–35 pts): ``(spread_width − ask) / ask`` for a call
       debit spread.  A higher ratio means more potential profit per dollar
       risked.
    2. **OTM sweet-spot** (0–30 pts): rewards strikes 3–10 % OTM, where call
       debit spreads offer the best leverage-vs-probability balance.
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

