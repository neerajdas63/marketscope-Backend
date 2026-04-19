# trade_planner.py — Intraday Trade Planner (institutional-grade, both directions)
#
# Strategies generated:
#   LONG_BREAKOUT   — near day high, strong volume/rfactor, above VWAP
#   LONG_PULLBACK   — uptrend pulled back to VWAP support zone
#   SHORT_BREAKDOWN — near day low, bearish momentum, below VWAP
#   SHORT_PULLBACK  — downtrend bounced to VWAP resistance zone
#   RANGE_LONG      — at VWAP -1σ band support in range-bound stock
#   RANGE_SHORT     — at VWAP +1σ band resistance in range-bound stock
#
# Plans are dropped (not returned) when:
#   - ltp / day_high / day_low / vwap are 0 or missing
#   - rfactor < 2.0 (insufficient momentum conviction)
#   - abs(change_pct) < 0.25 (stock is flat — no intraday edge)

import logging
from typing import Any, Dict, List

logger = logging.getLogger("trade_planner")

_CONFIDENCE_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

# Thresholds
_MIN_RFACTOR    = 2.0   # below this: not enough conviction
_MIN_CHANGE_PCT = 0.25  # abs% below this: stock is flat, no trade
_NEAR_EXTREME   = 0.8   # within 0.8% of day high/low = breakout/breakdown zone


def get_trade_plan(symbol: str, stock: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an institutional intraday trade plan (LONG or SHORT) for a stock.
    Returns empty dict when data is invalid or no clean setup exists.
    """
    try:
        ltp          = float(stock.get("ltp",           0) or 0)
        vwap         = float(stock.get("vwap",          0) or 0)
        day_high     = float(stock.get("day_high",      0) or 0)
        day_low      = float(stock.get("day_low",       0) or 0)
        change_pct   = float(stock.get("change_pct",    0) or 0)
        rfactor      = float(stock.get("rfactor",       0) or 0)
        boost        = float(stock.get("boost_score",   0) or 0)
        delivery_pct = float(stock.get("delivery_pct",  0) or 0)
        rsi          = float(stock.get("rsi",          50) or 50)

        # VWAP band fields from vwap_bands.py
        band_1_upper = float(stock.get("band_1_upper", 0) or 0)
        band_1_lower = float(stock.get("band_1_lower", 0) or 0)
        band_2_upper = float(stock.get("band_2_upper", 0) or 0)
        band_2_lower = float(stock.get("band_2_lower", 0) or 0)

        # ── Hard data quality gate ───────────────────────────────────────────
        if ltp <= 0 or day_high <= 0 or day_low <= 0 or vwap <= 0:
            return {}
        if day_high <= day_low:
            return {}

        # ── Momentum / activity gate ─────────────────────────────────────────
        if rfactor < _MIN_RFACTOR or abs(change_pct) < _MIN_CHANGE_PCT:
            return {}

        # ── ATR proxy (half day range) ────────────────────────────────────────
        atr = (day_high - day_low) * 0.5
        if atr <= 0:
            atr = ltp * 0.01

        # ── Fill band defaults if vwap_bands.py hasn't run ───────────────────
        if band_1_upper <= 0:
            std_dev      = (day_high - day_low) / 4.0
            band_1_upper = round(vwap + std_dev,     2)
            band_1_lower = round(vwap - std_dev,     2)
            band_2_upper = round(vwap + 2 * std_dev, 2)
            band_2_lower = round(vwap - 2 * std_dev, 2)

        reasons: List[str] = []

        # ── Trend bias ────────────────────────────────────────────────────────
        if ltp >= vwap and change_pct > 0:
            trend_bias = "BULLISH"
        elif ltp <= vwap and change_pct < 0:
            trend_bias = "BEARISH"
        else:
            trend_bias = "NEUTRAL"

        # ── Position within day range ─────────────────────────────────────────
        pct_from_high = (day_high - ltp) / day_high * 100
        pct_from_low  = (ltp - day_low)  / day_low  * 100
        near_high     = pct_from_high <= _NEAR_EXTREME
        near_low      = pct_from_low  <= _NEAR_EXTREME
        bullish       = trend_bias == "BULLISH"
        bearish       = trend_bias == "BEARISH"

        # ── Strategy selection ────────────────────────────────────────────────
        strategy  = None
        direction = None

        # LONG BREAKOUT: near day high + bullish + strong rfactor
        if near_high and bullish and rfactor >= 3.0:
            strategy  = "LONG_BREAKOUT"
            direction = "LONG"
            reasons.append(f"Near day high ({pct_from_high:.2f}% away)")
            if rsi > 60:
                reasons.append(f"RSI {rsi:.0f} — bullish momentum")

        # SHORT BREAKDOWN: near day low + bearish + strong rfactor
        elif near_low and bearish and rfactor >= 3.0:
            strategy  = "SHORT_BREAKDOWN"
            direction = "SHORT"
            reasons.append(f"Near day low ({pct_from_low:.2f}% away)")
            if rsi < 40:
                reasons.append(f"RSI {rsi:.0f} — bearish momentum")

        # LONG PULLBACK: bullish trend, pulled back to VWAP ±1σ zone
        elif bullish and pct_from_high > 0.8 and band_1_lower <= ltp <= band_1_upper:
            strategy  = "LONG_PULLBACK"
            direction = "LONG"
            reasons.append("Bullish trend — VWAP support pullback")
            if rsi > 50:
                reasons.append(f"RSI {rsi:.0f} still bullish")

        # SHORT PULLBACK: bearish trend, bounced back to VWAP ±1σ zone
        elif bearish and pct_from_low > 0.8 and band_1_lower <= ltp <= band_1_upper:
            strategy  = "SHORT_PULLBACK"
            direction = "SHORT"
            reasons.append("Bearish trend — VWAP resistance bounce")
            if rsi < 50:
                reasons.append(f"RSI {rsi:.0f} still bearish")

        # RANGE LONG: at VWAP -1σ band (buy the dip in a sideways market)
        elif ltp <= band_1_lower and ltp >= band_2_lower and rfactor >= 2.5:
            strategy  = "RANGE_LONG"
            direction = "LONG"
            reasons.append("At VWAP -1σ band support")

        # RANGE SHORT: at VWAP +1σ band (sell the rip in a sideways market)
        elif ltp >= band_1_upper and ltp <= band_2_upper and rfactor >= 2.5:
            strategy  = "RANGE_SHORT"
            direction = "SHORT"
            reasons.append("At VWAP +1σ band resistance")

        if strategy is None:
            return {}

        # ── Entry / Target / Stop-loss ────────────────────────────────────────
        if strategy == "LONG_BREAKOUT":
            entry_low  = round(day_high,           2)
            entry_high = round(day_high * 1.002,   2)
            target_1   = round(day_high + atr,      2)
            target_2   = round(day_high + 2 * atr,  2)
            stop_loss  = round(day_high - atr * 0.5, 2)

        elif strategy == "SHORT_BREAKDOWN":
            entry_high = round(day_low,            2)
            entry_low  = round(day_low  * 0.998,   2)
            target_1   = round(day_low  - atr,      2)
            target_2   = round(day_low  - 2 * atr,  2)
            stop_loss  = round(day_low  + atr * 0.5, 2)

        elif strategy == "LONG_PULLBACK":
            entry_low  = round(vwap * 0.997,       2)
            entry_high = round(vwap * 1.003,       2)
            target_1   = round(day_high,            2)
            target_2   = round(day_high + atr * 0.5, 2)
            stop_loss  = round(vwap - atr * 0.6,   2)

        elif strategy == "SHORT_PULLBACK":
            entry_low  = round(vwap * 0.997,       2)
            entry_high = round(vwap * 1.003,       2)
            target_1   = round(day_low,             2)
            target_2   = round(day_low  - atr * 0.5, 2)
            stop_loss  = round(vwap + atr * 0.6,   2)

        elif strategy == "RANGE_LONG":
            entry_low  = round(band_1_lower * 0.999, 2)
            entry_high = round(band_1_lower * 1.001, 2)
            target_1   = round(vwap,                 2)
            target_2   = round(band_1_upper,         2)
            stop_loss  = round(band_2_lower,         2)

        else:  # RANGE_SHORT
            entry_low  = round(band_1_upper * 0.999, 2)
            entry_high = round(band_1_upper * 1.001, 2)
            target_1   = round(vwap,                 2)
            target_2   = round(band_1_lower,         2)
            stop_loss  = round(band_2_upper,         2)

        # ── Risk / Reward ─────────────────────────────────────────────────────
        ref_entry = entry_low if direction == "LONG" else entry_high
        risk      = abs(ref_entry - stop_loss)
        reward    = abs(target_1  - ref_entry)
        rr        = round(reward / risk, 2) if risk > 0 else 0.0

        # ── Confidence ────────────────────────────────────────────────────────
        if rfactor >= 3.5 and boost > 3 and rr >= 2.0:
            confidence = "HIGH"
            reasons.append("Strong RFactor + Boost + R:R≥2")
        elif rfactor >= 2.5 and rr >= 1.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        if delivery_pct > 40:
            reasons.append(f"Delivery {delivery_pct:.0f}% — institutional conviction")
        elif delivery_pct > 0:
            reasons.append(f"Delivery {delivery_pct:.0f}%")
        if boost > 3:
            reasons.append(f"Boost score {boost:.1f}")

        return {
            "symbol":             symbol.upper(),
            "direction":          direction,           # "LONG" or "SHORT"
            "strategy":           strategy,
            "trend_bias":         trend_bias,
            "entry_zone":         [entry_low, entry_high],
            "entry_zone_low":     entry_low,
            "entry_zone_high":    entry_high,
            "target_1":           target_1,
            "target_2":           target_2,
            "stop_loss":          stop_loss,
            "risk_reward":        rr,
            "opening_range_high": round(day_high, 2),
            "opening_range_low":  round(day_low,  2),
            "vwap":               round(vwap,     2),
            "confidence":         confidence,
            "reasons":            reasons,
            "rfactor":            round(rfactor,  2),
            "boost_score":        round(boost,    2),
            "rsi":                round(rsi,      1),
            "ltp":                round(ltp,      2),
            "change_pct":         round(change_pct, 2),
            "delivery_pct":       round(delivery_pct, 1) if delivery_pct > 0 else None,
        }

    except Exception as e:
        logger.error("get_trade_plan failed for %s: %s", symbol, e, exc_info=True)
        return {}


def get_bulk_trade_plans(sym_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate trade plans for all F&O stocks in sym_data.
    Returns both LONG and SHORT plans sorted by confidence → R:R → rfactor.
    """
    from stocks import ACTIVE_FO_STOCK_SET

    plans: List[Dict[str, Any]] = []
    for sym, stock in sym_data.items():
        if sym not in ACTIVE_FO_STOCK_SET:
            continue
        plan = get_trade_plan(sym, stock)
        if plan:
            plans.append(plan)

    plans.sort(
        key=lambda p: (
            _CONFIDENCE_ORDER.get(p.get("confidence", "LOW"), 2),
            -p.get("risk_reward", 0),
            -p.get("rfactor", 0),
        )
    )
    return plans
