from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _split_sessions(df: pd.DataFrame) -> List[pd.DataFrame]:
    if df is None or df.empty:
        return []
    clean = df.sort_index()
    return [clean[clean.index.date == session_date].copy() for session_date in sorted(set(clean.index.date))]


def _session_vwap(session_df: pd.DataFrame) -> float:
    if session_df is None or session_df.empty:
        return 0.0
    typical = (session_df["High"] + session_df["Low"] + session_df["Close"]) / 3.0
    volume = session_df["Volume"].fillna(0)
    total_volume = _safe_float(volume.sum())
    if total_volume <= 0:
        return _safe_float(session_df["Close"].iloc[-1])
    return _safe_float((typical * volume).sum() / total_volume)


def get_nifty_market_state(nifty_5m_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze Nifty's real-time market state from 5-min bars.
    Returns market context that filters individual stock trades.
    """
    default = {
        "nifty_session_change_pct": 0.0,
        "nifty_or_high": 0.0,
        "nifty_or_low": 0.0,
        "nifty_or_break_side": "INSIDE",
        "nifty_trend_direction": "CHOPPY",
        "nifty_bar_consistency": "mixed",
        "nifty_bar_consistency_ratio": 0.5,
        "nifty_vwap_position": "FLAT",
        "market_mode": "CHOPPY",
        "tradeable_side": "SKIP",
        "market_confidence_score": 20.0,
        "skip_reason": "Nifty data unavailable - no clear direction",
    }
    sessions = _split_sessions(nifty_5m_df)
    if not sessions:
        return default

    current_session = sessions[-1]
    if current_session.empty:
        return default
    prev_day_close = _safe_float(sessions[-2]["Close"].dropna().iloc[-1]) if len(sessions) >= 2 else _safe_float(current_session["Open"].iloc[0])
    current_close = _safe_float(current_session["Close"].dropna().iloc[-1])
    session_change_pct = round(((current_close - prev_day_close) / prev_day_close) * 100.0, 2) if prev_day_close > 0 else 0.0

    opening_bars = current_session.head(3)
    nifty_or_high = _safe_float(opening_bars["High"].max()) if not opening_bars.empty else 0.0
    nifty_or_low = _safe_float(opening_bars["Low"].min()) if not opening_bars.empty else 0.0
    if nifty_or_high > 0 and current_close > nifty_or_high:
        or_break_side = "LONG"
    elif nifty_or_low > 0 and current_close < nifty_or_low:
        or_break_side = "SHORT"
    else:
        or_break_side = "INSIDE"

    last_6 = current_session.tail(6)
    close_up_count = 0
    close_down_count = 0
    closes = list(last_6["Close"])
    for idx in range(1, len(closes)):
        if _safe_float(closes[idx]) > _safe_float(closes[idx - 1]):
            close_up_count += 1
        elif _safe_float(closes[idx]) < _safe_float(closes[idx - 1]):
            close_down_count += 1
    if close_up_count >= 4:
        trend_direction = "UPTREND"
    elif close_down_count >= 4:
        trend_direction = "DOWNTREND"
    else:
        trend_direction = "CHOPPY"

    bullish_count = int((last_6["Close"] > last_6["Open"]).sum()) if not last_6.empty else 0
    consistency_ratio = bullish_count / max(len(last_6), 1)
    if consistency_ratio > 0.65:
        bar_consistency = "strong_bull"
    elif consistency_ratio < 0.35:
        bar_consistency = "strong_bear"
    else:
        bar_consistency = "mixed"

    vwap = _session_vwap(current_session)
    vwap_position = "ABOVE" if current_close > vwap else "BELOW" if current_close < vwap else "FLAT"

    was_down = session_change_pct < -0.3
    was_up = session_change_pct > 0.3
    recent_recovering = len(last_6) >= 2 and _safe_float(last_6["Close"].iloc[-1]) > _safe_float(last_6["Close"].iloc[0])
    recent_fading = len(last_6) >= 2 and _safe_float(last_6["Close"].iloc[-1]) < _safe_float(last_6["Close"].iloc[0])

    if or_break_side == "LONG" and trend_direction == "UPTREND" and bar_consistency == "strong_bull":
        market_mode = "TRENDING_UP"
        tradeable_side = "LONG_ONLY"
    elif or_break_side == "SHORT" and trend_direction == "DOWNTREND" and bar_consistency == "strong_bear":
        market_mode = "TRENDING_DOWN"
        tradeable_side = "SHORT_ONLY"
    elif was_down and recent_recovering and vwap_position == "ABOVE":
        market_mode = "RECOVERING"
        tradeable_side = "BOTH"
    elif was_up and recent_fading and vwap_position == "BELOW":
        market_mode = "FADING"
        tradeable_side = "BOTH"
    else:
        market_mode = "CHOPPY"
        tradeable_side = "SKIP"

    bullish_aligned = [
        or_break_side == "LONG",
        trend_direction == "UPTREND",
        bar_consistency == "strong_bull",
        vwap_position == "ABOVE",
    ]
    bearish_aligned = [
        or_break_side == "SHORT",
        trend_direction == "DOWNTREND",
        bar_consistency == "strong_bear",
        vwap_position == "BELOW",
    ]
    aligned_count = max(sum(bullish_aligned), sum(bearish_aligned))
    confidence_by_count = {4: 90.0, 3: 70.0, 2: 45.0, 1: 22.0, 0: 15.0}
    confidence = confidence_by_count.get(aligned_count, 20.0)
    if abs(session_change_pct) > 0.5 and aligned_count >= 3:
        confidence = min(100.0, confidence + 8.0)

    skip_reason = ""
    if tradeable_side == "SKIP":
        if or_break_side == "INSIDE":
            skip_reason = "Nifty inside opening range - no clear direction"
        elif trend_direction == "CHOPPY" or bar_consistency == "mixed":
            skip_reason = "Nifty choppy - mixed bullish/bearish bars"
        else:
            skip_reason = "Nifty signals are not aligned"

    return {
        "nifty_session_change_pct": session_change_pct,
        "nifty_or_high": round(nifty_or_high, 2),
        "nifty_or_low": round(nifty_or_low, 2),
        "nifty_or_break_side": or_break_side,
        "nifty_trend_direction": trend_direction,
        "nifty_bar_consistency": bar_consistency,
        "nifty_bar_consistency_ratio": round(consistency_ratio, 2),
        "nifty_vwap_position": vwap_position,
        "market_mode": market_mode,
        "tradeable_side": tradeable_side,
        "market_confidence_score": round(confidence, 1),
        "skip_reason": skip_reason,
    }


def get_market_breadth(all_stock_results: List[Dict]) -> Dict[str, Any]:
    """
    Quick breadth analysis from momentum pulse results.
    More stocks going up than down = bullish breadth.
    """
    total = len(all_stock_results or [])
    long_count = sum(1 for row in (all_stock_results or []) if str(row.get("direction")).upper() == "LONG")
    short_count = sum(1 for row in (all_stock_results or []) if str(row.get("direction")).upper() == "SHORT")
    long_pct = (long_count / total * 100.0) if total else 0.0
    short_pct = (short_count / total * 100.0) if total else 0.0
    strong_long_count = sum(
        1 for row in (all_stock_results or [])
        if str(row.get("direction")).upper() == "LONG" and _safe_float(row.get("momentum_pulse_score")) >= 65
    )
    strong_short_count = sum(
        1 for row in (all_stock_results or [])
        if str(row.get("direction")).upper() == "SHORT" and _safe_float(row.get("momentum_pulse_score")) >= 65
    )
    if long_pct >= 60.0 and strong_long_count >= 15:
        breadth_signal = "BULLISH"
        breadth_score = 70.0 + min(strong_long_count, 30)
    elif short_pct >= 60.0 and strong_short_count >= 15:
        breadth_signal = "BEARISH"
        breadth_score = 70.0 + min(strong_short_count, 30)
    else:
        breadth_signal = "NEUTRAL"
        breadth_score = 50.0
    return {
        "total_stocks": total,
        "long_count": long_count,
        "short_count": short_count,
        "long_pct": round(long_pct, 1),
        "short_pct": round(short_pct, 1),
        "strong_long_count": strong_long_count,
        "strong_short_count": strong_short_count,
        "breadth_signal": breadth_signal,
        "breadth_score": round(breadth_score, 1),
    }


def get_combined_market_filter(
    nifty_state: Dict,
    breadth: Dict,
    nifty_change_pct: float,
) -> Dict[str, Any]:
    """
    Combine nifty state + breadth to give final trade filter.
    """
    tradeable_side = str((nifty_state or {}).get("tradeable_side", "BOTH")).upper()
    confidence = _safe_float((nifty_state or {}).get("market_confidence_score"), 50.0)
    market_mode = str((nifty_state or {}).get("market_mode", "CHOPPY")).upper()
    breadth_signal = str((breadth or {}).get("breadth_signal", "NEUTRAL")).upper()
    reason = str((nifty_state or {}).get("skip_reason") or "")

    if confidence >= 75:
        quality_multiplier = 1.15
    elif confidence >= 50:
        quality_multiplier = 1.0
    else:
        quality_multiplier = 0.85

    if -0.3 <= _safe_float(nifty_change_pct) <= 0.3:
        return {
            "allow_long": False,
            "allow_short": False,
            "skip_day": True,
            "preferred_side": "NONE",
            "reason": "Nifty flat - no clear directional edge",
            "confidence": min(confidence, 35.0),
            "quality_multiplier": 0.85,
            "min_rs_for_long": 2.0,
            "min_rs_for_short": -2.0,
            "market_mode": "CHOPPY",
            "breadth_signal": breadth_signal,
        }

    if tradeable_side == "SKIP":
        return {
            "allow_long": False,
            "allow_short": False,
            "skip_day": True,
            "preferred_side": "NONE",
            "reason": reason or "Market choppy today - no clear direction",
            "confidence": confidence,
            "quality_multiplier": quality_multiplier,
            "min_rs_for_long": 2.0,
            "min_rs_for_short": -2.0,
            "market_mode": market_mode,
            "breadth_signal": breadth_signal,
        }

    if tradeable_side == "LONG_ONLY":
        allow_long = True
        allow_short = False
        preferred_side = "LONG"
        min_rs_for_long = 0.5
        min_rs_for_short = -2.0
    elif tradeable_side == "SHORT_ONLY":
        allow_long = False
        allow_short = True
        preferred_side = "SHORT"
        min_rs_for_long = 2.0
        min_rs_for_short = -0.5
    else:
        allow_long = True
        allow_short = True
        min_rs_for_long = 2.0 if breadth_signal == "BEARISH" else 0.5
        min_rs_for_short = -2.0 if breadth_signal == "BULLISH" else -0.5
        if breadth_signal == "BULLISH":
            preferred_side = "LONG"
        elif breadth_signal == "BEARISH":
            preferred_side = "SHORT"
        else:
            preferred_side = "BOTH"
            quality_multiplier = min(quality_multiplier, 1.0)

    return {
        "allow_long": allow_long,
        "allow_short": allow_short,
        "skip_day": False,
        "preferred_side": preferred_side,
        "reason": reason or f"Market mode {market_mode}, breadth {breadth_signal}",
        "confidence": confidence,
        "quality_multiplier": quality_multiplier,
        "min_rs_for_long": min_rs_for_long,
        "min_rs_for_short": min_rs_for_short,
        "market_mode": market_mode,
        "breadth_signal": breadth_signal,
    }
