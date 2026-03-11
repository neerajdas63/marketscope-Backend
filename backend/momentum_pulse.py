import logging
import math
import threading
import time
from datetime import date, datetime, time as dt_time
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from nse_fetcher import fetch_nse_index_quotes

logger = logging.getLogger("momentum_pulse")

IST = pytz.timezone("Asia/Kolkata")
LOOKBACK_SESSIONS = 20
MAX_HISTORY_POINTS = 24
REFRESH_COOLDOWN_SECONDS = 240
MIN_SESSION_BARS = 6

_pulse_cache: Dict[str, Any] = {
    "source_key": "",
    "computed_at": 0.0,
    "last_updated": "",
    "benchmark_change_pct": 0.0,
    "results": [],
    "is_loading": False,
    "last_attempt": 0.0,
    "error": "",
}
_score_state: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _now_ist_str() -> str:
    return datetime.now(IST).strftime("%H:%M:%S")


def _build_source_key(last_updated: str, scanner_stocks: Sequence[Dict[str, Any]]) -> str:
    today_key = datetime.now(IST).date().isoformat()
    return f"{today_key}|{last_updated}|{len(scanner_stocks)}"


def _snapshot_cache() -> Dict[str, Any]:
    with _lock:
        return {
            "source_key": str(_pulse_cache.get("source_key") or ""),
            "computed_at": _safe_float(_pulse_cache.get("computed_at")),
            "last_updated": str(_pulse_cache.get("last_updated") or ""),
            "benchmark_change_pct": _safe_float(_pulse_cache.get("benchmark_change_pct")),
            "results": list(_pulse_cache.get("results") or []),
            "is_loading": bool(_pulse_cache.get("is_loading")),
            "last_attempt": _safe_float(_pulse_cache.get("last_attempt")),
            "error": str(_pulse_cache.get("error") or ""),
        }


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def normalize_score(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return round(100.0 * _clamp((value - lower) / (upper - lower)), 1)


def get_tier(score: float) -> str:
    if score >= 75:
        return "strong"
    if score >= 60:
        return "moderate"
    if score >= 45:
        return "weak"
    return "veryweak"


def _score_from_anchors(value: float, anchors: Sequence[Tuple[float, float]]) -> float:
    if not anchors:
        return 0.0
    ordered = sorted((float(x), float(y)) for x, y in anchors)
    if value <= ordered[0][0]:
        return round(ordered[0][1], 1)
    for left, right in zip(ordered, ordered[1:]):
        x1, y1 = left
        x2, y2 = right
        if value <= x2:
            ratio = (value - x1) / max(x2 - x1, 1e-6)
            return round(y1 + (y2 - y1) * ratio, 1)
    return round(ordered[-1][1], 1)


def _normalize_intraday_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    try:
        if df is None or df.empty:
            return None
        clean = df.copy()
        clean = clean.sort_index()
        if clean.index.tz is not None:
            clean.index = clean.index.tz_convert(IST).tz_localize(None)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        for column in needed:
            if column not in clean.columns:
                return None
        clean = clean[needed].dropna(how="all")
        clean = clean.between_time("09:15", "15:30")
        return clean if not clean.empty else None
    except Exception:
        return None


def _get_sym_df(raw: Any, symbol: str) -> Optional[pd.DataFrame]:
    try:
        if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol not in raw.columns.get_level_values(0):
                return None
            return _normalize_intraday_df(raw[symbol])
        return _normalize_intraday_df(raw)
    except Exception:
        return None


def _split_sessions(df: Optional[pd.DataFrame]) -> List[Tuple[date, pd.DataFrame]]:
    if df is None or df.empty:
        return []
    sessions: List[Tuple[date, pd.DataFrame]] = []
    for session_date in sorted(set(df.index.date)):
        session_df = df[df.index.date == session_date].copy()
        if len(session_df) >= 3:
            sessions.append((session_date, session_df))
    return sessions


def _current_cutoff(session_df: pd.DataFrame) -> dt_time:
    return session_df.index[-1].time()


def _same_time_slice(session_df: pd.DataFrame, cutoff: dt_time) -> pd.DataFrame:
    return session_df[(session_df.index.time <= cutoff)].copy()


def _infer_prev_close(
    sessions: Sequence[Tuple[date, pd.DataFrame]],
    session_index: int,
    fallback: float,
) -> float:
    if session_index > 0:
        prev_df = sessions[session_index - 1][1]
        if prev_df is not None and not prev_df.empty:
            return _safe_float(prev_df["Close"].dropna().iloc[-1], fallback)
    return fallback


def calculate_same_time_cum_volume_baseline(
    session_df: pd.DataFrame,
    historical_sessions: Sequence[Tuple[date, pd.DataFrame]],
    cutoff: dt_time,
) -> Tuple[float, float, float, float]:
    today_slice = _same_time_slice(session_df, cutoff)
    today_cum_volume = _safe_float(pd.Series(today_slice["Volume"], dtype="float64").fillna(0.0).sum())
    baseline_values: List[float] = []
    for _, hist_df in historical_sessions[-LOOKBACK_SESSIONS:]:
        hist_slice = _same_time_slice(hist_df, cutoff)
        if hist_slice.empty:
            continue
        baseline_values.append(_safe_float(pd.Series(hist_slice["Volume"], dtype="float64").fillna(0.0).sum()))
    avg_same_time = round(mean(baseline_values), 1) if baseline_values else 0.0
    ratio = round(today_cum_volume / avg_same_time, 2) if avg_same_time > 0 else 0.0
    score = _score_from_anchors(
        ratio,
        [
            (0.6, 10.0),
            (1.0, 35.0),
            (1.5, 60.0),
            (2.0, 82.0),
            (3.0, 100.0),
        ],
    )
    return round(today_cum_volume, 1), avg_same_time, ratio, score


def calculate_same_time_range_baseline(
    session_df: pd.DataFrame,
    historical_sessions: Sequence[Tuple[date, pd.DataFrame]],
    cutoff: dt_time,
    current_prev_close: float,
) -> Tuple[float, float, float, float, float, float]:
    current_slice = _same_time_slice(session_df, cutoff)
    intraday_range_abs = _safe_float(current_slice["High"].max()) - _safe_float(current_slice["Low"].min())
    intraday_range_pct = round((intraday_range_abs / max(current_prev_close, 1e-6)) * 100.0, 2) if current_prev_close > 0 else 0.0

    hist_abs_values: List[float] = []
    hist_pct_values: List[float] = []
    hist_sessions = list(historical_sessions[-LOOKBACK_SESSIONS:])
    for index, (_, hist_df) in enumerate(hist_sessions):
        hist_slice = _same_time_slice(hist_df, cutoff)
        if hist_slice.empty:
            continue
        hist_range_abs = _safe_float(hist_slice["High"].max()) - _safe_float(hist_slice["Low"].min())
        prev_close = _infer_prev_close(hist_sessions, index, _safe_float(hist_df["Open"].iloc[0]))
        hist_range_pct = (hist_range_abs / max(prev_close, 1e-6)) * 100.0 if prev_close > 0 else 0.0
        hist_abs_values.append(hist_range_abs)
        hist_pct_values.append(hist_range_pct)

    avg_abs = round(mean(hist_abs_values), 2) if hist_abs_values else 0.0
    avg_pct = round(mean(hist_pct_values), 2) if hist_pct_values else 0.0
    ratio = round(intraday_range_abs / avg_abs, 2) if avg_abs > 0 else 0.0
    score = _score_from_anchors(
        ratio,
        [
            (0.7, 10.0),
            (1.0, 30.0),
            (1.3, 50.0),
            (1.8, 78.0),
            (2.5, 100.0),
        ],
    )
    return (
        round(intraday_range_abs, 2),
        intraday_range_pct,
        avg_abs,
        avg_pct,
        ratio,
        score,
    )


def calculate_relative_strength_scores(change_pct: float, nifty_change_pct: float) -> Tuple[float, float, float]:
    relative_strength = round(change_pct - nifty_change_pct, 2)
    long_score = _score_from_anchors(
        relative_strength,
        [(-1.5, 0.0), (-0.2, 20.0), (0.4, 40.0), (1.2, 68.0), (2.5, 100.0)],
    )
    short_score = _score_from_anchors(
        -relative_strength,
        [(-1.5, 0.0), (-0.2, 20.0), (0.4, 40.0), (1.2, 68.0), (2.5, 100.0)],
    )
    return relative_strength, long_score, short_score


def calculate_directional_consistency(session_df: pd.DataFrame, lookback: int = 8) -> Tuple[float, float, bool]:
    recent = session_df.tail(min(lookback, len(session_df))).copy()
    if recent.empty:
        return 0.0, 0.0, False

    close = pd.Series(recent["Close"], dtype="float64")
    open_ = pd.Series(recent["Open"], dtype="float64")
    high = pd.Series(recent["High"], dtype="float64")
    low = pd.Series(recent["Low"], dtype="float64")
    close_diff = close.diff().dropna()
    candle_range = (high - low).replace(0.0, np.nan)
    close_pos = (close - low) / candle_range

    bullish_ratio = float((close > open_).mean()) if len(close) else 0.0
    bearish_ratio = float((close < open_).mean()) if len(close) else 0.0
    higher_close_ratio = float((close_diff > 0).mean()) if not close_diff.empty else 0.0
    lower_close_ratio = float((close_diff < 0).mean()) if not close_diff.empty else 0.0
    upper_half_ratio = float((close_pos >= 0.55).mean()) if len(close_pos.dropna()) else 0.0
    lower_half_ratio = float((close_pos <= 0.45).mean()) if len(close_pos.dropna()) else 0.0

    long_score = round(
        bullish_ratio * 40.0 + higher_close_ratio * 35.0 + upper_half_ratio * 25.0,
        1,
    )
    short_score = round(
        bearish_ratio * 40.0 + lower_close_ratio * 35.0 + lower_half_ratio * 25.0,
        1,
    )

    session_range = _safe_float(high.max()) - _safe_float(low.min())
    latest_range = _safe_float(high.iloc[-1]) - _safe_float(low.iloc[-1]) if len(high) else 0.0
    one_bar_spike = session_range > 0 and latest_range / session_range >= 0.42
    return long_score, short_score, one_bar_spike


def calculate_vwap_alignment(
    session_df: pd.DataFrame,
    live_price: float,
    live_vwap: float,
) -> Tuple[float, float, float, float, bool]:
    volume = pd.Series(session_df["Volume"], dtype="float64").fillna(0.0)
    typical_price = (
        pd.Series(session_df["High"], dtype="float64")
        + pd.Series(session_df["Low"], dtype="float64")
        + pd.Series(session_df["Close"], dtype="float64")
    ) / 3.0
    cum_volume = volume.cumsum().replace(0.0, np.nan)
    vwap_series = ((typical_price * volume).cumsum() / cum_volume).dropna()
    vwap = live_vwap if live_vwap > 0 else _safe_float(vwap_series.iloc[-1])
    if vwap <= 0:
        return 0.0, 0.0, 0.0, False

    distance_pct = round(((live_price - vwap) / vwap) * 100.0, 2) if live_price > 0 else 0.0
    abs_distance = abs(distance_pct)
    is_extended = abs_distance >= 1.4

    stretch_penalty = _score_from_anchors(abs_distance, [(0.2, 0.0), (0.8, 6.0), (1.4, 14.0), (2.4, 28.0)])
    long_base = 76.0 if distance_pct >= 0 else 34.0
    short_base = 76.0 if distance_pct <= 0 else 34.0
    long_score = round(max(0.0, min(100.0, long_base - (stretch_penalty if distance_pct > 0 else stretch_penalty * 0.35))), 1)
    short_score = round(max(0.0, min(100.0, short_base - (stretch_penalty if distance_pct < 0 else stretch_penalty * 0.35))), 1)
    return round(vwap, 2), distance_pct, long_score, short_score, is_extended


def _score_series_metrics(scores: Sequence[float]) -> Dict[str, Any]:
    score_history = [round(_safe_float(value), 1) for value in scores][-MAX_HISTORY_POINTS:]
    if not score_history:
        return {
            "score_history": [],
            "score_change_5m": 0.0,
            "score_change_10m": 0.0,
            "score_change_15m": 0.0,
            "score_slope": 0.0,
            "score_acceleration": 0.0,
            "improving_streak": 0,
            "weakening_streak": 0,
            "pulse_trend_label": "Flat",
            "pulse_trend_strength": 0.0,
        }

    def _change(steps: int) -> float:
        if len(score_history) <= steps:
            return 0.0
        return round(score_history[-1] - score_history[-(steps + 1)], 1)

    diffs = [round(score_history[index] - score_history[index - 1], 1) for index in range(1, len(score_history))]
    improving_streak = 0
    for diff in reversed(diffs):
        if diff > 0:
            improving_streak += 1
            continue
        break
    weakening_streak = 0
    for diff in reversed(diffs):
        if diff < 0:
            weakening_streak += 1
            continue
        break

    score_slope = 0.0
    if len(score_history) >= 3:
        x_axis = np.arange(len(score_history), dtype="float64")
        score_slope = round(float(np.polyfit(x_axis, np.array(score_history, dtype="float64"), 1)[0]), 2)

    score_acceleration = 0.0
    if len(diffs) >= 2:
        score_acceleration = round(diffs[-1] - diffs[-2], 2)

    score_change_5m = _change(1)
    score_change_10m = _change(2)
    score_change_15m = _change(3)

    if score_change_10m >= 2.0 or score_slope >= 1.0 or improving_streak >= 2:
        pulse_trend_label = "Rising"
    elif score_change_10m <= -2.0 or score_slope <= -1.0 or weakening_streak >= 2:
        pulse_trend_label = "Falling"
    else:
        pulse_trend_label = "Flat"

    trend_strength = min(
        100.0,
        abs(score_change_10m) * 9.0
        + abs(score_slope) * 20.0
        + max(improving_streak, weakening_streak) * 8.0
        + abs(score_acceleration) * 7.0,
    )
    if pulse_trend_label == "Flat":
        trend_strength *= 0.55

    return {
        "score_history": score_history,
        "score_change_5m": score_change_5m,
        "score_change_10m": score_change_10m,
        "score_change_15m": score_change_15m,
        "score_slope": round(score_slope, 2),
        "score_acceleration": round(score_acceleration, 2),
        "improving_streak": improving_streak,
        "weakening_streak": weakening_streak,
        "pulse_trend_label": pulse_trend_label,
        "pulse_trend_strength": round(trend_strength, 1),
    }


def calculate_pulse_trend(
    symbol: str,
    current_score: float,
    bar_key: str,
    session_key: str,
    commit: bool,
) -> Dict[str, Any]:
    with _lock:
        state = _score_state.setdefault(symbol, {"session_key": session_key, "points": []})
        if state.get("session_key") != session_key:
            state["session_key"] = session_key
            state["points"] = []

        points = [dict(point) for point in state.get("points", [])]
        if points and points[-1].get("bar_key") == bar_key:
            points[-1]["score"] = round(current_score, 1)
        else:
            points.append({"bar_key": bar_key, "score": round(current_score, 1)})
        points = points[-MAX_HISTORY_POINTS:]

        if commit:
            state["points"] = points

    return _score_series_metrics([point["score"] for point in points])


def infer_direction(long_score: float, short_score: float) -> Tuple[str, float]:
    score_diff = round(long_score - short_score, 1)
    if score_diff >= 4.0:
        return "LONG", normalize_score(score_diff, 3.0, 18.0)
    if score_diff <= -4.0:
        return "SHORT", normalize_score(abs(score_diff), 3.0, 18.0)
    return "NEUTRAL", normalize_score(abs(score_diff), 4.0, 18.0) * 0.55


def _time_bucket_for(timestamp: datetime) -> str:
    current_time = timestamp.time()
    if current_time < dt_time(10, 0):
        return "DISCOVERY"
    if current_time < dt_time(13, 0):
        return "TREND"
    return "LATE"


def _time_adjustment(score_time_bucket: str, pulse_trend_label: str) -> Tuple[float, float]:
    if score_time_bucket == "DISCOVERY":
        return -1.5, 0.92
    if score_time_bucket == "TREND":
        return 1.0, 1.05
    if pulse_trend_label == "Rising":
        return 0.5, 1.02
    return -0.8, 0.97


def _fallback_prev_close(stock: Dict[str, Any], session_df: pd.DataFrame) -> float:
    live_price = _safe_float(stock.get("ltp"))
    change_pct = _safe_float(stock.get("change_pct"))
    if live_price > 0 and abs(change_pct) < 95:
        base = live_price / (1.0 + (change_pct / 100.0))
        if base > 0:
            return round(base, 2)
    return _safe_float(session_df["Open"].iloc[0], live_price)


def _build_warning_flags(
    direction: str,
    distance_from_vwap_pct: float,
    long_consistency: float,
    short_consistency: float,
    one_bar_spike: bool,
    relative_strength_score: float,
    pulse_trend_label: str,
    volume_pace_ratio: float,
    is_extended: bool,
) -> List[str]:
    warning_flags: List[str] = []
    consistency_score = long_consistency if direction == "LONG" else short_consistency if direction == "SHORT" else max(long_consistency, short_consistency)

    if is_extended or abs(distance_from_vwap_pct) >= 1.4:
        warning_flags.append("far_from_vwap")
    if consistency_score < 45:
        warning_flags.append("low_consistency")
    if one_bar_spike:
        warning_flags.append("one_bar_spike")
    if relative_strength_score < 40:
        warning_flags.append("weak_relative_strength")
    if pulse_trend_label == "Falling":
        warning_flags.append("fading_score")
    if volume_pace_ratio < 1.0:
        warning_flags.append("low_volume_confirmation")
    return warning_flags


def _nifty_change_from_sources(raw: Optional[pd.DataFrame]) -> float:
    try:
        quotes = fetch_nse_index_quotes()
        for key in ("NIFTY 50", "NIFTY50", "NIFTY 50 PR 2X LEV", "NIFTY"):
            quote = quotes.get(key)
            if quote and quote.get("percentChange") is not None:
                return round(_safe_float(quote.get("percentChange")), 2)
    except Exception as exc:
        logger.warning("Momentum Pulse Nifty quote fetch failed, using history fallback: %s", exc)

    nifty_df = _get_sym_df(raw, "^NSEI")
    sessions = _split_sessions(nifty_df)
    if len(sessions) >= 2:
        current_close = _safe_float(sessions[-1][1]["Close"].dropna().iloc[-1])
        prev_close = _safe_float(sessions[-2][1]["Close"].dropna().iloc[-1])
        if current_close > 0 and prev_close > 0:
            return round(((current_close - prev_close) / prev_close) * 100.0, 2)
    return 0.0


def _evaluate_symbol(
    stock: Dict[str, Any],
    df_5m: Optional[pd.DataFrame],
    nifty_change_pct: float,
) -> Optional[Dict[str, Any]]:
    if df_5m is None or df_5m.empty:
        return None

    sessions = _split_sessions(df_5m)
    if len(sessions) < 2:
        return None

    current_session_date, session_df = sessions[-1]
    if len(session_df) < MIN_SESSION_BARS:
        return None

    historical_sessions = sessions[:-1][-LOOKBACK_SESSIONS:]
    if not historical_sessions:
        return None

    latest_ts = session_df.index[-1]
    cutoff = _current_cutoff(session_df)
    live_price = _safe_float(stock.get("ltp"), _safe_float(session_df["Close"].iloc[-1]))
    if live_price <= 0:
        return None

    current_prev_close = _safe_float(historical_sessions[-1][1]["Close"].dropna().iloc[-1], _fallback_prev_close(stock, session_df))
    if current_prev_close <= 0:
        return None

    change_pct = _safe_float(stock.get("change_pct"), round(((live_price - current_prev_close) / current_prev_close) * 100.0, 2))

    today_cum_volume, avg_20d_cum_volume_same_time, volume_pace_ratio, volume_pace_score = calculate_same_time_cum_volume_baseline(
        session_df,
        historical_sessions,
        cutoff,
    )
    (
        intraday_range_abs,
        intraday_range_pct,
        avg_20d_range_same_time_abs,
        avg_20d_range_pct_same_time,
        range_expansion_ratio,
        range_expansion_score,
    ) = calculate_same_time_range_baseline(
        session_df,
        historical_sessions,
        cutoff,
        current_prev_close,
    )
    relative_strength, long_rs_score, short_rs_score = calculate_relative_strength_scores(change_pct, nifty_change_pct)
    long_consistency, short_consistency, one_bar_spike = calculate_directional_consistency(session_df)
    vwap, distance_from_vwap_pct, long_vwap_score, short_vwap_score, is_extended = calculate_vwap_alignment(
        session_df,
        live_price,
        _safe_float(stock.get("vwap")),
    )

    base_long_score = (
        volume_pace_score * 0.30
        + range_expansion_score * 0.25
        + long_rs_score * 0.20
        + long_consistency * 0.10
        + long_vwap_score * 0.05
    )
    base_short_score = (
        volume_pace_score * 0.30
        + range_expansion_score * 0.25
        + short_rs_score * 0.20
        + short_consistency * 0.10
        + short_vwap_score * 0.05
    )
    provisional_score = round(max(base_long_score, base_short_score), 1)
    session_key = current_session_date.isoformat()
    bar_key = latest_ts.strftime("%Y-%m-%d %H:%M")
    provisional_trend = calculate_pulse_trend(
        str(stock.get("symbol") or "").upper(),
        provisional_score,
        bar_key,
        session_key,
        commit=False,
    )

    pulse_trend_strength = _safe_float(provisional_trend.get("pulse_trend_strength"))
    score_time_bucket = _time_bucket_for(latest_ts)

    long_score = (
        volume_pace_score * 0.30
        + range_expansion_score * 0.25
        + long_rs_score * 0.20
        + long_consistency * 0.10
        + long_vwap_score * 0.05
        + pulse_trend_strength * 0.10
    )
    short_score = (
        volume_pace_score * 0.30
        + range_expansion_score * 0.25
        + short_rs_score * 0.20
        + short_consistency * 0.10
        + short_vwap_score * 0.05
        + pulse_trend_strength * 0.10
    )

    direction, direction_confidence = infer_direction(long_score, short_score)
    preview_label = str(provisional_trend.get("pulse_trend_label", "Flat"))
    score_adjustment, confidence_multiplier = _time_adjustment(score_time_bucket, preview_label)
    long_score = round(max(0.0, min(100.0, long_score + score_adjustment)), 1)
    short_score = round(max(0.0, min(100.0, short_score + score_adjustment)), 1)

    direction, direction_confidence = infer_direction(long_score, short_score)
    direction_confidence = round(max(0.0, min(100.0, direction_confidence * confidence_multiplier)), 1)
    momentum_pulse_score = round(max(long_score, short_score), 1)

    committed_trend = calculate_pulse_trend(
        str(stock.get("symbol") or "").upper(),
        momentum_pulse_score,
        bar_key,
        session_key,
        commit=True,
    )
    pulse_trend_strength = _safe_float(committed_trend.get("pulse_trend_strength"))
    pulse_trend_label = str(committed_trend.get("pulse_trend_label", "Flat"))
    if pulse_trend_label != preview_label:
        score_adjustment, confidence_multiplier = _time_adjustment(score_time_bucket, pulse_trend_label)
        long_score = round(max(0.0, min(100.0, base_long_score + pulse_trend_strength * 0.10 + score_adjustment)), 1)
        short_score = round(max(0.0, min(100.0, base_short_score + pulse_trend_strength * 0.10 + score_adjustment)), 1)
        direction, direction_confidence = infer_direction(long_score, short_score)
        direction_confidence = round(max(0.0, min(100.0, direction_confidence * confidence_multiplier)), 1)
        momentum_pulse_score = round(max(long_score, short_score), 1)
        committed_trend = calculate_pulse_trend(
            str(stock.get("symbol") or "").upper(),
            momentum_pulse_score,
            bar_key,
            session_key,
            commit=True,
        )
        pulse_trend_strength = _safe_float(committed_trend.get("pulse_trend_strength"))
        pulse_trend_label = str(committed_trend.get("pulse_trend_label", "Flat"))

    selected_rs_score = long_rs_score if direction == "LONG" else short_rs_score if direction == "SHORT" else max(long_rs_score, short_rs_score)
    warning_flags = _build_warning_flags(
        direction,
        distance_from_vwap_pct,
        long_consistency,
        short_consistency,
        one_bar_spike,
        selected_rs_score,
        pulse_trend_label,
        volume_pace_ratio,
        is_extended,
    )

    return {
        "symbol": stock.get("symbol"),
        "ltp": round(live_price, 2),
        "change_pct": round(change_pct, 2),
        "direction": direction,
        "direction_confidence": round(direction_confidence, 1),
        "momentum_pulse_score": momentum_pulse_score,
        "tier": get_tier(momentum_pulse_score),
        "volume_pace_score": round(volume_pace_score, 1),
        "volume_pace_ratio": round(volume_pace_ratio, 2),
        "range_expansion_score": round(range_expansion_score, 1),
        "range_expansion_ratio": round(range_expansion_ratio, 2),
        "relative_strength": round(relative_strength, 2),
        "long_relative_strength_score": round(long_rs_score, 1),
        "short_relative_strength_score": round(short_rs_score, 1),
        "long_directional_consistency_score": round(long_consistency, 1),
        "short_directional_consistency_score": round(short_consistency, 1),
        "long_vwap_alignment_score": round(long_vwap_score, 1),
        "short_vwap_alignment_score": round(short_vwap_score, 1),
        "pulse_trend_strength": round(pulse_trend_strength, 1),
        "today_cum_volume": round(today_cum_volume, 1),
        "avg_20d_cum_volume_same_time": round(avg_20d_cum_volume_same_time, 1),
        "intraday_range_abs": round(intraday_range_abs, 2),
        "intraday_range_pct": round(intraday_range_pct, 2),
        "avg_20d_range_same_time_abs": round(avg_20d_range_same_time_abs, 2),
        "avg_20d_range_pct_same_time": round(avg_20d_range_pct_same_time, 2),
        "score_history": committed_trend.get("score_history", []),
        "score_change_5m": committed_trend.get("score_change_5m", 0.0),
        "score_change_10m": committed_trend.get("score_change_10m", 0.0),
        "score_change_15m": committed_trend.get("score_change_15m", 0.0),
        "score_slope": committed_trend.get("score_slope", 0.0),
        "score_acceleration": committed_trend.get("score_acceleration", 0.0),
        "improving_streak": committed_trend.get("improving_streak", 0),
        "weakening_streak": committed_trend.get("weakening_streak", 0),
        "pulse_trend_label": pulse_trend_label,
        "vwap": round(vwap, 2),
        "distance_from_vwap_pct": round(distance_from_vwap_pct, 2),
        "score_time_bucket": score_time_bucket,
        "is_extended": is_extended,
        "warning_flags": warning_flags,
        "volume_surge": volume_pace_ratio >= 1.5,
        "range_expansion": range_expansion_ratio >= 1.3,
        "index_outperformer": relative_strength >= 0.75 if direction == "LONG" else relative_strength <= -0.75 if direction == "SHORT" else abs(relative_strength) >= 1.0,
        "trend_consistent": (long_consistency >= 60.0) if direction == "LONG" else (short_consistency >= 60.0) if direction == "SHORT" else max(long_consistency, short_consistency) >= 65.0,
        "improving_now": pulse_trend_label == "Rising" and _safe_float(committed_trend.get("score_change_10m")) > 0,
        "long_score": round(long_score, 1),
        "short_score": round(short_score, 1),
        "latest_bar_time": latest_ts.strftime("%H:%M"),
    }


def _compute_momentum_pulse(scanner_stocks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    symbols_ns = []
    seen = set()
    for stock in scanner_stocks:
        symbol = str(stock.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols_ns.append(f"{symbol}.NS")

    if not symbols_ns:
        return [], 0.0

    raw = yf.download(
        tickers=" ".join(symbols_ns + ["^NSEI"]),
        period="35d",
        interval="5m",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
        timeout=25,
    )
    benchmark_change_pct = _nifty_change_from_sources(raw)

    results: List[Dict[str, Any]] = []
    for stock in scanner_stocks:
        symbol = str(stock.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        df_5m = _get_sym_df(raw, f"{symbol}.NS")
        try:
            row = _evaluate_symbol(stock, df_5m, benchmark_change_pct)
        except Exception as exc:
            logger.warning("Momentum Pulse failed for %s: %s", symbol, exc)
            row = None
        if row:
            results.append(row)

    results.sort(
        key=lambda item: (
            _safe_float(item.get("momentum_pulse_score")),
            _safe_float(item.get("pulse_trend_strength")),
            _safe_float(item.get("direction_confidence")),
            abs(_safe_float(item.get("relative_strength"))),
        ),
        reverse=True,
    )
    for index, item in enumerate(results, start=1):
        item["rank"] = index
    return results, benchmark_change_pct


def _refresh_momentum_pulse_cache(source_key: str, scanner_stocks: List[Dict[str, Any]]) -> None:
    try:
        results, benchmark_change_pct = _compute_momentum_pulse(scanner_stocks)
        with _lock:
            _pulse_cache["source_key"] = source_key
            _pulse_cache["computed_at"] = time.time()
            _pulse_cache["last_updated"] = _now_ist_str()
            _pulse_cache["benchmark_change_pct"] = benchmark_change_pct
            _pulse_cache["results"] = results
            _pulse_cache["error"] = ""
    except Exception as exc:
        logger.error("Momentum Pulse refresh failed: %s", exc, exc_info=True)
        with _lock:
            _pulse_cache["error"] = str(exc)
    finally:
        with _lock:
            _pulse_cache["is_loading"] = False


def schedule_momentum_pulse_refresh(
    scanner_stocks: List[Dict[str, Any]],
    last_updated: str,
    force: bool = False,
) -> bool:
    if not scanner_stocks:
        return False

    snapshot = _snapshot_cache()
    source_key = _build_source_key(last_updated, scanner_stocks)
    cache_fresh = (
        snapshot["source_key"] == source_key
        and bool(snapshot["results"])
        and (time.time() - _safe_float(snapshot["computed_at"])) <= REFRESH_COOLDOWN_SECONDS
    )
    cooldown_elapsed = (time.time() - _safe_float(snapshot["last_attempt"])) >= 15.0

    if snapshot["is_loading"]:
        return False
    if not force and (cache_fresh or not cooldown_elapsed):
        return False

    with _lock:
        _pulse_cache["is_loading"] = True
        _pulse_cache["last_attempt"] = time.time()
        _pulse_cache["error"] = ""

    thread = threading.Thread(
        target=_refresh_momentum_pulse_cache,
        args=(source_key, list(scanner_stocks)),
        daemon=True,
        name="momentum-pulse-refresh",
    )
    thread.start()
    return True


def get_momentum_pulse_cache_status() -> Dict[str, Any]:
    snapshot = _snapshot_cache()
    return {
        "last_updated": snapshot["last_updated"],
        "is_loading": snapshot["is_loading"],
        "error": snapshot["error"],
        "total_cached": len(snapshot["results"]),
    }


def _filter_results(
    results: Sequence[Dict[str, Any]],
    direction: str,
    include_veryweak: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    filtered = list(results)
    if direction in {"LONG", "SHORT"}:
        filtered = [item for item in filtered if str(item.get("direction")) == direction]
    if not include_veryweak:
        filtered = [item for item in filtered if str(item.get("tier")) != "veryweak"]
    filtered.sort(
        key=lambda item: (
            _safe_float(item.get("momentum_pulse_score")),
            _safe_float(item.get("pulse_trend_strength")),
            _safe_float(item.get("direction_confidence")),
            abs(_safe_float(item.get("relative_strength"))),
        ),
        reverse=True,
    )
    trimmed = filtered[:limit] if limit > 0 else filtered
    for index, item in enumerate(trimmed, start=1):
        item["rank"] = index
    return trimmed


def get_momentum_pulse(
    scanner_stocks: List[Dict[str, Any]],
    last_updated: str,
    direction: str = "ALL",
    include_veryweak: bool = False,
    limit: int = 40,
) -> Dict[str, Any]:
    normalized_direction = str(direction or "ALL").strip().upper()
    if normalized_direction not in {"ALL", "LONG", "SHORT"}:
        normalized_direction = "ALL"

    source_key = _build_source_key(last_updated, scanner_stocks)
    snapshot = _snapshot_cache()
    cached_source_key = str(snapshot["source_key"] or "")
    cached_results = list(snapshot["results"] or [])
    cached_benchmark = _safe_float(snapshot["benchmark_change_pct"])
    cached_at = _safe_float(snapshot["computed_at"])
    pulse_last_updated = str(snapshot["last_updated"] or "")
    is_loading = bool(snapshot["is_loading"])
    last_attempt = _safe_float(snapshot["last_attempt"])
    cache_error = str(snapshot["error"] or "")

    if cached_source_key == source_key and cached_results and (time.time() - cached_at) <= REFRESH_COOLDOWN_SECONDS:
        stocks = _filter_results(cached_results, normalized_direction, include_veryweak, limit)
        return {
            "stocks": stocks,
            "total": len(stocks),
            "last_updated": pulse_last_updated or last_updated,
            "market_data_last_updated": last_updated,
            "direction": normalized_direction,
            "include_veryweak": include_veryweak,
            "benchmark_change_pct": round(cached_benchmark, 2),
            "is_loading": False,
            "status": "ready",
        }

    should_refresh = (source_key != cached_source_key) or not cached_results
    cooldown_elapsed = (time.time() - last_attempt) >= 15.0
    if should_refresh and not is_loading and cooldown_elapsed:
        schedule_momentum_pulse_refresh(scanner_stocks, last_updated)

    stocks = _filter_results(cached_results, normalized_direction, include_veryweak, limit)
    has_any_cached_results = bool(cached_results)
    status = "warming_up" if not has_any_cached_results else "stale_refreshing" if should_refresh else "ready"
    return {
        "stocks": stocks,
        "total": len(stocks),
        "last_updated": pulse_last_updated or last_updated,
        "market_data_last_updated": last_updated,
        "direction": normalized_direction,
        "include_veryweak": include_veryweak,
        "benchmark_change_pct": round(cached_benchmark, 2),
        "is_loading": bool(_snapshot_cache().get("is_loading")),
        "status": status,
        "message": cache_error or ("Momentum Pulse cache is warming up" if not has_any_cached_results else "Momentum Pulse refresh is running in background" if should_refresh else ""),
    }