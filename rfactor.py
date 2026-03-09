import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("rfactor")


@dataclass(frozen=True)
class BreakoutLevel:
    name: str
    value: float
    side: str
    weight: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _positive_score(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return round(100.0 * _clamp(math.tanh(max(0.0, value) / scale)), 1)


def _signed_score(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return round(100.0 * math.tanh(value / scale), 1)


def calculate_rsi(closes: pd.Series, period: int = 14) -> float:
    try:
        series = pd.Series(closes, dtype="float64").dropna()
        if len(series) < period + 1:
            return 50.0
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        val = float(rsi.dropna().iloc[-1])
        return round(val, 1) if not np.isnan(val) else 50.0
    except Exception:
        return 50.0


def _calculate_rsi_series(closes: pd.Series, period: int = 14) -> pd.Series:
    series = pd.Series(closes, dtype="float64").dropna()
    if len(series) < period + 1:
        return pd.Series(dtype="float64")
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> float:
    try:
        if df is None or df.empty:
            return 50.0
        high = pd.Series(df["High"], dtype="float64")
        low = pd.Series(df["Low"], dtype="float64")
        close = pd.Series(df["Close"], dtype="float64")
        volume = pd.Series(df["Volume"], dtype="float64")
        tp = (high + low + close) / 3.0
        raw_mf = tp * volume
        pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
        neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
        pos_sum = pos_mf.rolling(period).sum()
        neg_sum = neg_mf.rolling(period).sum()
        money_ratio = pos_sum / neg_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + money_ratio))
        val = float(mfi.dropna().iloc[-1])
        return round(val, 1) if not np.isnan(val) else 50.0
    except Exception:
        return 50.0


def get_rfactor_color_tier(score: float) -> str:
    if score >= 3.5:
        return "strong"
    if score >= 2.5:
        return "moderate"
    if score >= 1.5:
        return "weak"
    return "very_weak"


def _get_sym_df(data: Any, symbol: str) -> Optional[pd.DataFrame]:
    try:
        if data is None:
            return None
        if not isinstance(data, pd.DataFrame):
            return None
        if isinstance(data.columns, pd.MultiIndex):
            if symbol not in data.columns.get_level_values(0):
                return None
            df = data[symbol].copy()
        else:
            df = data.copy()
        if df.empty:
            return None
        df = df.sort_index()
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col not in df.columns:
                return None
        return df
    except Exception:
        return None


def _session_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    try:
        if df is None or df.empty:
            return None
        latest_date = df.index[-1].date()
        session = df[df.index.date == latest_date].copy()
        return session if not session.empty else None
    except Exception:
        return None


def _previous_session_df(df: Optional[pd.DataFrame], session_date) -> Optional[pd.DataFrame]:
    try:
        if df is None or df.empty:
            return None
        prev = df[df.index.date < session_date].copy()
        if prev.empty:
            return None
        prev_date = prev.index[-1].date()
        prev = prev[prev.index.date == prev_date].copy()
        return prev if not prev.empty else None
    except Exception:
        return None


def _intraday_vwap(df: Optional[pd.DataFrame]) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    volume = pd.Series(df["Volume"], dtype="float64").fillna(0.0)
    typical_price = (
        pd.Series(df["High"], dtype="float64")
        + pd.Series(df["Low"], dtype="float64")
        + pd.Series(df["Close"], dtype="float64")
    ) / 3.0
    cum_volume = volume.cumsum().replace(0, np.nan)
    return (typical_price * volume).cumsum() / cum_volume


def _obv_series(close: pd.Series, volume: pd.Series) -> pd.Series:
    close_series = pd.Series(close, dtype="float64").ffill()
    volume_series = pd.Series(volume, dtype="float64").fillna(0.0)
    if close_series.empty:
        return pd.Series(dtype="float64")
    direction = np.sign(close_series.diff().fillna(0.0))
    obv = (direction * volume_series).cumsum()
    return pd.Series(obv, index=close_series.index, dtype="float64")


def _normalized_slope(series: pd.Series, lookback: int = 6, normalizer: Optional[float] = None) -> float:
    try:
        clean = pd.Series(series, dtype="float64").dropna().tail(lookback)
        if len(clean) < 3:
            return 0.0
        x = np.arange(len(clean), dtype="float64")
        slope = float(np.polyfit(x, clean.to_numpy(dtype="float64"), 1)[0])
        denom = normalizer if normalizer and normalizer > 0 else max(abs(clean.mean()), 1e-6)
        return slope / denom
    except Exception:
        return 0.0


def _atr(df: Optional[pd.DataFrame], period: int = 14) -> float:
    try:
        if df is None or df.empty:
            return 0.0
        high = pd.Series(df["High"], dtype="float64")
        low = pd.Series(df["Low"], dtype="float64")
        close = pd.Series(df["Close"], dtype="float64")
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_series = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().dropna()
        return _safe_float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    except Exception:
        return 0.0


def _opening_range_levels(session_df: Optional[pd.DataFrame], bars: int = 6) -> Tuple[float, float]:
    if session_df is None or session_df.empty:
        return 0.0, 0.0
    segment = session_df.iloc[: min(bars, len(session_df))]
    return _safe_float(segment["High"].max()), _safe_float(segment["Low"].min())


def _swing_levels(session_df: Optional[pd.DataFrame], lookback: int = 20, exclude_latest: int = 2) -> Tuple[float, float]:
    if session_df is None or len(session_df) <= exclude_latest + 2:
        return 0.0, 0.0
    base = session_df.iloc[: len(session_df) - exclude_latest].tail(lookback)
    if base.empty:
        return 0.0, 0.0
    return _safe_float(base["High"].max()), _safe_float(base["Low"].min())


def _prev_day_levels(df_5m: Optional[pd.DataFrame], session_date) -> Tuple[float, float, float]:
    prev_df = _previous_session_df(df_5m, session_date)
    if prev_df is None or prev_df.empty:
        return 0.0, 0.0, 0.0
    prev_high = _safe_float(prev_df["High"].max())
    prev_low = _safe_float(prev_df["Low"].min())
    prev_close = _safe_float(prev_df["Close"].dropna().iloc[-1]) if not prev_df["Close"].dropna().empty else 0.0
    return prev_high, prev_low, prev_close


def _build_breakout_levels(df_5m: Optional[pd.DataFrame], session_df: Optional[pd.DataFrame]) -> List[BreakoutLevel]:
    if session_df is None or session_df.empty:
        return []
    session_date = session_df.index[-1].date()
    prev_day_high, prev_day_low, _ = _prev_day_levels(df_5m, session_date)
    or_high, or_low = _opening_range_levels(session_df)
    swing_high, swing_low = _swing_levels(session_df)

    levels: List[BreakoutLevel] = []
    if prev_day_high > 0:
        levels.append(BreakoutLevel("prev_day_high", prev_day_high, "LONG", 1.0))
    if prev_day_low > 0:
        levels.append(BreakoutLevel("prev_day_low", prev_day_low, "SHORT", 1.0))
    if or_high > 0:
        levels.append(BreakoutLevel("opening_range_high", or_high, "LONG", 0.75))
    if or_low > 0:
        levels.append(BreakoutLevel("opening_range_low", or_low, "SHORT", 0.75))
    if swing_high > 0:
        levels.append(BreakoutLevel("swing_high_20", swing_high, "LONG", 0.55))
    if swing_low > 0:
        levels.append(BreakoutLevel("swing_low_20", swing_low, "SHORT", 0.55))
    return levels


def _breakout_levels_payload(levels: List[BreakoutLevel]) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for level in levels:
        payload[level.name] = round(level.value, 2)
    return payload


def _level_distance_pct(close: float, level: BreakoutLevel) -> float:
    if level.value <= 0:
        return 999.0
    return ((close - level.value) / level.value) * 100.0


def _proximity_for_side(close: float, levels: List[BreakoutLevel], side: str) -> Tuple[float, str, float]:
    side_levels = [level for level in levels if level.side == side and level.value > 0]
    if not side_levels or close <= 0:
        return 0.0, "", 999.0

    best_score = 0.0
    best_name = ""
    best_dist = 999.0
    for level in side_levels:
        dist_pct = _level_distance_pct(close, level)
        if side == "LONG":
            if dist_pct <= -1.5:
                candidate = 0.0
            elif dist_pct <= 0:
                candidate = 45.0 * level.weight
            else:
                candidate = 100.0 * math.exp(-dist_pct / 0.65) * level.weight
        else:
            if dist_pct >= 1.5:
                candidate = 0.0
            elif dist_pct >= 0:
                candidate = 45.0 * level.weight
            else:
                candidate = 100.0 * math.exp(-abs(dist_pct) / 0.65) * level.weight

        if candidate > best_score:
            best_score = candidate
            best_name = level.name
            best_dist = round(dist_pct, 2)

    return round(best_score, 1), best_name, best_dist


def _compression_score(session_df: Optional[pd.DataFrame], atr_value: float) -> float:
    if session_df is None or len(session_df) < 8 or atr_value <= 0:
        return 0.0
    recent = session_df.tail(min(8, len(session_df)))
    recent_range = _safe_float(recent["High"].max()) - _safe_float(recent["Low"].min())
    normalized = recent_range / max(atr_value * max(len(recent) / 2.5, 1.0), 1e-6)
    return round(100.0 * (1.0 - _clamp((normalized - 0.8) / 1.6)), 1)


def _volume_acceleration(session_df: Optional[pd.DataFrame]) -> Tuple[float, float]:
    if session_df is None or len(session_df) < 8:
        return 1.0, 0.0
    volume = pd.Series(session_df["Volume"], dtype="float64").fillna(0.0)
    recent = _safe_float(volume.tail(3).mean())
    base = _safe_float(volume.iloc[:-3].tail(10).mean())
    if base <= 0:
        return 1.0, 0.0
    ratio = recent / base
    score = round(100.0 * _clamp(math.tanh(max(0.0, ratio - 1.0) / 0.55)), 1)
    return round(ratio, 2), score


def _range_cleanliness_penalty(session_df: Optional[pd.DataFrame], vwap_series: pd.Series) -> float:
    if session_df is None or len(session_df) < 6 or vwap_series.empty:
        return 0.0
    close = pd.Series(session_df["Close"], dtype="float64")
    vwap = vwap_series.reindex(close.index).ffill()
    relation = np.sign(close - vwap)
    crosses_vwap = int((relation.diff().abs() == 2).sum())

    or_high, or_low = _opening_range_levels(session_df)
    or_mid = (or_high + or_low) / 2.0 if or_high > 0 and or_low > 0 else 0.0
    crosses_or = 0
    if or_mid > 0:
        or_relation = np.sign(close - or_mid)
        crosses_or = int((or_relation.diff().abs() == 2).sum())

    total_crosses = crosses_vwap + crosses_or
    return round(_clamp(total_crosses / 6.0), 2)


def _rsi_zone_quality(rsi_value: float, direction: str) -> float:
    if direction == "LONG":
        if 50 <= rsi_value <= 65:
            return 100.0
        if 65 < rsi_value <= 72:
            return 75.0
        if 45 <= rsi_value < 50:
            return 60.0
        if rsi_value > 72:
            return max(10.0, 75.0 - (rsi_value - 72.0) * 6.0)
        return max(0.0, 40.0 - (45.0 - rsi_value) * 3.0)
    if direction == "SHORT":
        if 35 <= rsi_value <= 50:
            return 100.0
        if 28 <= rsi_value < 35:
            return 75.0
        if 50 < rsi_value <= 55:
            return 60.0
        if rsi_value < 28:
            return max(10.0, 75.0 - (28.0 - rsi_value) * 6.0)
        return max(0.0, 40.0 - (rsi_value - 55.0) * 3.0)
    return 0.0


def _candle_quality(last_row: pd.Series, direction: str) -> float:
    high = _safe_float(last_row.get("High"))
    low = _safe_float(last_row.get("Low"))
    close = _safe_float(last_row.get("Close"))
    candle_range = high - low
    if candle_range <= 0:
        return 0.0
    close_pos = (close - low) / candle_range
    if direction == "LONG":
        return round(100.0 * _clamp((close_pos - 0.45) / 0.35), 1)
    if direction == "SHORT":
        return round(100.0 * _clamp((0.55 - close_pos) / 0.35), 1)
    return 0.0


def _one_min_confirmation(df_1m: Optional[pd.DataFrame], direction: str) -> float:
    if df_1m is None or df_1m.empty or direction == "NEUTRAL":
        return 0.0
    session_df = _session_df(df_1m)
    if session_df is None or len(session_df) < 5:
        return 0.0
    close = pd.Series(session_df["Close"], dtype="float64")
    volume = pd.Series(session_df["Volume"], dtype="float64").fillna(0.0)
    rsi_series = _calculate_rsi_series(close)
    rsi_slope = _normalized_slope(rsi_series, lookback=5, normalizer=6.0)
    vol_recent = _safe_float(volume.tail(3).mean())
    vol_base = _safe_float(volume.iloc[:-3].tail(10).mean())
    vol_ratio = vol_recent / vol_base if vol_base > 0 else 1.0
    close_slope = _normalized_slope(close, lookback=5, normalizer=max(abs(_safe_float(close.tail(5).mean())), 1e-6))

    if direction == "LONG":
        alignment = max(0.0, rsi_slope) + max(0.0, close_slope)
    else:
        alignment = max(0.0, -rsi_slope) + max(0.0, -close_slope)

    score = 100.0 * _clamp(math.tanh(alignment / 0.012)) * _clamp(math.tanh(max(0.0, vol_ratio - 1.0) / 0.5))
    return round(score, 1)


def _infer_direction(
    rsi_slope: float,
    obv_slope: float,
    price_slope: float,
    long_proximity: float,
    short_proximity: float,
) -> Tuple[str, float]:
    bullish_weight = 0.0
    bearish_weight = 0.0
    bullish_votes = 0
    bearish_votes = 0

    signals = [
        (rsi_slope, 0.012, 1.2),
        (obv_slope, 0.02, 0.9),
        (price_slope, 0.0018, 0.9),
    ]
    for value, threshold, weight in signals:
        if value >= threshold:
            bullish_votes += 1
            bullish_weight += weight
        elif value <= -threshold:
            bearish_votes += 1
            bearish_weight += weight

    if long_proximity >= short_proximity + 12:
        bullish_votes += 1
        bullish_weight += 1.1
    elif short_proximity >= long_proximity + 12:
        bearish_votes += 1
        bearish_weight += 1.1

    if bullish_votes >= 2 and bullish_weight > bearish_weight:
        edge = bullish_weight - bearish_weight
        conf = round(100.0 * _clamp(0.35 + 0.18 * bullish_votes + 0.12 * edge), 1)
        return "LONG", conf
    if bearish_votes >= 2 and bearish_weight > bullish_weight:
        edge = bearish_weight - bullish_weight
        conf = round(100.0 * _clamp(0.35 + 0.18 * bearish_votes + 0.12 * edge), 1)
        return "SHORT", conf
    conf = round(100.0 * _clamp(abs(bullish_weight - bearish_weight) * 0.15), 1)
    return "NEUTRAL", conf


def _trigger_breakout_quality(
    close: float,
    previous_close: float,
    levels: List[BreakoutLevel],
    direction: str,
    candle_quality: float,
    atr_pct: float,
) -> Tuple[float, str, float]:
    if direction == "NEUTRAL" or close <= 0:
        return 0.0, "", 999.0

    candidates = [level for level in levels if level.side == direction and level.value > 0]
    if not candidates:
        return 0.0, "", 999.0

    best_quality = 0.0
    best_name = ""
    best_dist = 999.0

    for level in candidates:
        dist_pct = _level_distance_pct(close, level)
        prev_dist_pct = _level_distance_pct(previous_close, level)
        if direction == "LONG":
            if dist_pct <= 0:
                continue
            initial_break = prev_dist_pct <= 0 < dist_pct
            hold_score = 100.0 * _clamp(dist_pct / max(atr_pct * 0.85, 0.12))
        else:
            if dist_pct >= 0:
                continue
            initial_break = prev_dist_pct >= 0 > dist_pct
            hold_score = 100.0 * _clamp(abs(dist_pct) / max(atr_pct * 0.85, 0.12))

        quality = hold_score * 0.55 + candle_quality * 0.25 + (20.0 if initial_break else 10.0)
        quality *= level.weight
        if quality > best_quality:
            best_quality = quality
            best_name = level.name
            best_dist = round(dist_pct, 2)

    return round(min(best_quality, 100.0), 1), best_name, best_dist


def _vwap_acceptance_score(close_series: pd.Series, vwap_series: pd.Series, direction: str, atr_pct: float) -> float:
    if close_series.empty or vwap_series.empty or direction == "NEUTRAL":
        return 0.0
    closes = pd.Series(close_series, dtype="float64").tail(3)
    vwaps = pd.Series(vwap_series, dtype="float64").reindex(closes.index).ffill()
    if vwaps.isna().all():
        return 0.0
    aligned = 0
    for close, vwap in zip(closes, vwaps):
        if direction == "LONG" and close >= vwap:
            aligned += 1
        elif direction == "SHORT" and close <= vwap:
            aligned += 1
    acceptance = 100.0 * (aligned / max(len(closes), 1))
    dist_pct = abs((_safe_float(closes.iloc[-1]) - _safe_float(vwaps.iloc[-1])) / max(_safe_float(vwaps.iloc[-1]), 1e-6) * 100.0)
    stretch_penalty = 25.0 * _clamp((dist_pct - max(atr_pct * 0.9, 0.25)) / max(atr_pct * 1.2, 0.35))
    return round(max(0.0, acceptance - stretch_penalty), 1)


def _stage_weights(stage: str) -> Tuple[float, float]:
    if stage == "WARMING":
        return 0.75, 0.25
    if stage == "PRE_SIGNAL":
        return 0.65, 0.35
    if stage == "BREAKING":
        return 0.45, 0.55
    if stage == "CONFIRMED":
        return 0.35, 0.65
    if stage == "EXTENDED":
        return 0.25, 0.75
    return 0.55, 0.45


def _determine_stage(pre_score: float, trigger_score: float, breakout_quality: float, is_chase: bool) -> str:
    if is_chase and trigger_score >= 55:
        return "EXTENDED"
    if trigger_score >= 75 and breakout_quality >= 65:
        return "CONFIRMED"
    if trigger_score >= 55 and pre_score >= 45:
        return "BREAKING"
    if pre_score >= 60:
        return "PRE_SIGNAL"
    if pre_score >= 38:
        return "WARMING"
    return "NEUTRAL"


def _opportunity_score(pre_score: float, trigger_score: float, direction_conf: float, chop_penalty: float, is_chase: bool) -> float:
    raw = (
        pre_score * 0.40
        + trigger_score * 0.30
        + direction_conf * 0.20
        + (100.0 * (1.0 - chop_penalty)) * 0.10
    )
    if is_chase:
        raw *= 0.62
    return round(_clamp(raw / 100.0) * 100.0, 1)


def _change_pct_from_prev_close(close: float, prev_close: float, fallback: float) -> float:
    if prev_close > 0:
        return round(((close - prev_close) / prev_close) * 100.0, 2)
    return round(fallback, 2)


def _evaluate_symbol(
    stock: Dict[str, Any],
    df_5m: Optional[pd.DataFrame],
    df_1m: Optional[pd.DataFrame],
    nse_quote: Dict[str, Any],
    nifty_avg: float,
    live_change_pct: Optional[float] = None,
) -> Dict[str, Any]:
    close = _safe_float(stock.get("ltp"))
    vwap_live = _safe_float(stock.get("vwap"))
    day_high = _safe_float(stock.get("day_high"), close)
    day_low = _safe_float(stock.get("day_low"), close)

    session_df = _session_df(df_5m)
    latest_close = close
    previous_bar_close = close
    if session_df is not None and not session_df.empty:
        latest_close = _safe_float(session_df["Close"].dropna().iloc[-1], close)
        if close <= 0:
            close = latest_close
        previous_bar_close = _safe_float(session_df["Close"].dropna().iloc[-2], latest_close) if len(session_df["Close"].dropna()) >= 2 else latest_close
        day_high = _safe_float(session_df["High"].max(), day_high)
        day_low = _safe_float(session_df["Low"].min(), day_low)

    if close <= 0:
        close = latest_close

    vwap_series = _intraday_vwap(session_df)
    vwap = vwap_live if vwap_live > 0 else _safe_float(vwap_series.iloc[-1])

    session_date = session_df.index[-1].date() if session_df is not None and not session_df.empty else None
    prev_day_close = 0.0
    if session_date is not None:
        _, _, prev_day_close = _prev_day_levels(df_5m, session_date)

    close_series = pd.Series(dtype="float64") if session_df is None else pd.Series(session_df["Close"], dtype="float64").dropna()
    rsi_series = _calculate_rsi_series(close_series)
    rsi_value = round(_safe_float(rsi_series.iloc[-1], 50.0), 1) if not rsi_series.empty else 50.0
    mfi_value = calculate_mfi(session_df) if session_df is not None else 50.0
    atr_value = _atr(session_df)
    atr_pct = (atr_value / close * 100.0) if close > 0 and atr_value > 0 else 0.0

    levels = _build_breakout_levels(df_5m, session_df)
    breakout_levels = _breakout_levels_payload(levels)
    long_proximity, long_level_name, long_level_dist = _proximity_for_side(close, levels, "LONG")
    short_proximity, short_level_name, short_level_dist = _proximity_for_side(close, levels, "SHORT")

    rsi_slope_raw = _normalized_slope(rsi_series, lookback=6, normalizer=6.0)
    rsi_slope_5m = round(rsi_slope_raw * 1000.0, 2)
    obv = _obv_series(close_series, session_df["Volume"] if session_df is not None else pd.Series(dtype="float64"))
    avg_tail_volume = _safe_float(pd.Series(session_df["Volume"], dtype="float64").tail(10).mean()) if session_df is not None else 1.0
    obv_slope_raw = _normalized_slope(obv, lookback=6, normalizer=max(avg_tail_volume, 1.0))
    price_slope_raw = _normalized_slope(close_series, lookback=6, normalizer=max(abs(_safe_float(close_series.tail(6).mean(), close)), 1e-6))

    inferred_direction, direction_conf = _infer_direction(
        rsi_slope=rsi_slope_raw,
        obv_slope=obv_slope_raw,
        price_slope=price_slope_raw,
        long_proximity=long_proximity,
        short_proximity=short_proximity,
    )

    compression = _compression_score(session_df, atr_value)
    vol_accel_ratio, vol_accel_score = _volume_acceleration(session_df)
    chop_penalty = _range_cleanliness_penalty(session_df, vwap_series)
    one_min_boost = _one_min_confirmation(df_1m, inferred_direction)

    if inferred_direction == "LONG":
        obv_slope_score = _positive_score(obv_slope_raw, 0.03)
        rsi_lead_score = _positive_score(rsi_slope_raw, 0.018)
        proximity_score = long_proximity
        nearest_level = long_level_name
        dist_pct = long_level_dist
    elif inferred_direction == "SHORT":
        obv_slope_score = _positive_score(-obv_slope_raw, 0.03)
        rsi_lead_score = _positive_score(-rsi_slope_raw, 0.018)
        proximity_score = short_proximity
        nearest_level = short_level_name
        dist_pct = short_level_dist
    else:
        obv_slope_score = 0.0
        rsi_lead_score = 0.0
        proximity_score = max(long_proximity, short_proximity) * 0.45
        nearest_level = long_level_name if long_proximity >= short_proximity else short_level_name
        dist_pct = long_level_dist if long_proximity >= short_proximity else short_level_dist

    direction_multiplier = 0.45 + 0.55 * _clamp(direction_conf / 100.0)
    pre_score = (
        compression * 0.24
        + obv_slope_score * 0.16
        + rsi_lead_score * 0.24
        + vol_accel_score * 0.16
        + proximity_score * 0.15
        + one_min_boost * 0.05
    )
    pre_score *= direction_multiplier
    pre_score *= 1.0 - (0.40 * chop_penalty)
    pre_score = round(max(0.0, min(pre_score, 100.0)), 1)

    live_change = live_change_pct if live_change_pct is not None else _safe_float(stock.get("change_pct"), 0.0)
    live_change = _change_pct_from_prev_close(close, prev_day_close, live_change)
    relative_strength = round(live_change - nifty_avg, 2)

    candle_quality = _candle_quality(session_df.iloc[-1] if session_df is not None and not session_df.empty else pd.Series(dtype="float64"), inferred_direction)
    breakout_quality, breakout_level_name, breakout_dist_pct = _trigger_breakout_quality(
        close=close,
        previous_close=previous_bar_close,
        levels=levels,
        direction=inferred_direction,
        candle_quality=candle_quality,
        atr_pct=atr_pct,
    )
    vwap_acceptance = _vwap_acceptance_score(close_series, vwap_series, inferred_direction, atr_pct)
    volume_confirmation = round(min(100.0, vol_accel_score * 1.1), 1)
    rsi_zone = _rsi_zone_quality(rsi_value, inferred_direction)

    trigger_score = (
        breakout_quality * 0.38
        + volume_confirmation * 0.22
        + vwap_acceptance * 0.18
        + rsi_zone * 0.14
        + candle_quality * 0.08
    )
    if inferred_direction == "LONG" and live_change < 0:
        trigger_score *= 0.85
    elif inferred_direction == "SHORT" and live_change > 0:
        trigger_score *= 0.85
    trigger_score = round(max(0.0, min(trigger_score, 100.0)), 1)

    chase_flags: List[str] = []
    vwap_dist_pct = abs((close - vwap) / vwap * 100.0) if close > 0 and vwap > 0 else 0.0
    if vwap_dist_pct > max(1.1, atr_pct * 1.1):
        chase_flags.append("far_from_vwap")
    if inferred_direction == "LONG" and rsi_value >= 72:
        chase_flags.append("rsi_stretched")
    if inferred_direction == "SHORT" and rsi_value <= 28:
        chase_flags.append("rsi_stretched")
    if breakout_level_name:
        if inferred_direction == "LONG" and breakout_dist_pct > max(0.45, atr_pct * 0.9):
            chase_flags.append("above_breakout_band")
        if inferred_direction == "SHORT" and abs(breakout_dist_pct) > max(0.45, atr_pct * 0.9):
            chase_flags.append("below_breakout_band")
    if session_df is not None and len(session_df) >= 4 and atr_value > 0:
        move_3 = abs(_safe_float(session_df["Close"].iloc[-1]) - _safe_float(session_df["Close"].iloc[-4]))
        if move_3 > atr_value * 1.8:
            chase_flags.append("vertical_extension")

    is_chase = len(set(chase_flags)) >= 2
    chase_reason = ", ".join(sorted(set(chase_flags))) if is_chase else ""

    setup_stage = _determine_stage(pre_score, trigger_score, breakout_quality, is_chase)
    alert_stage = setup_stage
    pre_weight, trigger_weight = _stage_weights(setup_stage)
    rfactor_raw = pre_score * pre_weight + trigger_score * trigger_weight
    if is_chase:
        rfactor_raw *= 0.84
    rfactor = round(_clamp(rfactor_raw / 100.0) * 5.0, 2)
    opportunity_score = _opportunity_score(pre_score, trigger_score, direction_conf, chop_penalty, is_chase)

    selected_level = breakout_level_name or nearest_level
    selected_dist_pct = breakout_dist_pct if breakout_level_name else dist_pct

    if inferred_direction == "LONG":
        trend_value = _signed_score(rsi_slope_raw + max(0.0, obv_slope_raw) * 0.25, 0.02)
    elif inferred_direction == "SHORT":
        trend_value = _signed_score((-rsi_slope_raw) + max(0.0, -obv_slope_raw) * 0.25, 0.02)
    else:
        trend_value = 0.0

    delivery_pct = _safe_float(nse_quote.get("delivery_pct"))
    bid_ask_ratio = _safe_float(nse_quote.get("bid_ask_ratio"))

    return {
        "rfactor": rfactor,
        "tier": get_rfactor_color_tier(rfactor),
        "rsi": rsi_value,
        "mfi": round(mfi_value, 1),
        "relative_strength": relative_strength,
        "setup_stage": setup_stage,
        "alert_stage": alert_stage,
        "opportunity_score": opportunity_score,
        "rfactor_trend_15m": round(trend_value / 100.0, 2),
        "rfactor_trend_points": [rfactor],
        "pre_score": round(pre_score, 1),
        "trigger_score": round(trigger_score, 1),
        "inferred_direction": inferred_direction,
        "direction_conf": round(direction_conf, 1),
        "compression": round(compression, 1),
        "obv_slope_score": round(obv_slope_score, 1),
        "vol_accel": vol_accel_ratio,
        "rsi_slope_5m": rsi_slope_5m,
        "nearest_level": selected_level,
        "proximity_score": round(proximity_score, 1),
        "dist_pct": round(selected_dist_pct, 2),
        "breakout_levels": breakout_levels,
        "breakout_quality": round(breakout_quality, 1),
        "vwap_acceptance": round(vwap_acceptance, 1),
        "is_chase": is_chase,
        "chase_reason": chase_reason,
        "rfactor_trend_acceleration": 0.0,
        "delivery_pct": round(delivery_pct, 1) if delivery_pct > 0 else stock.get("delivery_pct"),
        "bid_ask_ratio": round(bid_ask_ratio, 2) if bid_ask_ratio > 0 else stock.get("bid_ask_ratio"),
        "day_high": day_high,
        "day_low": day_low,
        "vwap": vwap,
        "change_pct": live_change,
    }


def _trend_snapshots(
    stock: Dict[str, Any],
    df_5m: Optional[pd.DataFrame],
    df_1m: Optional[pd.DataFrame],
    nse_quote: Dict[str, Any],
    nifty_avg: float,
) -> Tuple[List[float], float, float]:
    if df_5m is None or df_5m.empty:
        base = round(_safe_float(stock.get("rfactor")), 2)
        return [base], 0.0, 0.0

    session_df = _session_df(df_5m)
    if session_df is None or len(session_df) < 3:
        base = round(_safe_float(stock.get("rfactor")), 2)
        return [base], 0.0, 0.0

    start_pos = max(3, len(session_df) - 2)
    end_positions = list(range(start_pos, len(session_df) + 1))
    points: List[float] = []
    for end_pos in end_positions:
        hist_end_ts = session_df.index[end_pos - 1]
        hist_df_5m = df_5m[df_5m.index <= hist_end_ts].copy()
        hist_session = _session_df(hist_df_5m)
        if hist_session is None or hist_session.empty:
            continue

        hist_stock = dict(stock)
        hist_stock["ltp"] = _safe_float(hist_session["Close"].dropna().iloc[-1], _safe_float(stock.get("ltp")))
        hist_stock["day_high"] = _safe_float(hist_session["High"].max(), hist_stock["ltp"])
        hist_stock["day_low"] = _safe_float(hist_session["Low"].min(), hist_stock["ltp"])
        hist_vwap = _intraday_vwap(hist_session)
        hist_stock["vwap"] = _safe_float(hist_vwap.iloc[-1], _safe_float(hist_stock.get("vwap")))

        hist_df_1m = None
        if df_1m is not None and not df_1m.empty:
            hist_df_1m = df_1m[df_1m.index <= hist_end_ts].copy()

        snapshot = _evaluate_symbol(
            stock=hist_stock,
            df_5m=hist_df_5m,
            df_1m=hist_df_1m,
            nse_quote=nse_quote,
            nifty_avg=nifty_avg,
            live_change_pct=_safe_float(stock.get("change_pct")),
        )
        points.append(round(_safe_float(snapshot.get("rfactor")), 2))

    if not points:
        base = round(_safe_float(stock.get("rfactor")), 2)
        return [base], 0.0, 0.0

    points[-1] = round(_safe_float(stock.get("rfactor")), 2)
    trend_15m = round(points[-1] - points[-2], 2) if len(points) >= 2 else 0.0
    acceleration = round((points[-1] - points[-2]) - (points[-2] - points[-3]), 2) if len(points) >= 3 else 0.0
    return points, trend_15m, acceleration


def calculate_rfactor_for_all(
    sym_data: Dict[str, Any],
    intraday_data=None,
    data_15min=None,
    nse_data=None,
    data_1min=None,
    data_5min=None,
) -> Dict[str, Any]:
    primary_intraday = data_5min if data_5min is not None else data_15min

    nifty_symbols = [
        "RELIANCE", "HDFCBANK", "INFY", "TCS", "ICICIBANK", "SBIN",
        "BHARTIARTL", "HCLTECH", "WIPRO", "AXISBANK", "KOTAKBANK", "LT",
        "MARUTI", "NTPC", "ONGC", "POWERGRID", "BAJFINANCE", "M&M",
        "TITAN", "ADANIPORTS",
    ]
    nifty_changes = [_safe_float(sym_data.get(symbol, {}).get("change_pct")) for symbol in nifty_symbols if symbol in sym_data]
    nifty_avg = float(np.mean(nifty_changes)) if nifty_changes else 0.0

    for clean_sym, stock in sym_data.items():
        try:
            symbol_ns = f"{clean_sym}.NS"
            df_5m = _get_sym_df(primary_intraday, symbol_ns)
            df_1m = _get_sym_df(data_1min, symbol_ns)
            nse_quote = nse_data.get(clean_sym, {}) if nse_data else {}

            metrics = _evaluate_symbol(
                stock=stock,
                df_5m=df_5m,
                df_1m=df_1m,
                nse_quote=nse_quote,
                nifty_avg=nifty_avg,
                live_change_pct=_safe_float(stock.get("change_pct")),
            )
            stock.update(metrics)

            trend_points, trend_15m, trend_accel = _trend_snapshots(
                stock=stock,
                df_5m=df_5m,
                df_1m=df_1m,
                nse_quote=nse_quote,
                nifty_avg=nifty_avg,
            )
            stock["rfactor_trend_points"] = trend_points
            stock["rfactor_trend_15m"] = trend_15m
            stock["rfactor_trend_acceleration"] = trend_accel

            chop_penalty = _range_cleanliness_penalty(_session_df(df_5m), _intraday_vwap(_session_df(df_5m))) if df_5m is not None else 0.0
            stock["opportunity_score"] = _opportunity_score(
                pre_score=_safe_float(stock.get("pre_score")),
                trigger_score=_safe_float(stock.get("trigger_score")),
                direction_conf=_safe_float(stock.get("direction_conf")),
                chop_penalty=chop_penalty,
                is_chase=bool(stock.get("is_chase")),
            )

            if len(trend_points) >= 2 and not stock.get("is_chase"):
                if stock.get("setup_stage") == "WARMING" and stock["rfactor_trend_15m"] >= 0.18:
                    stock["setup_stage"] = "PRE_SIGNAL"
                    stock["alert_stage"] = "PRE_SIGNAL"
                elif stock.get("setup_stage") == "PRE_SIGNAL" and _safe_float(stock.get("trigger_score")) >= 55:
                    stock["setup_stage"] = "BREAKING"
                    stock["alert_stage"] = "BREAKING"

        except Exception as exc:
            logger.warning("R-Factor v3 failed for %s: %s", clean_sym, exc)
            stock.setdefault("rfactor", 0.0)
            stock.setdefault("tier", "very_weak")
            stock.setdefault("rsi", 50.0)
            stock.setdefault("mfi", 50.0)
            stock.setdefault("relative_strength", 0.0)
            stock.setdefault("setup_stage", "NEUTRAL")
            stock.setdefault("alert_stage", "NEUTRAL")
            stock.setdefault("opportunity_score", 0.0)
            stock.setdefault("rfactor_trend_15m", 0.0)
            stock.setdefault("rfactor_trend_points", [stock.get("rfactor", 0.0)])
            stock.setdefault("rfactor_trend_acceleration", 0.0)
            stock.setdefault("pre_score", 0.0)
            stock.setdefault("trigger_score", 0.0)
            stock.setdefault("inferred_direction", "NEUTRAL")
            stock.setdefault("direction_conf", 0.0)
            stock.setdefault("compression", 0.0)
            stock.setdefault("obv_slope_score", 0.0)
            stock.setdefault("vol_accel", 1.0)
            stock.setdefault("rsi_slope_5m", 0.0)
            stock.setdefault("nearest_level", "")
            stock.setdefault("proximity_score", 0.0)
            stock.setdefault("dist_pct", 999.0)
            stock.setdefault("breakout_levels", {})
            stock.setdefault("breakout_quality", 0.0)
            stock.setdefault("vwap_acceptance", 0.0)
            stock.setdefault("is_chase", False)
            stock.setdefault("chase_reason", "")

    return sym_data


def get_alerts(
    sym_data: Dict[str, Any],
    min_pre_score: float = 55.0,
    min_trigger_score: float = 40.0,
    include_extended: bool = False,
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    for stock in sym_data.values():
        stage = str(stock.get("alert_stage", "NEUTRAL") or "NEUTRAL")
        if stage == "NEUTRAL":
            continue
        if not include_extended and stage == "EXTENDED":
            continue
        if _safe_float(stock.get("pre_score")) < min_pre_score and _safe_float(stock.get("trigger_score")) < min_trigger_score:
            continue
        alerts.append(stock)

    alerts.sort(
        key=lambda item: (
            _safe_float(item.get("opportunity_score")),
            _safe_float(item.get("rfactor")),
            _safe_float(item.get("direction_conf")),
        ),
        reverse=True,
    )
    return alerts


def get_dashboard_rows(sym_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stock in sym_data.values():
        rows.append({
            "symbol": stock.get("symbol"),
            "rfactor": stock.get("rfactor", 0.0),
            "tier": stock.get("tier", "very_weak"),
            "rsi": stock.get("rsi", 50.0),
            "mfi": stock.get("mfi", 50.0),
            "relative_strength": stock.get("relative_strength", 0.0),
            "setup_stage": stock.get("setup_stage", "NEUTRAL"),
            "alert_stage": stock.get("alert_stage", "NEUTRAL"),
            "opportunity_score": stock.get("opportunity_score", 0.0),
            "rfactor_trend_15m": stock.get("rfactor_trend_15m", 0.0),
            "rfactor_trend_points": stock.get("rfactor_trend_points", [stock.get("rfactor", 0.0)]),
            "pre_score": stock.get("pre_score", 0.0),
            "trigger_score": stock.get("trigger_score", 0.0),
            "inferred_direction": stock.get("inferred_direction", "NEUTRAL"),
            "direction_conf": stock.get("direction_conf", 0.0),
            "compression": stock.get("compression", 0.0),
            "vol_accel": stock.get("vol_accel", 1.0),
            "nearest_level": stock.get("nearest_level", ""),
            "dist_pct": stock.get("dist_pct", 999.0),
            "breakout_quality": stock.get("breakout_quality", 0.0),
            "vwap_acceptance": stock.get("vwap_acceptance", 0.0),
            "is_chase": stock.get("is_chase", False),
            "chase_reason": stock.get("chase_reason", ""),
        })

    rows.sort(
        key=lambda item: (
            _safe_float(item.get("opportunity_score")),
            _safe_float(item.get("rfactor")),
        ),
        reverse=True,
    )
    return rows
