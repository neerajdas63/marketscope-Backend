# intraday_boost.py — Measures intraday acceleration / momentum burst for each stock

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("intraday_boost")

_WEIGHTS = {
    "relative_volume_burst": 0.30,
    "price_velocity_burst": 0.25,
    "range_expansion_quality": 0.20,
    "directional_efficiency": 0.15,
    "institutional_hint": 0.10,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if np.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _scale(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return _clamp((value - lower) / (upper - lower))


def _get_sym_df(data: Any, symbol: str):
    """Extract a single-symbol DataFrame from a flat or MultiIndex market data result."""
    try:
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            if isinstance(data.columns, pd.MultiIndex):
                if symbol in data.columns.get_level_values(0):
                    df = data[symbol]
                    return df if not df.empty else None
                return None
            return data if not data.empty else None
        return None
    except Exception:
        return None


def normalize_intraday_frame(raw_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Normalize intraday candles into a common schema independent of source."""
    if raw_df is None or raw_df.empty:
        return None

    column_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "vwap": "VWAP",
        "avgprice": "VWAP",
        "buyqty": "BuyQty",
        "totbuyquan": "BuyQty",
        "sellqty": "SellQty",
        "totsellquan": "SellQty",
        "oi": "OI",
        "openinterest": "OI",
        "opninterest": "OI",
    }

    df = raw_df.copy()
    renamed = {}
    for column in df.columns:
        normalized = column_map.get(str(column).strip().lower())
        if normalized:
            renamed[column] = normalized
    df = df.rename(columns=renamed)

    if "Close" not in df.columns:
        return None
    for required in ("Open", "High", "Low", "Volume"):
        if required not in df.columns:
            if required == "Volume":
                df[required] = 0.0
            else:
                df[required] = df["Close"]

    keep = [column for column in ("Open", "High", "Low", "Close", "Volume", "VWAP", "BuyQty", "SellQty", "OI") if column in df.columns]
    df = df[keep].copy()
    df = df.sort_index()
    numeric_cols = [column for column in df.columns if column in keep]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Close"])
    return df if not df.empty else None


def normalize_daily_frame(raw_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if raw_df is None or raw_df.empty:
        return None
    df = normalize_intraday_frame(raw_df)
    if df is None or df.empty:
        return None
    return df


def extract_smartapi_quote_features(stock: Dict[str, Any]) -> Dict[str, float]:
    """Map quote/depth fields into a normalized schema. These are heuristic proxies only."""
    return {
        "ltp": _safe_float(stock.get("ltp")),
        "change_pct": _safe_float(stock.get("change_pct")),
        "day_high": _safe_float(stock.get("day_high"), _safe_float(stock.get("ltp"))),
        "day_low": _safe_float(stock.get("day_low"), _safe_float(stock.get("ltp"))),
        "day_open": _safe_float(stock.get("day_open"), _safe_float(stock.get("ltp"))),
        "vwap": _safe_float(stock.get("vwap")),
        "volume_ratio": _safe_float(stock.get("volume_ratio"), 1.0),
        "delivery_pct": _safe_float(stock.get("delivery_pct")),
        "bid_ask_ratio": _safe_float(stock.get("bid_ask_ratio"), 1.0),
        "bid_qty": _safe_float(stock.get("bid_qty")),
        "ask_qty": _safe_float(stock.get("ask_qty")),
        "oi": _safe_float(stock.get("oi") or stock.get("open_interest") or stock.get("opnInterest")),
    }


def _compute_intraday_atr(df: pd.DataFrame, window: int = 14) -> Tuple[float, float]:
    high = pd.Series(df["High"], dtype="float64")
    low = pd.Series(df["Low"], dtype="float64")
    close = pd.Series(df["Close"], dtype="float64")
    prev_close = close.shift(1)
    true_range = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1).dropna()
    if true_range.empty:
        return 0.0, 0.0
    atr_abs = _safe_float(true_range.tail(window).mean())
    atr_pct = (atr_abs / max(_safe_float(close.iloc[-1]), 1e-6)) * 100.0 if len(close) else 0.0
    return atr_abs, atr_pct


def _compute_daily_context(daily_df: Optional[pd.DataFrame], quote: Dict[str, float]) -> Dict[str, float]:
    if daily_df is None or daily_df.empty or len(daily_df) < 3:
        ltp = max(quote.get("ltp", 0.0), 1e-6)
        proxy_range_pct = ((quote.get("day_high", ltp) - quote.get("day_low", ltp)) / ltp) * 100.0 if ltp > 0 else 0.0
        return {
            "daily_atr_pct": max(proxy_range_pct, 0.8),
            "avg_daily_range_pct": max(proxy_range_pct, 1.0),
        }

    daily_close = pd.Series(daily_df["Close"], dtype="float64").dropna()
    daily_high = pd.Series(daily_df["High"], dtype="float64").dropna()
    daily_low = pd.Series(daily_df["Low"], dtype="float64").dropna()
    prev_close = daily_close.shift(1)
    true_range = pd.concat([
        (daily_high - daily_low),
        (daily_high - prev_close).abs(),
        (daily_low - prev_close).abs(),
    ], axis=1).max(axis=1).dropna()
    atr_abs = _safe_float(true_range.tail(10).mean())
    last_close = max(_safe_float(daily_close.iloc[-1]), 1e-6)
    avg_daily_range_pct = _safe_float((((daily_high - daily_low) / daily_close.replace(0.0, np.nan)) * 100.0).dropna().tail(10).mean(), 1.2)
    return {
        "daily_atr_pct": (atr_abs / last_close) * 100.0 if atr_abs > 0 else max(avg_daily_range_pct, 1.0),
        "avg_daily_range_pct": max(avg_daily_range_pct, 1.0),
    }


def _compute_relative_volume_burst(df: pd.DataFrame, quote: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    volume = pd.Series(df["Volume"], dtype="float64").fillna(0.0)
    if len(volume) < 2:
        proxy = _scale(quote.get("volume_ratio", 1.0), 1.0, 2.8)
        return proxy, {"recent_avg": 0.0, "baseline_avg": 0.0, "ratio": quote.get("volume_ratio", 1.0), "zscore": 0.0}

    recent_window = min(3, len(volume))
    recent_avg = _safe_float(volume.tail(recent_window).mean())
    baseline = volume.iloc[:-recent_window].tail(24)
    if baseline.empty:
        baseline = volume.iloc[:-recent_window]
    baseline_avg = _safe_float(baseline.mean(), recent_avg)
    baseline_std = _safe_float(baseline.std(ddof=0), 0.0)
    ratio = recent_avg / max(baseline_avg, 1e-6)
    zscore = (recent_avg - baseline_avg) / baseline_std if baseline_std > 0 else 0.0
    ratio_score = _scale(ratio, 1.05, 2.8)
    zscore_score = _scale(zscore, 0.2, 2.2)
    score = 0.7 * ratio_score + 0.3 * zscore_score
    return score, {
        "recent_avg": round(recent_avg, 2),
        "baseline_avg": round(baseline_avg, 2),
        "ratio": round(ratio, 2),
        "zscore": round(zscore, 2),
    }


def _compute_price_velocity_burst(df: pd.DataFrame, quote: Dict[str, float], daily_context: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    close = pd.Series(df["Close"], dtype="float64").dropna()
    if close.empty:
        proxy_move = abs(quote.get("change_pct", 0.0))
        denom = max(daily_context.get("daily_atr_pct", 1.0) * 0.35, 0.3)
        normalized = proxy_move / denom
        return _scale(normalized, 0.8, 2.8), {"move_pct": round(proxy_move, 2), "normalized": round(normalized, 2)}

    lookback = min(6, len(close) - 1) if len(close) >= 2 else 0
    if lookback <= 0:
        return 0.0, {"move_pct": 0.0, "normalized": 0.0}
    move_pct = abs((close.iloc[-1] - close.iloc[-(lookback + 1)]) / max(close.iloc[-(lookback + 1)], 1e-6) * 100.0)
    _, intraday_atr_pct = _compute_intraday_atr(df.tail(max(lookback + 6, 8)))
    denom = max(intraday_atr_pct * max(1.0, lookback / 3.0), daily_context.get("daily_atr_pct", 1.0) * 0.30, 0.2)
    normalized = move_pct / denom
    return _scale(normalized, 0.85, 2.75), {
        "move_pct": round(move_pct, 2),
        "intraday_atr_pct": round(intraday_atr_pct, 2),
        "normalized": round(normalized, 2),
    }


def _compute_range_expansion_quality(df: pd.DataFrame, quote: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    high = pd.Series(df["High"], dtype="float64")
    low = pd.Series(df["Low"], dtype="float64")
    open_ = pd.Series(df["Open"], dtype="float64")
    close = pd.Series(df["Close"], dtype="float64")
    true_range = (high - low).abs().replace(0.0, np.nan)
    body = (close - open_).abs()

    if true_range.dropna().empty or len(close) < 2:
        ltp = quote.get("ltp", 0.0)
        day_high = quote.get("day_high", ltp)
        day_low = quote.get("day_low", ltp)
        day_range_pct = ((day_high - day_low) / max(ltp, 1e-6)) * 100.0 if ltp > 0 else 0.0
        move_eff = abs(quote.get("change_pct", 0.0)) / max(day_range_pct, 0.25)
        score = 0.6 * _scale(day_range_pct, 0.8, 4.0) + 0.4 * _scale(move_eff, 0.35, 1.15)
        return score, {"expansion_ratio": round(day_range_pct, 2), "body_ratio": round(move_eff, 2), "breakout_efficiency": round(move_eff, 2)}

    recent_tr = _safe_float(true_range.tail(3).mean())
    baseline_tr = _safe_float(true_range.iloc[:-3].tail(18).mean(), recent_tr)
    expansion_ratio = recent_tr / max(baseline_tr, 1e-6)
    body_ratio = _safe_float((body.tail(3) / true_range.tail(3)).replace([np.inf, -np.inf], np.nan).dropna().mean())
    net_move = abs(_safe_float(close.iloc[-1]) - _safe_float(close.iloc[-4])) if len(close) >= 4 else abs(_safe_float(close.iloc[-1]) - _safe_float(close.iloc[0]))
    path = _safe_float(true_range.tail(3).sum())
    breakout_efficiency = net_move / max(path, 1e-6)
    score = (
        0.45 * _scale(expansion_ratio, 1.0, 2.4)
        + 0.35 * _scale(body_ratio, 0.30, 0.85)
        + 0.20 * _scale(breakout_efficiency, 0.18, 0.80)
    )
    return score, {
        "expansion_ratio": round(expansion_ratio, 2),
        "body_ratio": round(body_ratio, 2),
        "breakout_efficiency": round(breakout_efficiency, 2),
    }


def compute_directional_efficiency(df: pd.DataFrame, quote: Dict[str, float]) -> Tuple[float, str, Dict[str, float]]:
    close = pd.Series(df["Close"], dtype="float64").dropna()
    if len(close) < 2:
        change_pct = quote.get("change_pct", 0.0)
        direction = "up" if change_pct > 0.2 else "down" if change_pct < -0.2 else "flat"
        proxy_eff = _scale(abs(change_pct), 0.25, 2.0)
        return proxy_eff, direction, {"efficiency": round(proxy_eff, 2), "persistence": round(proxy_eff, 2)}

    lookback = min(6, len(close) - 1)
    recent = close.tail(lookback + 1)
    diffs = recent.diff().dropna()
    net_move = _safe_float(recent.iloc[-1] - recent.iloc[0])
    path = _safe_float(diffs.abs().sum())
    efficiency = abs(net_move) / max(path, 1e-6)
    direction_sign = 1 if net_move > 0 else -1 if net_move < 0 else 0
    direction = "up" if direction_sign > 0 else "down" if direction_sign < 0 else "flat"
    persistence = float(((diffs * direction_sign) > 0).mean()) if direction_sign != 0 and not diffs.empty else 0.0
    score = 0.75 * _scale(efficiency, 0.22, 0.86) + 0.25 * _scale(persistence, 0.45, 0.90)
    return score, direction, {"efficiency": round(efficiency, 2), "persistence": round(persistence, 2)}


def _intraday_vwap(df: pd.DataFrame, quote: Dict[str, float]) -> float:
    if "VWAP" in df.columns and not pd.Series(df["VWAP"]).dropna().empty:
        return _safe_float(pd.Series(df["VWAP"]).dropna().iloc[-1])
    volume = pd.Series(df["Volume"], dtype="float64").fillna(0.0)
    if volume.sum() <= 0:
        return quote.get("vwap", 0.0)
    typical = (pd.Series(df["High"], dtype="float64") + pd.Series(df["Low"], dtype="float64") + pd.Series(df["Close"], dtype="float64")) / 3.0
    vwap_series = (typical * volume).cumsum() / volume.cumsum().replace(0.0, np.nan)
    return _safe_float(vwap_series.dropna().iloc[-1], quote.get("vwap", 0.0))


def compute_institutional_hint(df: Optional[pd.DataFrame], quote: Dict[str, float], volume_score: float, impulse_score: float, direction: str) -> Tuple[float, Dict[str, float]]:
    """Heuristic only: a soft participation hint, not proof of institutional activity."""
    vwap_value = _intraday_vwap(df, quote) if df is not None and not df.empty else quote.get("vwap", 0.0)
    ltp = quote.get("ltp", 0.0)
    vwap_hold_score = 0.5
    if vwap_value > 0 and ltp > 0 and direction != "flat":
        if (direction == "up" and ltp >= vwap_value) or (direction == "down" and ltp <= vwap_value):
            vwap_hold_score = 0.8
        else:
            vwap_hold_score = 0.25

    bid_qty = quote.get("bid_qty", 0.0)
    ask_qty = quote.get("ask_qty", 0.0)
    bid_ask_ratio = quote.get("bid_ask_ratio", 1.0)
    if bid_qty > 0 or ask_qty > 0:
        depth_balance = (bid_qty - ask_qty) / max(bid_qty + ask_qty, 1.0)
        depth_score = _scale(abs(depth_balance), 0.05, 0.45)
    else:
        depth_score = _scale(abs(bid_ask_ratio - 1.0), 0.08, 0.65)

    delivery_pct = quote.get("delivery_pct", 0.0)
    delivery_score = _scale(delivery_pct, 25.0, 65.0) if delivery_pct > 0 else 0.35

    oi_value = quote.get("oi", 0.0)
    oi_score = _scale(oi_value, 1.0, max(oi_value * 1.5, 2.0)) if oi_value > 0 else 0.35

    score = (
        0.35 * volume_score
        + 0.25 * impulse_score
        + 0.20 * vwap_hold_score
        + 0.10 * depth_score
        + 0.05 * delivery_score
        + 0.05 * oi_score
    )
    return score, {
        "vwap_hold": round(vwap_hold_score, 2),
        "depth_imbalance": round(depth_score, 2),
        "delivery_support": round(delivery_score, 2),
        "oi_support": round(oi_score, 2),
    }


def _compute_quote_proxy_components(stock: Dict[str, Any], daily_context: Dict[str, float]) -> Tuple[Dict[str, float], str, float]:
    quote = extract_smartapi_quote_features(stock)
    ltp = quote.get("ltp", 0.0)
    day_high = quote.get("day_high", ltp)
    day_low = quote.get("day_low", ltp)
    change_pct = quote.get("change_pct", 0.0)
    day_range_pct = ((day_high - day_low) / max(ltp, 1e-6)) * 100.0 if ltp > 0 else 0.0

    direction = "up" if change_pct > 0.2 else "down" if change_pct < -0.2 else "flat"
    volume_score = _scale(quote.get("volume_ratio", 1.0), 1.0, 2.8)

    velocity_norm = abs(change_pct) / max(day_range_pct * 0.60, daily_context.get("daily_atr_pct", 1.0) * 0.30, 0.30)
    velocity_score = _scale(velocity_norm, 0.8, 2.8)

    impulse_eff = abs(change_pct) / max(day_range_pct, 0.25)
    range_score = 0.55 * _scale(day_range_pct, 0.8, 4.2) + 0.45 * _scale(impulse_eff, 0.35, 1.20)

    efficiency_score = 0.65 * _scale(impulse_eff, 0.35, 1.10)
    vwap = quote.get("vwap", 0.0)
    if vwap > 0 and ltp > 0 and direction != "flat":
        aligned = (direction == "up" and ltp >= vwap) or (direction == "down" and ltp <= vwap)
        efficiency_score += 0.35 * (0.9 if aligned else 0.25)
    else:
        efficiency_score += 0.35 * 0.5
    efficiency_score = _clamp(efficiency_score)

    institutional_score, _ = compute_institutional_hint(None, quote, volume_score, range_score, direction)
    components = {
        "relative_volume_burst": volume_score,
        "price_velocity_burst": velocity_score,
        "range_expansion_quality": range_score,
        "directional_efficiency": efficiency_score,
        "institutional_hint": institutional_score,
    }
    proxy_raw = sum(components[name] * _WEIGHTS[name] for name in _WEIGHTS)
    return components, direction, proxy_raw


def _compute_candle_components(stock: Dict[str, Any], intraday_df: pd.DataFrame, daily_context: Dict[str, float]) -> Tuple[Dict[str, float], str, Dict[str, Dict[str, float]]]:
    quote = extract_smartapi_quote_features(stock)
    volume_score, volume_diag = _compute_relative_volume_burst(intraday_df, quote)
    velocity_score, velocity_diag = _compute_price_velocity_burst(intraday_df, quote, daily_context)
    impulse_score, impulse_diag = _compute_range_expansion_quality(intraday_df, quote)
    efficiency_score, direction, eff_diag = compute_directional_efficiency(intraday_df, quote)
    institutional_score, inst_diag = compute_institutional_hint(intraday_df, quote, volume_score, impulse_score, direction)

    return {
        "relative_volume_burst": volume_score,
        "price_velocity_burst": velocity_score,
        "range_expansion_quality": impulse_score,
        "directional_efficiency": efficiency_score,
        "institutional_hint": institutional_score,
    }, direction, {
        "relative_volume_burst": volume_diag,
        "price_velocity_burst": velocity_diag,
        "range_expansion_quality": impulse_diag,
        "directional_efficiency": eff_diag,
        "institutional_hint": inst_diag,
    }


def _reduced_confidence(len_bars: int) -> float:
    if len_bars <= 0:
        return 0.0
    if len_bars < 6:
        return _scale(float(len_bars), 2.0, 6.0) * 0.45 + 0.25
    return _scale(float(len_bars), 6.0, 18.0) * 0.35 + 0.65


def _blend_components(primary: Dict[str, float], fallback: Dict[str, float], confidence: float) -> Dict[str, float]:
    return {
        key: round((_safe_float(primary.get(key)) * confidence) + (_safe_float(fallback.get(key)) * (1.0 - confidence)), 4)
        for key in _WEIGHTS
    }


def calculate_intraday_boost(
    sym_data: Dict[str, Any],
    intraday_data,
    daily_data,
) -> Dict[str, Any]:
    """Add a 0–5 intraday burst score to every stock in sym_data."""
    for clean_sym, stock in sym_data.items():
        try:
            symbol_ns = clean_sym + ".NS"
            intraday_df = normalize_intraday_frame(_get_sym_df(intraday_data, symbol_ns))
            daily_df = normalize_daily_frame(_get_sym_df(daily_data, symbol_ns))
            quote = extract_smartapi_quote_features(stock)
            daily_context = _compute_daily_context(daily_df, quote)

            proxy_components, proxy_direction, proxy_raw = _compute_quote_proxy_components(stock, daily_context)
            diagnostics: Dict[str, Any] = {
                "data_mode": "quote_proxy",
                "confidence": 0.0,
                "daily_context": {k: round(v, 2) for k, v in daily_context.items()},
                "component_details": {},
            }

            if intraday_df is not None and not intraday_df.empty:
                candle_components, candle_direction, component_details = _compute_candle_components(stock, intraday_df, daily_context)
                confidence = _reduced_confidence(len(intraday_df))
                blended_components = _blend_components(candle_components, proxy_components, confidence)
                raw_score = sum(blended_components[name] * _WEIGHTS[name] for name in _WEIGHTS)
                boost_direction = candle_direction if candle_direction != "flat" else proxy_direction
                diagnostics.update({
                    "data_mode": "candles_blended" if confidence < 0.99 else "candles",
                    "confidence": round(confidence, 2),
                    "component_details": component_details,
                })
            else:
                blended_components = proxy_components
                raw_score = proxy_raw
                boost_direction = proxy_direction
                diagnostics["confidence"] = 0.35

            boost_score = round(_clamp(raw_score) * 5.0, 2)
            stock["boost_score"] = boost_score
            stock["boost_direction"] = boost_direction
            stock["institutional_hint_score"] = round(_safe_float(blended_components.get("institutional_hint")) * 100.0, 1)
            stock["boost_components"] = {
                "relative_volume_burst": round(_safe_float(blended_components.get("relative_volume_burst")) * 100.0, 1),
                "price_velocity_burst": round(_safe_float(blended_components.get("price_velocity_burst")) * 100.0, 1),
                "range_expansion_quality": round(_safe_float(blended_components.get("range_expansion_quality")) * 100.0, 1),
                "directional_efficiency": round(_safe_float(blended_components.get("directional_efficiency")) * 100.0, 1),
                "institutional_hint": round(_safe_float(blended_components.get("institutional_hint")) * 100.0, 1),
                "confidence": diagnostics.get("confidence", 0.0),
                "data_mode": diagnostics.get("data_mode", "quote_proxy"),
                "details": diagnostics.get("component_details", {}),
                "daily_context": diagnostics.get("daily_context", {}),
            }

        except Exception as exc:
            logger.warning("boost_score failed for %s: %s", clean_sym, exc)
            stock["boost_score"] = 0.0
            stock["boost_direction"] = "flat"
            stock["institutional_hint_score"] = 0.0
            stock["boost_components"] = {
                "relative_volume_burst": 0.0,
                "price_velocity_burst": 0.0,
                "range_expansion_quality": 0.0,
                "directional_efficiency": 0.0,
                "institutional_hint": 0.0,
                "confidence": 0.0,
                "data_mode": "error",
                "details": {},
                "daily_context": {},
            }

    return sym_data
