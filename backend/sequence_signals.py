import logging
import threading
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from stocks import SCANNER_STOCKS

logger = logging.getLogger("sequence_signals")

IST = pytz.timezone("Asia/Kolkata")
SUPPORTED_TIMEFRAMES = ("3m", "5m", "15m")
REFRESH_COOLDOWN_SECONDS = 300
MAX_KEEP = 12
SWING_LEN = 4
ATR_LEN = 14
DISPLACEMENT_ATR_MULT = 1.0
RETEST_BARS = 10
MIN_OB_SCORE_SIGNAL = 3
OB_ACTIVE_BARS = 12
MAX_SIGNALS_PER_DAY = 12

_sequence_cache: Dict[str, Any] = {
    "cache_key": "",
    "computed_at": 0.0,
    "signals": [],
    "source": "yfinance",
    "error": "",
}
_lock = threading.Lock()


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


def _chunked(items: Sequence[str], size: int) -> List[List[str]]:
    return [list(items[index:index + size]) for index in range(0, len(items), size)]


def _normalize_intraday_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    try:
        if df is None or df.empty:
            return None
        clean = df.copy().sort_index()
        if clean.index.tz is None:
            clean.index = clean.index.tz_localize("UTC").tz_convert(IST)
        else:
            clean.index = clean.index.tz_convert(IST)
        clean.index = clean.index.tz_localize(None)
        required = ["Open", "High", "Low", "Close", "Volume"]
        for column in required:
            if column not in clean.columns:
                return None
        clean = clean[required].dropna(subset=["Open", "High", "Low", "Close"])
        clean["Volume"] = pd.to_numeric(clean["Volume"], errors="coerce").fillna(0.0)
        clean = clean.between_time("09:15", "15:30")
        return clean if not clean.empty else None
    except Exception:
        return None


def _extract_symbol_df(raw: Optional[pd.DataFrame], symbol: str) -> Optional[pd.DataFrame]:
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


def _download_intraday_batches(symbols_ns: Sequence[str], period: str, interval: str) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    for batch in _chunked(list(symbols_ns), 25):
        try:
            raw = yf.download(
                tickers=" ".join(batch),
                period=period,
                interval=interval,
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=False,
                timeout=20,
            )
        except Exception as exc:
            logger.warning("Sequence Signals download failed for %s %s batch %s: %s", period, interval, batch, exc)
            time.sleep(1)
            continue
        for symbol in batch:
            symbol_df = _extract_symbol_df(raw, symbol)
            if symbol_df is not None and not symbol_df.empty:
                results[symbol] = symbol_df
        time.sleep(1)
    return results


def _resample_intraday(df: Optional[pd.DataFrame], rule: str) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    frames: List[pd.DataFrame] = []
    for session_date in sorted(set(df.index.date)):
        session_df = df[df.index.date == session_date].copy()
        if session_df.empty:
            continue
        resampled = session_df.resample(
            rule,
            origin="start_day",
            offset="15min",
            label="right",
            closed="right",
        ).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"])
        resampled = resampled.between_time("09:15", "15:30")
        if not resampled.empty:
            frames.append(resampled)
    if not frames:
        return None
    combined = pd.concat(frames).sort_index()
    return combined if not combined.empty else None


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = pd.Series(df["High"], dtype="float64")
    low = pd.Series(df["Low"], dtype="float64")
    close = pd.Series(df["Close"], dtype="float64")
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    return _true_range(df).rolling(length, min_periods=1).mean()


def _with_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    typical_price = (enriched["High"] + enriched["Low"] + enriched["Close"]) / 3.0
    frames: List[pd.DataFrame] = []
    for session_date in sorted(set(enriched.index.date)):
        session_df = enriched[enriched.index.date == session_date].copy()
        if session_df.empty:
            continue
        tp = typical_price.loc[session_df.index]
        cum_tpv = (tp * session_df["Volume"]).cumsum()
        cum_vol = session_df["Volume"].cumsum().replace(0.0, np.nan)
        cum_tpv2 = ((tp * tp) * session_df["Volume"]).cumsum()
        vwap = cum_tpv / cum_vol
        variance = np.maximum((cum_tpv2 / cum_vol) - (vwap * vwap), 0.0)
        session_df["VWAP"] = vwap.ffill().fillna(0.0)
        session_df["UpperSD1"] = session_df["VWAP"] + np.sqrt(variance).fillna(0.0)
        session_df["LowerSD1"] = session_df["VWAP"] - np.sqrt(variance).fillna(0.0)
        frames.append(session_df)
    return pd.concat(frames).sort_index() if frames else enriched


def _compute_fvg_features(df: pd.DataFrame) -> pd.DataFrame:
    atr_fvg = _atr(df, 10)
    displacement = (df["Close"] - df["Open"]).abs() > (atr_fvg * 0.5)
    bull_fvg = (df["Low"] > df["High"].shift(2)) & displacement
    bear_fvg = (df["High"] < df["Low"].shift(2)) & displacement
    return pd.DataFrame(
        {
            "bull_fvg": bull_fvg.fillna(False),
            "bear_fvg": bear_fvg.fillna(False),
            "cont2_bull": (bull_fvg & bull_fvg.shift(1).fillna(False) & ~bull_fvg.shift(2).fillna(False)).fillna(False),
            "cont3_bull": (bull_fvg & bull_fvg.shift(1).fillna(False) & bull_fvg.shift(2).fillna(False)).fillna(False),
            "cont2_bear": (bear_fvg & bear_fvg.shift(1).fillna(False) & ~bear_fvg.shift(2).fillna(False)).fillna(False),
            "cont3_bear": (bear_fvg & bear_fvg.shift(1).fillna(False) & bear_fvg.shift(2).fillna(False)).fillna(False),
        },
        index=df.index,
    )


def _asof_bool_series(htf_series: pd.Series, lower_index: pd.Index) -> pd.Series:
    if htf_series.empty:
        return pd.Series(False, index=lower_index)
    left = pd.DataFrame({"ts": pd.Index(lower_index)}).sort_values("ts")
    right = pd.DataFrame({"ts": pd.Index(htf_series.index), "value": htf_series.astype(bool).values}).sort_values("ts")
    merged = pd.merge_asof(left, right, on="ts", direction="backward")
    return pd.Series(merged["value"].fillna(False).astype(bool).values, index=lower_index)


def _build_htf_features(lower_df: pd.DataFrame, htf_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if htf_df is None or htf_df.empty:
        return pd.DataFrame(
            {
                "bull_bias": False,
                "bear_bias": False,
                "bull_pattern": False,
                "bear_pattern": False,
            },
            index=lower_df.index,
        )
    close = pd.Series(htf_df["Close"], dtype="float64")
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    bull_bias = ((close > ema20) & (ema20 > ema50)).shift(1).fillna(False)
    bear_bias = ((close < ema20) & (ema20 < ema50)).shift(1).fillna(False)
    htf_fvg = _compute_fvg_features(htf_df)
    bull_pattern = (htf_fvg["cont2_bull"] | htf_fvg["cont3_bull"]).fillna(False)
    bear_pattern = (htf_fvg["cont2_bear"] | htf_fvg["cont3_bear"]).fillna(False)
    return pd.DataFrame(
        {
            "bull_bias": _asof_bool_series(bull_bias, lower_df.index),
            "bear_bias": _asof_bool_series(bear_bias, lower_df.index),
            "bull_pattern": _asof_bool_series(bull_pattern, lower_df.index),
            "bear_pattern": _asof_bool_series(bear_pattern, lower_df.index),
        },
        index=lower_df.index,
    )


def _pivot_value(values: pd.Series, index: int, swing_len: int, is_high: bool) -> Optional[float]:
    if index < swing_len * 2:
        return None
    center_index = index - swing_len
    window = values.iloc[index - (swing_len * 2): index + 1]
    center_value = _safe_float(values.iloc[center_index])
    if is_high and center_value >= _safe_float(window.max()):
        return center_value
    if not is_high and center_value <= _safe_float(window.min(), center_value):
        return center_value
    return None


def _find_last_opposite(df: pd.DataFrame, index: int, bullish: bool) -> Optional[int]:
    upper_bound = min(6, index)
    for lookback in range(1, upper_bound + 1):
        candidate = index - lookback
        if bullish and _safe_float(df["Close"].iloc[candidate]) < _safe_float(df["Open"].iloc[candidate]):
            return candidate
        if not bullish and _safe_float(df["Close"].iloc[candidate]) > _safe_float(df["Open"].iloc[candidate]):
            return candidate
    return None


def _score_text(score: int) -> str:
    if score >= 5:
        return f"Strong {score}/6"
    if score >= 3:
        return f"Okay {score}/6"
    return f"Weak {score}/6"


def _make_signal(
    symbol: str,
    timeframe: str,
    timestamp: datetime,
    side: str,
    signal_type: str,
    price: float,
    ob_score: int,
    mtf_label: str,
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "signal_type": signal_type,
        "signal": f"{side} {signal_type}",
        "signal_time": timestamp.strftime("%H:%M"),
        "signal_timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "price": round(price, 2),
        "ob_score": int(ob_score),
        "ob_score_text": f"{ob_score}/6",
        "ob_score_label": _score_text(ob_score),
        "mtf_label": mtf_label,
        "source": "yfinance",
    }


def _process_timeframe(
    symbol: str,
    timeframe: str,
    lower_df: Optional[pd.DataFrame],
    htf_df: Optional[pd.DataFrame],
    target_date: date,
) -> List[Dict[str, Any]]:
    if lower_df is None or lower_df.empty:
        return []

    lower_df = _with_session_vwap(lower_df)
    fvg_features = _compute_fvg_features(lower_df)
    htf_features = _build_htf_features(lower_df, htf_df)
    atr_ob = _atr(lower_df, ATR_LEN)

    signals: List[Dict[str, Any]] = []
    order_blocks: List[Dict[str, Any]] = []
    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    current_day: Optional[date] = None
    signal_count = 0

    timeframes_labels = {
        "3m": "3m+5m",
        "5m": "5m+15m",
        "15m": "15m+60m",
    }

    for index in range(len(lower_df)):
        timestamp = lower_df.index[index]
        bar_day = timestamp.date()
        if current_day != bar_day:
            current_day = bar_day
            signal_count = 0

        high = _safe_float(lower_df["High"].iloc[index])
        low = _safe_float(lower_df["Low"].iloc[index])
        open_ = _safe_float(lower_df["Open"].iloc[index])
        close = _safe_float(lower_df["Close"].iloc[index])
        prev_close = _safe_float(lower_df["Close"].iloc[index - 1], close) if index > 0 else close
        atr_now = _safe_float(atr_ob.iloc[index], 0.0)

        pivot_high = _pivot_value(lower_df["High"], index, SWING_LEN, True)
        pivot_low = _pivot_value(lower_df["Low"], index, SWING_LEN, False)
        if pivot_high is not None:
            last_swing_high = pivot_high
        if pivot_low is not None:
            last_swing_low = pivot_low

        bull_bos = last_swing_high is not None and prev_close <= last_swing_high < close
        bear_bos = last_swing_low is not None and prev_close >= last_swing_low > close
        bull_disp = close > open_ and (close - open_) > (atr_now * DISPLACEMENT_ATR_MULT)
        bear_disp = close < open_ and (open_ - close) > (atr_now * DISPLACEMENT_ATR_MULT)

        if bull_disp:
            opposite_index = _find_last_opposite(lower_df, index, True)
            if opposite_index is not None:
                top = _safe_float(lower_df["Open"].iloc[opposite_index])
                bottom = _safe_float(lower_df["Low"].iloc[opposite_index])
                duplicate = False
                for block in order_blocks:
                    near_same = abs(_safe_float(block["top"]) - top) <= (atr_now * 0.20) and abs(_safe_float(block["bottom"]) - bottom) <= (atr_now * 0.20)
                    if block["bias"] == 1 and not block["mitigated"] and near_same:
                        duplicate = True
                        break
                if not duplicate:
                    order_blocks.append(
                        {
                            "top": top,
                            "bottom": bottom,
                            "bias": 1,
                            "confirmed": False,
                            "mitigated": False,
                            "retested": False,
                            "rejected": False,
                            "left_index": index,
                            "score": 1,
                            "did_bull_c2": False,
                            "did_bull_c3": False,
                            "did_bull_mtf": False,
                            "did_bear_c2": False,
                            "did_bear_c3": False,
                            "did_bear_mtf": False,
                        }
                    )
        if bear_disp:
            opposite_index = _find_last_opposite(lower_df, index, False)
            if opposite_index is not None:
                top = _safe_float(lower_df["High"].iloc[opposite_index])
                bottom = _safe_float(lower_df["Open"].iloc[opposite_index])
                duplicate = False
                for block in order_blocks:
                    near_same = abs(_safe_float(block["top"]) - top) <= (atr_now * 0.20) and abs(_safe_float(block["bottom"]) - bottom) <= (atr_now * 0.20)
                    if block["bias"] == -1 and not block["mitigated"] and near_same:
                        duplicate = True
                        break
                if not duplicate:
                    order_blocks.append(
                        {
                            "top": top,
                            "bottom": bottom,
                            "bias": -1,
                            "confirmed": False,
                            "mitigated": False,
                            "retested": False,
                            "rejected": False,
                            "left_index": index,
                            "score": 1,
                            "did_bull_c2": False,
                            "did_bull_c3": False,
                            "did_bull_mtf": False,
                            "did_bear_c2": False,
                            "did_bear_c3": False,
                            "did_bear_mtf": False,
                        }
                    )

        if len(order_blocks) > MAX_KEEP:
            order_blocks = order_blocks[-MAX_KEEP:]

        best_bull_block: Optional[Dict[str, Any]] = None
        best_bear_block: Optional[Dict[str, Any]] = None
        best_bull_score = 0
        best_bear_score = 0
        bull_bias_now = bool(htf_features["bull_bias"].iloc[index])
        bear_bias_now = bool(htf_features["bear_bias"].iloc[index])

        for block in order_blocks:
            score = 1
            if not block["confirmed"]:
                if block["bias"] == 1 and bull_bos and index >= block["left_index"]:
                    block["confirmed"] = True
                if block["bias"] == -1 and bear_bos and index >= block["left_index"]:
                    block["confirmed"] = True
            if block["confirmed"]:
                score += 1
            if not block["mitigated"]:
                score += 1
            if (block["bias"] == 1 and bull_bias_now) or (block["bias"] == -1 and bear_bias_now):
                score += 1

            touch_bull = block["bias"] == 1 and low <= _safe_float(block["top"]) and high >= _safe_float(block["bottom"]) and index > block["left_index"] and (index - block["left_index"]) <= RETEST_BARS
            touch_bear = block["bias"] == -1 and high >= _safe_float(block["bottom"]) and low <= _safe_float(block["top"]) and index > block["left_index"] and (index - block["left_index"]) <= RETEST_BARS
            if not block["retested"] and (touch_bull or touch_bear):
                block["retested"] = True
            if block["retested"]:
                score += 1

            bull_reject = block["bias"] == 1 and block["retested"] and close > _safe_float(block["top"]) and low >= _safe_float(block["bottom"])
            bear_reject = block["bias"] == -1 and block["retested"] and close < _safe_float(block["bottom"]) and high <= _safe_float(block["top"])
            if not block["rejected"] and (bull_reject or bear_reject):
                block["rejected"] = True
            if block["rejected"]:
                score += 1

            if block["bias"] == 1 and low < _safe_float(block["bottom"]):
                block["mitigated"] = True
            if block["bias"] == -1 and high > _safe_float(block["top"]):
                block["mitigated"] = True

            block["score"] = score
            active_window = (index - block["left_index"]) <= OB_ACTIVE_BARS
            usable = block["confirmed"] and not block["mitigated"] and active_window and score >= MIN_OB_SCORE_SIGNAL

            if usable and block["bias"] == 1 and score >= best_bull_score:
                best_bull_score = score
                best_bull_block = block
            if usable and block["bias"] == -1 and score >= best_bear_score:
                best_bear_score = score
                best_bear_block = block

        if bar_day != target_date or signal_count >= MAX_SIGNALS_PER_DAY:
            continue

        cont2_bull = bool(fvg_features["cont2_bull"].iloc[index])
        cont3_bull = bool(fvg_features["cont3_bull"].iloc[index])
        cont2_bear = bool(fvg_features["cont2_bear"].iloc[index])
        cont3_bear = bool(fvg_features["cont3_bear"].iloc[index])
        valid_bull = bool(htf_features["bull_pattern"].iloc[index])
        valid_bear = bool(htf_features["bear_pattern"].iloc[index])
        new_signals = 0

        if best_bull_block is not None:
            if cont2_bull and not best_bull_block["did_bull_c2"]:
                signals.append(_make_signal(symbol, timeframe, timestamp, "BUY", "C2", close, int(best_bull_block["score"]), ""))
                best_bull_block["did_bull_c2"] = True
                new_signals += 1
            if cont3_bull and not best_bull_block["did_bull_c3"]:
                signals.append(_make_signal(symbol, timeframe, timestamp, "BUY", "C3", close, int(best_bull_block["score"]), ""))
                best_bull_block["did_bull_c3"] = True
                new_signals += 1
            if valid_bull and not best_bull_block["did_bull_mtf"]:
                signals.append(_make_signal(symbol, timeframe, timestamp, "BUY", "MTF", close, int(best_bull_block["score"]), timeframes_labels[timeframe]))
                best_bull_block["did_bull_mtf"] = True
                new_signals += 1

        if best_bear_block is not None:
            if cont2_bear and not best_bear_block["did_bear_c2"]:
                signals.append(_make_signal(symbol, timeframe, timestamp, "SELL", "C2", close, int(best_bear_block["score"]), ""))
                best_bear_block["did_bear_c2"] = True
                new_signals += 1
            if cont3_bear and not best_bear_block["did_bear_c3"]:
                signals.append(_make_signal(symbol, timeframe, timestamp, "SELL", "C3", close, int(best_bear_block["score"]), ""))
                best_bear_block["did_bear_c3"] = True
                new_signals += 1
            if valid_bear and not best_bear_block["did_bear_mtf"]:
                signals.append(_make_signal(symbol, timeframe, timestamp, "SELL", "MTF", close, int(best_bear_block["score"]), timeframes_labels[timeframe]))
                best_bear_block["did_bear_mtf"] = True
                new_signals += 1

        signal_count += new_signals

    return signals


def _build_symbol_frames(
    symbol_ns: str,
    one_minute_data: Dict[str, pd.DataFrame],
    five_minute_data: Dict[str, pd.DataFrame],
) -> Dict[str, Optional[pd.DataFrame]]:
    one_minute_df = one_minute_data.get(symbol_ns)
    five_minute_df = five_minute_data.get(symbol_ns)
    fifteen_minute_df = _resample_intraday(five_minute_df, "15min") if five_minute_df is not None else None
    sixty_minute_df = _resample_intraday(five_minute_df, "60min") if five_minute_df is not None else None
    return {
        "3m": _resample_intraday(one_minute_df, "3min") if one_minute_df is not None else None,
        "5m": five_minute_df,
        "15m": fifteen_minute_df,
        "15m_htf": sixty_minute_df,
    }


def _compute_sequence_signals(symbols: Sequence[str], target_date: date) -> List[Dict[str, Any]]:
    symbols_ns = [symbol if symbol.endswith(".NS") else f"{symbol}.NS" for symbol in symbols]
    one_minute_data = _download_intraday_batches(symbols_ns, period="1d", interval="1m")
    five_minute_data = _download_intraday_batches(symbols_ns, period="5d", interval="5m")

    all_signals: List[Dict[str, Any]] = []
    for symbol_ns in symbols_ns:
        clean_symbol = symbol_ns.replace(".NS", "")
        frames = _build_symbol_frames(symbol_ns, one_minute_data, five_minute_data)
        timeframe_map = {
            "3m": (frames["3m"], frames["5m"]),
            "5m": (frames["5m"], frames["15m"]),
            "15m": (frames["15m"], frames["15m_htf"]),
        }
        for timeframe in SUPPORTED_TIMEFRAMES:
            lower_df, htf_df = timeframe_map[timeframe]
            try:
                all_signals.extend(_process_timeframe(clean_symbol, timeframe, lower_df, htf_df, target_date))
            except Exception as exc:
                logger.warning("Sequence Signals failed for %s %s: %s", clean_symbol, timeframe, exc)

    all_signals.sort(key=lambda item: (item.get("signal_timestamp", ""), item.get("symbol", ""), item.get("timeframe", "")), reverse=True)
    for index, item in enumerate(all_signals, start=1):
        item["rank"] = index
    return all_signals


def get_sequence_signals(
    symbols: Optional[Sequence[str]] = None,
    timeframe: str = "ALL",
    side: str = "ALL",
    signal_type: str = "ALL",
    limit: int = 200,
    session_date: Optional[str] = None,
    market_data_last_updated: str = "",
) -> Dict[str, Any]:
    normalized_timeframe = str(timeframe or "ALL").strip().lower()
    if normalized_timeframe not in {"all", "3m", "5m", "15m"}:
        normalized_timeframe = "all"

    normalized_side = str(side or "ALL").strip().upper()
    if normalized_side not in {"ALL", "BUY", "SELL"}:
        normalized_side = "ALL"

    normalized_signal_type = str(signal_type or "ALL").strip().upper()
    if normalized_signal_type not in {"ALL", "C2", "C3", "MTF"}:
        normalized_signal_type = "ALL"

    target_date = datetime.now(IST).date() if not session_date else datetime.strptime(session_date, "%Y-%m-%d").date()
    symbol_list = list(symbols or [symbol.replace(".NS", "") for symbol in SCANNER_STOCKS])
    cache_key = f"{target_date.isoformat()}|{len(symbol_list)}"

    with _lock:
        cache_fresh = (
            _sequence_cache.get("cache_key") == cache_key
            and (time.time() - _safe_float(_sequence_cache.get("computed_at"))) <= REFRESH_COOLDOWN_SECONDS
            and isinstance(_sequence_cache.get("signals"), list)
        )
        if cache_fresh:
            all_signals = list(_sequence_cache.get("signals") or [])
            cache_error = str(_sequence_cache.get("error") or "")
        else:
            all_signals = []
            cache_error = ""

    if not all_signals:
        try:
            all_signals = _compute_sequence_signals(symbol_list, target_date)
            with _lock:
                _sequence_cache["cache_key"] = cache_key
                _sequence_cache["computed_at"] = time.time()
                _sequence_cache["signals"] = list(all_signals)
                _sequence_cache["error"] = ""
        except Exception as exc:
            logger.error("Sequence Signals refresh failed: %s", exc, exc_info=True)
            cache_error = str(exc)
            with _lock:
                _sequence_cache["error"] = cache_error
            all_signals = []

    filtered = list(all_signals)
    if normalized_timeframe != "all":
        filtered = [item for item in filtered if str(item.get("timeframe", "")).lower() == normalized_timeframe]
    if normalized_side != "ALL":
        filtered = [item for item in filtered if str(item.get("side", "")).upper() == normalized_side]
    if normalized_signal_type != "ALL":
        filtered = [item for item in filtered if str(item.get("signal_type", "")).upper() == normalized_signal_type]
    filtered = filtered[:limit] if limit > 0 else filtered
    for index, item in enumerate(filtered, start=1):
        item["rank"] = index

    timeframe_counts = {
        tf: sum(1 for item in filtered if item.get("timeframe") == tf)
        for tf in SUPPORTED_TIMEFRAMES
    }
    signal_type_counts = {
        key: sum(1 for item in filtered if item.get("signal_type") == key)
        for key in ("C2", "C3", "MTF")
    }
    side_counts = {
        key: sum(1 for item in filtered if item.get("side") == key)
        for key in ("BUY", "SELL")
    }

    return {
        "status": "ready" if filtered or not cache_error else "error",
        "message": cache_error or ("No strategy signals found for the selected filters" if not filtered else ""),
        "source": "yfinance",
        "session_date": target_date.isoformat(),
        "market_data_last_updated": market_data_last_updated,
        "last_updated": datetime.now(IST).strftime("%H:%M:%S"),
        "filters": {
            "timeframe": normalized_timeframe.upper() if normalized_timeframe != "all" else "ALL",
            "side": normalized_side,
            "signal_type": normalized_signal_type,
            "limit": limit,
        },
        "summary": {
            "total": len(filtered),
            "timeframes": timeframe_counts,
            "signal_types": signal_type_counts,
            "sides": side_counts,
        },
        "signals": filtered,
    }