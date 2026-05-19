from __future__ import annotations

import logging
import os
import threading
from collections import Counter
from datetime import date, datetime, timedelta, time
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pytz

from runtime_state import load_json_state, save_json_state
from upstox_client import get_intraday_history_batch
from backend.momentum_pulse_strategy import build_strategy_rows

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
STATE_FILE = "momentum_pulse_strategy_review.json"
REVIEW_CUTOFF_TIME = time(15, 30)
ACTIONABLE_GRADES = {"A_PLUS", "A"}
ACTIONABLE_ENTRY_STATES = {"ENTER_NOW", "ENTER_ON_RETEST", "WAIT_CONFIRMATION"}
SIGNAL_RECORD_START = time(9, 35)
SIGNAL_RECORD_END_EXCLUSIVE = time(12, 0)
MAX_STORED_DAYS = 30
MAX_SIGNALS_PER_DAY = 250
MAX_REVIEW_DAYS = 20
BACKFILL_MAX_SYMBOLS = max(20, int(os.getenv("MPS_REVIEW_BACKFILL_MAX_SYMBOLS", "80")))

_STATE_LOCK = threading.Lock()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    return str(value).strip() if value is not None else default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return default


def _round_price(value: Any) -> Optional[float]:
    number = _safe_float(value)
    return round(number, 2) if number > 0 else None


def _today_ist() -> date:
    return datetime.now(IST).date()


def _now_ist() -> datetime:
    return datetime.now(IST)


def _parse_date(value: str) -> Optional[date]:
    try:
        return datetime.strptime(str(value or "").strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _parse_time(value: Any) -> Optional[time]:
    text = _safe_str(value)
    if not text:
        return None
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return None


def _load_state() -> Dict[str, Any]:
    state = load_json_state(STATE_FILE, {"signals": {}, "capture_status": {}})
    if not isinstance(state, dict):
        return {"signals": {}, "capture_status": {}}
    signals = state.get("signals")
    if not isinstance(signals, dict):
        state["signals"] = {}
    capture_status = state.get("capture_status")
    if not isinstance(capture_status, dict):
        state["capture_status"] = {}
    return state


def _save_state(state: Dict[str, Any]) -> None:
    save_json_state(STATE_FILE, state)


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _normalise_signal_row(row: Dict[str, Any]) -> Dict[str, Any]:
    symbol = _safe_str(row.get("symbol")).upper()
    trade_date = _safe_str(row.get("trade_date")) or _today_ist().isoformat()
    signal_bar_time = _safe_str(row.get("signal_bar_time", row.get("scan_time")))
    trade_side = _safe_str(row.get("trade_side")).upper()
    grade = _safe_str(row.get("grade")).upper()
    entry_state = _safe_str(row.get("entry_state")).upper()

    return {
        "id": f"{trade_date}|{signal_bar_time}|{symbol}|{trade_side}|{grade}",
        "symbol": symbol,
        "trade_date": trade_date,
        "signal_bar_time": signal_bar_time,
        "scan_time": _safe_str(row.get("scan_time", signal_bar_time)),
        "refresh_time": _safe_str(row.get("refresh_time", row.get("refreshtime"))),
        "signal_age_minutes": row.get("signal_age_minutes"),
        "signal_freshness": _safe_str(row.get("signal_freshness")),
        "trade_side": trade_side,
        "grade": grade,
        "entry_state": entry_state,
        "execution_rank": _safe_float(row.get("execution_rank")),
        "score": _safe_float(row.get("score")),
        "momentum_pulse_score": _safe_float(row.get("momentum_pulse_score")),
        "price_at_scan": _round_price(row.get("price_at_scan", row.get("ltp"))),
        "entry_price": _round_price(row.get("entry_price")),
        "stop_loss": _round_price(row.get("stop_loss")),
        "target_1": _round_price(row.get("target_1")),
        "target_2": _round_price(row.get("target_2")),
        "rr_t1": row.get("rr_t1"),
        "rr_t2": row.get("rr_t2"),
        "vwap": _round_price(row.get("vwap")),
        "or_high": _round_price(row.get("or_high")),
        "or_low": _round_price(row.get("or_low")),
        "vwap_distance_pct": round(_safe_float(row.get("vwap_distance_pct")), 2),
        "volume_ratio": round(_safe_float(row.get("volume_ratio")), 2),
        "range_ratio": round(_safe_float(row.get("range_ratio")), 2),
        "chase_risk": _safe_str(row.get("chase_risk")).upper(),
        "retest_ok": _safe_bool(row.get("retest_ok")),
        "grade_stability_score": _safe_float(row.get("grade_stability_score")),
        "reasons": [str(item) for item in _as_list(row.get("reasons"))[:8]],
        "major_risks": [str(item) for item in _as_list(row.get("major_risks"))[:8]],
        "reversal_flags": [str(item) for item in _as_list(row.get("reversal_flags"))[:8]],
        "warning_flags": [str(item) for item in _as_list(row.get("warning_flags"))[:8]],
        "recorded_at": _now_ist().strftime("%H:%M:%S"),
    }


def _is_recordable_signal(row: Dict[str, Any]) -> bool:
    grade = _safe_str(row.get("grade")).upper()
    trade_side = _safe_str(row.get("trade_side")).upper()
    signal_time = _parse_time(row.get("signal_bar_time", row.get("scan_time")))
    return (
        trade_side in {"LONG", "SHORT"}
        and grade in ACTIONABLE_GRADES
        and signal_time is not None
        and SIGNAL_RECORD_START <= signal_time < SIGNAL_RECORD_END_EXCLUSIVE
        and _safe_float(row.get("entry_price")) > 0
        and _safe_float(row.get("stop_loss")) > 0
    )


def _capture_status_from_rows(rows: Sequence[Dict[str, Any]], recordable_count: int) -> Dict[str, Any]:
    row_list = [row for row in (rows or []) if isinstance(row, dict)]
    in_window = []
    a_grade = []
    a_grade_in_window = []
    for row in row_list:
        grade = _safe_str(row.get("grade")).upper()
        signal_time = _parse_time(row.get("signal_bar_time", row.get("scan_time")))
        is_in_window = signal_time is not None and SIGNAL_RECORD_START <= signal_time < SIGNAL_RECORD_END_EXCLUSIVE
        if is_in_window:
            in_window.append(row)
        if grade in ACTIONABLE_GRADES:
            a_grade.append(row)
        if grade in ACTIONABLE_GRADES and is_in_window:
            a_grade_in_window.append(row)

    top_reasons = Counter()
    for row in row_list:
        top_reasons.update(str(reason) for reason in _as_list(row.get("reasons"))[:2])

    return {
        "last_capture_at": _now_ist().strftime("%H:%M:%S"),
        "last_capture_date": _today_ist().isoformat(),
        "rows_seen": len(row_list),
        "rows_in_signal_window": len(in_window),
        "a_or_a_plus_seen": len(a_grade),
        "a_or_a_plus_in_signal_window": len(a_grade_in_window),
        "recorded_count": recordable_count,
        "latest_signal_bar_time": max([_safe_str(row.get("signal_bar_time", row.get("scan_time"))) for row in row_list] or [""]),
        "top_reasons_seen": [{"reason": reason, "count": count} for reason, count in top_reasons.most_common(5)],
    }


def record_strategy_signals(strategy_payload: Dict[str, Any]) -> None:
    rows = list(strategy_payload.get("rows") or [])
    if not rows:
        return

    recordable = [_normalise_signal_row(row) for row in rows if _is_recordable_signal(row)]

    with _STATE_LOCK:
        state = _load_state()
        capture_status = _capture_status_from_rows(rows, len(recordable))
        capture_by_date = dict(state.get("capture_status") or {})
        capture_by_date[_safe_str(capture_status.get("last_capture_date"))] = capture_status
        state["capture_status"] = capture_by_date

        if not recordable:
            _save_state(state)
            return

        signals_by_date = dict(state.get("signals") or {})
        for signal in recordable:
            day = _safe_str(signal.get("trade_date"))
            existing = list(signals_by_date.get(day) or [])
            by_id = {
                _safe_str(item.get("id")): item
                for item in existing
                if isinstance(item, dict)
            }
            by_id[_safe_str(signal.get("id"))] = signal
            signals_by_date[day] = list(by_id.values())[-MAX_SIGNALS_PER_DAY:]

        keep_days = sorted(signals_by_date.keys())[-MAX_STORED_DAYS:]
        state["signals"] = {day: signals_by_date[day] for day in keep_days}
        _save_state(state)


def _selected_dates(target_date: str = "", days: int = 1) -> List[str]:
    clean_date = _safe_str(target_date)
    if clean_date:
        parsed = _parse_date(clean_date)
        return [parsed.isoformat()] if parsed else []

    span = min(max(int(days or 1), 1), MAX_REVIEW_DAYS)
    end = _today_ist()
    return [(end - timedelta(days=offset)).isoformat() for offset in range(span - 1, -1, -1)]


def _symbols_from_scanner(scanner_stocks: Optional[Sequence[Any]]) -> List[str]:
    symbols: List[str] = []
    seen = set()
    for item in scanner_stocks or []:
        if isinstance(item, dict):
            symbol = _safe_str(item.get("symbol"))
        else:
            symbol = _safe_str(item)
        symbol = symbol.replace(".NS", "").upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
        if len(symbols) >= BACKFILL_MAX_SYMBOLS:
            break
    return symbols


def _intraday_vwap(frame: pd.DataFrame) -> float:
    if frame is None or frame.empty:
        return 0.0
    volume = pd.Series(frame["Volume"], dtype="float64").fillna(0.0)
    typical = (
        pd.Series(frame["High"], dtype="float64")
        + pd.Series(frame["Low"], dtype="float64")
        + pd.Series(frame["Close"], dtype="float64")
    ) / 3.0
    total_volume = float(volume.sum())
    if total_volume <= 0:
        return _safe_float(frame["Close"].iloc[-1])
    return round(float((typical * volume).sum() / total_volume), 2)


def _directional_score(frame: pd.DataFrame, side: str) -> float:
    recent = frame.tail(min(6, len(frame))).copy()
    if len(recent) < 2:
        return 0.0
    close = pd.Series(recent["Close"], dtype="float64")
    open_ = pd.Series(recent["Open"], dtype="float64")
    close_diff = close.diff().dropna()
    if side == "LONG":
        candle_ratio = float((close > open_).mean())
        follow_ratio = float((close_diff > 0).mean()) if not close_diff.empty else 0.0
    else:
        candle_ratio = float((close < open_).mean())
        follow_ratio = float((close_diff < 0).mean()) if not close_diff.empty else 0.0
    return round((candle_ratio * 8.0) + (follow_ratio * 7.0), 1)


def _backfill_candidate_row(symbol: str, day: str, session_df: pd.DataFrame, bar_ts: Any) -> Optional[Dict[str, Any]]:
    upto = session_df[session_df.index <= bar_ts].copy()
    if len(upto) < 5:
        return None

    opening = session_df[session_df.index.time <= time(9, 30)]
    if opening.empty:
        opening = session_df.head(3)
    or_high = _safe_float(opening["High"].max())
    or_low = _safe_float(opening["Low"].min())
    price = _safe_float(upto["Close"].iloc[-1])
    vwap = _intraday_vwap(upto)
    if price <= 0 or vwap <= 0 or or_high <= 0 or or_low <= 0:
        return None

    vwap_dist = round(((price - vwap) / vwap) * 100.0, 2)
    direction = "NEUTRAL"
    if price > or_high and price > vwap and vwap_dist > 0:
        direction = "LONG"
    elif price < or_low and price < vwap and vwap_dist < 0:
        direction = "SHORT"
    if direction == "NEUTRAL":
        return None
    signal_time_text = bar_ts.strftime("%H:%M") if hasattr(bar_ts, "strftime") else _safe_str(bar_ts)
    try:
        review_reference_time = (datetime.combine(_parse_date(day) or _today_ist(), _parse_time(signal_time_text) or time(9, 35)) + timedelta(minutes=5)).strftime("%H:%M:%S")
    except Exception:
        review_reference_time = signal_time_text

    recent = upto.tail(min(6, len(upto)))
    avg_volume = _safe_float(recent["Volume"].mean())
    last_volume = _safe_float(recent["Volume"].iloc[-1])
    volume_ratio = round(last_volume / avg_volume, 2) if avg_volume > 0 else 1.0
    candle_ranges = (pd.Series(recent["High"], dtype="float64") - pd.Series(recent["Low"], dtype="float64")).abs()
    avg_range = _safe_float(candle_ranges.mean())
    current_range = _safe_float(candle_ranges.iloc[-1])
    range_ratio = round(current_range / avg_range, 2) if avg_range > 0 else 1.0

    long_score = _directional_score(upto, "LONG")
    short_score = _directional_score(upto, "SHORT")
    score_used = long_score if direction == "LONG" else short_score
    rank_grade = 4 if volume_ratio >= 1.5 and range_ratio >= 1.2 and score_used >= 12 else 3 if score_used >= 9 else 2

    return {
        "symbol": symbol,
        "trade_date": day,
        "scan_time": signal_time_text,
        "signal_bar_time": signal_time_text,
        "refresh_time": review_reference_time,
        "age_reference_time": review_reference_time,
        "market_data_last_updated": review_reference_time,
        "price_at_scan": price,
        "ltp": price,
        "prev_close": _safe_float(session_df["Open"].iloc[0]),
        "vwap": vwap,
        "vwap_distance_pct": vwap_dist,
        "or_high": or_high,
        "or_low": or_low,
        "volume_ratio": volume_ratio,
        "range_ratio": range_ratio,
        "direction": direction,
        "direction_confidence": max(long_score, short_score),
        "long_score": long_score,
        "short_score": short_score,
        "rank_grade": rank_grade,
        "momentum_pulse_score": min(100.0, round((volume_ratio * 18.0) + (range_ratio * 14.0) + (score_used * 3.0), 1)),
        "pulse_trend_label": "Rising" if score_used >= 9 else "Flat",
        "score_change_5m": 0.0,
        "score_change_10m": 0.0,
        "weakening_streak": 0,
        "improving_now": True,
        "reasons": ["Backfilled from 5-minute candles"],
        "review_source": "on_demand_backfill",
    }


def _backfill_signals_for_dates(selected: Sequence[str], scanner_stocks: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    symbols = _symbols_from_scanner(scanner_stocks)
    if not symbols:
        return []

    output: List[Dict[str, Any]] = []
    for day in selected:
        labels = [f"{symbol}.NS" for symbol in symbols]
        raw = None
        try:
            raw = get_intraday_history_batch(labels, from_date=day, to_date=day, interval_minutes=5)
        except Exception as exc:
            logger.warning("Strategy review backfill history fetch failed for %s: %s", day, exc)
            continue
        if raw is None or raw.empty:
            continue

        for symbol in symbols:
            frame = _get_symbol_frame(raw, symbol)
            if frame.empty:
                continue
            session = frame[(frame.index.time >= time(9, 15)) & (frame.index.time <= REVIEW_CUTOFF_TIME)].copy()
            signal_bars = session[
                (session.index.time >= SIGNAL_RECORD_START)
                & (session.index.time < SIGNAL_RECORD_END_EXCLUSIVE)
            ]
            for bar_ts in signal_bars.index:
                row = _backfill_candidate_row(symbol, day, session, bar_ts)
                if row:
                    output.append(row)

    strategy_rows = build_strategy_rows(output)
    return [
        _normalise_signal_row(row)
        for row in strategy_rows
        if _is_recordable_signal(row)
    ][:MAX_SIGNALS_PER_DAY]


def _get_symbol_frame(raw: Optional[pd.DataFrame], symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    label = _safe_str(symbol).upper()
    label_ns = label if label.endswith(".NS") else f"{label}.NS"
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            for candidate in (label_ns, label):
                if candidate in raw.columns.get_level_values(0):
                    frame = raw[candidate].copy()
                    return frame.dropna(how="all")
        return raw.copy().dropna(how="all")
    except Exception:
        return pd.DataFrame()


def _slice_after_signal(frame: pd.DataFrame, signal_time_value: Any) -> pd.DataFrame:
    signal_time = _parse_time(signal_time_value)
    if frame is None or frame.empty or signal_time is None:
        return pd.DataFrame()
    try:
        clean = frame.copy().sort_index()
        clean = clean[clean.index.time >= signal_time]
        clean = clean[clean.index.time <= REVIEW_CUTOFF_TIME]
        return clean
    except Exception:
        return pd.DataFrame()


def _reason_from_signal(signal: Dict[str, Any], outcome: str, event: str) -> List[str]:
    reasons: List[str] = []
    side = _safe_str(signal.get("trade_side")).upper()
    if outcome == "WIN":
        if _safe_float(signal.get("volume_ratio")) >= 1.5:
            reasons.append("volume_pace_supportive")
        if _safe_float(signal.get("range_ratio")) >= 1.2:
            reasons.append("range_expansion_supportive")
        if _safe_str(signal.get("chase_risk")).upper() == "LOW":
            reasons.append("low_chase_risk")
        if _safe_bool(signal.get("retest_ok")):
            reasons.append("retest_entry_worked")
        if not signal.get("reversal_flags"):
            reasons.append("no_reversal_flags_at_signal")
        if event == "TARGET_2":
            reasons.append("extended_follow_through")
        elif event == "TARGET_1":
            reasons.append("target_1_follow_through")
    elif outcome == "LOSS":
        if _safe_str(signal.get("signal_freshness")).upper() == "STALE":
            reasons.append("stale_signal")
        if _safe_str(signal.get("chase_risk")).upper() in {"MEDIUM", "HIGH"}:
            reasons.append("chase_risk")
        if signal.get("reversal_flags"):
            reasons.extend([f"reversal_{flag}" for flag in signal.get("reversal_flags", [])[:4]])
        if "far_from_vwap" in [str(x).lower() for x in signal.get("major_risks", [])]:
            reasons.append("far_from_vwap")
        if "one_bar_spike" in [str(x).lower() for x in signal.get("major_risks", [])]:
            reasons.append("one_bar_spike")
        if side == "LONG":
            reasons.append("stop_hit_before_upside_follow_through")
        elif side == "SHORT":
            reasons.append("stop_hit_before_downside_follow_through")
    return reasons or ["price_action_follow_through" if outcome == "WIN" else "price_action_failed"]


def _evaluate_signal(signal: Dict[str, Any], frame: pd.DataFrame, as_of_final: bool) -> Dict[str, Any]:
    side = _safe_str(signal.get("trade_side")).upper()
    entry = _safe_float(signal.get("entry_price"))
    stop = _safe_float(signal.get("stop_loss"))
    target_1 = _safe_float(signal.get("target_1"))
    target_2 = _safe_float(signal.get("target_2"))
    after = _slice_after_signal(frame, signal.get("signal_bar_time"))

    result = dict(signal)
    result.update(
        {
            "outcome": "NO_DATA",
            "outcome_event": "NO_CANDLES",
            "outcome_reason": "No post-signal candle data available yet",
            "win_loss_reason_codes": ["no_candle_data"],
            "max_favorable_pct": 0.0,
            "max_adverse_pct": 0.0,
            "exit_price": None,
            "exit_time": "",
            "bars_reviewed": 0,
        }
    )

    if side not in {"LONG", "SHORT"} or entry <= 0 or stop <= 0 or after.empty:
        return result

    highs = pd.Series(after["High"], dtype="float64")
    lows = pd.Series(after["Low"], dtype="float64")
    closes = pd.Series(after["Close"], dtype="float64")
    result["bars_reviewed"] = int(len(after))

    if side == "LONG":
        max_fav = (float(highs.max()) - entry) / entry * 100.0
        max_adv = (entry - float(lows.min())) / entry * 100.0
    else:
        max_fav = (entry - float(lows.min())) / entry * 100.0
        max_adv = (float(highs.max()) - entry) / entry * 100.0
    result["max_favorable_pct"] = round(max_fav, 2)
    result["max_adverse_pct"] = round(max_adv, 2)

    for ts, candle in after.iterrows():
        high = _safe_float(candle.get("High"))
        low = _safe_float(candle.get("Low"))
        close = _safe_float(candle.get("Close"))
        hit_stop = low <= stop if side == "LONG" else high >= stop
        hit_t2 = high >= target_2 if side == "LONG" and target_2 > 0 else low <= target_2 if side == "SHORT" and target_2 > 0 else False
        hit_t1 = high >= target_1 if side == "LONG" and target_1 > 0 else low <= target_1 if side == "SHORT" and target_1 > 0 else False

        if hit_stop and (hit_t1 or hit_t2):
            result.update({"outcome": "LOSS", "outcome_event": "AMBIGUOUS_STOP_FIRST", "exit_price": stop})
            result["exit_time"] = ts.strftime("%H:%M") if hasattr(ts, "strftime") else _safe_str(ts)
            result["win_loss_reason_codes"] = _reason_from_signal(signal, "LOSS", "AMBIGUOUS_STOP_FIRST")
            result["outcome_reason"] = "Same candle me target/stop overlap tha; conservative review ne stop-first maana"
            return result
        if hit_stop:
            result.update({"outcome": "LOSS", "outcome_event": "STOP_LOSS", "exit_price": stop})
            result["exit_time"] = ts.strftime("%H:%M") if hasattr(ts, "strftime") else _safe_str(ts)
            result["win_loss_reason_codes"] = _reason_from_signal(signal, "LOSS", "STOP_LOSS")
            result["outcome_reason"] = "Stop loss target se pehle hit hua"
            return result
        if hit_t2:
            result.update({"outcome": "WIN", "outcome_event": "TARGET_2", "exit_price": target_2})
            result["exit_time"] = ts.strftime("%H:%M") if hasattr(ts, "strftime") else _safe_str(ts)
            result["win_loss_reason_codes"] = _reason_from_signal(signal, "WIN", "TARGET_2")
            result["outcome_reason"] = "Target 2 hit hua; strong follow-through mila"
            return result
        if hit_t1:
            result.update({"outcome": "WIN", "outcome_event": "TARGET_1", "exit_price": target_1})
            result["exit_time"] = ts.strftime("%H:%M") if hasattr(ts, "strftime") else _safe_str(ts)
            result["win_loss_reason_codes"] = _reason_from_signal(signal, "WIN", "TARGET_1")
            result["outcome_reason"] = "Target 1 hit hua; first follow-through successful tha"
            return result
        result["exit_price"] = round(close, 2) if close > 0 else result.get("exit_price")
        result["exit_time"] = ts.strftime("%H:%M") if hasattr(ts, "strftime") else _safe_str(ts)

    final_close = _safe_float(closes.iloc[-1]) if not closes.empty else 0.0
    pnl_pct = ((final_close - entry) / entry * 100.0) if side == "LONG" else ((entry - final_close) / entry * 100.0)
    result["close_pnl_pct"] = round(pnl_pct, 2)
    if as_of_final:
        if pnl_pct > 0:
            result.update({"outcome": "WIN", "outcome_event": "EOD_POSITIVE"})
            result["win_loss_reason_codes"] = _reason_from_signal(signal, "WIN", "EOD_POSITIVE")
            result["outcome_reason"] = "Target hit nahi hua, par 15:30 tak trade positive close hua"
        elif pnl_pct < 0:
            result.update({"outcome": "LOSS", "outcome_event": "EOD_NEGATIVE"})
            result["win_loss_reason_codes"] = _reason_from_signal(signal, "LOSS", "EOD_NEGATIVE")
            result["outcome_reason"] = "Target hit nahi hua aur 15:30 tak trade negative close hua"
        else:
            result.update({"outcome": "FLAT", "outcome_event": "EOD_FLAT", "win_loss_reason_codes": ["flat_close"]})
            result["outcome_reason"] = "Trade roughly flat close hua"
    else:
        result.update({"outcome": "OPEN", "outcome_event": "INTRADAY_OPEN", "win_loss_reason_codes": ["market_not_closed_yet"]})
        result["outcome_reason"] = "Market close se pehle review provisional hai"
    return result


def _summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows or [])
    total = len(rows)
    wins = [row for row in rows if _safe_str(row.get("outcome")).upper() == "WIN"]
    losses = [row for row in rows if _safe_str(row.get("outcome")).upper() == "LOSS"]
    open_rows = [row for row in rows if _safe_str(row.get("outcome")).upper() == "OPEN"]
    win_reason_counter: Counter[str] = Counter()
    loss_reason_counter: Counter[str] = Counter()

    for row in wins:
        win_reason_counter.update(str(reason) for reason in row.get("win_loss_reason_codes", []))
    for row in losses:
        loss_reason_counter.update(str(reason) for reason in row.get("win_loss_reason_codes", []))

    return {
        "total_signals": total,
        "win_count": len(wins),
        "loss_count": len(losses),
        "open_count": len(open_rows),
        "no_data_count": len([row for row in rows if _safe_str(row.get("outcome")).upper() == "NO_DATA"]),
        "win_rate_pct": round((len(wins) / max(len(wins) + len(losses), 1)) * 100.0, 2),
        "avg_execution_rank": round(sum(_safe_float(row.get("execution_rank")) for row in rows) / total, 2) if total else 0.0,
        "top_win_reasons": [{"reason": key, "count": value} for key, value in win_reason_counter.most_common(8)],
        "top_loss_reasons": [{"reason": key, "count": value} for key, value in loss_reason_counter.most_common(8)],
    }


def build_strategy_review_payload(
    target_date: str = "",
    days: int = 1,
    limit: int = 200,
    scanner_stocks: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    selected = _selected_dates(target_date, days)
    with _STATE_LOCK:
        state = _load_state()
        stored_by_date = dict(state.get("signals") or {})
        capture_by_date = dict(state.get("capture_status") or {})

    capture_status = {
        "selected_dates": selected,
        "by_date": {day: capture_by_date.get(day, {}) for day in selected},
        "total_recorded_for_selected_dates": sum(
            len(stored_by_date.get(day, []) or [])
            for day in selected
        ),
    }

    signals: List[Dict[str, Any]] = []
    for day in selected:
        signals.extend([dict(item) for item in stored_by_date.get(day, []) if isinstance(item, dict)])

    if not signals:
        backfilled = _backfill_signals_for_dates(selected, scanner_stocks)
        if backfilled:
            with _STATE_LOCK:
                state = _load_state()
                signals_by_date = dict(state.get("signals") or {})
                for signal in backfilled:
                    day = _safe_str(signal.get("trade_date"))
                    existing = list(signals_by_date.get(day) or [])
                    by_id = {
                        _safe_str(item.get("id")): item
                        for item in existing
                        if isinstance(item, dict)
                    }
                    by_id[_safe_str(signal.get("id"))] = signal
                    signals_by_date[day] = list(by_id.values())[-MAX_SIGNALS_PER_DAY:]
                state["signals"] = signals_by_date
                _save_state(state)
                stored_by_date = dict(signals_by_date)
            signals = [dict(item) for item in backfilled]
            capture_status["backfill"] = {
                "status": "used",
                "signals_generated": len(backfilled),
                "max_symbols": BACKFILL_MAX_SYMBOLS,
                "source": "5-minute candles",
            }
        else:
            capture_status["backfill"] = {
                "status": "empty",
                "signals_generated": 0,
                "max_symbols": BACKFILL_MAX_SYMBOLS,
                "source": "5-minute candles",
            }

    max_limit = min(max(int(limit or 200), 1), 500)
    signals = signals[-max_limit:]
    if not signals:
        return {
            "feature": "Momentum Pulse Strategy Review",
            "feature_key": "momentum_pulse_strategy_review",
            "mode": "review",
            "status": "empty",
            "message": "No recorded signals found. Review needs signals captured during the live 09:35-12:00 window; future signals are now auto-recorded from Momentum Pulse refresh.",
            "dates": selected,
            "rows": [],
            "total": 0,
            "summary": _summary([]),
            "capture_status": capture_status,
        }

    rows: List[Dict[str, Any]] = []
    for day in sorted({str(signal.get("trade_date")) for signal in signals if signal.get("trade_date")}):
        day_signals = [signal for signal in signals if _safe_str(signal.get("trade_date")) == day]
        symbols = sorted({f"{_safe_str(signal.get('symbol')).upper()}.NS" for signal in day_signals if signal.get("symbol")})
        raw = None
        if symbols:
            try:
                raw = get_intraday_history_batch(symbols, from_date=day, to_date=day, interval_minutes=5)
            except Exception as exc:
                logger.warning("Strategy review history fetch failed for %s: %s", day, exc)
                raw = None

        review_date = _parse_date(day)
        as_of_final = bool(review_date and (review_date < _today_ist() or _now_ist().time() >= REVIEW_CUTOFF_TIME))
        for signal in day_signals:
            frame = _get_symbol_frame(raw, _safe_str(signal.get("symbol")))
            rows.append(_evaluate_signal(signal, frame, as_of_final=as_of_final))

    rows.sort(
        key=lambda row: (
            _safe_str(row.get("trade_date")),
            _safe_str(row.get("signal_bar_time")),
            _safe_float(row.get("execution_rank")),
        ),
        reverse=True,
    )

    final_ready = all(
        _parse_date(day) and (_parse_date(day) < _today_ist() or _now_ist().time() >= REVIEW_CUTOFF_TIME)
        for day in selected
    )
    return {
        "feature": "Momentum Pulse Strategy Review",
        "feature_key": "momentum_pulse_strategy_review",
        "mode": "review",
        "status": "final" if final_ready else "provisional",
        "message": "Final post-market review" if final_ready else "Provisional review; final result locks after 15:30 IST",
        "dates": selected,
        "rows": rows,
        "total": len(rows),
        "summary": _summary(rows),
        "capture_status": capture_status,
        "available_outcomes": ["ALL", "WIN", "LOSS", "OPEN", "FLAT", "NO_DATA"],
    }
