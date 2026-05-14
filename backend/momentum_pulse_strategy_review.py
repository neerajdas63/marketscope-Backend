from __future__ import annotations

import logging
import threading
from collections import Counter
from datetime import date, datetime, timedelta, time
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pytz

from runtime_state import load_json_state, save_json_state
from upstox_client import get_intraday_history_batch

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
STATE_FILE = "momentum_pulse_strategy_review.json"
REVIEW_CUTOFF_TIME = time(15, 30)
ACTIONABLE_GRADES = {"A_PLUS", "A"}
ACTIONABLE_ENTRY_STATES = {"ENTER_NOW", "ENTER_ON_RETEST", "WAIT_CONFIRMATION"}
MAX_STORED_DAYS = 30
MAX_SIGNALS_PER_DAY = 250
MAX_REVIEW_DAYS = 20

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
    state = load_json_state(STATE_FILE, {"signals": {}})
    if not isinstance(state, dict):
        return {"signals": {}}
    signals = state.get("signals")
    if not isinstance(signals, dict):
        state["signals"] = {}
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
    entry_state = _safe_str(row.get("entry_state")).upper()
    trade_side = _safe_str(row.get("trade_side")).upper()
    freshness = _safe_str(row.get("signal_freshness")).upper()
    return (
        trade_side in {"LONG", "SHORT"}
        and grade in ACTIONABLE_GRADES
        and entry_state in ACTIONABLE_ENTRY_STATES
        and freshness != "STALE"
        and _safe_float(row.get("entry_price")) > 0
        and _safe_float(row.get("stop_loss")) > 0
    )


def record_strategy_signals(strategy_payload: Dict[str, Any]) -> None:
    rows = list(strategy_payload.get("rows") or [])
    if not rows:
        return

    recordable = [_normalise_signal_row(row) for row in rows if _is_recordable_signal(row)]
    if not recordable:
        return

    with _STATE_LOCK:
        state = _load_state()
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


def build_strategy_review_payload(target_date: str = "", days: int = 1, limit: int = 200) -> Dict[str, Any]:
    selected = _selected_dates(target_date, days)
    with _STATE_LOCK:
        state = _load_state()
        stored_by_date = dict(state.get("signals") or {})

    signals: List[Dict[str, Any]] = []
    for day in selected:
        signals.extend([dict(item) for item in stored_by_date.get(day, []) if isinstance(item, dict)])

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
        "available_outcomes": ["ALL", "WIN", "LOSS", "OPEN", "FLAT", "NO_DATA"],
    }
