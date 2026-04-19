from __future__ import annotations

from datetime import date as calendar_date, datetime, time, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pytz

from backend.momentum_pulse import (
    LOOKBACK_SESSIONS,
    MOMENTUM_PULSE_BATCH_SIZE,
    _SYMBOL_TO_SECTOR,
    _download_intraday_batch_for_range,
    _evaluate_symbol,
    _get_sym_df,
    _split_sessions,
)
from stocks import SCANNER_STOCKS

IST = pytz.timezone("Asia/Kolkata")
ENTRY_WINDOW_START = time(9, 35)
ENTRY_WINDOW_END_EXCLUSIVE = time(11, 0)
_VALID_DIRECTION_FILTERS = {"ALL", "LONG", "SHORT"}
_VALID_GRADE_FILTERS = {"ALL", "A_PLUS", "A", "FAILED_OR_CHOP", "NO_TRADE"}
_ACTIONABLE_GRADES = {"A_PLUS", "A"}
_HISTORICAL_LOOKBACK_CALENDAR_DAYS = max(35, LOOKBACK_SESSIONS + 10)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any, default: str = "") -> str:
    return str(value).strip() if value is not None else default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        x = value.strip().lower()
        if x == "true":
            return True
        if x == "false":
            return False
    return default


def _round_price(value: Any) -> Optional[float]:
    v = _safe_float(value)
    return round(v, 2) if v > 0 else None


def _parse_scan_time(value: Any) -> Optional[time]:
    if value is None:
        return None
    if isinstance(value, time):
        return value
    text = str(value).strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return None


def _in_early_window(scan_time_value: Any) -> bool:
    t = _parse_scan_time(scan_time_value)
    if t is None:
        return False
    return ENTRY_WINDOW_START <= t < ENTRY_WINDOW_END_EXCLUSIVE


def _has_text_item(items: Sequence[Any], needle: str) -> bool:
    needle = needle.strip().lower()
    return any(str(x).strip().lower() == needle for x in (items or []))


def _risk_reward(entry: Optional[float], stop_loss: Optional[float], target: Optional[float]) -> Optional[float]:
    if entry is None or stop_loss is None or target is None:
        return None
    risk = abs(entry - stop_loss)
    reward = abs(target - entry)
    if risk <= 0:
        return None
    return round(reward / risk, 2)


def _get_directional_scores(row: Dict[str, Any]) -> Dict[str, float]:
    return {
        "long_score": _safe_float(row.get("longscore", row.get("long_score", 0.0))),
        "short_score": _safe_float(row.get("shortscore", row.get("short_score", 0.0))),
    }


def _major_risk_flags(row: Dict[str, Any]) -> List[str]:
    warning_flags = list(row.get("warningflags", row.get("warning_flags", [])) or [])
    risks: List[str] = []

    if _has_text_item(warning_flags, "farfromvwap") or _has_text_item(warning_flags, "far_from_vwap"):
        risks.append("far_from_vwap")
    if _has_text_item(warning_flags, "momentumdecay") or _has_text_item(warning_flags, "momentum_decay"):
        risks.append("momentum_decay")
    if _has_text_item(warning_flags, "onebarspike") or _has_text_item(warning_flags, "one_bar_spike"):
        risks.append("one_bar_spike")
    if _has_text_item(warning_flags, "lowconsistency") or _has_text_item(warning_flags, "low_consistency"):
        risks.append("low_consistency")
    if _has_text_item(warning_flags, "weakrelativestrength") or _has_text_item(warning_flags, "weak_relative_strength"):
        risks.append("weak_relative_strength")
    if _has_text_item(warning_flags, "fadingscore") or _has_text_item(warning_flags, "fading_score"):
        risks.append("fading_score")
    if _has_text_item(warning_flags, "lowvolumeconfirmation") or _has_text_item(warning_flags, "low_volume_confirmation"):
        risks.append("low_volume_confirmation")

    behavior_state = _safe_str(row.get("behaviorstate", row.get("behavior_state"))).upper()
    if behavior_state == "EXTENDED":
        risks.append("extended")

    if _safe_bool(row.get("long_failed_fast")):
        risks.append("long_failed_fast")
    if _safe_bool(row.get("short_failed_fast")):
        risks.append("short_failed_fast")

    return risks


def _is_failed_or_chop(row: Dict[str, Any], direction: str) -> bool:
    reasons = " | ".join(str(x) for x in (row.get("reasons") or []))
    reasons = reasons.lower()

    if "chop risk" in reasons:
        return True
    if "setup incomplete" in reasons:
        return True
    if "directional edge clear nahi" in reasons:
        return True
    if direction == "LONG" and _safe_bool(row.get("long_failed_fast")):
        return True
    if direction == "SHORT" and _safe_bool(row.get("short_failed_fast")):
        return True
    return False


def classify_trade(row: Dict[str, Any]) -> Dict[str, Any]:
    direction = _safe_str(row.get("direction", row.get("trade_side", "NO_TRADE"))).upper()
    scan_time = row.get("scan_time")
    in_early_window = _in_early_window(scan_time)

    price = _safe_float(row.get("price_at_scan", row.get("ltp")))
    vwap = _safe_float(row.get("vwap"))
    vwap_dist = _safe_float(row.get("vwap_distance_pct", row.get("distance_from_vwap_pct")))
    or_high = _safe_float(row.get("or_high", row.get("opening_range_high")))
    or_low = _safe_float(row.get("or_low", row.get("opening_range_low")))
    volume_ratio = _safe_float(row.get("volume_ratio", row.get("volume_pace_ratio", row.get("volumepaceratio"))))
    range_ratio = _safe_float(row.get("range_ratio", row.get("range_expansion_ratio", row.get("rangeexpansionratio"))))
    change_pct = _safe_float(row.get("change_pct_at_scan", row.get("changepct", row.get("change_pct"))))
    scores = _get_directional_scores(row)
    long_score = scores["long_score"]
    short_score = scores["short_score"]
    reasons = list(row.get("reasons") or [])
    rank_grade = _safe_int(row.get("rank_grade"))
    risks = _major_risk_flags(row)

    if direction not in {"LONG", "SHORT"}:
        return {
            "trade_side": "NO_TRADE",
            "grade": "NO_TRADE",
            "eligible_time_window": in_early_window,
            "score": 0.0,
            "reasons": ["Directional edge clear nahi tha"],
        }

    if not in_early_window:
        return {
            "trade_side": "NO_TRADE",
            "grade": "NO_TRADE",
            "eligible_time_window": False,
            "score": 0.0,
            "reasons": ["Strategy entry sirf 09:35 ke baad aur 11:00 se pehle valid hogi"],
        }

    if _is_failed_or_chop(row, direction):
        return {
            "trade_side": "NO_TRADE",
            "grade": "FAILED_OR_CHOP",
            "eligible_time_window": True,
            "score": 0.0,
            "reasons": ["No clean trade / failed setup / chop risk"],
        }

    score_used = long_score if direction == "LONG" else short_score

    if direction == "LONG":
        structure_ok = price > or_high and price > vwap and vwap_dist > 0
        high_conviction = (
            structure_ok
            and volume_ratio >= 1.50
            and range_ratio >= 1.20
            and 0.20 <= vwap_dist <= 1.20
            and score_used >= 12
            and rank_grade >= 4
            and "long_failed_fast" not in risks
            and "momentum_decay" not in risks
            and "fading_score" not in risks
        )
        medium_conviction = (
            structure_ok
            and volume_ratio >= 0.90
            and range_ratio >= 1.00
            and 0.15 <= vwap_dist <= 1.30
            and score_used >= 9
            and rank_grade >= 3
            and "long_failed_fast" not in risks
            and "momentum_decay" not in risks
        )
    else:
        structure_ok = price < or_low and price < vwap and vwap_dist < 0
        high_conviction = (
            structure_ok
            and volume_ratio >= 1.50
            and range_ratio >= 1.20
            and -1.20 <= vwap_dist <= -0.20
            and score_used >= 12
            and rank_grade >= 4
            and "short_failed_fast" not in risks
            and "momentum_decay" not in risks
            and "fading_score" not in risks
        )
        medium_conviction = (
            structure_ok
            and volume_ratio >= 0.90
            and range_ratio >= 1.00
            and -1.30 <= vwap_dist <= -0.15
            and score_used >= 9
            and rank_grade >= 3
            and "short_failed_fast" not in risks
            and "momentum_decay" not in risks
        )

    if high_conviction:
        grade = "A_PLUS"
    elif medium_conviction:
        grade = "A"
    else:
        grade = "NO_TRADE"

    output_reasons: List[str] = []

    if grade == "NO_TRADE":
        if not structure_ok:
            output_reasons.append("Setup incomplete tha")
        else:
            output_reasons.append("Confirmation threshold meet nahi hua")
        return {
            "trade_side": "NO_TRADE",
            "grade": "NO_TRADE",
            "eligible_time_window": True,
            "score": round(score_used, 1),
            "reasons": output_reasons,
        }

    if direction == "LONG":
        if volume_ratio >= 1.50:
            output_reasons.append(f"Volume pace supportive ({round(volume_ratio, 2)}x baseline)")
        if range_ratio >= 1.20:
            output_reasons.append(f"Range expansion strong ({round(range_ratio, 2)}x baseline)")
        output_reasons.append("Price VWAP ke upar sustain")
        output_reasons.append("Opening range breakout")
        if change_pct > 0:
            output_reasons.append("Bullish directional intent clear tha")
    else:
        if volume_ratio >= 1.50:
            output_reasons.append(f"Volume pace supportive ({round(volume_ratio, 2)}x baseline)")
        if range_ratio >= 1.20:
            output_reasons.append(f"Range expansion strong ({round(range_ratio, 2)}x baseline)")
        output_reasons.append("Price VWAP ke neeche sustain")
        output_reasons.append("Opening range breakdown")
        if change_pct < 0:
            output_reasons.append("Bearish directional intent clear tha")

    output_reasons.extend([r for r in reasons if str(r).strip()])

    return {
        "trade_side": direction,
        "grade": grade,
        "eligible_time_window": True,
        "score": round(score_used, 1),
        "reasons": output_reasons,
    }


def build_entry_stop_exit(row: Dict[str, Any], classified: Dict[str, Any]) -> Dict[str, Any]:
    direction = _safe_str(classified.get("trade_side")).upper()
    grade = _safe_str(classified.get("grade")).upper()

    price = _safe_float(row.get("price_at_scan", row.get("ltp")))
    vwap = _safe_float(row.get("vwap"))
    or_high = _safe_float(row.get("or_high", row.get("opening_range_high")))
    or_low = _safe_float(row.get("or_low", row.get("opening_range_low")))

    entry_notes: List[str] = []
    stop_notes: List[str] = []
    exit_notes: List[str] = []

    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None

    range_size = max(or_high - or_low, max(price * 0.004, 0.05))

    if grade in {"NO_TRADE", "FAILED_OR_CHOP"} or direction not in {"LONG", "SHORT"}:
        return {
            "entry_price": None,
            "stop_loss": None,
            "target_1": None,
            "target_2": None,
            "entry_notes": ["No clean trade / failed setup / chop risk"],
            "stop_notes": ["Avoid forced entry"],
            "exit_notes": ["Wait for cleaner setup"],
        }

    if direction == "LONG":
        entry_price = round(max(price, or_high + 0.05), 2)
        stop_loss = round(min(or_high, vwap) - max(price * 0.0015, 0.10), 2)
        target_1 = round(entry_price + range_size * 0.8, 2)
        target_2 = round(entry_price + range_size * 1.4, 2)

        entry_notes.append("Direct entry: breakout candle close above OR high")
        entry_notes.append("Safer entry: OR high / VWAP retest hold after breakout")
        if grade == "A_PLUS":
            entry_notes.append("A+ trade: early aggressive sizing possible if breakout body strong ho")
        else:
            entry_notes.append("A trade: better to wait for confirmation or small retest")

        stop_notes.append("Primary stop: OR high retest zone ke neeche")
        stop_notes.append("Alternate stop: VWAP ke neeche 5-minute close")

        exit_notes.append("Partial after first sharp expansion")
        exit_notes.append("Trail below previous 5-minute swing low")
        exit_notes.append("Full exit on VWAP close breakdown or OR reclaim failure")

    elif direction == "SHORT":
        entry_price = round(min(price, or_low - 0.05), 2)
        stop_loss = round(max(or_low, vwap) + max(price * 0.0015, 0.10), 2)
        target_1 = round(entry_price - range_size * 0.8, 2)
        target_2 = round(entry_price - range_size * 1.4, 2)

        entry_notes.append("Direct entry: breakdown candle close below OR low")
        entry_notes.append("Safer entry: OR low / VWAP retest reject after breakdown")
        if grade == "A_PLUS":
            entry_notes.append("A+ trade: aggressive short possible if selling pressure real ho")
        else:
            entry_notes.append("A trade: confirmation ke baad short lena better")

        stop_notes.append("Primary stop: OR low rejection zone ke upar")
        stop_notes.append("Alternate stop: VWAP reclaim ke upar 5-minute close")

        exit_notes.append("Partial after first sharp downside flush")
        exit_notes.append("Trail above previous 5-minute swing high")
        exit_notes.append("Full exit on VWAP reclaim or OR low reclaim")

    rr1 = _risk_reward(entry_price, stop_loss, target_1)
    rr2 = _risk_reward(entry_price, stop_loss, target_2)

    return {
        "entry_price": _round_price(entry_price),
        "stop_loss": _round_price(stop_loss),
        "target_1": _round_price(target_1),
        "target_2": _round_price(target_2),
        "rr_t1": rr1,
        "rr_t2": rr2,
        "entry_notes": entry_notes,
        "stop_notes": stop_notes,
        "exit_notes": exit_notes,
    }


def build_strategy_row(row: Dict[str, Any]) -> Dict[str, Any]:
    classified = classify_trade(row)
    plan = build_entry_stop_exit(row, classified)

    result = dict(row)
    result.update(classified)
    result.update(plan)

    result["scan_time"] = _safe_str(row.get("scan_time"))
    result["symbol"] = _safe_str(row.get("symbol"))
    result["trade_date"] = _safe_str(row.get("trade_date"))
    result["price_at_scan"] = _round_price(row.get("price_at_scan", row.get("ltp")))
    result["prev_close"] = _round_price(row.get("prev_close"))
    result["vwap"] = _round_price(row.get("vwap"))
    result["or_high"] = _round_price(row.get("or_high", row.get("opening_range_high")))
    result["or_low"] = _round_price(row.get("or_low", row.get("opening_range_low")))
    result["vwap_distance_pct"] = round(_safe_float(row.get("vwap_distance_pct", row.get("distance_from_vwap_pct"))), 2)
    result["volume_ratio"] = round(_safe_float(row.get("volume_ratio", row.get("volume_pace_ratio", row.get("volumepaceratio")))), 2)
    result["range_ratio"] = round(_safe_float(row.get("range_ratio", row.get("range_expansion_ratio", row.get("rangeexpansionratio")))), 2)

    return result


def build_strategy_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _sort_strategy_rows([build_strategy_row(row) for row in (rows or [])])


def _sort_strategy_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = list(rows or [])
    grade_priority = {
        "A_PLUS": 4,
        "A": 3,
        "FAILED_OR_CHOP": 2,
        "NO_TRADE": 1,
    }
    output.sort(
        key=lambda x: (
            grade_priority.get(_safe_str(x.get("grade")).upper(), 0),
            _safe_float(x.get("score")),
            _safe_float(x.get("volume_ratio")),
            _safe_float(x.get("range_ratio")),
        ),
        reverse=True,
    )
    return output


def summarize_strategy(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows or [])
    a_plus = [r for r in rows if _safe_str(r.get("grade")).upper() == "A_PLUS"]
    a_only = [r for r in rows if _safe_str(r.get("grade")).upper() == "A"]
    failed = [r for r in rows if _safe_str(r.get("grade")).upper() == "FAILED_OR_CHOP"]
    no_trade = [r for r in rows if _safe_str(r.get("grade")).upper() == "NO_TRADE"]
    longs = [r for r in rows if _safe_str(r.get("trade_side")).upper() == "LONG"]
    shorts = [r for r in rows if _safe_str(r.get("trade_side")).upper() == "SHORT"]

    def avg(items: Sequence[Dict[str, Any]], key: str) -> float:
        vals = [_safe_float(i.get(key)) for i in items if i.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    return {
        "total": len(rows),
        "a_plus_count": len(a_plus),
        "a_count": len(a_only),
        "failed_or_chop_count": len(failed),
        "no_trade_count": len(no_trade),
        "long_count": len(longs),
        "short_count": len(shorts),
        "avg_volume_ratio": avg(rows, "volume_ratio"),
        "avg_range_ratio": avg(rows, "range_ratio"),
        "avg_abs_change_pct": round(
            sum(abs(_safe_float(r.get("change_pct_at_scan", r.get("changepct", r.get("change_pct"))))) for r in rows) / len(rows),
            2
        ) if rows else 0.0,
        "a_plus_common": {
            "avg_score": avg(a_plus, "score"),
            "avg_vwap_dist": avg(a_plus, "vwap_distance_pct"),
            "avg_volume_ratio": avg(a_plus, "volume_ratio"),
            "avg_range_ratio": avg(a_plus, "range_ratio"),
        },
        "a_common": {
            "avg_score": avg(a_only, "score"),
            "avg_vwap_dist": avg(a_only, "vwap_distance_pct"),
            "avg_volume_ratio": avg(a_only, "volume_ratio"),
            "avg_range_ratio": avg(a_only, "range_ratio"),
        },
    }


def _infer_prev_close_from_pulse_row(row: Dict[str, Any]) -> float:
    direct = _safe_float(row.get("prev_close"))
    if direct > 0:
        return direct
    ltp = _safe_float(row.get("ltp"))
    change_pct = _safe_float(row.get("change_pct"))
    if ltp > 0 and abs(change_pct) < 95:
        base = ltp / (1.0 + (change_pct / 100.0))
        if base > 0:
            return round(base, 2)
    return 0.0


def _derive_rank_grade(row: Dict[str, Any]) -> int:
    score = _safe_float(row.get("momentum_pulse_score"))
    rank = max(1, _safe_int(row.get("rank"), 9999))
    tier = _safe_str(row.get("tier")).lower()
    volume_ratio = _safe_float(row.get("volume_pace_ratio"))
    trend_strength = _safe_float(row.get("pulse_trend_strength"))

    if score >= 72 and rank <= 12 and tier == "strong" and volume_ratio >= 1.2:
        return 4
    if score >= 58 and rank <= 30 and tier in {"strong", "moderate"} and trend_strength >= 18:
        return 3
    if score >= 45 or tier == "weak":
        return 2
    return 1


def _derive_failed_fast(row: Dict[str, Any]) -> bool:
    momentum_decay_pct = _safe_float(row.get("momentum_decay_pct"))
    score_change_5m = _safe_float(row.get("score_change_5m"))
    score_change_10m = _safe_float(row.get("score_change_10m"))
    pulse_trend_label = _safe_str(row.get("pulse_trend_label")).lower()
    return (
        momentum_decay_pct >= 10.0
        and (score_change_5m <= -2.0 or score_change_10m <= -3.0)
        and pulse_trend_label == "falling"
    )


def _seed_reasons_from_pulse_row(row: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    direction = _safe_str(row.get("direction")).upper()
    direction_confidence = _safe_float(row.get("direction_confidence"))
    volume_ratio = _safe_float(row.get("volume_pace_ratio"))
    range_ratio = _safe_float(row.get("range_expansion_ratio"))
    behavior_state = _safe_str(row.get("behavior_state")).upper()
    warning_flags = [str(flag).strip().lower() for flag in (row.get("warning_flags") or [])]
    quality_tags = [str(tag).strip().lower() for tag in (row.get("quality_tags") or [])]

    if direction not in {"LONG", "SHORT"} or direction_confidence < 12:
        reasons.append("Directional edge clear nahi tha")
    if volume_ratio < 0.9 or range_ratio < 1.0 or "low_consistency" in warning_flags:
        reasons.append("Setup incomplete tha")
    if behavior_state == "EXTENDED" or "one_bar_spike" in warning_flags or "overextended_risk" in warning_flags:
        reasons.append("Chop risk")
    if "sector_leader" in quality_tags:
        reasons.append("Sector leader confirmation")
    if "or_breakout" in quality_tags:
        reasons.append("Opening range confirmation")
    if "strong_accumulation" in quality_tags:
        reasons.append("Participation broad tha")
    return reasons


def _prepare_strategy_input_row(row: Dict[str, Any], trade_date: str) -> Dict[str, Any]:
    prepared = dict(row)
    direction = _safe_str(row.get("direction")).upper()
    failed_fast = _derive_failed_fast(row)
    prepared.update(
        {
            "scan_time": _safe_str(row.get("latest_bar_time", row.get("scan_time"))),
            "trade_date": trade_date,
            "price_at_scan": _safe_float(row.get("ltp")),
            "prev_close": _infer_prev_close_from_pulse_row(row),
            "or_high": _safe_float(row.get("opening_range_high", row.get("or_high"))),
            "or_low": _safe_float(row.get("opening_range_low", row.get("or_low"))),
            "vwap_distance_pct": _safe_float(row.get("distance_from_vwap_pct", row.get("vwap_distance_pct"))),
            "change_pct_at_scan": _safe_float(row.get("change_pct")),
            "volume_ratio": _safe_float(row.get("volume_pace_ratio", row.get("volume_ratio"))),
            "range_ratio": _safe_float(row.get("range_expansion_ratio", row.get("range_ratio"))),
            "rank_grade": _derive_rank_grade(row),
            "reasons": _seed_reasons_from_pulse_row(row),
            "long_failed_fast": direction == "LONG" and failed_fast,
            "short_failed_fast": direction == "SHORT" and failed_fast,
        }
    )
    return prepared


def _normalize_direction_filter(value: str) -> str:
    normalized = _safe_str(value, "ALL").upper()
    return normalized if normalized in _VALID_DIRECTION_FILTERS else "ALL"


def _normalize_grade_filter(value: str) -> str:
    normalized = _safe_str(value, "ALL").upper()
    return normalized if normalized in _VALID_GRADE_FILTERS else "ALL"


def _filter_strategy_rows(
    rows: Sequence[Dict[str, Any]],
    direction: str,
    grade: str,
    limit: int,
) -> Tuple[List[Dict[str, Any]], str, str]:
    normalized_direction = _normalize_direction_filter(direction)
    normalized_grade = _normalize_grade_filter(grade)

    filtered_rows = list(rows or [])
    if normalized_direction in {"LONG", "SHORT"}:
        filtered_rows = [row for row in filtered_rows if _safe_str(row.get("trade_side")).upper() == normalized_direction]
    if normalized_grade != "ALL":
        filtered_rows = [row for row in filtered_rows if _safe_str(row.get("grade")).upper() == normalized_grade]
    if limit > 0:
        filtered_rows = filtered_rows[:limit]

    return filtered_rows, normalized_direction, normalized_grade


def _build_strategy_response(
    strategy_rows: Sequence[Dict[str, Any]],
    *,
    status: str,
    message: str,
    last_updated: str,
    market_data_last_updated: str,
    benchmark_change_pct: float,
    direction: str,
    grade: str,
    limit: int,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ordered_rows = _sort_strategy_rows(strategy_rows)
    filtered_rows, normalized_direction, normalized_grade = _filter_strategy_rows(
        ordered_rows,
        direction=direction,
        grade=grade,
        limit=limit,
    )

    response = {
        "feature": "Momentum Pulse Strategy",
        "feature_key": "momentum_pulse_strategy",
        "status": _safe_str(status, "ready"),
        "message": _safe_str(message),
        "last_updated": _safe_str(last_updated),
        "market_data_last_updated": _safe_str(market_data_last_updated, last_updated),
        "benchmark_change_pct": round(_safe_float(benchmark_change_pct), 2),
        "direction": normalized_direction,
        "grade": normalized_grade,
        "rows": filtered_rows,
        "total": len(filtered_rows),
        "total_candidates": len(ordered_rows),
        "summary": summarize_strategy(filtered_rows),
        "overall_summary": summarize_strategy(ordered_rows),
        "available_directions": ["ALL", "LONG", "SHORT"],
        "available_grades": ["ALL", "A_PLUS", "A", "FAILED_OR_CHOP", "NO_TRADE"],
    }
    if extra_fields:
        response.update(extra_fields)
    return response


def _chunked(items: Sequence[Any], size: int) -> List[List[Any]]:
    step = max(1, int(size or 1))
    return [list(items[index:index + step]) for index in range(0, len(items), step)]


def _normalize_scanner_symbols(items: Optional[Sequence[Any]]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for item in items or []:
        raw = item.get("symbol") if isinstance(item, dict) else item
        symbol = _safe_str(raw).upper().replace(".NS", "")
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
    return normalized


def _iter_entry_window_scan_times() -> List[time]:
    anchor = datetime.now(IST).date()
    current = datetime.combine(anchor, ENTRY_WINDOW_START)
    end = datetime.combine(anchor, ENTRY_WINDOW_END_EXCLUSIVE)
    slots: List[time] = []
    while current < end:
        slots.append(current.time())
        current += timedelta(minutes=5)
    return slots


def _slice_vwap(session_slice: pd.DataFrame) -> float:
    if session_slice is None or session_slice.empty:
        return 0.0
    volume = pd.Series(session_slice["Volume"], dtype="float64").fillna(0.0)
    total_volume = float(volume.sum())
    if total_volume <= 0:
        return 0.0
    typical_price = (
        pd.Series(session_slice["High"], dtype="float64")
        + pd.Series(session_slice["Low"], dtype="float64")
        + pd.Series(session_slice["Close"], dtype="float64")
    ) / 3.0
    return round(float((typical_price * volume).sum() / total_volume), 2)


def _build_historical_stock_snapshot(symbol: str, session_slice: pd.DataFrame, prev_close: float) -> Optional[Dict[str, Any]]:
    if session_slice is None or session_slice.empty:
        return None
    live_price = _safe_float(session_slice["Close"].iloc[-1])
    if live_price <= 0 or prev_close <= 0:
        return None
    change_pct = round(((live_price - prev_close) / prev_close) * 100.0, 2)
    return {
        "symbol": symbol,
        "ltp": round(live_price, 2),
        "change_pct": change_pct,
        "vwap": _slice_vwap(session_slice),
    }


def _build_sector_data_from_snapshots(stock_snapshots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    sector_values: Dict[str, List[float]] = {}
    for snapshot in stock_snapshots:
        symbol = _safe_str(snapshot.get("symbol")).upper()
        sector = _SYMBOL_TO_SECTOR.get(symbol)
        if not sector:
            continue
        sector_values.setdefault(sector, []).append(_safe_float(snapshot.get("change_pct")))

    return {
        sector_name: {"result": {"current": round(sum(changes) / len(changes), 2)}}
        for sector_name, changes in sector_values.items()
        if changes
    }


def _historical_session_view(
    sessions: Sequence[Tuple[calendar_date, pd.DataFrame]],
    trade_date: calendar_date,
    scan_time: time,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], float]:
    historical_frames: List[pd.DataFrame] = []
    current_slice: Optional[pd.DataFrame] = None

    for session_date, session_df in sessions:
        if session_date < trade_date:
            historical_frames.append(session_df)
            continue
        if session_date == trade_date:
            current_slice = session_df[session_df.index.time <= scan_time].copy()
            break
        break

    if current_slice is None or current_slice.empty or not historical_frames:
        return None, None, 0.0

    prev_close = _safe_float(historical_frames[-1]["Close"].dropna().iloc[-1])
    combined = pd.concat(list(historical_frames[-LOOKBACK_SESSIONS:]) + [current_slice], axis=0)
    return combined, current_slice, prev_close


def _build_historical_nifty_change_map(trade_date: calendar_date) -> Dict[str, float]:
    from_date = (trade_date - timedelta(days=_HISTORICAL_LOOKBACK_CALENDAR_DAYS)).isoformat()
    to_date = trade_date.isoformat()
    nifty_df = None

    try:
        raw = _download_intraday_batch_for_range(["NIFTY"], from_date=from_date, to_date=to_date)
        nifty_df = _get_sym_df(raw, "NIFTY")
    except Exception:
        nifty_df = None

    if nifty_df is None or nifty_df.empty:
        try:
            import os
            os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
            import yfinance as yf

            end_exclusive = (trade_date + timedelta(days=1)).isoformat()
            raw = yf.download(
                tickers="^NSEI",
                start=from_date,
                end=end_exclusive,
                interval="5m",
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=20,
            )
            nifty_df = _get_sym_df(raw, "^NSEI")
        except Exception:
            nifty_df = None

    sessions = _split_sessions(nifty_df)
    if len(sessions) < 2:
        return {}

    target_session = next((session_df for session_date, session_df in sessions if session_date == trade_date), None)
    previous_sessions = [session_df for session_date, session_df in sessions if session_date < trade_date]
    if target_session is None or not previous_sessions:
        return {}

    prev_close = _safe_float(previous_sessions[-1]["Close"].dropna().iloc[-1])
    if prev_close <= 0:
        return {}

    change_map: Dict[str, float] = {}
    for timestamp, bar in target_session.iterrows():
        close = _safe_float(bar.get("Close"))
        if close > 0:
            change_map[timestamp.strftime("%H:%M")] = round(((close - prev_close) / prev_close) * 100.0, 2)
    return change_map


def _choose_representative_daily_row(
    existing: Optional[Dict[str, Any]],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    if existing is None:
        return candidate

    existing_grade = _safe_str(existing.get("grade")).upper()
    candidate_grade = _safe_str(candidate.get("grade")).upper()
    existing_actionable = existing_grade in _ACTIONABLE_GRADES
    candidate_actionable = candidate_grade in _ACTIONABLE_GRADES

    if existing_actionable:
        return existing
    if candidate_actionable:
        return candidate

    grade_priority = {"FAILED_OR_CHOP": 2, "NO_TRADE": 1}
    existing_priority = grade_priority.get(existing_grade, 0)
    candidate_priority = grade_priority.get(candidate_grade, 0)
    if candidate_priority != existing_priority:
        return candidate if candidate_priority > existing_priority else existing

    existing_time = _parse_scan_time(existing.get("scan_time")) or time(23, 59)
    candidate_time = _parse_scan_time(candidate.get("scan_time")) or time(23, 59)
    if candidate_time != existing_time:
        return candidate if candidate_time < existing_time else existing

    if _safe_float(candidate.get("score")) != _safe_float(existing.get("score")):
        return candidate if _safe_float(candidate.get("score")) > _safe_float(existing.get("score")) else existing

    return candidate if _safe_float(candidate.get("range_ratio")) > _safe_float(existing.get("range_ratio")) else existing


def _simulate_trade_outcome(
    row: Dict[str, Any],
    sessions: Sequence[Tuple[calendar_date, pd.DataFrame]],
) -> Dict[str, Any]:
    grade = _safe_str(row.get("grade")).upper()
    direction = _safe_str(row.get("trade_side")).upper()
    entry_price = _safe_float(row.get("entry_price"))
    stop_loss = _safe_float(row.get("stop_loss"))
    target_1 = _safe_float(row.get("target_1"))
    target_2 = _safe_float(row.get("target_2"))
    scan_time = _parse_scan_time(row.get("scan_time"))
    trade_date_text = _safe_str(row.get("trade_date"))

    if grade not in _ACTIONABLE_GRADES or direction not in {"LONG", "SHORT"}:
        return {
            "historical_outcome": "NO_ENTRY",
            "historical_exit_time": "",
            "historical_exit_price": None,
            "historical_pnl_pct": 0.0,
            "historical_rr_realized": 0.0,
            "historical_outcome_reason": "",
            "loss_reason": _safe_str(row.get("loss_reason")),
        }

    target_session = next(
        (session_df for session_date, session_df in sessions if session_date.isoformat() == trade_date_text),
        None,
    )
    if target_session is None or scan_time is None or entry_price <= 0 or stop_loss <= 0:
        return {
            "historical_outcome": "NO_DATA",
            "historical_exit_time": "",
            "historical_exit_price": None,
            "historical_pnl_pct": 0.0,
            "historical_rr_realized": 0.0,
            "historical_outcome_reason": "Historical session data unavailable",
            "loss_reason": "Historical session data unavailable",
        }

    future_session = target_session[target_session.index.time > scan_time].copy()
    risk = abs(entry_price - stop_loss)
    if risk <= 0:
        return {
            "historical_outcome": "NO_DATA",
            "historical_exit_time": "",
            "historical_exit_price": None,
            "historical_pnl_pct": 0.0,
            "historical_rr_realized": 0.0,
            "historical_outcome_reason": "Invalid risk definition",
            "loss_reason": "Invalid risk definition",
        }

    def _finalize(outcome: str, exit_price: float, exit_time: str, reason: str) -> Dict[str, Any]:
        points = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
        pnl_pct = round((points / max(entry_price, 1e-6)) * 100.0, 2)
        rr_realized = round(points / risk, 2)
        loss_reason = reason if points < 0 else _safe_str(row.get("loss_reason"))
        return {
            "historical_outcome": outcome,
            "historical_exit_time": exit_time,
            "historical_exit_price": _round_price(exit_price),
            "historical_pnl_pct": pnl_pct,
            "historical_rr_realized": rr_realized,
            "historical_outcome_reason": reason,
            "loss_reason": loss_reason,
        }

    for timestamp, bar in future_session.iterrows():
        high = _safe_float(bar.get("High"))
        low = _safe_float(bar.get("Low"))

        if direction == "LONG":
            stop_hit = low <= stop_loss
            target_2_hit = target_2 > 0 and high >= target_2
            target_1_hit = target_1 > 0 and high >= target_1
            if stop_hit and (target_1_hit or target_2_hit):
                return _finalize(
                    "STOP_LOSS",
                    stop_loss,
                    timestamp.strftime("%H:%M"),
                    "Stop and target both touched in same candle; conservative stop-first assumption",
                )
            if stop_hit:
                return _finalize("STOP_LOSS", stop_loss, timestamp.strftime("%H:%M"), "Stop loss hit before target")
            if target_2_hit:
                return _finalize("TARGET_2", target_2, timestamp.strftime("%H:%M"), "Target 2 hit")
            if target_1_hit:
                return _finalize("TARGET_1", target_1, timestamp.strftime("%H:%M"), "Target 1 hit")
        else:
            stop_hit = high >= stop_loss
            target_2_hit = target_2 > 0 and low <= target_2
            target_1_hit = target_1 > 0 and low <= target_1
            if stop_hit and (target_1_hit or target_2_hit):
                return _finalize(
                    "STOP_LOSS",
                    stop_loss,
                    timestamp.strftime("%H:%M"),
                    "Stop and target both touched in same candle; conservative stop-first assumption",
                )
            if stop_hit:
                return _finalize("STOP_LOSS", stop_loss, timestamp.strftime("%H:%M"), "Stop loss hit before target")
            if target_2_hit:
                return _finalize("TARGET_2", target_2, timestamp.strftime("%H:%M"), "Target 2 hit")
            if target_1_hit:
                return _finalize("TARGET_1", target_1, timestamp.strftime("%H:%M"), "Target 1 hit")

    if future_session.empty:
        eod_price = _safe_float(row.get("price_at_scan", row.get("entry_price")))
        eod_time = _safe_str(row.get("scan_time"))
    else:
        eod_price = _safe_float(future_session["Close"].iloc[-1], _safe_float(row.get("price_at_scan", row.get("entry_price"))))
        eod_time = future_session.index[-1].strftime("%H:%M")
    eod_points = (eod_price - entry_price) if direction == "LONG" else (entry_price - eod_price)
    if abs(eod_points) <= max(entry_price * 0.0005, 0.01):
        return _finalize("EOD_FLAT", eod_price, eod_time, "No stop/target hit; exited near flat at session close")
    if eod_points > 0:
        return _finalize("EOD_PROFIT", eod_price, eod_time, "No stop/target hit; exited in profit at session close")
    return _finalize("EOD_LOSS", eod_price, eod_time, "No stop/target hit; exited in loss at session close")


def _summarize_performance(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    actionable_rows = [
        row for row in (rows or [])
        if _safe_str(row.get("grade")).upper() in _ACTIONABLE_GRADES
        and _safe_str(row.get("historical_outcome")).upper() not in {"", "NO_ENTRY", "NO_DATA"}
    ]
    if not actionable_rows:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "target_1_hits": 0,
            "target_2_hits": 0,
            "stop_loss_hits": 0,
            "eod_profit_count": 0,
            "eod_loss_count": 0,
            "eod_flat_count": 0,
            "avg_pnl_pct": 0.0,
            "avg_rr_realized": 0.0,
        }

    wins = [
        row for row in actionable_rows
        if _safe_str(row.get("historical_outcome")).upper() in {"TARGET_1", "TARGET_2", "EOD_PROFIT"}
    ]
    losses = [
        row for row in actionable_rows
        if _safe_str(row.get("historical_outcome")).upper() in {"STOP_LOSS", "EOD_LOSS"}
    ]

    return {
        "trades": len(actionable_rows),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round((len(wins) / len(actionable_rows)) * 100.0, 2) if actionable_rows else 0.0,
        "target_1_hits": sum(1 for row in actionable_rows if _safe_str(row.get("historical_outcome")).upper() == "TARGET_1"),
        "target_2_hits": sum(1 for row in actionable_rows if _safe_str(row.get("historical_outcome")).upper() == "TARGET_2"),
        "stop_loss_hits": sum(1 for row in actionable_rows if _safe_str(row.get("historical_outcome")).upper() == "STOP_LOSS"),
        "eod_profit_count": sum(1 for row in actionable_rows if _safe_str(row.get("historical_outcome")).upper() == "EOD_PROFIT"),
        "eod_loss_count": sum(1 for row in actionable_rows if _safe_str(row.get("historical_outcome")).upper() == "EOD_LOSS"),
        "eod_flat_count": sum(1 for row in actionable_rows if _safe_str(row.get("historical_outcome")).upper() == "EOD_FLAT"),
        "avg_pnl_pct": round(sum(_safe_float(row.get("historical_pnl_pct")) for row in actionable_rows) / len(actionable_rows), 2),
        "avg_rr_realized": round(sum(_safe_float(row.get("historical_rr_realized")) for row in actionable_rows) / len(actionable_rows), 2),
    }


def build_strategy_payload(
    pulse_result: Dict[str, Any],
    direction: str = "ALL",
    grade: str = "ALL",
    limit: int = 40,
) -> Dict[str, Any]:
    pulse_stocks = list(pulse_result.get("stocks") or [])
    last_updated = _safe_str(pulse_result.get("last_updated"))
    market_data_last_updated = _safe_str(pulse_result.get("market_data_last_updated", last_updated))
    trade_date = datetime.now(IST).strftime("%Y-%m-%d")

    prepared_rows = [_prepare_strategy_input_row(row, trade_date) for row in pulse_stocks]
    strategy_rows = build_strategy_rows(prepared_rows)
    return _build_strategy_response(
        strategy_rows,
        status=_safe_str(pulse_result.get("status", "ready")),
        message=_safe_str(pulse_result.get("message")),
        last_updated=last_updated,
        market_data_last_updated=market_data_last_updated,
        benchmark_change_pct=_safe_float(pulse_result.get("benchmark_change_pct")),
        direction=direction,
        grade=grade,
        limit=limit,
    )


def build_historical_strategy_payload(
    target_date: str,
    direction: str = "ALL",
    grade: str = "ALL",
    limit: int = 40,
    include_veryweak: bool = True,
    scanner_symbols: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    try:
        trade_date = datetime.strptime(_safe_str(target_date), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date format '{target_date}': use YYYY-MM-DD") from exc

    if trade_date > datetime.now(IST).date():
        raise ValueError(f"Historical strategy date '{target_date}' cannot be in the future")

    normalized_symbols = _normalize_scanner_symbols(scanner_symbols) or _normalize_scanner_symbols(SCANNER_STOCKS)
    symbols_ns = [f"{symbol}.NS" for symbol in normalized_symbols]
    if not symbols_ns:
        return _build_strategy_response(
            [],
            status="ready",
            message=f"No scanner symbols configured for historical replay on {trade_date.isoformat()}",
            last_updated=trade_date.isoformat(),
            market_data_last_updated=trade_date.isoformat(),
            benchmark_change_pct=0.0,
            direction=direction,
            grade=grade,
            limit=limit,
            extra_fields={
                "mode": "historical",
                "requested_date": trade_date.isoformat(),
                "performance_summary": _summarize_performance([]),
                "overall_performance_summary": _summarize_performance([]),
            },
        )

    from_date = (trade_date - timedelta(days=_HISTORICAL_LOOKBACK_CALENDAR_DAYS)).isoformat()
    sessions_by_symbol: Dict[str, List[Tuple[calendar_date, pd.DataFrame]]] = {}

    for batch in _chunked(symbols_ns, MOMENTUM_PULSE_BATCH_SIZE):
        raw = _download_intraday_batch_for_range(batch, from_date=from_date, to_date=trade_date.isoformat())
        if raw is None or raw.empty:
            continue
        for symbol_ns in batch:
            frame = _get_sym_df(raw, symbol_ns)
            if frame is None or frame.empty:
                continue
            clean_symbol = symbol_ns.replace(".NS", "")
            sessions = _split_sessions(frame)
            if any(session_date == trade_date for session_date, _ in sessions):
                sessions_by_symbol[clean_symbol] = sessions

    if not sessions_by_symbol:
        return _build_strategy_response(
            [],
            status="ready",
            message=f"No 5-minute history found for {trade_date.isoformat()}",
            last_updated=trade_date.isoformat(),
            market_data_last_updated=trade_date.isoformat(),
            benchmark_change_pct=0.0,
            direction=direction,
            grade=grade,
            limit=limit,
            extra_fields={
                "mode": "historical",
                "requested_date": trade_date.isoformat(),
                "performance_summary": _summarize_performance([]),
                "overall_performance_summary": _summarize_performance([]),
            },
        )

    nifty_change_map = _build_historical_nifty_change_map(trade_date)
    scan_times = _iter_entry_window_scan_times()
    trend_state: Dict[str, Dict[str, Any]] = {}
    representative_by_symbol: Dict[str, Dict[str, Any]] = {}

    for scan_time in scan_times:
        snapshot_inputs: List[Tuple[str, pd.DataFrame, Dict[str, Any]]] = []
        for symbol, sessions in sessions_by_symbol.items():
            combined_df, session_slice, prev_close = _historical_session_view(sessions, trade_date, scan_time)
            if combined_df is None or session_slice is None or prev_close <= 0:
                continue
            snapshot = _build_historical_stock_snapshot(symbol, session_slice, prev_close)
            if snapshot:
                snapshot_inputs.append((symbol, combined_df, snapshot))

        if not snapshot_inputs:
            continue

        sector_data = _build_sector_data_from_snapshots([snapshot for _, _, snapshot in snapshot_inputs])
        scan_key = scan_time.strftime("%H:%M")
        nifty_change_pct = _safe_float(nifty_change_map.get(scan_key))

        pulse_rows: List[Dict[str, Any]] = []
        for _symbol, combined_df, snapshot in snapshot_inputs:
            pulse_row = _evaluate_symbol(
                snapshot,
                combined_df,
                nifty_change_pct,
                sector_data=sector_data,
                oi_cache={},
                trend_state=trend_state,
            )
            if pulse_row is None:
                continue
            if not include_veryweak and _safe_str(pulse_row.get("tier")).lower() == "veryweak":
                continue
            pulse_rows.append(pulse_row)

        pulse_rows.sort(
            key=lambda item: (
                _safe_float(item.get("momentum_pulse_score")),
                _safe_float(item.get("pulse_trend_strength")),
                _safe_float(item.get("direction_confidence")),
                abs(_safe_float(item.get("relative_strength"))),
            ),
            reverse=True,
        )
        for index, pulse_row in enumerate(pulse_rows, start=1):
            pulse_row["rank"] = index
            prepared = _prepare_strategy_input_row(pulse_row, trade_date.isoformat())
            strategy_row = build_strategy_row(prepared)
            strategy_row.update(_simulate_trade_outcome(strategy_row, sessions_by_symbol.get(_safe_str(strategy_row.get("symbol")).upper(), [])))
            symbol_key = _safe_str(strategy_row.get("symbol")).upper()
            representative_by_symbol[symbol_key] = _choose_representative_daily_row(
                representative_by_symbol.get(symbol_key),
                strategy_row,
            )

    strategy_rows = _sort_strategy_rows(representative_by_symbol.values())
    filtered_rows, _, _ = _filter_strategy_rows(strategy_rows, direction=direction, grade=grade, limit=limit)

    return _build_strategy_response(
        strategy_rows,
        status="ready",
        message=f"Historical strategy replay for {trade_date.isoformat()} built from 5-minute data",
        last_updated=trade_date.isoformat(),
        market_data_last_updated=trade_date.isoformat(),
        benchmark_change_pct=_safe_float(nifty_change_map.get("10:55", 0.0)),
        direction=direction,
        grade=grade,
        limit=limit,
        extra_fields={
            "mode": "historical",
            "requested_date": trade_date.isoformat(),
            "performance_summary": _summarize_performance(filtered_rows),
            "overall_performance_summary": _summarize_performance(strategy_rows),
        },
    )
