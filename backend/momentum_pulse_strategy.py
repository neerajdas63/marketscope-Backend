from __future__ import annotations

import threading
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytz

IST = pytz.timezone("Asia/Kolkata")

# Strategy entry validity window (IST).
ENTRY_WINDOW_START = time(9, 35)
ENTRY_WINDOW_END_EXCLUSIVE = time(11, 0)

_VALID_DIRECTION_FILTERS = {"ALL", "LONG", "SHORT"}
_VALID_GRADE_FILTERS = {"ALL", "A_PLUS", "A", "FAILED_OR_CHOP", "NO_TRADE"}
_ACTIONABLE_GRADES = {"A_PLUS", "A"}

# Rolling live-grade state: symbol -> {"trade_date": "YYYY-MM-DD", "grades": [...], "last_scan_time": "HH:MM"}
_GRADE_STATE_LOCK = threading.Lock()
_GRADE_STATE: Dict[str, Dict[str, Any]] = {}
_GRADE_HISTORY_MAX = 6


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


def _normalize_direction_filter(value: str) -> str:
    normalized = _safe_str(value, "ALL").upper()
    return normalized if normalized in _VALID_DIRECTION_FILTERS else "ALL"


def _normalize_grade_filter(value: str) -> str:
    normalized = _safe_str(value, "ALL").upper()
    return normalized if normalized in _VALID_GRADE_FILTERS else "ALL"


def _extract_warning_flags(row: Dict[str, Any]) -> List[str]:
    raw = row.get("warning_flags", row.get("warningflags", []))
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
        return [p for p in parts if p]
    if isinstance(raw, (list, tuple, set)):
        return [str(flag).strip() for flag in raw if str(flag).strip()]
    return []


def _major_risk_flags(row: Dict[str, Any]) -> List[str]:
    warning_flags = _extract_warning_flags(row)
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
            "rr_t1": None,
            "rr_t2": None,
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
        entry_notes.append("A+ trade: early aggressive sizing possible if breakout body strong ho" if grade == "A_PLUS" else "A trade: better to wait for confirmation or small retest")

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
        entry_notes.append("A+ trade: aggressive short possible if selling pressure real ho" if grade == "A_PLUS" else "A trade: confirmation ke baad short lena better")

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
    volume_ratio = _safe_float(row.get("volume_pace_ratio", row.get("volume_ratio")))
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
    warning_flags = [str(flag).strip().lower() for flag in _extract_warning_flags(row)]
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
            "price_at_scan": _safe_float(row.get("ltp", row.get("price_at_scan"))),
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


def _grade_value(grade: str) -> int:
    value = _safe_str(grade).upper()
    return {"A_PLUS": 4, "A": 3, "FAILED_OR_CHOP": 2, "NO_TRADE": 1}.get(value, 0)


def _grade_stability_score(history: Sequence[str]) -> float:
    grades = [str(g).upper() for g in (history or []) if str(g).strip()]
    if not grades:
        return 0.0

    values = [_grade_value(g) for g in grades]
    n = len(values)
    latest = values[-1]
    avg_val = sum(values) / max(n, 1)
    flips = sum(1 for i in range(1, n) if grades[i] != grades[i - 1])
    actionable_ratio = sum(1 for g in grades if g in _ACTIONABLE_GRADES) / max(n, 1)

    score = (avg_val / 4.0) * 55.0 + (latest / 4.0) * 25.0 + actionable_ratio * 15.0 - flips * 10.0
    return round(max(0.0, min(100.0, score)), 1)


def _update_grade_state(symbol: str, trade_date: str, grade: str, scan_time_value: Any) -> List[str]:
    symbol_key = _safe_str(symbol).upper()
    if not symbol_key:
        return []

    scan_time = _safe_str(scan_time_value)
    normalized_grade = _safe_str(grade).upper()

    with _GRADE_STATE_LOCK:
        record = _GRADE_STATE.setdefault(symbol_key, {"trade_date": trade_date, "grades": [], "last_scan_time": ""})
        if record.get("trade_date") != trade_date:
            record["trade_date"] = trade_date
            record["grades"] = []
            record["last_scan_time"] = ""

        history = list(record.get("grades") or [])
        last_grade = history[-1] if history else ""
        last_scan_time = _safe_str(record.get("last_scan_time"))

        if scan_time and scan_time == last_scan_time and normalized_grade == _safe_str(last_grade).upper():
            return history

        record["last_scan_time"] = scan_time
        history.append(normalized_grade)
        history = history[-_GRADE_HISTORY_MAX:]
        record["grades"] = history
        return list(history)


def _retest_ok(trade_side: str, price: float, vwap: float, or_high: float, or_low: float) -> bool:
    if price <= 0:
        return False
    side = _safe_str(trade_side).upper()
    if side == "LONG":
        ref = max(_safe_float(or_high), _safe_float(vwap))
        if ref <= 0:
            return False
        dist_pct = abs(price - ref) / ref * 100.0
        return price >= ref and dist_pct <= 0.35
    if side == "SHORT":
        ref = min(_safe_float(or_low), _safe_float(vwap))
        if ref <= 0:
            return False
        dist_pct = abs(price - ref) / ref * 100.0
        return price <= ref and dist_pct <= 0.35
    return False


def _or_stretch_pct(trade_side: str, price: float, or_high: float, or_low: float) -> float:
    side = _safe_str(trade_side).upper()
    if price <= 0:
        return 0.0
    if side == "LONG" and or_high > 0:
        return max(0.0, round(((price - or_high) / or_high) * 100.0, 2))
    if side == "SHORT" and or_low > 0:
        return max(0.0, round(((or_low - price) / or_low) * 100.0, 2))
    return 0.0


def _chase_risk(
    trade_side: str,
    grade: str,
    vwap_distance_pct: float,
    or_stretch_pct: float,
    risks: Sequence[str],
) -> str:
    side = _safe_str(trade_side).upper()
    normalized_grade = _safe_str(grade).upper()
    if normalized_grade not in _ACTIONABLE_GRADES or side not in {"LONG", "SHORT"}:
        return "NA"

    abs_vwap = abs(_safe_float(vwap_distance_pct))
    abs_or = max(0.0, _safe_float(or_stretch_pct))
    risk_set = {str(r).strip().lower() for r in (risks or [])}

    if "far_from_vwap" in risk_set or "extended" in risk_set or "one_bar_spike" in risk_set:
        return "HIGH"
    if abs_vwap >= 1.40 or abs_or >= 1.00:
        return "HIGH"
    if abs_vwap >= 0.95 or abs_or >= 0.65:
        return "MEDIUM"
    if abs_vwap <= 0.55 and abs_or <= 0.45:
        return "LOW"
    return "MEDIUM"


def _entry_state(
    trade_side: str,
    grade: str,
    chase_risk: str,
    retest_ok: bool,
    momentum_pulse_score: float,
    pulse_trend_label: str,
    direction_confidence: float,
    volume_ratio: float,
    range_ratio: float,
    risks: Sequence[str],
) -> str:
    normalized_grade = _safe_str(grade).upper()
    side = _safe_str(trade_side).upper()
    if normalized_grade not in _ACTIONABLE_GRADES or side not in {"LONG", "SHORT"}:
        return "CANCEL_SETUP"
    if _safe_str(chase_risk).upper() == "HIGH":
        return "AVOID_CHASE"
    if retest_ok:
        return "ENTER_ON_RETEST"

    risk_set = {str(r).strip().lower() for r in (risks or [])}
    trend_label = _safe_str(pulse_trend_label).lower()
    healthy = (
        _safe_float(momentum_pulse_score) >= 58.0
        and _safe_float(direction_confidence) >= 14.0
        and trend_label != "falling"
        and _safe_float(volume_ratio) >= 1.0
        and _safe_float(range_ratio) >= 1.0
        and "momentum_decay" not in risk_set
        and "fading_score" not in risk_set
        and "low_consistency" not in risk_set
    )
    if _safe_str(chase_risk).upper() == "LOW" and healthy:
        return "ENTER_NOW"
    return "WAIT_CONFIRMATION"


def _execution_rank(
    trade_side: str,
    grade: str,
    momentum_pulse_score: float,
    grade_stability_score: float,
    direction_confidence: float,
    volume_ratio: float,
    range_ratio: float,
    vwap_distance_pct: float,
    chase_risk: str,
    entry_state: str,
    risks: Sequence[str],
    retest_ok: bool,
) -> float:
    side = _safe_str(trade_side).upper()
    normalized_grade = _safe_str(grade).upper()

    grade_points = {"A_PLUS": 32.0, "A": 24.0, "FAILED_OR_CHOP": 8.0, "NO_TRADE": 0.0}.get(normalized_grade, 0.0)
    momentum_component = max(0.0, min(100.0, _safe_float(momentum_pulse_score))) * 0.35
    stability_component = max(0.0, min(100.0, _safe_float(grade_stability_score))) * 0.18
    confidence_component = max(0.0, min(100.0, _safe_float(direction_confidence))) * 0.08

    vol = max(0.0, min(3.0, _safe_float(volume_ratio)))
    rng = max(0.0, min(3.0, _safe_float(range_ratio)))
    volume_component = (vol / 3.0) * 10.0
    range_component = (rng / 3.0) * 8.0

    score = grade_points + momentum_component + stability_component + confidence_component + volume_component + range_component
    if retest_ok:
        score += 4.0

    entry_bonus = {
        "ENTER_NOW": 6.0,
        "ENTER_ON_RETEST": 5.0,
        "WAIT_CONFIRMATION": 2.0,
        "AVOID_CHASE": -6.0,
        "CANCEL_SETUP": -12.0,
    }.get(_safe_str(entry_state).upper(), 0.0)
    score += entry_bonus

    abs_vwap = abs(_safe_float(vwap_distance_pct))
    if abs_vwap > 0.60:
        score -= min(24.0, (abs_vwap - 0.60) * 10.0)
    if abs_vwap > 1.40:
        score -= min(12.0, (abs_vwap - 1.40) * 8.0)

    chase_penalty = {"HIGH": 14.0, "MEDIUM": 7.0, "LOW": 0.0, "NA": 10.0}.get(_safe_str(chase_risk).upper(), 6.0)
    score -= chase_penalty

    risk_weights = {
        "far_from_vwap": 10.0,
        "momentum_decay": 7.0,
        "one_bar_spike": 8.0,
        "low_consistency": 6.0,
        "weak_relative_strength": 6.0,
        "fading_score": 5.0,
        "low_volume_confirmation": 4.0,
        "extended": 8.0,
        "long_failed_fast": 10.0,
        "short_failed_fast": 10.0,
    }
    risk_set = {str(r).strip().lower() for r in (risks or [])}
    for key, weight in risk_weights.items():
        if key in risk_set:
            score -= weight

    if side not in {"LONG", "SHORT"}:
        score -= 10.0

    return round(max(0.0, min(100.0, score)), 2)


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

    result["direction"] = _safe_str(row.get("direction", result.get("trade_side"))).upper()
    result["momentum_pulse_score"] = _safe_float(row.get("momentum_pulse_score"))
    result["warning_flags"] = _extract_warning_flags(row)

    trade_date = _safe_str(result.get("trade_date")) or datetime.now(IST).strftime("%Y-%m-%d")
    symbol = _safe_str(result.get("symbol")).upper()
    grade = _safe_str(result.get("grade")).upper()

    grade_history = _update_grade_state(symbol, trade_date, grade, result.get("scan_time"))
    stability_score = _grade_stability_score(grade_history)

    trade_side = _safe_str(result.get("trade_side")).upper()
    price = _safe_float(result.get("price_at_scan", result.get("ltp")))
    vwap = _safe_float(result.get("vwap"))
    or_high = _safe_float(result.get("or_high"))
    or_low = _safe_float(result.get("or_low"))
    vwap_dist = _safe_float(result.get("vwap_distance_pct"))
    volume_ratio = _safe_float(result.get("volume_ratio"))
    range_ratio = _safe_float(result.get("range_ratio"))
    direction_confidence = _safe_float(row.get("direction_confidence"))
    pulse_trend_label = _safe_str(row.get("pulse_trend_label"))
    risks = _major_risk_flags(result)

    retest_ok = _retest_ok(trade_side, price, vwap, or_high, or_low)
    or_stretch = _or_stretch_pct(trade_side, price, or_high, or_low)
    chase_risk = _chase_risk(trade_side, grade, vwap_dist, or_stretch, risks)
    entry_state = _entry_state(
        trade_side=trade_side,
        grade=grade,
        chase_risk=chase_risk,
        retest_ok=retest_ok,
        momentum_pulse_score=_safe_float(result.get("momentum_pulse_score")),
        pulse_trend_label=pulse_trend_label,
        direction_confidence=direction_confidence,
        volume_ratio=volume_ratio,
        range_ratio=range_ratio,
        risks=risks,
    )
    execution_rank = _execution_rank(
        trade_side=trade_side,
        grade=grade,
        momentum_pulse_score=_safe_float(result.get("momentum_pulse_score")),
        grade_stability_score=stability_score,
        direction_confidence=direction_confidence,
        volume_ratio=volume_ratio,
        range_ratio=range_ratio,
        vwap_distance_pct=vwap_dist,
        chase_risk=chase_risk,
        entry_state=entry_state,
        risks=risks,
        retest_ok=retest_ok,
    )

    result["grade_history"] = grade_history
    result["grade_stability_score"] = stability_score
    result["retest_ok"] = bool(retest_ok)
    result["chase_risk"] = chase_risk
    result["entry_state"] = entry_state
    result["execution_rank"] = execution_rank
    result["or_stretch_pct"] = or_stretch
    result["major_risks"] = list(risks)

    return result


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
            _safe_float(x.get("execution_rank")),
            grade_priority.get(_safe_str(x.get("grade")).upper(), 0),
            _safe_float(x.get("score")),
            _safe_float(x.get("volume_ratio")),
            _safe_float(x.get("range_ratio")),
        ),
        reverse=True,
    )
    return output


def build_strategy_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _sort_strategy_rows([build_strategy_row(row) for row in (rows or [])])


def summarize_strategy(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows or [])
    a_plus = [r for r in rows if _safe_str(r.get("grade")).upper() == "A_PLUS"]
    a_only = [r for r in rows if _safe_str(r.get("grade")).upper() == "A"]
    failed = [r for r in rows if _safe_str(r.get("grade")).upper() == "FAILED_OR_CHOP"]
    no_trade = [r for r in rows if _safe_str(r.get("grade")).upper() == "NO_TRADE"]
    longs = [r for r in rows if _safe_str(r.get("trade_side")).upper() == "LONG"]
    shorts = [r for r in rows if _safe_str(r.get("trade_side")).upper() == "SHORT"]
    enter_now = [r for r in rows if _safe_str(r.get("entry_state")).upper() == "ENTER_NOW"]
    enter_retest = [r for r in rows if _safe_str(r.get("entry_state")).upper() == "ENTER_ON_RETEST"]
    avoid = [r for r in rows if _safe_str(r.get("entry_state")).upper() in {"AVOID_CHASE", "CANCEL_SETUP"}]

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
        "enter_now_count": len(enter_now),
        "enter_on_retest_count": len(enter_retest),
        "avoid_count": len(avoid),
        "avg_volume_ratio": avg(rows, "volume_ratio"),
        "avg_range_ratio": avg(rows, "range_ratio"),
        "avg_execution_rank": avg(rows, "execution_rank"),
        "avg_abs_change_pct": round(
            sum(abs(_safe_float(r.get("change_pct_at_scan", r.get("changepct", r.get("change_pct"))))) for r in rows) / len(rows),
            2
        ) if rows else 0.0,
        "a_plus_common": {
            "avg_score": avg(a_plus, "score"),
            "avg_vwap_dist": avg(a_plus, "vwap_distance_pct"),
            "avg_volume_ratio": avg(a_plus, "volume_ratio"),
            "avg_range_ratio": avg(a_plus, "range_ratio"),
            "avg_execution_rank": avg(a_plus, "execution_rank"),
        },
        "a_common": {
            "avg_score": avg(a_only, "score"),
            "avg_vwap_dist": avg(a_only, "vwap_distance_pct"),
            "avg_volume_ratio": avg(a_only, "volume_ratio"),
            "avg_range_ratio": avg(a_only, "range_ratio"),
            "avg_execution_rank": avg(a_only, "execution_rank"),
        },
    }


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


def _thin_row_for_bucket(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "symbol": _safe_str(row.get("symbol")),
        "scan_time": _safe_str(row.get("scan_time")),
        "trade_side": _safe_str(row.get("trade_side")),
        "grade": _safe_str(row.get("grade")),
        "entry_state": _safe_str(row.get("entry_state")),
        "execution_rank": _safe_float(row.get("execution_rank")),
        "price_at_scan": row.get("price_at_scan"),
        "vwap_distance_pct": _safe_float(row.get("vwap_distance_pct")),
        "volume_ratio": _safe_float(row.get("volume_ratio")),
        "range_ratio": _safe_float(row.get("range_ratio")),
        "entry_price": row.get("entry_price"),
        "stop_loss": row.get("stop_loss"),
        "target_1": row.get("target_1"),
        "target_2": row.get("target_2"),
        "rr_t1": row.get("rr_t1"),
        "rr_t2": row.get("rr_t2"),
        "or_high": row.get("or_high"),
        "or_low": row.get("or_low"),
        "retest_ok": row.get("retest_ok"),
        "chase_risk": _safe_str(row.get("chase_risk")),
        "grade_stability_score": _safe_float(row.get("grade_stability_score")),
        "reasons": list(row.get("reasons") or [])[:6],
        "major_risks": list(row.get("major_risks") or [])[:6],
    }


def build_best_stock_buckets(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = _sort_strategy_rows(rows)

    def actionable(row: Dict[str, Any]) -> bool:
        return (
            _safe_str(row.get("grade")).upper() in _ACTIONABLE_GRADES
            and _safe_str(row.get("trade_side")).upper() in {"LONG", "SHORT"}
            and _safe_str(row.get("entry_state")).upper() not in {"AVOID_CHASE", "CANCEL_SETUP"}
        )

    actionable_rows = [r for r in ordered if actionable(r)]

    overall_best = [_thin_row_for_bucket(r) for r in actionable_rows[:8]]
    best_longs = [
        _thin_row_for_bucket(r)
        for r in actionable_rows
        if _safe_str(r.get("trade_side")).upper() == "LONG"
    ][:6]
    best_shorts = [
        _thin_row_for_bucket(r)
        for r in actionable_rows
        if _safe_str(r.get("trade_side")).upper() == "SHORT"
    ][:6]

    avoid_list = [
        _thin_row_for_bucket(r)
        for r in ordered
        if _safe_str(r.get("entry_state")).upper() in {"AVOID_CHASE", "CANCEL_SETUP"}
        or _safe_str(r.get("grade")).upper() in {"FAILED_OR_CHOP", "NO_TRADE"}
    ][:12]

    return {
        "overall_best": overall_best,
        "best_longs": best_longs,
        "best_shorts": best_shorts,
        "avoid_list": avoid_list,
    }


def build_live_strategy_payload(
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
    filtered_rows, normalized_direction, normalized_grade = _filter_strategy_rows(
        strategy_rows,
        direction=direction,
        grade=grade,
        limit=limit,
    )

    status = _safe_str(pulse_result.get("status", "ready"))
    message = _safe_str(pulse_result.get("message"))
    benchmark_change_pct = round(_safe_float(pulse_result.get("benchmark_change_pct")), 2)

    return {
        "feature": "Momentum Pulse Strategy",
        "feature_key": "momentum_pulse_strategy",
        "mode": "live",
        "status": status,
        "message": message,
        "last_updated": last_updated,
        "market_data_last_updated": market_data_last_updated,
        "benchmark_change_pct": benchmark_change_pct,
        "direction": normalized_direction,
        "grade": normalized_grade,
        "rows": filtered_rows,
        "total": len(filtered_rows),
        "total_candidates": len(strategy_rows),
        "summary": summarize_strategy(filtered_rows),
        "overall_summary": summarize_strategy(strategy_rows),
        "best_stocks": build_best_stock_buckets(strategy_rows),
        "available_directions": ["ALL", "LONG", "SHORT"],
        "available_grades": ["ALL", "A_PLUS", "A", "FAILED_OR_CHOP", "NO_TRADE"],
    }


def build_strategy_payload(
    pulse_result: Dict[str, Any],
    direction: str = "ALL",
    grade: str = "ALL",
    limit: int = 40,
) -> Dict[str, Any]:
    # Backward-compatible wrapper used by existing route integrations.
    return build_live_strategy_payload(pulse_result=pulse_result, direction=direction, grade=grade, limit=limit)


def build_historical_strategy_payload(
    target_date: str,
    direction: str = "ALL",
    grade: str = "ALL",
    limit: int = 40,
    include_veryweak: bool = True,
    scanner_symbols: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    # Historical replay is intentionally disabled in this module to keep 512MB deployments safe.
    # This function remains for route compatibility only.
    _ = include_veryweak
    _ = scanner_symbols

    filtered_rows: List[Dict[str, Any]] = []
    normalized_direction = _normalize_direction_filter(direction)
    normalized_grade = _normalize_grade_filter(grade)

    return {
        "feature": "Momentum Pulse Strategy",
        "feature_key": "momentum_pulse_strategy",
        "mode": "historical",
        "status": "disabled",
        "message": f"Historical replay is disabled on this deployment (requested_date={_safe_str(target_date)})",
        "last_updated": "",
        "market_data_last_updated": "",
        "benchmark_change_pct": 0.0,
        "direction": normalized_direction,
        "grade": normalized_grade,
        "rows": filtered_rows,
        "total": 0,
        "total_candidates": 0,
        "summary": summarize_strategy(filtered_rows),
        "overall_summary": summarize_strategy([]),
        "best_stocks": build_best_stock_buckets([]),
        "available_directions": ["ALL", "LONG", "SHORT"],
        "available_grades": ["ALL", "A_PLUS", "A", "FAILED_OR_CHOP", "NO_TRADE"],
    }

