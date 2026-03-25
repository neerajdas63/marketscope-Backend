import threading
from datetime import datetime
from typing import Any, Dict, List, Sequence

import pytz

from backend.momentum_pulse import get_momentum_pulse
from runtime_state import load_json_state, save_json_state
from stocks import SECTORS

IST = pytz.timezone("Asia/Kolkata")
DEFAULT_FETCH_LIMIT = 120
TOP_ZONE_RANK = 15
LEADER_REPLACE_MARGIN = 4.0
LEADER_CONFIRMATION_STEPS = 2
SECTOR_TAB_LIMIT = 5
SECTOR_AGGREGATE_DEPTH = 3

_navigator_lock = threading.RLock()
_DEFAULT_NAVIGATOR_STATE: Dict[str, Any] = {
    "source_key": "",
    "current_ranks": {},
    "fresh_symbols": [],
    "leader_session_key": "",
    "leaders": {
        "LONG": {"symbol": "", "challenger": "", "challenger_steps": 0},
        "SHORT": {"symbol": "", "challenger": "", "challenger_steps": 0},
    },
}
_navigator_state: Dict[str, Any] = {
    **_DEFAULT_NAVIGATOR_STATE,
    **dict(load_json_state("pulse_navigator_state.json", {}) or {}),
}


def _persist_navigator_state() -> None:
    with _navigator_lock:
        save_json_state(
            "pulse_navigator_state.json",
            {
                "source_key": str(_navigator_state.get("source_key") or ""),
                "current_ranks": dict(_navigator_state.get("current_ranks") or {}),
                "fresh_symbols": list(_navigator_state.get("fresh_symbols") or []),
                "leader_session_key": str(_navigator_state.get("leader_session_key") or ""),
                "leaders": dict(_navigator_state.get("leaders") or {}),
            },
        )

_ACTIONABILITY_PRIORITY = {
    "clean_setup": 3,
    "needs_pullback": 2,
    "extended": 1,
    "risky_spike": 0,
}

_BEHAVIOR_PRIORITY = {
    "EARLY": 4,
    "ACTIVE": 3,
    "LATE": 2,
    "EXTENDED": 0,
}

_PRESET_CONFIG: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "min_score": 56.0,
        "max_warning_count": 2,
        "fo_only": False,
        "allow_extended": True,
    },
    "safe": {
        "min_score": 62.0,
        "max_warning_count": 1,
        "fo_only": False,
        "allow_extended": False,
    },
    "aggressive": {
        "min_score": 50.0,
        "max_warning_count": 3,
        "fo_only": False,
        "allow_extended": True,
    },
    "fo_focus": {
        "min_score": 56.0,
        "max_warning_count": 2,
        "fo_only": True,
        "allow_extended": True,
    },
}

_SECTOR_BY_SYMBOL = {
    symbol.replace(".NS", ""): sector_name
    for sector_name, symbols in SECTORS.items()
    for symbol in symbols
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_direction(direction: str) -> str:
    value = str(direction or "ALL").strip().upper()
    return value if value in {"ALL", "LONG", "SHORT"} else "ALL"


def _normalize_preset(preset: str) -> str:
    value = str(preset or "balanced").strip().lower()
    return value if value in _PRESET_CONFIG else "balanced"


def _source_key(last_updated: str, total_items: int) -> str:
    return f"{datetime.now(IST).date().isoformat()}|{last_updated}|{total_items}"


def _rank_key(item: Dict[str, Any]) -> tuple:
    return (
        _ACTIONABILITY_PRIORITY.get(str(item.get("actionability_label")), 0),
        _safe_float(item.get("momentum_pulse_score")),
        _safe_float(item.get("pulse_trend_strength")),
        _safe_float(item.get("direction_confidence")),
        abs(_safe_float(item.get("relative_strength"))),
    )


def _leader_session_key() -> str:
    return datetime.now(IST).date().isoformat()


def _avg_score_history(item: Dict[str, Any]) -> float:
    history = [_safe_float(value) for value in item.get("score_history") or []]
    if not history:
        return _safe_float(item.get("momentum_pulse_score"))
    return round(sum(history) / len(history), 1)


def _directional_consistency_score(item: Dict[str, Any]) -> float:
    direction = str(item.get("direction") or "NEUTRAL")
    if direction == "LONG":
        return _safe_float(item.get("long_directional_consistency_score"))
    if direction == "SHORT":
        return _safe_float(item.get("short_directional_consistency_score"))
    return 0.0


def _directional_vwap_score(item: Dict[str, Any]) -> float:
    direction = str(item.get("direction") or "NEUTRAL")
    if direction == "LONG":
        return _safe_float(item.get("long_vwap_alignment_score"))
    if direction == "SHORT":
        return _safe_float(item.get("short_vwap_alignment_score"))
    return 0.0


def _session_leader_score(item: Dict[str, Any]) -> float:
    direction = str(item.get("direction") or "NEUTRAL")
    if direction not in {"LONG", "SHORT"}:
        return 0.0

    current_score = _safe_float(item.get("momentum_pulse_score"))
    average_score = _avg_score_history(item)
    consistency_score = _directional_consistency_score(item)
    vwap_score = _directional_vwap_score(item)
    direction_confidence = _safe_float(item.get("direction_confidence"))
    pulse_trend_strength = _safe_float(item.get("pulse_trend_strength"))
    relative_strength = _safe_float(item.get("relative_strength"))
    aligned_relative_strength = relative_strength if direction == "LONG" else -relative_strength
    relative_strength_score = max(0.0, min(100.0, aligned_relative_strength * 28.0 + 50.0))
    behavior_bonus = _BEHAVIOR_PRIORITY.get(str(item.get("behavior_state") or ""), 0) * 2.5
    warning_penalty = int(item.get("warning_count", 0) or 0) * 3.5
    extension_penalty = 7.0 if bool(item.get("is_extended")) else 0.0
    spike_penalty = 10.0 if "one_bar_spike" in {str(flag) for flag in item.get("warning_flags") or []} else 0.0

    score = (
        current_score * 0.28
        + average_score * 0.24
        + direction_confidence * 0.14
        + relative_strength_score * 0.16
        + consistency_score * 0.10
        + vwap_score * 0.05
        + pulse_trend_strength * 0.03
        + behavior_bonus
        - warning_penalty
        - extension_penalty
        - spike_penalty
    )
    return round(max(0.0, min(100.0, score)), 1)


def _leader_rank_key(item: Dict[str, Any]) -> tuple:
    direction = str(item.get("direction") or "NEUTRAL")
    relative_strength = _safe_float(item.get("relative_strength"))
    aligned_relative_strength = relative_strength if direction == "LONG" else -relative_strength
    return (
        _safe_float(item.get("session_leader_score")),
        _avg_score_history(item),
        aligned_relative_strength,
        _safe_float(item.get("direction_confidence")),
        -int(item.get("warning_count", 0) or 0),
    )


def _aligned_relative_strength(item: Dict[str, Any]) -> float:
    direction = str(item.get("direction") or "NEUTRAL")
    relative_strength = _safe_float(item.get("relative_strength"))
    if direction == "LONG":
        return relative_strength
    if direction == "SHORT":
        return -relative_strength
    return 0.0


def _stock_opportunity_score(item: Dict[str, Any]) -> float:
    actionability_label = str(item.get("actionability_label") or "needs_pullback")
    actionability_bonus = {
        "clean_setup": 8.0,
        "needs_pullback": 3.0,
        "extended": -4.0,
        "risky_spike": -12.0,
    }.get(actionability_label, 0.0)
    behavior_bonus = _BEHAVIOR_PRIORITY.get(str(item.get("behavior_state") or ""), 0) * 1.5
    aligned_relative_strength = _aligned_relative_strength(item)
    relative_strength_score = max(0.0, min(100.0, aligned_relative_strength * 24.0 + 50.0))
    warning_penalty = int(item.get("warning_count", 0) or 0) * 2.5
    extension_penalty = 4.0 if bool(item.get("is_extended")) else 0.0

    score = (
        _safe_float(item.get("momentum_pulse_score")) * 0.52
        + _safe_float(item.get("pulse_trend_strength")) * 0.16
        + _safe_float(item.get("direction_confidence")) * 0.14
        + relative_strength_score * 0.10
        + behavior_bonus
        + actionability_bonus
        - warning_penalty
        - extension_penalty
    )
    return round(max(0.0, min(100.0, score)), 1)


def _sector_stock_rank_key(item: Dict[str, Any]) -> tuple:
    return (
        _safe_float(item.get("sector_opportunity_score")),
        _leader_rank_key(item),
    )


def _leader_reason(item: Dict[str, Any]) -> str:
    direction = str(item.get("direction") or "NEUTRAL")
    consistency_score = _directional_consistency_score(item)
    relative_strength = _safe_float(item.get("relative_strength"))
    distance_from_vwap_pct = _safe_float(item.get("distance_from_vwap_pct"))

    parts: List[str] = []
    if direction == "LONG" and relative_strength > 0.5:
        parts.append(f"holding {relative_strength:.2f}% RS vs Nifty")
    elif direction == "SHORT" and relative_strength < -0.5:
        parts.append(f"leading downside by {abs(relative_strength):.2f}% vs Nifty")

    if consistency_score >= 65.0:
        parts.append("trend has stayed consistent through the session")
    if abs(distance_from_vwap_pct) <= 0.9:
        if direction == "LONG":
            parts.append("still holding cleanly around VWAP")
        elif direction == "SHORT":
            parts.append("still respecting pressure around VWAP")
    if str(item.get("behavior_state") or "") in {"EARLY", "ACTIVE"}:
        parts.append("leadership is still active")

    if not parts:
        return "session leader based on sustained score, relative strength, and trend quality"
    return "; ".join(parts[:3])


def _fresh_reason(item: Dict[str, Any]) -> str:
    score_change_10m = _safe_float(item.get("score_change_10m"))
    pulse_trend_strength = _safe_float(item.get("pulse_trend_strength"))
    direction = str(item.get("direction") or "NEUTRAL")
    bias_text = "bullish" if direction == "LONG" else "bearish" if direction == "SHORT" else "directional"
    return (
        f"fresh {bias_text} improvement with {score_change_10m:.1f} score gain in 10m "
        f"and trend strength {pulse_trend_strength:.1f}"
    )


def _select_stable_session_leader(items: Sequence[Dict[str, Any]], direction: str) -> Dict[str, Any] | None:
    candidates = [
        item for item in items
        if str(item.get("direction")) == direction
        and str(item.get("actionability_label")) != "risky_spike"
        and _safe_float(item.get("session_leader_score")) >= 50.0
    ]
    if not candidates:
        return None

    candidates.sort(key=_leader_rank_key, reverse=True)
    preferred = candidates[0]
    session_key = _leader_session_key()

    with _navigator_lock:
        if str(_navigator_state.get("leader_session_key") or "") != session_key:
            _navigator_state["leader_session_key"] = session_key
            _navigator_state["leaders"] = {
                "LONG": {"symbol": "", "challenger": "", "challenger_steps": 0},
                "SHORT": {"symbol": "", "challenger": "", "challenger_steps": 0},
            }

        leader_state = dict((_navigator_state.get("leaders") or {}).get(direction) or {})
        incumbent_symbol = str(leader_state.get("symbol") or "")
        incumbent = next((item for item in candidates if str(item.get("symbol") or "") == incumbent_symbol), None)

        if incumbent is None:
            _navigator_state["leaders"][direction] = {
                "symbol": str(preferred.get("symbol") or ""),
                "challenger": "",
                "challenger_steps": 0,
            }
            _persist_navigator_state()
            return preferred

        incumbent_score = _safe_float(incumbent.get("session_leader_score"))
        preferred_score = _safe_float(preferred.get("session_leader_score"))
        if str(preferred.get("symbol") or "") == incumbent_symbol or preferred_score < incumbent_score + LEADER_REPLACE_MARGIN:
            _navigator_state["leaders"][direction] = {
                "symbol": incumbent_symbol,
                "challenger": "",
                "challenger_steps": 0,
            }
            _persist_navigator_state()
            return incumbent

        challenger_symbol = str(preferred.get("symbol") or "")
        previous_challenger = str(leader_state.get("challenger") or "")
        previous_steps = int(leader_state.get("challenger_steps", 0) or 0)
        challenger_steps = previous_steps + 1 if challenger_symbol == previous_challenger else 1

        if challenger_steps >= LEADER_CONFIRMATION_STEPS:
            _navigator_state["leaders"][direction] = {
                "symbol": challenger_symbol,
                "challenger": "",
                "challenger_steps": 0,
            }
            _persist_navigator_state()
            return preferred

        _navigator_state["leaders"][direction] = {
            "symbol": incumbent_symbol,
            "challenger": challenger_symbol,
            "challenger_steps": challenger_steps,
        }
        _persist_navigator_state()
        return incumbent


def _market_mode(items: Sequence[Dict[str, Any]]) -> str:
    if not items:
        return "Quiet"

    long_count = sum(1 for item in items if str(item.get("direction")) == "LONG")
    short_count = sum(1 for item in items if str(item.get("direction")) == "SHORT")
    strong_count = sum(1 for item in items if _safe_float(item.get("momentum_pulse_score")) >= 60.0)
    clean_count = sum(1 for item in items if str(item.get("actionability_label")) == "clean_setup")

    if strong_count <= 6:
        return "Narrow"
    if long_count >= max(short_count * 1.6, 8) and clean_count >= 4:
        return "Trend Up"
    if short_count >= max(long_count * 1.6, 8) and clean_count >= 4:
        return "Trend Down"
    return "Rotational"


def _build_reasons(item: Dict[str, Any]) -> List[str]:
    direction = str(item.get("direction") or "NEUTRAL")
    relative_strength = _safe_float(item.get("relative_strength"))
    volume_pace_ratio = _safe_float(item.get("volume_pace_ratio"))
    range_expansion_ratio = _safe_float(item.get("range_expansion_ratio"))
    score_change_10m = _safe_float(item.get("score_change_10m"))
    distance_from_vwap_pct = _safe_float(item.get("distance_from_vwap_pct"))

    candidates: List[tuple[float, str]] = []
    if volume_pace_ratio >= 1.2:
        candidates.append((volume_pace_ratio, f"Volume pace {volume_pace_ratio:.2f}x same-time average"))
    if range_expansion_ratio >= 1.15:
        candidates.append((range_expansion_ratio, f"Range expansion {range_expansion_ratio:.2f}x normal"))
    if direction == "LONG" and relative_strength > 0.2:
        candidates.append((relative_strength + 1.0, f"Beating Nifty by {relative_strength:.2f}%"))
    if direction == "SHORT" and relative_strength < -0.2:
        candidates.append((abs(relative_strength) + 1.0, f"Underperforming Nifty by {abs(relative_strength):.2f}%"))
    if str(item.get("pulse_trend_label")) == "Rising" and score_change_10m > 0:
        candidates.append((score_change_10m + 0.5, f"Pulse score up {score_change_10m:.1f} in last 10m"))
    if bool(item.get("trend_consistent")):
        candidates.append((1.4, "Directional candles are holding consistently"))
    if abs(distance_from_vwap_pct) <= 0.6 and not bool(item.get("is_extended")):
        candidates.append((1.2, "Still trading close to VWAP"))

    if not candidates:
        candidates.append((_safe_float(item.get("direction_confidence")) / 100.0, "Directional bias is improving"))

    candidates.sort(key=lambda entry: entry[0], reverse=True)
    return [text for _, text in candidates[:3]]


def _actionability_label(item: Dict[str, Any]) -> str:
    warning_flags = {str(flag) for flag in item.get("warning_flags") or []}
    if "one_bar_spike" in warning_flags:
        return "risky_spike"
    if bool(item.get("is_extended")) or "far_from_vwap" in warning_flags:
        return "extended"
    if (
        str(item.get("direction")) in {"LONG", "SHORT"}
        and _safe_float(item.get("direction_confidence")) >= 55.0
        and bool(item.get("trend_consistent"))
        and str(item.get("pulse_trend_label")) == "Rising"
        and len(warning_flags) <= 1
    ):
        return "clean_setup"
    return "needs_pullback"


def _decorate_item(item: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(item)
    symbol = str(enriched.get("symbol") or "")
    warning_flags = [str(flag) for flag in enriched.get("warning_flags") or []]
    enriched["sector"] = str(enriched.get("sector") or _SECTOR_BY_SYMBOL.get(symbol) or "")
    enriched["warning_flags"] = warning_flags
    enriched["warning_count"] = len(warning_flags)
    enriched["reasons"] = _build_reasons(enriched)
    enriched["actionability_label"] = _actionability_label(enriched)
    enriched["session_leader_score"] = _session_leader_score(enriched)
    enriched["sector_aligned_relative_strength"] = round(_aligned_relative_strength(enriched), 2)
    enriched["sector_opportunity_score"] = _stock_opportunity_score(enriched)
    enriched["leader_reason"] = _leader_reason(enriched)
    enriched["ui_tags"] = [
        str(enriched.get("direction") or "NEUTRAL"),
        str(enriched.get("tier") or "veryweak"),
        str(enriched.get("actionability_label") or "needs_pullback"),
    ]
    return enriched


def _apply_preset_filters(
    items: Sequence[Dict[str, Any]],
    preset: str,
    direction: str,
) -> List[Dict[str, Any]]:
    config = _PRESET_CONFIG[preset]
    filtered = list(items)

    if direction in {"LONG", "SHORT"}:
        filtered = [item for item in filtered if str(item.get("direction")) == direction]
    if config["fo_only"]:
        filtered = [item for item in filtered if bool(item.get("fo"))]

    filtered = [
        item for item in filtered
        if _safe_float(item.get("momentum_pulse_score")) >= _safe_float(config.get("min_score"))
        and int(item.get("warning_count", 0) or 0) <= int(config.get("max_warning_count", 99))
    ]
    if not config["allow_extended"]:
        filtered = [item for item in filtered if str(item.get("actionability_label")) != "extended"]

    filtered.sort(key=_rank_key, reverse=True)
    return filtered


def _slice(items: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    return list(items[:max(limit, 0)]) if limit > 0 else list(items)


def _build_discover_tab(items: Sequence[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    clean_setups = [item for item in items if str(item.get("actionability_label")) == "clean_setup"]
    early_movers = [item for item in items if str(item.get("time_context_bucket")) == "DISCOVERY"]
    trend_continuation = [
        item for item in items
        if str(item.get("time_context_bucket")) == "TREND" and str(item.get("pulse_trend_label")) == "Rising"
    ]
    late_strength = [
        item for item in items
        if str(item.get("time_context_bucket")) == "LATE" and str(item.get("direction")) in {"LONG", "SHORT"}
    ]

    return {
        "tab": "discover",
        "title": "Discover",
        "buckets": [
            {"id": "curated_now", "title": "Curated Now", "stocks": _slice(items, limit)},
            {"id": "clean_setups", "title": "Clean Setups", "stocks": _slice(clean_setups, limit)},
            {"id": "early_movers", "title": "Early Movers", "stocks": _slice(early_movers, limit)},
            {"id": "trend_continuation", "title": "Trend Continuation", "stocks": _slice(trend_continuation, limit)},
            {"id": "late_strength", "title": "Late Strength", "stocks": _slice(late_strength, limit)},
        ],
    }


def _detect_fresh_entries(items: Sequence[Dict[str, Any]], source_key: str) -> List[str]:
    current_ranks = {
        str(item.get("symbol") or ""): index
        for index, item in enumerate(items, start=1)
        if item.get("symbol")
    }
    with _navigator_lock:
        previous_ranks = dict(_navigator_state.get("current_ranks") or {})
        if str(_navigator_state.get("source_key") or "") != source_key:
            fresh_symbols = [
                symbol
                for symbol, rank in current_ranks.items()
                if rank <= TOP_ZONE_RANK and (previous_ranks.get(symbol) is None or int(previous_ranks.get(symbol)) > TOP_ZONE_RANK)
            ]
            _navigator_state["source_key"] = source_key
            _navigator_state["current_ranks"] = current_ranks
            _navigator_state["fresh_symbols"] = fresh_symbols
            _persist_navigator_state()
        return list(_navigator_state.get("fresh_symbols") or [])


def _build_fresh_tab(items: Sequence[Dict[str, Any]], source_key: str, limit: int) -> Dict[str, Any]:
    fresh_symbols = set(_detect_fresh_entries(items, source_key))
    selection_mode = "top_zone_entry"
    fresh_candidates = [
        item for item in items
        if str(item.get("symbol") or "") in fresh_symbols
        and (_safe_float(item.get("score_change_10m")) > 0 or bool(item.get("improving_now")))
    ]
    if not fresh_candidates:
        selection_mode = "trend_improvers"
        fresh_candidates = [
            item for item in items
            if str(item.get("direction") or "") in {"LONG", "SHORT"}
            and str(item.get("pulse_trend_label") or "") == "Rising"
            and (
                _safe_float(item.get("score_change_10m")) >= 1.0
                or bool(item.get("improving_now"))
            )
        ]
    if not fresh_candidates:
        selection_mode = "active_directional_strength"
        fresh_candidates = [
            item for item in items
            if str(item.get("direction") or "") in {"LONG", "SHORT"}
            and str(item.get("behavior_state") or "") in {"EARLY", "ACTIVE"}
            and _safe_float(item.get("momentum_pulse_score")) >= 58.0
            and _safe_float(item.get("pulse_trend_strength")) >= 28.0
        ]
    fresh_candidates.sort(
        key=lambda item: (
            _safe_float(item.get("score_change_10m")),
            _safe_float(item.get("pulse_trend_strength")),
            _safe_float(item.get("momentum_pulse_score")),
            _safe_float(item.get("direction_confidence")),
        ),
        reverse=True,
    )
    return {
        "tab": "fresh",
        "title": "Fresh Movers",
        "selection_mode": selection_mode,
        "longs": _slice([item for item in fresh_candidates if str(item.get("direction")) == "LONG"], limit),
        "shorts": _slice([item for item in fresh_candidates if str(item.get("direction")) == "SHORT"], limit),
        "stocks": _slice(fresh_candidates, limit),
    }


def _build_leaders_tab(items: Sequence[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    long_candidates = [item for item in items if str(item.get("direction")) == "LONG"]
    short_candidates = [item for item in items if str(item.get("direction")) == "SHORT"]
    long_candidates.sort(key=_leader_rank_key, reverse=True)
    short_candidates.sort(key=_leader_rank_key, reverse=True)
    return {
        "tab": "leaders",
        "title": "Session Leaders",
        "longs": _slice(long_candidates, limit),
        "shorts": _slice(short_candidates, limit),
    }


def _build_sector_tab(items: Sequence[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        sector = str(item.get("sector") or "").strip()
        if not sector:
            continue
        grouped.setdefault(sector, []).append(item)

    sector_cards: List[Dict[str, Any]] = []
    for sector_name, sector_items in grouped.items():
        direction_groups = {
            direction_name: sorted(
                [
                    item for item in sector_items
                    if str(item.get("direction") or "") == direction_name
                    and str(item.get("actionability_label") or "") != "risky_spike"
                ],
                key=_sector_stock_rank_key,
                reverse=True,
            )
            for direction_name in ("LONG", "SHORT")
        }
        direction_summaries: List[Dict[str, Any]] = []
        for direction_name, ranked_candidates in direction_groups.items():
            if not ranked_candidates:
                continue
            top_candidates = ranked_candidates[:SECTOR_AGGREGATE_DEPTH]
            direction_summaries.append(
                {
                    "direction": direction_name,
                    "ranked": ranked_candidates,
                    "sector_score": round(
                        sum(_safe_float(candidate.get("sector_opportunity_score")) for candidate in top_candidates) / len(top_candidates),
                        1,
                    ),
                    "market_relative_score": round(
                        sum(_safe_float(candidate.get("sector_aligned_relative_strength")) for candidate in top_candidates) / len(top_candidates),
                        2,
                    ),
                }
            )

        if direction_summaries:
            direction_summaries.sort(
                key=lambda summary: (
                    _safe_float(summary.get("sector_score")),
                    _safe_float(summary.get("market_relative_score")),
                    len(summary.get("ranked") or []),
                ),
                reverse=True,
            )
            winning_summary = direction_summaries[0]
            ranked = list(winning_summary.get("ranked") or [])
            sector_direction = str(winning_summary.get("direction") or "")
            sector_score = round(_safe_float(winning_summary.get("sector_score")), 1)
            market_relative_score = round(_safe_float(winning_summary.get("market_relative_score")), 2)
        else:
            ranked = sorted(sector_items, key=_sector_stock_rank_key, reverse=True)
            if not ranked:
                continue
            first_item = ranked[0]
            sector_direction = str(first_item.get("direction") or "NEUTRAL")
            sector_score = round(_safe_float(first_item.get("sector_opportunity_score")), 1)
            market_relative_score = round(_safe_float(first_item.get("sector_aligned_relative_strength")), 2)

        weakest = sorted(sector_items, key=lambda item: _safe_float(item.get("sector_opportunity_score")))
        leader = ranked[0] if ranked else None
        challenger = ranked[1] if len(ranked) > 1 else None
        laggard = weakest[0] if weakest else None
        if leader is None:
            continue

        avg_change_pct = round(
            sum(_safe_float(entry.get("change_pct")) for entry in sector_items) / len(sector_items),
            2,
        ) if sector_items else 0.0
        sector_cards.append(
            {
                "sector": sector_name,
                "sector_direction": sector_direction,
                "best_stock": leader,
                "leader": leader,
                "challenger": challenger,
                "laggard": laggard,
                "sector_score": sector_score,
                "market_relative_score": market_relative_score,
                "average_change_pct": avg_change_pct,
                "candidate_count": len(ranked),
                "top_stocks": _slice(ranked, 3),
            }
        )

    sector_cards.sort(
        key=lambda item: (
            _safe_float(item.get("sector_score")),
            _safe_float(item.get("market_relative_score")),
            _safe_float(item.get("average_change_pct")),
            int(item.get("candidate_count", 0) or 0),
        ),
        reverse=True,
    )
    return {
        "tab": "sectors",
        "title": "Sector Leaders",
        "sectors": _slice(sector_cards, min(max(limit, 0), SECTOR_TAB_LIMIT) if limit > 0 else SECTOR_TAB_LIMIT),
    }


def _build_hero(
    items: Sequence[Dict[str, Any]],
    fresh_tab: Dict[str, Any],
    sector_tab: Dict[str, Any],
    leaders_tab: Dict[str, Any],
) -> Dict[str, Any]:
    leader_long = _select_stable_session_leader(items, "LONG")
    leader_short = _select_stable_session_leader(items, "SHORT")
    fresh_long = next(iter(fresh_tab.get("longs") or []), None)
    fresh_short = next(iter(fresh_tab.get("shorts") or []), None)
    strongest_sector = next(iter(sector_tab.get("sectors") or []), None)
    best_fresh = next(iter(fresh_tab.get("stocks") or []), None)
    return {
        "market_mode": _market_mode(items),
        "best_long": leader_long,
        "best_short": leader_short,
        "leader_long": leader_long,
        "leader_short": leader_short,
        "fresh_long": fresh_long,
        "fresh_short": fresh_short,
        "best_fresh": best_fresh,
        "strongest_sector": strongest_sector,
        "leaders_overview": {
            "long_count": len(leaders_tab.get("longs") or []),
            "short_count": len(leaders_tab.get("shorts") or []),
        },
    }


def get_pulse_navigator(
    scanner_stocks: List[Dict[str, Any]],
    last_updated: str,
    preset: str = "balanced",
    direction: str = "ALL",
    limit: int = 12,
) -> Dict[str, Any]:
    normalized_preset = _normalize_preset(preset)
    normalized_direction = _normalize_direction(direction)
    pulse_result = get_momentum_pulse(
        scanner_stocks=scanner_stocks,
        last_updated=last_updated,
        direction="ALL",
        include_veryweak=True,
        limit=max(DEFAULT_FETCH_LIMIT, limit * 8, len(scanner_stocks)),
    )
    pulse_status = str(pulse_result.get("status") or "warming_up")
    pulse_stocks = list(pulse_result.get("stocks") or [])

    if pulse_status == "warming_up" and not pulse_stocks:
        return {
            "feature": "Pulse Navigator",
            "feature_key": "pulse_navigator",
            "status": pulse_status,
            "message": str(pulse_result.get("message") or "Momentum data is warming up"),
            "last_updated": str(pulse_result.get("last_updated") or last_updated),
            "market_data_last_updated": str(pulse_result.get("market_data_last_updated") or last_updated),
            "tabs": {
                "discover": {"tab": "discover", "title": "Discover", "buckets": []},
                "fresh": {"tab": "fresh", "title": "Fresh Movers", "stocks": []},
                "leaders": {"tab": "leaders", "title": "Session Leaders", "longs": [], "shorts": []},
                "sectors": {"tab": "sectors", "title": "Sector Leaders", "sectors": []},
            },
            "hero": {"market_mode": "Warming Up"},
            "preset": normalized_preset,
            "direction": normalized_direction,
        }

    raw_items = [_decorate_item(item) for item in pulse_stocks]
    filtered_items = _apply_preset_filters(raw_items, normalized_preset, normalized_direction)
    source_key = _source_key(str(pulse_result.get("last_updated") or last_updated), len(raw_items))

    discover_tab = _build_discover_tab(filtered_items, limit)
    fresh_tab = _build_fresh_tab(filtered_items, source_key, limit)
    leaders_tab = _build_leaders_tab(filtered_items, limit)
    sector_tab = _build_sector_tab(filtered_items, limit)
    hero = _build_hero(filtered_items, fresh_tab, sector_tab, leaders_tab)

    return {
        "feature": "Pulse Navigator",
        "feature_key": "pulse_navigator",
        "status": pulse_status if pulse_status in {"ready", "stale_refreshing"} else "ready",
        "message": str(pulse_result.get("message") or ""),
        "last_updated": str(pulse_result.get("last_updated") or last_updated),
        "market_data_last_updated": str(pulse_result.get("market_data_last_updated") or last_updated),
        "benchmark_change_pct": _safe_float(pulse_result.get("benchmark_change_pct")),
        "preset": normalized_preset,
        "direction": normalized_direction,
        "available_presets": list(_PRESET_CONFIG.keys()),
        "available_tabs": ["discover", "leaders", "fresh", "sectors"],
        "total_candidates": len(filtered_items),
        "hero": hero,
        "tabs": {
            "discover": discover_tab,
            "leaders": leaders_tab,
            "fresh": fresh_tab,
            "sectors": sector_tab,
        },
    }


def get_pulse_navigator_tab(
    scanner_stocks: List[Dict[str, Any]],
    last_updated: str,
    tab: str,
    preset: str = "balanced",
    direction: str = "ALL",
    limit: int = 12,
) -> Dict[str, Any]:
    result = get_pulse_navigator(
        scanner_stocks=scanner_stocks,
        last_updated=last_updated,
        preset=preset,
        direction=direction,
        limit=limit,
    )
    selected_tab = str(tab or "discover").strip().lower()
    if selected_tab not in {"discover", "leaders", "fresh", "sectors"}:
        selected_tab = "discover"

    return {
        "feature": result.get("feature"),
        "feature_key": result.get("feature_key"),
        "status": result.get("status"),
        "last_updated": result.get("last_updated"),
        "market_data_last_updated": result.get("market_data_last_updated"),
        "benchmark_change_pct": result.get("benchmark_change_pct"),
        "preset": result.get("preset"),
        "direction": result.get("direction"),
        "available_presets": result.get("available_presets"),
        "available_tabs": result.get("available_tabs"),
        "hero": result.get("hero"),
        "tab": result.get("tabs", {}).get(selected_tab, {}),
    }
