import threading
from datetime import datetime
from typing import Any, Dict, List, Sequence

import pytz

from backend.momentum_pulse import get_momentum_pulse
from stocks import SECTORS

IST = pytz.timezone("Asia/Kolkata")
DEFAULT_FETCH_LIMIT = 120
TOP_ZONE_RANK = 15

_navigator_lock = threading.Lock()
_navigator_state: Dict[str, Any] = {
    "source_key": "",
    "current_ranks": {},
    "fresh_symbols": [],
}

_ACTIONABILITY_PRIORITY = {
    "clean_setup": 3,
    "needs_pullback": 2,
    "extended": 1,
    "risky_spike": 0,
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
        return list(_navigator_state.get("fresh_symbols") or [])


def _build_fresh_tab(items: Sequence[Dict[str, Any]], source_key: str, limit: int) -> Dict[str, Any]:
    fresh_symbols = set(_detect_fresh_entries(items, source_key))
    fresh_candidates = [
        item for item in items
        if str(item.get("symbol") or "") in fresh_symbols
        and (_safe_float(item.get("score_change_10m")) > 0 or bool(item.get("improving_now")))
    ]
    if not fresh_candidates:
        fresh_candidates = [
            item for item in items
            if (_safe_float(item.get("score_change_10m")) >= 3.0 or bool(item.get("improving_now")))
        ]
    fresh_candidates.sort(
        key=lambda item: (
            _safe_float(item.get("score_change_10m")),
            _safe_float(item.get("pulse_trend_strength")),
            _safe_float(item.get("momentum_pulse_score")),
        ),
        reverse=True,
    )
    return {
        "tab": "fresh",
        "title": "Fresh Movers",
        "stocks": _slice(fresh_candidates, limit),
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
        ranked = sorted(sector_items, key=_rank_key, reverse=True)
        weakest = sorted(sector_items, key=lambda item: _safe_float(item.get("momentum_pulse_score")))
        leader = ranked[0] if ranked else None
        challenger = ranked[1] if len(ranked) > 1 else None
        laggard = weakest[0] if weakest else None
        if leader is None:
            continue
        sector_cards.append(
            {
                "sector": sector_name,
                "leader": leader,
                "challenger": challenger,
                "laggard": laggard,
                "sector_score": round(_safe_float(leader.get("momentum_pulse_score")), 1),
            }
        )

    sector_cards.sort(key=lambda item: _safe_float(item.get("sector_score")), reverse=True)
    return {
        "tab": "sectors",
        "title": "Sector Leaders",
        "sectors": _slice(sector_cards, limit),
    }


def _build_hero(items: Sequence[Dict[str, Any]], fresh_tab: Dict[str, Any], sector_tab: Dict[str, Any]) -> Dict[str, Any]:
    best_long = next((item for item in items if str(item.get("direction")) == "LONG"), None)
    best_short = next((item for item in items if str(item.get("direction")) == "SHORT"), None)
    strongest_sector = next(iter(sector_tab.get("sectors") or []), None)
    best_fresh = next(iter(fresh_tab.get("stocks") or []), None)
    return {
        "market_mode": _market_mode(items),
        "best_long": best_long,
        "best_short": best_short,
        "best_fresh": best_fresh,
        "strongest_sector": strongest_sector,
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

    if str(pulse_result.get("status") or "") != "ready":
        return {
            "feature": "Pulse Navigator",
            "feature_key": "pulse_navigator",
            "status": str(pulse_result.get("status") or "warming_up"),
            "message": str(pulse_result.get("message") or "Momentum data is warming up"),
            "last_updated": str(pulse_result.get("last_updated") or last_updated),
            "market_data_last_updated": str(pulse_result.get("market_data_last_updated") or last_updated),
            "tabs": {
                "discover": {"tab": "discover", "title": "Discover", "buckets": []},
                "fresh": {"tab": "fresh", "title": "Fresh Movers", "stocks": []},
                "sectors": {"tab": "sectors", "title": "Sector Leaders", "sectors": []},
            },
            "hero": {"market_mode": "Warming Up"},
            "preset": normalized_preset,
            "direction": normalized_direction,
        }

    raw_items = [_decorate_item(item) for item in pulse_result.get("stocks") or []]
    filtered_items = _apply_preset_filters(raw_items, normalized_preset, normalized_direction)
    source_key = _source_key(str(pulse_result.get("last_updated") or last_updated), len(raw_items))

    discover_tab = _build_discover_tab(filtered_items, limit)
    fresh_tab = _build_fresh_tab(filtered_items, source_key, limit)
    sector_tab = _build_sector_tab(filtered_items, limit)
    hero = _build_hero(filtered_items, fresh_tab, sector_tab)

    return {
        "feature": "Pulse Navigator",
        "feature_key": "pulse_navigator",
        "status": "ready",
        "last_updated": str(pulse_result.get("last_updated") or last_updated),
        "market_data_last_updated": str(pulse_result.get("market_data_last_updated") or last_updated),
        "benchmark_change_pct": _safe_float(pulse_result.get("benchmark_change_pct")),
        "preset": normalized_preset,
        "direction": normalized_direction,
        "available_presets": list(_PRESET_CONFIG.keys()),
        "available_tabs": ["discover", "fresh", "sectors"],
        "total_candidates": len(filtered_items),
        "hero": hero,
        "tabs": {
            "discover": discover_tab,
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
    if selected_tab not in {"discover", "fresh", "sectors"}:
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
