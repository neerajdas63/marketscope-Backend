"""sector_momentum.py — Momentum Acceleration tracker (9:15–10:00 AM, every 5 min)"""

import os
os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
import logging
from datetime import datetime, timedelta
from statistics import mean
from typing import Any, Dict, List, Optional

import pytz

from upstox_client import get_daily_history_batch, get_intraday_history_batch

logger = logging.getLogger("sector_momentum")

IST = pytz.timezone("Asia/Kolkata")

TIME_SLOTS: List[str] = [
    "9:15", "9:20", "9:25", "9:30",
    "9:35", "9:40", "9:45", "9:50",
    "9:55", "10:00",
]

_momentum_data: Dict[str, Dict[str, float]] = {}
_last_snapshot_time: Optional[str] = None
_cache_ref: Optional[Any] = None
# Stores sector_name → change_pct at the 10:00 AM final slot.
# Populated once at 10:00 snapshot (or by backfill). Persists for the rest of the day.
_final_snapshot: Dict[str, float] = {}


def _extract_symbol_frame(raw: Any, symbol: str):
    if raw is None:
        return None
    try:
        import pandas as pd

        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0)
            level1 = raw.columns.get_level_values(1)
            if symbol in level0:
                df = raw[symbol]
            elif symbol in level1:
                df = raw.xs(symbol, axis=1, level=1)
            else:
                return None
        else:
            df = raw
        return df if df is not None and not df.empty else None
    except Exception:
        return None


def _normalize_to_naive_ist_index(df):
    """Convert a dataframe index to naive IST for reliable slot matching."""
    if df is None or df.empty:
        return df
    try:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata").tz_localize(None)
        else:
            df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    except Exception:
        pass
    return df


def _get_slot_close(df, target_dt: datetime, max_gap_minutes: int = 7) -> Optional[float]:
    """Return the closest close at or before the target slot within an acceptable gap."""
    if df is None or df.empty:
        return None
    try:
        eligible = df[df.index <= target_dt]
        if eligible.empty:
            return None
        last_row = eligible.iloc[-1]
        last_ts = eligible.index[-1]
        if (target_dt - last_ts) > timedelta(minutes=max_gap_minutes):
            return None
        return float(last_row["Close"])
    except Exception:
        return None


def set_cache_ref(cache: Any) -> None:
    global _cache_ref
    _cache_ref = cache


def _get_current_slot(now: datetime) -> Optional[str]:
    """
    Return the label of the most recently passed (or exact) slot for *now*.

    This means any time between 9:15 and 10:04:59 returns a valid slot, so:
    - Cron-fired calls at exact boundaries work (exact match).
    - Catch-up calls at mid-interval times (e.g. 9:54) still map to the
      last completed slot (e.g. "9:50") instead of returning None.
    """
    slots = [
        (9, 15, "9:15"),  (9, 20, "9:20"),  (9, 25, "9:25"),
        (9, 30, "9:30"),  (9, 35, "9:35"),  (9, 40, "9:40"),
        (9, 45, "9:45"),  (9, 50, "9:50"),  (9, 55, "9:55"),
        (10,  0, "10:00"),
    ]
    best_label: Optional[str] = None
    best_diff: float = float("inf")
    for h, m, label in slots:
        slot_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
        diff = (now - slot_dt).total_seconds()
        # Only consider slots that have already passed (diff >= 0) and are the closest
        if 0 <= diff < best_diff:
            best_diff = diff
            best_label = label
    # Only valid if we're within the opening window (up to 5 min after last slot)
    return best_label if best_diff <= 300 else None


def _calculate_result_from_snapshots(data: Dict[str, float]) -> Dict[str, Any]:
    eod_slots = [s for s in ["EOD"] if s in data]
    recorded = [s for s in TIME_SLOTS if s in data] or eod_slots

    if not recorded:
        return {"label": "NO DATA", "delta": 0.0, "current": 0.0, "color": "gray"}

    if len(recorded) == 1:
        val = data[recorded[0]]
        return {
            "label": "BASE",
            "delta": 0.0,
            "current": round(val, 2),
            "prev_avg": round(val, 2),
            "color": "#00C853" if val > 0 else "#FF1744",
        }

    current_val = data[recorded[-1]]
    window_size = min(3, len(recorded) - 1)
    prev_slots = recorded[-(window_size + 1):-1]
    prev_avg = sum(data[s] for s in prev_slots) / len(prev_slots)
    delta = round(current_val - prev_avg, 2)

    if delta > 0.30:
        label, color = "STRONG UP 🚀", "#00C853"
    elif delta > 0.15:
        label, color = "ACCELERATING ↑", "#69F0AE"
    elif delta > 0.05:
        label, color = "GAINING →↑", "#B9F6CA"
    elif delta >= -0.05:
        label, color = "STABLE →", "#9E9E9E"
    elif delta >= -0.15:
        label, color = "SLOWING →↓", "#FF6D00"
    elif delta >= -0.30:
        label, color = "FADING ↓", "#FF1744"
    else:
        label, color = "BREAKING DOWN 🔴", "#B71C1C"

    return {
        "label": label,
        "delta": delta,
        "current": round(current_val, 2),
        "prev_avg": round(prev_avg, 2),
        "color": color,
    }


def _calculate_result(sector_name: str) -> Dict[str, Any]:
    return _calculate_result_from_snapshots(_momentum_data.get(sector_name, {}))


def _is_opening_window() -> bool:
    now = datetime.now(IST)
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=10, minute=0, second=0, microsecond=0)
    return start <= now <= end


def backfill_today_snapshots() -> None:
    """
    Called once at startup when the server starts mid-window (after 9:15).
    Downloads today's 5-min intraday data via yfinance and fills all elapsed
    time slots so the Opening tracker shows the full picture, not just "--".
    """
    from stocks import ACTIVE_SECTORS
    import os
    os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
    import yfinance as yf

    now = datetime.now(IST)
    today = now.date()

    # Collect only slots that have already passed
    SLOT_TIMES = [
        (9, 15, "9:15"),  (9, 20, "9:20"),  (9, 25, "9:25"),
        (9, 30, "9:30"),  (9, 35, "9:35"),  (9, 40, "9:40"),
        (9, 45, "9:45"),  (9, 50, "9:50"),  (9, 55, "9:55"),
        (10,  0, "10:00"),
    ]
    elapsed_slots = [
        (h, m, label) for h, m, label in SLOT_TIMES
        if now >= now.replace(hour=h, minute=m, second=0, microsecond=0)
    ]
    if not elapsed_slots:
        return

    logger.info("Backfilling %d elapsed opening slots for today (%s)...", len(elapsed_slots), today)

    sector_samples = {
        name: [(s if s.endswith(".NS") else s + ".NS") for s in syms[:5]]
        for name, syms in ACTIVE_SECTORS.items()
    }
    sampled_symbols = list({s for syms in sector_samples.values() for s in syms})
    from_date = (today - timedelta(days=14)).isoformat()
    to_date = today.isoformat()

    daily_raw = None
    try:
        daily_raw = get_daily_history_batch(sampled_symbols, from_date=from_date, to_date=to_date)
    except Exception as exc:
        logger.warning("Backfill: Upstox daily history failed: %s", exc)

    daily_fallback = None
    missing_daily = [symbol for symbol in sampled_symbols if _extract_symbol_frame(daily_raw, symbol) is None]
    if missing_daily:
        try:
            daily_fallback = yf.download(
                tickers=" ".join(missing_daily),
                period="5d",
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
                timeout=30,
            )
        except Exception as exc:
            logger.warning("Backfill: daily yfinance fallback failed: %s", exc)

    def get_prev_close(symbol: str) -> float:
        for raw in (daily_raw, daily_fallback):
            df = _extract_symbol_frame(raw, symbol)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            try:
                clean_df = _normalize_to_naive_ist_index(df.copy())
                closes = clean_df[clean_df.index.date < today]["Close"].dropna()
                if not closes.empty:
                    return float(closes.iloc[-1])
            except Exception:
                continue
        return 0.0

    intra_raw = None
    try:
        intra_raw = get_intraday_history_batch(sampled_symbols, from_date=to_date, to_date=to_date, interval_minutes=5)
    except Exception as exc:
        logger.warning("Backfill: Upstox intraday history failed: %s", exc)

    intra_fallback = None
    missing_intra = [symbol for symbol in sampled_symbols if _extract_symbol_frame(intra_raw, symbol) is None]
    if missing_intra:
        try:
            intra_fallback = yf.download(
                tickers=" ".join(missing_intra),
                period="1d",
                interval="5m",
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
                timeout=30,
            )
        except Exception as exc:
            logger.warning("Backfill: intraday yfinance fallback failed: %s", exc)

    if (intra_raw is None or intra_raw.empty) and (intra_fallback is None or intra_fallback.empty):
        logger.warning("Backfill: empty intraday data")
        return

    def get_sym_intra(symbol: str):
        for raw in (intra_raw, intra_fallback):
            df = _extract_symbol_frame(raw, symbol)
            if df is None or df.empty:
                continue
            df = _normalize_to_naive_ist_index(df.copy())
            df = df[df.index.date == today]
            if not df.empty:
                return df
        return None

    # ── Build snapshots per sector ────────────────────────────────────────────
    for sector_name, symbols in sector_samples.items():
        sector_snaps: Dict[str, float] = {}
        for h, m, label in elapsed_slots:
            changes = []
            slot_dt = datetime.combine(today, datetime.min.time()).replace(hour=h, minute=m)
            for sym in symbols:
                df = get_sym_intra(sym)
                if df is None:
                    continue
                prev_close = get_prev_close(sym)
                if prev_close <= 0:
                    continue
                slot_close = _get_slot_close(df, slot_dt)
                if slot_close is None:
                    continue
                try:
                    changes.append(round((slot_close - prev_close) / prev_close * 100, 2))
                except Exception:
                    continue
            if changes:
                sector_snaps[label] = round(sum(changes) / len(changes), 2)

        if sector_snaps:
            # Merge into _momentum_data without overwriting already-live slots
            existing = _momentum_data.setdefault(sector_name, {})
            for slot, val in sector_snaps.items():
                existing.setdefault(slot, val)  # don't overwrite live data

    filled = sum(len(v) for v in _momentum_data.values())
    logger.info("Backfill complete — %d slot entries across %d sectors.", filled, len(_momentum_data))

    # If the opening window has already fully closed (all 10 slots filled),
    # derive the final snapshot from the 10:00 slot values so Strong/Weak panels populate.
    global _final_snapshot
    sectors_with_final = [
        name for name, snaps in _momentum_data.items() if "10:00" in snaps
    ]
    if sectors_with_final:
        _final_snapshot = {
            name: _momentum_data[name]["10:00"]
            for name in sectors_with_final
        }
        logger.info("_final_snapshot set from backfill — %d sectors.", len(_final_snapshot))


def get_historical_momentum(target_date: str) -> Dict[str, Any]:
    import yfinance as yf
    from datetime import timedelta
    from stocks import ACTIVE_SECTORS

    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date format '{target_date}': use YYYY-MM-DD") from exc

    next_day = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    target_date_obj = dt.date()

    logger.info("Historical momentum fetch for %s", target_date)

    SLOT_IST: Dict[str, tuple] = {
        "9:15":  (9, 15), "9:20":  (9, 20), "9:25":  (9, 25),
        "9:30":  (9, 30), "9:35":  (9, 35), "9:40":  (9, 40),
        "9:45":  (9, 45), "9:50":  (9, 50), "9:55":  (9, 55),
        "10:00": (10,  0),
    }

    sector_samples = {
        sector_name: [sym if sym.endswith(".NS") else sym + ".NS" for sym in list(symbols)[:5]]
        for sector_name, symbols in ACTIVE_SECTORS.items()
    }
    sampled_symbols = list({symbol for symbols in sector_samples.values() for symbol in symbols})
    daily_from_date = (dt - timedelta(days=10)).strftime("%Y-%m-%d")

    daily_raw = None
    try:
        daily_raw = get_daily_history_batch(sampled_symbols, from_date=daily_from_date, to_date=target_date)
    except Exception as exc:
        logger.warning("Historical momentum Upstox daily fetch failed: %s", exc)

    daily_fallback = None
    missing_daily = [symbol for symbol in sampled_symbols if _extract_symbol_frame(daily_raw, symbol) is None]
    if missing_daily:
        try:
            daily_fallback = yf.download(
                tickers=" ".join(missing_daily),
                start=daily_from_date,
                end=target_date,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.warning("Historical momentum daily yfinance fallback failed: %s", exc)

    intra_raw = None
    try:
        intra_raw = get_intraday_history_batch(sampled_symbols, from_date=target_date, to_date=target_date, interval_minutes=5)
    except Exception as exc:
        logger.warning("Historical momentum Upstox intraday fetch failed: %s", exc)

    intra_fallback = None
    missing_intra = [symbol for symbol in sampled_symbols if _extract_symbol_frame(intra_raw, symbol) is None]
    if missing_intra:
        try:
            intra_fallback = yf.download(
                tickers=" ".join(missing_intra),
                start=target_date,
                end=next_day,
                interval="5m",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.warning("Historical momentum intraday yfinance fallback failed: %s", exc)

    result_data: Dict[str, Dict[str, float]] = {}

    for sector_name, sample_symbols in sector_samples.items():
        prev_closes: Dict[str, float] = {}
        sym_dfs: Dict[str, Any] = {}

        for sym_ns in sample_symbols:
            for raw in (daily_raw, daily_fallback):
                daily_df = _extract_symbol_frame(raw, sym_ns)
                if daily_df is None or daily_df.empty or "Close" not in daily_df.columns:
                    continue
                try:
                    clean_daily_df = _normalize_to_naive_ist_index(daily_df.copy())
                    closes = clean_daily_df[clean_daily_df.index.date < target_date_obj]["Close"].dropna()
                    if not closes.empty:
                        prev_closes[sym_ns] = float(closes.iloc[-1])
                        break
                except Exception:
                    continue

            for raw in (intra_raw, intra_fallback):
                intra_df = _extract_symbol_frame(raw, sym_ns)
                if intra_df is None or intra_df.empty:
                    continue
                clean_intra_df = _normalize_to_naive_ist_index(intra_df.copy())
                clean_intra_df = clean_intra_df[clean_intra_df.index.date == target_date_obj]
                if not clean_intra_df.empty:
                    sym_dfs[sym_ns] = clean_intra_df
                    break

        logger.info("Sector %s prev_closes: %s", sector_name, prev_closes)
        logger.info("Sector %s: %d symbols available", sector_name, len(sym_dfs))

        if not sym_dfs:
            continue

        sector_snapshots: Dict[str, float] = {}
        for slot, (ist_h, ist_m) in SLOT_IST.items():
            changes: List[float] = []
            for sym, df in sym_dfs.items():
                try:
                    prev_close = prev_closes.get(sym, 0.0)
                    slot_dt = datetime.combine(target_date_obj, datetime.min.time()).replace(hour=ist_h, minute=ist_m)
                    slot_close = _get_slot_close(df, slot_dt)
                    if slot_close is None or prev_close <= 0:
                        continue
                    changes.append(round((slot_close - prev_close) / prev_close * 100, 2))
                except Exception:
                    continue

            if changes:
                sector_snapshots[slot] = round(sum(changes) / len(changes), 2)

        logger.info("Sector %s snapshots: %s", sector_name, sector_snapshots)

        if sector_snapshots:
            result_data[sector_name] = sector_snapshots

    sectors_result: Dict[str, Any] = {}
    for sector_name, snapshots in result_data.items():
        sectors_result[sector_name] = {
            "snapshots": snapshots,
            "result": _calculate_result_from_snapshots(snapshots),
        }

    recorded_slots = [
        s for s in TIME_SLOTS
        if any(s in v["snapshots"] for v in sectors_result.values())
    ]

    sorted_sectors = sorted(
        sectors_result.items(),
        key=lambda x: x[1]["result"].get("current", 0.0),
        reverse=True,
    )

    top_long  = [s for s, v in sorted_sectors if v["result"].get("current", 0.0) > 0][:3]
    top_short = [s for s, v in sorted_sectors if v["result"].get("current", 0.0) < 0][-3:][::-1]

    return {
        "sectors": dict(sorted_sectors),
        "slots": recorded_slots,
        "date": target_date,
        "top_long": top_long,
        "top_short": top_short,
        "is_live": False,
        "is_historical": True,
        "last_updated": datetime.now(IST).strftime("%H:%M:%S"),
    }


def calculate_trend(snapshots: Dict[str, float]) -> str:
    values = [snapshots[s] for s in TIME_SLOTS if s in snapshots]
    if len(values) < 2:
        return "FLAT"
    last3 = values[-3:] if len(values) >= 3 else values
    if all(last3[i] < last3[i + 1] for i in range(len(last3) - 1)):
        return "UP"
    if all(last3[i] > last3[i + 1] for i in range(len(last3) - 1)):
        return "DOWN"
    return "FLAT"


def take_snapshot(sectors_data: List[Dict[str, Any]]) -> None:
    global _last_snapshot_time

    now = datetime.now(IST)
    slot = _get_current_slot(now)
    if slot is None:
        logger.debug("Momentum snapshot skipped — outside 9:15–10:00 window")
        return

    if not sectors_data:
        logger.warning("Momentum snapshot: sectors_data is empty, skipping")
        return

    for sector in sectors_data:
        name = sector.get("name", "UNKNOWN")
        stocks = sector.get("stocks", [])
        if not stocks:
            continue

        try:
            avg_change = round(float(sector.get("change_pct") or 0.0), 2)
        except Exception as exc:
            logger.warning("Momentum: failed official change for %s: %s", name, exc)
            avg_change = 0.0

        _momentum_data.setdefault(name, {})[slot] = avg_change

    _last_snapshot_time = slot
    logger.info("Momentum snapshot taken at %s", slot)

    # At the final 10:00 AM slot, freeze the strong/weak snapshot for the rest of the day
    if slot == "10:00":
        global _final_snapshot
        _final_snapshot = {
            name: _momentum_data[name].get("10:00", 0.0)
            for name in _momentum_data
            if "10:00" in _momentum_data[name]
        }
        logger.info("Final 10:00 AM snapshot saved for %d sectors.", len(_final_snapshot))


def _build_eod_snapshot() -> None:
    global _momentum_data
    try:
        if _cache_ref is None:
            return
        cached = _cache_ref.get()
        if not cached:
            return
        sectors_data = cached.get("sectors", [])
        for sector in sectors_data:
            name = sector.get("name", "UNKNOWN")
            stocks = sector.get("stocks", [])
            if not stocks or not name:
                continue
            try:
                avg_change = round(float(sector.get("change_pct") or 0.0), 2)
            except Exception:
                avg_change = 0.0
            _momentum_data[name] = {"EOD": avg_change}
        logger.info("EOD momentum snapshot built from cache (%d sectors)", len(_momentum_data))
    except Exception as exc:
        logger.error("_build_eod_snapshot failed: %s", exc)


def get_momentum_data() -> Dict[str, Any]:
    if not _momentum_data:
        _build_eod_snapshot()

    recorded_slots = [
        s for s in TIME_SLOTS
        if any(s in v for v in _momentum_data.values())
    ]
    if not recorded_slots:
        recorded_slots = ["EOD"] if any(
            "EOD" in v for v in _momentum_data.values()
        ) else []

    sectors_result: Dict[str, Any] = {}
    for sector_name, snapshots in _momentum_data.items():
        result = _calculate_result(sector_name)
        sectors_result[sector_name] = {
            "snapshots": snapshots,
            "result": result,
        }

    sorted_sectors = sorted(
        sectors_result.items(),
        key=lambda x: x[1]["result"].get("current", 0.0),
        reverse=True,
    )

    top_long  = [s for s, v in sorted_sectors if v["result"].get("current", 0.0) > 0][:3]
    top_short = [s for s, v in sorted_sectors if v["result"].get("current", 0.0) < 0][-3:][::-1]

    # After 10 AM: use _final_snapshot to build persistent strong/weak lists
    now_ist = datetime.now(IST)
    past_window = now_ist >= now_ist.replace(hour=10, minute=0, second=0, microsecond=0)

    if past_window and _final_snapshot:
        # strong = positive value at 10:00 AND positive delta from result
        # weak   = negative value at 10:00 AND negative delta from result
        final_strong = [
            s for s in _final_snapshot
            if _final_snapshot[s] > 0
            and sectors_result.get(s, {}).get("result", {}).get("delta", 0) >= 0
        ]
        final_weak = [
            s for s in _final_snapshot
            if _final_snapshot[s] < 0
            and sectors_result.get(s, {}).get("result", {}).get("delta", 0) <= 0
        ]
        # Sort by magnitude
        final_strong.sort(key=lambda s: _final_snapshot.get(s, 0), reverse=True)
        final_weak.sort(key=lambda s: _final_snapshot.get(s, 0))
    else:
        final_strong = top_long
        final_weak   = top_short

    return {
        "sectors":            dict(sorted_sectors),
        "slots":              recorded_slots,
        "top_long":           top_long,
        "top_short":          top_short,
        "strong_sectors":     final_strong,
        "weak_sectors":       final_weak,
        "has_final_snapshot": bool(_final_snapshot),
        "is_live":            _is_opening_window(),
        "last_updated":       datetime.now(IST).strftime("%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Feature 3: Relative Sector Strength vs Nifty
# ---------------------------------------------------------------------------

def get_relative_sector_strength(sectors_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute each sector's performance relative to Nifty50.

    Returns:
        {
            sector_name: {
                "absolute":  float,   # sector avg change_pct
                "relative":  float,   # sector - nifty
                "vs_nifty":  str      # OUTPERFORMING | UNDERPERFORMING | IN LINE
            },
            ...
        }
    """
    try:
        from stocks import ACTIVE_SECTORS as _SECTORS

        # Build sym → change_pct map from sectors_data
        sym_change: Dict[str, float] = {}
        sector_avgs: Dict[str, float] = {}

        for sector in sectors_data:
            name   = sector.get("name", "")
            stocks = sector.get("stocks", [])
            if not name or not stocks:
                continue
            changes = [float(s.get("change_pct", 0) or 0) for s in stocks]
            if changes:
                sector_avgs[name] = round(sum(changes) / len(changes), 2)
            for s in stocks:
                sym = s.get("symbol")
                if sym:
                    sym_change[sym] = float(s.get("change_pct", 0) or 0)

        # Nifty50 average: use SECTORS["NIFTY 50"] symbols
        nifty_syms = [
            s.replace(".NS", "") for s in _SECTORS.get("NIFTY 50", [])
        ]
        nifty_changes = [sym_change[s] for s in nifty_syms if s in sym_change]
        nifty_avg = round(sum(nifty_changes) / len(nifty_changes), 2) if nifty_changes else 0.0

        result: Dict[str, Any] = {}
        for name, avg in sector_avgs.items():
            rel = round(avg - nifty_avg, 2)
            if rel > 0.5:
                vs = "OUTPERFORMING"
            elif rel < -0.5:
                vs = "UNDERPERFORMING"
            else:
                vs = "IN LINE"
            result[name] = {
                "absolute":  avg,
                "relative":  rel,
                "vs_nifty":  vs,
                "nifty_avg": nifty_avg,
            }

        return result

    except Exception as e:
        logger.error("get_relative_sector_strength failed: %s", e, exc_info=True)
        return {}
