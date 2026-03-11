"""sector_momentum.py — Momentum Acceleration tracker (9:15–10:00 AM, every 5 min)"""

import os
os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
import logging
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional

import pytz

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
    from stocks import SECTORS
    import os
    os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
    import yfinance as yf
    import pandas as pd

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

    all_symbols = [
        sym if sym.endswith(".NS") else sym + ".NS"
        for syms in SECTORS.values()
        for sym in syms
    ]
    # Use sample of first 5 per sector (mirrors get_historical_momentum)
    sector_samples = {
        name: [(s if s.endswith(".NS") else s + ".NS") for s in syms[:5]]
        for name, syms in SECTORS.items()
    }
    sampled_symbols = list({s for syms in sector_samples.values() for s in syms})
    tickers_str = " ".join(sampled_symbols)

    # ── Daily data for prev_close ─────────────────────────────────────────────
    import time as _time
    import pandas as pd
    # Batch daily download
    daily_batches = []
    for i in range(0, len(sampled_symbols), 30):
        batch = sampled_symbols[i:i+30]
        try:
            df = yf.download(
                tickers=" ".join(batch),
                period="5d",
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
                timeout=30,
            )
            if df is not None and not df.empty:
                daily_batches.append(df)
        except Exception as exc:
            logger.warning(f"Backfill: daily batch failed: {exc}")
        _time.sleep(2)
    if daily_batches:
        daily_raw = pd.concat(daily_batches, axis=1)
    else:
        daily_raw = None

    def get_prev_close(symbol: str) -> float:
        try:
            if isinstance(daily_raw.columns, pd.MultiIndex):
                if symbol not in daily_raw.columns.get_level_values(0):
                    return 0.0
                closes = daily_raw[symbol]["Close"].dropna()
            else:
                closes = daily_raw["Close"].dropna()
            return float(closes.iloc[-2]) if len(closes) >= 2 else 0.0
        except Exception:
            return 0.0

    # ── Intraday data for today ───────────────────────────────────────────────
    # Batch intraday download
    intra_batches = []
    for i in range(0, len(sampled_symbols), 30):
        batch = sampled_symbols[i:i+30]
        try:
            df = yf.download(
                tickers=" ".join(batch),
                period="1d",
                interval="5m",
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
                timeout=30,
            )
            if df is not None and not df.empty:
                intra_batches.append(df)
        except Exception as exc:
            logger.warning(f"Backfill: intraday batch failed: {exc}")
        _time.sleep(2)
    if intra_batches:
        intra_raw = pd.concat(intra_batches, axis=1)
    else:
        intra_raw = None

    if intra_raw is None or intra_raw.empty:
        logger.warning("Backfill: empty intraday data")
        return

    # Normalise index to naive IST
    if intra_raw.index.tz is not None:
        intra_raw.index = intra_raw.index.tz_convert("Asia/Kolkata").tz_localize(None)

    def get_sym_intra(symbol: str):
        try:
            if isinstance(intra_raw.columns, pd.MultiIndex):
                if symbol not in intra_raw.columns.get_level_values(0):
                    return None
                df = intra_raw[symbol]
            else:
                df = intra_raw
            return df if not df.empty else None
        except Exception:
            return None

    # ── Build snapshots per sector ────────────────────────────────────────────
    for sector_name, symbols in sector_samples.items():
        sector_snaps: Dict[str, float] = {}
        for h, m, label in elapsed_slots:
            changes = []
            for sym in symbols:
                df = get_sym_intra(sym)
                if df is None:
                    continue
                prev_close = get_prev_close(sym)
                if prev_close <= 0:
                    continue
                # Find the 5-min candle at exactly this slot time
                mask = (df.index.hour == h) & (df.index.minute == m)
                rows = df[mask]
                if rows.empty:
                    continue
                try:
                    slot_close = float(rows["Close"].iloc[0])
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
    import pandas as pd
    from datetime import timedelta
    from stocks import SECTORS

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

    result_data: Dict[str, Dict[str, float]] = {}

    for sector_name, symbols in SECTORS.items():
        sample = list(symbols)[:5]
        tickers_str = " ".join(s + ".NS" if not s.endswith(".NS") else s for s in sample)

        # ── Fetch previous trading day's close (same baseline the heatmap uses) ──
        prev_closes: Dict[str, float] = {}
        try:
            daily_raw = yf.download(
                tickers=tickers_str,
                start=(dt - timedelta(days=7)).strftime("%Y-%m-%d"),
                end=target_date,   # exclusive → last row = prev trading day
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if daily_raw is not None and not daily_raw.empty:
                if daily_raw.index.tz is not None:
                    daily_raw.index = daily_raw.index.tz_convert("Asia/Kolkata").tz_localize(None)
                if isinstance(daily_raw.columns, pd.MultiIndex):
                    for sym in sample:
                        sym_ns = sym if sym.endswith(".NS") else sym + ".NS"
                        try:
                            avail = daily_raw.columns.get_level_values(1).unique().tolist()
                            if sym_ns in avail:
                                sym_daily = daily_raw.xs(sym_ns, axis=1, level=1)
                                closes = sym_daily["Close"].dropna()
                                if not closes.empty:
                                    prev_closes[sym_ns] = float(closes.iloc[-1])
                        except Exception:
                            pass
                else:
                    sym_ns = sample[0] if sample[0].endswith(".NS") else sample[0] + ".NS"
                    closes = daily_raw["Close"].dropna()
                    if not closes.empty:
                        prev_closes[sym_ns] = float(closes.iloc[-1])
        except Exception as exc:
            logger.warning("Daily prev_close download failed for %s: %s", sector_name, exc)

        logger.info("Sector %s prev_closes: %s", sector_name, prev_closes)

        try:
            raw = yf.download(
                tickers=tickers_str,
                start=target_date,
                end=next_day,
                interval="5m",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.warning("Historical download failed for %s: %s", sector_name, exc)
            continue

        if raw is None or raw.empty:
            logger.warning("Empty download for sector %s", sector_name)
            continue

        # ── CRITICAL FIX: Convert UTC index → naive IST BEFORE anything else ──
        if raw.index.tz is not None:
            raw.index = raw.index.tz_convert("Asia/Kolkata").tz_localize(None)

        # Keep only rows matching target_date in IST
        raw = raw[raw.index.date == target_date_obj]

        if raw.empty:
            logger.warning("No IST rows for %s on %s", sector_name, target_date)
            continue

        logger.info("Sector %s: %d rows after IST filter, first=%s",
                    sector_name, len(raw), raw.index[0])

        # Build {symbol: df} — MultiIndex is (field, symbol)
        sym_dfs: Dict[str, Any] = {}
        if isinstance(raw.columns, pd.MultiIndex):
            available_syms = raw.columns.get_level_values(1).unique().tolist()
            for sym in sample:
                sym_ns = sym if sym.endswith(".NS") else sym + ".NS"
                if sym_ns in available_syms:
                    try:
                        df = raw.xs(sym_ns, axis=1, level=1)
                        if not df.empty:
                            sym_dfs[sym_ns] = df
                    except Exception as e:
                        logger.warning("xs failed for %s: %s", sym_ns, e)
        else:
            # Single ticker
            sym_key = sample[0]
            sym_key = sym_key if sym_key.endswith(".NS") else sym_key + ".NS"
            sym_dfs[sym_key] = raw

        logger.info("Sector %s: %d symbols available", sector_name, len(sym_dfs))

        if not sym_dfs:
            continue

        sector_snapshots: Dict[str, float] = {}
        for slot, (ist_h, ist_m) in SLOT_IST.items():
            changes: List[float] = []
            for sym, df in sym_dfs.items():
                try:
                    slot_mask = (df.index.hour == ist_h) & (df.index.minute == ist_m)
                    slot_rows = df[slot_mask]

                    if slot_rows.empty:
                        continue

                    prev_close = prev_closes.get(sym, 0.0)
                    slot_close = float(slot_rows["Close"].iloc[0])

                    if slot == "9:15":
                        logger.info("[DEBUG] %s prev_close=%.2f slot_close=%.2f",
                                    sym, prev_close, slot_close)

                    if prev_close > 0:
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
        from stocks import SECTORS as _SECTORS

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
