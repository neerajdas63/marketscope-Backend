# breakout_scanner.py — Breakout scoring from already-cached sym_data (no extra API calls)

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

from upstox_client import get_bulk_full_quotes as get_upstox_bulk_full_quotes, get_daily_history_batch

logger = logging.getLogger("breakout_scanner")

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------

_cache: Dict[str, Any] = {
    "breakouts": [],
    "last_updated": None,
    "is_loading": False,
    "last_attempt": 0,      # unix timestamp of last refresh attempt
}

_COOLDOWN_SECONDS = 300     # minimum gap between refreshes (5 minutes)


# ---------------------------------------------------------------------------
# Per-symbol scorers
# ---------------------------------------------------------------------------

def _score_breakout(clean_sym: str, stock: Dict[str, Any]) -> Dict | None:
    """
    LONG breakout scoring (max 13 pts).
    Returns result dict with direction="LONG", or None if score < 3.
    """
    ltp = float(stock.get("ltp") or 0.0)
    if ltp <= 0:
        return None

    vol_ratio  = float(stock.get("volume_ratio") or 1.0)
    rsi        = float(stock.get("rsi") or 50.0)
    rs_val     = float(stock.get("relative_strength") or 0.0)
    change_pct = float(stock.get("change_pct") or 0.0)
    rfactor    = float(stock.get("rfactor") or 0.0)

    score = 0
    signals: List[str] = []

    # ── Volume ──────────────────────────────────────────────────────
    if vol_ratio >= 2.0:
        score += 3; signals.append("VOL_SURGE_2X")
    elif vol_ratio >= 1.5:
        score += 2; signals.append("VOL_SURGE_1.5X")
    elif vol_ratio >= 1.2:
        score += 1; signals.append("VOL_ABOVE_AVG")

    # ── RSI ──────────────────────────────────────────────────────────
    if 50.0 <= rsi <= 80.0:
        score += 2; signals.append("IDEAL_RSI")
    elif 80.0 < rsi <= 90.0:
        score += 1; signals.append("HIGH_RSI")
    elif rsi > 90.0:
        score -= 1; signals.append("OVERBOUGHT")

    # ── Relative Strength ────────────────────────────────────────────
    if rs_val >= 3.0:
        score += 3; signals.append("RS_VERY_STRONG")
    elif rs_val >= 1.5:
        score += 2; signals.append("RS_STRONG")
    elif rs_val >= 0.5:
        score += 1; signals.append("RS_POSITIVE")

    # ── Today's candle ───────────────────────────────────────────────
    if change_pct >= 3.0:
        score += 2; signals.append("STRONG_CANDLE")
    elif change_pct >= 1.5:
        score += 1; signals.append("GOOD_CANDLE")
    # change_pct >= 0.0 → neutral (no points, no penalty)

    # ── R-Factor bonus ───────────────────────────────────────────────
    if rfactor >= 2.5:
        score += 2; signals.append("HIGH_RFACTOR")
    elif rfactor >= 2.0:
        score += 1; signals.append("GOOD_RFACTOR")

    if score < 3:
        return None

    if score >= 9:
        btype = "EXPLOSIVE"
    elif score >= 6:
        btype = "STRONG"
    else:
        btype = "MODERATE"

    return {
        **stock,
        "symbol": clean_sym,
        "breakout_score": score,
        "breakout_type": btype,
        "direction": "LONG",
        "signals": signals,
    }


def _score_breakdown(clean_sym: str, stock: Dict[str, Any]) -> Dict | None:
    """
    SHORT breakdown scoring (max 14 pts).
    Returns result dict with direction="SHORT", or None if score < 3.
    """
    ltp = float(stock.get("ltp") or 0.0)
    if ltp <= 0:
        return None

    vol_ratio  = float(stock.get("volume_ratio") or 1.0)
    rsi        = float(stock.get("rsi") or 50.0)
    rs_val     = float(stock.get("relative_strength") or 0.0)
    change_pct = float(stock.get("change_pct") or 0.0)
    rfactor    = float(stock.get("rfactor") or 0.0)

    score = 0
    signals: List[str] = []

    # ── SIGNAL 1: Heavy selling volume ───────────────────────────────
    if vol_ratio >= 1.5:
        score += 3; signals.append("SELL_VOL_1.5X")
    elif vol_ratio >= 1.2:
        score += 2; signals.append("SELL_VOL_AVG")
    elif vol_ratio >= 1.0:
        score += 1; signals.append("SELL_VOL_LOW")
    # vol_ratio < 1.0 → no points (panic selling without volume = weak signal)

    # ── SIGNAL 2: RSI weak zone ───────────────────────────────────────
    if rsi <= 30.0:
        score += 3; signals.append("RSI_OVERSOLD")
    elif rsi <= 40.0:
        score += 2; signals.append("RSI_WEAK")
    elif rsi <= 50.0:
        score += 1; signals.append("RSI_BELOW_MID")

    # ── SIGNAL 3: Negative relative strength ─────────────────────────
    if rs_val <= -3.0:
        score += 3; signals.append("RS_VERY_WEAK")
    elif rs_val <= -1.5:
        score += 2; signals.append("RS_WEAK")
    elif rs_val <= -0.5:
        score += 1; signals.append("RS_NEGATIVE")

    # ── SIGNAL 4: Strong red candle ───────────────────────────────────
    if change_pct <= -3.0:
        score += 3; signals.append("STRONG_RED_CANDLE")
    elif change_pct <= -1.5:
        score += 2; signals.append("RED_CANDLE")
    elif change_pct <= -0.5:
        score += 1; signals.append("SLIGHT_RED")

    # ── SIGNAL 5: Low rfactor (no institutional support) ─────────────
    if rfactor < 1.5:
        score += 2; signals.append("LOW_RFACTOR")
    elif rfactor < 2.0:
        score += 1; signals.append("WEAK_RFACTOR")

    if score < 3:
        return None

    if score >= 9:
        btype = "STRONG_SHORT"
    elif score >= 6:
        btype = "MODERATE_SHORT"
    else:
        btype = "WATCH_SHORT"

    return {
        **stock,
        "symbol": clean_sym,
        "breakout_score": score,
        "breakout_type": btype,
        "direction": "SHORT",
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# Core aggregator
# ---------------------------------------------------------------------------

def _compute_breakouts_from_boost(sym_data: Dict[str, Any], limit: int = 30) -> List[Dict]:
    """
    Score every symbol for LONG breakout first; if it doesn't qualify as LONG,
    try SHORT breakdown. Returns top 15 longs + top 15 shorts (max 30).
    """
    if not sym_data:
        logger.warning("Breakout: sym_data is empty — no stocks to score")
        return []

    longs:  List[Dict] = []
    shorts: List[Dict] = []

    for clean_sym, stock in sym_data.items():
        change  = float(stock.get("change_pct") or 0.0)
        vol     = float(stock.get("volume_ratio") or 0.0)
        rs      = float(stock.get("relative_strength") or 0.0)

        # Try LONG — gate: must be green, above-avg volume, not weaker than market
        if change >= 0.0 and vol >= 1.2 and rs >= 0.0:
            long_result = _score_breakout(clean_sym, stock)
            if long_result:
                longs.append(long_result)
                continue  # qualifies as LONG — skip SHORT check

        # Try SHORT — gate: must be red, some volume, weaker than market
        if change <= -0.5 and vol >= 1.0 and rs <= 0.0:
            short_result = _score_breakdown(clean_sym, stock)
            if short_result:
                shorts.append(short_result)

    longs.sort(key=lambda x: x["breakout_score"], reverse=True)
    shorts.sort(key=lambda x: x["breakout_score"], reverse=True)

    top_longs  = longs[:15]
    top_shorts = shorts[:15]

    logger.info(
        "Breakout: scored %d symbols → %d LONG, %d SHORT candidates",
        len(sym_data), len(top_longs), len(top_shorts),
    )
    return top_longs + top_shorts


# ---------------------------------------------------------------------------
# Background refresh with cooldown guard
# ---------------------------------------------------------------------------

def refresh_breakout_cache(sym_data: Dict[str, Any], limit: int = 30) -> None:
    """Run in a background thread. Respects 5-minute cooldown between attempts."""
    global _cache

    # Cooldown guard — prevent hammering if fetch keeps returning 0 results
    if time.time() - _cache["last_attempt"] < _COOLDOWN_SECONDS:
        return

    if _cache["is_loading"]:
        return

    _cache["is_loading"] = True
    _cache["last_attempt"] = time.time()

    try:
        results = _compute_breakouts_from_boost(sym_data, limit=limit)
        _cache["breakouts"] = results
        _cache["last_updated"] = datetime.now().strftime("%H:%M:%S")
        logger.info("Breakout cache refreshed — %d results", len(results))
    except Exception as exc:
        logger.error("Breakout cache error: %s", exc)
    finally:
        _cache["is_loading"] = False


# ---------------------------------------------------------------------------
# Public API — returns immediately from cache
# ---------------------------------------------------------------------------

def get_breakout_stocks(sym_data: Dict[str, Any], limit: int = 30) -> Dict[str, Any]:
    """
    Returns immediately from the module-level cache.
    Triggers a background refresh if cache is empty and not already loading.
    Subsequent calls within 5 minutes return the same cached result instantly.
    """
    global _cache

    cache_empty = not _cache["breakouts"]
    cooldown_elapsed = time.time() - _cache["last_attempt"] >= _COOLDOWN_SECONDS

    if (cache_empty or cooldown_elapsed) and not _cache["is_loading"]:
        t = threading.Thread(
            target=refresh_breakout_cache,
            args=(sym_data, limit),
            daemon=True,
        )
        t.start()

    return {
        "breakouts": _cache["breakouts"],
        "last_updated": _cache["last_updated"],
        "is_loading": _cache["is_loading"],
        "count": len(_cache["breakouts"]),
    }


# ---------------------------------------------------------------------------
# Feature 6: 52-Week High Breakout Scanner
# ---------------------------------------------------------------------------

_52w_cache: Dict[str, Any] = {
    "results":      [],
    "last_updated": None,
    "is_loading":   False,
    "last_attempt": 0,
}
_52W_COOLDOWN = 900   # 15 minutes


def _compute_52w_breakouts() -> List[Dict[str, Any]]:
    """
    Compute 52-week high breakout candidates for all F&O stocks.
    Uses batch yfinance download (1-year daily). Expensive — called in background only.
    """
    import os
    os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
    import os
    os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
    import yfinance as yf
    from stocks import FO_STOCKS

    symbols_ns = [sym + ".NS" for sym in FO_STOCKS]
    logger.info("52W scanner: downloading 1-year daily data for %d symbols.", len(symbols_ns))

    from datetime import timedelta
    import pandas as pd

    to_date = datetime.now().date().isoformat()
    from_date = (datetime.now().date() - timedelta(days=380)).isoformat()

    raw = None
    try:
        raw = get_daily_history_batch(symbols_ns, from_date=from_date, to_date=to_date)
    except Exception as exc:
        logger.warning("52W Upstox daily history fetch failed: %s", exc)

    def _extract_symbol_frame(source: Any, symbol: str):
        try:
            if source is None or not isinstance(source, pd.DataFrame) or source.empty:
                return None
            if isinstance(source.columns, pd.MultiIndex):
                level0 = source.columns.get_level_values(0)
                level1 = source.columns.get_level_values(1)
                if symbol in level0:
                    frame = source[symbol]
                elif symbol in level1:
                    frame = source.xs(symbol, axis=1, level=1)
                else:
                    return None
            else:
                frame = source
            return frame if frame is not None and not frame.empty else None
        except Exception:
            return None

    fallback_raw = None
    missing_history = [symbol for symbol in symbols_ns if _extract_symbol_frame(raw, symbol) is None]
    if missing_history:
        try:
            fallback_raw = yf.download(
                tickers=" ".join(missing_history),
                period="1y",
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                progress=False,
                threads=True,
                timeout=60,
            )
        except Exception as exc:
            logger.error("52W yfinance fallback failed for %s: %s", missing_history, exc)

    live_quotes: Dict[str, Dict[str, Any]] = {}
    try:
        live_quotes = get_upstox_bulk_full_quotes(FO_STOCKS)
    except Exception as exc:
        logger.warning("52W Upstox live quote fetch failed: %s", exc)

    results: List[Dict[str, Any]] = []

    for sym_ns in symbols_ns:
        clean = sym_ns.replace(".NS", "")
        try:
            df = _extract_symbol_frame(raw, sym_ns) or _extract_symbol_frame(fallback_raw, sym_ns)
            if df is None or df.empty:
                continue

            closes  = df["Close"].dropna()
            volumes = df["Volume"].dropna()

            if len(closes) < 22:   # need at least 1 month of data
                continue

            quote = live_quotes.get(clean) or {}
            ltp = float(quote.get("ltp") or closes.iloc[-1])
            today_vol = float(quote.get("total_traded_volume") or 0.0) * 100_000
            if today_vol <= 0:
                today_vol = float(volumes.iloc[-1]) if not volumes.empty else 0.0

            # 52W high = max close over last 252 trading days
            w52_high = float(closes.iloc[-252:].max()) if len(closes) >= 252 else float(closes.max())

            # 20-day average volume baseline
            avg_20d_vol = float(volumes.iloc[-21:-1].mean()) if len(volumes) >= 21 else float(volumes.mean())

            if w52_high <= 0 or avg_20d_vol <= 0:
                continue

            pct_from_high = round((w52_high - ltp) / w52_high * 100, 2)
            vol_ratio     = round(today_vol / avg_20d_vol, 2)

            # Breakout conditions
            at_or_above_52w = ltp >= w52_high * 0.995    # within 0.5%
            volume_surge    = vol_ratio >= 2.0

            if not (at_or_above_52w and volume_surge):
                continue

            breakout_type = "ABOVE_HIGH" if ltp > w52_high else "AT_HIGH"
            strength      = "STRONG" if vol_ratio >= 3.0 else "MODERATE"

            results.append({
                "symbol":           clean,
                "ltp":              round(ltp, 2),
                "52w_high":         round(w52_high, 2),
                "pct_from_52w_high": pct_from_high,
                "volume_ratio":     vol_ratio,
                "breakout_type":    breakout_type,
                "strength":         strength,
            })

        except Exception as sym_err:
            logger.debug("52W: skipping %s — %s", clean, sym_err)
            continue

    # Sort: STRONG first, then by volume_ratio desc
    results.sort(
        key=lambda x: (0 if x["strength"] == "STRONG" else 1, -x["volume_ratio"])
    )
    logger.info("52W scanner complete — %d breakout candidates.", len(results))
    return results


def _refresh_52w_cache() -> None:
    global _52w_cache
    if _52w_cache["is_loading"]:
        return
    _52w_cache["is_loading"] = True
    _52w_cache["last_attempt"] = time.time()
    try:
        results = _compute_52w_breakouts()
        _52w_cache["results"]      = results
        _52w_cache["last_updated"] = datetime.now().strftime("%H:%M:%S")
        logger.info("52W cache refreshed — %d results.", len(results))
    except Exception as exc:
        logger.error("52W cache refresh error: %s", exc)
    finally:
        _52w_cache["is_loading"] = False


def scan_52w_breakouts() -> Dict[str, Any]:
    """
    Returns 52-week high breakout candidates from cache.
    Triggers a background refresh if cache is stale (>15 min) or empty.
    """
    global _52w_cache

    cache_empty      = not _52w_cache["results"]
    cooldown_elapsed = time.time() - _52w_cache["last_attempt"] >= _52W_COOLDOWN

    if (cache_empty or cooldown_elapsed) and not _52w_cache["is_loading"]:
        t = threading.Thread(target=_refresh_52w_cache, daemon=True)
        t.start()

    return {
        "results":      _52w_cache["results"],
        "last_updated": _52w_cache["last_updated"],
        "is_loading":   _52w_cache["is_loading"],
        "count":        len(_52w_cache["results"]),
    }

