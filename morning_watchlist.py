# morning_watchlist.py — ORB + Sector Momentum + Volume Watchlist Generator

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytz

logger = logging.getLogger("morning_watchlist")

IST = pytz.timezone("Asia/Kolkata")

# 15-minute Opening Range = first 3 candles of the day (9:15, 9:20, 9:25)
ORB_MINUTES: List[Tuple[int, int]] = [(9, 15), (9, 20), (9, 25)]

# Day has ~375 trading minutes; 5-min candles = 75 candles.
# First 3 candles (15 min) ≈ 1/25 of the day → avg_15min_vol ≈ avg_daily_vol / 25
_AVG_CANDLES_PER_DAY = 25.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_top_sector_ranks(date: str, is_live: bool = False) -> List[Tuple[str, int]]:
    """
    Returns [(sector_name, rank), ...] for the top 3 sectors by momentum
    'current' value.  Uses live cache when is_live=True, else historical fetch.
    """
    try:
        if is_live:
            from sector_momentum import get_momentum_data
            momentum = get_momentum_data()
        else:
            from sector_momentum import get_historical_momentum
            momentum = get_historical_momentum(date)

        sectors_dict = momentum.get("sectors", {})
        ranked = sorted(
            sectors_dict.items(),
            key=lambda kv: kv[1].get("result", {}).get("current", 0.0),
            reverse=True,
        )
        return [(name, idx + 1) for idx, (name, _) in enumerate(ranked[:5])]
    except Exception as exc:
        logger.error("_get_top_sector_ranks failed: %s", exc)
        return []


def _build_watchlist(date: str, top_sector_ranks: List[Tuple[str, int]]) -> Dict[str, Any]:
    """
    Core computation: downloads price data and scores every stock in the
    top-3 sectors.
    """
    import pandas as pd
    import os
    os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
    import yfinance as yf
    from stocks import SECTORS

    if not top_sector_ranks:
        return {
            "date": date,
            "generated_at": datetime.now(IST).strftime("%H:%M:%S"),
            "watchlist": [],
            "top_long": [],
            "top_short": [],
            "top_sectors": [],
            "error": "No sector momentum data available",
        }

    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date format '{date}': use YYYY-MM-DD") from exc

    next_day        = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
    daily_start     = (dt - timedelta(days=14)).strftime("%Y-%m-%d")  # ~10 trading days
    target_date_obj = dt.date()

    top_sectors    = [name for name, _ in top_sector_ranks]
    sector_rank_map = {name: rank for name, rank in top_sector_ranks}

    logger.info("Building watchlist for %s | top sectors: %s", date, top_sectors)

    watchlist: List[Dict[str, Any]] = []

    for sector_name in top_sectors:
        symbols    = list(SECTORS.get(sector_name, []))
        symbols_ns = [s if s.endswith(".NS") else s + ".NS" for s in symbols]
        if not symbols_ns:
            continue

        tickers_str = " ".join(symbols_ns)

        # ── Daily data: prev_day close/high/low + avg volume ──────────────
        daily_raw: Optional[Any] = None
        try:
            daily_raw = yf.download(
                tickers=tickers_str,
                start=daily_start,
                end=date,           # exclusive → last row = prev trading day
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if daily_raw is not None and not daily_raw.empty:
                if daily_raw.index.tz is not None:
                    daily_raw.index = (
                        daily_raw.index
                        .tz_convert("Asia/Kolkata")
                        .tz_localize(None)
                    )
        except Exception as exc:
            logger.warning("Daily download failed for %s: %s", sector_name, exc)

        # ── Intraday 5-min data ───────────────────────────────────────────
        try:
            intra_raw = yf.download(
                tickers=tickers_str,
                start=date,
                end=next_day,
                interval="5m",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.warning("Intraday download failed for %s: %s", sector_name, exc)
            continue

        if intra_raw is None or intra_raw.empty:
            logger.warning("Empty intraday for %s", sector_name)
            continue

        # UTC → naive IST
        if intra_raw.index.tz is not None:
            intra_raw.index = (
                intra_raw.index
                .tz_convert("Asia/Kolkata")
                .tz_localize(None)
            )

        # Keep only target date rows
        intra_raw = intra_raw[intra_raw.index.date == target_date_obj]
        if intra_raw.empty:
            logger.warning("No IST intraday rows for %s on %s", sector_name, date)
            continue

        # ── Per-symbol processing ─────────────────────────────────────────
        for sym_ns in symbols_ns:
            sym_display = sym_ns.replace(".NS", "")
            try:
                # Extract this symbol's intraday slice
                if isinstance(intra_raw.columns, pd.MultiIndex):
                    avail = intra_raw.columns.get_level_values(1).unique().tolist()
                    if sym_ns not in avail:
                        continue
                    intra_df = intra_raw.xs(sym_ns, axis=1, level=1)
                else:
                    intra_df = intra_raw  # single-ticker fallback

                if intra_df.empty:
                    continue

                # ── ORB: first three 5-min candles (9:15, 9:20, 9:25) ────
                orb_mask = intra_df.index.map(
                    lambda t: (t.hour, t.minute) in ORB_MINUTES
                )
                orb_df = intra_df[orb_mask]
                if orb_df.empty:
                    continue

                orb_high          = round(float(orb_df["High"].max()), 2)
                orb_low           = round(float(orb_df["Low"].min()), 2)
                first_15min_vol   = float(orb_df["Volume"].sum())

                # Current price = last available close
                current_price = round(float(intra_df["Close"].dropna().iloc[-1]), 2)

                # ── Previous-day stats ────────────────────────────────────
                prev_close    = 0.0
                prev_day_high = 0.0
                prev_day_low  = 0.0
                avg_15min_vol = 0.0

                if daily_raw is not None and not daily_raw.empty:
                    try:
                        if isinstance(daily_raw.columns, pd.MultiIndex):
                            d_avail = daily_raw.columns.get_level_values(1).unique().tolist()
                            if sym_ns in d_avail:
                                daily_df = daily_raw.xs(sym_ns, axis=1, level=1).dropna(how="all")
                                if not daily_df.empty:
                                    prev_close     = round(float(daily_df["Close"].iloc[-1]), 2)
                                    prev_day_high  = round(float(daily_df["High"].iloc[-1]), 2)
                                    prev_day_low   = round(float(daily_df["Low"].iloc[-1]), 2)
                                    avg_daily_vol  = float(daily_df["Volume"].mean())
                                    avg_15min_vol  = avg_daily_vol / _AVG_CANDLES_PER_DAY
                        else:
                            daily_df = daily_raw.dropna(how="all")
                            if not daily_df.empty:
                                prev_close     = round(float(daily_df["Close"].iloc[-1]), 2)
                                prev_day_high  = round(float(daily_df["High"].iloc[-1]), 2)
                                prev_day_low   = round(float(daily_df["Low"].iloc[-1]), 2)
                                avg_daily_vol  = float(daily_df["Volume"].mean())
                                avg_15min_vol  = avg_daily_vol / _AVG_CANDLES_PER_DAY
                    except Exception as exc:
                        logger.warning("Daily stats error for %s: %s", sym_ns, exc)

                # ── Volume ratio ──────────────────────────────────────────
                volume_ratio = (
                    round(first_15min_vol / avg_15min_vol, 2)
                    if avg_15min_vol > 0 else 0.0
                )

                # ── Bias ──────────────────────────────────────────────────
                if current_price > orb_high:
                    bias = "LONG"
                elif current_price < orb_low:
                    bias = "SHORT"
                else:
                    bias = "WAIT"

                # ── Scoring ───────────────────────────────────────────────
                sector_rank       = sector_rank_map.get(sector_name, 3)
                sector_rank_score = 4 - sector_rank   # rank1→3, rank2→2, rank3→1

                if volume_ratio > 3:
                    volume_score = 3
                elif volume_ratio > 2:
                    volume_score = 2
                elif volume_ratio > 1.5:
                    volume_score = 1
                else:
                    volume_score = 0

                # Broke ORB = 2pts; within 0.3% of either ORB line = 1pt
                if bias in ("LONG", "SHORT"):
                    orb_score = 2
                elif (
                    orb_high > 0 and abs(current_price - orb_high) / orb_high < 0.003
                ) or (
                    orb_low > 0 and abs(current_price - orb_low) / orb_low < 0.003
                ):
                    orb_score = 1
                else:
                    orb_score = 0

                total_score = sector_rank_score + volume_score + orb_score

                logger.debug(
                    "%s | bias=%s score=%d (sec=%d vol=%d orb=%d) vr=%.2f",
                    sym_display, bias, total_score,
                    sector_rank_score, volume_score, orb_score, volume_ratio,
                )

                watchlist.append({
                    "symbol":        sym_display,
                    "sector":        sector_name,
                    "sector_rank":   sector_rank,
                    "orb_high":      orb_high,
                    "orb_low":       orb_low,
                    "current_price": current_price,
                    "prev_close":    prev_close,
                    "prev_day_high": prev_day_high,
                    "prev_day_low":  prev_day_low,
                    "volume_ratio":  volume_ratio,
                    "bias":          bias,
                    "action":        "BUY" if (bias == "LONG" and total_score >= 5) else "SELL" if (bias == "SHORT" and total_score >= 5) else "WATCH",
                    "score":         total_score,
                    "total_score":   total_score,
                    "breakdown": {
                        "sector": sector_rank_score,
                        "volume": volume_score,
                        "orb":    orb_score,
                    },
                })

            except Exception as exc:
                logger.warning("Error processing %s (%s): %s", sym_ns, sector_name, exc)
                continue

    # Sort by score descending, then by volume_ratio as tiebreaker
    watchlist.sort(key=lambda x: (x["total_score"], x["volume_ratio"]), reverse=True)

    top_long  = [w for w in watchlist if w["bias"] == "LONG"  and w["total_score"] >= 6]
    top_short = [w for w in watchlist if w["bias"] == "SHORT" and w["total_score"] >= 6]

    # Build enriched top_sectors list with majority bias and avg change_pct
    top_sectors_info: List[Dict[str, Any]] = []
    for sector_name in top_sectors:
        sector_stocks = [w for w in watchlist if w["sector"] == sector_name]
        if sector_stocks:
            bias_counts: Dict[str, int] = {"LONG": 0, "SHORT": 0, "WAIT": 0}
            change_pcts: List[float] = []
            for w in sector_stocks:
                bias_counts[w["bias"]] = bias_counts.get(w["bias"], 0) + 1
                if w["prev_close"] > 0:
                    change_pcts.append(
                        round((w["current_price"] - w["prev_close"]) / w["prev_close"] * 100, 2)
                    )
            majority_bias  = max(bias_counts, key=lambda k: bias_counts[k])
            avg_change_pct = round(sum(change_pcts) / len(change_pcts), 2) if change_pcts else 0.0
        else:
            majority_bias  = "WAIT"
            avg_change_pct = 0.0
        top_sectors_info.append({
            "name":       sector_name,
            "bias":       majority_bias,
            "change_pct": avg_change_pct,
        })

    return {
        "date":          date,
        "generated_at":  datetime.now(IST).strftime("%H:%M:%S"),
        "watchlist":     watchlist,
        "top_long":      top_long,
        "top_short":     top_short,
        "top_sectors":   top_sectors_info,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_morning_watchlist(date: str) -> Dict[str, Any]:
    """
    Historical morning watchlist for a given date (YYYY-MM-DD).
    Derives top sectors from get_historical_momentum(date).
    """
    from sector_momentum import get_historical_momentum
    momentum = get_historical_momentum(date)
    sectors_dict = momentum.get("sectors", {})

    ranked = sorted(
        sectors_dict.items(),
        key=lambda kv: kv[1].get("result", {}).get("current", 0.0),
        reverse=True,
    )
    top_sector_ranks = [(name, idx + 1) for idx, (name, _) in enumerate(ranked[:5])]

    # Build top_sectors_info directly from momentum data (authoritative source)
    top_sectors_info: List[Dict[str, Any]] = []
    for name, _rank in top_sector_ranks:
        current_val = sectors_dict.get(name, {}).get("result", {}).get("current", 0.0)
        bias = "LONG" if current_val > 0.5 else "SHORT" if current_val < -0.5 else "WAIT"
        top_sectors_info.append({
            "name":       name,
            "bias":       bias,
            "change_pct": round(current_val, 2),
        })

    result = _build_watchlist(date, top_sector_ranks)
    result["top_sectors"] = top_sectors_info
    return result


def get_live_watchlist() -> Dict[str, Any]:
    """
    Today's morning watchlist using the live in-memory sector momentum cache.
    """
    today = datetime.now(IST).strftime("%Y-%m-%d")
    top_sector_ranks = _get_top_sector_ranks(today, is_live=True)
    return _build_watchlist(today, top_sector_ranks)
