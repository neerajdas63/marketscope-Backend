# fetcher.py — Downloads and processes Indian stock market data via yfinance

import time
import logging
import threading
from statistics import mean
from datetime import datetime, time as dt_time
from typing import Any, Dict, List

import pandas as pd
import pytz

import os
os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
import yfinance as yf

from angel_client import get_bulk_ltp, get_bulk_full_quotes
from upstox_client import get_bulk_daily_ohlc, get_bulk_full_quotes as get_upstox_bulk_full_quotes
from stocks import ALL_SYMBOLS, SECTORS, FO_STOCKS, SCANNER_STOCKS
from nse_fetcher import fetch_nse_index_quotes
from intraday_boost import calculate_intraday_boost

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")
DAILY_BASELINE_REFRESH_SECONDS = int(os.getenv("DAILY_BASELINE_REFRESH_SECONDS", "21600"))

_daily_baseline_lock = threading.Lock()
_daily_baseline_cache: Dict[str, Any] = {
    "trading_date": "",
    "fetched_at": 0.0,
    "prev_close_by_symbol": {},
    "avg_volume_by_symbol": {},
    "is_loading": False,
    "last_attempt": 0.0,
}


def _get_sector_index_change_pct(
    sector_name: str,
    official_indices: Dict[str, Dict[str, Any]],
    fallback_change_pct: float,
) -> float:
    official = official_indices.get(sector_name) or official_indices.get(sector_name.strip())
    if official:
        return round(float(official.get("percentChange", fallback_change_pct) or fallback_change_pct), 2)
    return round(fallback_change_pct, 2)


def _is_market_open() -> bool:
    """Return True if the NSE market is currently open (Mon–Fri, 09:15–15:30 IST)."""
    now_ist = datetime.now(IST)
    weekday = now_ist.weekday()  # 0 = Monday, 6 = Sunday
    if weekday > 4:
        return False
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now_ist <= market_close


def _safe_float(value: Any) -> float:
    """Safely convert a value to float, returning 0.0 on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _get_sym_df(data: Any, symbol: str):
    """Extract a single symbol dataframe from a flat or MultiIndex yfinance response."""
    try:
        if data is None:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            if symbol in data.columns.get_level_values(0):
                df = data[symbol]
                return df if not df.empty else None
            return None
        return data if not data.empty else None
    except Exception:
        return None


def _batch_download(symbols: List[str], **kwargs):
    results = []
    for index in range(0, len(symbols), 30):
        batch = symbols[index:index + 30]
        try:
            df = yf.download(tickers=" ".join(batch), **kwargs)
            if df is not None and not df.empty:
                results.append(df)
        except Exception as exc:
            logger.warning("yfinance batch download failed for %s: %s", batch, exc)
        time.sleep(1)
    if results:
        return pd.concat(results, axis=1)
    return None


def _snapshot_daily_baseline() -> Dict[str, Any]:
    with _daily_baseline_lock:
        return {
            "trading_date": str(_daily_baseline_cache.get("trading_date") or ""),
            "fetched_at": _safe_float(_daily_baseline_cache.get("fetched_at")),
            "prev_close_by_symbol": dict(_daily_baseline_cache.get("prev_close_by_symbol") or {}),
            "avg_volume_by_symbol": dict(_daily_baseline_cache.get("avg_volume_by_symbol") or {}),
            "is_loading": bool(_daily_baseline_cache.get("is_loading")),
            "last_attempt": _safe_float(_daily_baseline_cache.get("last_attempt")),
        }


def _daily_baseline_is_stale(snapshot: Dict[str, Any]) -> bool:
    today_key = datetime.now(IST).date().isoformat()
    if str(snapshot.get("trading_date") or "") != today_key:
        return True
    fetched_at = _safe_float(snapshot.get("fetched_at"))
    if fetched_at <= 0:
        return True
    return (time.time() - fetched_at) >= DAILY_BASELINE_REFRESH_SECONDS


def _refresh_daily_baseline_cache(symbols: List[str]) -> None:
    try:
        logger.info("Refreshing daily baseline cache for %d symbols...", len(symbols))
        upstox_prev_close_by_symbol: Dict[str, float] = {}
        try:
            ohlc_rows = get_bulk_daily_ohlc([symbol.replace(".NS", "") for symbol in symbols])
            upstox_prev_close_by_symbol = {
                symbol: _safe_float(payload.get("prev_close"))
                for symbol, payload in ohlc_rows.items()
                if _safe_float(payload.get("prev_close")) > 0
            }
            if upstox_prev_close_by_symbol:
                logger.info("Upstox daily OHLC provided prev-close data for %d symbols.", len(upstox_prev_close_by_symbol))
        except Exception as exc:
            logger.warning("Upstox daily OHLC baseline fetch failed: %s", exc)

        daily_data = _batch_download(
            symbols,
            period="5d",
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=True,
            timeout=20,
        )
        prev_close_by_symbol: Dict[str, float] = {}
        avg_volume_by_symbol: Dict[str, float] = {}
        for symbol in symbols:
            clean_symbol = symbol.replace(".NS", "")
            if _safe_float(upstox_prev_close_by_symbol.get(clean_symbol)) > 0:
                prev_close_by_symbol[clean_symbol] = _safe_float(upstox_prev_close_by_symbol.get(clean_symbol))
            df = _get_sym_df(daily_data, symbol)
            if df is None:
                continue
            close_series = df.get("Close")
            volume_series = df.get("Volume")
            if close_series is not None:
                close_values = close_series.dropna()
                if len(close_values) >= 2 and clean_symbol not in prev_close_by_symbol:
                    prev_close_by_symbol[clean_symbol] = _safe_float(close_values.iloc[-2])
            if volume_series is not None:
                volume_values = volume_series.dropna()
                if len(volume_values) >= 3:
                    avg_volume_by_symbol[clean_symbol] = float(volume_values.tail(20).mean())

        with _daily_baseline_lock:
            _daily_baseline_cache["trading_date"] = datetime.now(IST).date().isoformat()
            _daily_baseline_cache["fetched_at"] = time.time()
            _daily_baseline_cache["prev_close_by_symbol"] = prev_close_by_symbol
            _daily_baseline_cache["avg_volume_by_symbol"] = avg_volume_by_symbol
        logger.info(
            "Daily baseline cache refreshed: %d prev closes, %d avg-volume entries.",
            len(prev_close_by_symbol),
            len(avg_volume_by_symbol),
        )
    except Exception as exc:
        logger.warning("Daily baseline cache refresh failed: %s", exc)
    finally:
        with _daily_baseline_lock:
            _daily_baseline_cache["is_loading"] = False


def _ensure_daily_baseline_cache(symbols: List[str]) -> Dict[str, Any]:
    snapshot = _snapshot_daily_baseline()
    if not _daily_baseline_is_stale(snapshot):
        return snapshot

    if snapshot["is_loading"]:
        return snapshot

    with _daily_baseline_lock:
        if _daily_baseline_cache["is_loading"]:
            return _snapshot_daily_baseline()
        _daily_baseline_cache["is_loading"] = True
        _daily_baseline_cache["last_attempt"] = time.time()

    threading.Thread(
        target=_refresh_daily_baseline_cache,
        args=(list(symbols),),
        daemon=True,
        name="daily-baseline-refresh",
    ).start()
    return _snapshot_daily_baseline()


def _apply_neutral_rfactor_fields(sym_data: Dict[str, Any]) -> Dict[str, Any]:
    """Populate neutral R-Factor fields so downstream features keep working when R-Factor is disabled."""
    for stock in sym_data.values():
        stock.setdefault("rfactor", 0.0)
        stock.setdefault("tier", "very_weak")
        stock.setdefault("rsi", 50.0)
        stock.setdefault("mfi", 50.0)
        stock.setdefault("relative_strength", 0.0)
        stock.setdefault("setup_stage", "NEUTRAL")
        stock.setdefault("alert_stage", "NEUTRAL")
        stock.setdefault("opportunity_score", 0.0)
        stock.setdefault("rfactor_trend_15m", 0.0)
        stock.setdefault("rfactor_trend_points", [0.0])
        stock.setdefault("rfactor_trend_acceleration", 0.0)
        stock.setdefault("pre_score", 0.0)
        stock.setdefault("trigger_score", 0.0)
        stock.setdefault("inferred_direction", "NEUTRAL")
        stock.setdefault("direction_conf", 0.0)
        stock.setdefault("compression", 0.0)
        stock.setdefault("obv_slope_score", 0.0)
        stock.setdefault("vol_accel", 1.0)
        stock.setdefault("rsi_slope_5m", 0.0)
        stock.setdefault("nearest_level", "")
        stock.setdefault("proximity_score", 0.0)
        stock.setdefault("dist_pct", 999.0)
        stock.setdefault("breakout_levels", {})
        stock.setdefault("breakout_quality", 0.0)
        stock.setdefault("vwap_acceptance", 0.0)
        stock.setdefault("is_chase", False)
        stock.setdefault("chase_reason", "")
    return sym_data


def _compute_change_pct(ltp: float, prev_close: float, fallback_change_pct: float = 0.0) -> float:
    if prev_close > 0:
        return round(((ltp - prev_close) / prev_close) * 100, 2)
    return round(fallback_change_pct, 2)


def _build_sym_data(
    symbols: List[str],
    primary_quotes: Dict[str, Any],
    secondary_quotes: Dict[str, Any],
    angel_ltp: Dict[str, float],
    prev_close_reference: Dict[str, float],
    avg_volume_by_symbol: Dict[str, float],
) -> Dict[str, Any]:
    sym_data: Dict[str, Any] = {}
    for symbol in symbols:
        try:
            clean = symbol.replace(".NS", "")
            q = primary_quotes.get(clean) or secondary_quotes.get(clean)
            angel_price = angel_ltp.get(clean)

            if not q or q.get("ltp", 0) <= 0:
                prev_close_fb = _safe_float(prev_close_reference.get(clean))
                ltp_fb = round(float(angel_price or 0), 2)
                if ltp_fb <= 0 or prev_close_fb <= 0:
                    continue
                change_fb = _compute_change_pct(ltp_fb, prev_close_fb)
                sym_data[clean] = {
                    "symbol": clean,
                    "ltp": ltp_fb,
                    "change_pct": change_fb,
                    "volume_ratio": 1.0,
                    "fo": clean in FO_STOCKS,
                    "day_high": ltp_fb,
                    "day_low": ltp_fb,
                    "day_open": ltp_fb,
                    "vwap": 0.0,
                    "quote_source": "yfinance_fallback" if not angel_price else "angel_ltp_fallback",
                    "delivery_source": "unavailable_fallback",
                    "delivery_pct": None,
                    "bid_ask_ratio": None,
                    "bid_qty": None,
                    "ask_qty": None,
                }
                continue

            live_ltp = round(float(angel_price or q.get("ltp", 0) or 0), 2)
            quote_source = str(q.get("quote_source") or "")
            raw_change_pct = float(q.get("change_pct", 0) or 0)
            prev_close = float(q.get("prev_close", 0) or 0)
            if quote_source == "upstox_full":
                prev_close = _safe_float(prev_close_reference.get(clean)) or prev_close
            elif prev_close <= 0:
                prev_close = _safe_float(prev_close_reference.get(clean))
            change_pct = _compute_change_pct(live_ltp, prev_close, raw_change_pct)

            vol_ratio = 1.0
            try:
                nse_vol_shares = q.get("total_traded_volume", 0.0) * 100_000
                avg_20d = _safe_float(avg_volume_by_symbol.get(clean))
                if avg_20d > 0 and nse_vol_shares > 0:
                    vol_ratio = round(nse_vol_shares / avg_20d, 2)
            except Exception:
                pass

            sym_data[clean] = {
                "symbol": clean,
                "ltp": live_ltp,
                "change_pct": change_pct,
                "volume_ratio": vol_ratio,
                "fo": clean in FO_STOCKS,
                "day_high": q.get("day_high", live_ltp or q["ltp"]),
                "day_low": q.get("day_low", live_ltp or q["ltp"]),
                "day_open": q.get("day_open", live_ltp or q["ltp"]),
                "vwap": q.get("vwap", 0.0),
                "quote_source": q.get("quote_source", "smartapi_full"),
                "delivery_source": q.get("delivery_source", "unknown"),
                "delivery_pct": round(float(q.get("delivery_pct", 0) or 0), 1) if q.get("delivery_pct") is not None else None,
                "bid_ask_ratio": round(float(q.get("bid_ask_ratio", 0) or 0), 2) if q.get("bid_qty") or q.get("ask_qty") else None,
                "bid_qty": q.get("bid_qty") if q.get("bid_qty") or q.get("ask_qty") else None,
                "ask_qty": q.get("ask_qty") if q.get("bid_qty") or q.get("ask_qty") else None,
            }
        except Exception as exc:
            logger.warning("Skipping %s: %s", symbol, exc)
            continue
    return sym_data


def _build_heatmap_diagnostics(sym_data: Dict[str, Any]) -> Dict[str, Any]:
    quote_source_counts: Dict[str, int] = {}
    zero_change_symbols: List[str] = []
    near_zero_change_symbols: List[str] = []
    non_zero_change_count = 0
    neutral_direction_count = 0

    for symbol, stock in sym_data.items():
        quote_source = str(stock.get("quote_source") or "unknown")
        quote_source_counts[quote_source] = quote_source_counts.get(quote_source, 0) + 1

        change_pct = _safe_float(stock.get("change_pct"))
        if change_pct == 0.0:
            zero_change_symbols.append(symbol)
        else:
            non_zero_change_count += 1
            if abs(change_pct) < 0.05:
                near_zero_change_symbols.append(symbol)

        if str(stock.get("inferred_direction") or "NEUTRAL") == "NEUTRAL":
            neutral_direction_count += 1

    total_symbols = len(sym_data)
    return {
        "total_symbols": total_symbols,
        "quote_source_counts": quote_source_counts,
        "zero_change_count": len(zero_change_symbols),
        "non_zero_change_count": non_zero_change_count,
        "zero_change_ratio": round((len(zero_change_symbols) / total_symbols), 3) if total_symbols else 0.0,
        "neutral_direction_count": neutral_direction_count,
        "zero_change_samples": zero_change_symbols[:20],
        "near_zero_change_samples": near_zero_change_symbols[:20],
    }


def _fetch_prev_close(symbol: str) -> float:
    """
    Fetch the official previous trading day closing price for a symbol
    using daily candles (auto_adjust=False to match NSE prices exactly).

    Returns 0.0 on any failure.
    """
    try:
        daily_data = yf.download(
            tickers=symbol,
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if daily_data is None or daily_data.empty:
            return 0.0

        # yfinance may return multi-level columns even for a single ticker
        if isinstance(daily_data.columns, pd.MultiIndex):
            close_series = daily_data["Close"].squeeze().dropna()
        else:
            close_series = daily_data["Close"].dropna()

        # Need at least 2 rows: [..., prev_day, today]
        if len(close_series) < 2:
            return 0.0

        return _safe_float(close_series.iloc[-2])
    except Exception as e:
        logger.warning(f"Failed to fetch daily prev_close for {symbol}: {e}")
        return 0.0


def fetch_sector(sector_name: str, symbols: List[str]) -> Dict[str, Any]:
    """
    Download intraday data for all symbols in a single sector batch and
    compute per-stock metrics.

    Returns a sector dict:
    {
        "name": str,
        "change_pct": float,
        "stocks": [ { symbol, ltp, change_pct, volume_ratio, fo }, ... ]
    }
    """
    if not symbols:
        return {"name": sector_name, "change_pct": 0.0, "stocks": []}

    stocks_data: List[Dict[str, Any]] = []

    try:
        tickers_data = yf.download(
            tickers=" ".join(symbols),
            period="2d",
            interval="5m",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        for symbol in symbols:
            try:
                # Handle yfinance single vs multi-ticker structure difference
                if len(symbols) == 1:
                    df = tickers_data
                else:
                    if symbol not in tickers_data.columns.get_level_values(0):
                        logger.warning(f"Symbol {symbol} not found in downloaded data, skipping.")
                        continue
                    df = tickers_data[symbol]

                if df is None or df.empty:
                    logger.warning(f"Empty dataframe for {symbol}, skipping.")
                    continue

                # Convert index to IST and filter to market hours only
                df_ist = df.copy()
                if df_ist.index.tzinfo is None:
                    df_ist.index = df_ist.index.tz_localize("UTC").tz_convert(IST)
                else:
                    df_ist.index = df_ist.index.tz_convert(IST)

                df_market = df_ist.between_time("09:15", "15:30")

                # Fallback to full df if market-hours filter yields nothing (e.g. market closed)
                if df_market.empty:
                    logger.warning(f"No market-hours candles for {symbol}, falling back to full data.")
                    df_market = df_ist

                # Split into today's and previous day's market-hours candles
                today_date = datetime.now(IST).date()
                today_df = df_market[df_market.index.date == today_date]
                prev_df = df_market[df_market.index.date < today_date]

                # If today has no candles yet (pre-market), fall back to all available data
                if today_df.empty:
                    logger.warning(f"No today candles for {symbol}, falling back to full market data.")
                    today_df = df_market

                close_today = today_df["Close"].dropna()
                volume_today = today_df["Volume"].dropna()

                if close_today.empty:
                    logger.warning(f"No price data for {symbol}, skipping.")
                    continue

                ltp = _safe_float(close_today.iloc[-1])

                # prev_close: use official daily candle (auto_adjust=False) for exact NSE match
                prev_close = _fetch_prev_close(symbol)

                # Fallback: last Close of previous day's intraday candles
                if prev_close == 0.0 and not prev_df.empty:
                    prev_close_series = prev_df["Close"].dropna()
                    prev_close = _safe_float(prev_close_series.iloc[-1]) if not prev_close_series.empty else 0.0

                # Last resort: today's first Open
                if prev_close == 0.0:
                    open_today = today_df["Open"].dropna()
                    prev_close = _safe_float(open_today.iloc[0]) if not open_today.empty else 0.0

                if prev_close == 0.0:
                    logger.warning(f"Zero prev_close for {symbol}, skipping.")
                    continue

                change_pct = round(((ltp - prev_close) / prev_close) * 100, 2)

                curr_vol = _safe_float(volume_today.iloc[-1]) if not volume_today.empty else 0.0
                avg_vol = _safe_float(volume_today.mean()) if not volume_today.empty else 0.0
                volume_ratio = round(curr_vol / avg_vol, 1) if avg_vol > 0 else 1.0

                # Strip .NS to check F&O eligibility
                base_symbol = symbol.replace(".NS", "")
                fo = base_symbol in FO_STOCKS

                stocks_data.append({
                    "symbol": base_symbol,
                    "ltp": round(ltp, 2),
                    "change_pct": change_pct,
                    "volume_ratio": volume_ratio,
                    "fo": fo,
                })

            except Exception as sym_err:
                logger.error(f"Error processing symbol {symbol}: {sym_err}")
                continue

    except Exception as batch_err:
        logger.error(f"Error downloading batch for sector '{sector_name}': {batch_err}")
        return {"name": sector_name, "change_pct": 0.0, "stocks": []}

    # Sort stocks by absolute change_pct descending
    stocks_data.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

    # Sector-level average change
    sector_change_pct = (
        round(mean(s["change_pct"] for s in stocks_data), 2)
        if stocks_data
        else 0.0
    )

    return {
        "name": sector_name,
        "change_pct": sector_change_pct,
        "stocks": stocks_data,
    }


def fetch_all_sectors() -> Dict[str, Any]:
    """
    Fetch daily + intraday data for ALL symbols in exactly two batched
    yf.download() calls, then assemble sector results.
    This is significantly faster than per-sector or per-symbol fetching.

    Returns:
    {
        "sectors": [ { name, change_pct, stocks: [ ... ] }, ... ],
        "last_updated": "HH:MM:SS",
        "market_open": bool
    }
    """
    start = time.time()

    # Combined unique symbols: heatmap sectors + scanner universe
    _combined_symbols = list({sym for sym in ALL_SYMBOLS + SCANNER_STOCKS})
    scanner_symbols = list(dict.fromkeys(SCANNER_STOCKS))
    clean_symbols = [s.replace(".NS", "") for s in _combined_symbols]
    scanner_clean_symbols = [s.replace(".NS", "") for s in scanner_symbols]

    def get_sym_df(data, symbol):
        """Extract a single symbol's DataFrame from a (possibly multi-ticker) download result."""
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if symbol in data.columns.get_level_values(0):
                    df = data[symbol]
                    return df if not df.empty else None
                return None
            else:
                return data
        except Exception:
            return None

    # STEP 1 — Upstox FULL quotes (primary live market data).
    # Returns per symbol: ltp, change_pct, prev_close, vwap, day_high, day_low,
    #                     total_traded_volume (lakhs), bid_ask_ratio, bid_qty, ask_qty
    logger.info(f"Fetching Upstox FULL quotes for {len(scanner_clean_symbols)} scanner symbols...")
    upstox_full: Dict[str, Any] = {}
    try:
        upstox_full = get_upstox_bulk_full_quotes(scanner_clean_symbols)
        logger.info(f"Upstox quotes received for {len(upstox_full)}/{len(scanner_clean_symbols)} scanner symbols.")
    except Exception as e:
        logger.warning(f"Upstox full quote fetch failed: {e}")

    logger.info(f"Fetching SmartAPI FULL quotes for {len(clean_symbols)} symbols...")
    smartapi_full: Dict[str, Any] = {}
    try:
        smartapi_full = get_bulk_full_quotes(clean_symbols)
        logger.info(f"SmartAPI quotes received for {len(smartapi_full)}/{len(clean_symbols)} symbols.")
    except Exception as e:
        logger.warning(f"SmartAPI full quote fetch failed: {e}")

    angel_ltp: Dict[str, float] = {}
    missing_for_ltp = [symbol for symbol in clean_symbols if symbol not in upstox_full and symbol not in smartapi_full]
    if missing_for_ltp:
        logger.info(f"Falling back to Angel LTP for {len(missing_for_ltp)} symbols...")
        try:
            angel_ltp = get_bulk_ltp(missing_for_ltp)
            logger.info(f"Angel LTP received for {len(angel_ltp)}/{len(missing_for_ltp)} symbols.")
        except Exception as e:
            logger.warning(f"Angel LTP fetch failed: {e}")

    logger.info("Fetching official NSE sector indices...")
    official_indices: Dict[str, Dict[str, Any]] = {}
    try:
        official_indices = fetch_nse_index_quotes()
        logger.info("Official NSE indices received for %d entries.", len(official_indices))
    except Exception as e:
        logger.warning(f"NSE index quote fetch failed: {e}")

    baseline_snapshot = _ensure_daily_baseline_cache(ALL_SYMBOLS)
    prev_close_by_symbol = baseline_snapshot.get("prev_close_by_symbol", {})
    avg_volume_by_symbol = baseline_snapshot.get("avg_volume_by_symbol", {})
    live_prev_close_by_symbol: Dict[str, float] = {}
    if not prev_close_by_symbol:
        logger.info("Daily baseline cache is warming up; using live quote fields and neutral volume ratios for now.")
        try:
            ohlc_rows = get_bulk_daily_ohlc(clean_symbols)
            live_prev_close_by_symbol = {
                symbol: _safe_float(payload.get("prev_close"))
                for symbol, payload in ohlc_rows.items()
                if _safe_float(payload.get("prev_close")) > 0
            }
            if live_prev_close_by_symbol:
                logger.info(
                    "Filled live prev-close data for %d symbols from Upstox daily OHLC during heatmap fetch.",
                    len(live_prev_close_by_symbol),
                )
        except Exception as exc:
            logger.warning("Synchronous Upstox prev-close fill failed during heatmap fetch: %s", exc)

    prev_close_reference = dict(prev_close_by_symbol)
    if live_prev_close_by_symbol:
        prev_close_reference.update(live_prev_close_by_symbol)

    heatmap_sym_data = _build_sym_data(
        symbols=_combined_symbols,
        primary_quotes=smartapi_full,
        secondary_quotes=upstox_full,
        angel_ltp=angel_ltp,
        prev_close_reference=prev_close_reference,
        avg_volume_by_symbol=avg_volume_by_symbol,
    )
    momentum_sym_data = _build_sym_data(
        symbols=scanner_symbols,
        primary_quotes=upstox_full,
        secondary_quotes=smartapi_full,
        angel_ltp=angel_ltp,
        prev_close_reference=prev_close_reference,
        avg_volume_by_symbol=avg_volume_by_symbol,
    )

    logger.info(
        "sym_data built: heatmap=%d (SmartAPI primary), momentum=%d (Upstox primary).",
        len(heatmap_sym_data),
        len(momentum_sym_data),
    )

    # STEP 5 — quote depth data for rfactor (delivery%, bid/ask) — sourced from live quotes
    if not upstox_full and not smartapi_full:
        logger.warning("Live depth data empty — using neutral placeholder values.")

    logger.info("R-Factor remains disabled — skipping unused 15-minute downloads and filling neutral placeholder fields.")
    heatmap_sym_data = _apply_neutral_rfactor_fields(heatmap_sym_data)
    heatmap_sym_data = calculate_intraday_boost(heatmap_sym_data, None, None)
    momentum_sym_data = _apply_neutral_rfactor_fields(momentum_sym_data)
    momentum_sym_data = calculate_intraday_boost(momentum_sym_data, None, None)

    # STEP 6 — VWAP standard deviation bands (Feature 4)
    # Attaches vwap_position, band_1_upper/lower, band_2_upper/lower to each stock
    from vwap_bands import calculate_vwap_bands
    for stock in heatmap_sym_data.values():
        calculate_vwap_bands(stock)
    for stock in momentum_sym_data.values():
        calculate_vwap_bands(stock)

    # Build scanner flat list from SCANNER_STOCKS
    scanner_stocks_result: List[Dict[str, Any]] = []
    for sym in SCANNER_STOCKS:
        clean = sym.replace(".NS", "")
        if clean in heatmap_sym_data:
            scanner_stocks_result.append(heatmap_sym_data[clean])
    scanner_stocks_result.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    logger.info(f"Scanner stocks assembled: {len(scanner_stocks_result)}/{len(SCANNER_STOCKS)} symbols.")

    scanner_stocks_upstox_result: List[Dict[str, Any]] = []
    for sym in SCANNER_STOCKS:
        clean = sym.replace(".NS", "")
        if clean in momentum_sym_data:
            scanner_stocks_upstox_result.append(momentum_sym_data[clean])
    scanner_stocks_upstox_result.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    logger.info(
        "Momentum scanner stocks assembled: %d/%d symbols.",
        len(scanner_stocks_upstox_result),
        len(SCANNER_STOCKS),
    )

    # STEP 4 — Build sector results from sym_data (heatmap only)
    sectors_result: List[Dict[str, Any]] = []
    for sector_name, sector_symbols in SECTORS.items():
        stocks = []
        for sym in sector_symbols:
            clean = sym.replace(".NS", "")
            if clean in heatmap_sym_data:
                stocks.append(heatmap_sym_data[clean])

        if not stocks:
            continue

        stocks.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
        fallback_sector_pct = round(sum(s["change_pct"] for s in stocks) / len(stocks), 2)
        sector_pct = _get_sector_index_change_pct(sector_name, official_indices, fallback_sector_pct)
        sectors_result.append({
            "name": sector_name,
            "change_pct": sector_pct,
            "stocks": stocks,
        })

    sectors_result.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

    now_ist = datetime.now(IST)
    market_open = (
        now_ist.weekday() < 5
        and dt_time(9, 15) <= now_ist.time() <= dt_time(15, 30)
    )

    elapsed = round(time.time() - start, 1)
    logger.info(
        f"Fetch completed in {elapsed}s. "
        f"Last updated: {now_ist.strftime('%H:%M:%S')} IST. "
        f"Market open: {market_open}"
    )

    diagnostics = _build_heatmap_diagnostics(heatmap_sym_data)
    logger.info(
        "Heatmap diagnostics: %d symbols, %d zero-change, sources=%s",
        diagnostics["total_symbols"],
        diagnostics["zero_change_count"],
        diagnostics["quote_source_counts"],
    )

    return {
        "sectors": sectors_result,
        "scanner_stocks": scanner_stocks_result,
        "scanner_stocks_upstox": scanner_stocks_upstox_result,
        "last_updated": now_ist.strftime("%H:%M:%S"),
        "market_open": market_open,
        "diagnostics": diagnostics,
    }
