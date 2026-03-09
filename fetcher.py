# fetcher.py — Downloads and processes Indian stock market data via yfinance

import time
import logging
from statistics import mean
from datetime import datetime, time as dt_time
from typing import Any, Dict, List

import pandas as pd
import pytz
import yfinance as yf

from angel_client import get_bulk_ltp, get_bulk_full_quotes
from stocks import ALL_SYMBOLS, SECTORS, FO_STOCKS, SCANNER_STOCKS
from rfactor import calculate_rfactor_for_all
from nse_fetcher import fetch_nse_index_quotes
from intraday_boost import calculate_intraday_boost

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


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


def _compute_change_pct(ltp: float, prev_close: float, fallback_change_pct: float = 0.0) -> float:
    if prev_close > 0:
        return round(((ltp - prev_close) / prev_close) * 100, 2)
    return round(fallback_change_pct, 2)


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
    all_syms_str  = " ".join(_combined_symbols)
    clean_symbols = [s.replace(".NS", "") for s in _combined_symbols]

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

    # STEP 1 — SmartAPI FULL quotes (replaces NSE website scraping).
    # Returns per symbol: ltp, change_pct, prev_close, vwap, day_high, day_low,
    #                     total_traded_volume (lakhs), bid_ask_ratio, bid_qty, ask_qty
    logger.info(f"Fetching SmartAPI FULL quotes for {len(clean_symbols)} symbols...")
    nse_full: Dict[str, Any] = {}
    try:
        nse_full = get_bulk_full_quotes(clean_symbols)
        logger.info(f"SmartAPI quotes received for {len(nse_full)}/{len(clean_symbols)} symbols.")
    except Exception as e:
        logger.warning(f"SmartAPI full quote fetch failed: {e}")

    angel_ltp: Dict[str, float] = {}
    if not nse_full:
        logger.info(f"Falling back to Angel LTP for {len(clean_symbols)} symbols...")
        try:
            angel_ltp = get_bulk_ltp(clean_symbols)
            logger.info(f"Angel LTP received for {len(angel_ltp)}/{len(clean_symbols)} symbols.")
        except Exception as e:
            logger.warning(f"Angel LTP fetch failed: {e}")

    logger.info("Fetching official NSE sector indices...")
    official_indices: Dict[str, Dict[str, Any]] = {}
    try:
        official_indices = fetch_nse_index_quotes()
        logger.info("Official NSE indices received for %d entries.", len(official_indices))
    except Exception as e:
        logger.warning(f"NSE index quote fetch failed: {e}")

    # STEP 2 — Daily yfinance: 20-day volume average baseline (fast; 1d interval, 5d period)
    logger.info(f"Downloading daily data for volume baseline ({len(ALL_SYMBOLS)} symbols)...")
    daily_data = yf.download(
        tickers=all_syms_str,
        period="5d",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
        timeout=30,
    )

    # STEP 3 — 15-min yfinance: RSI / MFI candle series (no free NSE equivalent)
    logger.info(f"Downloading 15-min data for RSI/MFI ({len(ALL_SYMBOLS)} symbols)...")
    data_15min = yf.download(
        tickers=all_syms_str,
        period="5d",
        interval="15m",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
        timeout=30,
    )

    # STEP 4 — Build sym_data: primary values from SmartAPI, volume_ratio from yfinance daily
    sym_data: Dict[str, Any] = {}
    for symbol in _combined_symbols:
        try:
            clean = symbol.replace(".NS", "")
            q = nse_full.get(clean)
            angel_price = angel_ltp.get(clean)

            # Fall back to yfinance daily if NSE quote unavailable
            if not q or q.get("ltp", 0) <= 0:
                d_df = get_sym_df(daily_data, symbol)
                if d_df is None:
                    continue
                close_s = d_df["Close"].dropna()
                if len(close_s) < 2:
                    continue
                prev_close_fb = float(close_s.iloc[-2])
                ltp_fb        = round(float(angel_price or close_s.iloc[-1]), 2)
                change_fb     = _compute_change_pct(ltp_fb, prev_close_fb)
                sym_data[clean] = {
                    "symbol": clean, "ltp": ltp_fb, "change_pct": change_fb,
                    "volume_ratio": 1.0, "fo": clean in FO_STOCKS,
                    "day_high": ltp_fb, "day_low": ltp_fb, "day_open": ltp_fb, "vwap": 0.0,
                    "quote_source": "yfinance_fallback" if not angel_price else "angel_ltp_fallback",
                    "delivery_source": "unavailable_fallback",
                    "delivery_pct": None,
                    "bid_ask_ratio": None,
                    "bid_qty": None,
                    "ask_qty": None,
                }
                continue

            live_ltp = round(float(angel_price or q.get("ltp", 0) or 0), 2)
            prev_close = float(q.get("prev_close", 0) or 0)
            if prev_close <= 0:
                d_df = get_sym_df(daily_data, symbol)
                if d_df is not None:
                    close_s = d_df["Close"].dropna()
                    if len(close_s) >= 2:
                        prev_close = float(close_s.iloc[-2])
            change_pct = _compute_change_pct(live_ltp, prev_close, float(q.get("change_pct", 0) or 0))

            # Volume ratio: NSE total traded (lakhs → shares) vs 20-day yfinance avg
            vol_ratio = 1.0
            try:
                nse_vol_shares = q.get("total_traded_volume", 0.0) * 100_000
                d_df = get_sym_df(daily_data, symbol)
                if d_df is not None and nse_vol_shares > 0:
                    avg_20d = float(d_df["Volume"].dropna().tail(20).mean()) if len(d_df["Volume"].dropna()) >= 3 else 0.0
                    if avg_20d > 0:
                        vol_ratio = round(nse_vol_shares / avg_20d, 2)
            except Exception:
                pass

            sym_data[clean] = {
                "symbol":       clean,
                "ltp":          live_ltp,
                "change_pct":   change_pct,
                "volume_ratio": vol_ratio,
                "fo":           clean in FO_STOCKS,
                # Live intraday fields — used by rfactor price_action and boost
                "day_high":     q.get("day_high",  live_ltp or q["ltp"]),
                "day_low":      q.get("day_low",   live_ltp or q["ltp"]),
                "day_open":     q.get("day_open",  live_ltp or q["ltp"]),
                "vwap":         q.get("vwap",       0.0),
                # Delivery & bid-ask written here directly so they survive even if rfactor skips
                "quote_source":  q.get("quote_source", "smartapi_full"),
                "delivery_source": q.get("delivery_source", "unknown"),
                "delivery_pct":  round(float(q.get("delivery_pct", 0) or 0), 1) if q.get("delivery_pct") is not None else None,
                "bid_ask_ratio": round(float(q.get("bid_ask_ratio", 0) or 0), 2) if q.get("bid_qty") or q.get("ask_qty") else None,
                "bid_qty":       q.get("bid_qty") if q.get("bid_qty") or q.get("ask_qty") else None,
                "ask_qty":       q.get("ask_qty") if q.get("bid_qty") or q.get("ask_qty") else None,
            }

        except Exception as e:
            logger.warning(f"Skipping {symbol}: {e}")
            continue

    logger.info(f"sym_data built for {len(sym_data)} symbols (SmartAPI primary, yfinance fallback).")

    # STEP 5 — quote depth data for rfactor (delivery%, bid/ask) — sourced from live quotes
    nse_data: Dict[str, Any] = {
        sym: {
            "delivery_pct":  q.get("delivery_pct") or 0.0,
            "bid_ask_ratio": q.get("bid_ask_ratio") or 1.0,
            "bid_qty":       q.get("bid_qty") or 0,
            "ask_qty":       q.get("ask_qty") or 0,
        }
        for sym, q in nse_full.items()
    }
    if not nse_data:
        logger.warning("Live depth data empty — rfactor will use neutral delivery/bid values.")

    sym_data = calculate_rfactor_for_all(sym_data, None, data_15min, nse_data)
    sym_data = calculate_intraday_boost(sym_data, None, daily_data)

    # STEP 6 — VWAP standard deviation bands (Feature 4)
    # Attaches vwap_position, band_1_upper/lower, band_2_upper/lower to each stock
    from vwap_bands import calculate_vwap_bands
    for stock in sym_data.values():
        calculate_vwap_bands(stock)

    # Build scanner flat list from SCANNER_STOCKS
    scanner_stocks_result: List[Dict[str, Any]] = []
    for sym in SCANNER_STOCKS:
        clean = sym.replace(".NS", "")
        if clean in sym_data:
            scanner_stocks_result.append(sym_data[clean])
    scanner_stocks_result.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    logger.info(f"Scanner stocks assembled: {len(scanner_stocks_result)}/{len(SCANNER_STOCKS)} symbols.")

    # STEP 4 — Build sector results from sym_data (heatmap only)
    sectors_result: List[Dict[str, Any]] = []
    for sector_name, sector_symbols in SECTORS.items():
        stocks = []
        for sym in sector_symbols:
            clean = sym.replace(".NS", "")
            if clean in sym_data:
                stocks.append(sym_data[clean])

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

    return {
        "sectors": sectors_result,
        "scanner_stocks": scanner_stocks_result,
        "last_updated": now_ist.strftime("%H:%M:%S"),
        "market_open": market_open,
    }
