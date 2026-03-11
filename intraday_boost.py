# intraday_boost.py — Measures intraday acceleration / momentum burst for each stock

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict

logger = logging.getLogger("intraday_boost")



def _get_sym_df(data, symbol: str):
    """Extract a single-symbol DataFrame from a flat or MultiIndex yfinance result."""
    import os
    os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
    try:
        if data is None:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            if symbol in data.columns.get_level_values(0):
                df = data[symbol]
                return df if not df.empty else None
            return None
        return data
    except Exception:
        return None


def calculate_intraday_boost(
    sym_data: Dict[str, Any],
    intraday_data,
    daily_data,
) -> Dict[str, Any]:
    """
    Adds boost_score (0–5) to every stock in sym_data.

    Boost measures intraday ACCELERATION — how aggressively the stock is moving
    right now vs its normal behaviour. Both up and down moves score equally
    (direction-neutral), so a crashing stock can still have a high boost.

    Components
    ----------
    35%  Volume acceleration   — last-3-candle avg vol vs whole-day avg vol
    30%  Price velocity        — abs price change over last 6 candles
    20%  Daily range position  — how far price has moved within today's H-L range
    15%  Candle consistency    — fraction of last 6 candles moving in same direction

    Args:
        sym_data:      dict keyed by clean symbol
        intraday_data: yfinance 5-min multi-ticker dataframe
        daily_data:    yfinance daily multi-ticker dataframe (for daily range ref)

    Returns:
        sym_data with boost_score added in-place.
    """
    for clean_sym, stock in sym_data.items():
        try:
            symbol_ns = clean_sym + ".NS"
            df = _get_sym_df(intraday_data, symbol_ns)
            d_df = _get_sym_df(daily_data, symbol_ns)

            if df is None or df.empty or len(df) < 6:
                # ── NSE-based boost when no candle data available ──────────────
                try:
                    ltp_val    = float(stock.get("ltp",          0) or 0)
                    day_high   = float(stock.get("day_high", ltp_val) or ltp_val)
                    day_low    = float(stock.get("day_low",  ltp_val) or ltp_val)
                    vwap_nse   = float(stock.get("vwap",    ltp_val) or ltp_val)
                    change_pct = float(stock.get("change_pct",   0) or 0)
                    vol_ratio  = float(stock.get("volume_ratio",  1) or 1)

                    # Volume acceleration proxy: vol_ratio vs "normal" (1×)
                    vol_accel_score = min(1.0, vol_ratio / 3.0)

                    # Price velocity proxy: abs % change today
                    velocity_score = min(1.0, abs(change_pct) / 2.0)

                    # Daily range position: how far from centre of today's H-L range
                    day_range = day_high - day_low
                    if day_range > 0:
                        center = (day_high + day_low) / 2
                        range_score = min(1.0, abs(ltp_val - center) / (day_range / 2))
                    else:
                        range_score = 0.0

                    # Candle consistency proxy: ltp vs VWAP aligned with direction
                    if vwap_nse > 0 and ltp_val > 0:
                        above_vwap = ltp_val > vwap_nse
                        moving_up  = change_pct > 0
                        consistency_score = 0.8 if above_vwap == moving_up else 0.3
                    else:
                        consistency_score = 0.5

                    raw = (
                        vol_accel_score   * 0.35
                        + velocity_score  * 0.30
                        + range_score     * 0.20
                        + consistency_score * 0.15
                    )
                    stock["boost_score"] = round(raw * 5, 2)
                except Exception as nse_err:
                    logger.warning(f"NSE boost failed for {clean_sym}: {nse_err}")
                    stock["boost_score"] = 0.0
                continue

            close = df["Close"].dropna()
            volume = df["Volume"].dropna()

            # ── COMPONENT 1: Volume acceleration (35%) ──
            # Recent 3-candle avg vs whole-day avg
            recent_vol_avg = float(volume.tail(3).mean()) if len(volume) >= 3 else 0.0
            day_vol_avg = float(volume.mean()) if not volume.empty else 1.0
            vol_accel = recent_vol_avg / day_vol_avg if day_vol_avg > 0 else 1.0
            vol_accel_score = min(1.0, vol_accel / 3.0)  # 3× recent burst = full score

            # ── COMPONENT 2: Price velocity (30%) ──
            # Abs % change over last 6 candles
            price_6_ago = float(close.iloc[-6]) if len(close) >= 6 else float(close.iloc[0])
            price_now = float(close.iloc[-1])
            velocity_pct = abs(price_now - price_6_ago) / price_6_ago * 100 if price_6_ago != 0 else 0.0
            velocity_score = min(1.0, velocity_pct / 2.0)  # 2% move in 6 candles = full

            # ── COMPONENT 3: Daily range position (20%) ──
            # How far into today's high-low range has price moved
            day_high = float(df["High"].dropna().max()) if not df["High"].dropna().empty else price_now
            day_low = float(df["Low"].dropna().min()) if not df["Low"].dropna().empty else price_now
            day_range = day_high - day_low
            if day_range > 0:
                range_pos = abs(price_now - ((day_high + day_low) / 2)) / (day_range / 2)
                range_score = min(1.0, range_pos)  # at extreme of range = 1.0
            else:
                range_score = 0.0

            # ── COMPONENT 4: Candle consistency (15%) ──
            # Fraction of last 6 candles that moved in the same direction as the net move
            last6_close = close.tail(6).values
            if len(last6_close) >= 2:
                net_dir = 1 if last6_close[-1] >= last6_close[0] else -1
                consistent = sum(
                    1 for i in range(1, len(last6_close))
                    if (last6_close[i] - last6_close[i - 1]) * net_dir > 0
                )
                consistency_score = consistent / (len(last6_close) - 1)
            else:
                consistency_score = 0.0

            # ── FINAL BOOST SCORE (0–5) ──
            raw = (
                vol_accel_score    * 0.35
                + velocity_score   * 0.30
                + range_score      * 0.20
                + consistency_score * 0.15
            )
            stock["boost_score"] = round(raw * 5, 2)

        except Exception as e:
            logger.warning(f"boost_score failed for {clean_sym}: {e}")
            stock["boost_score"] = 0.0

    return sym_data
