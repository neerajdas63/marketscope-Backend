# oi_analysis.py — OI Analysis using NSE option chain API

import logging
import threading
import os
os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz
from upstox_client import get_nearest_option_expiry, get_option_chain as get_upstox_option_chain, get_underlying_snapshot, is_upstox_configured
from stocks import ACTIVE_FO_STOCK_SET
_IST = pytz.timezone("Asia/Kolkata")

logger = logging.getLogger("oi_analysis")

_WORKERS = 3          # Limited concurrency keeps OI refresh fast without fully flooding NSE.
_BUDGET_SECS = 120.0  # Sequential + delays: ~3s/symbol × 30 = 90s; give 120s budget
_REQUEST_DELAY = 0.4  # Limited concurrency + smaller delay reduces bulk OI latency.
_OC_EQUITIES_URL = "https://www.nseindia.com/api/option-chain-equities"
_OC_INDICES_URL  = "https://www.nseindia.com/api/option-chain-indices"

_INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

# Default symbols for bulk OI — indices first, then highest-OI F&O stocks.
# Keeping this list small (30-35) ensures results come back within the budget.
_DEFAULT_OI_SYMBOLS = [
    "NIFTY", "BANKNIFTY",
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN",
    "AXISBANK", "BAJFINANCE", "KOTAKBANK", "LT", "TATAMOTORS", "WIPRO",
    "SUNPHARMA", "ADANIPORTS", "HINDUNILVR", "ONGC", "NTPC", "POWERGRID",
    "BHARTIARTL", "HCLTECH", "TITAN", "MARUTI", "TATASTEEL",
    "BAJAJ-AUTO", "DRREDDY", "CIPLA", "DIVISLAB", "TECHM",
]
_DEFAULT_OI_SYMBOLS = _DEFAULT_OI_SYMBOLS[:2] + [
    symbol for symbol in _DEFAULT_OI_SYMBOLS[2:] if symbol in ACTIVE_FO_STOCK_SET
]
_FO_RADAR_MAX_SYMBOLS = max(10, int(os.getenv("FO_RADAR_MAX_SYMBOLS", "36")))
_FO_RADAR_MIN_REFRESH_SECONDS = max(60, int(os.getenv("FO_RADAR_MIN_REFRESH_SECONDS", "900")))


def _build_compact_chain_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for rec in records:
        strike = float(rec.get("strikePrice", 0) or 0)
        ce = rec.get("CE") or {}
        pe = rec.get("PE") or {}
        if strike <= 0:
            continue
        compact.append(
            {
                "strike": strike,
                "call_oi": int(ce.get("openInterest", 0) or 0),
                "call_prev_oi": int(ce.get("prevOpenInterest", ce.get("openInterest", 0)) or 0),
                "put_oi": int(pe.get("openInterest", 0) or 0),
                "put_prev_oi": int(pe.get("prevOpenInterest", pe.get("openInterest", 0)) or 0),
            }
        )
    return compact


def _build_compact_upstox_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in rows:
        strike = float(row.get("strike_price", 0) or 0)
        if strike <= 0:
            continue
        call_market = (row.get("call_options") or {}).get("market_data") or {}
        put_market = (row.get("put_options") or {}).get("market_data") or {}
        compact.append(
            {
                "strike": strike,
                "call_oi": int(call_market.get("oi", 0) or 0),
                "call_prev_oi": int(call_market.get("prev_oi", call_market.get("oi", 0)) or 0),
                "put_oi": int(put_market.get("oi", 0) or 0),
                "put_prev_oi": int(put_market.get("prev_oi", put_market.get("oi", 0)) or 0),
            }
        )
    return compact


def _compute_max_pain_from_compact_records(records: List[Dict[str, Any]]) -> float:
    strikes = sorted({float(rec.get("strike", 0) or 0) for rec in records if float(rec.get("strike", 0) or 0) > 0})
    if not strikes:
        return 0.0

    min_pain = float("inf")
    max_pain_strike = strikes[0]
    for test_strike in strikes:
        total_loss = 0.0
        for rec in records:
            strike = float(rec.get("strike", 0) or 0)
            if strike <= 0:
                continue
            total_loss += max(0.0, test_strike - strike) * int(rec.get("call_oi", 0) or 0)
            total_loss += max(0.0, strike - test_strike) * int(rec.get("put_oi", 0) or 0)
        if total_loss < min_pain:
            min_pain = total_loss
            max_pain_strike = test_strike

    return float(max_pain_strike)


def _compute_oi_analysis_from_compact_records(
    symbol: str,
    compact_records: List[Dict[str, Any]],
    underlying_price: float,
    price_change: float,
    oi_source: str,
) -> Dict[str, Any]:
    if not compact_records:
        return {}

    total_call_oi = 0
    total_put_oi = 0
    total_call_oi_prev = 0
    total_put_oi_prev = 0
    call_oi_by_strike: Dict[float, int] = {}
    put_oi_by_strike: Dict[float, int] = {}

    for rec in compact_records:
        strike = float(rec.get("strike", 0) or 0)
        call_oi = int(rec.get("call_oi", 0) or 0)
        call_prev_oi = int(rec.get("call_prev_oi", call_oi) or 0)
        put_oi = int(rec.get("put_oi", 0) or 0)
        put_prev_oi = int(rec.get("put_prev_oi", put_oi) or 0)

        total_call_oi += call_oi
        total_put_oi += put_oi
        total_call_oi_prev += call_prev_oi
        total_put_oi_prev += put_prev_oi

        if strike > 0:
            call_oi_by_strike[strike] = call_oi_by_strike.get(strike, 0) + call_oi
            put_oi_by_strike[strike] = put_oi_by_strike.get(strike, 0) + put_oi

    total_oi_now = total_call_oi + total_put_oi
    total_oi_prev = total_call_oi_prev + total_put_oi_prev
    oi_change_pct = round(((total_oi_now - total_oi_prev) / total_oi_prev) * 100, 2) if total_oi_prev > 0 else 0.0

    pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0.0
    if pcr > 1.2:
        pcr_signal = "OVERSOLD"
    elif pcr < 0.7:
        pcr_signal = "OVERBOUGHT"
    else:
        pcr_signal = "NEUTRAL"

    oi_increased = total_oi_now > total_oi_prev
    price_up = price_change > 0
    if price_up and oi_increased:
        oi_signal = "LONG_BUILDUP"
    elif not price_up and oi_increased:
        oi_signal = "SHORT_BUILDUP"
    elif price_up and not oi_increased:
        oi_signal = "SHORT_COVERING"
    else:
        oi_signal = "LONG_UNWINDING"

    support_strikes = sorted(put_oi_by_strike, key=lambda strike: put_oi_by_strike[strike], reverse=True)[:3]
    resistance_strikes = sorted(call_oi_by_strike, key=lambda strike: call_oi_by_strike[strike], reverse=True)[:3]
    max_pain = _compute_max_pain_from_compact_records(compact_records)

    return {
        "symbol": symbol.upper(),
        "price": round(underlying_price, 2),
        "current_price": round(underlying_price, 2),
        "oi_signal": oi_signal,
        "oi_change_pct": oi_change_pct,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "pcr": pcr,
        "pcr_signal": pcr_signal,
        "max_pain": max_pain,
        "support_strikes": sorted(support_strikes),
        "resistance_strikes": sorted(resistance_strikes),
        "oi_source": oi_source,
    }


def _get_upstox_oi_analysis(symbol: str) -> Dict[str, Any]:
    if not is_upstox_configured():
        return {}
    payload = get_upstox_option_chain(symbol)
    rows = list(payload.get("data") or [])
    if not rows:
        return {}

    compact_records = _build_compact_upstox_records(rows)
    if not compact_records:
        return {}

    underlying_price = float((rows[0] or {}).get("underlying_spot_price", 0) or 0)
    snapshot = get_underlying_snapshot(symbol)
    price_change = float(snapshot.get("net_change", 0) or 0)
    if underlying_price <= 0:
        underlying_price = float(snapshot.get("last_price", 0) or 0)

    result = _compute_oi_analysis_from_compact_records(
        symbol=symbol,
        compact_records=compact_records,
        underlying_price=underlying_price,
        price_change=price_change,
        oi_source="upstox_option_chain",
    )
    if result:
        result["expiry_date"] = str(payload.get("expiry_date") or get_nearest_option_expiry(symbol) or "")
    return result


def fetch_option_chain(symbol: str) -> Optional[Dict]:
    """
    Fetch raw option chain JSON from NSE.

    Key requirements from NSE's anti-bot checks:
      1. Session must have visited /option-chain page (sets OC cookies)
      2. Each request must carry Referer: .../option-chain
      3. sec-fetch-* headers must be present (set in _BASE_HEADERS)
      4. Requests must be sequential at ~1/s (parallel = instant 403)
      5. Do NOT reset session on 403 — it causes a rebuild storm that
         hammers NSE with cookie requests and worsens the block.
    """
    from nse_fetcher import _get_oi_session
    sym_upper = symbol.upper()
    url = _OC_INDICES_URL if sym_upper in _INDEX_SYMBOLS else _OC_EQUITIES_URL
    oc_headers = {
        "Referer": "https://www.nseindia.com/option-chain",
        "Accept":  "application/json, text/plain, */*",
    }
    try:
        time.sleep(_REQUEST_DELAY)
        sess = _get_oi_session()
        r = sess.get(url, params={"symbol": sym_upper}, headers=oc_headers, timeout=12)
        if r.status_code in (401, 403):
            logger.warning("OI fetch 403 for %s (rate-limited, NOT resetting session).", symbol)
            return None
        if r.status_code != 200:
            logger.warning("OI fetch failed for %s: HTTP %d", symbol, r.status_code)
            return None
        content_type = r.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            logger.warning(
                "OI fetch got non-JSON for %s (Content-Type: %s) — cookies may be stale.",
                symbol, content_type,
            )
            return None
        return r.json()
    except Exception as e:
        logger.warning("OI fetch exception for %s: %s", symbol, e)
        return None


def _compute_max_pain(records: List[Dict]) -> float:
    """Calculate max pain strike: the strike where total option writers' loss is minimum."""
    strikes: Dict[float, Dict[str, int]] = {}
    for rec in records:
        strike = float(rec.get("strikePrice", 0) or 0)
        ce_oi = int((rec.get("CE") or {}).get("openInterest", 0) or 0)
        pe_oi = int((rec.get("PE") or {}).get("openInterest", 0) or 0)
        if strike > 0:
            strikes[strike] = {"call_oi": ce_oi, "put_oi": pe_oi}

    all_strikes = sorted(strikes.keys())
    if not all_strikes:
        return 0.0

    min_pain = float("inf")
    max_pain_strike = all_strikes[0]
    for test_strike in all_strikes:
        total_loss = 0
        for s, oi in strikes.items():
            total_loss += max(0.0, test_strike - s) * oi["call_oi"]
            total_loss += max(0.0, s - test_strike) * oi["put_oi"]
        if total_loss < min_pain:
            min_pain = total_loss
            max_pain_strike = test_strike

    return float(max_pain_strike)


def _get_nse_oi_analysis(symbol: str) -> Dict[str, Any]:
    from nse_fetcher import _get_session

    raw = fetch_option_chain(symbol)
    if not raw:
        return {}

    records_wrapper = raw.get("records", {})
    records: List[Dict] = records_wrapper.get("data", [])
    underlying_price = float(records_wrapper.get("underlyingValue", 0) or 0)
    if not records:
        logger.warning("Empty option chain records for %s", symbol)
        return {}

    compact_records = _build_compact_chain_records(records)
    price_change = 0.0
    try:
        sym_upper = symbol.upper()
        if sym_upper not in _INDEX_SYMBOLS:
            sess = _get_session()
            r = sess.get(
                "https://www.nseindia.com/api/quote-equity",
                params={"symbol": sym_upper},
                timeout=5,
            )
            if r.status_code == 200:
                price_change = float(r.json().get("priceInfo", {}).get("change", 0) or 0)
        else:
            snapshot = get_underlying_snapshot(symbol)
            price_change = float(snapshot.get("net_change", 0) or 0)
            if underlying_price <= 0:
                underlying_price = float(snapshot.get("last_price", 0) or 0)
    except Exception:
        pass

    return _compute_oi_analysis_from_compact_records(
        symbol=symbol,
        compact_records=compact_records,
        underlying_price=underlying_price,
        price_change=price_change,
        oi_source="nse_option_chain",
    )


def get_oi_analysis(symbol: str) -> Dict[str, Any]:
    """
    Return full OI analysis dict for a given F&O stock or index symbol.
    Returns empty dict on any failure.

    Fields returned:
        symbol, price, oi_signal, oi_change_pct,
        total_call_oi, total_put_oi, pcr, pcr_signal,
        max_pain, support_strikes, resistance_strikes
    """
    try:
        if is_upstox_configured():
            upstox_result = _get_upstox_oi_analysis(symbol)
            if upstox_result:
                return upstox_result

        return _get_nse_oi_analysis(symbol)

    except Exception as e:
        logger.error("get_oi_analysis failed for %s: %s", symbol, e, exc_info=True)
        return {}


def get_bulk_oi(symbols: List[str]) -> Dict[str, Any]:
    """
    Fetch OI analysis for multiple symbols in parallel.
    Uses 8 workers with a 45-second total budget.
    """
    result: Dict[str, Any] = {}
    deadline = time.monotonic() + _BUDGET_SECS

    with ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="oi-bulk") as pool:
        futures = {pool.submit(get_oi_analysis, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            if time.monotonic() > deadline:
                logger.warning("OI bulk fetch: budget exceeded, stopping early.")
                break
            try:
                data = fut.result(timeout=15)  # must be > _REQUEST_DELAY (1.2s) + request time
                if data:
                    result[sym] = data
            except Exception as e:
                logger.warning("OI bulk: %s failed: %s", sym, e)

    logger.info("OI bulk complete — %d/%d symbols fetched.", len(result), len(symbols))
    return result


# ── F&O Trade Radar ────────────────────────────────────────────────────────────

# Background cache populated by refresh_fo_radar_cache() after each scheduled fetch.
_fo_radar_cache: List[Dict] = []
_fo_radar_cache_at: float   = 0.0
_fo_radar_lock = threading.Lock()


def _select_fo_radar_symbols(fo_symbols: List[str], stock_map: Dict[str, Dict]) -> List[str]:
    available_symbols = [symbol for symbol in fo_symbols if symbol in stock_map]
    if not available_symbols:
        return []

    selected: List[str] = []
    seen: set[str] = set()

    for symbol in _DEFAULT_OI_SYMBOLS:
        clean_symbol = str(symbol or "").strip().upper().replace(".NS", "")
        if clean_symbol in stock_map and clean_symbol not in seen:
            selected.append(clean_symbol)
            seen.add(clean_symbol)
            if len(selected) >= _FO_RADAR_MAX_SYMBOLS:
                return selected

    ranked_candidates = sorted(
        available_symbols,
        key=lambda sym: (
            abs(float(stock_map[sym].get("change_pct", 0) or 0)),
            float(stock_map[sym].get("volume_ratio", 0) or 0),
            abs(float(stock_map[sym].get("boost_score", 0) or 0)),
        ),
        reverse=True,
    )
    for symbol in ranked_candidates:
        if symbol in seen:
            continue
        selected.append(symbol)
        seen.add(symbol)
        if len(selected) >= _FO_RADAR_MAX_SYMBOLS:
            break

    return selected


def compute_fo_trade_signal(oi_data: dict, stock_data: dict) -> dict:
    """
    Merge OI + price-action fields and compute a BUY / SELL / AVOID trade signal.

    Parameters
    ----------
    oi_data   : result of get_oi_analysis() — may be {} if OI unavailable
    stock_data: enriched stock dict from the main cache sectors (has rsi, vwap, etc.)

    Returns a fully merged dict ready to be stored in _fo_radar_cache.
    """
    # ── Price-action fields ───────────────────────────────────────────────────
    ltp          = float(stock_data.get("ltp",          0) or 0)
    change_pct   = float(stock_data.get("change_pct",   0) or 0)
    rsi          = float(stock_data.get("rsi",          50) or 50)
    vwap         = float(stock_data.get("vwap",         0) or 0)
    delivery_pct = float(stock_data.get("delivery_pct", 0) or 0)
    bid_ask      = float(stock_data.get("bid_ask_ratio", 1) or 1)
    rfactor      = float(stock_data.get("rfactor",      0) or 0)
    volume_ratio = float(stock_data.get("volume_ratio", 1) or 1)

    # ── OI fields (default to neutral when unavailable) ───────────────────────
    has_oi       = bool(oi_data)
    oi_signal    = oi_data.get("oi_signal",    "NO_DATA") if has_oi else "NO_DATA"
    oi_change_pct= float(oi_data.get("oi_change_pct", 0) or 0) if has_oi else 0.0
    pcr          = oi_data.get("pcr") if has_oi else None
    pcr_val      = float(pcr) if pcr is not None else 1.0
    pcr_signal   = oi_data.get("pcr_signal", "NEUTRAL") if has_oi else "NEUTRAL"
    max_pain     = oi_data.get("max_pain") if has_oi else None
    support_strikes     = oi_data.get("support_strikes",     []) if has_oi else []
    resistance_strikes  = oi_data.get("resistance_strikes",  []) if has_oi else []

    # Distance from max pain (%)
    dist_from_max_pain: Optional[float] = None
    if max_pain and ltp > 0:
        dist_from_max_pain = round((ltp - max_pain) / max_pain * 100, 2)

    above_vwap = (ltp > vwap) if vwap > 0 else False

    # ── BUY scoring ───────────────────────────────────────────────────────────
    buy_pts: int = 0
    buy_reasons: List[str] = []

    if oi_signal == "LONG_BUILDUP":
        buy_pts += 1; buy_reasons.append("OI rising with price (Long Buildup)")
    if oi_signal == "SHORT_COVERING":
        buy_pts += 1; buy_reasons.append("Short sellers covering (bullish)")
    if pcr_val > 1.1:
        buy_pts += 1; buy_reasons.append(f"PCR bullish ({pcr_val:.2f})")
    if vwap > 0 and ltp > vwap:
        buy_pts += 1; buy_reasons.append("Price above VWAP")
    if 50 <= rsi <= 70:
        buy_pts += 1; buy_reasons.append(f"RSI {rsi:.1f} — momentum zone")
    if delivery_pct >= 40:
        buy_pts += 1; buy_reasons.append(f"Delivery {delivery_pct:.1f}% — institutional interest")
    if bid_ask >= 1.2:
        buy_pts += 1; buy_reasons.append(f"Bid/Ask {bid_ask:.2f} — buy pressure")
    if dist_from_max_pain is not None and dist_from_max_pain > -2 and ltp < (max_pain or 0):
        buy_pts += 1; buy_reasons.append("Below max pain — gravity pull up")

    # ── SELL scoring ──────────────────────────────────────────────────────────
    sell_pts: int = 0
    sell_reasons: List[str] = []

    if oi_signal == "SHORT_BUILDUP":
        sell_pts += 1; sell_reasons.append("OI rising with price falling (Short Buildup)")
    if oi_signal == "LONG_UNWINDING":
        sell_pts += 1; sell_reasons.append("Long positions being unwound (bearish)")
    if pcr_val < 0.8:
        sell_pts += 1; sell_reasons.append(f"PCR bearish ({pcr_val:.2f})")
    if vwap > 0 and ltp < vwap:
        sell_pts += 1; sell_reasons.append("Price below VWAP")
    if rsi > 70:
        sell_pts += 1; sell_reasons.append(f"RSI {rsi:.1f} — overbought, reversal risk")
    if rsi < 30:
        sell_pts += 1; sell_reasons.append(f"RSI {rsi:.1f} — oversold, downtrend")
    if bid_ask < 0.8:
        sell_pts += 1; sell_reasons.append(f"Bid/Ask {bid_ask:.2f} — sell pressure")
    if dist_from_max_pain is not None and ltp > (max_pain or 0) and dist_from_max_pain > 2:
        sell_pts += 1; sell_reasons.append("Above max pain — gravity pull down")
    # ── Price-action SELL signals (work even without OI data) ─────────────────
    if change_pct < -2.5 and volume_ratio > 1.5:
        sell_pts += 2; sell_reasons.append(f"High-volume selloff {change_pct:.1f}% — distribution signal")
    elif change_pct < -1.5 and volume_ratio > 1.3:
        sell_pts += 1; sell_reasons.append(f"Selling with elevated volume ({change_pct:.1f}%)")
    if 30 <= rsi < 43:
        sell_pts += 1; sell_reasons.append(f"RSI {rsi:.1f} — weak zone, bearish bias")
    if rfactor < 0:
        sell_pts += 1; sell_reasons.append(f"R-Factor {rfactor:.2f} — bearish momentum")

    # ── Decision ─────────────────────────────────────────────────────────────
    diff = abs(buy_pts - sell_pts)
    if buy_pts == sell_pts:
        trade_signal = "AVOID"
        reasons      = buy_reasons + sell_reasons
    elif buy_pts > sell_pts:
        trade_signal = "BUY"
        reasons      = buy_reasons
    else:
        trade_signal = "SELL"
        reasons      = sell_reasons

    # Confidence based on margin of victory
    if diff >= 4:
        confidence = 3
    elif diff >= 2:
        confidence = 2
    else:
        confidence = 1

    return {
        "symbol":                stock_data.get("symbol", ""),
        "ltp":                   round(ltp, 2),
        "change_pct":            round(change_pct, 2),
        "sector":                stock_data.get("sector", ""),
        # OI fields
        "pcr":                   round(pcr_val, 2) if has_oi else None,
        "pcr_signal":            pcr_signal,
        "oi_signal":             oi_signal,
        "oi_change_pct":         round(oi_change_pct, 2),
        "max_pain":              round(max_pain, 2) if max_pain else None,
        "dist_from_max_pain_pct": dist_from_max_pain,
        "support_strikes":       support_strikes,
        "resistance_strikes":    resistance_strikes,
        # Price action fields
        "rfactor":               round(rfactor, 2),
        "rsi":                   round(rsi, 1),
        "vwap":                  round(vwap, 2),
        "above_vwap":            above_vwap,
        "delivery_pct":          round(delivery_pct, 1) if delivery_pct else None,
        "bid_ask_ratio":         round(bid_ask, 2) if bid_ask != 1.0 else None,
        "volume_ratio":          round(volume_ratio, 2),
        # Trade decision
        "trade_signal":          trade_signal,
        "confidence":            confidence,
        "reasons":               reasons,
    }


def refresh_fo_radar_cache(fo_symbols: List[str], price_data: List[Dict], force: bool = False) -> None:
    """
    Background job: fetch OI for all F&O symbols sequentially, merge with cache,
    compute trade signals, and store result in _fo_radar_cache.

    Called after each scheduled fetch completes so the endpoint is always fast.

    Parameters
    ----------
    fo_symbols : list of clean symbol strings from FO_STOCKS (no .NS suffix)
    price_data : flat list of stock dicts (scanner_stocks) — each dict has
                 symbol, ltp, change_pct, rsi, vwap, etc.
    """
    global _fo_radar_cache, _fo_radar_cache_at

    # Build a flat symbol → stock dict for quick lookup.
    # Accepts both flat stock lists and legacy sector-grouped lists.
    stock_map: Dict[str, Dict] = {}
    for item in price_data:
        if "stocks" in item:
            # Legacy sector-grouped format
            for stock in item.get("stocks", []):
                sym = stock.get("symbol", "")
                if sym:
                    stock_map[sym] = {**stock, "sector": item.get("name", "")}
        else:
            # Flat stock dict (scanner_stocks format)
            sym = item.get("symbol", "")
            if sym:
                stock_map[sym] = item

    with _fo_radar_lock:
        cache_age = time.monotonic() - _fo_radar_cache_at if _fo_radar_cache_at > 0 else float("inf")
    if not force and cache_age < _FO_RADAR_MIN_REFRESH_SECONDS:
        logger.info(
            "F&O Radar refresh skipped — cache age %.1fs is below minimum refresh interval %ss.",
            cache_age,
            _FO_RADAR_MIN_REFRESH_SECONDS,
        )
        return

    # Only process a capped high-signal subset that exists in the cache (has price data)
    symbols_to_fetch = _select_fo_radar_symbols(fo_symbols, stock_map)
    logger.info(
        "F&O Radar refresh: fetching OI for %d symbols sequentially%s…",
        len(symbols_to_fetch),
        " (forced)" if force else "",
    )

    results: List[Dict] = []
    for sym in symbols_to_fetch:
        try:
            oi_data   = get_oi_analysis(sym)        # includes _REQUEST_DELAY sleep
            entry     = compute_fo_trade_signal(oi_data or {}, stock_map[sym])
            results.append(entry)
        except Exception as exc:
            logger.warning("F&O Radar: skipping %s — %s", sym, exc)
            # Still include with price-action-only signal
            try:
                entry = compute_fo_trade_signal({}, stock_map[sym])
                results.append(entry)
            except Exception:
                pass

    # Sort by confidence desc, then by abs(change_pct) desc
    results.sort(key=lambda x: (x["confidence"], abs(x.get("change_pct", 0))), reverse=True)

    with _fo_radar_lock:
        _fo_radar_cache    = results
        _fo_radar_cache_at = time.monotonic()

    logger.info(
        "F&O Radar cache refreshed — %d stocks (%d BUY / %d SELL / %d AVOID).",
        len(results),
        sum(1 for r in results if r["trade_signal"] == "BUY"),
        sum(1 for r in results if r["trade_signal"] == "SELL"),
        sum(1 for r in results if r["trade_signal"] == "AVOID"),
    )


def get_fo_radar_snapshot() -> List[Dict]:
    """Return a copy of the current F&O Radar cache (thread-safe)."""
    with _fo_radar_lock:
        return list(_fo_radar_cache)


def fo_radar_cache_age_seconds() -> float:
    """Return how many seconds ago the F&O Radar cache was last refreshed."""
    if _fo_radar_cache_at == 0.0:
        return float("inf")
    return time.monotonic() - _fo_radar_cache_at


def get_cached_oi_signals() -> Dict[str, Dict[str, Any]]:
    """Return symbol -> {oi_signal, oi_change_pct} from cached F&O radar data (no API calls)."""
    with _fo_radar_lock:
        signals: Dict[str, Dict[str, Any]] = {}
        for entry in _fo_radar_cache:
            sym = str(entry.get("symbol", "")).upper()
            if sym:
                signals[sym] = {
                    "oi_signal": entry.get("oi_signal", ""),
                    "oi_change_pct": float(entry.get("oi_change_pct", 0) or 0),
                }
        return signals
