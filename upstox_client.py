import gzip
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dt_time, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

_API_BASE_URL = os.getenv("UPSTOX_API_BASE_URL", "https://api.upstox.com")
_INSTRUMENT_URL = os.getenv(
    "UPSTOX_INSTRUMENT_URL",
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
)
_INSTRUMENT_CACHE_TTL_SECONDS = int(os.getenv("UPSTOX_INSTRUMENT_CACHE_TTL_SECONDS", "43200"))
_QUOTE_CACHE_TTL_SECONDS = float(os.getenv("UPSTOX_QUOTE_CACHE_TTL_SECONDS", "2.0"))
_CHUNK_SIZE = int(os.getenv("UPSTOX_BATCH_CHUNK_SIZE", "400"))
_TIMEOUT_SECONDS = float(os.getenv("UPSTOX_TIMEOUT_SECONDS", "15"))
_HISTORY_WORKERS = int(os.getenv("UPSTOX_HISTORY_WORKERS", "8"))
_DAILY_HISTORY_WORKERS = int(os.getenv("UPSTOX_DAILY_HISTORY_WORKERS", "2"))
_HISTORICAL_CACHE_TTL_SECONDS = float(os.getenv("UPSTOX_HISTORICAL_CACHE_TTL_SECONDS", "21600"))
_INTRADAY_CACHE_TTL_SECONDS = float(os.getenv("UPSTOX_INTRADAY_CACHE_TTL_SECONDS", "120"))

_client_lock = threading.Lock()
_client: Optional[httpx.Client] = None
_client_token = ""

_instrument_lock = threading.Lock()
_instrument_cache: Dict[str, Dict[str, str]] = {}
_instrument_cache_at = 0.0

_quote_lock = threading.Lock()
_quote_cache: Dict[tuple[str, str], tuple[float, Dict[str, Any]]] = {}

_search_lock = threading.Lock()
_search_cache: Dict[tuple[str, str, str], tuple[float, List[Dict[str, Any]]]] = {}

_option_lock = threading.Lock()
_option_expiry_cache: Dict[str, tuple[float, List[str]]] = {}

_history_lock = threading.Lock()
_history_cache: Dict[tuple[str, str, str, int, str], tuple[float, pd.DataFrame]] = {}

_config_warning_lock = threading.Lock()
_missing_token_warned = False


def _warn_missing_token_once() -> None:
    global _missing_token_warned
    with _config_warning_lock:
        if _missing_token_warned:
            return
        logger.warning("Upstox disabled: missing environment variable UPSTOX_ACCESS_TOKEN. Using fallback providers.")
        _missing_token_warned = True


def is_upstox_configured() -> bool:
    return bool(str(os.getenv("UPSTOX_ACCESS_TOKEN", "") or "").strip())


def _require_token() -> str:
    token = str(os.getenv("UPSTOX_ACCESS_TOKEN", "") or "").strip()
    if not token:
        raise RuntimeError("Missing required environment variable: UPSTOX_ACCESS_TOKEN")
    return token


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace(".NS", "").strip()


def _is_stale(timestamp: float, ttl_seconds: float) -> bool:
    return timestamp <= 0 or (time.time() - timestamp) >= ttl_seconds


def _chunked(items: List[str], chunk_size: int) -> Iterable[List[str]]:
    for index in range(0, len(items), chunk_size):
        yield items[index:index + chunk_size]


def _is_market_open_now() -> bool:
    now = datetime.now().astimezone().astimezone(tz=None)
    try:
        now = pd.Timestamp.now(tz="Asia/Kolkata").to_pydatetime()
    except Exception:
        pass
    if now.weekday() >= 5:
        return False
    return dt_time(9, 15) <= now.time() <= dt_time(15, 30)


def _get_client() -> httpx.Client:
    global _client, _client_token

    token = _require_token()
    with _client_lock:
        if _client is not None and _client_token == token:
            return _client

        if _client is not None:
            _client.close()

        _client = httpx.Client(
            base_url=_API_BASE_URL,
            timeout=_TIMEOUT_SECONDS,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )
        _client_token = token
        return _client


def _fetch_json(url: str) -> Any:
    response = httpx.get(url, timeout=max(_TIMEOUT_SECONDS, 20.0), follow_redirects=True)
    response.raise_for_status()
    content = response.content
    if url.endswith(".gz"):
        content = gzip.decompress(content)
    return json.loads(content.decode("utf-8"))


def _load_instrument_cache(force_refresh: bool = False) -> Dict[str, Dict[str, str]]:
    global _instrument_cache, _instrument_cache_at

    with _instrument_lock:
        if not force_refresh and _instrument_cache and not _is_stale(_instrument_cache_at, _INSTRUMENT_CACHE_TTL_SECONDS):
            return _instrument_cache

        raw = _fetch_json(_INSTRUMENT_URL)
        selected: Dict[str, Dict[str, str]] = {}
        for record in raw:
            if str(record.get("segment", "")).upper() != "NSE_EQ":
                continue
            if str(record.get("instrument_type", "")).upper() != "EQ":
                continue
            symbol = _normalize_symbol(record.get("trading_symbol", ""))
            instrument_key = str(record.get("instrument_key", "") or "").strip()
            if not symbol or not instrument_key:
                continue
            selected[symbol] = {
                "instrument_key": instrument_key,
                "trading_symbol": str(record.get("trading_symbol", "") or "").strip(),
                "exchange_token": str(record.get("exchange_token", "") or "").strip(),
            }

        _instrument_cache = selected
        _instrument_cache_at = time.time()
        logger.info("Upstox instrument cache loaded for %d NSE_EQ symbols.", len(selected))
        return _instrument_cache


def get_symbol_instrument_keys(symbols: Iterable[str]) -> Dict[str, str]:
    if not is_upstox_configured():
        _warn_missing_token_once()
        return {}
    instrument_cache = _load_instrument_cache()
    result: Dict[str, str] = {}
    for symbol in symbols:
        normalized = _normalize_symbol(symbol)
        record = instrument_cache.get(normalized)
        if record:
            result[normalized] = record["instrument_key"]
    return result


def _request_json(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = _get_client().get(path, params=params)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"Unexpected Upstox response for {path}: {type(payload).__name__}")
            status = str(payload.get("status") or "").lower()
            if status and status != "success":
                raise RuntimeError(f"Upstox returned non-success payload for {path}: {payload}")
            return payload
        except httpx.HTTPStatusError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else 0
            if status_code not in {429, 500, 502, 503, 504} or attempt >= 2:
                raise
            time.sleep(0.8 * (attempt + 1))
        except Exception as exc:
            last_error = exc
            if attempt >= 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Upstox request failed for {path}")


def _normalize_history_date_range(from_date: str, to_date: str) -> tuple[str, str]:
    start = pd.Timestamp(from_date).date()
    end = pd.Timestamp(to_date).date()
    while end.weekday() >= 5:
        end -= timedelta(days=1)
    while start.weekday() >= 5:
        start += timedelta(days=1)
    if start > end:
        start = end
    return start.isoformat(), end.isoformat()


def _previous_trading_day(day_value) -> Any:
    current = pd.Timestamp(day_value).date() - timedelta(days=1)
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def _normalize_minute_history_window(from_date: str, to_date: str) -> tuple[str, str]:
    start_str, end_str = _normalize_history_date_range(from_date, to_date)
    start = pd.Timestamp(start_str).date()
    end = pd.Timestamp(end_str).date()
    today_ist = pd.Timestamp.now(tz="Asia/Kolkata").date()

    # Upstox minute-history rejects current-session dates during live market hours.
    # We fetch today's candles separately from the intraday endpoint, so historical
    # requests should stop at the previous trading day while the market is open.
    if end >= today_ist and _is_market_open_now():
        end = _previous_trading_day(today_ist)
        if start > end:
            start = end

    return start.isoformat(), end.isoformat()


def _search_instruments(query: str, exchanges: str, segments: str, records: int = 10) -> List[Dict[str, Any]]:
    if not is_upstox_configured():
        _warn_missing_token_once()
        return []
    cache_key = (str(query or "").strip().upper(), exchanges.upper(), segments.upper())
    with _search_lock:
        cached = _search_cache.get(cache_key)
        if cached and not _is_stale(cached[0], 300.0):
            return list(cached[1])

    payload = _request_json(
        "/v2/instruments/search",
        {
            "query": query,
            "exchanges": exchanges,
            "segments": segments,
            "page_number": 1,
            "records": min(max(int(records), 1), 30),
        },
    )
    rows = list(payload.get("data") or [])
    with _search_lock:
        _search_cache[cache_key] = (time.time(), rows)
    return rows


def _get_cached_result(cache_prefix: str, symbol: str) -> Optional[Dict[str, Any]]:
    with _quote_lock:
        cached = _quote_cache.get((cache_prefix, _normalize_symbol(symbol)))
        if not cached:
            return None
        cached_at, payload = cached
        if _is_stale(cached_at, _QUOTE_CACHE_TTL_SECONDS):
            _quote_cache.pop((cache_prefix, _normalize_symbol(symbol)), None)
            return None
        return dict(payload)


def _set_cached_result(cache_prefix: str, symbol: str, payload: Dict[str, Any]) -> None:
    with _quote_lock:
        _quote_cache[(cache_prefix, _normalize_symbol(symbol))] = (time.time(), dict(payload))


def _get_cached_history(cache_key: tuple[str, str, str, int, str], ttl_seconds: float) -> Optional[pd.DataFrame]:
    with _history_lock:
        cached = _history_cache.get(cache_key)
        if not cached:
            return None
        cached_at, frame = cached
        if _is_stale(cached_at, ttl_seconds):
            _history_cache.pop(cache_key, None)
            return None
        return frame.copy()


def _set_cached_history(cache_key: tuple[str, str, str, int, str], frame: pd.DataFrame) -> None:
    with _history_lock:
        _history_cache[cache_key] = (time.time(), frame.copy())


def _candles_to_frame(candles: List[List[Any]]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    rows: List[Dict[str, Any]] = []
    for candle in candles:
        if not isinstance(candle, list) or len(candle) < 6:
            continue
        rows.append(
            {
                "timestamp": candle[0],
                "Open": float(candle[1] or 0),
                "High": float(candle[2] or 0),
                "Low": float(candle[3] or 0),
                "Close": float(candle[4] or 0),
                "Volume": float(candle[5] or 0),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    frame = frame.set_index("timestamp").sort_index()
    return frame[["Open", "High", "Low", "Close", "Volume"]]


def _merge_candles(*candle_lists: List[List[Any]]) -> List[List[Any]]:
    merged: Dict[str, List[Any]] = {}
    for candle_list in candle_lists:
        for candle in candle_list or []:
            if isinstance(candle, list) and candle:
                merged[str(candle[0])] = candle
    return list(merged.values())


def _fetch_symbol_history_frame(symbol: str, from_date: str, to_date: str, interval_minutes: int = 5) -> pd.DataFrame:
    normalized = _normalize_symbol(symbol)
    instrument_key = get_underlying_instrument_key(normalized)
    if not instrument_key:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    history_from_date, history_to_date = _normalize_minute_history_window(from_date, to_date)

    historical_key = (normalized, history_from_date, history_to_date, interval_minutes, "historical")
    historical_frame = _get_cached_history(historical_key, _HISTORICAL_CACHE_TTL_SECONDS)
    if historical_frame is None:
        payload = _request_json(
            f"/v3/historical-candle/{instrument_key}/minutes/{interval_minutes}/{history_to_date}/{history_from_date}",
            {},
        )
        historical_candles = list((payload.get("data") or {}).get("candles") or [])
        historical_frame = _candles_to_frame(historical_candles)
        _set_cached_history(historical_key, historical_frame)

    intraday_frame = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if _is_market_open_now():
        intraday_key = (normalized, from_date, to_date, interval_minutes, "intraday")
        cached_intraday = _get_cached_history(intraday_key, _INTRADAY_CACHE_TTL_SECONDS)
        if cached_intraday is None:
            try:
                payload = _request_json(
                    f"/v3/historical-candle/intraday/{instrument_key}/minutes/{interval_minutes}",
                    {},
                )
                intraday_candles = list((payload.get("data") or {}).get("candles") or [])
                cached_intraday = _candles_to_frame(intraday_candles)
                _set_cached_history(intraday_key, cached_intraday)
            except Exception:
                cached_intraday = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        intraday_frame = cached_intraday

    combined = pd.concat([historical_frame, intraday_frame], axis=0) if not historical_frame.empty or not intraday_frame.empty else pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if combined.empty:
        return combined
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def get_intraday_history_batch(symbols: Sequence[str], from_date: str, to_date: str, interval_minutes: int = 5) -> Optional[pd.DataFrame]:
    if not is_upstox_configured():
        _warn_missing_token_once()
        return None
    labels = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
    if not labels:
        return None

    frames: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max(1, _HISTORY_WORKERS), thread_name_prefix="upstox-history") as pool:
        futures = {
            pool.submit(_fetch_symbol_history_frame, label, from_date, to_date, interval_minutes): label
            for label in labels
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                frame = future.result()
                if frame is not None and not frame.empty:
                    frames[label] = frame
            except Exception as exc:
                logger.warning("Upstox historical fetch failed for %s: %s", label, exc)

    if not frames:
        return None
    return pd.concat(frames, axis=1)


def _fetch_symbol_daily_history_frame(symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
    normalized = _normalize_symbol(symbol)
    instrument_key = get_underlying_instrument_key(normalized)
    if not instrument_key:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    from_date, to_date = _normalize_history_date_range(from_date, to_date)

    daily_key = (normalized, from_date, to_date, 1, "daily")
    daily_frame = _get_cached_history(daily_key, _HISTORICAL_CACHE_TTL_SECONDS)
    if daily_frame is None:
        payload = _request_json(
            f"/v2/historical-candle/{instrument_key}/day/{to_date}/{from_date}",
            {},
        )
        daily_candles = list((payload.get("data") or {}).get("candles") or [])
        daily_frame = _candles_to_frame(daily_candles)
        _set_cached_history(daily_key, daily_frame)
    return daily_frame


def get_daily_history_batch(symbols: Sequence[str], from_date: str, to_date: str) -> Optional[pd.DataFrame]:
    if not is_upstox_configured():
        _warn_missing_token_once()
        return None
    labels = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
    if not labels:
        return None

    frames: Dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max(1, _DAILY_HISTORY_WORKERS), thread_name_prefix="upstox-daily") as pool:
        futures = {
            pool.submit(_fetch_symbol_daily_history_frame, label, from_date, to_date): label
            for label in labels
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                frame = future.result()
                if frame is not None and not frame.empty:
                    frames[label] = frame
            except Exception as exc:
                logger.warning("Upstox daily history fetch failed for %s: %s", label, exc)

    if not frames:
        return None
    return pd.concat(frames, axis=1)


def get_bulk_full_quotes(symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    if not is_upstox_configured():
        _warn_missing_token_once()
        return {}
    normalized_symbols = [_normalize_symbol(symbol) for symbol in symbols if _normalize_symbol(symbol)]
    result: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for symbol in normalized_symbols:
        cached = _get_cached_result("full", symbol)
        if cached is not None:
            result[symbol] = cached
        else:
            missing.append(symbol)

    if not missing:
        return result

    symbol_to_key = get_symbol_instrument_keys(missing)
    key_to_symbol = {instrument_key: symbol for symbol, instrument_key in symbol_to_key.items()}
    key_to_symbol.update({symbol: symbol for symbol in missing})

    for chunk_symbols in _chunked(list(symbol_to_key.keys()), _CHUNK_SIZE):
        instrument_keys = [symbol_to_key[symbol] for symbol in chunk_symbols]
        payload = _request_json("/v2/market-quote/quotes", {"instrument_key": ",".join(instrument_keys)})
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            continue
        for row in data.values():
            if not isinstance(row, dict):
                continue
            instrument_token = str(row.get("instrument_token") or row.get("instrument_key") or "").strip()
            symbol = key_to_symbol.get(instrument_token) or _normalize_symbol(row.get("symbol", ""))
            if not symbol:
                continue

            ohlc = row.get("ohlc") or {}
            last_price = float(row.get("last_price", 0) or 0)
            raw_net_change = float(row.get("net_change", 0) or 0)
            prev_close_from_net = (last_price - raw_net_change) if last_price > 0 else 0.0
            ohlc_close = float(ohlc.get("close", 0) or 0)
            prev_close = prev_close_from_net if prev_close_from_net > 0 else ohlc_close
            if prev_close <= 0 and ohlc_close > 0:
                prev_close = ohlc_close
            total_buy = int(row.get("total_buy_quantity", 0) or 0)
            total_sell = int(row.get("total_sell_quantity", 0) or 0)
            volume = float(row.get("volume", 0) or 0)
            normalized_row = {
                "ltp": round(last_price, 2),
                "change_pct": round(((last_price - prev_close) / prev_close) * 100, 2) if prev_close > 0 else 0.0,
                "net_change": round(raw_net_change, 2),
                "prev_close": round(prev_close, 2),
                "day_open": round(float(ohlc.get("open", 0) or 0), 2),
                "day_high": round(float(ohlc.get("high", 0) or 0), 2),
                "day_low": round(float(ohlc.get("low", 0) or 0), 2),
                "vwap": round(float(row.get("average_price", 0) or 0), 2),
                "total_traded_volume": round(volume / 100_000, 2),
                "bid_qty": total_buy,
                "ask_qty": total_sell,
                "bid_ask_ratio": round(total_buy / total_sell, 2) if total_sell > 0 else 1.0,
                "delivery_pct": None,
                "quote_source": "upstox_full",
                "delivery_source": "unavailable_upstox",
            }
            result[symbol] = normalized_row
            _set_cached_result("full", symbol, normalized_row)

    return result


def get_underlying_instrument_key(symbol: str) -> Optional[str]:
    if not is_upstox_configured():
        _warn_missing_token_once()
        return None
    normalized = _normalize_symbol(symbol)
    if not normalized:
        return None

    instrument_cache = _load_instrument_cache()
    record = instrument_cache.get(normalized)
    if record:
        return record.get("instrument_key")

    rows = _search_instruments(normalized, exchanges="NSE", segments="INDEX", records=10)
    if rows:
        for row in rows:
            trading_symbol = _normalize_symbol(row.get("trading_symbol", ""))
            if trading_symbol == normalized:
                return str(row.get("instrument_key", "") or "").strip() or None
        instrument_key = str(rows[0].get("instrument_key", "") or "").strip()
        if instrument_key:
            return instrument_key

    rows = _search_instruments(normalized, exchanges="NSE", segments="EQ", records=10)
    if rows:
        for row in rows:
            trading_symbol = _normalize_symbol(row.get("trading_symbol", ""))
            if trading_symbol == normalized:
                return str(row.get("instrument_key", "") or "").strip() or None
        instrument_key = str(rows[0].get("instrument_key", "") or "").strip()
        if instrument_key:
            return instrument_key

    return None


def get_full_quotes_by_instrument_keys(instrument_keys: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    if not is_upstox_configured():
        _warn_missing_token_once()
        return {}
    keys = [str(key or "").strip() for key in instrument_keys if str(key or "").strip()]
    result: Dict[str, Dict[str, Any]] = {}
    for chunk in _chunked(keys, _CHUNK_SIZE):
        payload = _request_json("/v2/market-quote/quotes", {"instrument_key": ",".join(chunk)})
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            continue
        for row in data.values():
            if not isinstance(row, dict):
                continue
            instrument_token = str(row.get("instrument_token") or row.get("instrument_key") or "").strip()
            if instrument_token:
                result[instrument_token] = dict(row)
    return result


def get_underlying_snapshot(symbol: str) -> Dict[str, Any]:
    if not is_upstox_configured():
        _warn_missing_token_once()
        return {}
    instrument_key = get_underlying_instrument_key(symbol)
    if not instrument_key:
        return {}
    rows = get_full_quotes_by_instrument_keys([instrument_key])
    row = rows.get(instrument_key) or {}
    if not row:
        return {}
    ohlc = row.get("ohlc") or {}
    return {
        "instrument_key": instrument_key,
        "last_price": round(float(row.get("last_price", 0) or 0), 2),
        "net_change": round(float(row.get("net_change", 0) or 0), 2),
        "prev_close": round(float(ohlc.get("close", 0) or 0), 2),
    }


def get_option_contract_expiries(symbol: str) -> List[str]:
    if not is_upstox_configured():
        _warn_missing_token_once()
        return []
    instrument_key = get_underlying_instrument_key(symbol)
    if not instrument_key:
        return []

    with _option_lock:
        cached = _option_expiry_cache.get(instrument_key)
        if cached and not _is_stale(cached[0], 900.0):
            return list(cached[1])

    payload = _request_json("/v2/option/contract", {"instrument_key": instrument_key})
    rows = list(payload.get("data") or [])
    expiries = sorted({str(row.get("expiry") or "").strip() for row in rows if str(row.get("expiry") or "").strip()})

    with _option_lock:
        _option_expiry_cache[instrument_key] = (time.time(), expiries)
    return expiries


def get_nearest_option_expiry(symbol: str) -> str:
    expiries = get_option_contract_expiries(symbol)
    if not expiries:
        return ""

    today = time.strftime("%Y-%m-%d")
    future_expiries = [expiry for expiry in expiries if expiry >= today]
    return future_expiries[0] if future_expiries else expiries[0]


def get_option_chain(symbol: str, expiry_date: Optional[str] = None) -> Dict[str, Any]:
    if not is_upstox_configured():
        _warn_missing_token_once()
        return {}
    instrument_key = get_underlying_instrument_key(symbol)
    if not instrument_key:
        return {}

    selected_expiry = str(expiry_date or get_nearest_option_expiry(symbol) or "").strip()
    if not selected_expiry:
        return {}

    payload = _request_json(
        "/v2/option/chain",
        {"instrument_key": instrument_key, "expiry_date": selected_expiry},
    )
    return {
        "instrument_key": instrument_key,
        "expiry_date": selected_expiry,
        "data": list(payload.get("data") or []),
    }


def get_bulk_daily_ohlc(symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    if not is_upstox_configured():
        _warn_missing_token_once()
        return {}
    normalized_symbols = [_normalize_symbol(symbol) for symbol in symbols if _normalize_symbol(symbol)]
    result: Dict[str, Dict[str, Any]] = {}

    symbol_to_key = get_symbol_instrument_keys(normalized_symbols)
    key_to_symbol = {instrument_key: symbol for symbol, instrument_key in symbol_to_key.items()}

    for chunk_symbols in _chunked(list(symbol_to_key.keys()), _CHUNK_SIZE):
        instrument_keys = [symbol_to_key[symbol] for symbol in chunk_symbols]
        payload = _request_json(
            "/v3/market-quote/ohlc",
            {"instrument_key": ",".join(instrument_keys), "interval": "1d"},
        )
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            continue
        for row in data.values():
            if not isinstance(row, dict):
                continue
            instrument_token = str(row.get("instrument_token") or row.get("instrument_key") or "").strip()
            symbol = key_to_symbol.get(instrument_token)
            if not symbol:
                continue
            prev_ohlc = row.get("prev_ohlc") or {}
            live_ohlc = row.get("live_ohlc") or {}
            result[symbol] = {
                "prev_close": round(float(prev_ohlc.get("close", 0) or 0), 2),
                "prev_volume": float(prev_ohlc.get("volume", 0) or 0),
                "live_volume": float(live_ohlc.get("volume", 0) or 0),
                "last_price": round(float(row.get("last_price", 0) or 0), 2),
            }

    return result