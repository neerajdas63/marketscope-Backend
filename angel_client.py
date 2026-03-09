import json
import logging
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.request import Request, urlopen

import pyotp
from SmartApi import SmartConnect

logger = logging.getLogger(__name__)

_EXCHANGE = "NSE"
_MASTER_URL = os.getenv(
    "ANGEL_INSTRUMENT_MASTER_URL",
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
)
_SESSION_TTL_SECONDS = int(os.getenv("ANGEL_SESSION_TTL_SECONDS", "3300"))
_MASTER_TTL_SECONDS = int(os.getenv("ANGEL_MASTER_TTL_SECONDS", "43200"))
_LTP_CACHE_TTL_SECONDS = float(os.getenv("ANGEL_LTP_CACHE_TTL_SECONDS", "2.0"))
_CANDLE_CACHE_TTL_SECONDS = float(os.getenv("ANGEL_CANDLE_CACHE_TTL_SECONDS", "30.0"))
_BATCH_CHUNK_SIZE = int(os.getenv("ANGEL_BATCH_CHUNK_SIZE", "50"))
_FALLBACK_WORKERS = int(os.getenv("ANGEL_FALLBACK_WORKERS", "8"))
_QUOTE_CACHE_TTL_SECONDS = float(os.getenv("ANGEL_QUOTE_CACHE_TTL_SECONDS", "2.0"))

_client_lock = threading.Lock()
_client: Optional[SmartConnect] = None
_client_created_at = 0.0

_instrument_lock = threading.Lock()
_instrument_cache: Dict[str, Dict[str, str]] = {}
_instrument_cache_at = 0.0

_ltp_lock = threading.Lock()
_ltp_cache: Dict[str, Tuple[float, float]] = {}

_quote_lock = threading.Lock()
_quote_cache: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}

_candles_lock = threading.Lock()
_candles_cache: Dict[Tuple[str, str, str, str], Tuple[float, List[List[Any]]]] = {}


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _normalize_symbol(symbol: str) -> str:
    return symbol.upper().replace(".NS", "").strip()


def _normalize_angel_symbol(symbol: str) -> str:
    base = symbol.upper().strip()
    if base.endswith("-EQ"):
        base = base[:-3]
    return base


def _is_stale(timestamp: float, ttl_seconds: float) -> bool:
    return timestamp <= 0 or (time.time() - timestamp) >= ttl_seconds


def _build_totp() -> str:
    secret = _require_env("ANGEL_TOTP_SECRET")
    return pyotp.TOTP(secret).now()


def _create_logged_in_client() -> SmartConnect:
    api_key = _require_env("ANGEL_API_KEY")
    client_id = _require_env("ANGEL_CLIENT_ID")
    password = os.getenv("ANGEL_PIN", "").strip() or os.getenv("ANGEL_PASSWORD", "").strip()
    if not password:
        raise RuntimeError("Missing required environment variable: ANGEL_PIN")

    client = SmartConnect(api_key=api_key)
    session = client.generateSession(client_id, password, _build_totp())
    if not isinstance(session, dict) or not session.get("status"):
        message = "Unknown SmartAPI login failure"
        if isinstance(session, dict):
            message = str(session.get("message") or session.get("errorcode") or message)
        raise RuntimeError(message)
    return client


def get_smart_client(force_refresh: bool = False) -> SmartConnect:
    global _client, _client_created_at

    with _client_lock:
        if not force_refresh and _client is not None and not _is_stale(_client_created_at, _SESSION_TTL_SECONDS):
            return _client

        _client = _create_logged_in_client()
        _client_created_at = time.time()
        logger.info("SmartAPI session established successfully.")
        return _client


def _fetch_json(url: str, timeout: int = 20) -> Any:
    request = Request(url, headers={"User-Agent": "MarketScope/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _select_best_record(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    eq_records = [r for r in records if str(r.get("symbol", "")).upper().endswith("-EQ")]
    if eq_records:
        return eq_records[0]
    return records[0] if records else None


def _load_instrument_cache(force_refresh: bool = False) -> Dict[str, Dict[str, str]]:
    global _instrument_cache, _instrument_cache_at

    with _instrument_lock:
        if not force_refresh and _instrument_cache and not _is_stale(_instrument_cache_at, _MASTER_TTL_SECONDS):
            return _instrument_cache

        raw = _fetch_json(_MASTER_URL)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in raw:
            if str(record.get("exch_seg", "")).upper() != _EXCHANGE:
                continue
            symbol_token = str(record.get("token", "")).strip()
            trading_symbol = str(record.get("symbol", "")).strip()
            if not symbol_token or not trading_symbol:
                continue
            normalized = _normalize_angel_symbol(trading_symbol)
            grouped.setdefault(normalized, []).append(record)

        selected: Dict[str, Dict[str, str]] = {}
        for normalized, records in grouped.items():
            best = _select_best_record(records)
            if not best:
                continue
            selected[normalized] = {
                "token": str(best.get("token", "")).strip(),
                "trading_symbol": str(best.get("symbol", "")).strip(),
            }

        _instrument_cache = selected
        _instrument_cache_at = time.time()
        logger.info("Angel instrument master loaded for %d NSE symbols.", len(selected))
        return _instrument_cache


def get_symbol_token(symbol: str) -> Optional[str]:
    record = _load_instrument_cache().get(_normalize_symbol(symbol))
    return record.get("token") if record else None


def get_symbol_tokens(symbols: Iterable[str]) -> Dict[str, str]:
    instrument_cache = _load_instrument_cache()
    result: Dict[str, str] = {}
    for symbol in symbols:
        normalized = _normalize_symbol(symbol)
        record = instrument_cache.get(normalized)
        if record:
            result[normalized] = record["token"]
    return result


def _get_trading_symbol(symbol: str) -> Optional[str]:
    record = _load_instrument_cache().get(_normalize_symbol(symbol))
    return record.get("trading_symbol") if record else None


def _get_cached_quote(mode: str, symbol: str) -> Optional[Dict[str, Any]]:
    cache_key = (mode.upper(), _normalize_symbol(symbol))
    with _quote_lock:
        cached = _quote_cache.get(cache_key)
        if not cached:
            return None
        cached_at, payload = cached
        if _is_stale(cached_at, _QUOTE_CACHE_TTL_SECONDS):
            _quote_cache.pop(cache_key, None)
            return None
        return dict(payload)


def _set_cached_quote(mode: str, symbol: str, payload: Dict[str, Any]) -> None:
    cache_key = (mode.upper(), _normalize_symbol(symbol))
    with _quote_lock:
        _quote_cache[cache_key] = (time.time(), dict(payload))


def _get_cached_ltp(symbol: str) -> Optional[float]:
    normalized = _normalize_symbol(symbol)
    with _ltp_lock:
        cached = _ltp_cache.get(normalized)
        if not cached:
            return None
        cached_at, ltp = cached
        if _is_stale(cached_at, _LTP_CACHE_TTL_SECONDS):
            _ltp_cache.pop(normalized, None)
            return None
        return ltp


def _set_cached_ltp(symbol: str, ltp: float) -> None:
    normalized = _normalize_symbol(symbol)
    with _ltp_lock:
        _ltp_cache[normalized] = (time.time(), round(float(ltp), 2))


def _is_auth_failure(error: Exception) -> bool:
    message = str(error).upper()
    return any(fragment in message for fragment in ("SESSION", "TOKEN", "AUTH", "AG8001", "ACCESS DENIED"))


def _call_with_reauth(func):
    try:
        return func(get_smart_client())
    except Exception as exc:
        if not _is_auth_failure(exc):
            raise
        logger.warning("SmartAPI request failed due to auth issue, retrying with a fresh session.")
        return func(get_smart_client(force_refresh=True))


def _get_market_data_method(client: SmartConnect):
    method = getattr(client, "market_data", None)
    if method is None:
        method = getattr(client, "getMarketData", None)
    if method is None:
        method = getattr(client, "marketData", None)
    return method


def _normalize_quote_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "exchange": str(row.get("exchange", _EXCHANGE)),
        "tradingSymbol": str(row.get("tradingSymbol") or row.get("tradingsymbol") or ""),
        "symbolToken": str(row.get("symbolToken") or row.get("symboltoken") or ""),
    }
    for field in (
        "ltp", "open", "high", "low", "close", "lastTradeQty", "netChange",
        "percentChange", "avgPrice", "tradeVolume", "opnInterest", "lowerCircuit",
        "upperCircuit", "totBuyQuan", "totSellQuan", "52WeekLow", "52WeekHigh",
        "exchFeedTime", "exchTradeTime", "depth",
    ):
        if field in row:
            normalized[field] = row[field]
    return normalized


def get_market_quotes(symbols: Iterable[str], mode: str = "LTP") -> Dict[str, Dict[str, Any]]:
    requested_mode = mode.upper()
    if requested_mode not in {"LTP", "OHLC", "FULL"}:
        raise ValueError(f"Unsupported Angel market data mode: {mode}")

    normalized_symbols = [_normalize_symbol(symbol) for symbol in symbols]
    result: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for symbol in normalized_symbols:
        cached = _get_cached_quote(requested_mode, symbol)
        if cached is not None:
            result[symbol] = cached
        else:
            missing.append(symbol)

    if not missing:
        return result

    instrument_cache = _load_instrument_cache()
    token_to_symbol: Dict[str, str] = {}
    trading_symbol_to_symbol: Dict[str, str] = {}
    token_list: List[str] = []
    for symbol in missing:
        record = instrument_cache.get(symbol)
        if not record:
            continue
        token = record["token"]
        token_to_symbol[token] = symbol
        trading_symbol_to_symbol[record["trading_symbol"]] = symbol
        token_list.append(token)

    if not token_list:
        return result

    market_data_method = _get_market_data_method(get_smart_client())
    if market_data_method is None:
        raise RuntimeError("SmartAPI client does not expose market quote API")

    for chunk in _chunked(token_list, _BATCH_CHUNK_SIZE):
        response = _call_with_reauth(
            lambda client: _get_market_data_method(client)(requested_mode, {_EXCHANGE: chunk})
        )
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected SmartAPI market data response: {response}")

        success = response.get("status")
        if success is False:
            raise RuntimeError(str(response.get("message") or response.get("errorcode") or "Market data request failed"))

        payload = response.get("data") or {}
        fetched_rows = payload.get("fetched") or []
        for row in fetched_rows:
            symbol_token = str(row.get("symbolToken") or row.get("symboltoken") or "")
            symbol = token_to_symbol.get(symbol_token)
            if not symbol:
                trading_symbol = str(row.get("tradingSymbol") or row.get("tradingsymbol") or "")
                symbol = trading_symbol_to_symbol.get(trading_symbol)
            if not symbol:
                continue
            normalized_row = _normalize_quote_row(row)
            result[symbol] = normalized_row
            _set_cached_quote(requested_mode, symbol, normalized_row)
            if "ltp" in normalized_row and normalized_row["ltp"]:
                _set_cached_ltp(symbol, float(normalized_row["ltp"]))

        unfetched_rows = payload.get("unfetched") or []
        if unfetched_rows:
            logger.warning("Angel market data unfetched for %d symbols in %s mode.", len(unfetched_rows), requested_mode)

    return result


def get_ohlc_quotes(symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    return get_market_quotes(symbols, mode="OHLC")


def get_full_quotes(symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    return get_market_quotes(symbols, mode="FULL")


def get_ltp(symbol: str, token: str) -> float:
    normalized = _normalize_symbol(symbol)
    cached = _get_cached_ltp(normalized)
    if cached is not None:
        return cached

    quote = get_market_quotes([normalized], mode="LTP").get(normalized)
    if not quote:
        raise RuntimeError(f"No Angel LTP quote found for {normalized}")
    ltp_value = round(float(quote.get("ltp", 0) or 0), 2)
    if ltp_value <= 0:
        raise RuntimeError(f"Invalid LTP response for {normalized}")
    _set_cached_ltp(normalized, ltp_value)
    return ltp_value


def _chunked(items: List[str], chunk_size: int) -> Iterable[List[str]]:
    for index in range(0, len(items), chunk_size):
        yield items[index:index + chunk_size]


def get_bulk_ltp(symbols: Iterable[str]) -> Dict[str, float]:
    quotes = get_market_quotes(symbols, mode="LTP")
    return {
        symbol: round(float(payload.get("ltp", 0) or 0), 2)
        for symbol, payload in quotes.items()
        if float(payload.get("ltp", 0) or 0) > 0
    }


def get_bulk_full_quotes(symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch FULL market data via SmartAPI for all symbols and normalise into
    the same shape that fetcher.py previously expected from NSE scraping.

    Returns dict keyed by clean symbol with:
        ltp, change_pct, prev_close, day_open, day_high, day_low, vwap,
        total_traded_volume (in lakhs), bid_qty, ask_qty, bid_ask_ratio
    """
    raw = get_market_quotes(symbols, mode="FULL")
    result: Dict[str, Dict[str, Any]] = {}
    for symbol, row in raw.items():
        ltp = float(row.get("ltp", 0) or 0)
        if ltp <= 0:
            continue
        prev_close = float(row.get("close", 0) or 0)
        tot_buy = int(row.get("totBuyQuan", 0) or 0)
        tot_sell = int(row.get("totSellQuan", 0) or 0)
        trade_vol = float(row.get("tradeVolume", 0) or 0)
        result[symbol] = {
            "ltp":                round(ltp, 2),
            "change_pct":         round(float(row.get("percentChange", 0) or 0), 2),
            "prev_close":         round(prev_close, 2),
            "day_open":           round(float(row.get("open", 0) or 0), 2),
            "day_high":           round(float(row.get("high", 0) or 0), 2),
            "day_low":            round(float(row.get("low", 0) or 0), 2),
            "vwap":               round(float(row.get("avgPrice", 0) or 0), 2),
            "total_traded_volume": round(trade_vol / 100_000, 2),  # shares → lakhs
            "bid_qty":            tot_buy,
            "ask_qty":            tot_sell,
            "bid_ask_ratio":      round(tot_buy / tot_sell, 2) if tot_sell > 0 else 1.0,
            "delivery_pct":       None,
            "quote_source":       "smartapi_full",
            "delivery_source":    "unavailable_smartapi",
        }
    return result


def get_intraday_candles(symbol_token: str, interval: str, from_date: str, to_date: str) -> List[List[Any]]:
    cache_key = (str(symbol_token), interval, from_date, to_date)
    with _candles_lock:
        cached = _candles_cache.get(cache_key)
        if cached and not _is_stale(cached[0], _CANDLE_CACHE_TTL_SECONDS):
            return cached[1]

    params = {
        "exchange": _EXCHANGE,
        "symboltoken": str(symbol_token),
        "interval": interval,
        "fromdate": from_date,
        "todate": to_date,
    }

    def _request(client: SmartConnect) -> List[List[Any]]:
        response = client.getCandleData(params)
        if not isinstance(response, dict) or not response.get("status"):
            raise RuntimeError(str(response.get("message") if isinstance(response, dict) else response))
        return list(response.get("data") or [])

    candles = _call_with_reauth(_request)
    with _candles_lock:
        _candles_cache[cache_key] = (time.time(), candles)
    return candles