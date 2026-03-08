# nse_fetcher.py — Fetches delivery% and bid-ask data from NSE API directly

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_PER_SYMBOL_TIMEOUT: float = 5.0    # skip symbol if NSE doesn't respond in 5 s
_TOTAL_BUDGET_SECS: float  = 25.0   # give up entire batch after 25 s
_MAX_WORKERS: int           = 6     # parallel requests to NSE

_BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":           "application/json, text/plain, */*",
    "Accept-Language":  "en-US,en;q=0.9",
    "Accept-Encoding":  "gzip, deflate",
    "Referer":          "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua":        '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest":   "empty",
    "sec-fetch-mode":   "cors",
    "sec-fetch-site":   "same-origin",
    "Connection":       "keep-alive",
}

_COOKIE_URL      = "https://www.nseindia.com/"
_QUOTE_PAGE_URL  = "https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE"  # warm-up for quote API cookies
_QUOTE_URL       = "https://www.nseindia.com/api/quote-equity"
_OC_PAGE_URL     = "https://www.nseindia.com/option-chain"  # warm-up URL for OC cookies
_OC_INDICES_URL  = "https://www.nseindia.com/api/option-chain-indices"

# Separate budget for bulk real-time quote fetching (all symbols, not just F&O)
_QUOTE_ALL_BUDGET_SECS: float = 90.0
_QUOTE_ALL_WORKERS: int       = 4     # reduced — fewer concurrent connections to NSE
_QUOTE_SUBMIT_DELAY: float    = 0.15   # 150 ms stagger between task submissions (prevents NSE bot-detect)

# Module-level session — reused across all calls, refreshed when cookies expire
_session: Optional[requests.Session] = None
_session_created_at: float = 0.0
_SESSION_TTL: float = 240.0  # refresh every 4 min — forces rebuild between 5-min scheduler cycles
_session_lock = threading.Lock()  # prevents thundering-herd on concurrent session creation

# Separate session for option-chain API — uses curl_cffi (Chrome TLS fingerprint) to bypass Akamai
_oi_session = None  # curl_cffi.Session or requests.Session depending on availability
_oi_session_created_at: float = 0.0
_oi_session_lock = threading.Lock()


# ── Session management ─────────────────────────────────────────────────────────

def _build_session() -> requests.Session:
    """Create a new requests.Session with NSE cookies initialised.

    Two-step warm-up (mirrors the OI session pattern):
      1. GET homepage        → base cookies (nseappid, bm_*, etc.)
      2. pause 1.5 s
      3. GET /get-quotes/equity page  → quote-API-specific cookies
      4. pause 1.0 s

    Visiting only the homepage is not enough — the /api/quote-equity
    endpoint requires cookies that NSE only sets when the browser visits
    the equity quote page first (same reason OI needs /option-chain).
    """
    s = requests.Session()
    s.headers.update(_BASE_HEADERS)
    try:
        s.get(_COOKIE_URL, timeout=10)           # step 1: base cookies
        time.sleep(1.5)                           # step 2: human-like pause
        s.headers.update({"Referer": "https://www.nseindia.com/"})
        s.get(_QUOTE_PAGE_URL, timeout=10)        # step 3: quote-API cookies
        time.sleep(1.0)                           # step 4: let page settle
        logger.info("NSE session initialised (quote-page cookies acquired).")
    except Exception as e:
        logger.warning(f"NSE cookie warm-up failed: {e}")
    return s


def _build_oi_session() -> requests.Session:
    """
    Create a session for the NSE option-chain API.

    NOTE: NSE's Akamai Bot Manager blocks all programmatic access to the
    option-chain endpoint (/api/option-chain-equities) — plain requests,
    curl_cffi TLS impersonation, and headless Playwright all return either
    {} or ERR_HTTP2_PROTOCOL_ERROR.  Only a real interactive browser with
    full JavaScript execution passes the challenge.

    This function builds the best session we can without a real browser.
    It will work occasionally (e.g. when NSE relaxes bot detection outside
    market hours) but will often return {} during live trading.
    """
    s = requests.Session()
    s.headers.update(_BASE_HEADERS)
    try:
        s.get(_COOKIE_URL, timeout=10)
        time.sleep(1.5)
        s.headers.update({"Referer": "https://www.nseindia.com/"})
        s.get(_OC_PAGE_URL, timeout=10)
        time.sleep(1.0)
        s.headers.update({"Referer": "https://www.nseindia.com/option-chain"})
        logger.info("NSE OI session initialised.")
    except Exception as e:
        logger.warning(f"NSE OI session warm-up failed: {e}")
    return s


def _get_session() -> requests.Session:
    """Return a valid (possibly cached) NSE session, refreshing if stale.

    Thread-safe: only one thread rebuilds the session at a time.
    All other threads wait and then reuse the freshly built session.
    This prevents the thundering-herd where 8+ OI threads all hit NSE
    simultaneously for cookies, which triggers 403 rate-limiting.
    """
    global _session, _session_created_at
    now = time.monotonic()
    # Fast path: session is valid, no lock needed
    if _session is not None and (now - _session_created_at) <= _SESSION_TTL:
        return _session
    # Slow path: acquire lock so only one thread rebuilds
    with _session_lock:
        # Re-check after acquiring lock (another thread may have just rebuilt it)
        now = time.monotonic()
        if _session is None or (now - _session_created_at) > _SESSION_TTL:
            _session = _build_session()
            _session_created_at = now
    return _session


def _get_oi_session() -> requests.Session:
    """Return a valid (possibly cached) OI-specific NSE session."""
    global _oi_session, _oi_session_created_at
    now = time.monotonic()
    if _oi_session is not None and (now - _oi_session_created_at) <= _SESSION_TTL:
        return _oi_session
    with _oi_session_lock:
        now = time.monotonic()
        if _oi_session is None or (now - _oi_session_created_at) > _SESSION_TTL:
            _oi_session = _build_oi_session()
            _oi_session_created_at = now
    return _oi_session


def _reset_oi_session() -> None:
    """Force a fresh OI session on next call."""
    global _oi_session, _oi_session_created_at
    with _oi_session_lock:
        _oi_session = None
        _oi_session_created_at = 0.0


def _reset_session() -> None:
    """Force a new session on next call (e.g. after 403 / cookie expiry)."""
    global _session, _session_created_at
    with _session_lock:
        _session = None
        _session_created_at = 0.0


# ── Per-symbol fetch ───────────────────────────────────────────────────────────

def fetch_nse_delivery(symbol: str) -> dict:
    """
    Fetch delivery % and bid/ask data for *symbol* (clean, no .NS suffix)
    from the NSE trade-info API endpoint.

    Returns a dict with:
        delivery_pct        — float, e.g. 55.4
        total_traded_volume — float, in lakhs
        bid_ask_ratio       — float, totalBuyQty / totalSellQty
        bid_qty             — int,   total buy quantity in order book
        ask_qty             — int,   total sell quantity in order book
        best_bid            — float, top bid price
        best_ask            — float, top ask price
    """
    session = _get_session()
    try:
        resp = session.get(
            _QUOTE_URL,
            params={"symbol": symbol.upper(), "section": "trade_info"},
            timeout=_PER_SYMBOL_TIMEOUT,
        )

        if resp.status_code in (401, 403):
            logger.warning(f"NSE {resp.status_code} for {symbol} (trade_info) — skipping.")
            return {}
        if resp.status_code != 200:
            return {}
        # NSE silent-fail: returns HTTP 200 with empty body when session is stale
        if not resp.content or not resp.content.strip():
            logger.warning(f"NSE returned empty body for {symbol} — skipping.")
            return {}

        data = resp.json()

        # ── Delivery % ────────────────────────────────────────────────────────
        dp_block = data.get("securityWiseDP", {}) or {}
        delivery_pct = float(dp_block.get("deliveryToTradedQuantity", 0) or 0)

        # ── Bid / Ask order book ──────────────────────────────────────────────
        ob = data.get("marketDeptOrderBook", {}) or {}
        ti = ob.get("tradeInfo", {}) or {}
        total_buy  = int(ob.get("totalBuyQuantity",  0) or 0)
        total_sell = int(ob.get("totalSellQuantity", 0) or 0)
        bid_ask_ratio = round(total_buy / total_sell, 2) if total_sell > 0 else 1.0

        bids = ob.get("bid", []) or []
        asks = ob.get("ask", []) or []
        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 0.0
        total_traded_vol = float(ti.get("totalTradedVolume", 0) or 0)

        return {
            "delivery_pct":        delivery_pct,
            "total_traded_volume": total_traded_vol,
            "bid_ask_ratio":       bid_ask_ratio,
            "bid_qty":             total_buy,
            "ask_qty":             total_sell,
            "best_bid":            best_bid,
            "best_ask":            best_ask,
        }

    except requests.Timeout:
        logger.warning(f"NSE fetch timed out for {symbol}")
        return {}
    except Exception as e:
        logger.warning(f"NSE fetch failed for {symbol}: {e}")
        return {}


def fetch_nse_full_quote(symbol: str) -> dict:
    """
    Fetch FULL real-time data for *symbol* from two NSE endpoints in sequence:
      1. /api/quote-equity          → priceInfo  (ltp, change%, vwap, open, high, low)
      2. /api/quote-equity?section=trade_info → volume, delivery%, bid/ask

    Returns a unified dict:
        ltp, change_pct, prev_close, day_open, vwap, day_high, day_low
        total_traded_volume (lakhs shares),
        delivery_pct, bid_ask_ratio, bid_qty, ask_qty
    Returns {} on failure.
    """
    session = _get_session()
    result: dict = {}

    # ── Call 1: priceInfo ─────────────────────────────────────────────────────
    try:
        resp = session.get(
            _QUOTE_URL,
            params={"symbol": symbol.upper()},
            timeout=_PER_SYMBOL_TIMEOUT,
        )
        if resp.status_code in (401, 403):
            logger.warning(f"NSE {resp.status_code} for {symbol} priceInfo — skipping.")
            return {}
        if resp.status_code != 200:
            return {}
        # NSE silent-fail: returns HTTP 200 with empty body when session is stale
        if not resp.content or not resp.content.strip():
            logger.warning(f"NSE empty body for {symbol} priceInfo — skipping.")
            return {}
        pi = resp.json().get("priceInfo", {}) or {}
        ltp = float(pi.get("lastPrice", 0) or 0)
        if ltp <= 0:
            return {}
        hl = pi.get("intraDayHighLow") or {}
        result = {
            "ltp":        round(ltp, 2),
            "change_pct": round(float(pi.get("pChange",       0) or 0), 2),
            "prev_close": round(float(pi.get("previousClose", 0) or 0), 2),
            "day_open":   round(float(pi.get("open",          0) or 0), 2),
            "vwap":       round(float(pi.get("vwap",          0) or 0), 2),
            "day_high":   round(float(hl.get("max", ltp)), 2),
            "day_low":    round(float(hl.get("min", ltp)), 2),
        }
    except Exception as e:
        logger.warning(f"NSE priceInfo failed for {symbol}: {e}")
        return {}

    # ── Call 2: trade_info (volume, delivery%, bid/ask) ───────────────────────
    try:
        resp2 = session.get(
            _QUOTE_URL,
            params={"symbol": symbol.upper(), "section": "trade_info"},
            timeout=_PER_SYMBOL_TIMEOUT,
        )
        if resp2.status_code == 200:
            d2   = resp2.json()
            dp   = d2.get("securityWiseDP", {}) or {}
            ob   = d2.get("marketDeptOrderBook", {}) or {}
            ti   = ob.get("tradeInfo", {}) or {}
            tbuy  = int(ob.get("totalBuyQuantity",  0) or 0)
            tsell = int(ob.get("totalSellQuantity", 0) or 0)
            bids  = ob.get("bid", []) or []
            asks  = ob.get("ask", []) or []
            result.update({
                "delivery_pct":        float(dp.get("deliveryToTradedQuantity", 0) or 0),
                "total_traded_volume": float(ti.get("totalTradedVolume", 0) or 0),  # in lakhs
                "bid_ask_ratio":       round(tbuy / tsell, 2) if tsell > 0 else 1.0,
                "bid_qty":             tbuy,
                "ask_qty":             tsell,
                "best_bid":            float(bids[0]["price"]) if bids else 0.0,
                "best_ask":            float(asks[0]["price"]) if asks else 0.0,
            })
    except Exception:
        pass  # trade_info is optional — priceInfo result is still valid

    # Log if trade_info didn't enrich the result (helps diagnose delivery% gaps)
    if "delivery_pct" not in result:
        logger.debug(f"trade_info call 2 not parsed for {symbol} — delivery/bid-ask unavailable")

    return result


def _run_nse_full_quote_batch(symbols: list, budget: float) -> Dict[str, dict]:
    """Run one batch of fetch_nse_full_quote calls with staggered submission.

    Tasks are submitted with _QUOTE_SUBMIT_DELAY between each one so NSE
    never sees a burst of simultaneous requests (which triggers empty-body
    bot-detection responses).
    """
    results: Dict[str, dict] = {}
    deadline = time.monotonic() + budget
    with ThreadPoolExecutor(max_workers=_QUOTE_ALL_WORKERS, thread_name_prefix="nse-full") as pool:
        futures = {}
        for sym in symbols:
            if time.monotonic() >= deadline:
                break
            futures[pool.submit(fetch_nse_full_quote, sym)] = sym
            time.sleep(_QUOTE_SUBMIT_DELAY)  # stagger: don't blast NSE simultaneously

        for future, sym in futures.items():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("NSE full-quote batch budget exceeded — stopping early.")
                future.cancel()
                continue
            try:
                data = future.result(timeout=min(_PER_SYMBOL_TIMEOUT * 2, remaining))
                if data:
                    results[sym] = data
            except FuturesTimeout:
                future.cancel()
            except Exception:
                pass
    return results


def fetch_all_nse_full_quotes(symbols: list) -> Dict[str, dict]:
    """
    Fetch complete real-time data for every symbol in *symbols* in parallel.

    Uses fetch_nse_full_quote (2 HTTP calls per symbol) with _QUOTE_ALL_WORKERS
    workers and _QUOTE_ALL_BUDGET_SECS hard budget.

    If all symbols fail (e.g. concurrent 403s expire the session simultaneously),
    the session is force-rebuilt and the batch retried once before giving up.

    Returns dict keyed by clean symbol (no .NS suffix).
    """
    if not symbols:
        return {}

    results = _run_nse_full_quote_batch(symbols, _QUOTE_ALL_BUDGET_SECS)
    logger.info(f"NSE full quotes fetched for {len(results)}/{len(symbols)} symbols.")

    if not results:
        # All workers returned {} — typically means session cookies expired and
        # all 8 threads got a simultaneous 403.  Force a fresh session and retry.
        logger.warning("NSE full-quote batch returned 0 results — rebuilding session and retrying.")
        _reset_session()
        _get_session()          # rebuild now (includes 1.5 s warm-up pause)
        results = _run_nse_full_quote_batch(symbols, _QUOTE_ALL_BUDGET_SECS)
        logger.info(f"NSE full quotes retry: {len(results)}/{len(symbols)} symbols.")

    return results

def fetch_all_nse_data(symbols: list) -> Dict[str, dict]:
    """
    Fetch NSE trade-info for all *symbols* in parallel with a hard time budget.

    Each symbol gets at most _PER_SYMBOL_TIMEOUT seconds; the entire batch is
    capped at _TOTAL_BUDGET_SECS so a slow NSE response never stalls the main fetch.
    """
    if not symbols:
        return {}

    results: Dict[str, dict] = {}
    deadline = time.monotonic() + _TOTAL_BUDGET_SECS

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="nse-fetch") as pool:
        futures = {pool.submit(fetch_nse_delivery, sym): sym for sym in symbols}
        for future, sym in futures.items():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("NSE fetch total budget exceeded — stopping early.")
                future.cancel()
                continue
            try:
                data = future.result(timeout=min(_PER_SYMBOL_TIMEOUT, remaining))
                if data:
                    results[sym] = data
            except FuturesTimeout:
                logger.warning(f"NSE fetch timed out for {sym}, skipping.")
                future.cancel()
            except Exception:
                pass  # skip silently

    logger.info(f"NSE data fetched for {len(results)}/{len(symbols)} symbols.")
    return results


# Legacy aliases so any remaining references don't break
def fetch_nse_data_for_all(fo_symbols: list) -> Dict[str, dict]:
    return fetch_all_nse_data(fo_symbols)

# Backward-compat shim (previously only returned ltp/change_pct/prev_close)
def fetch_all_nse_quotes(symbols: list) -> Dict[str, dict]:
    return fetch_all_nse_full_quotes(symbols)
