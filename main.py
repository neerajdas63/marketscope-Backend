# main.py — MarketScope FastAPI application entry point

import os
os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.auth_access import authorize_request
from cache import InMemoryCache
from backend.momentum_pulse import get_momentum_pulse, schedule_momentum_pulse_refresh
from backend.pulse_navigator import get_pulse_navigator, get_pulse_navigator_tab
from backend.trade_guardian import (
    acknowledge_alert,
    close_trade,
    create_trade,
    get_trade_detail,
    get_trade_guardian_summary,
    init_trade_guardian_storage,
    list_alerts,
    list_trades,
    run_trade_guardian_monitor_cycle,
    send_trade_guardian_test_alert,
)
from fetcher import fetch_all_sectors
from apscheduler.triggers.combining import OrTrigger
from apscheduler.triggers.cron import CronTrigger
from scheduler import is_market_hours, start_scheduler
from sector_momentum import get_momentum_data as get_sector_momentum_data
from sector_momentum import get_historical_momentum
from sector_momentum import set_cache_ref as momentum_set_cache_ref
from sector_momentum import take_snapshot as momentum_take_snapshot
from sector_momentum import backfill_today_snapshots
from sector_momentum import get_relative_sector_strength
from sector_scope import calculate_sector_scope
from morning_watchlist import get_morning_watchlist, get_live_watchlist
from oi_analysis import get_oi_analysis, get_bulk_oi
from oi_analysis import refresh_fo_radar_cache, get_fo_radar_snapshot, fo_radar_cache_age_seconds
from market_breadth import get_market_breadth
from trade_planner import get_trade_plan, get_bulk_trade_plans

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Shared cache instance (used by routes and scheduler)
cache: InMemoryCache = InMemoryCache()

CACHE_DURATION_SECONDS: int = int(os.getenv("CACHE_DURATION_SECONDS", 300))
INITIAL_CACHE_RETRY_ATTEMPTS: int = int(os.getenv("INITIAL_CACHE_RETRY_ATTEMPTS", 3))
INITIAL_CACHE_RETRY_DELAY_SECONDS: float = float(os.getenv("INITIAL_CACHE_RETRY_DELAY_SECONDS", 5))
TRADE_GUARDIAN_POLL_SECONDS: int = max(2, int(os.getenv("TRADE_GUARDIAN_POLL_SECONDS", "10")))
TRADE_GUARDIAN_STARTUP_DELAY_SECONDS: int = max(0, int(os.getenv("TRADE_GUARDIAN_STARTUP_DELAY_SECONDS", "20")))

_trade_guardian_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trade-guardian")


class TradeGuardianCreateRequest(BaseModel):
    symbol: str = Field(..., min_length=1)
    direction: str = Field(..., min_length=4)
    entry_price: float = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0)
    target_1: float = Field(..., gt=0)
    target_2: float = Field(..., gt=0)
    quantity: float = Field(default=0, ge=0)
    notes: str = Field(default="")


class TradeGuardianCloseRequest(BaseModel):
    reason: str = Field(default="closed_manual")


class TradeGuardianTestAlertRequest(BaseModel):
    text: str = Field(default="Trade Guardian test alert")


def _warming_up_response(message: str = "Data cache is warming up", **payload: Any) -> Dict[str, Any]:
    """Return a structured warm-up response without changing route paths."""
    response = dict(payload)
    response.setdefault("status", "warming_up")
    response.setdefault("message", message)
    return response


# ---------------------------------------------------------------------------
# Sector momentum snapshot helpers
# ---------------------------------------------------------------------------

def _is_momentum_window() -> bool:
    """Return True if current IST time is within 9:15–10:00 AM on a weekday."""
    import pytz as _pytz
    now = datetime.now(_pytz.timezone("Asia/Kolkata"))
    if now.weekday() > 4:
        return False
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=10, minute=0, second=0, microsecond=0)
    return start <= now <= end


def _is_after_open_today() -> bool:
    """Return True if it's a weekday and current IST time is at or after 9:15 AM
    (regardless of whether the opening window has already closed)."""
    import pytz as _pytz
    now = datetime.now(_pytz.timezone("Asia/Kolkata"))
    if now.weekday() > 4:
        return False
    open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    return now >= open_time


def _should_backfill_opening_window() -> bool:
    """Return True only while the 9:15-10:00 opening backfill window is still relevant."""
    return _is_momentum_window()


def _momentum_snapshot_job() -> None:
    """Scheduled every 5 min; captures a sector momentum snapshot during the open window."""
    if not _is_momentum_window():
        return
    data = cache.get()
    if not data:
        return
    momentum_take_snapshot(data.get("sectors", []))


async def _trade_guardian_monitor_job() -> None:
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(_trade_guardian_executor, run_trade_guardian_monitor_cycle)
    except Exception as exc:
        logger.error("Trade Guardian monitor job failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Background startup helpers
# ---------------------------------------------------------------------------

async def _bg_init_fetch() -> None:
    """Run the initial market data fetch before the app reports ready."""
    last_exc: Exception | None = None
    for attempt in range(1, INITIAL_CACHE_RETRY_ATTEMPTS + 1):
        try:
            logger.info("[INIT] Starting initial market data fetch (attempt %d/%d)...", attempt, INITIAL_CACHE_RETRY_ATTEMPTS)
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mscope-init") as ex:
                initial_data = await loop.run_in_executor(ex, fetch_all_sectors)
            cache.set(initial_data)
            schedule_momentum_pulse_refresh(
                scanner_stocks=list(initial_data.get("scanner_stocks", [])),
                last_updated=str(initial_data.get("last_updated", "") or ""),
                force=True,
            )
            logger.info("[INIT] Initial fetch completed successfully.")
            break
        except Exception as exc:
            last_exc = exc
            logger.error("[INIT] Initial fetch failed on attempt %d/%d: %s", attempt, INITIAL_CACHE_RETRY_ATTEMPTS, exc, exc_info=True)
            if attempt >= INITIAL_CACHE_RETRY_ATTEMPTS:
                raise RuntimeError("Initial cache warm-up failed") from exc
            await asyncio.sleep(INITIAL_CACHE_RETRY_DELAY_SECONDS)

    # Immediately seed the F&O Radar OI cache in its own background thread.
    # This runs after the initial fetch so stock price data is already available.
    try:
        from stocks import FO_STOCKS
        fo_clean          = [s.replace(".NS", "") for s in FO_STOCKS]
        scanner_stocks    = cache.get().get("scanner_stocks", [])
        loop              = asyncio.get_running_loop()
        _radar_ex         = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fo-radar-init")
        loop.run_in_executor(_radar_ex, refresh_fo_radar_cache, fo_clean, scanner_stocks, False)
        logger.info("[BG] F&O Radar OI refresh started in background (%d symbols).", len(fo_clean))
    except Exception as exc:
        logger.warning("[BG] Could not start F&O Radar refresh: %s", exc)


async def _bg_backfill() -> None:
    """Backfill opening momentum slots after a post-open server start."""
    if not _should_backfill_opening_window():
        logger.info("[BG] Skipping opening backfill outside the 9:15-10:00 momentum window.")
        return
    try:
        logger.info("[BG] Backfilling opening momentum slots...")
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mscope-backfill") as ex:
            await loop.run_in_executor(ex, backfill_today_snapshots)
        logger.info("[BG] Backfill done.")
        if _is_momentum_window():
            logger.info("[BG] Still inside momentum window — taking live catch-up snapshot.")
            _momentum_snapshot_job()
    except Exception as exc:
        logger.error(f"[BG] Backfill failed: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the critical cache before reporting the server as ready."""
    logger.info("MarketScope starting...")
    init_trade_guardian_storage()

    # Give sector_momentum a reference to the cache for EOD fallback snapshots
    momentum_set_cache_ref(cache)

    # Start the background scheduler
    scheduler = start_scheduler(cache)

    # Add sector momentum snapshot job — fires exactly at 9:15,9:20,...,9:55 and 10:00 IST
    _snapshot_trigger = OrTrigger([
        CronTrigger(day_of_week="mon-fri", hour=9, minute="15,20,25,30,35,40,45,50,55", timezone="Asia/Kolkata"),
        CronTrigger(day_of_week="mon-fri", hour=10, minute=0, timezone="Asia/Kolkata"),
    ])
    scheduler.add_job(
        _momentum_snapshot_job,
        trigger=_snapshot_trigger,
        id="sector_momentum_snapshot",
        name="Sector momentum snapshot at slot times",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )
    logger.info("Sector momentum snapshot job registered (cron-aligned).")

    scheduler.add_job(
        _trade_guardian_monitor_job,
        trigger="interval",
        seconds=TRADE_GUARDIAN_POLL_SECONDS,
        id="trade_guardian_monitor",
        name="Trade Guardian live monitor",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=15,
        next_run_time=datetime.now() + timedelta(seconds=TRADE_GUARDIAN_STARTUP_DELAY_SECONDS),
    )
    logger.info(
        "Trade Guardian monitor job registered (%ss interval, %ss startup delay).",
        TRADE_GUARDIAN_POLL_SECONDS,
        TRADE_GUARDIAN_STARTUP_DELAY_SECONDS,
    )

    # Start initial fetch in the background (non-blocking)
    asyncio.create_task(_bg_init_fetch())

    # Catch-up backfill also runs in the background if started after market open.
    if _should_backfill_opening_window():
        logger.info("Server started during opening window — scheduling background backfill...")
        asyncio.create_task(_bg_backfill())
    elif _is_after_open_today():
        logger.info("Server started after opening window — skipping background backfill.")

    logger.info("MarketScope startup complete — server is ready.")

    yield  # Application is now running and accepting requests

    logger.info("MarketScope shutting down...")
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MarketScope API",
    description="Indian Stock Market Heatmap Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


_PUBLIC_PATHS = {"/", "/health", "/openapi.json", "/docs", "/redoc", "/favicon.ico"}
_PUBLIC_PREFIXES = ("/docs", "/redoc")


@app.middleware("http")
async def access_control_middleware(request: Request, call_next):
    if request.method.upper() == "OPTIONS":
        return await call_next(request)

    path = request.url.path or "/"
    if path in _PUBLIC_PATHS or any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES):
        return await call_next(request)

    try:
        request.state.current_user = await authorize_request(request.headers.get("Authorization", ""))
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return await call_next(request)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", summary="Welcome", tags=["General"])
def root() -> Dict[str, Any]:
    """Welcome endpoint — returns app info and available endpoints."""
    return {
        "app": "MarketScope API",
        "endpoints": ["/heatmap", "/momentum-pulse", "/health"],
        "description": "Indian Stock Market Heatmap Backend",
    }


@app.get("/heatmap", summary="Full sector & stock heatmap data", tags=["Market Data"])
async def get_heatmap() -> Dict[str, Any]:
    """
    Returns the latest sector and stock heatmap data.

    - If the cache is stale (> CACHE_DURATION_SECONDS) AND the market is open,
      a fresh fetch is triggered synchronously before responding.
    - If no data is available yet, returns an error with a retry hint.
    """
    # If cache is empty or stale, return a warming up/loading response
    data = cache.get()
    if data is None or cache.is_stale(CACHE_DURATION_SECONDS):
        return _warming_up_response(stocks=[], sectors=[], last_updated="", total=0)

    # Ensure market_open reflects the current moment
    response = dict(data)
    response["market_open"] = is_market_hours()
    return response


@app.get("/rfactor", summary="Top stocks ranked by R-Factor score", tags=["Market Data"])
async def get_rfactor(
    limit: int = 20,
    fo_only: bool = False,
    min_score: float = 0,
    sort_by: str = "rfactor",
) -> Dict[str, Any]:
    """
    Returns stocks ranked by R-Factor score (composite of volume, price action,
    RSI, MFI, and relative strength).

    Query params:
    - limit: number of stocks to return (default 20)
    - fo_only: filter to F&O eligible stocks only
    - min_score: minimum R-Factor score threshold
    - sort_by: rfactor | opportunity | trend
    """
    return {
        "stocks": [],
        "last_updated": cache.get().get("last_updated", "") if cache.get() else "",
        "total": 0,
        "sort_by": str(sort_by or "rfactor").strip().lower(),
        "status": "disabled",
        "message": "R-Factor is disabled to reduce backend refresh load.",
    }


@app.get("/momentum-pulse", summary="Live intraday abnormal activity discovery engine", tags=["Market Data"])
async def momentum_pulse_endpoint(
    limit: int = 40,
    direction: str = "ALL",
    include_veryweak: bool = False,
) -> Dict[str, Any]:
    """
    Returns live intraday movers ranked by abnormal same-time volume/range activity,
    relative strength vs Nifty, directional consistency, light VWAP alignment,
    and score trend improvement.

    Query params:
    - limit: number of stocks to return (default 40)
    - direction: ALL | LONG | SHORT
    - include_veryweak: include veryweak tier names in response
    """
    cached = cache.get()
    if not cached:
        return _warming_up_response(stocks=[], total=0, last_updated="", direction="ALL")

    try:
        result = get_momentum_pulse(
            scanner_stocks=list(cached.get("scanner_stocks", [])),
            last_updated=str(cached.get("last_updated", "") or ""),
            direction=direction,
            include_veryweak=include_veryweak,
            limit=limit,
        )
        return result
    except Exception as exc:
        logger.error("Momentum Pulse endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Momentum Pulse computation failed") from exc


@app.get("/pulse-navigator", summary="Curated discovery upgrade built on Momentum Pulse", tags=["Market Data"])
async def pulse_navigator_endpoint(
    limit: int = 12,
    preset: str = "balanced",
    direction: str = "ALL",
) -> Dict[str, Any]:
    cached = cache.get()
    if not cached:
        return _warming_up_response(
            message="Pulse Navigator is warming up",
            feature="Pulse Navigator",
            feature_key="pulse_navigator",
            tabs={
                "discover": {"tab": "discover", "title": "Discover", "buckets": []},
                "leaders": {"tab": "leaders", "title": "Session Leaders", "longs": [], "shorts": []},
                "fresh": {"tab": "fresh", "title": "Fresh Movers", "stocks": []},
                "sectors": {"tab": "sectors", "title": "Sector Leaders", "sectors": []},
            },
        )

    try:
        return get_pulse_navigator(
            scanner_stocks=list(cached.get("scanner_stocks", [])),
            last_updated=str(cached.get("last_updated", "") or ""),
            preset=preset,
            direction=direction,
            limit=limit,
        )
    except Exception as exc:
        logger.error("Pulse Navigator endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Pulse Navigator computation failed") from exc


@app.get("/pulse-navigator/discover", summary="Pulse Navigator discover tab", tags=["Market Data"])
async def pulse_navigator_discover_endpoint(
    limit: int = 12,
    preset: str = "balanced",
    direction: str = "ALL",
) -> Dict[str, Any]:
    cached = cache.get()
    if not cached:
        return _warming_up_response(message="Pulse Navigator discover tab is warming up", tab={"tab": "discover", "title": "Discover", "buckets": []})

    try:
        return get_pulse_navigator_tab(
            scanner_stocks=list(cached.get("scanner_stocks", [])),
            last_updated=str(cached.get("last_updated", "") or ""),
            tab="discover",
            preset=preset,
            direction=direction,
            limit=limit,
        )
    except Exception as exc:
        logger.error("Pulse Navigator discover endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Pulse Navigator discover computation failed") from exc


@app.get("/pulse-navigator/fresh", summary="Pulse Navigator fresh movers tab", tags=["Market Data"])
async def pulse_navigator_fresh_endpoint(
    limit: int = 12,
    preset: str = "balanced",
    direction: str = "ALL",
) -> Dict[str, Any]:
    cached = cache.get()
    if not cached:
        return _warming_up_response(message="Pulse Navigator fresh tab is warming up", tab={"tab": "fresh", "title": "Fresh Movers", "stocks": []})

    try:
        return get_pulse_navigator_tab(
            scanner_stocks=list(cached.get("scanner_stocks", [])),
            last_updated=str(cached.get("last_updated", "") or ""),
            tab="fresh",
            preset=preset,
            direction=direction,
            limit=limit,
        )
    except Exception as exc:
        logger.error("Pulse Navigator fresh endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Pulse Navigator fresh computation failed") from exc


@app.get("/pulse-navigator/leaders", summary="Pulse Navigator session leaders tab", tags=["Market Data"])
async def pulse_navigator_leaders_endpoint(
    limit: int = 12,
    preset: str = "balanced",
    direction: str = "ALL",
) -> Dict[str, Any]:
    cached = cache.get()
    if not cached:
        return _warming_up_response(
            message="Pulse Navigator leaders tab is warming up",
            tab={"tab": "leaders", "title": "Session Leaders", "longs": [], "shorts": []},
        )

    try:
        return get_pulse_navigator_tab(
            scanner_stocks=list(cached.get("scanner_stocks", [])),
            last_updated=str(cached.get("last_updated", "") or ""),
            tab="leaders",
            preset=preset,
            direction=direction,
            limit=limit,
        )
    except Exception as exc:
        logger.error("Pulse Navigator leaders endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Pulse Navigator leaders computation failed") from exc


@app.get("/pulse-navigator/sectors", summary="Pulse Navigator sector leaders tab", tags=["Market Data"])
async def pulse_navigator_sectors_endpoint(
    limit: int = 12,
    preset: str = "balanced",
    direction: str = "ALL",
) -> Dict[str, Any]:
    cached = cache.get()
    if not cached:
        return _warming_up_response(message="Pulse Navigator sectors tab is warming up", tab={"tab": "sectors", "title": "Sector Leaders", "sectors": []})

    try:
        return get_pulse_navigator_tab(
            scanner_stocks=list(cached.get("scanner_stocks", [])),
            last_updated=str(cached.get("last_updated", "") or ""),
            tab="sectors",
            preset=preset,
            direction=direction,
            limit=limit,
        )
    except Exception as exc:
        logger.error("Pulse Navigator sectors endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Pulse Navigator sectors computation failed") from exc


@app.get("/scanner", summary="Scan stocks by change, volume, direction and sector", tags=["Market Data"])
async def get_scanner(
    min_change: float = 1.0,
    direction: str = "ALL",
    fo_only: bool = False,
    min_volume: float = 0,
    sectors: str = "",
) -> Dict[str, Any]:
    """
    Filters and tags stocks based on momentum criteria.

    Query params:
    - min_change: minimum absolute % change (default 1.0)
    - direction: ALL | GAINERS | LOSERS
    - fo_only: filter to F&O eligible stocks only
    - min_volume: minimum volume_ratio threshold
    - sectors: comma-separated sector names to filter (e.g. "IT,AUTO")
    """
    cached = cache.get()
    if not cached:
        return _warming_up_response(stocks=[], total=0, last_updated="")

    # Use the scanner_stocks universe (not heatmap sectors)
    all_stocks = list(cached.get("scanner_stocks", []))

    # Apply filters
    if fo_only:
        all_stocks = [s for s in all_stocks if s.get("fo")]

    if sectors:
        sector_list = [s.strip().upper() for s in sectors.split(",")]
        all_stocks = [s for s in all_stocks if s["sector"].upper() in sector_list]

    if direction == "GAINERS":
        all_stocks = [s for s in all_stocks if s["change_pct"] > 0]
    elif direction == "LOSERS":
        all_stocks = [s for s in all_stocks if s["change_pct"] < 0]

    if min_change > 0:
        all_stocks = [s for s in all_stocks if abs(s["change_pct"]) >= min_change]

    if min_volume > 0:
        all_stocks = [s for s in all_stocks if s.get("volume_ratio", 0) >= min_volume]

    # Add signal tag
    for stock in all_stocks:
        vol = stock.get("volume_ratio", 0)
        chg = abs(stock.get("change_pct", 0))
        if chg >= 2.0 and vol >= 2.0:
            stock["signal"] = "MOMENTUM"
        elif vol >= 2.0:
            stock["signal"] = "VOLUME SPIKE"
        elif chg >= 3.0:
            stock["signal"] = "BREAKOUT"
        else:
            stock["signal"] = ""

    # Sort by abs change_pct descending
    all_stocks.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

    return {
        "stocks": all_stocks,
        "total": len(all_stocks),
        "last_updated": cached.get("last_updated", ""),
    }


@app.get("/boost", summary="Top stocks ranked by intraday acceleration score (alias)", tags=["Market Data"])
@app.get("/intraday-boost", summary="Top stocks ranked by intraday acceleration score", tags=["Market Data"])
async def get_intraday_boost(
    limit: int = 20,
    fo_only: bool = False,
    min_score: float = 0,
) -> Dict[str, Any]:
    """
    Returns stocks ranked by boost_score — a direction-neutral measure of intraday
    acceleration based on volume burst, price velocity, range position, and candle
    consistency.

    Query params:
    - limit: number of stocks to return (default 20)
    - fo_only: filter to F&O eligible stocks only
    - min_score: minimum boost_score threshold
    """
    cached = cache.get()
    if not cached:
        return _warming_up_response(stocks=[], total=0, last_updated="")

    # Use the scanner_stocks universe (not heatmap sectors)
    all_stocks = list(cached.get("scanner_stocks", []))

    if fo_only:
        all_stocks = [s for s in all_stocks if s.get("fo")]
    if min_score > 0:
        all_stocks = [s for s in all_stocks if s.get("boost_score", 0) >= min_score]

    all_stocks.sort(key=lambda x: x.get("boost_score", 0), reverse=True)
    all_stocks = all_stocks[:limit]

    return {
        "stocks": all_stocks,
        "total": len(all_stocks),
        "last_updated": cached.get("last_updated", ""),
    }


@app.get("/sequence-signals", summary="Today's OB + FVG + MTF sequence strategy signals", tags=["Market Data"])
async def get_sequence_strategy_signals(
    limit: int = 200,
    timeframe: str = "ALL",
    side: str = "ALL",
    signal_type: str = "ALL",
    session_date: str = "",
) -> Dict[str, Any]:
    """
    Sequence Signals is temporarily disabled to protect core feature stability.
    """
    return {
        "status": "disabled",
        "message": "Sequence Signals is temporarily disabled to avoid impacting core backend stability.",
        "source": "disabled",
        "session_date": session_date or datetime.now().strftime("%Y-%m-%d"),
        "market_data_last_updated": cache.get().get("last_updated", "") if cache.get() else "",
        "last_updated": cache.get().get("last_updated", "") if cache.get() else "",
        "filters": {
            "timeframe": str(timeframe or "ALL").upper(),
            "side": str(side or "ALL").upper(),
            "signal_type": str(signal_type or "ALL").upper(),
            "limit": limit,
        },
        "summary": {
            "total": 0,
            "timeframes": {"3m": 0, "5m": 0, "15m": 0},
            "signal_types": {"C2": 0, "C3": 0, "MTF": 0},
            "sides": {"BUY": 0, "SELL": 0},
        },
        "signals": [],
    }


@app.get("/sector-scope", summary="Intra-sector relative strength ranking", tags=["Market Data"])
async def get_sector_scope(
    sector: str = "",
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Returns stocks ranked by scope_score within their sector.

    Query params:
    - sector: sector name to filter (e.g. "AUTO"). Omit to get top N from every sector.
    - limit: stocks to return per sector (default 5)
    """
    cached = cache.get()
    if not cached:
        if sector:
            return _warming_up_response(sector=sector.strip().upper(), stocks=[], total=0, last_updated="")
        return _warming_up_response(sectors=[], last_updated="")

    # Compute scope scores on a shallow copy of the cached sector list
    import copy
    sectors_copy = copy.deepcopy(cached["sectors"])
    sectors_copy = calculate_sector_scope(sectors_copy)

    if sector:
        target = sector.strip().upper()
        matched = [
            s for s in sectors_copy
            if s["name"].upper() == target
        ]
        if not matched:
            raise HTTPException(status_code=404, detail=f"Sector '{sector}' not found")
        sec = matched[0]
        top_stocks = sorted(
            sec["stocks"], key=lambda x: x.get("scope_score", 0), reverse=True
        )[:limit]
        return {
            "sector": sec["name"],
            "stocks": top_stocks,
            "total": len(top_stocks),
            "last_updated": cached.get("last_updated", ""),
        }
    else:
        # Top N from each sector
        result = []
        for sec in sectors_copy:
            top_stocks = sorted(
                sec["stocks"], key=lambda x: x.get("scope_score", 0), reverse=True
            )[:limit]
            result.append({
                "sector": sec["name"],
                "stocks": top_stocks,
            })
        return {
            "sectors": result,
            "last_updated": cached.get("last_updated", ""),
        }


def get_symbols() -> Dict[str, Any]:
    """Build a flat sym_data dict from cached sectors (heatmap — deduped by symbol)."""
    cached = cache.get()
    if not cached:
        return {}
    sym_data: Dict[str, Any] = {}
    for sector in cached.get("sectors", []):
        for stock in sector.get("stocks", []):
            sym = stock.get("symbol")
            if sym and sym not in sym_data:
                sym_data[sym] = stock
    return sym_data


def get_scanner_symbols() -> Dict[str, Any]:
    """Build a flat sym_data dict from the scanner_stocks cache (used by scanner/boost/rfactor/planner/fo-radar)."""
    cached = cache.get()
    if not cached:
        return {}
    sym_data: Dict[str, Any] = {}
    for stock in cached.get("scanner_stocks", []):
        sym = stock.get("symbol")
        if sym and sym not in sym_data:
            sym_data[sym] = stock
    return sym_data


@app.get("/breakout", summary="Top breakout stocks (52W high, volume, RSI, RS signals)", tags=["Market Data"])
def breakout_endpoint() -> Dict[str, Any]:
    """
    Returns breakout candidates instantly from background cache.
    First call triggers a background fetch (1-year daily data, parallel).
    Subsequent calls return cached results immediately.

    Response includes `is_loading: true` while the first scan runs.
    """
    from breakout_scanner import get_breakout_stocks
    try:
        symbols = get_symbols()
        result = get_breakout_stocks(symbols)
        breakouts = result.get("breakouts", [])
        return {
            **result,
            "count": len(breakouts),
            "long_count": sum(1 for b in breakouts if b.get("direction") == "LONG"),
            "short_count": sum(1 for b in breakouts if b.get("direction") == "SHORT"),
        }
    except Exception as exc:
        logger.error("Breakout endpoint error: %s", exc)
        return {"error": str(exc), "breakouts": [], "count": 0, "long_count": 0, "short_count": 0}


@app.get("/sector-momentum/history", summary="Historical sector momentum for a given date", tags=["Market Data"])
def sector_momentum_history_endpoint(date: str = None) -> Dict[str, Any]:
    """
    Returns reconstructed sector momentum snapshots for a past date.
    Pass ?date=YYYY-MM-DD; defaults to yesterday when omitted.
    """
    try:
        from datetime import timedelta
        if not date:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        return get_historical_momentum(date)
    except Exception as exc:
        logger.error("Historical momentum endpoint error: %s", exc)
        return {"error": str(exc), "sectors": {}, "slots": [], "date": date or ""}


@app.get("/sector-momentum", summary="Sector momentum trend tracker (9:15–10:00 AM)", tags=["Market Data"])
def sector_momentum_endpoint() -> Dict[str, Any]:
    """
    Returns sector-wise avg change_pct snapshots taken every 5 minutes
    from 9:15 AM to 10:00 AM IST.

    Response includes per-sector trend (UP/DOWN/FLAT), momentum score,
    and top 3 LONG / SHORT sectors.
    """
    try:
        return get_sector_momentum_data()
    except Exception as exc:
        logger.error("Sector momentum endpoint error: %s", exc)
        return {"error": str(exc), "sectors": {}, "slots": []}


@app.get("/api/morning-watchlist/live", summary="Live morning watchlist (ORB + momentum + volume)", tags=["Market Data"])
def morning_watchlist_live_endpoint() -> Dict[str, Any]:
    """
    Returns today's morning watchlist using live sector momentum.
    Ranks stocks in the top-3 momentum sectors by ORB breakout, volume ratio,
    and sector rank score.
    """
    try:
        return get_live_watchlist()
    except Exception as exc:
        logger.error("Live watchlist endpoint error: %s", exc)
        return {
            "error": str(exc),
            "watchlist": [],
            "top_long": [],
            "top_short": [],
            "top_sectors": [],
        }


@app.get("/api/morning-watchlist", summary="Historical morning watchlist for a given date", tags=["Market Data"])
@app.get("/morning-watchlist")
def morning_watchlist_endpoint(date: str = None) -> Dict[str, Any]:
    try:
        from datetime import timedelta
        import pytz
        IST = pytz.timezone("Asia/Kolkata")
        today = datetime.now(IST).strftime("%Y-%m-%d")

        if not date:
            date = (datetime.now(IST) - timedelta(days=1)).strftime("%Y-%m-%d")

        if date == today:
            from scheduler import is_market_hours
            if not is_market_hours():
                date = (datetime.now(IST) - timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info("Market closed — using previous day: %s", date)

        return get_morning_watchlist(date)
    except Exception as exc:
        logger.error("Morning watchlist endpoint error: %s", exc)
        return {
            "error": str(exc),
            "watchlist": [],
            "stocks": [],
            "top_sectors": [],
            "date": date or "",
        }


# ---------------------------------------------------------------------------
# Feature 1: OI Analysis
# ---------------------------------------------------------------------------

@app.get("/api/oi/bulk", summary="Bulk OI analysis for multiple symbols", tags=["OI Analysis"])
@app.get("/oi/bulk", include_in_schema=False)
def oi_bulk_endpoint(symbols: str = "") -> Dict[str, Any]:
    """
    Returns OI analysis for multiple F&O symbols.
    Pass ?symbols=RELIANCE,TCS,NIFTY or leave blank for all F&O stocks.
    """
    from stocks import FO_STOCKS
    try:
        # Frontend may pass the sentinel string "ALL_FO_SYMBOLS" — treat it as empty
        cleaned = symbols.strip()
        if cleaned and cleaned.upper() != "ALL_FO_SYMBOLS":
            sym_list = [s.strip().upper() for s in cleaned.split(",") if s.strip()]
        else:
            # Use the curated default list (indices + top 30 high-OI stocks)
            # Fetching all 200+ FO symbols exceeds NSE's rate limit
            from oi_analysis import _DEFAULT_OI_SYMBOLS
            sym_list = _DEFAULT_OI_SYMBOLS
        result = get_bulk_oi(sym_list)

        # Split into nifty/banknifty index summaries and stock list
        nifty      = result.pop("NIFTY",      result.pop("nifty",      {}))
        banknifty  = result.pop("BANKNIFTY",  result.pop("banknifty",  {}))
        stocks     = list(result.values())
        fetched    = len(stocks) + (1 if nifty else 0) + (1 if banknifty else 0)

        return {
            "nifty":      nifty,
            "banknifty":  banknifty,
            "stocks":     stocks,
            "total":      len(sym_list),
            "fetched":    fetched,
        }
    except Exception as exc:
        logger.error("OI bulk endpoint error: %s", exc)
        return {"data": {}, "count": 0, "error": str(exc)}


@app.get("/api/oi/{symbol}", summary="OI analysis for a single symbol", tags=["OI Analysis"])
@app.get("/oi/{symbol}", include_in_schema=False)
def oi_single_endpoint(symbol: str) -> Dict[str, Any]:
    """Returns OI analysis for a single F&O stock or index (NIFTY/BANKNIFTY)."""
    try:
        result = get_oi_analysis(symbol.upper())
        if not result:
            raise HTTPException(status_code=404, detail=f"OI data not available for {symbol}")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("OI single endpoint error for %s: %s", symbol, exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Feature 2: Market Breadth Dashboard
# ---------------------------------------------------------------------------

@app.get("/api/breadth", summary="Market breadth — advances/declines/VWAP stats", tags=["Market Data"])
@app.get("/breadth", include_in_schema=False)
def breadth_endpoint() -> Dict[str, Any]:
    """
    Returns real-time market breadth computed from cached sector data.
    No new API calls — instant response.
    """
    cached = cache.get()
    if not cached:
        return _warming_up_response(
            advances=0,
            declines=0,
            unchanged=0,
            advance_decline_ratio=0.0,
            adr=0.0,
            pct_above_vwap=0.0,
            pct_positive=0.0,
            nifty50_breadth=0.0,
            breadth_signal="WARMING_UP",
            sector_breadth={},
            sector_breadth_list=[],
            total_stocks=0,
        )
    try:
        return get_market_breadth(cached.get("sectors", []))
    except Exception as exc:
        logger.error("Breadth endpoint error: %s", exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Feature 3: Relative Sector Strength vs Nifty
# ---------------------------------------------------------------------------

@app.get("/api/sector-relative-strength", summary="Sector strength relative to Nifty50", tags=["Market Data"])
@app.get("/sector-relative-strength", include_in_schema=False)
def sector_relative_strength_endpoint() -> Dict[str, Any]:
    """
    Returns each sector's change_pct and its performance relative to Nifty50.
    Classification: OUTPERFORMING (+0.5%), IN LINE, UNDERPERFORMING (-0.5%).
    """
    cached = cache.get()
    if not cached:
        return _warming_up_response(sectors=[], last_updated="")
    try:
        data = get_relative_sector_strength(cached.get("sectors", []))
        return {
            "sectors": data,
            "last_updated": cached.get("last_updated", ""),
        }
    except Exception as exc:
        logger.error("Relative strength endpoint error: %s", exc)
        return {"sectors": {}, "error": str(exc)}


# ---------------------------------------------------------------------------
# Feature 5: Intraday Trade Planner
# ---------------------------------------------------------------------------

@app.get("/api/trade-plan/bulk", summary="Trade plans for all F&O stocks", tags=["Trade Planner"])
@app.get("/trade-plan/bulk", include_in_schema=False)
def trade_plan_bulk_endpoint(direction: str = "") -> Dict[str, Any]:
    """
    Returns actionable trade plans (LONG and SHORT) for all F&O stocks.
    Optional ?direction=LONG or ?direction=SHORT to filter.
    Sorted by confidence (HIGH first) then R:R then RFactor descending.
    """
    sym_data = get_scanner_symbols()
    if not sym_data:
        return _warming_up_response(plans=[], count=0, long_count=0, short_count=0, last_updated="")
    try:
        plans = get_bulk_trade_plans(sym_data)
        if direction.upper() in ("LONG", "SHORT"):
            plans = [p for p in plans if p.get("direction") == direction.upper()]
        return {
            "plans": plans,
            "count": len(plans),
            "long_count":  sum(1 for p in plans if p.get("direction") == "LONG"),
            "short_count": sum(1 for p in plans if p.get("direction") == "SHORT"),
            "last_updated": cache.get().get("last_updated", "") if cache.get() else "",
        }
    except Exception as exc:
        logger.error("Trade plan bulk error: %s", exc)
        return {"plans": [], "count": 0, "error": str(exc)}


@app.get("/api/trade-plan/{symbol}", summary="Trade plan for a single stock", tags=["Trade Planner"])
@app.get("/trade-plan/{symbol}", include_in_schema=False)
def trade_plan_single_endpoint(symbol: str) -> Dict[str, Any]:
    """Returns a trade plan for a single stock symbol."""
    clean = symbol.upper().replace(".NS", "")
    sym_data = get_scanner_symbols()
    if not sym_data:
        return _warming_up_response(symbol=clean, strategy="WARMING_UP")
    stock = sym_data.get(clean)
    if not stock:
        raise HTTPException(status_code=404, detail=f"Symbol {clean} not found in cache")
    try:
        plan = get_trade_plan(clean, stock)
        if not plan:
            return {"symbol": clean, "strategy": "AVOID", "message": "Insufficient data or no setup"}
        return plan
    except Exception as exc:
        logger.error("Trade plan error for %s: %s", symbol, exc)
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Feature 6: 52-Week High Breakout Scanner
# ---------------------------------------------------------------------------

@app.get("/api/52w-breakouts", summary="52-week high institutional breakout scanner", tags=["Market Data"])
@app.get("/52w-breakouts", include_in_schema=False)
def breakouts_52w_endpoint() -> Dict[str, Any]:
    """
    Returns stocks within 0.5% of their 52-week high with 2x+ volume surge.
    Results are cached for 15 minutes. First call triggers background computation.
    """
    from breakout_scanner import scan_52w_breakouts
    try:
        return scan_52w_breakouts()
    except Exception as exc:
        logger.error("52W breakout endpoint error: %s", exc)
        return {"results": [], "count": 0, "is_loading": False, "error": str(exc)}


@app.get("/health", summary="Server health check", tags=["General"])
def health() -> Dict[str, Any]:
    """Returns server health information including cache status and stock counts."""
    data = cache.get()
    total_sectors = len(data["sectors"]) if data else 0
    total_stocks = (
        sum(len(s["stocks"]) for s in data["sectors"]) if data else 0
    )
    return {
        "status": "ok",
        "last_updated": cache.last_updated_str(),
        "market_open": is_market_hours(),
        "total_sectors": total_sectors,
        "total_stocks": total_stocks,
        "trade_guardian_poll_seconds": TRADE_GUARDIAN_POLL_SECONDS,
    }


# ---------------------------------------------------------------------------
# Trade Guardian
# ---------------------------------------------------------------------------

@app.get("/api/trade-guardian", summary="Trade Guardian summary", tags=["Trade Guardian"])
def trade_guardian_summary_endpoint(request: Request) -> Dict[str, Any]:
    try:
        return get_trade_guardian_summary(request.state.current_user)
    except Exception as exc:
        logger.error("Trade Guardian summary error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Trade Guardian summary failed") from exc


@app.get("/api/trade-guardian/trades", summary="List Trade Guardian trades", tags=["Trade Guardian"])
def trade_guardian_trades_endpoint(request: Request, include_closed: bool = False) -> Dict[str, Any]:
    try:
        return list_trades(request.state.current_user, include_closed=include_closed)
    except Exception as exc:
        logger.error("Trade Guardian trade list error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Trade Guardian trade list failed") from exc


@app.get("/api/trade-guardian/trades/{trade_id}", summary="Trade Guardian trade detail", tags=["Trade Guardian"])
def trade_guardian_trade_detail_endpoint(trade_id: str, request: Request) -> Dict[str, Any]:
    try:
        return get_trade_detail(request.state.current_user, trade_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Trade Guardian trade detail error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Trade Guardian trade detail failed") from exc


@app.post("/api/trade-guardian/trades", summary="Create Trade Guardian trade", tags=["Trade Guardian"])
def trade_guardian_create_trade_endpoint(payload: TradeGuardianCreateRequest, request: Request) -> Dict[str, Any]:
    try:
        return create_trade(request.state.current_user, payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Trade Guardian create trade error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Trade Guardian create trade failed") from exc


@app.post("/api/trade-guardian/trades/{trade_id}/close", summary="Close Trade Guardian trade", tags=["Trade Guardian"])
def trade_guardian_close_trade_endpoint(trade_id: str, payload: TradeGuardianCloseRequest, request: Request) -> Dict[str, Any]:
    try:
        return close_trade(request.state.current_user, trade_id, payload.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Trade Guardian close trade error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Trade Guardian close trade failed") from exc


@app.get("/api/trade-guardian/alerts", summary="List Trade Guardian alerts", tags=["Trade Guardian"])
def trade_guardian_alerts_endpoint(request: Request, include_resolved: bool = False) -> Dict[str, Any]:
    try:
        return list_alerts(request.state.current_user, include_resolved=include_resolved)
    except Exception as exc:
        logger.error("Trade Guardian alert list error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Trade Guardian alert list failed") from exc


@app.post("/api/trade-guardian/alerts/{alert_id}/acknowledge", summary="Acknowledge Trade Guardian alert", tags=["Trade Guardian"])
def trade_guardian_ack_alert_endpoint(alert_id: str, request: Request) -> Dict[str, Any]:
    try:
        return acknowledge_alert(request.state.current_user, alert_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Trade Guardian alert acknowledge error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Trade Guardian alert acknowledge failed") from exc


@app.post("/api/trade-guardian/monitor", summary="Run Trade Guardian monitor cycle", tags=["Trade Guardian"])
def trade_guardian_monitor_endpoint() -> Dict[str, Any]:
    try:
        return run_trade_guardian_monitor_cycle()
    except Exception as exc:
        logger.error("Trade Guardian monitor error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Trade Guardian monitor failed") from exc


@app.post("/api/trade-guardian/test-telegram", summary="Send Trade Guardian Telegram test", tags=["Trade Guardian"])
def trade_guardian_test_telegram_endpoint(payload: TradeGuardianTestAlertRequest | None = None, request: Request = None) -> Dict[str, Any]:
    try:
        if request is None:
            raise HTTPException(status_code=500, detail="Trade Guardian Telegram test failed")
        message_text = (payload.text if payload else "Trade Guardian test alert")
        return send_trade_guardian_test_alert(request.state.current_user, message_text)
    except Exception as exc:
        logger.error("Trade Guardian Telegram test failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Trade Guardian Telegram test failed") from exc


# ---------------------------------------------------------------------------
# F&O Trade Radar
# ---------------------------------------------------------------------------

@app.get("/api/fo-radar", summary="F&O Trade Radar — BUY/SELL/AVOID for every F&O stock", tags=["OI Analysis"])
@app.get("/fo-radar", include_in_schema=False)
def fo_radar_endpoint(
    signal: str = "ALL",
    min_confidence: int = 1,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Returns trade decisions (BUY/SELL/AVOID) for F&O stocks combining OI and
    price-action signals. Reads from a background cache refreshed every 5 minutes.

    Query params:
    - signal: ALL | BUY | SELL | AVOID
    - min_confidence: 1 (Low) | 2 (Medium) | 3 (High)
    - limit: max stocks to return (default 50)
    """
    stocks = get_fo_radar_snapshot()

    if not stocks:
        cached = cache.get()
        if not cached:
            return _warming_up_response(
                message="F&O Radar cache is warming up",
                stocks=[],
                total=0,
                buy_count=0,
                sell_count=0,
                avoid_count=0,
                last_updated=cache.last_updated_str(),
                cache_age_seconds=None,
                note="OI cache not yet populated — refreshes automatically each cycle.",
            )
        # Cache exists but radar hasn't run yet — trigger a quick price-action-only snapshot
        try:
            from stocks import FO_STOCKS
            # Build price-action-only entries from scanner_stocks (not heatmap sectors)
            stock_map: Dict[str, Any] = {}
            for s in cached.get("scanner_stocks", []):
                sym = s.get("symbol", "")
                if sym:
                    stock_map[sym] = s
            from oi_analysis import compute_fo_trade_signal
            clean_fo = [s.replace(".NS", "") for s in FO_STOCKS]
            stocks = [
                compute_fo_trade_signal({}, stock_map[sym])
                for sym in clean_fo if sym in stock_map
            ]
            stocks.sort(key=lambda x: (x["confidence"], abs(x.get("change_pct", 0))), reverse=True)
        except Exception as exc:
            logger.warning("fo-radar fallback build failed: %s", exc)
            return {
                "stocks": [],
                "total": 0,
                "buy_count": 0,
                "sell_count": 0,
                "avoid_count": 0,
                "last_updated": cache.last_updated_str(),
                "cache_age_seconds": None,
                "note": "OI cache not yet populated — refreshes automatically each cycle.",
            }

    # Apply filters
    sig_upper = signal.upper()
    if sig_upper != "ALL":
        stocks = [s for s in stocks if s["trade_signal"] == sig_upper]
    if min_confidence > 1:
        stocks = [s for s in stocks if s["confidence"] >= min_confidence]

    total_filtered = len(stocks)
    stocks = stocks[:limit]

    all_snap = get_fo_radar_snapshot() or stocks
    return {
        "stocks":        stocks,
        "total":         total_filtered,
        "buy_count":     sum(1 for s in all_snap if s["trade_signal"] == "BUY"),
        "sell_count":    sum(1 for s in all_snap if s["trade_signal"] == "SELL"),
        "avoid_count":   sum(1 for s in all_snap if s["trade_signal"] == "AVOID"),
        "last_updated":  cache.last_updated_str(),
        "cache_age_seconds": (lambda a: None if a == float("inf") else round(a))(fo_radar_cache_age_seconds()),
    }


@app.post("/api/fo-radar/refresh", summary="Trigger a background F&O Radar OI refresh", tags=["OI Analysis"])
async def fo_radar_refresh_endpoint() -> Dict[str, Any]:
    """
    Kick off a background refresh of the F&O Radar OI cache.
    Returns immediately — the refresh runs in a background thread.
    """
    cached = cache.get()
    if not cached:
        return _warming_up_response(status="warming_up", symbols=0)
    from stocks import FO_STOCKS
    fo_clean = [s.replace(".NS", "") for s in FO_STOCKS]
    # Use scanner_stocks for price data (not heatmap sectors)
    scanner_stocks_data = cached.get("scanner_stocks", [])
    loop = asyncio.get_running_loop()
    _radar_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fo-radar")
    loop.run_in_executor(
        _radar_executor,
        refresh_fo_radar_cache,
        fo_clean,
        scanner_stocks_data,
        True,
    )
    return {"status": "refresh_started", "symbols": len(fo_clean)}


# ---------------------------------------------------------------------------
# Direct run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,  # reload=True causes infinite restart loop on Windows (watchfiles respawns before startup completes)
    )
