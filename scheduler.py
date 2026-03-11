# scheduler.py — APScheduler background job for periodic market data refresh

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING

import os
os.environ["YFINANCE_CACHE"] = "/tmp/yfinance_cache"
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from backend.momentum_pulse import schedule_momentum_pulse_refresh
from fetcher import fetch_all_sectors

# Dedicated thread-pool so fetches never block the async event loop
_fetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mscope-fetch")

if TYPE_CHECKING:
    from cache import InMemoryCache

logger = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


def is_market_hours() -> bool:
    """
    Return True if the current IST time falls within NSE trading hours.
    Trading days: Monday–Friday (weekday 0–4).
    Trading hours: 09:00–15:35 IST (slight buffer around 09:15–15:30).
    """
    now_ist = datetime.now(IST)
    weekday = now_ist.weekday()  # 0 = Monday … 6 = Sunday
    if weekday > 4:
        return False
    market_start = now_ist.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = now_ist.replace(hour=15, minute=35, second=0, microsecond=0)
    return market_start <= now_ist <= market_end


async def scheduled_fetch(cache_obj: "InMemoryCache") -> None:
    """
    Async job executed by the scheduler every 5 minutes.
    Skips execution outside market hours.
    Never raises — all exceptions are caught and logged.

    fetch_all_sectors() is CPU/IO-heavy and completely synchronous, so it is
    offloaded to a thread-pool executor to avoid blocking the asyncio event loop.

    asyncio.shield() protects the running thread from being cancelled if APScheduler
    fires the next interval before this one finishes (e.g. slow yfinance response).
    CancelledError is a BaseException (not Exception), so it must be caught explicitly.
    """
    if not is_market_hours():
        logger.info("Market closed, skipping scheduled fetch.")
        return

    try:
        logger.info("Scheduled fetch started (background thread)...")
        loop = asyncio.get_event_loop()
        # shield() ensures the executor thread runs to completion even if this
        # coroutine is cancelled by APScheduler before the fetch finishes.
        data = await asyncio.shield(loop.run_in_executor(_fetch_executor, fetch_all_sectors))
        cache_obj.set(data)
        schedule_momentum_pulse_refresh(
            scanner_stocks=list(data.get("scanner_stocks", [])),
            last_updated=str(data.get("last_updated", "") or ""),
            force=True,
        )
        now_ist = datetime.now(IST).strftime("%H:%M:%S")
        logger.info(f"Scheduled fetch completed successfully at {now_ist} IST.")

        # Kick off F&O Radar OI refresh in background (sequential ~1.2s/symbol,
        # so we never await it — it runs independently in its own thread)
        try:
            from stocks import FO_STOCKS
            from oi_analysis import refresh_fo_radar_cache
            fo_clean = [s.replace(".NS", "") for s in FO_STOCKS]
            scanner_stocks = data.get("scanner_stocks", [])
            _radar_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fo-radar")
            loop.run_in_executor(_radar_executor, refresh_fo_radar_cache, fo_clean, scanner_stocks)
            logger.info("F&O Radar background refresh started (%d symbols).", len(fo_clean))
        except Exception as radar_exc:
            logger.warning("F&O Radar refresh could not start: %s", radar_exc)
    except asyncio.CancelledError:
        # Coroutine was cancelled (next interval fired or server shutting down).
        # shield() keeps the background thread alive; we just skip the cache update.
        logger.warning("Scheduled fetch coroutine cancelled — fetch may still complete in background.")
    except Exception as exc:
        logger.error(f"Scheduled fetch failed: {exc}", exc_info=True)


def start_scheduler(cache_obj: "InMemoryCache") -> AsyncIOScheduler:
    """
    Initialise and start the APScheduler with a 5-minute interval job.

    Args:
        cache_obj: The shared InMemoryCache instance to update after each fetch.

    Returns:
        The running AsyncIOScheduler instance.
    """
    scheduler = AsyncIOScheduler(timezone=IST)
    scheduler.add_job(
        scheduled_fetch,
        trigger="interval",
        minutes=5,
        args=[cache_obj],
        id="market_data_refresh",
        name="MarketScope 5-min data refresh",
        replace_existing=True,
        max_instances=1,      # never run two fetches simultaneously
        coalesce=True,        # if missed intervals, run once not multiple times
        misfire_grace_time=60, # allow 60s grace for slow jobs
    )
    scheduler.start()
    logger.info("Scheduler started — market data will refresh every 5 minutes.")
    return scheduler
