# cache.py — In-memory cache for MarketScope heatmap data

import time
import pytz
from datetime import datetime, timezone
from typing import Any, Optional

from runtime_state import load_json_state, save_json_state

IST = pytz.timezone("Asia/Kolkata")
_CACHE_STATE_FILE = "market_cache.json"


class InMemoryCache:
    """Thread-safe in-memory cache with staleness tracking.

    NOTE: Market data is intentionally NOT persisted to disk.
    Persisting 200+ stocks of OHLC + metadata causes large JSON
    serialization on every cache.set() call, doubling RAM usage
    and causing memory spikes on Render's 512MB instance.

    Only `updated_at` is persisted so the app knows on restart
    how stale its last data was — it will re-fetch fresh data anyway.
    """

    def __init__(self) -> None:
        persisted = load_json_state(_CACHE_STATE_FILE, {})
        # Data is never loaded from disk — always starts cold (None).
        # Fresh fetch happens via _bg_init_fetch() in lifespan.
        self.data: Optional[Any] = None
        self.updated_at: float = float(persisted.get("updated_at") or 0)

    def set(self, data: Any) -> None:
        """Store data in memory and persist only the timestamp to disk."""
        self.data = data
        self.updated_at = time.time()
        # FIX: Only save the timestamp — NOT the full market data payload.
        # Saving data caused double-RAM usage during JSON serialization.
        save_json_state(
            _CACHE_STATE_FILE,
            {"updated_at": self.updated_at},
        )

    def get(self) -> Optional[Any]:
        """Return the currently cached data (may be None)."""
        return self.data

    def is_stale(self, seconds: int) -> bool:
        """Return True if the cache has not been updated within `seconds`."""
        return (time.time() - self.updated_at) > seconds

    def last_updated_str(self) -> str:
        """Return the last-updated time as an IST-formatted HH:MM:SS string, or 'Never'."""
        if self.updated_at == 0:
            return "Never"
        utc_dt = datetime.fromtimestamp(self.updated_at, timezone.utc)
        ist_dt = utc_dt.astimezone(IST)
        return ist_dt.strftime("%H:%M:%S")