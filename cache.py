# cache.py — In-memory cache for MarketScope heatmap data

import time
import pytz
from datetime import datetime, timezone
from typing import Any, Optional

from runtime_state import load_json_state, save_json_state

IST = pytz.timezone("Asia/Kolkata")
_CACHE_STATE_FILE = "market_cache.json"


class InMemoryCache:
    """Thread-safe in-memory cache with staleness tracking."""

    def __init__(self) -> None:
        # Only restore the timestamp — NOT the full market data.
        # Loading full market data on startup wastes RAM with stale data
        # that will be replaced on the first fetch anyway.
        persisted = load_json_state(_CACHE_STATE_FILE, {})
        self.data: Optional[Any] = None  # always start fresh
        self.updated_at: float = float(persisted.get("updated_at") or 0)

    def set(self, data: Any) -> None:
        """Store data and record the current timestamp."""
        self.data = data
        self.updated_at = time.time()
        # Only persist the timestamp — NOT the full market data dict.
        # Serializing hundreds of stocks on every refresh cycle creates
        # a second full copy of the data in RAM during JSON serialization.
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