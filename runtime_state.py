import json
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

_STATE_DIR = os.getenv("MARKETSCOPE_STATE_DIR", ".runtime_state")


def _state_file_path(filename: str) -> str:
    os.makedirs(_STATE_DIR, exist_ok=True)
    return os.path.join(_STATE_DIR, filename)


def load_json_state(filename: str, default: Any) -> Any:
    path = _state_file_path(filename)
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        logger.warning("Failed to load runtime state from %s: %s", path, exc)
        return default


def save_json_state(filename: str, payload: Any) -> None:
    path = _state_file_path(filename)
    directory = os.path.dirname(path)

    try:
        with tempfile.NamedTemporaryFile("w", delete=False, dir=directory, encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
            temp_path = handle.name
        os.replace(temp_path, path)
    except Exception as exc:
        logger.warning("Failed to save runtime state to %s: %s", path, exc)
        try:
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except OSError:
            pass