import logging
import os
from typing import Any, Dict

import httpx

logger = logging.getLogger("telegram_alerts")

_TELEGRAM_API_BASE = "https://api.telegram.org"
_TIMEOUT_SECONDS = float(os.getenv("TELEGRAM_TIMEOUT_SECONDS", "10"))


class TelegramAlertError(RuntimeError):
    pass


def get_telegram_config() -> Dict[str, str]:
    return {
        "bot_token": str(os.getenv("TELEGRAM_BOT_TOKEN", "") or "").strip(),
        "chat_id": str(os.getenv("TELEGRAM_CHAT_ID", "") or "").strip(),
    }


def telegram_is_configured() -> bool:
    config = get_telegram_config()
    return bool(config["bot_token"] and config["chat_id"])


def _build_url(bot_token: str, method: str) -> str:
    return f"{_TELEGRAM_API_BASE}/bot{bot_token}/{method}"


def send_telegram_message(
    text: str,
    disable_notification: bool = False,
    chat_id: str | None = None,
    bot_token: str | None = None,
) -> Dict[str, Any]:
    config = get_telegram_config()
    resolved_bot_token = str(bot_token or config["bot_token"] or "").strip()
    resolved_chat_id = str(chat_id or config["chat_id"] or "").strip()
    if not resolved_bot_token:
        raise TelegramAlertError("TELEGRAM_BOT_TOKEN is not configured")
    if not resolved_chat_id:
        raise TelegramAlertError("TELEGRAM_CHAT_ID is not configured")

    payload = {
        "chat_id": resolved_chat_id,
        "text": str(text or "").strip(),
        "disable_notification": bool(disable_notification),
    }
    if not payload["text"]:
        raise TelegramAlertError("Telegram message text is empty")

    response = httpx.post(_build_url(resolved_bot_token, "sendMessage"), json=payload, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not data.get("ok"):
        raise TelegramAlertError(f"Telegram sendMessage failed: {data}")
    return data


def fetch_telegram_updates(bot_token: str | None = None) -> Dict[str, Any]:
    token = str(bot_token or get_telegram_config()["bot_token"] or "").strip()
    if not token:
        raise TelegramAlertError("TELEGRAM_BOT_TOKEN is not configured")

    response = httpx.get(_build_url(token, "getUpdates"), timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not data.get("ok"):
        raise TelegramAlertError(f"Telegram getUpdates failed: {data}")
    return data
