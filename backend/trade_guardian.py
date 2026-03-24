import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pytz

from backend.telegram_alerts import TelegramAlertError, send_telegram_message, telegram_is_configured
from upstox_client import get_bulk_full_quotes

logger = logging.getLogger("trade_guardian")

IST = pytz.timezone("Asia/Kolkata")
_DB_LOCK = threading.Lock()
_ROOT_DIR = Path(__file__).resolve().parent.parent
_DB_PATH = Path(os.getenv("TRADE_GUARDIAN_DB_PATH", str(_ROOT_DIR / "trade_guardian.db")))
_POLL_SECONDS = max(2, int(os.getenv("TRADE_GUARDIAN_POLL_SECONDS", "10")))
_REPEAT_SECONDS = max(30, int(os.getenv("TRADE_GUARDIAN_REPEAT_SECONDS", "60")))
_REPEATING_ALERT_TYPES = {"stop_loss_hit", "target_1_hit", "target_2_hit"}
_MONITORED_STATUSES = {"pending", "active", "t1_hit"}
_TERMINAL_STATUSES = {"t2_hit", "sl_hit", "closed_manual", "cancelled"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _now_ist() -> datetime:
    return datetime.now(IST)


def _now_iso() -> str:
    return _now_ist().isoformat()


def _user_scope(user: Dict[str, Any]) -> Tuple[str, str]:
    user_id = str(user.get("id") or user.get("email") or "").strip()
    email = str(user.get("email") or "").strip().lower()
    if not user_id:
        raise ValueError("Authenticated user id is required")
    return user_id, email


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace(".NS", "")


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(_DB_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def init_trade_guardian_storage() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _DB_LOCK:
        connection = _connect()
        try:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_guardian_trades (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    user_email TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    target_1 REAL NOT NULL,
                    target_2 REAL NOT NULL,
                    quantity REAL NOT NULL DEFAULT 0,
                    notes TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    telegram_chat_id TEXT NOT NULL DEFAULT '',
                    last_price REAL NOT NULL DEFAULT 0,
                    last_price_at TEXT NOT NULL DEFAULT '',
                    last_alert_message TEXT NOT NULL DEFAULT '',
                    activated_at TEXT NOT NULL DEFAULT '',
                    closed_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_guardian_alerts (
                    id TEXT PRIMARY KEY,
                    trade_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL,
                    repeat_every_seconds INTEGER NOT NULL DEFAULT 0,
                    repeat_count INTEGER NOT NULL DEFAULT 0,
                    first_triggered_at TEXT NOT NULL,
                    last_sent_at TEXT NOT NULL DEFAULT '',
                    acknowledged_at TEXT NOT NULL DEFAULT '',
                    resolved_at TEXT NOT NULL DEFAULT '',
                    snoozed_until TEXT NOT NULL DEFAULT '',
                    last_price REAL NOT NULL DEFAULT 0
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_guardian_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    last_price REAL NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_guardian_trades_user_status ON trade_guardian_trades(user_id, status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_guardian_alerts_trade_status ON trade_guardian_alerts(trade_id, status)")
            connection.commit()
        finally:
            connection.close()


def _row_to_dict(row: sqlite3.Row | Dict[str, Any] | None) -> Dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    return {key: row[key] for key in row.keys()}


def _format_trade_message(trade: Dict[str, Any], header: str, detail: str) -> str:
    return (
        f"{header}\n"
        f"Symbol: {trade.get('symbol')}\n"
        f"Side: {trade.get('direction')}\n"
        f"Entry: {_safe_float(trade.get('entry_price')):.2f}\n"
        f"SL: {_safe_float(trade.get('stop_loss')):.2f}\n"
        f"T1: {_safe_float(trade.get('target_1')):.2f}\n"
        f"T2: {_safe_float(trade.get('target_2')):.2f}\n"
        f"LTP: {_safe_float(trade.get('last_price')):.2f}\n"
        f"Status: {str(trade.get('status') or '').upper()}\n"
        f"{detail}"
    )


def _normalize_trade_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    symbol = _normalize_symbol(payload.get("symbol") or "")
    direction = str(payload.get("direction") or "").strip().upper()
    entry_price = _safe_float(payload.get("entry_price"))
    stop_loss = _safe_float(payload.get("stop_loss"))
    target_1 = _safe_float(payload.get("target_1"))
    target_2 = _safe_float(payload.get("target_2"))
    quantity = _safe_float(payload.get("quantity"))
    notes = str(payload.get("notes") or "").strip()
    telegram_chat_id = str(payload.get("telegram_chat_id") or os.getenv("TELEGRAM_CHAT_ID", "") or "").strip()

    if not symbol:
        raise ValueError("symbol is required")
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("direction must be LONG or SHORT")
    if min(entry_price, stop_loss, target_1, target_2) <= 0:
        raise ValueError("entry_price, stop_loss, target_1, and target_2 must be greater than 0")

    if direction == "LONG":
        if not (stop_loss < entry_price < target_1 <= target_2):
            raise ValueError("For LONG trades use stop_loss < entry_price < target_1 <= target_2")
    else:
        if not (stop_loss > entry_price > target_1 >= target_2):
            raise ValueError("For SHORT trades use stop_loss > entry_price > target_1 >= target_2")

    return {
        "symbol": symbol,
        "direction": direction,
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),
        "quantity": round(quantity, 2),
        "notes": notes,
        "telegram_chat_id": telegram_chat_id,
    }


def _fetch_trade(connection: sqlite3.Connection, trade_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    row = connection.execute(
        "SELECT * FROM trade_guardian_trades WHERE id = ? AND user_id = ? LIMIT 1",
        (trade_id, user_id),
    ).fetchone()
    return _row_to_dict(row) if row else None


def _serialize_trade(connection: sqlite3.Connection, trade: Dict[str, Any]) -> Dict[str, Any]:
    alert = connection.execute(
        """
        SELECT * FROM trade_guardian_alerts
        WHERE trade_id = ?
        ORDER BY first_triggered_at DESC, id DESC
        LIMIT 1
        """,
        (trade["id"],),
    ).fetchone()
    alert_payload = _row_to_dict(alert) if alert else None
    return {
        "id": trade["id"],
        "symbol": trade["symbol"],
        "direction": trade["direction"],
        "entry_price": round(_safe_float(trade["entry_price"]), 2),
        "stop_loss": round(_safe_float(trade["stop_loss"]), 2),
        "target_1": round(_safe_float(trade["target_1"]), 2),
        "target_2": round(_safe_float(trade["target_2"]), 2),
        "quantity": round(_safe_float(trade["quantity"]), 2),
        "notes": trade.get("notes") or "",
        "status": trade["status"],
        "last_price": round(_safe_float(trade.get("last_price")), 2),
        "last_price_at": trade.get("last_price_at") or "",
        "activated_at": trade.get("activated_at") or "",
        "closed_at": trade.get("closed_at") or "",
        "created_at": trade.get("created_at") or "",
        "updated_at": trade.get("updated_at") or "",
        "latest_alert": {
            "id": alert_payload.get("id"),
            "alert_type": alert_payload.get("alert_type"),
            "message": alert_payload.get("message"),
            "status": alert_payload.get("status"),
            "repeat_count": int(alert_payload.get("repeat_count") or 0),
            "last_sent_at": alert_payload.get("last_sent_at") or "",
            "acknowledged_at": alert_payload.get("acknowledged_at") or "",
        } if alert_payload else None,
    }


def _serialize_alert(alert: Dict[str, Any], trade: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "id": alert["id"],
        "trade_id": alert["trade_id"],
        "alert_type": alert["alert_type"],
        "message": alert["message"],
        "status": alert["status"],
        "repeat_every_seconds": int(alert.get("repeat_every_seconds") or 0),
        "repeat_count": int(alert.get("repeat_count") or 0),
        "first_triggered_at": alert.get("first_triggered_at") or "",
        "last_sent_at": alert.get("last_sent_at") or "",
        "acknowledged_at": alert.get("acknowledged_at") or "",
        "resolved_at": alert.get("resolved_at") or "",
        "last_price": round(_safe_float(alert.get("last_price")), 2),
    }
    if trade:
        payload["symbol"] = trade.get("symbol") or ""
        payload["direction"] = trade.get("direction") or ""
        payload["trade_status"] = trade.get("status") or ""
    return payload


def _record_event(
    connection: sqlite3.Connection,
    trade_id: str,
    user_id: str,
    event_type: str,
    message: str,
    last_price: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    connection.execute(
        """
        INSERT INTO trade_guardian_events(trade_id, user_id, event_type, message, event_time, last_price, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade_id,
            user_id,
            str(event_type or "").strip(),
            str(message or "").strip(),
            _now_iso(),
            round(_safe_float(last_price), 2),
            json.dumps(metadata or {}, separators=(",", ":")),
        ),
    )


def create_trade(user: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    init_trade_guardian_storage()
    user_id, user_email = _user_scope(user)
    clean = _normalize_trade_payload(payload)
    trade_id = uuid.uuid4().hex
    now_iso = _now_iso()
    with _DB_LOCK:
        connection = _connect()
        try:
            connection.execute(
                """
                INSERT INTO trade_guardian_trades(
                    id, user_id, user_email, symbol, direction, entry_price, stop_loss, target_1, target_2,
                    quantity, notes, status, telegram_chat_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id,
                    user_id,
                    user_email,
                    clean["symbol"],
                    clean["direction"],
                    clean["entry_price"],
                    clean["stop_loss"],
                    clean["target_1"],
                    clean["target_2"],
                    clean["quantity"],
                    clean["notes"],
                    "pending",
                    clean["telegram_chat_id"],
                    now_iso,
                    now_iso,
                ),
            )
            _record_event(
                connection,
                trade_id,
                user_id,
                "trade_created",
                f"Trade created for {clean['symbol']} {clean['direction']} at {clean['entry_price']:.2f}",
                metadata={
                    "entry_price": clean["entry_price"],
                    "stop_loss": clean["stop_loss"],
                    "target_1": clean["target_1"],
                    "target_2": clean["target_2"],
                },
            )
            trade = _fetch_trade(connection, trade_id, user_id)
            connection.commit()
            return _serialize_trade(connection, trade or {})
        finally:
            connection.close()


def list_trades(user: Dict[str, Any], include_closed: bool = False) -> Dict[str, Any]:
    init_trade_guardian_storage()
    user_id, _ = _user_scope(user)
    with _DB_LOCK:
        connection = _connect()
        try:
            if include_closed:
                rows = connection.execute(
                    "SELECT * FROM trade_guardian_trades WHERE user_id = ? ORDER BY updated_at DESC, created_at DESC",
                    (user_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM trade_guardian_trades WHERE user_id = ? AND status NOT IN ('closed_manual', 'cancelled') ORDER BY updated_at DESC, created_at DESC",
                    (user_id,),
                ).fetchall()
            trades = [_serialize_trade(connection, _row_to_dict(row)) for row in rows]
            return {
                "trades": trades,
                "total": len(trades),
                "monitor_poll_seconds": _POLL_SECONDS,
                "repeat_every_seconds": _REPEAT_SECONDS,
            }
        finally:
            connection.close()


def get_trade_detail(user: Dict[str, Any], trade_id: str) -> Dict[str, Any]:
    init_trade_guardian_storage()
    user_id, _ = _user_scope(user)
    with _DB_LOCK:
        connection = _connect()
        try:
            trade = _fetch_trade(connection, trade_id, user_id)
            if not trade:
                raise KeyError("Trade not found")
            events = connection.execute(
                "SELECT * FROM trade_guardian_events WHERE trade_id = ? ORDER BY event_time DESC, id DESC LIMIT 50",
                (trade_id,),
            ).fetchall()
            payload = _serialize_trade(connection, trade)
            payload["events"] = [
                {
                    "id": row["id"],
                    "event_type": row["event_type"],
                    "message": row["message"],
                    "event_time": row["event_time"],
                    "last_price": round(_safe_float(row["last_price"]), 2),
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                }
                for row in events
            ]
            return payload
        finally:
            connection.close()


def close_trade(user: Dict[str, Any], trade_id: str, reason: str = "closed_manual") -> Dict[str, Any]:
    init_trade_guardian_storage()
    user_id, _ = _user_scope(user)
    now_iso = _now_iso()
    with _DB_LOCK:
        connection = _connect()
        try:
            trade = _fetch_trade(connection, trade_id, user_id)
            if not trade:
                raise KeyError("Trade not found")
            terminal_status = "cancelled" if trade["status"] == "pending" else "closed_manual"
            connection.execute(
                "UPDATE trade_guardian_trades SET status = ?, closed_at = ?, updated_at = ? WHERE id = ?",
                (terminal_status, now_iso, now_iso, trade_id),
            )
            connection.execute(
                "UPDATE trade_guardian_alerts SET status = 'resolved', resolved_at = ? WHERE trade_id = ? AND status = 'active'",
                (now_iso, trade_id),
            )
            _record_event(
                connection,
                trade_id,
                user_id,
                terminal_status,
                f"Trade closed manually: {reason or terminal_status}",
                last_price=_safe_float(trade.get("last_price")),
            )
            updated = _fetch_trade(connection, trade_id, user_id)
            connection.commit()
            return _serialize_trade(connection, updated or {})
        finally:
            connection.close()


def list_alerts(user: Dict[str, Any], include_resolved: bool = False) -> Dict[str, Any]:
    init_trade_guardian_storage()
    user_id, _ = _user_scope(user)
    with _DB_LOCK:
        connection = _connect()
        try:
            if include_resolved:
                rows = connection.execute(
                    "SELECT * FROM trade_guardian_alerts WHERE user_id = ? ORDER BY first_triggered_at DESC, id DESC LIMIT 100",
                    (user_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM trade_guardian_alerts WHERE user_id = ? AND status = 'active' ORDER BY first_triggered_at DESC, id DESC LIMIT 100",
                    (user_id,),
                ).fetchall()

            alerts: List[Dict[str, Any]] = []
            for row in rows:
                alert = _row_to_dict(row)
                trade = _fetch_trade(connection, alert["trade_id"], user_id)
                alerts.append(_serialize_alert(alert, trade))
            return {
                "alerts": alerts,
                "total": len(alerts),
                "repeat_every_seconds": _REPEAT_SECONDS,
            }
        finally:
            connection.close()


def acknowledge_alert(user: Dict[str, Any], alert_id: str) -> Dict[str, Any]:
    init_trade_guardian_storage()
    user_id, _ = _user_scope(user)
    now_iso = _now_iso()
    with _DB_LOCK:
        connection = _connect()
        try:
            row = connection.execute(
                "SELECT * FROM trade_guardian_alerts WHERE id = ? AND user_id = ? LIMIT 1",
                (alert_id, user_id),
            ).fetchone()
            if not row:
                raise KeyError("Alert not found")
            alert = _row_to_dict(row)
            connection.execute(
                "UPDATE trade_guardian_alerts SET status = 'acknowledged', acknowledged_at = ? WHERE id = ?",
                (now_iso, alert_id),
            )
            trade = _fetch_trade(connection, alert["trade_id"], user_id)
            if trade:
                _record_event(
                    connection,
                    trade["id"],
                    user_id,
                    "alert_acknowledged",
                    f"Alert acknowledged: {alert['alert_type']}",
                    last_price=_safe_float(alert.get("last_price")),
                )
            updated = connection.execute(
                "SELECT * FROM trade_guardian_alerts WHERE id = ? LIMIT 1",
                (alert_id,),
            ).fetchone()
            connection.commit()
            return _serialize_alert(_row_to_dict(updated), trade)
        finally:
            connection.close()


def get_trade_guardian_summary(user: Dict[str, Any]) -> Dict[str, Any]:
    trades_payload = list_trades(user, include_closed=False)
    alerts_payload = list_alerts(user, include_resolved=False)
    return {
        "status": "ok",
        "trades": trades_payload["trades"],
        "alerts": alerts_payload["alerts"],
        "trade_count": trades_payload["total"],
        "alert_count": alerts_payload["total"],
        "monitor_poll_seconds": _POLL_SECONDS,
        "repeat_every_seconds": _REPEAT_SECONDS,
        "telegram_configured": telegram_is_configured(),
    }


def _resolve_trade_state(trade: Dict[str, Any], price: float) -> List[Tuple[str, str, str]]:
    direction = str(trade.get("direction") or "").upper()
    status = str(trade.get("status") or "pending").lower()
    entry_price = _safe_float(trade.get("entry_price"))
    stop_loss = _safe_float(trade.get("stop_loss"))
    target_1 = _safe_float(trade.get("target_1"))
    target_2 = _safe_float(trade.get("target_2"))

    events: List[Tuple[str, str, str]] = []

    if status == "pending":
        if (direction == "LONG" and price >= entry_price) or (direction == "SHORT" and price <= entry_price):
            events.append(("entry_triggered", "active", f"Entry triggered for {trade['symbol']} at {price:.2f}"))
            status = "active"

    if status in {"active", "t1_hit"}:
        if direction == "LONG":
            if price <= stop_loss:
                events.append(("stop_loss_hit", "sl_hit", f"Stop loss hit for {trade['symbol']} at {price:.2f}"))
            elif price >= target_2:
                events.append(("target_2_hit", "t2_hit", f"Target 2 hit for {trade['symbol']} at {price:.2f}"))
            elif status == "active" and price >= target_1:
                events.append(("target_1_hit", "t1_hit", f"Target 1 hit for {trade['symbol']} at {price:.2f}"))
        elif direction == "SHORT":
            if price >= stop_loss:
                events.append(("stop_loss_hit", "sl_hit", f"Stop loss hit for {trade['symbol']} at {price:.2f}"))
            elif price <= target_2:
                events.append(("target_2_hit", "t2_hit", f"Target 2 hit for {trade['symbol']} at {price:.2f}"))
            elif status == "active" and price <= target_1:
                events.append(("target_1_hit", "t1_hit", f"Target 1 hit for {trade['symbol']} at {price:.2f}"))

    return events


def _resolve_prior_alerts(connection: sqlite3.Connection, trade_id: str, except_alert_type: str = "") -> None:
    now_iso = _now_iso()
    if except_alert_type:
        connection.execute(
            "UPDATE trade_guardian_alerts SET status = 'resolved', resolved_at = ? WHERE trade_id = ? AND status = 'active' AND alert_type != ?",
            (now_iso, trade_id, except_alert_type),
        )
    else:
        connection.execute(
            "UPDATE trade_guardian_alerts SET status = 'resolved', resolved_at = ? WHERE trade_id = ? AND status = 'active'",
            (now_iso, trade_id),
        )


def _ensure_alert(connection: sqlite3.Connection, trade: Dict[str, Any], alert_type: str, message: str, last_price: float) -> None:
    existing = connection.execute(
        "SELECT * FROM trade_guardian_alerts WHERE trade_id = ? AND alert_type = ? AND status = 'active' LIMIT 1",
        (trade["id"], alert_type),
    ).fetchone()
    repeat_every_seconds = _REPEAT_SECONDS if alert_type in _REPEATING_ALERT_TYPES else 0
    now_iso = _now_iso()
    if existing:
        connection.execute(
            "UPDATE trade_guardian_alerts SET message = ?, last_price = ? WHERE id = ?",
            (message, round(last_price, 2), existing["id"]),
        )
        return

    if alert_type in {"target_2_hit", "stop_loss_hit"}:
        _resolve_prior_alerts(connection, trade["id"], alert_type)

    alert_id = uuid.uuid4().hex
    connection.execute(
        """
        INSERT INTO trade_guardian_alerts(
            id, trade_id, user_id, alert_type, message, status, repeat_every_seconds,
            first_triggered_at, last_price
        ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)
        """,
        (
            alert_id,
            trade["id"],
            trade["user_id"],
            alert_type,
            message,
            repeat_every_seconds,
            now_iso,
            round(last_price, 2),
        ),
    )


def _send_due_alerts(connection: sqlite3.Connection, limit: int = 50) -> int:
    now_ts = time.time()
    sent = 0
    rows = connection.execute(
        "SELECT * FROM trade_guardian_alerts WHERE status = 'active' ORDER BY first_triggered_at ASC, id ASC LIMIT ?",
        (limit,),
    ).fetchall()
    for row in rows:
        alert = _row_to_dict(row)
        last_sent_at = str(alert.get("last_sent_at") or "").strip()
        acknowledged_at = str(alert.get("acknowledged_at") or "").strip()
        resolved_at = str(alert.get("resolved_at") or "").strip()
        snoozed_until = str(alert.get("snoozed_until") or "").strip()
        repeat_every_seconds = int(alert.get("repeat_every_seconds") or 0)

        if acknowledged_at or resolved_at:
            continue
        if snoozed_until:
            try:
                if datetime.fromisoformat(snoozed_until).timestamp() > now_ts:
                    continue
            except Exception:
                pass

        if last_sent_at:
            try:
                last_sent_ts = datetime.fromisoformat(last_sent_at).timestamp()
            except Exception:
                last_sent_ts = 0.0
            if repeat_every_seconds <= 0 or (now_ts - last_sent_ts) < repeat_every_seconds:
                continue

        trade_row = connection.execute(
            "SELECT * FROM trade_guardian_trades WHERE id = ? LIMIT 1",
            (alert["trade_id"],),
        ).fetchone()
        trade = _row_to_dict(trade_row) if trade_row else {}
        if not trade:
            connection.execute(
                "UPDATE trade_guardian_alerts SET status = 'resolved', resolved_at = ? WHERE id = ?",
                (_now_iso(), alert["id"]),
            )
            continue

        message = str(alert.get("message") or "").strip()
        if alert["alert_type"] in _REPEATING_ALERT_TYPES and last_sent_at:
            message = _format_trade_message(
                trade,
                header=f"Repeat alert: {alert['alert_type'].replace('_', ' ').upper()}",
                detail=message,
            )
        elif not message.startswith("Trade alert"):
            message = _format_trade_message(
                trade,
                header=f"Trade alert: {alert['alert_type'].replace('_', ' ').upper()}",
                detail=message,
            )

        try:
            send_telegram_message(message, chat_id=str(trade.get("telegram_chat_id") or "").strip() or None)
            sent += 1
            connection.execute(
                "UPDATE trade_guardian_alerts SET last_sent_at = ?, repeat_count = repeat_count + 1 WHERE id = ?",
                (_now_iso(), alert["id"]),
            )
            connection.execute(
                "UPDATE trade_guardian_trades SET last_alert_message = ?, updated_at = ? WHERE id = ?",
                (str(alert.get("message") or "").strip(), _now_iso(), trade["id"]),
            )
        except TelegramAlertError as exc:
            logger.warning("Trade Guardian Telegram delivery failed for %s: %s", trade.get("symbol"), exc)
        except Exception as exc:
            logger.warning("Trade Guardian alert dispatch failed for %s: %s", trade.get("symbol"), exc)

    return sent


def run_trade_guardian_monitor_cycle() -> Dict[str, Any]:
    init_trade_guardian_storage()
    with _DB_LOCK:
        connection = _connect()
        try:
            active_rows = connection.execute(
                "SELECT * FROM trade_guardian_trades WHERE status IN ('pending', 'active', 't1_hit') ORDER BY created_at ASC"
            ).fetchall()
            trades = [_row_to_dict(row) for row in active_rows]
            symbols = sorted({str(trade.get("symbol") or "").strip().upper() for trade in trades if str(trade.get("symbol") or "").strip()})
            quotes = get_bulk_full_quotes(symbols) if symbols else {}
            transitions = 0

            for trade in trades:
                symbol = str(trade.get("symbol") or "").strip().upper()
                quote = quotes.get(symbol) or {}
                price = _safe_float(quote.get("ltp"), _safe_float(trade.get("last_price")))
                if price <= 0:
                    continue

                connection.execute(
                    "UPDATE trade_guardian_trades SET last_price = ?, last_price_at = ?, updated_at = ? WHERE id = ?",
                    (round(price, 2), _now_iso(), _now_iso(), trade["id"]),
                )
                trade["last_price"] = round(price, 2)

                events = _resolve_trade_state(trade, price)
                for alert_type, new_status, message in events:
                    transitions += 1
                    status_changed = str(trade.get("status") or "") != new_status
                    update_values: List[Any] = [new_status, _now_iso(), trade["id"]]
                    update_sql = "UPDATE trade_guardian_trades SET status = ?, updated_at = ?"
                    if new_status == "active" and not str(trade.get("activated_at") or ""):
                        update_sql += ", activated_at = ?"
                        update_values = [new_status, _now_iso(), _now_iso(), trade["id"]]
                    elif new_status in _TERMINAL_STATUSES:
                        update_sql += ", closed_at = ?"
                        update_values = [new_status, _now_iso(), _now_iso(), trade["id"]]
                    update_sql += " WHERE id = ?"
                    connection.execute(update_sql, tuple(update_values))
                    trade["status"] = new_status
                    _record_event(connection, trade["id"], trade["user_id"], alert_type, message, last_price=price)
                    _ensure_alert(connection, trade, alert_type, message, last_price=price)
                    if status_changed and new_status in _TERMINAL_STATUSES:
                        _resolve_prior_alerts(connection, trade["id"], alert_type)

            sent_alerts = _send_due_alerts(connection)
            connection.commit()
            return {
                "status": "ok",
                "polled_symbols": len(symbols),
                "tracked_trades": len(trades),
                "transitions": transitions,
                "alerts_sent": sent_alerts,
                "monitor_poll_seconds": _POLL_SECONDS,
            }
        finally:
            connection.close()


def send_trade_guardian_test_alert(user: Dict[str, Any], text: str = "Trade Guardian test alert") -> Dict[str, Any]:
    user_id, _ = _user_scope(user)
    if not telegram_is_configured():
        raise RuntimeError("Telegram is not configured")
    message = f"Trade Guardian test\nUser: {user.get('email') or user_id}\n{text}"
    send_telegram_message(message)
    return {
        "status": "sent",
        "message": message,
    }