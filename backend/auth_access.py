import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException, status


SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or ""
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""


def _require_config() -> None:
    if SUPABASE_URL and SUPABASE_ANON_KEY and SUPABASE_SERVICE_ROLE_KEY:
        return
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Authentication service is not configured",
    )


def _extract_bearer_token(authorization_header: str) -> str:
    raw = str(authorization_header or "").strip()
    prefix = "Bearer "
    if not raw.startswith(prefix):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    token = raw[len(prefix):].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return token


async def _fetch_user(access_token: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {access_token}",
            },
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login session",
        )
    user = response.json()
    if not isinstance(user, dict):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login session",
        )
    email = str(user.get("email") or "").strip().lower()
    if not email:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email address is required for access",
        )
    return user


async def _fetch_allowed_access(email: str) -> Optional[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/allowed_access",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            },
            params={
                "select": "email,is_enabled,access_expires_at,plan_name,notes",
                "email": f"eq.{email}",
                "limit": "1",
            },
        )
    if response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Access control lookup failed",
        )
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        return None
    entry = payload[0]
    return entry if isinstance(entry, dict) else None


def _parse_timestamp(raw_value: Any) -> Optional[datetime]:
    value = str(raw_value or "").strip()
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Invalid access expiry configuration",
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


async def authorize_request(authorization_header: str) -> Dict[str, Any]:
    _require_config()
    access_token = _extract_bearer_token(authorization_header)
    user = await _fetch_user(access_token)
    email = str(user.get("email") or "").strip().lower()
    access_entry = await _fetch_allowed_access(email)
    if access_entry is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access not enabled for this account",
        )

    if not bool(access_entry.get("is_enabled", False)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access not enabled for this account",
        )

    expires_at = _parse_timestamp(access_entry.get("access_expires_at"))
    if expires_at is not None and expires_at <= datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Membership expired",
        )

    return {
        "id": user.get("id"),
        "email": email,
        "plan_name": access_entry.get("plan_name") or "",
        "access_expires_at": access_entry.get("access_expires_at") or None,
        "notes": access_entry.get("notes") or "",
    }