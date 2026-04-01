# services/auth.py
from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Request

logger = logging.getLogger("auth")


def _ensure_firebase_app() -> None:
    """Initialize firebase_admin app if not already initialized."""
    import firebase_admin  # type: ignore

    if not firebase_admin._apps:
        firebase_admin.initialize_app()


async def verify_firebase_token(request: Request) -> dict:
    """
    FastAPI dependency that extracts and verifies a Firebase Auth ID token
    from the Authorization header.

    Returns the decoded token dict (contains 'uid', 'email', etc.).
    Raises HTTP 401 if the token is missing or invalid.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or invalid Authorization header")

    token_str = auth_header[len("Bearer "):]
    if not token_str:
        raise HTTPException(status_code=401, detail="missing token")

    try:
        _ensure_firebase_app()

        from firebase_admin import auth  # type: ignore

        decoded = auth.verify_id_token(token_str)
        return decoded
    except Exception as e:
        logger.warning("[AUTH] token verification failed: %s", e)
        raise HTTPException(status_code=401, detail="invalid or expired token")


def get_current_user_id(token: dict) -> str:
    """Extract uid from a decoded Firebase token."""
    uid = token.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="uid not found in token")
    return uid
