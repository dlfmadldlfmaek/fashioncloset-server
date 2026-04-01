# services/premium.py
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from services.firestore import get_db

logger = logging.getLogger(__name__)

# Optional TTL cache for hot paths
_PREMIUM_CACHE_TTL_SEC = int(os.getenv("PREMIUM_CACHE_TTL_SEC", "300"))  # 5 min default
_premium_cache: dict[str, tuple[float, bool]] = {}


def is_premium_user(user_id: str) -> bool:
    """
    Return whether the user is premium.

    Notes:
      - Uses Firestore users/{userId}.isPremium
      - Applies a small TTL cache to reduce read load in hot paths.
    """
    if not user_id:
        return False

    now = time.time()
    cached = _premium_cache.get(user_id)
    if cached:
        ts, val = cached
        if now - ts < _PREMIUM_CACHE_TTL_SEC:
            return val

    try:
        db = get_db()
        # Firestore Python doesn't support projection in all versions consistently,
        # so we read doc and pick the field.
        doc = db.collection("users").document(user_id).get()
        if not doc.exists:
            _premium_cache[user_id] = (now, False)
            return False

        data = doc.to_dict() or {}
        val = bool(data.get("isPremium", False))
        _premium_cache[user_id] = (now, val)
        return val
    except Exception:
        logger.exception("is_premium_user failed user_id=%s", user_id)
        # Fail closed: treat as non-premium on errors
        return False
