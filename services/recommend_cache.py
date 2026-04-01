# services/recommend_cache.py
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL_SEC = 20 * 60
MAX_ENTRIES = 1000

_LOCK = threading.Lock()


@dataclass
class CacheEntry:
    data: Dict[str, Any]
    expires_at: float


# LRU (oldest first)
_CACHE: "OrderedDict[str, CacheEntry]" = OrderedDict()

_hits = 0
_misses = 0
_evicted = 0


def _now() -> float:
    return time.time()


def _prune_expired(now: float) -> None:
    # OrderedDict: scan from oldest; stop once not expired
    keys_to_delete = []
    for k, entry in _CACHE.items():
        if entry.expires_at <= now:
            keys_to_delete.append(k)
        else:
            break
    for k in keys_to_delete:
        _CACHE.pop(k, None)


def _evict_if_needed() -> None:
    global _evicted
    while len(_CACHE) > MAX_ENTRIES:
        _CACHE.popitem(last=False)  # evict oldest
        _evicted += 1


def clothes_hash(clothes: list[dict]) -> str:
    """
    Stable hash for clothes list.

    Why: avoid cache busting on volatile fields (e.g. lastWornAt, updatedAt).
    """
    def pick(c: dict) -> dict:
        return {
            "id": c.get("id"),
            "category": c.get("category") or c.get("mainCategory"),
            "tags": c.get("tags") or [],
            "season": c.get("season"),
            "color": c.get("color"),
            "thickness": c.get("thickness"),
        }

    normalized = sorted((pick(c) for c in (clothes or [])), key=lambda x: str(x.get("id") or ""))
    raw = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_cache_key(
    user_id: str,
    nx: int,
    ny: int,
    time_slot: str,
    clothes_hash_key: str,
) -> str:
    return f"{user_id}:{nx}:{ny}:{time_slot}:{clothes_hash_key}"


def get_cached_recommend(key: str) -> Optional[Dict[str, Any]]:
    global _hits, _misses
    now = _now()
    with _LOCK:
        _prune_expired(now)
        entry = _CACHE.get(key)
        if entry is None:
            _misses += 1
            return None

        if entry.expires_at <= now:
            _CACHE.pop(key, None)
            _misses += 1
            return None

        # touch LRU
        _CACHE.move_to_end(key, last=True)
        _hits += 1
        logger.debug("[CACHE HIT] key=%s hits=%s misses=%s size=%s", key, _hits, _misses, len(_CACHE))
        return entry.data


def set_cached_recommend(key: str, data: Dict[str, Any], minutes: int = 20) -> None:
    ttl_sec = max(1, int(minutes * 60))
    expires_at = _now() + ttl_sec
    with _LOCK:
        _CACHE[key] = CacheEntry(data=data, expires_at=expires_at)
        _CACHE.move_to_end(key, last=True)
        _prune_expired(_now())
        _evict_if_needed()
        logger.debug("[CACHE SET] key=%s ttl=%ss size=%s", key, ttl_sec, len(_CACHE))


def cache_stats() -> Dict[str, int]:
    with _LOCK:
        return {
            "size": len(_CACHE),
            "hits": _hits,
            "misses": _misses,
            "evicted": _evicted,
        }
