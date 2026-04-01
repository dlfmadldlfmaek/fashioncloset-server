# services/recommend_history.py
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import firestore

from services.firestore import get_db  # 너가 만든 singleton get_db 사용
from services.geo import latlon_to_grid  # 있으면 사용, 없으면 nx/ny 인자로 받게 바꿔도 됨

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_weather(weather: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep weather shape stable for analysis.
    """
    return {
        "temp": float(weather.get("temp", 20.0)) if weather.get("temp") is not None else None,
        "feelsLike": weather.get("feelsLike"),
        "wind": weather.get("wind"),
        "pty": str(weather.get("pty", "SUNNY")),
    }


def _make_history_doc_id(
    *,
    date_str: str,
    time_slot: str,
    nx: int,
    ny: int,
) -> str:
    # ex) 2026-01-30:morning:60:127
    return f"{date_str}:{time_slot}:{nx}:{ny}"


def save_recommend_history(
    user_id: str,
    weather: Dict[str, Any],
    items: List[Dict[str, Any]],
    *,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    time_slot: str = "default",
    date_str: Optional[str] = None,
) -> str:
    """
    Save recommendation history.

    items: [{"id": "...", "finalScore": 1.23, "mainCategory": "...", "color": "...", ...}, ...]

    Returns:
      doc_id (so caller can update likedIds later).
    """
    if not user_id:
        raise ValueError("user_id is required")

    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if nx is None or ny is None:
        if lat is None or lon is None:
            raise ValueError("provide (nx, ny) or (lat, lon)")
        nx, ny = latlon_to_grid(float(lat), float(lon))

    norm_weather = _normalize_weather(weather)

    # Store as list objects (avoid huge map keys)
    norm_items: List[Dict[str, Any]] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        item_id = it.get("id")
        if not item_id:
            continue
        norm_items.append(
            {
                "id": str(item_id),
                "finalScore": float(it.get("finalScore", 0.0)),
                "score": float(it.get("score", 0.0)) if it.get("score") is not None else None,
                "mainCategory": it.get("mainCategory"),
                "color": it.get("color"),
                "season": it.get("season"),
                "tags": it.get("tags") or [],
            }
        )

    doc_id = _make_history_doc_id(date_str=date_str, time_slot=time_slot, nx=int(nx), ny=int(ny))
    doc = {
        "createdAt": firestore.SERVER_TIMESTAMP,
        "createdAtIso": _utc_now_iso(),
        "date": date_str,
        "timeSlot": time_slot,
        "nx": int(nx),
        "ny": int(ny),
        "weather": norm_weather,
        "items": norm_items,
        "itemIds": [it["id"] for it in norm_items],
        "likedIds": [],
        "version": 2,
    }

    db = get_db()
    ref = (
        db.collection("users")
        .document(user_id)
        .collection("recommend_history")
        .document(doc_id)
    )

    # merge=True: allow incremental schema evolution
    ref.set(doc, merge=True)
    logger.info("[HISTORY] saved user=%s doc=%s items=%s", user_id, doc_id, len(norm_items))
    return doc_id


def append_liked_to_history(user_id: str, doc_id: str, liked_ids: List[str]) -> None:
    """
    Append liked ids to a history document.
    """
    if not liked_ids:
        return
    db = get_db()
    ref = (
        db.collection("users")
        .document(user_id)
        .collection("recommend_history")
        .document(doc_id)
    )
    ref.update({"likedIds": firestore.ArrayUnion([str(x) for x in liked_ids if x])})
