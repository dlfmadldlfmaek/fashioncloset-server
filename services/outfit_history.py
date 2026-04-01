# services/outfit_history.py
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from google.cloud import firestore

logger = logging.getLogger(__name__)

_DB: Optional[firestore.Client] = None
_DB_LOCK = threading.Lock()


def get_db() -> firestore.Client:
    """
    Singleton Firestore client per process.

    Why: per-call Client() in Cloud Run adds overhead under concurrency.
    """
    global _DB
    if _DB is not None:
        return _DB
    with _DB_LOCK:
        if _DB is None:
            _DB = firestore.Client()
    return _DB


def make_outfit_hash(item_ids: Iterable[str]) -> str:
    """
    Order-insensitive outfit key.

    - Deduplicates IDs to avoid ["A","A","B"] != ["A","B"]
    - Escapes delimiter to be safe even if IDs contain '|'
    """
    ids = [str(x) for x in (item_ids or []) if x is not None and str(x).strip()]
    if not ids:
        return ""
    uniq = sorted(set(ids))
    escaped = [s.replace("|", "%7C") for s in uniq]
    return "|".join(escaped)


def _docid(date_dt: datetime) -> str:
    return date_dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _as_dt(doc_id: str) -> Optional[datetime]:
    try:
        return datetime.strptime(doc_id, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def load_recent_outfits(user_id: str, days: int = 7) -> Dict[str, datetime]:
    """
    Load outfit combinations worn in recent N days.

    Returns:
      { "A|B": last_worn_datetime_utc, ... }
    """
    if not user_id:
        raise ValueError("user_id is required")
    if days <= 0:
        return {}

    db = get_db()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    since_str = _docid(since)
    until_str = _docid(now + timedelta(days=1))  # exclusive upper bound

    ref = (
        db.collection("users")
        .document(user_id)
        .collection("outfits")
        .where(firestore.FieldPath.document_id(), ">=", since_str)
        .where(firestore.FieldPath.document_id(), "<", until_str)
    )

    recent_pairs: Dict[str, datetime] = {}

    for doc in ref.stream():
        data = doc.to_dict() or {}
        clothes_ids = data.get("clothesIds") or []
        if not clothes_ids:
            continue

        pair_key = make_outfit_hash(clothes_ids)
        if not pair_key:
            continue

        dt: Optional[datetime] = None

        ts = data.get("createdAt")
        if ts:
            try:
                # google.cloud.firestore_v1._helpers.TimestampWithNanoseconds supports to_datetime()
                dt = ts.to_datetime()
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
            except Exception:
                dt = None

        if dt is None:
            dt = _as_dt(doc.id)

        if dt is None:
            continue

        prev = recent_pairs.get(pair_key)
        if prev is None or dt > prev:
            recent_pairs[pair_key] = dt

    return recent_pairs


def calculate_combination_penalty(
    candidate_items: List[Dict[str, Any]],
    recent_history: Dict[str, datetime],
    *,
    now: Optional[datetime] = None,
) -> float:
    """
    Penalize if the exact combination was worn recently.
    """
    if not candidate_items:
        return 0.0

    try:
        current_ids = [str(it.get("id")) for it in candidate_items if it.get("id") is not None]
    except Exception:
        return 0.0

    current_key = make_outfit_hash(current_ids)
    if not current_key:
        return 0.0

    last_worn = recent_history.get(current_key)
    if not last_worn:
        return 0.0

    now_dt = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    diff = now_dt - last_worn
    diff_days = diff.total_seconds() / 86400.0

    if diff_days <= 1.0:
        return -50.0
    if diff_days <= 3.0:
        return -20.0
    if diff_days <= 7.0:
        return -5.0
    return 0.0
