# services/firestore.py
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence

from google.cloud import firestore

logger = logging.getLogger(__name__)

_DB: Optional[firestore.Client] = None
_DB_LOCK = threading.Lock()


def get_db() -> firestore.Client:
    """Singleton Firestore client per process (Cloud Run friendly)."""
    global _DB
    if _DB is not None:
        return _DB
    with _DB_LOCK:
        if _DB is None:
            _DB = firestore.Client()
    return _DB


def _norm_tags(tags: Optional[Iterable[Any]]) -> List[str]:
    if not tags:
        return []
    out: List[str] = []
    for t in tags:
        if t is None:
            continue
        s = str(t).strip()
        if s:
            out.append(s)
    return out


def _norm_vector(vector: Sequence[Any]) -> List[float]:
    if vector is None:
        raise ValueError("vector is required")
    out: List[float] = []
    for x in vector:
        if x is None:
            continue
        try:
            out.append(float(x))
        except (TypeError, ValueError) as e:
            raise ValueError(f"vector contains non-numeric value: {x!r}") from e
    if not out:
        raise ValueError("vector must not be empty")
    return out


def set_like(
    user_id: str,
    clothes_id: str,
    main_category: Optional[str],
    tags: Optional[Iterable[Any]] = None,
) -> None:
    """
    Upsert like doc at users/{userId}/likes/{clothesId}.
    Uses fixed doc id to support unlike via direct delete.
    """
    if not user_id:
        raise ValueError("user_id is required")
    if not clothes_id:
        raise ValueError("clothes_id is required")

    db = get_db()
    ref = (
        db.collection("users")
        .document(user_id)
        .collection("likes")
        .document(clothes_id)
    )

    payload: Dict[str, Any] = {
        "clothesId": clothes_id,
        "mainCategory": (main_category or "").strip(),
        "tags": _norm_tags(tags),
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> None:
        snap = ref.get(transaction=transaction)
        if not snap.exists:
            payload["createdAt"] = firestore.SERVER_TIMESTAMP
        transaction.set(ref, payload, merge=True)

    tx = db.transaction()
    _txn(tx)


def delete_like(user_id: str, clothes_id: str) -> None:
    """Delete users/{userId}/likes/{clothesId} (idempotent)."""
    if not user_id:
        raise ValueError("user_id is required")
    if not clothes_id:
        raise ValueError("clothes_id is required")

    db = get_db()
    ref = (
        db.collection("users")
        .document(user_id)
        .collection("likes")
        .document(clothes_id)
    )
    ref.delete()


def save_like_log(
    user_id: str,
    clothes_id: str,
    category: str,
    season: str,
    color: str,
) -> None:
    """Legacy random-ID like log (optional)."""
    db = get_db()
    ref = (
        db.collection("users")
        .document(user_id)
        .collection("likes")
        .document()
    )
    ref.set(
        {
            "clothesId": clothes_id,
            "category": (category or "").strip(),
            "season": (season or "").strip(),
            "color": (color or "").strip(),
            "createdAt": firestore.SERVER_TIMESTAMP,
        }
    )


def save_style_anchor(
    user_id: str,
    style_name: str,
    vector: Sequence[Any],
) -> None:
    """users/{uid}/styleAnchors/{styleName}."""
    if not user_id:
        raise ValueError("user_id is required")
    if not style_name:
        raise ValueError("style_name is required")

    db = get_db()
    ref = (
        db.collection("users")
        .document(user_id)
        .collection("styleAnchors")
        .document(style_name)
    )
    vec = _norm_vector(vector)
    ref.set(
        {
            "vector": vec,
            "dim": len(vec),
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )
