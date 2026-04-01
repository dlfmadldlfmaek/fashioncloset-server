# services/analytics.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Set, List

from google.cloud import firestore

from services.firestore import get_db


def _as_utc_day_range(date_str: str) -> tuple[datetime, datetime]:
    """
    Convert YYYY-MM-DD into [start, end) UTC range.

    Why: Avoid naive datetime mismatches with Firestore Timestamp (UTC).
    """
    start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _extract_recommended_ids(doc: Dict[str, Any]) -> Set[str]:
    """
    Normalize recommended item ids from multiple possible schema shapes.
    """
    ids: Set[str] = set()

    raw = doc.get("itemIds")
    if isinstance(raw, list):
        ids |= {str(x) for x in raw if x is not None}

    items = doc.get("items")
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and it.get("id") is not None:
                ids.add(str(it["id"]))

    recommended = doc.get("recommended")
    if isinstance(recommended, list):
        for it in recommended:
            if isinstance(it, dict) and it.get("id") is not None:
                ids.add(str(it["id"]))

    return ids


def analyze_recommend_vs_wear(user_id: str, target_date_str: str) -> Optional[Dict[str, Any]]:
    """
    Compare a day's recommendation history vs actual worn outfit.

    Collections:
      - users/{uid}/outfits/{YYYY-MM-DD}
      - users/{uid}/recommend_history where createdAt is Firestore Timestamp
    """
    db = get_db()

    outfit_doc = (
        db.collection("users")
        .document(user_id)
        .collection("outfits")
        .document(target_date_str)
        .get()
    )

    if not outfit_doc.exists:
        return None

    outfit_data = outfit_doc.to_dict() or {}
    worn_ids: Set[str] = {str(x) for x in (outfit_data.get("clothesIds") or []) if x is not None}
    if not worn_ids:
        return {
            "date": target_date_str,
            "recommendedIds": [],
            "wornIds": [],
            "matchedIds": [],
            "success": False,
            "accuracy": 0.0,
        }

    start_dt, end_dt = _as_utc_day_range(target_date_str)

    history_query = (
        db.collection("users")
        .document(user_id)
        .collection("recommend_history")
        .where("createdAt", ">=", start_dt)
        .where("createdAt", "<", end_dt)
        .order_by("createdAt", direction=firestore.Query.DESCENDING)  # latest of the day
        .limit(1)
    )

    docs_iter = history_query.stream()
    first = next(docs_iter, None)
    if first is None:
        return None

    recommend_doc = first.to_dict() or {}
    recommended_ids = _extract_recommended_ids(recommend_doc)

    matched_ids = recommended_ids & worn_ids

    return {
        "date": target_date_str,
        "recommendedIds": sorted(recommended_ids),
        "wornIds": sorted(worn_ids),
        "matchedIds": sorted(matched_ids),
        "success": len(matched_ids) > 0,
        "accuracy": (len(matched_ids) / len(recommended_ids)) if recommended_ids else 0.0,
    }


def apply_learning(user_id: str, recommended_ids: List[str], worn_ids: Set[str]) -> None:
    """
    Update learning weights based on recommendation vs actual wear.

    Note: Consider batching/queueing if update_learning_weight writes per item.
    """
    try:
        from services.learning import update_learning_weight
    except ImportError:
        return

    rec_set = {str(x) for x in (recommended_ids or []) if x is not None}
    worn_set = {str(x) for x in (worn_ids or set()) if x is not None}

    for item_id in rec_set:
        update_learning_weight(user_id, item_id, "SUCCESS" if item_id in worn_set else "FAILURE")

    for item_id in worn_set:
        if item_id not in rec_set:
            update_learning_weight(user_id, item_id, "DISCOVERY")
