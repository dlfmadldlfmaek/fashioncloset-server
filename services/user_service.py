# services/user_service.py
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Optional

from google.cloud import firestore

logger = logging.getLogger(__name__)

_DB: Optional[firestore.Client] = None
_DB_LOCK = threading.Lock()


def _get_db() -> firestore.Client:
    """
    Singleton Firestore client per process.

    Why: per-call firestore.Client() adds overhead under Cloud Run concurrency.
    """
    global _DB
    if _DB is not None:
        return _DB
    with _DB_LOCK:
        if _DB is None:
            _DB = firestore.Client()
    return _DB


def _norm_key(v: Any) -> str:
    """
    Keep normalization consistent with learning.py.
    """
    if v is None:
        return "UNKNOWN"
    s = str(v).strip()
    if not s:
        return "UNKNOWN"
    return s.upper()


def load_user_preference(user_id: str) -> Dict[str, DefaultDict[str, float]]:
    """
    Load RL-style weights from learning_weights/{userId}.

    Returns:
      {
        "category": defaultdict(float, {...}),
        "season":   defaultdict(float, {...}),
        "color":    defaultdict(float, {...}),
      }
    """
    if not user_id:
        raise ValueError("user_id is required")

    db = _get_db()
    doc = db.collection("learning_weights").document(user_id).get()

    if not doc.exists:
        return {
            "category": defaultdict(float),
            "season": defaultdict(float),
            "color": defaultdict(float),
        }

    data = doc.to_dict() or {}

    # Normalize keys defensively to match runtime item keys
    cat = { _norm_key(k): float(v) for k, v in (data.get("categoryWeight") or {}).items() }
    sea = { _norm_key(k): float(v) for k, v in (data.get("seasonWeight") or {}).items() }
    col = { _norm_key(k): float(v) for k, v in (data.get("colorWeight") or {}).items() }

    return {
        "category": defaultdict(float, cat),
        "season": defaultdict(float, sea),
        "color": defaultdict(float, col),
    }


def process_user_feedback(user_id: str, item: Dict[str, Any], is_positive: bool) -> None:
    """
    Wrapper for learning updates.

    IMPORTANT:
      - If your learning.py uses (user_id, item, success: bool), call it that way.
      - If your learning.py was upgraded to (user_id, item, outcome: "SUCCESS"/"FAILURE"/"DISCOVERY"),
        call the upgraded signature.

    This function supports both without breaking.
    """
    if not user_id:
        raise ValueError("user_id is required")
    if not isinstance(item, dict):
        raise ValueError("item must be a dict")

    # Ensure item carries normalized keys (optional, but improves matching quality)
    if "mainCategory" in item:
        item["mainCategory"] = _norm_key(item.get("mainCategory"))
    if "season" in item:
        item["season"] = _norm_key(item.get("season"))
    if "color" in item:
        item["color"] = _norm_key(item.get("color"))

    try:
        # Try upgraded API first: update_learning_weight(user_id, item, outcome)
        from services.learning import update_learning_weight  # type: ignore

        outcome = "SUCCESS" if is_positive else "FAILURE"
        update_learning_weight(user_id, item, outcome)  # upgraded signature
        return
    except TypeError:
        # Fallback to legacy API: update_learning_weight(user_id, item, success=bool)
        from services.learning import update_learning_weight  # type: ignore

        update_learning_weight(user_id, item, success=is_positive)
        return
    except Exception:
        logger.exception("[FEEDBACK] update failed user=%s item=%s", user_id, item.get("id", "UNKNOWN"))
        raise
