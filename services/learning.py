# services/learning.py
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from google.cloud import firestore
from services.firestore import get_db

logger = logging.getLogger("learning")

# Firestore client: use shared singleton from services.firestore

# Learning constants
LR: float = 0.03
MIN_W: float = -0.6
MAX_W: float = 0.6

Outcome = Literal["SUCCESS", "FAILURE", "DISCOVERY"]


def _clamp(v: float) -> float:
    """Clamp weight within [MIN_W, MAX_W]."""
    return max(MIN_W, min(MAX_W, float(v)))


def _norm_key(v: Optional[Any]) -> str:
    """
    Normalize map keys to avoid key explosion.

    Why: Firestore map keys are effectively unbounded; normalize casing/whitespace.
    """
    if v is None:
        return "UNKNOWN"
    s = str(v).strip()
    if not s:
        return "UNKNOWN"
    return s.upper()


def _ensure_maps(w: Dict[str, Any]) -> None:
    for k in (
        "categoryWeight",
        "colorWeight",
        "seasonWeight",
        "categoryFail",
        "colorFail",
        "seasonFail",
    ):
        if k not in w or not isinstance(w.get(k), dict):
            w[k] = {}


@firestore.transactional
def _update_in_transaction(
    transaction: firestore.Transaction,
    ref: firestore.DocumentReference,
    item: Dict[str, Any],
    outcome: Outcome,
) -> None:
    """
    Transactional update to user learning weights.

    outcome:
      - SUCCESS   : user liked/wore (positive)
      - FAILURE   : user skipped/disliked (negative, progressive)
      - DISCOVERY : user wore something not recommended (mild positive)
    """
    snapshot = ref.get(transaction=transaction)
    w: Dict[str, Any] = snapshot.to_dict() if snapshot.exists else {}
    _ensure_maps(w)

    main_cat = _norm_key(item.get("mainCategory"))
    color = _norm_key(item.get("color"))
    season = _norm_key(item.get("season"))

    if outcome == "SUCCESS":
        delta = LR
        w["categoryFail"][main_cat] = 0
        w["colorFail"][color] = 0
        w["seasonFail"][season] = 0

    elif outcome == "DISCOVERY":
        # Slight positive: "you missed it but user wore it"
        delta = LR * 0.5
        # Do not reset fail counters aggressively for discovery

    else:  # FAILURE
        cat_fail = int(w["categoryFail"].get(main_cat, 0)) + 1
        col_fail = int(w["colorFail"].get(color, 0)) + 1
        sea_fail = int(w["seasonFail"].get(season, 0)) + 1

        w["categoryFail"][main_cat] = cat_fail
        w["colorFail"][color] = col_fail
        w["seasonFail"][season] = sea_fail

        penalty_multiplier = 1 + min(max(cat_fail, col_fail, sea_fail), 5) * 0.5
        delta = -LR * penalty_multiplier

    w["categoryWeight"][main_cat] = _clamp(float(w["categoryWeight"].get(main_cat, 0.0)) + delta)
    w["colorWeight"][color] = _clamp(float(w["colorWeight"].get(color, 0.0)) + delta)
    w["seasonWeight"][season] = _clamp(float(w["seasonWeight"].get(season, 0.0)) + delta)

    # Prefer server-side timestamp for consistency
    w["updatedAt"] = firestore.SERVER_TIMESTAMP
    # Keep createdAt stable if you want (optional)
    if "createdAt" not in w:
        w["createdAt"] = firestore.SERVER_TIMESTAMP

    # merge=True prevents accidental deletion of unrelated fields
    transaction.set(ref, w, merge=True)


def update_learning_weight(
    user_id: str,
    item: Dict[str, Any],
    outcome: Outcome,
) -> None:
    """
    Public API.

    item: dict must include (at least) id plus optional mainCategory/color/season.
    outcome: "SUCCESS" | "FAILURE" | "DISCOVERY"
    """
    if not user_id:
        raise ValueError("user_id is required")
    if outcome not in ("SUCCESS", "FAILURE", "DISCOVERY"):
        raise ValueError(f"invalid outcome: {outcome}")

    db = get_db()
    ref = db.collection("learning_weights").document(user_id)
    transaction = db.transaction()

    try:
        _update_in_transaction(transaction, ref, item, outcome)
        logger.info(
            "[LEARNING] user=%s outcome=%s item=%s",
            user_id,
            outcome,
            item.get("id", "UNKNOWN"),
        )
    except Exception:
        logger.exception("[LEARNING] update failed user=%s item=%s", user_id, item.get("id", "UNKNOWN"))
        # keep raising if you want caller to handle, or swallow:
        # return
        raise


# Backward-compat wrapper (if some callers still pass success: bool)
def update_learning_weight_bool(user_id: str, item: Dict[str, Any], success: bool) -> None:
    update_learning_weight(user_id, item, "SUCCESS" if success else "FAILURE")
