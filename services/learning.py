# services/learning.py
from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

from google.cloud import firestore
from services.firestore import get_db

logger = logging.getLogger("learning")

# Phase 3: 적응적 LR + 확장된 가중치 범위
BASE_LR: float = 0.03
MIN_W: float = -1.0
MAX_W: float = 1.0

# DISCOVERY 가중치 상향 (0.5 → 0.7)
DISCOVERY_LR_RATIO: float = 0.7

# 적응적 LR: 성공 횟수 기반 decay
# LR = BASE_LR / (1 + success_count * DECAY_RATE)
DECAY_RATE: float = 0.05

Outcome = Literal["SUCCESS", "FAILURE", "DISCOVERY"]


def _adaptive_lr(base_lr: float, success_count: int) -> float:
    """성공 횟수가 많을수록 LR 감소 (이미 잘 학습된 영역은 천천히 조정)."""
    count = max(0, int(success_count))
    return base_lr / (1.0 + count * DECAY_RATE)


def _clamp(v: float) -> float:
    return max(MIN_W, min(MAX_W, float(v)))


def _norm_key(v: Optional[Any]) -> str:
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
        "categorySuccess",
        "colorSuccess",
        "seasonSuccess",
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
    snapshot = ref.get(transaction=transaction)
    w: Dict[str, Any] = snapshot.to_dict() if snapshot.exists else {}
    _ensure_maps(w)

    main_cat = _norm_key(item.get("mainCategory"))
    color = _norm_key(item.get("color"))
    season = _norm_key(item.get("season"))

    # 성공 카운터 로드 (적응적 LR용)
    cat_success = int(w["categorySuccess"].get(main_cat, 0))
    col_success = int(w["colorSuccess"].get(color, 0))
    sea_success = int(w["seasonSuccess"].get(season, 0))
    avg_success = (cat_success + col_success + sea_success) / 3.0

    lr = _adaptive_lr(BASE_LR, int(avg_success))

    if outcome == "SUCCESS":
        delta = lr
        w["categoryFail"][main_cat] = 0
        w["colorFail"][color] = 0
        w["seasonFail"][season] = 0
        # 성공 카운터 증가
        w["categorySuccess"][main_cat] = cat_success + 1
        w["colorSuccess"][color] = col_success + 1
        w["seasonSuccess"][season] = sea_success + 1

    elif outcome == "DISCOVERY":
        # Phase 3: DISCOVERY LR 상향 (0.5 → 0.7)
        delta = lr * DISCOVERY_LR_RATIO

    else:  # FAILURE
        cat_fail = int(w["categoryFail"].get(main_cat, 0)) + 1
        col_fail = int(w["colorFail"].get(color, 0)) + 1
        sea_fail = int(w["seasonFail"].get(season, 0)) + 1

        w["categoryFail"][main_cat] = cat_fail
        w["colorFail"][color] = col_fail
        w["seasonFail"][season] = sea_fail

        penalty_multiplier = 1 + min(max(cat_fail, col_fail, sea_fail), 5) * 0.5
        delta = -lr * penalty_multiplier

    w["categoryWeight"][main_cat] = _clamp(float(w["categoryWeight"].get(main_cat, 0.0)) + delta)
    w["colorWeight"][color] = _clamp(float(w["colorWeight"].get(color, 0.0)) + delta)
    w["seasonWeight"][season] = _clamp(float(w["seasonWeight"].get(season, 0.0)) + delta)

    w["updatedAt"] = firestore.SERVER_TIMESTAMP
    if "createdAt" not in w:
        w["createdAt"] = firestore.SERVER_TIMESTAMP

    transaction.set(ref, w, merge=True)


def update_learning_weight(
    user_id: str,
    item: Dict[str, Any],
    outcome: Outcome,
) -> None:
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
        raise


def get_learning_weights(user_id: str) -> Dict[str, Any]:
    """Phase 3: 유저 학습 가중치 조회 (scoring.py에서 사용)."""
    if not user_id:
        return {}

    try:
        db = get_db()
        ref = db.collection("learning_weights").document(user_id)
        snapshot = ref.get()
        if snapshot.exists:
            return snapshot.to_dict() or {}
    except Exception:
        logger.exception("[LEARNING] get_learning_weights failed user=%s", user_id)

    return {}


# Backward-compat wrapper
def update_learning_weight_bool(user_id: str, item: Dict[str, Any], success: bool) -> None:
    update_learning_weight(user_id, item, "SUCCESS" if success else "FAILURE")
