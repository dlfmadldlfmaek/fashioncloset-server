# services/style_vector.py
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from google.cloud import firestore

logger = logging.getLogger(__name__)

# Phase 3: EMA 계수 (α=0.3 → 최근 착용이 30% 영향, 누적이 70%)
EMA_ALPHA: float = 0.3


def get_db() -> firestore.Client:
    return firestore.Client()


def _as_1d_float_vector(x: object) -> Optional[np.ndarray]:
    try:
        v = np.asarray(x, dtype=np.float32)
    except Exception:
        return None
    if v.ndim != 1 or v.size == 0:
        return None
    if not np.isfinite(v).all():
        return None
    return v


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        return v
    return v / norm


@firestore.transactional
def _update_in_transaction(
    transaction: firestore.Transaction,
    ref: firestore.DocumentReference,
    outfit_vector: np.ndarray,
) -> None:
    snapshot = ref.get(transaction=transaction)
    outfit_vector = _l2_normalize(outfit_vector)

    if snapshot.exists:
        data = snapshot.to_dict() or {}

        old_raw = data.get("vector")
        old_vec = _as_1d_float_vector(old_raw)
        if old_vec is None:
            transaction.set(
                ref,
                {
                    "vector": outfit_vector.tolist(),
                    "count": 1,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                },
                merge=True,
            )
            logger.warning("[STYLE_VECTOR] reset invalid existing vector style=%s", ref.id)
            return

        if old_vec.shape[0] != outfit_vector.shape[0]:
            logger.warning(
                "[STYLE_VECTOR] dim mismatch style=%s old=%d new=%d (skip update)",
                ref.id,
                old_vec.shape[0],
                outfit_vector.shape[0],
            )
            return

        count = int(data.get("count", 1))
        if count < 1:
            count = 1

        # Phase 3: EMA (지수이동평균) — 최근 착용이 더 큰 영향
        # 기존: 단순 평균 (old * count + new) / (count + 1)
        # 변경: new_vec = α * new + (1 - α) * old
        new_vec = EMA_ALPHA * outfit_vector + (1.0 - EMA_ALPHA) * old_vec
        new_vec = _l2_normalize(new_vec)

        transaction.update(
            ref,
            {
                "vector": new_vec.tolist(),
                "count": count + 1,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            },
        )

        logger.info("[STYLE_VECTOR] updated (EMA α=%.2f) style=%s count=%d", EMA_ALPHA, ref.id, count + 1)

    else:
        transaction.set(
            ref,
            {
                "vector": outfit_vector.tolist(),
                "count": 1,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            },
        )
        logger.info("[STYLE_VECTOR] created style=%s count=1", ref.id)


def update_style_vector(user_id: str, style: str, outfit_vector: object) -> None:
    vec = _as_1d_float_vector(outfit_vector)
    if vec is None:
        logger.warning("[STYLE_VECTOR] invalid outfit_vector user=%s style=%s", user_id, style)
        return

    try:
        db = get_db()
        ref = (
            db.collection("users")
            .document(user_id)
            .collection("style_vectors")
            .document(style)
        )
        tx = db.transaction()
        _update_in_transaction(tx, ref, vec)
    except Exception as e:
        logger.exception("[STYLE_VECTOR] update failed user=%s style=%s err=%s", user_id, style, e)


def get_user_style_vector(user_id: str, style: str) -> Optional[np.ndarray]:
    """Phase 3: 유저 스타일 벡터 조회 (api/recommend.py에서 앵커 혼합용)."""
    try:
        db = get_db()
        ref = (
            db.collection("users")
            .document(user_id)
            .collection("style_vectors")
            .document(style)
        )
        snapshot = ref.get()
        if snapshot.exists:
            data = snapshot.to_dict() or {}
            return _as_1d_float_vector(data.get("vector"))
    except Exception as e:
        logger.exception("[STYLE_VECTOR] get failed user=%s style=%s err=%s", user_id, style, e)

    return None
