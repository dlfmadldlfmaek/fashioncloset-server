# services/style_vector.py
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from google.cloud import firestore

logger = logging.getLogger(__name__)


def get_db() -> firestore.Client:
    """
    Cloud Run 안전: 요청 시점에 Client 생성.
    (네 프로젝트에서 이미 이 정책을 쓰고 있으니 통일)
    """
    return firestore.Client()


def _as_1d_float_vector(x: object) -> Optional[np.ndarray]:
    """Return finite 1D float32 vector or None."""
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

    # 항상 정규화된 값으로 들어가게
    outfit_vector = _l2_normalize(outfit_vector)

    if snapshot.exists:
        data = snapshot.to_dict() or {}

        old_raw = data.get("vector")
        old_vec = _as_1d_float_vector(old_raw)
        if old_vec is None:
            # 기존 데이터가 깨진 경우: 새 벡터로 리셋(운영 안전)
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
            # 차원 불일치: 안전하게 스킵 or 리셋 선택
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

        # 점진 평균
        new_vec = (old_vec * count + outfit_vector) / (count + 1)
        new_vec = _l2_normalize(new_vec)

        transaction.update(
            ref,
            {
                "vector": new_vec.tolist(),
                "count": count + 1,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            },
        )

        logger.info("[STYLE_VECTOR] updated style=%s count=%d", ref.id, count + 1)

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
    """
    users/{uid}/style_vectors/{style}
    - outfit_vector는 list/np.ndarray 모두 가능.
    - 내부에서 검증/정규화 후 트랜잭션으로 원자 업데이트.
    """
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
        # 필요하면 여기서 raise
