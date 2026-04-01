# api/quota.py
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from services.rate_limit import limiter

from services.auth import verify_firebase_token
from services.premium import is_premium_user

logger = logging.getLogger("quota")

router = APIRouter(prefix="/quota", tags=["quota"])


# =========================================================
# Policy (env로 조절 가능)
# =========================================================
# 무료 사용자 일일 사용 가능 횟수 (예: 착장 생성 / AI 추천 등)
FREE_DAILY_QUOTA = int(os.getenv("FREE_DAILY_QUOTA", "5"))

# 프리미엄은 사실상 무제한. 숫자로 내려줘야 하면 큰 값 사용
PREMIUM_DAILY_QUOTA = int(os.getenv("PREMIUM_DAILY_QUOTA", "999999"))

# quota를 “일 단위”로 리셋하는 기준 (UTC 기준으로 하는 게 운영에 편함)
RESET_TZ = os.getenv("QUOTA_RESET_TZ", "UTC").strip().upper()  # "UTC" or "LOCAL"
SECONDS_PER_DAY = 86400


def _day_key(now: Optional[float] = None) -> str:
    """Return a stable day bucket key (e.g. '2026-02-15')."""
    now = now or time.time()
    if RESET_TZ == "LOCAL":
        lt = time.localtime(now)
        return f"{lt.tm_year:04d}-{lt.tm_mon:02d}-{lt.tm_mday:02d}"
    gt = time.gmtime(now)
    return f"{gt.tm_year:04d}-{gt.tm_mon:02d}-{gt.tm_mday:02d}"


# =========================================================
# Storage (Firestore optional, memory fallback)
# =========================================================
@dataclass
class QuotaRecord:
    day: str
    used: int


# 메모리 fallback: {(userId, feature, day): used}
_MEM: Dict[Tuple[str, str, str], int] = {}


def _try_get_firestore():
    """
    Optional Firestore client.
    - firebase_admin가 설치/초기화되어 있으면 사용
    - 아니면 None (서버는 죽지 않음)
    """
    try:
        import firebase_admin  # type: ignore
        from firebase_admin import firestore  # type: ignore

        if not firebase_admin._apps:
            # 환경에 따라 자동 초기화가 필요할 수 있음.
            # 보통 Cloud Run에서는 GOOGLE_APPLICATION_CREDENTIALS 또는 기본 SA로 인증됨.
            firebase_admin.initialize_app()

        return firestore.client()
    except Exception as e:
        logger.info("[QUOTA] firestore not available -> memory fallback (%s)", e)
        return None


_DB = _try_get_firestore()


def _quota_doc(user_id: str, feature: str, day: str) -> Any:
    # users/{uid}/quota/{feature_day} 형태
    # (원하는 구조가 있으면 여기만 바꾸면 됨)
    return (
        _DB.collection("users")
        .document(user_id)
        .collection("quota")
        .document(f"{feature}_{day}")
    )


def _get_used(user_id: str, feature: str, day: str) -> int:
    if _DB is None:
        return int(_MEM.get((user_id, feature, day), 0))

    try:
        doc = _quota_doc(user_id, feature, day).get()
        if not doc.exists:
            return 0
        data = doc.to_dict() or {}
        return int(data.get("used", 0))
    except Exception as e:
        logger.warning("[QUOTA] firestore read failed -> fallback memory (%s)", e)
        return int(_MEM.get((user_id, feature, day), 0))


def _set_used(user_id: str, feature: str, day: str, used: int) -> None:
    used = max(0, int(used))

    if _DB is None:
        _MEM[(user_id, feature, day)] = used
        return

    try:
        _quota_doc(user_id, feature, day).set(
            {
                "day": day,
                "feature": feature,
                "used": used,
                "updatedAt": time.time(),
            },
            merge=True,
        )
    except Exception as e:
        logger.warning("[QUOTA] firestore write failed -> fallback memory (%s)", e)
        _MEM[(user_id, feature, day)] = used


def _inc_used(user_id: str, feature: str, day: str, amount: int) -> int:
    """
    Increment used and return new used.
    Firestore가 있으면 transaction으로 정확히(권장).
    없으면 메모리로.
    """
    amount = max(0, int(amount))

    if _DB is None:
        key = (user_id, feature, day)
        _MEM[key] = int(_MEM.get(key, 0)) + amount
        return int(_MEM[key])

    # Firestore transaction
    try:
        from google.cloud.firestore import Transaction  # type: ignore

        @_DB.transactional
        def _txn_inc(txn: Transaction) -> int:
            ref = _quota_doc(user_id, feature, day)
            snap = ref.get(transaction=txn)
            used = 0
            if snap.exists:
                used = int((snap.to_dict() or {}).get("used", 0))
            used_new = used + amount
            txn.set(
                ref,
                {
                    "day": day,
                    "feature": feature,
                    "used": used_new,
                    "updatedAt": time.time(),
                },
                merge=True,
            )
            return used_new


        return int(_txn_inc(_DB.transaction()))
    except Exception as e:
        logger.warning("[QUOTA] firestore txn failed -> fallback non-txn (%s)", e)
        used = _get_used(user_id, feature, day)
        used_new = used + amount
        _set_used(user_id, feature, day, used_new)
        return used_new


# =========================================================
# API Schemas
# =========================================================
class QuotaStatusResponse(BaseModel):
    userId: str
    feature: str
    day: str
    premium: bool
    limit: int
    used: int
    remaining: int


class ConsumeRequest(BaseModel):
    userId: str = Field(..., min_length=1)
    feature: str = Field(default="tryon_generate", min_length=1)
    amount: int = Field(default=1, ge=1)


class ConsumeResponse(BaseModel):
    ok: bool
    status: QuotaStatusResponse


# =========================================================
# Helpers
# =========================================================
def _get_limit(premium: bool) -> int:
    return PREMIUM_DAILY_QUOTA if premium else FREE_DAILY_QUOTA


def _status(user_id: str, feature: str, day: str) -> QuotaStatusResponse:
    premium = bool(is_premium_user(user_id))
    limit = _get_limit(premium)
    used = 0 if premium else _get_used(user_id, feature, day)
    remaining = max(0, limit - used) if not premium else limit
    return QuotaStatusResponse(
        userId=user_id,
        feature=feature,
        day=day,
        premium=premium,
        limit=limit,
        used=used,
        remaining=remaining,
    )


# =========================================================
# Routes
# =========================================================
@router.get("", response_model=QuotaStatusResponse)
def get_quota(
    userId: str,
    feature: str = "tryon_generate",
    token: dict = Depends(verify_firebase_token),
):
    """
    현재 quota 상태 조회
    """
    if token["uid"] != userId:
        raise HTTPException(403, "forbidden: userId mismatch")

    day = _day_key()
    return _status(userId, feature, day)


@router.post("/consume", response_model=ConsumeResponse)
@limiter.limit("20/minute")
def consume_quota(request: Request, req: ConsumeRequest, token: dict = Depends(verify_firebase_token)):
    """
    quota 차감.
    - premium: 차감하지 않고 ok=true
    - free: remaining 부족하면 429
    """
    if token["uid"] != req.userId:
        raise HTTPException(403, "forbidden: userId mismatch")

    day = _day_key()
    premium = bool(is_premium_user(req.userId))
    limit = _get_limit(premium)

    if premium:
        st = _status(req.userId, req.feature, day)
        return ConsumeResponse(ok=True, status=st)

    used = _get_used(req.userId, req.feature, day)
    if used + req.amount > limit:
        st = _status(req.userId, req.feature, day)
        raise HTTPException(status_code=429, detail={"message": "quota exceeded", "status": st.model_dump()})

    new_used = _inc_used(req.userId, req.feature, day, req.amount)
    st = _status(req.userId, req.feature, day)
    # status.used를 정확히 맞추고 싶으면 new_used로 덮어도 됨
    st.used = new_used
    st.remaining = max(0, limit - new_used)

    return ConsumeResponse(ok=True, status=st)


@router.post("/reset", response_model=QuotaStatusResponse)
def reset_quota(
    userId: str,
    feature: str = "tryon_generate",
    x_admin_key: str = Header(default=""),
):
    """
    (관리/디버그용) 오늘 사용량 0으로 리셋 — ADMIN_KEY 인증 필요
    """
    expected = os.getenv("ADMIN_KEY", "")
    if not expected or x_admin_key != expected:
        raise HTTPException(status_code=401, detail="unauthorized")

    day = _day_key()
    premium = bool(is_premium_user(userId))
    if premium:
        return _status(userId, feature, day)

    _set_used(userId, feature, day, 0)
    return _status(userId, feature, day)
