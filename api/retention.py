# api/retention.py


import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request

from services.firestore import get_db
from services.premium import is_premium_user

router = APIRouter(prefix="/retention", tags=["retention"])
logger = logging.getLogger("retention")

SERVICE_KEY = os.getenv("SERVICE_KEY", "")

# Firestore batch limit
_BATCH_LIMIT = 450  # 안전 여유 (최대 500)


def _require_service_key(request: Request) -> None:
    """
    간단한 서버 보호:
    - Header: X-Service-Key: <SERVICE_KEY>
    """
    if not SERVICE_KEY:
        # 키를 환경변수에 안 넣었으면 운영에서 위험. 안전상 막음.
        raise HTTPException(status_code=500, detail="server_misconfigured: SERVICE_KEY missing")

    got = request.headers.get("X-Service-Key", "")
    if got != SERVICE_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")


def _cutoffs(days: int = 90) -> Tuple[str, int]:
    """
    반환:
      - cutoff_date_str: "YYYY-MM-DD" (outfits의 date 비교용)
      - cutoff_ms: epoch milliseconds (wearLogs의 wornAt 비교용)
    """
    now_utc = datetime.now(timezone.utc)
    cutoff_dt = now_utc - timedelta(days=days)

    cutoff_date_str = cutoff_dt.date().isoformat()  # yyyy-mm-dd
    cutoff_ms = int(cutoff_dt.timestamp() * 1000)
    return cutoff_date_str, cutoff_ms


def _delete_query_in_batches(query) -> int:
    """
    query 결과를 batch로 나눠 삭제.
    반환: 삭제된 문서 수
    """
    db = get_db()
    deleted = 0

    while True:
        snaps = query.limit(_BATCH_LIMIT).get()
        if not snaps:
            break

        batch = db.batch()
        for doc in snaps:
            batch.delete(doc.reference)
        batch.commit()

        deleted += len(snaps)

        # 다음 batch로 계속(같은 query 재실행)
        if len(snaps) < _BATCH_LIMIT:
            break

    return deleted


def _cleanup_user(user_id: str, days: int = 90) -> dict:
    """
    무료 유저: 90일 지난 outfits / wearLogs 삭제
    프리미엄: 삭제 안함
    """
    if not user_id:
        return {"userId": user_id, "skipped": True, "reason": "empty_user_id"}

    if is_premium_user(user_id):
        return {"userId": user_id, "skipped": True, "reason": "premium_user"}

    cutoff_date_str, cutoff_ms = _cutoffs(days=days)
    db = get_db()
    user_doc = db.collection("users").document(user_id)

    # 1) outfits: users/{uid}/outfits
    # - doc id가 yyyy-mm-dd여도, 안전하게 "date" 필드 기준으로 삭제
    outfits_q = (
        user_doc.collection("outfits")
        .where("date", "<", cutoff_date_str)
    )
    deleted_outfits = _delete_query_in_batches(outfits_q)

    # 2) wearLogs: users/{uid}/wearLogs (wornAt: ms)
    wearlogs_q = (
        user_doc.collection("wearLogs")
        .where("wornAt", "<", cutoff_ms)
    )
    deleted_wearlogs = _delete_query_in_batches(wearlogs_q)

    return {
        "userId": user_id,
        "skipped": False,
        "cutoffDate": cutoff_date_str,
        "deleted": {
            "outfits": deleted_outfits,
            "wearLogs": deleted_wearlogs,
        },
    }


@router.post("/cleanup")
def cleanup_retention(
    request: Request,
    userId: Optional[str] = Query(None, description="특정 유저만 정리 (없으면 전체 유저)"),
    days: int = Query(90, ge=1, le=3650, description="무료 보관 기간(일) 기본 90일"),
    limitUsers: int = Query(300, ge=1, le=5000, description="전체 정리 시 한 번에 처리할 유저 수 제한"),
):
    """
    ✅ 무료 3개월(90일) 지난 캘린더 데이터 삭제
    - premium: 무제한 (삭제 안함)
    - free: outfits / wearLogs에서 오래된 것 삭제

    보안:
      - Header: X-Service-Key 필요 (환경변수 SERVICE_KEY와 일치)
    """
    _require_service_key(request)

    # 단일 유저
    if userId:
        result = _cleanup_user(userId, days=days)
        return {"mode": "single", "result": result}

    # 전체 유저 스캔 (users collection)
    db = get_db()
    users = db.collection("users").limit(limitUsers).get()

    results = []
    for u in users:
        uid = u.id
        try:
            results.append(_cleanup_user(uid, days=days))
        except Exception as e:
            logger.exception("cleanup failed user=%s err=%s", uid, e)
            results.append({"userId": uid, "skipped": True, "reason": f"error:{type(e).__name__}"})

    summary = {
        "processedUsers": len(results),
        "skippedPremium": sum(1 for r in results if r.get("reason") == "premium_user"),
        "deletedOutfits": sum((r.get("deleted") or {}).get("outfits", 0) for r in results if not r.get("skipped")),
        "deletedWearLogs": sum((r.get("deleted") or {}).get("wearLogs", 0) for r in results if not r.get("skipped")),
    }

    return {"mode": "all", "summary": summary, "results": results}
