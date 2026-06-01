# api/campaign.py
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from services.campaign import list_campaigns, run_campaign, send_test_to_uid

router = APIRouter(prefix="/campaign", tags=["campaign"])
logger = logging.getLogger("campaign_api")

# retention.py와 동일한 시크릿 헤더 패턴 재사용.
SERVICE_KEY = os.getenv("SERVICE_KEY", "")


def _require_service_key(request: Request) -> None:
    if not SERVICE_KEY:
        raise HTTPException(status_code=500, detail="server_misconfigured: SERVICE_KEY missing")
    if request.headers.get("X-Service-Key", "") != SERVICE_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")


@router.get("/list")
def campaigns_list(request: Request):
    """사용 가능한 캠페인 이름 목록."""
    _require_service_key(request)
    return {"campaigns": list_campaigns()}


@router.post("/run/{name}")
def campaign_run(
    request: Request,
    name: str,
    dry_run: bool = Query(True, description="기본 True: 발송 없이 타겟 수만 집계"),
    scan_limit: int = Query(3000, ge=1, le=20000),
):
    """
    Cloud Scheduler가 호출하는 캠페인 발송 엔드포인트.
    기본 dry_run=True (안전). 실제 발송은 dry_run=false 명시 필요.

    보안: Header X-Service-Key 필요.
    """
    _require_service_key(request)
    try:
        result = run_campaign(name, dry_run=dry_run, scan_limit=scan_limit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result


@router.post("/test/{name}")
def campaign_test(
    request: Request,
    name: str,
    uid: str = Query(..., description="테스트 푸시를 받을 유저 uid"),
):
    """세그먼트 무시하고 특정 uid 1명에게 대표 메시지 발송 — 디바이스 동작 검증용."""
    _require_service_key(request)
    try:
        return send_test_to_uid(name, uid)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
