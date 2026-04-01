# api/admin.py


import logging
import os
from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from services.calendar_retention import cleanup_outfits_for_non_premium

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger("admin")

_ADMIN_KEY = os.getenv("ADMIN_KEY", "")


class CleanupResponse(BaseModel):
    dryRun: bool
    cutoffDate: str
    usersScanned: int
    usersProcessed: int
    outfitsDeleted: int


@router.post("/cleanup-calendar", response_model=CleanupResponse)
def cleanup_calendar(
    x_admin_key: str = Header(default=""),
    dry_run: bool = Query(True, description="dry run (true: delete하지 않고 카운트만)"),
    retention_days_free: int = Query(90, ge=7, le=3650, description="free retention days (default 90)"),
):
    if not _ADMIN_KEY or x_admin_key != _ADMIN_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    result = cleanup_outfits_for_non_premium(
        retention_days_free=retention_days_free,
        dry_run=dry_run,
    )
    return CleanupResponse(**result)
