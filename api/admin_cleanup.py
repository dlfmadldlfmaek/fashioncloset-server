# api/admin_cleanup.py
import os

from fastapi import APIRouter, HTTPException, Header
from services.firestore import get_db
from services.calendar_retention import cleanup_user_calendar

router = APIRouter(prefix="/admin", tags=["admin"])

_ADMIN_KEY = os.getenv("ADMIN_KEY", "")

@router.post("/cleanup-calendar")
def cleanup_calendar(x_admin_key: str = Header(default="")):
    if not _ADMIN_KEY or x_admin_key != _ADMIN_KEY:
        raise HTTPException(status_code=403, detail="forbidden")

    db = get_db()
    users = db.collection("users").stream()
    results = []
    for u in users:
        results.append(cleanup_user_calendar(u.id))
    return {"ok": True, "results": results}
