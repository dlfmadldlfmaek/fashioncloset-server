# api/like.py
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from schemas.request import LikeRequest
from services.auth import verify_firebase_token
from services.firestore import set_like, delete_like
from services.rate_limit import limiter

logger = logging.getLogger("like")
router = APIRouter(prefix="/like", tags=["like"])


@router.post("")
@limiter.limit("60/minute")
def like(request: Request, req: LikeRequest, token: dict = Depends(verify_firebase_token)):
    if token["uid"] != req.userId:
        raise HTTPException(403, "forbidden: userId mismatch")

    logger.info("[LIKE] user=%s item=%s liked=%s", req.userId, req.id, req.liked)

    if req.liked:
        set_like(
            user_id=req.userId,
            clothes_id=req.id,
            main_category=req.mainCategory,
            tags=req.tags,
        )
        return {"ok": True, "liked": True}

    delete_like(user_id=req.userId, clothes_id=req.id)
    return {"ok": True, "liked": False}
