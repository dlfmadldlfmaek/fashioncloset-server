# api/outfit.py


import logging
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from services.auth import verify_firebase_token
from services.firestore import get_db
from services.outfit_encoder import encode_text  # 또는 encode_outfit_image/bytes/url
from services.style_vector import update_style_vector
from services.analytics import analyze_recommend_vs_wear, apply_learning  # 네가 올린 분석/학습 유틸

router = APIRouter(prefix="/outfit", tags=["outfit"])
logger = logging.getLogger(__name__)


@router.post("/confirm")
def confirm_outfit(
    req: dict,
    background_tasks: BackgroundTasks,
    token: dict = Depends(verify_firebase_token),
):
    """
    req 예시:
      {
        "userId": "...",
        "date": "2026-01-30",      # optional, default today(UTC)
        "clothesIds": ["a","b"],
        "style": "casual",        # optional
        "tags": ["후드","데님"],   # optional (없으면 clothes에서 모아도 됨)
      }
    """
    user_id = req.get("userId")
    if not user_id:
        raise HTTPException(400, "userId required")

    if token["uid"] != user_id:
        raise HTTPException(403, "forbidden: userId mismatch")

    date_str = req.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    clothes_ids = req.get("clothesIds") or []
    if not clothes_ids:
        raise HTTPException(400, "clothesIds required")

    style = (req.get("style") or "").strip().lower() or "casual"
    tags = req.get("tags") or []

    # 1) outfits 저장
    db = get_db()
    db.collection("users").document(user_id).collection("outfits").document(date_str).set(
        {
            "clothesIds": clothes_ids,
            "createdAt": datetime.now(timezone.utc),  # 또는 SERVER_TIMESTAMP
        },
        merge=True,
    )

    # 2) 백그라운드: 추천 vs 착용 분석 → learning 반영
    def _learning_job():
        res = analyze_recommend_vs_wear(user_id, date_str)
        if not res:
            return
        apply_learning(user_id, res["recommendedIds"], set(res["wornIds"]))

    background_tasks.add_task(_learning_job)

    # 3) 백그라운드: style_vector 업데이트 (착용 확정 기반)
    def _style_vector_job():
        # 가장 안전한 최소 정보는 tags 기반 텍스트 임베딩(이미지 없어도 됨)
        vec = encode_text(tags if tags else [style])
        update_style_vector(user_id, style, vec)

    background_tasks.add_task(_style_vector_job)

    return {"ok": True, "date": date_str}
