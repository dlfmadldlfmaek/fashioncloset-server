# api/recommend.py
import logging
from fastapi import APIRouter

from schemas.request import RecommendRequest
from schemas.response import RecommendResponse
from services.scoring import (
    personalization_weight,
    recently_worn_penalty,
)

router = APIRouter(prefix="/recommend", tags=["recommend"])
logger = logging.getLogger("recommend")


@router.post("", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    logger.info(f"[RECOMMEND] user={req.userId}")

    # -------------------------------------------------
    # 1️⃣ 날씨 조회 (요청 시점 import / 실패해도 절대 예외 없음)
    # -------------------------------------------------
    try:
        from services.weather import get_current_weather
        weather_raw = get_current_weather(req.lat, req.lon)
    except Exception:
        weather_raw = {
            "temp": 0.0,
            "feelsLike": 0.0,
            "wind": 0.0,
            "pty": "SUNNY",
        }

    weather = {
        "temp": weather_raw.get("temp", 0.0),
        "feelsLike": weather_raw.get("feelsLike", 0.0),
        "wind": weather_raw.get("wind", 0.0),
        "pty": weather_raw.get("pty", "SUNNY"),
    }

    # -------------------------------------------------
    # 2️⃣ 사용자 선호도 (현재 단계: Cloud Run → Firestore ❌)
    # -------------------------------------------------
    pref = {
        "category": {},
        "season": {},
        "color": {},
    }

    # -------------------------------------------------
    # 3️⃣ 점수 계산
    # -------------------------------------------------
    results = []
    base_score = 1.0

    for item in req.clothes:
        try:
            pref_score = personalization_weight(item, pref)
            worn_penalty = recently_worn_penalty(item.lastWornDate)

            # 날씨 보너스 (단순 / 안전)
            weather_bonus = 0.0
            if weather["feelsLike"] < 0:
                weather_bonus += 0.2
            if weather["pty"] in ("RAIN", "SNOW"):
                weather_bonus += 0.1

            final_score = (
                base_score
                + pref_score * 0.4
                + weather_bonus
                + worn_penalty
            )

            results.append({
                "id": item.id,
                "score": round(base_score, 2),
                "finalScore": round(final_score, 3),
            })

        except Exception as e:
            # 개별 아이템 실패는 전체 추천을 막지 않음
            logger.warning(f"[ITEM SKIP] id={item.id} err={e}")

    # 점수 높은 순 정렬
    results.sort(key=lambda x: x["finalScore"], reverse=True)

    # -------------------------------------------------
    # 4️⃣ 응답
    # -------------------------------------------------
    return RecommendResponse(
        weather=weather,
        recommended=results
    )
