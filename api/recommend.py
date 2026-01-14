# api/recommend.py
import logging
from fastapi import APIRouter

from schemas.request import RecommendRequest
from schemas.response import RecommendResponse

router = APIRouter(prefix="/recommend", tags=["recommend"])
logger = logging.getLogger("recommend")


@router.post("", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    # 🔥 날씨는 완전히 보호
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

    # 🔥 계산 최소화 (외부 의존성 없음)
    results = [
        {
            "id": item.id,
            "score": 1.0,
            "finalScore": 1.0,
        }
        for item in req.clothes
    ]

    return RecommendResponse(
        weather=weather,
        recommended=results
    )
