# api/weather.py
#
# 클라이언트(FeedCard, RecommendPage)가 KMA 날씨를 직접 받아갈 수 있는 단독 엔드포인트.
# 1.12.1+91 이전엔 클라이언트가 open-meteo로 직접 호출했고, recommend.py의 _build_weather가
# 클라이언트 temp를 그대로 echo만 해서 보라색 weather 카드도 open-meteo 값이었음.
# 이 엔드포인트로 통일하면 Feed 카드 ° / 보라색 카드가 같은 KMA 값으로 표시된다.

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from services.rate_limit import limiter
from services.weather import get_current_weather

router = APIRouter(prefix="/weather", tags=["weather"])
logger = logging.getLogger("weather")


@router.get("")
@limiter.limit("60/minute")
def current_weather(
    request: Request,
    lat: float = Query(..., description="위도"),
    lon: float = Query(..., description="경도"),
):
    """현재 위치의 KMA 단기 관측 (T1H/WSD/PTY)."""
    try:
        w = get_current_weather(lat, lon)
        return {
            "temp": w["temp"],
            "feelsLike": w.get("feelsLike", w["temp"]),
            "wind": w.get("wind", 0.0),
            "pty": w.get("pty", "SUNNY"),
        }
    except Exception as e:
        logger.warning("weather fetch failed lat=%s lon=%s err=%s", lat, lon, e)
        raise HTTPException(status_code=502, detail="weather_unavailable")
