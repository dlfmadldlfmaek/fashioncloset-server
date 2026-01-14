from pydantic import BaseModel
from typing import List, Optional


# -------------------------------------------------
# 🌤 날씨 응답
# -------------------------------------------------
class WeatherResponse(BaseModel):
    # ⚠️ 날씨 API 실패 / fallback 대비
    temp: Optional[float] = None
    feelsLike: Optional[float] = None
    wind: Optional[float] = None
    pty: str = "UNKNOWN"


# -------------------------------------------------
# 👕 추천 아이템
# -------------------------------------------------
class RecommendItem(BaseModel):
    id: str
    score: float
    finalScore: float


# -------------------------------------------------
# 📦 추천 API 응답
# -------------------------------------------------
class RecommendResponse(BaseModel):
    weather: WeatherResponse
    recommended: List[RecommendItem]
