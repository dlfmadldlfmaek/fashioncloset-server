# schemas/response.py
from pydantic import BaseModel
from typing import List, Optional


class WeatherResponse(BaseModel):
    temp: float
    pty: str
    feelsLike: Optional[float] = None
    wind: Optional[float] = None


class RecommendItem(BaseModel):
    id: str
    score: float
    finalScore: float


class RecommendResponse(BaseModel):
    weather: WeatherResponse
    recommended: List[RecommendItem]
