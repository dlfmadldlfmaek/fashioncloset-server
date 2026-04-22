# schemas/response.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class WeatherResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    temp: Optional[float] = None
    feelsLike: Optional[float] = None
    wind: Optional[float] = None
    pty: str


class RecommendItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    score: float
    finalScore: float

    # app compatibility
    mainCategory: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    # optional extras
    category: Optional[str] = None
    season: Optional[str] = None
    color: Optional[str] = None
    thickness: Optional[str] = None
    imageUrl: Optional[str] = None

    _debug: Optional[Dict[str, Any]] = None


class RecommendResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    weather: WeatherResponse
    recommended: List[RecommendItem] = Field(default_factory=list)


class OutfitSet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: List[RecommendItem] = Field(default_factory=list)
    outfitScore: float = 0.0

    styleSim: Optional[float] = None       # cosine similarity (-1~1)
    styleScore: Optional[float] = None     # 0~100 변환 점수

    _debug: Optional[Dict[str, Any]] = None


class RecommendOutfitResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    weather: WeatherResponse
    outfits: List[OutfitSet] = Field(default_factory=list)
