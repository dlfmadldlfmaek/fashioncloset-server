# schemas/request.py
from pydantic import BaseModel
from typing import List, Optional


class ClothesItem(BaseModel):
    id: str
    mainCategory: str
    season: str
    color: str

    # 앱에서 넘어오는 선택 필드
    lastWornDate: Optional[str] = None
    imageUrl: Optional[str] = None


class RecommendRequest(BaseModel):
    userId: str
    lat: float
    lon: float

    # 없어도 안전
    style: Optional[str] = None

    clothes: List[ClothesItem]
