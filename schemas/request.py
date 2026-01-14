from pydantic import BaseModel
from typing import List

class ClothesItem(BaseModel):
    id: str
    mainCategory: str
    season: str
    color: str

class RecommendRequest(BaseModel):
    userId: str
    lat: float
    lon: float
    clothes: List[ClothesItem]

class LikeLogRequest(BaseModel):
    userId: str
    id: str
    mainCategory: str
    season: str
    color: str
