# schemas/request.py
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict
from pydantic.aliases import AliasChoices


class ClothesItem(BaseModel):
    """
    lastWornAt: unix epoch seconds (or ms). Server should normalize if mixed.
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str

    category: str = Field(
        default="TOP",
        validation_alias=AliasChoices("category", "mainCategory"),
    )

    tags: List[str] = Field(default_factory=list)

    season: Optional[str] = None
    color: Optional[str] = None
    thickness: Optional[str] = None  # was "THIN"

    lastWornAt: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("lastWornAt", "lastWornDate"),
    )

    imageUrl: Optional[str] = None


class RecommendRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    userId: str = Field(..., min_length=1)
    lat: float = Field(default=0.0, ge=-90, le=90)
    lon: float = Field(default=0.0, ge=-180, le=180)
    temp: Optional[float] = Field(default=None, description="Temperature in Celsius. If provided, skips weather API.")
    style: Optional[str] = None
    clothes: List[ClothesItem]
    excludeItemSets: Optional[List[List[str]]] = Field(
        default=None,
        description="List of item-id sets to exclude from outfit results (e.g. [['id1','id2'], ['id3','id4']])",
    )
    bodyType: Optional[str] = Field(
        default=None,
        description="User body type for personalized recommendations",
    )


class LikeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    userId: str
    id: str
    mainCategory: str
    tags: List[str] = Field(default_factory=list)
    liked: bool = True
