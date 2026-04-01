# tests/test_schemas.py
"""Tests for Pydantic request/response schema validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.request import ClothesItem, LikeRequest, RecommendRequest


# ---------------------------------------------------------------------------
# ClothesItem
# ---------------------------------------------------------------------------
class TestClothesItem:
    def test_minimal_valid(self):
        item = ClothesItem(id="abc123")
        assert item.id == "abc123"
        assert item.category == "TOP"  # default
        assert item.tags == []
        assert item.season is None
        assert item.color is None
        assert item.lastWornAt is None
        assert item.imageUrl is None

    def test_full_fields(self):
        item = ClothesItem(
            id="item-1",
            category="BOTTOM",
            tags=["casual", "daily"],
            season="SUMMER",
            color="BLUE",
            thickness="THIN",
            lastWornAt=1700000000000,
            imageUrl="https://example.com/img.jpg",
        )
        assert item.category == "BOTTOM"
        assert item.tags == ["casual", "daily"]
        assert item.season == "SUMMER"
        assert item.lastWornAt == 1700000000000

    def test_alias_mainCategory(self):
        """mainCategory alias should map to category field."""
        item = ClothesItem(**{"id": "x", "mainCategory": "OUTER"})
        assert item.category == "OUTER"

    def test_alias_lastWornDate(self):
        """lastWornDate alias should map to lastWornAt field."""
        item = ClothesItem(**{"id": "x", "lastWornDate": 1700000000})
        assert item.lastWornAt == 1700000000

    def test_extra_fields_ignored(self):
        """Extra fields should be silently ignored (extra='ignore')."""
        item = ClothesItem(id="x", unknownField="value")
        assert item.id == "x"
        assert not hasattr(item, "unknownField")


# ---------------------------------------------------------------------------
# RecommendRequest
# ---------------------------------------------------------------------------
class TestRecommendRequest:
    def test_valid_request(self):
        req = RecommendRequest(
            userId="user-1",
            lat=37.5665,
            lon=126.9780,
            clothes=[ClothesItem(id="c1")],
        )
        assert req.userId == "user-1"
        assert req.lat == 37.5665
        assert req.lon == 126.9780
        assert req.style is None
        assert len(req.clothes) == 1

    def test_valid_request_with_style(self):
        req = RecommendRequest(
            userId="u1",
            lat=35.0,
            lon=129.0,
            style="casual",
            clothes=[],
        )
        assert req.style == "casual"

    def test_missing_userId_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            RecommendRequest(lat=37.0, lon=127.0, clothes=[])
        assert "userId" in str(exc_info.value)

    def test_missing_lat_raises(self):
        with pytest.raises(ValidationError):
            RecommendRequest(userId="u1", lon=127.0, clothes=[])

    def test_missing_lon_raises(self):
        with pytest.raises(ValidationError):
            RecommendRequest(userId="u1", lat=37.0, clothes=[])

    def test_empty_userId_raises(self):
        with pytest.raises(ValidationError):
            RecommendRequest(userId="", lat=37.0, lon=127.0, clothes=[])

    def test_lat_out_of_range_positive(self):
        with pytest.raises(ValidationError):
            RecommendRequest(userId="u1", lat=91.0, lon=127.0, clothes=[])

    def test_lat_out_of_range_negative(self):
        with pytest.raises(ValidationError):
            RecommendRequest(userId="u1", lat=-91.0, lon=127.0, clothes=[])

    def test_lon_out_of_range_positive(self):
        with pytest.raises(ValidationError):
            RecommendRequest(userId="u1", lat=37.0, lon=181.0, clothes=[])

    def test_lon_out_of_range_negative(self):
        with pytest.raises(ValidationError):
            RecommendRequest(userId="u1", lat=37.0, lon=-181.0, clothes=[])

    def test_boundary_lat_lon(self):
        """Boundary values should be accepted."""
        req = RecommendRequest(userId="u1", lat=90.0, lon=180.0, clothes=[])
        assert req.lat == 90.0
        assert req.lon == 180.0

        req2 = RecommendRequest(userId="u1", lat=-90.0, lon=-180.0, clothes=[])
        assert req2.lat == -90.0
        assert req2.lon == -180.0


# ---------------------------------------------------------------------------
# LikeRequest
# ---------------------------------------------------------------------------
class TestLikeRequest:
    def test_valid_like(self):
        req = LikeRequest(userId="u1", id="item-1", mainCategory="TOP")
        assert req.userId == "u1"
        assert req.id == "item-1"
        assert req.mainCategory == "TOP"
        assert req.tags == []
        assert req.liked is True

    def test_unlike(self):
        req = LikeRequest(
            userId="u1", id="item-1", mainCategory="BOTTOM", liked=False
        )
        assert req.liked is False

    def test_with_tags(self):
        req = LikeRequest(
            userId="u1",
            id="item-1",
            mainCategory="OUTER",
            tags=["street", "casual"],
        )
        assert req.tags == ["street", "casual"]

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            LikeRequest(userId="u1")  # missing id, mainCategory
