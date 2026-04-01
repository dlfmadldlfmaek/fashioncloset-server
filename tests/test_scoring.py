# tests/test_scoring.py
"""Tests for scoring utility functions (services/scoring.py)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pytest

from services.scoring import personalization_weight, recently_worn_penalty


# ---------------------------------------------------------------------------
# Helper: lightweight item mock
# ---------------------------------------------------------------------------
@dataclass
class _FakeItem:
    category: Optional[str] = None
    mainCategory: Optional[str] = None
    season: Optional[str] = None
    color: Optional[str] = None


# ---------------------------------------------------------------------------
# recently_worn_penalty
# ---------------------------------------------------------------------------
class TestRecentlyWornPenalty:
    def test_none_returns_zero(self):
        assert recently_worn_penalty(None) == 0.0

    def test_zero_int_returns_zero(self):
        assert recently_worn_penalty(0) == 0.0

    def test_zero_string_returns_zero(self):
        assert recently_worn_penalty("0") == 0.0

    def test_empty_string_returns_zero(self):
        assert recently_worn_penalty("") == 0.0

    def test_worn_today_penalized_heavily(self):
        """Item worn less than 1 day ago -> -100."""
        now_ms = int(time.time() * 1000)
        one_hour_ago = now_ms - 3_600_000
        assert recently_worn_penalty(one_hour_ago) == -100.0

    def test_worn_2_days_ago(self):
        """Item worn 2 days ago -> -30."""
        now_ms = int(time.time() * 1000)
        two_days_ago = now_ms - (2 * 86_400_000)
        assert recently_worn_penalty(two_days_ago) == -30.0

    def test_worn_4_days_ago(self):
        """Item worn 4 days ago -> -10."""
        now_ms = int(time.time() * 1000)
        four_days_ago = now_ms - (4 * 86_400_000)
        assert recently_worn_penalty(four_days_ago) == -10.0

    def test_worn_6_days_ago_no_penalty(self):
        """Item worn 6 days ago -> 0."""
        now_ms = int(time.time() * 1000)
        six_days_ago = now_ms - (6 * 86_400_000)
        assert recently_worn_penalty(six_days_ago) == 0.0

    def test_worn_30_days_ago_no_penalty(self):
        now_ms = int(time.time() * 1000)
        thirty_days_ago = now_ms - (30 * 86_400_000)
        assert recently_worn_penalty(thirty_days_ago) == 0.0

    def test_epoch_seconds_auto_converted(self):
        """If timestamp is in seconds (10 digits), it should auto-convert to ms."""
        now_sec = int(time.time())
        one_hour_ago_sec = now_sec - 3600
        assert recently_worn_penalty(one_hour_ago_sec) == -100.0

    def test_future_timestamp_returns_zero(self):
        """Future timestamps should return 0 (no penalty)."""
        future_ms = int(time.time() * 1000) + 86_400_000
        assert recently_worn_penalty(future_ms) == 0.0

    def test_invalid_string_returns_zero(self):
        assert recently_worn_penalty("not_a_number") == 0.0

    def test_float_string_accepted(self):
        """Float-formatted string timestamps should be parsed."""
        now_ms = int(time.time() * 1000)
        one_hour_ago = now_ms - 3_600_000
        assert recently_worn_penalty(str(float(one_hour_ago))) == -100.0


# ---------------------------------------------------------------------------
# personalization_weight
# ---------------------------------------------------------------------------
class TestPersonalizationWeight:
    def test_empty_pref_returns_zero(self):
        item = _FakeItem(category="TOP", season="SUMMER", color="BLUE")
        assert personalization_weight(item, {}) == 0.0

    def test_none_pref_returns_zero(self):
        item = _FakeItem(category="TOP")
        assert personalization_weight(item, None) == 0.0

    def test_category_preference(self):
        item = _FakeItem(category="TOP")
        pref = {"category": {"TOP": 10.0}, "season": {}, "color": {}}
        result = personalization_weight(item, pref)
        assert result == pytest.approx(10.0)  # cat_w * 1.0

    def test_season_preference(self):
        item = _FakeItem(season="SUMMER")
        pref = {"category": {}, "season": {"SUMMER": 6.0}, "color": {}}
        result = personalization_weight(item, pref)
        assert result == pytest.approx(3.0)  # sea_w * 0.5

    def test_color_preference(self):
        item = _FakeItem(color="BLUE")
        pref = {"category": {}, "season": {}, "color": {"BLUE": 5.0}}
        result = personalization_weight(item, pref)
        assert result == pytest.approx(4.0)  # col_w * 0.8

    def test_combined_preferences(self):
        item = _FakeItem(category="BOTTOM", season="WINTER", color="BLACK")
        pref = {
            "category": {"BOTTOM": 2.0},
            "season": {"WINTER": 4.0},
            "color": {"BLACK": 5.0},
        }
        expected = 2.0 * 1.0 + 4.0 * 0.5 + 5.0 * 0.8  # 2 + 2 + 4 = 8.0
        result = personalization_weight(item, pref)
        assert result == pytest.approx(expected)

    def test_missing_category_uses_default_TOP(self):
        """When category is None, default is TOP."""
        item = _FakeItem(category=None)
        pref = {"category": {"TOP": 3.0}, "season": {}, "color": {}}
        result = personalization_weight(item, pref)
        assert result == pytest.approx(3.0)

    def test_mainCategory_fallback(self):
        """If category is None, mainCategory should be used."""
        item = _FakeItem(category=None, mainCategory="OUTER")
        pref = {"category": {"OUTER": 7.0}, "season": {}, "color": {}}
        result = personalization_weight(item, pref)
        assert result == pytest.approx(7.0)

    def test_case_normalization(self):
        """Values should be upper-cased before lookup."""
        item = _FakeItem(category="bottom", season="summer", color="blue")
        pref = {
            "category": {"BOTTOM": 1.0},
            "season": {"SUMMER": 2.0},
            "color": {"BLUE": 3.0},
        }
        expected = 1.0 * 1.0 + 2.0 * 0.5 + 3.0 * 0.8
        result = personalization_weight(item, pref)
        assert result == pytest.approx(expected)

    def test_unmatched_preference_returns_zero(self):
        """If item attributes don't match any pref keys, weight is 0."""
        item = _FakeItem(category="DRESS", season="FALL", color="RED")
        pref = {
            "category": {"TOP": 5.0},
            "season": {"SUMMER": 5.0},
            "color": {"BLUE": 5.0},
        }
        assert personalization_weight(item, pref) == 0.0
