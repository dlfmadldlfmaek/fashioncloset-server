# tests/test_quota_logic.py
"""Tests for quota helper functions (pure logic, no Firebase)."""
from __future__ import annotations

import os
import re
import time

from api.quota import FREE_DAILY_QUOTA, PREMIUM_DAILY_QUOTA, _day_key, _get_limit


class TestDayKey:
    def test_returns_date_string_format(self):
        key = _day_key()
        # Should be YYYY-MM-DD format
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", key), f"Unexpected format: {key}"

    def test_known_timestamp_utc(self, monkeypatch):
        """Unix epoch 1.0 in UTC is 1970-01-01 (force UTC mode).
        Note: now=0.0 is falsy, so _day_key falls back to time.time().
        """
        monkeypatch.setattr("api.quota.RESET_TZ", "UTC")
        key = _day_key(now=1.0)  # 1 second after epoch
        assert key == "1970-01-01"

    def test_specific_date_utc(self, monkeypatch):
        """2026-03-16 12:00:00 UTC -> 2026-03-16 in UTC mode."""
        monkeypatch.setattr("api.quota.RESET_TZ", "UTC")
        # 2026-03-16 12:00:00 UTC
        ts = 1773662400.0
        key = _day_key(now=ts)
        assert key == "2026-03-16"

    def test_now_zero_uses_current_time(self):
        """now=0.0 is falsy, so _day_key should fallback to current time."""
        key_default = _day_key()
        key_zero = _day_key(now=0.0)
        # Both should return today's date (not 1970-01-01)
        assert key_zero == key_default

    def test_deterministic(self):
        """Same input should always return same output."""
        ts = 1700000000.0
        assert _day_key(now=ts) == _day_key(now=ts)

    def test_different_days_different_keys(self):
        ts1 = 1700000000.0
        ts2 = ts1 + 86400  # next day
        assert _day_key(now=ts1) != _day_key(now=ts2)


class TestGetLimit:
    def test_premium_limit(self):
        limit = _get_limit(premium=True)
        assert limit == PREMIUM_DAILY_QUOTA

    def test_free_limit(self):
        limit = _get_limit(premium=False)
        assert limit == FREE_DAILY_QUOTA

    def test_premium_greater_than_free(self):
        assert _get_limit(True) > _get_limit(False)
