# tests/test_geo.py
"""Tests for KMA DFS grid conversion (services/geo.py)."""
from __future__ import annotations

import pytest

from services.geo import latlon_to_grid


class TestLatLonToGrid:
    def test_seoul(self):
        """Seoul City Hall: known reference point for KMA grid."""
        x, y = latlon_to_grid(37.5665, 126.9780)
        # Seoul should be roughly nx=60, ny=127 area
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert 55 <= x <= 65, f"Seoul x={x} out of expected range"
        assert 120 <= y <= 135, f"Seoul y={y} out of expected range"

    def test_busan(self):
        """Busan: lat=35.1796, lon=129.0756."""
        x, y = latlon_to_grid(35.1796, 129.0756)
        assert isinstance(x, int)
        assert isinstance(y, int)
        # Busan is east/south of Seoul -> higher x, lower y
        assert 95 <= x <= 105, f"Busan x={x} out of expected range"
        assert 70 <= y <= 85, f"Busan y={y} out of expected range"

    def test_jeju(self):
        """Jeju: lat=33.4996, lon=126.5312."""
        x, y = latlon_to_grid(33.4996, 126.5312)
        assert isinstance(x, int)
        assert isinstance(y, int)
        # Jeju is far south
        assert 50 <= x <= 60, f"Jeju x={x} out of expected range"
        assert 30 <= y <= 45, f"Jeju y={y} out of expected range"

    def test_returns_integer_tuple(self):
        x, y = latlon_to_grid(37.0, 127.0)
        assert isinstance(x, int)
        assert isinstance(y, int)

    def test_equator_prime_meridian(self):
        """Edge case: (0, 0) should not crash, though outside Korea."""
        x, y = latlon_to_grid(0.0, 0.0)
        assert isinstance(x, int)
        assert isinstance(y, int)

    def test_lat_out_of_range_raises(self):
        with pytest.raises(ValueError, match="lat out of range"):
            latlon_to_grid(91.0, 127.0)

    def test_lat_negative_out_of_range_raises(self):
        with pytest.raises(ValueError, match="lat out of range"):
            latlon_to_grid(-91.0, 127.0)

    def test_lon_out_of_range_raises(self):
        with pytest.raises(ValueError, match="lon out of range"):
            latlon_to_grid(37.0, 181.0)

    def test_lon_negative_out_of_range_raises(self):
        with pytest.raises(ValueError, match="lon out of range"):
            latlon_to_grid(37.0, -181.0)

    def test_boundary_values_positive(self):
        """Positive boundary lat/lon should not raise ValueError."""
        latlon_to_grid(90.0, 180.0)

    def test_boundary_lat_negative_90_raises_math_error(self):
        """lat=-90 causes ZeroDivisionError in Lambert projection (south pole)."""
        with pytest.raises(ZeroDivisionError):
            latlon_to_grid(-90.0, -180.0)

    def test_zero_zero(self):
        """(0, 0) should not raise."""
        latlon_to_grid(0.0, 0.0)
