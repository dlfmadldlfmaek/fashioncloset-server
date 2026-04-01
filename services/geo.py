# utils/geo.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class KmaGridParams:
    """
    KMA (Korea Meteorological Administration) DFS grid conversion parameters.

    This implements Lambert Conformal Conic projection used by KMA's 5km grid.
    """
    re_km: float = 6371.00877
    grid_km: float = 5.0
    slat1_deg: float = 30.0
    slat2_deg: float = 60.0
    olon_deg: float = 126.0
    olat_deg: float = 38.0
    xo: float = 43.0
    yo: float = 136.0


@dataclass(frozen=True)
class _KmaGridPrecomp:
    re: float
    sn: float
    sf: float
    ro: float
    olon: float


def _precompute(p: KmaGridParams) -> _KmaGridPrecomp:
    deg2rad = math.pi / 180.0

    re = p.re_km / p.grid_km
    slat1 = p.slat1_deg * deg2rad
    slat2 = p.slat2_deg * deg2rad
    olon = p.olon_deg * deg2rad
    olat = p.olat_deg * deg2rad

    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(
        math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    )

    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = (sf**sn) * math.cos(slat1) / sn

    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro**sn)

    return _KmaGridPrecomp(re=re, sn=sn, sf=sf, ro=ro, olon=olon)


_DEFAULT_PARAMS = KmaGridParams()
_DEFAULT_PRECOMP = _precompute(_DEFAULT_PARAMS)


def _round_half_up(v: float) -> int:
    """
    Why: int(v + 0.5) fails for negatives; this is symmetric half-up.
    """
    return int(math.floor(v + 0.5)) if v >= 0 else int(math.ceil(v - 0.5))


def latlon_to_grid(lat: float, lon: float, *, params: KmaGridParams = _DEFAULT_PARAMS) -> Tuple[int, int]:
    """
    Convert WGS84 (lat, lon) to KMA DFS grid (x, y).

    Args:
        lat: Latitude in degrees (-90..90)
        lon: Longitude in degrees (-180..180)
        params: KMA grid parameters (default: 5km grid)

    Returns:
        (x, y) integer grid coordinates.
    """
    if not (-90.0 <= float(lat) <= 90.0):
        raise ValueError(f"lat out of range: {lat}")
    if not (-180.0 <= float(lon) <= 180.0):
        raise ValueError(f"lon out of range: {lon}")

    # If custom params are passed, precompute them once per call-site (simple approach).
    # For high-frequency use with custom params, you can memoize _precompute(params).
    pre = _DEFAULT_PRECOMP if params is _DEFAULT_PARAMS else _precompute(params)

    deg2rad = math.pi / 180.0
    ra = math.tan(math.pi * 0.25 + float(lat) * deg2rad * 0.5)
    ra = pre.re * pre.sf / (ra**pre.sn)

    theta = float(lon) * deg2rad - pre.olon
    theta *= pre.sn

    x = _round_half_up(ra * math.sin(theta) + params.xo)
    y = _round_half_up(pre.ro - ra * math.cos(theta) + params.yo)

    return x, y
