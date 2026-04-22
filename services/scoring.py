# services/scoring.py
from __future__ import annotations

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _norm(v: Any, default: str) -> str:
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    return s.upper()


def personalization_weight(item: Any, pref: Dict[str, Any]) -> float:
    """
    Preference-based additive weight.

    Expects pref:
      {
        "category": Mapping[str, float],
        "season":   Mapping[str, float],
        "color":    Mapping[str, float],
      }
    """
    if not pref:
        return 0.0

    category = _norm(getattr(item, "category", None) or getattr(item, "mainCategory", None), "TOP")
    season = _norm(getattr(item, "season", None), "ALL")
    color = _norm(getattr(item, "color", None), "UNKNOWN")

    cat_w = float(pref.get("category", {}).get(category, 0.0))
    sea_w = float(pref.get("season", {}).get(season, 0.0))
    col_w = float(pref.get("color", {}).get(color, 0.0))

    # Tuneable weights
    w = 0.0
    w += cat_w * 1.0
    w += sea_w * 0.5
    w += col_w * 0.8
    return w


def recently_worn_penalty(last_worn_at: Any) -> float:
    """
    Penalize recently worn items based on last_worn_at (ms or sec epoch).
    """
    if last_worn_at in (None, 0, "0", ""):
        return 0.0

    try:
        ts = int(float(last_worn_at))
    except Exception:
        return 0.0

    # If it's seconds (10 digits-ish), convert to ms
    if ts < 10_000_000_000:
        ts *= 1000

    now_ms = int(time.time() * 1000)
    diff_ms = now_ms - ts

    if diff_ms < 0:
        logger.debug("recently_worn_penalty got future timestamp: %s", last_worn_at)
        return 0.0

    diff_days = diff_ms / 86_400_000.0

    if diff_days < 1.0:
        return -100.0
    if diff_days < 3.0:
        return -30.0
    if diff_days < 5.0:
        return -10.0
    return 0.0
