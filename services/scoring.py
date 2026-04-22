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


def personalization_weight(
    item: Any,
    pref: Dict[str, Any],
    learning_weights: Dict[str, Any] | None = None,
) -> float:
    """
    Preference-based additive weight.
    Phase 3: learning_weights가 있으면 Firestore의 학습된 가중치를 곱연산으로 통합.

    pref:
      {"category": {str: float}, "season": {str: float}, "color": {str: float}}
    learning_weights:
      {"categoryWeight": {str: float}, "colorWeight": {str: float}, "seasonWeight": {str: float}}
    """
    if not pref:
        return 0.0

    category = _norm(getattr(item, "category", None) or getattr(item, "mainCategory", None), "TOP")
    season = _norm(getattr(item, "season", None), "ALL")
    color = _norm(getattr(item, "color", None), "UNKNOWN")

    cat_w = float(pref.get("category", {}).get(category, 0.0))
    sea_w = float(pref.get("season", {}).get(season, 0.0))
    col_w = float(pref.get("color", {}).get(color, 0.0))

    # 기본 가중치 계수
    cat_coeff = 1.0
    sea_coeff = 0.5
    col_coeff = 0.8

    # Phase 3: 학습 가중치 통합 (곱연산)
    if learning_weights:
        lw_cat = float(learning_weights.get("categoryWeight", {}).get(category, 0.0))
        lw_sea = float(learning_weights.get("seasonWeight", {}).get(season, 0.0))
        lw_col = float(learning_weights.get("colorWeight", {}).get(color, 0.0))

        # 학습 가중치를 승수로 변환: 0.0 -> 1.0, +1.0 -> 2.0, -1.0 -> 0.0
        cat_coeff *= (1.0 + lw_cat)
        sea_coeff *= (1.0 + lw_sea)
        col_coeff *= (1.0 + lw_col)

    w = 0.0
    w += cat_w * cat_coeff
    w += sea_w * sea_coeff
    w += col_w * col_coeff
    return w


def recently_worn_penalty(last_worn_at: Any) -> float:
    """Penalize recently worn items based on last_worn_at (ms or sec epoch)."""
    if last_worn_at in (None, 0, "0", ""):
        return 0.0

    try:
        ts = int(float(last_worn_at))
    except Exception:
        return 0.0

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
