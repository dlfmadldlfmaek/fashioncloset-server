# services/scoring.py
from datetime import datetime
from typing import Dict


def personalization_weight(item, pref: Dict) -> float:
    score = 0.0
    score += pref.get("category", {}).get(item.mainCategory, 0.0) * 0.6
    score += pref.get("season", {}).get(item.season, 0.0) * 0.25
    score += pref.get("color", {}).get(item.color, 0.0) * 0.15
    return score


def recently_worn_penalty(last_worn_date: str | None) -> float:
    if not last_worn_date:
        return 0.0
    try:
        worn = datetime.strptime(last_worn_date, "%Y-%m-%d")
        days = (datetime.now() - worn).days
        if days <= 1:
            return -0.6
        if days <= 3:
            return -0.3
        if days <= 7:
            return -0.15
    except Exception:
        pass
    return 0.0
