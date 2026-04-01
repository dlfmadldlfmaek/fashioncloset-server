# services/diversify.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


def diversify(
    items: List[Dict[str, Any]],
    *,
    limit: Optional[int] = None,
    max_per_category: int = 2,
    max_per_color: int = 2,
    key_id: str = "id",
    key_category: str = "mainCategory",
    key_color: str = "color",
) -> List[Dict[str, Any]]:
    """
    Pick top items with diversity constraints (stable greedy).

    Behavior:
      - Tries to satisfy both category and color limits.
      - If not enough, relaxes color constraint.
      - If still not enough, relaxes both.
      - Preserves input order as much as possible.

    Expected input order:
      - Usually sorted by finalScore desc before calling.
    """
    if not items:
        return []

    def norm(v: Optional[Any]) -> str:
        if v is None:
            return ""
        return str(v).strip().upper()

    target = limit if (limit is not None and limit > 0) else len(items)

    selected: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    cat_count: Dict[str, int] = {}
    color_count: Dict[str, int] = {}

    def can_take(it: Dict[str, Any], *, relax_color: bool, relax_cat: bool) -> bool:
        it_id = str(it.get(key_id, ""))
        if it_id and it_id in seen_ids:
            return False

        cat = norm(it.get(key_category))
        col = norm(it.get(key_color))

        cat_ok = True if relax_cat or cat == "" else (cat_count.get(cat, 0) < max_per_category)
        col_ok = True if relax_color or col == "" else (color_count.get(col, 0) < max_per_color)
        return cat_ok and col_ok

    def take(it: Dict[str, Any]) -> None:
        it_id = str(it.get(key_id, ""))
        if it_id:
            seen_ids.add(it_id)

        cat = norm(it.get(key_category))
        col = norm(it.get(key_color))

        if cat:
            cat_count[cat] = cat_count.get(cat, 0) + 1
        if col:
            color_count[col] = color_count.get(col, 0) + 1

        selected.append(it)

    # Pass 1: strict (category+color)
    for it in items:
        if len(selected) >= target:
            break
        if can_take(it, relax_color=False, relax_cat=False):
            take(it)

    # Pass 2: relax color only
    if len(selected) < target:
        for it in items:
            if len(selected) >= target:
                break
            if can_take(it, relax_color=True, relax_cat=False):
                take(it)

    # Pass 3: relax both (fill)
    if len(selected) < target:
        for it in items:
            if len(selected) >= target:
                break
            if can_take(it, relax_color=True, relax_cat=True):
                take(it)

    return selected
