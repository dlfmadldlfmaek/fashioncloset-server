# services/premium_policy.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta

from services.premium import is_premium_user

# 베타 스타일 목록 (필요한 만큼 추가)
BETA_STYLES = {"gorpcore", "archive", "workwear", "y2k", "techwear_beta"}

@dataclass(frozen=True)
class PremiumPolicy:
    # 추천/코디 더보기
    more_outfits_without_ad: bool
    max_outfit_sets: int

    # 스타일
    can_use_beta_styles: bool

    # 캘린더 보관
    calendar_retention_days: int  # 무료 90일(대략 3개월), 유료는 -1(무제한)

def get_policy(user_id: str) -> PremiumPolicy:
    if is_premium_user(user_id):
        return PremiumPolicy(
            more_outfits_without_ad=True,
            max_outfit_sets=10,          # 유료는 더 많이
            can_use_beta_styles=True,
            calendar_retention_days=-1,  # 무제한
        )
    return PremiumPolicy(
        more_outfits_without_ad=False,
        max_outfit_sets=3,             # 무료 기본
        can_use_beta_styles=False,     # 베타 스타일 제한
        calendar_retention_days=90,    # 3개월
    )

def is_beta_style(style: str | None) -> bool:
    s = (style or "").strip().lower()
    return s in BETA_STYLES
