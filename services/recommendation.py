# services/recommendation.py
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, Set, Dict
from zoneinfo import ZoneInfo

SEOUL = ZoneInfo("Asia/Seoul")


def _norm(s: str) -> str:
    return str(s or "").strip().lower()


def get_current_style_context(now: Optional[datetime] = None) -> str:
    """
    Default style context by Seoul time (fallback when user doesn't pick a style).
    """
    now = now.astimezone(SEOUL) if now else datetime.now(SEOUL)

    if 0 <= now.weekday() <= 3:
        return "minimal"
    if now.weekday() == 4:
        return "street" if now.hour >= 18 else "minimal"
    return "casual"


# -----------------------
# ✅ Style keyword profiles
# -----------------------
STYLE_POS: Dict[str, Set[str]] = {
    "casual": {
        "후드", "맨투맨", "티셔츠", "데님", "청바지", "스웨트", "조거", "스니커즈", "캐주얼",
        "니트", "가디건", "후리스", "운동화", "코튼", "면", "편한", "일상", "라운드넥",
        "크루넥", "롤업", "체크", "스트라이프", "폴로", "반팔", "반바지", "치노", "캔버스",
        "플리스", "집업", "져지", "피케", "헨리넥", "린넨", "면바지",
    },
    "formal": {
        "셔츠", "블라우스", "슬랙스", "코트", "자켓", "정장", "드레스", "로퍼", "포멀",
        "더비", "옥스포드", "셋업", "수트", "타이", "넥타이", "울", "캐시미어", "턱시도",
        "테일러드", "구두", "힐", "펌프스", "클래식", "카라", "와이셔츠", "드레스셔츠",
        "핀스트라이프", "체스터", "트렌치",
    },
    "minimal": {
        "무지", "미니멀", "베이직", "솔리드", "심플", "클린", "모노", "모노톤", "단색",
        "오프화이트", "아이보리", "그레이", "블랙", "화이트", "뉴트럴", "톤온톤", "슬림",
        "핏", "스트레이트", "텍스처", "울", "코튼", "린넨", "실크", "캐시미어",
        "차분", "절제", "깔끔", "정돈", "네이비", "베이지", "카키",
    },
    "street": {
        "스트릿", "오버사이즈", "힙합", "카고", "테크웨어", "그래픽", "로고", "프린트",
        "와이드", "조거", "후드", "스니커즈", "데님", "나이키", "아디다스", "뉴발란스",
        "컨버스", "반스", "볼캡", "버킷햇", "맨투맨", "바시티", "트랙", "나일론",
        "메쉬", "형광", "레터링", "패치", "빅로고", "스케이트", "보드",
    },
    "vintage": {
        "빈티지", "레트로", "클래식", "워싱", "플리츠", "트위드", "코듀로이", "올드스쿨",
        "앤틱", "70년대", "80년대", "90년대", "빈티지워싱", "디스트로이드", "페이드",
        "브라운", "버건디", "머스타드", "카키", "패치워크", "자수", "핸드메이드",
    },
}

# ✅ "이 스타일이면 싫어할만한 키워드(감점)"도 같이 둠
STYLE_NEG: Dict[str, Set[str]] = {
    "minimal": {
        "스트릿", "오버사이즈", "힙합", "카고", "테크웨어", "그래픽", "로고", "프린트", "패턴", "카모",
        "와이드", "조거", "형광", "빅로고", "레터링", "나일론", "메쉬",
    },
    "street": {
        "정장", "드레스", "블라우스", "포멀", "로퍼", "옥스포드", "더비", "셋업", "슬랙스",
        "캐시미어", "턱시도", "수트", "테일러드",
    },
    "formal": {
        "후드", "맨투맨", "스웨트", "조거", "스트릿", "힙합", "카고", "그래픽", "로고",
        "트랙", "나일론", "형광", "스케이트", "오버사이즈",
    },
    "casual": {
        "정장", "드레스", "셋업", "턱시도", "수트",
    },
    "vintage": {"테크웨어", "나일론", "메쉬", "형광"},
}


def _count_hits(tags: Iterable[str], targets: Set[str]) -> int:
    """
    Count hits using substring match.
    """
    tag_list = [_norm(t) for t in (tags or []) if isinstance(t, str) and t.strip()]
    if not tag_list or not targets:
        return 0

    targets_n = {_norm(t) for t in targets}

    hits = 0
    for t in tag_list:
        for key in targets_n:
            if key in t:
                hits += 1
                break
    return hits


def apply_time_score(item: object, style: Optional[str] = None, *, now: Optional[datetime] = None) -> float:
    """
    ✅ 스타일 분리 강하게 버전 (street vs minimal 차이 확실히)
    - 선택 스타일 pos hit: 강하게 가산
    - 선택 스타일 neg hit: 강하게 감산
    - 다른 스타일 pos가 더 잘 맞으면 감산(기존보다 강함)
    """
    style_key = _norm(style or "")
    if not style_key:
        style_key = get_current_style_context(now=now)

    tags = getattr(item, "tags", None) or []

    # pos/neg hit
    hit_pos_map = {k: _count_hits(tags, v) for k, v in STYLE_POS.items()}
    chosen_pos = hit_pos_map.get(style_key, 0)

    other_pos = [v for k, v in hit_pos_map.items() if k != style_key]
    max_other_pos = max(other_pos) if other_pos else 0

    chosen_neg = _count_hits(tags, STYLE_NEG.get(style_key, set()))

    # -----------------------
    # 1) 기본 가산 (선택 스타일 pos)
    # -----------------------
    # ✅ 기존보다 더 벌려줌
    if chosen_pos >= 4:
        mult = 1.45
    elif chosen_pos == 3:
        mult = 1.33
    elif chosen_pos == 2:
        mult = 1.22
    elif chosen_pos == 1:
        mult = 1.12
    else:
        mult = 1.00

    # -----------------------
    # 2) 감산 (선택 스타일 neg)
    # -----------------------
    # ✅ minimal에 스트릿 요소가 섞이면 확실히 떨어지게
    if chosen_neg >= 3:
        mult *= 0.78
    elif chosen_neg == 2:
        mult *= 0.86
    elif chosen_neg == 1:
        mult *= 0.93

    # -----------------------
    # 3) 감산 (다른 스타일이 더 잘 맞는 경우)
    # -----------------------
    # ✅ 기존보다 강하게 감산해서 "비슷비슷" 방지
    if chosen_pos == 0 and max_other_pos >= 2:
        mult *= 0.78
    elif chosen_pos == 0 and max_other_pos == 1:
        mult *= 0.88
    elif chosen_pos == 1 and max_other_pos >= 3:
        mult *= 0.85
    elif chosen_pos == 1 and max_other_pos == 2:
        mult *= 0.92

    # 하한/상한 (너무 튀는 것 방지)
    if mult < 0.70:
        mult = 0.70
    if mult > 1.55:
        mult = 1.55

    return float(mult)
