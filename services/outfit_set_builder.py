# services/outfit_set_builder.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from services.style_encoder import load_style_anchors, calc_style_similarity


def _upper(x: Optional[str]) -> str:
    return (x or "").strip().upper()


def _norm(x: Optional[str]) -> str:
    return (x or "").strip().lower()


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _canonical_category(raw: Any) -> str:
    s = str(raw or "").strip().upper()
    if s in {"TOP", "BOTTOM", "OUTER"}:
        return s
    if s in {"상의", "탑", "TOPS", "TOPWEAR"}:
        return "TOP"
    if s in {"하의", "바지", "치마", "BOTTOMS", "PANTS", "SKIRT"}:
        return "BOTTOM"
    if s in {"아우터", "겉옷", "OUTWEAR", "COAT", "JACKET"}:
        return "OUTER"
    return ""


def _pick_top(items: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda x: float(x.get("finalScore", 0.0)), reverse=True)[:n]


def _avg_score(items: List[Dict[str, Any]]) -> float:
    if not items:
        return 0.0
    return sum(float(it.get("finalScore", 0.0)) for it in items) / float(len(items))


def _weather_requires_outer(weather: Optional[Dict[str, Any]], *, cold_temp_threshold: float = 12.0) -> bool:
    if not weather:
        return False

    pty = _upper(str(weather.get("pty") or ""))
    if pty in {"RAIN", "SNOW"}:
        return True

    temp_raw = weather.get("temp")
    try:
        temp = float(temp_raw) if temp_raw is not None else None
    except Exception:
        temp = None

    return temp is not None and temp <= cold_temp_threshold


def _dedupe_and_limit(outfits: List[Dict[str, Any]], max_sets: int) -> List[Dict[str, Any]]:
    outfits.sort(key=lambda x: float(x.get("outfitScore", 0.0)), reverse=True)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for o in outfits:
        ids = [str(it.get("id")) for it in (o.get("items") or []) if it and it.get("id")]
        if not ids:
            continue
        sig = "|".join(sorted(ids))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(o)
        if len(out) >= max_sets:
            break
    return out


# -----------------------
# TAG 기반(보조) 스타일
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
    "casual": {"정장", "드레스", "셋업", "턱시도", "수트"},
    "vintage": {"테크웨어", "나일론", "메쉬", "형광"},
}


def _count_hits(tags: List[str], targets: Set[str]) -> int:
    tags_n = [_norm(t) for t in (tags or []) if isinstance(t, str) and t.strip()]
    if not tags_n or not targets:
        return 0
    targets_n = {_norm(t) for t in targets}
    hits = 0
    for t in tags_n:
        for key in targets_n:
            if key in t:
                hits += 1
                break
    return hits


def _style_bonus_tag(items: List[Dict[str, Any]], style: Optional[str]) -> Tuple[float, Dict[str, Any]]:
    style_key = _norm(style)
    if not style_key or style_key not in STYLE_POS:
        return 0.0, {"style": style_key, "pos": 0, "neg": 0, "bonus": 0.0, "method": "tag"}

    # TOP/OUTER 우선
    all_tags: List[str] = []
    preferred_tags: List[str] = []
    for it in items:
        cat = _canonical_category(it.get("mainCategory") or it.get("category"))
        tags = it.get("tags") or []
        all_tags.extend(tags)
        if cat in {"TOP", "OUTER"}:
            preferred_tags.extend(tags)

    tags_for_style = preferred_tags if preferred_tags else all_tags
    pos = _count_hits(tags_for_style, STYLE_POS.get(style_key, set()))
    neg = _count_hits(tags_for_style, STYLE_NEG.get(style_key, set()))

    bonus = 0.0
    if pos >= 6:
        bonus += 6.0
    elif pos >= 4:
        bonus += 4.0
    elif pos >= 2:
        bonus += 2.0
    elif pos >= 1:
        bonus += 1.0

    if neg >= 4:
        bonus -= 7.0
    elif neg >= 2:
        bonus -= 4.0
    elif neg >= 1:
        bonus -= 2.0

    return float(bonus), {"style": style_key, "pos": pos, "neg": neg, "bonus": float(bonus), "method": "tag"}


def _split_by_cat(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_cat: Dict[str, List[Dict[str, Any]]] = {"TOP": [], "BOTTOM": [], "OUTER": []}
    for it in items:
        raw_cat = it.get("mainCategory") or it.get("category")
        cat = _canonical_category(raw_cat)
        if cat in by_cat:
            by_cat[cat].append(it)
    return by_cat


# -----------------------
# Anchor 기반 스타일 보너스
# -----------------------
def _as_vec(x: Any) -> Optional[np.ndarray]:
    try:
        v = np.asarray(x, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if v.size == 0 or not np.isfinite(v).all():
        return None
    return v


def _outfit_image_vector(items: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    vecs: List[np.ndarray] = []
    weights: List[float] = []

    for it in items:
        emb = it.get("imageEmbedding")
        v = _as_vec(emb) if emb is not None else None
        if v is None:
            continue

        cat = _canonical_category(it.get("mainCategory") or it.get("category"))
        w = 1.0
        if cat in {"TOP", "OUTER"}:
            w = 1.25
        elif cat == "BOTTOM":
            w = 0.95

        vecs.append(v)
        weights.append(w)

    if not vecs:
        return None

    dim = int(vecs[0].shape[0])
    for v in vecs[1:]:
        if int(v.shape[0]) != dim:
            return None

    W = np.asarray(weights, dtype=np.float32)
    W = W / max(float(W.sum()), 1e-6)

    M = np.stack(vecs, axis=0)
    out = (M * W.reshape(-1, 1)).sum(axis=0)

    n = float(np.linalg.norm(out))
    if n > 0.0:
        out = (out / n).astype(np.float32)
    return out


def _style_bonus_anchor(
    items: List[Dict[str, Any]],
    style: Optional[str],
    *,
    anchors: Dict[str, np.ndarray],
    weight: float = 8.0,
) -> Tuple[float, Dict[str, Any]]:
    style_key = _norm(style)
    if not style_key:
        return 0.0, {"style": style_key, "sim": 0.0, "bonus": 0.0, "method": "anchor", "reason": "no_style"}

    anchor = anchors.get(style_key)
    if anchor is None:
        return 0.0, {"style": style_key, "sim": 0.0, "bonus": 0.0, "method": "anchor", "reason": "no_anchor"}

    outfit_vec = _outfit_image_vector(items)
    if outfit_vec is None:
        return 0.0, {"style": style_key, "sim": 0.0, "bonus": 0.0, "method": "anchor", "reason": "no_embedding"}

    sim = float(calc_style_similarity(outfit_vec, anchor))
    sim = max(-1.0, min(1.0, sim))
    bonus = float(sim) * float(weight)
    return bonus, {"style": style_key, "sim": round(sim, 4), "bonus": round(bonus, 3), "method": "anchor"}


# -----------------------
# ✅ 조합 품질 점수(룰 기반) 추가
# -----------------------
_NEUTRALS = {"BLACK", "WHITE", "GRAY", "GREY", "BEIGE", "IVORY", "NAVY", "BROWN", "KHAKI", "CREAM", "OFFWHITE", "OFF_WHITE", "CHARCOAL"}
_WARM = {"RED", "ORANGE", "YELLOW", "PINK", "CORAL", "BURGUNDY"}
_COOL = {"BLUE", "GREEN", "PURPLE", "TEAL", "MINT"}
_COLOR_ETC = {"ETC", "UNKNOWN", ""}

_THICK = {"THIN": 0, "LIGHT": 0, "MEDIUM": 1, "THICK": 2, "HEAVY": 3}
_SEASON = {"SPRING", "SUMMER", "FALL", "AUTUMN", "WINTER", "ALL"}

def _color_group(c: str) -> str:
    c = _upper(c)
    if c in _COLOR_ETC:
        return "ETC"
    if c in _NEUTRALS:
        return "NEUTRAL"
    if c in _WARM:
        return "WARM"
    if c in _COOL:
        return "COOL"
    return "OTHER"

def _thickness_level(t: Any) -> int:
    k = _upper(str(t or ""))
    return int(_THICK.get(k, 1))

def _season_ok(item_season: Any, weather_temp: Optional[float]) -> Tuple[bool, str]:
    s = _upper(str(item_season or "ALL"))
    if s not in _SEASON:
        s = "ALL"
    if s == "ALL":
        return True, "ALL"

    # temp가 없으면 시즌 체크를 약하게(OK 처리)
    if weather_temp is None:
        return True, s

    # 온도 기반 계절 적합 (경계선 관대하게)
    if weather_temp >= 26:
        ok = s in {"SUMMER", "ALL"}
    elif weather_temp >= 20:
        ok = s in {"SUMMER", "SPRING", "ALL"}
    elif weather_temp >= 12:
        ok = s in {"SPRING", "FALL", "AUTUMN", "ALL"}
    elif weather_temp >= 5:
        ok = s in {"WINTER", "FALL", "AUTUMN", "ALL"}
    else:
        ok = s in {"WINTER", "ALL"}

    return bool(ok), s

def _style_tag_presence(tags: List[str]) -> Dict[str, int]:
    # 특정 “강한 아이템성” 태그 감지(충돌 패널티용)
    # 하의에 “트랙/조거/스웻” 등 있으면 포멀과 충돌 가능성이 큼
    keys = {
        "formal": {"정장", "셋업", "슬랙스", "셔츠", "블라우스", "로퍼", "포멀"},
        "street": {"스트릿", "힙합", "카고", "테크웨어", "와이드", "조거", "그래픽", "로고", "프린트"},
        "sports": {"트랙", "스웻", "스웨트", "저지", "러닝", "운동", "트레이닝"},
        "denim": {"데님", "청바지", "진"},
    }
    out = {k: 0 for k in keys}
    tn = [_norm(t) for t in (tags or []) if isinstance(t, str)]
    for t in tn:
        for k, bag in keys.items():
            for w in bag:
                if _norm(w) in t:
                    out[k] += 1
                    break
    return out

def _pair_quality_score(
    items: List[Dict[str, Any]],
    *,
    weather_temp: Optional[float],
    need_outer: bool,
) -> Tuple[float, Dict[str, Any]]:
    """
    조합 품질 점수:
    + 색상/톤 매칭
    + 두께/날씨 적합
    - 시즌 충돌
    - 스타일 강충돌(포멀 상의 + 트레이닝/스트릿 하의 등)
    """
    top = next((x for x in items if _canonical_category(x.get("mainCategory") or x.get("category")) == "TOP"), None)
    bottom = next((x for x in items if _canonical_category(x.get("mainCategory") or x.get("category")) == "BOTTOM"), None)
    outer = next((x for x in items if _canonical_category(x.get("mainCategory") or x.get("category")) == "OUTER"), None)

    dbg: Dict[str, Any] = {
        "color": {},
        "season": {},
        "thickness": {},
        "conflict": {},
        "total": 0.0,
    }

    score = 0.0

    # ---- 1) 색상 매칭 (TOP-BOTTOM 중심)
    if top and bottom:
        ct = _color_group(str(top.get("color") or ""))
        cb = _color_group(str(bottom.get("color") or ""))
        dbg["color"] = {"top": ct, "bottom": cb}

        # 구체적 색상도 참조
        top_color = _upper(str(top.get("color") or ""))
        bot_color = _upper(str(bottom.get("color") or ""))

        if "ETC" in {ct, cb}:
            score += 0.0
        elif ct == "NEUTRAL" and cb == "NEUTRAL":
            # 같은 뉴트럴끼리도 세분화
            if top_color == bot_color and top_color not in {"BLACK", "WHITE"}:
                score += 1.0  # 같은 색 뉴트럴은 약간 밋밋
            else:
                score += 2.0
        elif ct == "NEUTRAL" and cb in {"WARM", "COOL", "OTHER"}:
            score += 1.5  # 뉴트럴 + 포인트 = 좋은 조합
        elif cb == "NEUTRAL" and ct in {"WARM", "COOL", "OTHER"}:
            score += 1.5
        elif ct == cb and ct == "WARM":
            score += 0.5  # 따뜻한 색끼리는 톤 충돌 주의
        elif ct == cb and ct == "COOL":
            score += 0.8  # 차가운 색끼리는 괜찮음
        elif {ct, cb} == {"WARM", "COOL"}:
            # 특정 조합은 허용 (네이비+버건디 등)
            if {top_color, bot_color} & {"NAVY", "BLUE"} and {top_color, bot_color} & {"BURGUNDY", "RED"}:
                score += 0.3
            else:
                score -= 0.8  # 기존 -1.2에서 완화
        else:
            score += 0.2

    # ---- 2) 시즌 충돌 패널티 (각 아이템 season vs temp)
    if weather_temp is not None:
        for it in items:
            ok, s = _season_ok(it.get("season"), weather_temp)
            dbg["season"][str(it.get("id") or "")] = {"season": s, "ok": ok}
            if not ok:
                score -= 1.0  # 시즌 미스 페널티 (경계선 완화)

    # ---- 3) 두께/날씨 적합
    if weather_temp is not None:
        # temp가 낮은데 THIN/LIGHT 조합이면 패널티
        levels = [_thickness_level(it.get("thickness")) for it in items]
        avg_lv = sum(levels) / max(len(levels), 1)
        dbg["thickness"] = {"avgLevel": round(avg_lv, 2), "levels": levels, "needOuter": need_outer}

        if weather_temp <= 10:
            if avg_lv <= 0.6 and not outer:
                score -= 2.5
            elif avg_lv <= 0.6 and outer:
                score -= 1.2
            elif avg_lv >= 2.0:
                score += 0.5
        elif weather_temp >= 24:
            if avg_lv >= 2.0:
                score -= 1.5
            elif avg_lv <= 1.0:
                score += 0.4

    # ---- 4) 강한 스타일 충돌 패널티
    if top and bottom:
        top_tags = _style_tag_presence(top.get("tags") or [])
        bottom_tags = _style_tag_presence(bottom.get("tags") or [])
        dbg["conflict"] = {"top": top_tags, "bottom": bottom_tags}

        # 포멀 상의 + 트레이닝/스트릿 하의는 이질감 큼
        if top_tags["formal"] > 0 and (bottom_tags["sports"] > 0 or bottom_tags["street"] > 0):
            score -= 3.5

        # 스트릿/트레이닝 상의 + 포멀 하의도 감점
        if (top_tags["sports"] > 0 or top_tags["street"] > 0) and bottom_tags["formal"] > 0:
            score -= 2.8

        # 미니멀인데 그래픽/로고 과다 -> 약간 감점(태그만으로 과한 컷 방지)
        if top_tags["street"] > 2 and bottom_tags["formal"] > 0:
            score -= 1.5

    dbg["total"] = round(score, 3)
    return float(score), dbg


# -----------------------
# 체형별 코디 보너스
# -----------------------
_BODY_TYPE_TAG_BONUS: Dict[str, Dict[str, float]] = {
    "하체비만": {
        # 좋은 아이템: A라인, 와이드팬츠, 상의 포인트
        "pos": {"A라인", "플레어", "와이드", "부츠컷", "미디", "롱스커트", "하이웨이스트"},
        "neg": {"스키니", "타이트", "레깅스", "숏팬츠", "핫팬츠", "미니스커트"},
    },
    "상체비만": {
        "pos": {"V넥", "브이넥", "랩", "세로줄", "스트라이프", "어두운", "블랙상의"},
        "neg": {"퍼프", "벌룬", "패드숄더", "터틀넥", "하이넥", "프릴", "러플"},
    },
    "통통체형": {
        "pos": {"세로라인", "스트레이트", "슬림핏", "허리마크", "하이웨이스트", "모노톤"},
        "neg": {"오버사이즈", "박시", "가로줄", "보더", "볼륨", "프릴"},
    },
    "마른체형": {
        "pos": {"레이어드", "오버사이즈", "볼륨", "프릴", "러플", "패딩", "퍼프"},
        "neg": {"슬림핏", "스키니", "타이트"},
    },
    "역삼각형": {
        "pos": {"와이드팬츠", "A라인", "플레어", "카고", "볼륨하의", "밝은하의"},
        "neg": {"패드숄더", "퍼프소매", "보트넥", "오프숄더"},
    },
    "골반넓음": {
        "pos": {"스트레이트", "부츠컷", "A라인", "롱아우터", "하이웨이스트"},
        "neg": {"스키니", "타이트", "핫팬츠", "힙라인"},
    },
    "골반좁음": {
        "pos": {"카고", "와이드", "플리츠", "주름", "포켓디테일", "배기"},
        "neg": {"스트레이트", "슬림핏"},
    },
    "어깨좁음": {
        "pos": {"패드숄더", "퍼프소매", "보트넥", "오프숄더", "스퀘어넥", "숄더패드"},
        "neg": {"래글런", "드롭숄더", "나시", "슬리브리스"},
    },
}


def _body_type_bonus(items: List[Dict[str, Any]], body_type: Optional[str]) -> Tuple[float, Dict[str, Any]]:
    bt = (body_type or "").strip()
    if not bt or bt == "균형체형" or bt not in _BODY_TYPE_TAG_BONUS:
        return 0.0, {"bodyType": bt, "pos": 0, "neg": 0, "bonus": 0.0}

    rules = _BODY_TYPE_TAG_BONUS[bt]
    pos_tags = rules.get("pos", set())
    neg_tags = rules.get("neg", set())

    all_tags: List[str] = []
    for it in items:
        all_tags.extend(it.get("tags") or [])

    pos = _count_hits(all_tags, pos_tags)
    neg = _count_hits(all_tags, neg_tags)

    bonus = 0.0
    if pos >= 3:
        bonus += 4.0
    elif pos >= 2:
        bonus += 2.5
    elif pos >= 1:
        bonus += 1.0

    if neg >= 3:
        bonus -= 5.0
    elif neg >= 2:
        bonus -= 3.0
    elif neg >= 1:
        bonus -= 1.5

    return bonus, {"bodyType": bt, "pos": pos, "neg": neg, "bonus": round(bonus, 2)}


# -----------------------
# Main
# -----------------------
def build_outfit_sets(
    scored_items: List[Dict[str, Any]],
    *,
    weather: Optional[Dict[str, Any]] = None,
    style: Optional[str] = None,
    top_n_each: int = 6,
    max_sets: int = 10,
    cold_temp_threshold: float = 12.0,
    min_base_score: float = 0.0,
    min_outfit_score: float = 0.0,
    threshold_include_style: bool = True,
    threshold_include_weather: bool = True,
    allow_singles: bool = True,
    min_sets: int = 3,  # ✅ 최소 3세트 보장
    anchor_weight: float = 8.0,
    tag_weight: float = 1.0,
    # ✅ 조합 품질 점수 가중치
    quality_weight: float = 1.0,
    body_type: Optional[str] = None,
) -> List[Dict[str, Any]]:

    if not scored_items:
        return []

    anchors = load_style_anchors()

    # 1) min_base_score 필터
    if min_base_score and min_base_score > 0:
        filtered = [it for it in scored_items if float(it.get("finalScore", 0.0)) >= float(min_base_score)]
    else:
        filtered = scored_items

    # 2) 필터 후 TOP/BOTTOM 비면 완화
    by_cat_filtered = _split_by_cat(filtered)
    if (not by_cat_filtered["TOP"]) or (not by_cat_filtered["BOTTOM"]):
        by_cat_all = _split_by_cat(scored_items)
        if not by_cat_filtered["TOP"] and by_cat_all["TOP"]:
            filtered += _pick_top(by_cat_all["TOP"], top_n_each)
        if not by_cat_filtered["BOTTOM"] and by_cat_all["BOTTOM"]:
            filtered += _pick_top(by_cat_all["BOTTOM"], top_n_each)

        seen_ids = set()
        uniq: List[Dict[str, Any]] = []
        for it in filtered:
            iid = str(it.get("id") or "")
            if not iid or iid in seen_ids:
                continue
            seen_ids.add(iid)
            uniq.append(it)
        filtered = uniq

    if not filtered:
        return []

    by_cat = _split_by_cat(filtered)
    tops = _pick_top(by_cat["TOP"], top_n_each)
    bottoms = _pick_top(by_cat["BOTTOM"], top_n_each)
    outers = _pick_top(by_cat["OUTER"], max(3, top_n_each // 2))

    temp = weather.get("temp") if weather else None
    try:
        temp_f = float(temp) if temp is not None else None
    except Exception:
        temp_f = None

    pty = _upper(str(weather.get("pty") or "")) if weather else ""
    need_outer = _weather_requires_outer(weather, cold_temp_threshold=cold_temp_threshold)
    strict_outer = _env_flag("STRICT_OUTFIT_OUTER", "0")

    def _finalize(
        items: List[Dict[str, Any]],
        mode: str,
        *,
        _min_outfit: float,
        _include_style: bool,
        _include_weather: bool,
        _apply_style_penalty: bool,
        _style_penalty: float = 6.0,
    ) -> Optional[Dict[str, Any]]:
        base = _avg_score(items)

        # weather bonus
        w_bonus = 0.0
        if _include_weather:
            if pty in {"RAIN", "SNOW"}:
                w_bonus += 1.5
            if need_outer and any(_canonical_category(it.get("mainCategory") or it.get("category")) == "OUTER" for it in items):
                w_bonus += 0.8

        # style bonus: anchor + tag
        s_bonus_total = 0.0
        dbg_style: Dict[str, Any] = {"style": _norm(style)}

        if _include_style:
            a_bonus, a_dbg = _style_bonus_anchor(items, style, anchors=anchors, weight=float(anchor_weight))
            s_bonus_total += a_bonus
            dbg_style["anchor"] = a_dbg

            t_bonus, t_dbg = _style_bonus_tag(items, style)
            t_bonus_scaled = float(t_bonus) * float(tag_weight)
            s_bonus_total += t_bonus_scaled
            dbg_style["tag"] = {**t_dbg, "scaledBonus": round(t_bonus_scaled, 3)}

            # 전멸 방지: 컷 대신 패널티
            neg = int((t_dbg or {}).get("neg", 0))
            if _apply_style_penalty and neg >= 2 and float(t_dbg.get("bonus", 0.0)) <= 0.0:
                s_bonus_total -= float(_style_penalty)
                dbg_style["penalty"] = float(_style_penalty)
            else:
                dbg_style["penalty"] = 0.0

        # ✅ 조합 품질 점수(룰 기반)
        q_bonus, q_dbg = _pair_quality_score(items, weather_temp=temp_f, need_outer=need_outer)
        q_bonus_scaled = float(q_bonus) * float(quality_weight)

        # ✅ 체형 보너스
        bt_bonus, bt_dbg = _body_type_bonus(items, body_type)

        score = float(base) + float(w_bonus) + float(s_bonus_total) + float(q_bonus_scaled) + float(bt_bonus)
        score = max(0.0, score)

        if _min_outfit and score < float(_min_outfit):
            return None

        return {
            "items": items,
            "outfitScore": round(score, 3),
            "_debug": {
                "mode": mode,
                "needOuter": need_outer,
                "strictOuter": strict_outer,
                "pty": pty,
                "temp": temp_f,
                "base": round(float(base), 3),
                "weatherBonus": round(float(w_bonus), 3),
                "style": dbg_style,
                "quality": {**q_dbg, "scaled": round(q_bonus_scaled, 3), "weight": float(quality_weight)},
                "bodyType": bt_dbg,
                "minOutfit": float(_min_outfit),
                "includeStyle": bool(_include_style),
                "includeWeather": bool(_include_weather),
                "minSets": int(min_sets),
            },
        }

    def _generate(
        *,
        _min_outfit: float,
        _include_style: bool,
        _include_weather: bool,
        _apply_style_penalty: bool,
    ) -> List[Dict[str, Any]]:
        outfits: List[Dict[str, Any]] = []

        if tops and bottoms:
            # 3피스
            if outers:
                outer_candidates = outers[:3]
                for t in tops:
                    for b in bottoms:
                        for o in outer_candidates:
                            out = _finalize(
                                [t, b, o],
                                "TOP+BOTTOM+OUTER",
                                _min_outfit=_min_outfit,
                                _include_style=_include_style,
                                _include_weather=_include_weather,
                                _apply_style_penalty=_apply_style_penalty,
                            )
                            if out:
                                outfits.append(out)

            # 2피스
            if not (strict_outer and need_outer and _include_weather):
                for t in tops:
                    for b in bottoms:
                        out = _finalize(
                            [t, b],
                            "TOP+BOTTOM",
                            _min_outfit=_min_outfit,
                            _include_style=_include_style,
                            _include_weather=_include_weather,
                            _apply_style_penalty=_apply_style_penalty,
                        )
                        if out:
                            outfits.append(out)

        # 최소 세트 채우기(전멸 방지)
        if allow_singles and len(outfits) < int(min_sets):
            for t in tops:
                o = _finalize(
                    [t],
                    "SINGLE_TOP",
                    _min_outfit=_min_outfit,
                    _include_style=_include_style,
                    _include_weather=_include_weather,
                    _apply_style_penalty=_apply_style_penalty,
                )
                if o:
                    o["_debug"]["reason"] = "FILL_MIN_SETS"
                    outfits.append(o)

            for b in bottoms:
                o = _finalize(
                    [b],
                    "SINGLE_BOTTOM",
                    _min_outfit=_min_outfit,
                    _include_style=_include_style,
                    _include_weather=_include_weather,
                    _apply_style_penalty=_apply_style_penalty,
                )
                if o:
                    o["_debug"]["reason"] = "FILL_MIN_SETS"
                    outfits.append(o)

            for o2 in outers:
                o = _finalize(
                    [o2],
                    "SINGLE_OUTER",
                    _min_outfit=_min_outfit,
                    _include_style=_include_style,
                    _include_weather=_include_weather,
                    _apply_style_penalty=_apply_style_penalty,
                )
                if o:
                    o["_debug"]["reason"] = "FILL_MIN_SETS"
                    outfits.append(o)

        return _dedupe_and_limit(outfits, max_sets)

    # Pass 1
    out = _generate(
        _min_outfit=float(min_outfit_score),
        _include_style=bool(threshold_include_style),
        _include_weather=bool(threshold_include_weather),
        _apply_style_penalty=True,
    )
    if len(out) >= int(min_sets):
        return out

    # Pass 2 (완화)
    relaxed_min = max(0.0, float(min_outfit_score) - 6.0)
    out2 = _generate(
        _min_outfit=relaxed_min,
        _include_style=bool(threshold_include_style),
        _include_weather=bool(threshold_include_weather),
        _apply_style_penalty=False,
    )
    out = _dedupe_and_limit(out + out2, max_sets)
    if len(out) >= int(min_sets):
        return out

    # Pass 3 (스타일 OFF)
    out3 = _generate(
        _min_outfit=max(0.0, relaxed_min - 6.0),
        _include_style=False,
        _include_weather=bool(threshold_include_weather),
        _apply_style_penalty=False,
    )
    out = _dedupe_and_limit(out + out3, max_sets)
    return out
