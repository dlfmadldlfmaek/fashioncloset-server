# api/recommend.py

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Query, Request

from services.rate_limit import limiter

from schemas.request import RecommendRequest
from schemas.response import (
    RecommendItem,
    RecommendOutfitResponse,
    RecommendResponse,
    OutfitSet,
    WeatherResponse,
)
from services.diversify import diversify
from services.learning import get_learning_weights
from services.outfit_encoder import encode_outfit_image_from_url
from services.outfit_set_builder import build_outfit_sets
from services.premium import is_premium_user
from services.recommend_cache import (
    build_cache_key,
    clothes_hash,
    get_cached_recommend,
    set_cached_recommend,
)
from services.recommendation import apply_time_score, get_current_style_context
from services.scoring import personalization_weight, recently_worn_penalty
from services.style_encoder import calc_style_similarity, load_style_anchors
from services.style_vector import get_user_style_vector
from services.user_service import load_user_preference
from services.weather import get_current_weather

router = APIRouter(prefix="/recommend", tags=["recommend"])
logger = logging.getLogger("recommend")


def _build_weather(req) -> WeatherResponse:
    if req.temp is not None:
        return WeatherResponse(temp=float(req.temp), feelsLike=float(req.temp), wind=0, pty="SUNNY")
    try:
        w_raw = get_current_weather(req.lat, req.lon)
        return WeatherResponse(
            temp=w_raw["temp"],
            feelsLike=w_raw.get("feelsLike"),
            wind=w_raw.get("wind"),
            pty=w_raw["pty"],
        )
    except Exception as e:
        logger.warning("[WEATHER FAIL] %s", e)
        return WeatherResponse(temp=20.0, feelsLike=20.0, wind=0, pty="SUNNY")


# -----------------------
# Style list (base + expanded + beta)
# -----------------------
BASE_STYLES = ["casual", "formal", "minimal", "street", "vintage"]
# Phase 1: 확장 스타일
EXPANDED_STYLES = ["gorpcore", "workwear", "preppy", "romantic", "sporty"]
ALL_STYLES = BASE_STYLES + EXPANDED_STYLES

BETA_STYLES = [
    "workwear_beta",
    "gorpcore_beta",
    "sporty_beta",
    "date_night_beta",
]

# -----------------------
# Premium policy
# -----------------------
PREMIUM_DEFAULT_MAX_SETS = 6
FREE_DEFAULT_MAX_SETS = 3
PREMIUM_HARD_CAP_MAX_SETS = 30

RECOMMEND_TOTAL = 10
RECOMMEND_MIN_TOP = 3
RECOMMEND_MIN_BOTTOM = 3


# -----------------------
# helpers (style gating)
# -----------------------
def _is_beta_style(style: Optional[str]) -> bool:
    s = (style or "").strip().lower()
    return s in set(BETA_STYLES)


def _enforce_style_for_user(style: Optional[str], premium: bool) -> str:
    s = (style or "").strip().lower()
    if not s:
        return "casual"

    if _is_beta_style(s) and (not premium):
        return "casual"

    if (s not in ALL_STYLES) and (not _is_beta_style(s)):
        return "casual"

    return s


def _enforce_styles_for_user(styles: Optional[List[str]], premium: bool) -> List[str]:
    """Phase 5: 멀티스타일 검증."""
    if not styles:
        return []
    return [_enforce_style_for_user(s, premium) for s in styles]


# -----------------------
# helpers (category/metadata)
# -----------------------
def normalize_category(raw: str) -> str:
    s = (raw or "").strip().upper()
    if not s:
        return "TOP"

    if s in {"TOP", "상의", "SHIRT", "TEE", "TSHIRT", "T-SHIRT", "HOODIE", "SWEATER", "KNIT"}:
        return "TOP"
    if s in {"BOTTOM", "하의", "PANTS", "JEANS", "DENIM", "SKIRT", "SLACKS"}:
        return "BOTTOM"
    if s in {"OUTER", "아우터", "JACKET", "COAT", "PADDING", "PARKA", "CARDIGAN"}:
        return "OUTER"

    if s in {
        "SHOES", "신발", "SNEAKER", "SNEAKERS", "BOOTS", "BOOT", "HEELS", "HEEL",
        "LOAFER", "LOAFERS", "SANDAL", "SANDALS", "SLIPPER", "SLIPPERS"
    }:
        return "SHOES"
    if s in {"BAG", "가방", "BACKPACK", "TOTE", "CROSS", "CROSSBAG", "SHOULDERBAG", "CLUTCH"}:
        return "BAG"
    if s in {"ACC", "ACCESSORY", "액세서리", "악세서리", "HAT", "CAP", "BELT", "SCARF", "MUFFLER", "SUNGLASSES", "WATCH"}:
        return "ACC"

    if ("BOTTOM" in s) or ("PANT" in s) or ("JEAN" in s) or ("SKIRT" in s) or ("하의" in s) or ("바지" in s):
        return "BOTTOM"
    if ("OUTER" in s) or ("COAT" in s) or ("JACKET" in s) or ("패딩" in s) or ("아우터" in s):
        return "OUTER"

    if any(k in s for k in ["SHOE", "SNEAKER", "BOOT", "HEEL", "LOAFER", "SANDAL", "SLIPPER", "신발", "운동화", "구두", "부츠", "샌들"]):
        return "SHOES"
    if any(k in s for k in ["BAG", "BACKPACK", "TOTE", "CLUTCH", "가방", "백팩"]):
        return "BAG"
    if any(k in s for k in ["ACC", "ACCESS", "HAT", "CAP", "BELT", "SCARF", "SUNGLASS", "WATCH", "모자", "벨트", "머플러", "시계"]):
        return "ACC"

    return "TOP"


def extract_info_from_tags(tags: List[str]) -> Tuple[str, str]:
    season_keywords = {
        "SPRING": "SPRING", "봄": "SPRING",
        "SUMMER": "SUMMER", "여름": "SUMMER",
        "AUTUMN": "AUTUMN", "FALL": "AUTUMN", "가을": "AUTUMN",
        "WINTER": "WINTER", "겨울": "WINTER",
    }
    thickness_keywords = {
        "THIN": "THIN", "얇음": "THIN", "SHEER": "THIN", "LINEN": "THIN", "린넨": "THIN",
        "MEDIUM": "MEDIUM",
        "THICK": "THICK", "HEAVY": "THICK", "FLEECE": "THICK", "기모": "THICK", "두꺼움": "THICK",
        "PADDING": "PADDING", "패딩": "PADDING",
    }

    detected_season = "ALL"
    detected_thickness = "MEDIUM"

    for tag in tags or []:
        upper_tag = str(tag).upper()
        for key, val in season_keywords.items():
            if key.upper() in upper_tag:
                detected_season = val
        for key, val in thickness_keywords.items():
            if key.upper() in upper_tag:
                detected_thickness = val

    return detected_season, detected_thickness


def calculate_weather_match(item, weather: WeatherResponse) -> float:
    score = 0.0
    temp = weather.temp if weather.temp is not None else 20.0

    thick = (getattr(item, "thickness", None) or "MEDIUM").upper()
    season = (getattr(item, "season", None) or "ALL").upper()
    cat = normalize_category(getattr(item, "category", None) or "TOP")
    tags = getattr(item, "tags", None) or []

    if temp <= 0:
        if thick in ["THICK", "PADDING", "HEAVY"]:
            score += 12
        elif thick == "MEDIUM":
            score += 2
        elif thick in ["THIN", "SHEER", "LIGHT"]:
            score -= 15
    elif temp <= 10:
        if thick in ["THICK", "PADDING", "HEAVY"]:
            score += 8
        elif thick == "MEDIUM":
            score += 4
        elif thick in ["THIN", "SHEER"]:
            score -= 8
    elif temp <= 20:
        if thick == "MEDIUM":
            score += 5
        elif thick in ["THIN", "LIGHT"]:
            score += 3
        elif thick in ["THICK", "PADDING"]:
            score -= 5
    elif temp <= 27:
        if thick in ["THIN", "SHEER", "LIGHT"]:
            score += 8
        elif thick == "MEDIUM":
            score += 2
        elif thick in ["THICK", "PADDING"]:
            score -= 15
    else:
        if thick in ["THIN", "SHEER", "LIGHT"]:
            score += 10
        elif thick in ["THICK", "PADDING"]:
            score -= 20

    if temp <= 5 and season in {"WINTER"}:
        score += 6
    elif temp <= 12 and season in {"WINTER", "FALL", "AUTUMN"}:
        score += 4
    elif 12 < temp <= 22 and season in {"SPRING", "FALL", "AUTUMN"}:
        score += 4
    elif temp >= 20 and season in {"SUMMER", "SPRING"}:
        score += 5

    if temp >= 25 and season == "WINTER":
        score -= 8
    elif temp <= 5 and season == "SUMMER":
        score -= 8

    if weather.pty in ["RAIN", "SNOW"]:
        if cat == "BOTTOM" and any(str(t).strip().upper() == "LONG" for t in tags):
            score -= 2
        if cat == "OUTER":
            score += 3

    return score


def _cat_counts(scored: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for it in scored:
        cat = str(it.get("mainCategory") or it.get("category") or "UNKNOWN")
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    if p <= 0:
        return arr[0]
    if p >= 1:
        return arr[-1]
    k = (len(arr) - 1) * p
    f = int(k)
    c = min(f + 1, len(arr) - 1)
    if f == c:
        return arr[f]
    return arr[f] + (arr[c] - arr[f]) * (k - f)


def _top_scores_by_cat(scored: List[dict], cat: str, k: int) -> List[float]:
    xs = [
        float(x.get("finalScore", 0.0))
        for x in scored
        if (x.get("mainCategory") == cat or x.get("category") == cat)
    ]
    xs.sort(reverse=True)
    return xs[:k]


def _tb_base_distribution(scored: List[dict], k: int = 6) -> List[float]:
    tops = _top_scores_by_cat(scored, "TOP", k)
    bottoms = _top_scores_by_cat(scored, "BOTTOM", k)
    if not tops or not bottoms:
        return []
    return [(t + b) / 2.0 for t in tops for b in bottoms]


def _adaptive_floor(ref: float, hard_floor: float, soft_floor: float) -> float:
    if ref <= 0:
        return soft_floor
    if ref < hard_floor:
        return min(hard_floor, max(soft_floor, ref - 0.8))
    return hard_floor


# -----------------------
# helpers (style score on OutfitSet)
# -----------------------
def _safe_vec(x: object) -> Optional[np.ndarray]:
    try:
        v = np.asarray(x, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if v.size == 0 or not np.isfinite(v).all():
        return None
    return v


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n <= 0.0 else (v / n).astype(np.float32)


def _outfit_embedding_from_items(items: List[dict]) -> Optional[np.ndarray]:
    vecs: List[np.ndarray] = []
    dim: Optional[int] = None

    for it in items:
        url = (it.get("imageUrl") or "").strip()
        if not url:
            continue
        try:
            raw = encode_outfit_image_from_url(url)
        except Exception:
            continue

        v = _safe_vec(raw)
        if v is None:
            continue

        if dim is None:
            dim = int(v.shape[0])
        elif int(v.shape[0]) != dim:
            continue

        vecs.append(v)

    if not vecs:
        return None

    mean = np.mean(np.stack(vecs, axis=0), axis=0)
    return _l2norm(mean)


def _sim_to_score(sim: float) -> float:
    s = float(sim)
    if s < -1.0:
        s = -1.0
    if s > 1.0:
        s = 1.0
    return round((s + 1.0) * 50.0, 1)


# -----------------------
# Phase 3: 앵커 + 유저 스타일 벡터 혼합
# -----------------------
def _get_mixed_anchor(
    style_key: str,
    user_id: str,
    anchors: Dict[str, np.ndarray],
    *,
    global_weight: float = 0.5,
) -> Optional[np.ndarray]:
    """글로벌 앵커 0.5 + 개인 벡터 0.5 혼합."""
    global_anchor = anchors.get(style_key)

    user_vec = None
    try:
        user_vec = get_user_style_vector(user_id, style_key)
    except Exception:
        pass

    if global_anchor is None and user_vec is None:
        return None
    if global_anchor is not None and user_vec is None:
        return global_anchor
    if global_anchor is None and user_vec is not None:
        return _l2norm(user_vec)

    # 둘 다 있으면 가중 평균
    ga = _safe_vec(global_anchor)
    uv = _safe_vec(user_vec)
    if ga is None:
        return _l2norm(uv) if uv is not None else None
    if uv is None:
        return ga

    if ga.shape[0] != uv.shape[0]:
        return ga

    mixed = global_weight * ga + (1.0 - global_weight) * uv
    return _l2norm(mixed)


# -----------------------
# Scoring (items)
# -----------------------
def _clip_style_score(image_url: str, style_key: str, anchors: dict) -> Tuple[float, Optional[list]]:
    if not image_url or not style_key or not anchors:
        return 0.0, None

    anchor = anchors.get(style_key.lower())
    if anchor is None:
        return 0.0, None

    try:
        item_vec = encode_outfit_image_from_url(image_url)
        vec = _safe_vec(item_vec)
        if vec is None:
            return 0.0, None

        anchor_vec = _safe_vec(anchor)
        if anchor_vec is None or vec.shape[0] != anchor_vec.shape[0]:
            return 0.0, vec.tolist()

        sim = float(calc_style_similarity(vec, anchor_vec))
        sim = max(-1.0, min(1.0, sim))
        bonus = sim * 6.0 if sim > 0 else sim * 3.0
        return round(bonus, 2), vec.tolist()
    except Exception as e:
        logger.debug("[CLIP_ITEM] failed url=%s err=%s", image_url[:60], e)
        return 0.0, None


def _score_items_raw(
    req: RecommendRequest,
    weather: WeatherResponse,
    style_ctx_override: Optional[str] = None,
) -> List[dict]:
    pref = load_user_preference(req.userId)

    # Phase 3: 학습 가중치 로드
    learning_weights = get_learning_weights(req.userId)

    style_ctx = (style_ctx_override or req.style or "").strip().lower()
    if not style_ctx:
        style_ctx = get_current_style_context()

    anchors = load_style_anchors()

    results: List[dict] = []
    base_score = 50.0

    for item in req.clothes:
        try:
            raw_cat = getattr(item, "category", None) or getattr(item, "mainCategory", None) or "TOP"
            item.category = normalize_category(str(raw_cat))

            inferred_season, inferred_thickness = extract_info_from_tags(getattr(item, "tags", []) or [])
            item.season = inferred_season
            item.thickness = inferred_thickness

            # Phase 3: 학습 가중치를 personalization_weight에 전달
            pref_score = personalization_weight(item, pref, learning_weights=learning_weights) * 10
            worn_penalty = recently_worn_penalty(getattr(item, "lastWornAt", None))
            weather_score = calculate_weather_match(item, weather)
            tpo_multiplier = apply_time_score(item, style_ctx)

            image_url = (getattr(item, "imageUrl", None) or "").strip()
            clip_bonus, embedding = _clip_style_score(image_url, style_ctx, anchors)

            final_score = (base_score + pref_score + weather_score + worn_penalty) * tpo_multiplier + clip_bonus
            if final_score < 0:
                final_score = 0.0

            base: dict = {}
            try:
                if hasattr(item, "model_dump"):
                    base = item.model_dump()
            except Exception:
                base = {}

            result_item = {
                **base,
                "id": item.id,
                "category": item.category,
                "mainCategory": item.category,
                "score": round(base_score, 2),
                "finalScore": round(final_score, 3),
                "tags": getattr(item, "tags", []) or [],
                "season": item.season,
                "color": getattr(item, "color", None),
                "thickness": item.thickness,
                "_debug": {
                    "pref": round(pref_score, 1),
                    "weather": weather_score,
                    "wornPenalty": worn_penalty,
                    "tpo": tpo_multiplier,
                    "clipBonus": clip_bonus,
                    "style": style_ctx,
                    "inferred": f"{inferred_season}/{inferred_thickness}",
                    "cat": item.category,
                    "hasLearningWeights": bool(learning_weights),
                },
            }

            if embedding is not None:
                result_item["imageEmbedding"] = embedding

            results.append(result_item)
        except Exception as e:
            logger.warning("[ITEM SKIP] id=%s err=%s", getattr(item, "id", "?"), e)

    results.sort(key=lambda x: x["finalScore"], reverse=True)
    return results


def _score_items(req: RecommendRequest, weather: WeatherResponse, style_ctx_override: Optional[str] = None) -> List[dict]:
    results = _score_items_raw(req, weather, style_ctx_override=style_ctx_override)
    try:
        results = diversify(results, max_per_category=6, max_per_color=4)
    except Exception:
        pass
    return results


# -----------------------
# Outfit policy switch
# -----------------------
@dataclass(frozen=True)
class OutfitPolicy:
    include_style_in_threshold: bool
    include_weather_in_threshold: bool
    base_ref_pctl: float
    base_margin: float
    final_margin: float
    base_floor: float
    final_floor: float
    allow_singles: bool


STYLE_POLICIES: Dict[str, OutfitPolicy] = {
    "casual": OutfitPolicy(False, True, 0.80, 3.5, 1.8, 60.0, 61.5, False),
    "formal": OutfitPolicy(True, True, 0.80, 3.0, 1.6, 61.0, 63.5, False),
    "minimal": OutfitPolicy(True, True, 0.78, 3.2, 1.7, 60.5, 63.0, False),
    "street": OutfitPolicy(True, True, 0.78, 3.2, 1.7, 60.0, 62.8, False),
    "vintage": OutfitPolicy(True, True, 0.78, 3.2, 1.7, 60.0, 62.8, False),
}
DEFAULT_POLICY = OutfitPolicy(True, True, 0.78, 3.5, 1.8, 60.0, 61.5, False)


def _get_policy(style: Optional[str]) -> OutfitPolicy:
    key = (style or "default").strip().lower()
    return STYLE_POLICIES.get(key, DEFAULT_POLICY)


# -----------------------
# Balanced recommend list
# -----------------------
def pick_balanced_recommend(scored: List[dict], total: int = 10, min_top: int = 3, min_bottom: int = 3) -> List[dict]:
    for x in scored:
        cat = normalize_category(str(x.get("mainCategory") or x.get("category") or "TOP"))
        x["mainCategory"] = cat
        x["category"] = cat

    tops = [x for x in scored if x["mainCategory"] == "TOP"]
    bottoms = [x for x in scored if x["mainCategory"] == "BOTTOM"]

    out: List[dict] = []
    out += tops[:min_top]
    out += bottoms[:min_bottom]

    used = {x.get("id") for x in out}
    rest = [x for x in scored if x.get("id") not in used]

    out += rest[: max(0, total - len(out))]
    return out[:total]


# -----------------------
# Routes
# -----------------------
@router.get("/styles")
@limiter.limit("30/minute")
def recommend_styles(request: Request, userId: str = Query(..., description="user id")):
    premium = is_premium_user(userId)
    styles: List[Dict[str, Any]] = []

    for s in BASE_STYLES:
        styles.append({"key": s, "label": s, "isBeta": False, "premiumRequired": False})

    # Phase 1: 확장 스타일도 노출
    for s in EXPANDED_STYLES:
        styles.append({"key": s, "label": s, "isBeta": False, "premiumRequired": False})

    for s in BETA_STYLES:
        styles.append({"key": s, "label": s, "isBeta": True, "premiumRequired": True})

    return {"premium": premium, "styles": styles}


@router.post("", response_model=RecommendResponse)
@limiter.limit("30/minute")
def recommend(req: RecommendRequest, background_tasks: BackgroundTasks, request: Request):
    premium = is_premium_user(req.userId)
    effective_style = _enforce_style_for_user(req.style, premium)

    logger.info(
        "[RECOMMEND] user=%s premium=%s style=%s->%s items=%d",
        req.userId, premium, (req.style or ""), effective_style, len(req.clothes)
    )

    weather = _build_weather(req)

    if not req.clothes:
        return RecommendResponse(weather=weather, recommended=[])

    cache_key: Optional[str] = None
    try:
        clothes_key = clothes_hash([c.model_dump() for c in req.clothes])
        style_for_cache = effective_style or "default"
        temp_key = int(req.temp) if req.temp is not None else 0
        cache_key = build_cache_key(req.userId, temp_key, 0, style_for_cache, clothes_key)

        cached = get_cached_recommend(cache_key)
        if cached:
            return RecommendResponse(**cached)
    except Exception:
        pass

    results = _score_items(req, weather, style_ctx_override=effective_style)
    balanced = pick_balanced_recommend(
        results,
        total=RECOMMEND_TOTAL,
        min_top=RECOMMEND_MIN_TOP,
        min_bottom=RECOMMEND_MIN_BOTTOM,
    )
    response_data = {"weather": weather, "recommended": balanced}

    if cache_key:
        try:
            set_cached_recommend(cache_key, response_data, minutes=20)
        except Exception:
            pass

    return RecommendResponse(**response_data)


@router.post("/outfits", response_model=RecommendOutfitResponse)
@limiter.limit("30/minute")
def recommend_outfits(
    req: RecommendRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    maxSets: Optional[int] = Query(
        None,
        ge=1,
        le=PREMIUM_HARD_CAP_MAX_SETS,
        description="how many outfit sets to return",
    ),
):
    premium = is_premium_user(req.userId)
    effective_style = _enforce_style_for_user(req.style, premium)

    # Phase 5: 멀티스타일 처리
    effective_styles = _enforce_styles_for_user(req.styles, premium) if req.styles else None

    if maxSets is None:
        effective_max_sets = PREMIUM_DEFAULT_MAX_SETS if premium else FREE_DEFAULT_MAX_SETS
    else:
        if premium:
            effective_max_sets = min(int(maxSets), PREMIUM_HARD_CAP_MAX_SETS)
        else:
            effective_max_sets = min(int(maxSets), FREE_DEFAULT_MAX_SETS)

    logger.info(
        "[RECOMMEND_OUTFITS] user=%s premium=%s style=%s->%s styles=%s items=%d maxSets=%s effectiveMax=%d",
        req.userId, premium, (req.style or ""), effective_style,
        effective_styles, len(req.clothes), maxSets, effective_max_sets
    )

    weather = _build_weather(req)

    if not req.clothes:
        return RecommendOutfitResponse(weather=weather, outfits=[])

    scored = _score_items_raw(req, weather, style_ctx_override=effective_style)

    if req.excludeItemSets:
        rng = np.random.default_rng()
        for item in scored:
            noise = rng.uniform(-3.0, 3.0)
            item["finalScore"] = max(0.0, float(item.get("finalScore", 0.0)) + noise)
        scored.sort(key=lambda x: x["finalScore"], reverse=True)

    scored_outfit = [
        x for x in scored
        if (x.get("mainCategory") or x.get("category")) in {"TOP", "BOTTOM", "OUTER"}
    ]

    logger.info(
        "[OUTFITS] usable=%d (TOP=%d BOTTOM=%d OUTER=%d) removed=%d",
        len(scored_outfit),
        sum(1 for x in scored_outfit if x.get("mainCategory") == "TOP"),
        sum(1 for x in scored_outfit if x.get("mainCategory") == "BOTTOM"),
        sum(1 for x in scored_outfit if x.get("mainCategory") == "OUTER"),
        len(scored) - len(scored_outfit),
    )

    counts = _cat_counts(scored_outfit)
    policy = _get_policy(effective_style)

    base_dist = _tb_base_distribution(scored_outfit, k=6)
    base_ref = _percentile(base_dist, policy.base_ref_pctl) if base_dist else 0.0

    base_floor = _adaptive_floor(base_ref, policy.base_floor, soft_floor=52.0)
    final_floor = _adaptive_floor(base_ref, policy.final_floor, soft_floor=53.5)

    if base_ref > 0:
        min_base_score = max(base_floor, base_ref - policy.base_margin)
        min_outfit_score = max(final_floor, base_ref - policy.final_margin)
    else:
        min_base_score = base_floor
        min_outfit_score = final_floor

    logger.info(
        "[OUTFITS] counts=%s base_ref=%.3f base_floor=%.3f final_floor=%.3f min_base=%.3f min_outfit=%.3f",
        counts, base_ref, base_floor, final_floor, min_base_score, min_outfit_score
    )

    exclude_sigs: set = set()
    if req.excludeItemSets:
        exclude_sigs = {"|".join(sorted(ids)) for ids in req.excludeItemSets}

    gen_max = effective_max_sets + len(exclude_sigs) * 2 if exclude_sigs else effective_max_sets

    outfits_raw = build_outfit_sets(
        scored_outfit,
        weather={"temp": weather.temp, "pty": weather.pty},
        style=effective_style,
        styles=effective_styles,
        max_sets=gen_max,
        min_base_score=min_base_score,
        min_outfit_score=min_outfit_score,
        threshold_include_style=policy.include_style_in_threshold,
        threshold_include_weather=policy.include_weather_in_threshold,
        allow_singles=policy.allow_singles,
        min_sets=3,
        body_type=req.bodyType,
    )

    if exclude_sigs:
        outfits_raw = [
            o for o in outfits_raw
            if "|".join(sorted(str(it.get("id")) for it in (o.get("items") or []) if it and it.get("id")))
            not in exclude_sigs
        ]

    logger.info("[OUTFITS] produced=%d lens=%s", len(outfits_raw), [len(o.get("items", [])) for o in outfits_raw])

    # fallback
    if not outfits_raw:
        t = float(weather.temp or 0.0)
        relax = 4.0
        if t <= -10:
            relax = 6.0
        elif t >= 30:
            relax = 5.0

        relaxed_min_base = max(0.0, float(min_base_score) - relax)
        relaxed_min_outfit = max(0.0, float(min_outfit_score) - relax)

        outfits_raw = build_outfit_sets(
            scored_outfit,
            weather={"temp": weather.temp, "pty": weather.pty},
            style=effective_style,
            styles=effective_styles,
            max_sets=gen_max,
            min_base_score=relaxed_min_base,
            min_outfit_score=relaxed_min_outfit,
            threshold_include_style=False,
            threshold_include_weather=True,
            allow_singles=True,
            body_type=req.bodyType,
        )

        if exclude_sigs:
            outfits_raw = [
                o for o in outfits_raw
                if "|".join(sorted(str(it.get("id")) for it in (o.get("items") or []) if it and it.get("id")))
                not in exclude_sigs
            ]

        logger.info("[OUTFITS] fallback produced=%d", len(outfits_raw))

    outfits_raw = outfits_raw[:effective_max_sets]

    # Phase 3: 앵커 + 유저 스타일 벡터 혼합
    anchors = load_style_anchors()
    anchor_key = (effective_style or "").strip().lower()
    mixed_anchor = _get_mixed_anchor(anchor_key, req.userId, anchors)

    outfits: List[OutfitSet] = []
    for o in outfits_raw:
        raw_items = o.get("items") or []
        items = [RecommendItem(**it) for it in raw_items]

        style_sim: Optional[float] = None
        style_score: Optional[float] = None

        if mixed_anchor is not None:
            outfit_vec = _outfit_embedding_from_items(raw_items)
            if outfit_vec is not None:
                style_sim = float(calc_style_similarity(outfit_vec, mixed_anchor))
                style_score = _sim_to_score(style_sim)

        outfits.append(
            OutfitSet(
                items=items,
                outfitScore=o.get("outfitScore", 0.0),
                styleSim=style_sim,
                styleScore=style_score,
                _debug=o.get("_debug"),
            )
        )

    return RecommendOutfitResponse(weather=weather, outfits=outfits)
