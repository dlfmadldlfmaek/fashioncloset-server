# api/classify.py
from __future__ import annotations

import base64
import logging
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel

from services.auth import verify_firebase_token
from services.rate_limit import limiter

router = APIRouter(prefix="/classify", tags=["classify"])
logger = logging.getLogger("classify")

GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/models",
)
GEMINI_MODEL = os.getenv("CLASSIFY_MODEL", "gemini-2.5-flash")

_ASCII_CODE_LIST_RE = re.compile(r"^[\d\s]+$")

_CATEGORY_RULES = """- mainCategory: 반드시 TOP, BOTTOM, OUTER, SET, SHOES, BAG, ACC 중 하나
  - TOP: 티셔츠, 셔츠, 블라우스, 니트, 맨투맨, 후드, 폴로 등 상의
  - BOTTOM: 바지, 치마, 청바지, 레깅스 등 하의
  - OUTER: 자켓, 코트, 점퍼, 가디건 등 겉옷
  - SET: 원피스, 점프수트, 래시가드, 올인원, 드레스, 슈트세트 등 상하의 일체형
  - SHOES: 신발, 운동화, 슬리퍼 등
  - BAG: 가방, 백팩, 클러치 등
  - ACC: 모자, 벨트, 시계, 안경, 목걸이 등 액세서리"""

_NAME_RULES = """- name: 색상 + 소재/직물 + 넥라인/디테일 + 종류 조합 (한국어)
  - 소재를 정확히 구분: 와플(격자 울퉁불퉁), 골지(세로 골), 니트(편직), 면(평직), 저지(신축성 편직), 데님, 코듀로이(굵은 세로골) 등
  - 넥라인을 정확히 구분: 폴로(카라+단추), 브이넥(V자), 라운드넥(둥근), 크루넥(높은 둥근), 헨리넥(단추만), 터틀넥 등
  - 로고나 텍스트가 보이면 브랜드명 포함 (예: 아디다스 와플 폴로 티)"""

CLASSIFY_PROMPT = f"""이 옷 이미지를 정확하게 분석해서 JSON으로 답변해주세요:
{{
  "name": "옷 이름 (예: 흰색 와플 폴로 티셔츠)",
  "mainCategory": "카테고리 (TOP, BOTTOM, OUTER, SET, SHOES, BAG, ACC 중 하나)",
  "tags": ["색상", "소재", "스타일", "핏", "시즌"]
}}

규칙:
{_NAME_RULES}
{_CATEGORY_RULES}
- tags: 최대 5개, 한국어 (색상, 소재, 넥라인, 핏, 시즌 등)
- JSON만 출력, 다른 텍스트 없이"""


CLASSIFY_BEST_PROMPT = f"""여러 옷 이미지를 비교 분석해서 JSON으로 답변해주세요. 모든 이미지는 같은 상품(같은 옷)입니다.
{{
  "name": "옷 이름 (예: 흰색 와플 폴로 티셔츠)",
  "mainCategory": "카테고리 (TOP, BOTTOM, OUTER, SET, SHOES, BAG, ACC 중 하나)",
  "tags": ["색상", "소재", "스타일", "핏", "시즌"],
  "bestIndex": 0
}}

규칙:
{_NAME_RULES}
{_CATEGORY_RULES}
- tags: 최대 5개, 한국어
- bestIndex: 옷 자체가 가장 잘 보이는 단독 정면샷 이미지의 0-based 인덱스
  - 모델 착용샷보다 상품 단독샷(흰 배경, 옷만 펼쳐진 사진)을 우선
  - 단독샷이 없으면 정면 착용샷, 그것도 없으면 가장 또렷한 사진
  - 절대 범위를 벗어난 인덱스를 출력하지 말 것
- JSON만 출력, 다른 텍스트 없이"""

VALID_CATEGORIES = {"TOP", "BOTTOM", "OUTER", "SET", "SHOES", "BAG", "ACC"}

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_BEST_IMAGES = 8


class ClassifyResponse(BaseModel):
    name: str
    mainCategory: str
    tags: list[str]


class ClassifyBestResponse(BaseModel):
    name: str
    mainCategory: str
    tags: list[str]
    bestIndex: int


class ClassifyBestFromUrlRequest(BaseModel):
    productUrl: str
    ogImageUrl: str = ""


class ClassifyBestFromUrlResponse(BaseModel):
    name: str
    mainCategory: str
    tags: list[str]
    bestIndex: int
    imageUrl: str


def _load_gemini_api_key() -> str:
    raw = os.getenv("GEMINI_API_KEY", "") or ""
    raw = raw.strip()

    if not raw:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    if "\r" in raw or "\n" in raw:
        if _ASCII_CODE_LIST_RE.match(raw):
            nums = [int(x) for x in raw.split()]
            try:
                raw = "".join(chr(n) for n in nums)
            except Exception:
                raise HTTPException(
                    status_code=500,
                    detail="GEMINI_API_KEY malformed",
                )
        else:
            raw = "".join(raw.split())

    raw = raw.strip()
    if not raw or any(ch in raw for ch in ("\r", "\n")):
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY contains invalid characters",
        )
    return raw


def _extract_json_str(text: str) -> str:
    """Gemini 응답에서 JSON 본문만 뽑아냄 — 코드블록/앞뒤 텍스트 제거."""
    json_str = text.strip()
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str)
    if code_block:
        return code_block.group(1).strip()
    brace_match = re.search(r"\{[\s\S]*\}", json_str)
    if brace_match:
        return brace_match.group(0)
    return json_str


def _parse_gemini_response(text: str) -> ClassifyResponse:
    import json as json_mod

    json_str = _extract_json_str(text)

    # JSON 파싱 시도
    try:
        data = json_mod.loads(json_str)
        name = data.get("name", "이름 없음")
        category = data.get("mainCategory", "TOP")
        main_category = category if category in VALID_CATEGORIES else "TOP"
        tags = [str(t) for t in data.get("tags", []) if t][:5]
        return ClassifyResponse(name=name, mainCategory=main_category, tags=tags)
    except (json_mod.JSONDecodeError, AttributeError) as exc:
        logger.warning("JSON parse failed (%s), falling back to regex. json_str repr: %s", exc, repr(json_str[:300]))

    # 폴백: regex 파싱
    name_match = re.search(r'"name"\s*:\s*"([^"]*)"', json_str)
    cat_match = re.search(r'"mainCategory"\s*:\s*"([^"]*)"', json_str)
    tags_match = re.search(r'"tags"\s*:\s*\[([^\]]*)\]', json_str)

    name = name_match.group(1) if name_match else "이름 없음"
    category = cat_match.group(1) if cat_match else "TOP"
    main_category = category if category in VALID_CATEGORIES else "TOP"

    tags: list[str] = []
    if tags_match:
        tag_values = re.findall(r'"([^"]*)"', tags_match.group(1))
        tags = [t for t in tag_values if t][:5]

    return ClassifyResponse(name=name, mainCategory=main_category, tags=tags)


def _parse_gemini_best_response(text: str, image_count: int) -> ClassifyBestResponse:
    """멀티이미지 응답 파싱 — bestIndex 추가, 범위 밖 인덱스는 0으로 클램프."""
    import json as json_mod

    json_str = _extract_json_str(text)

    name = "이름 없음"
    category = "TOP"
    tags: list[str] = []
    best_index = 0

    try:
        data = json_mod.loads(json_str)
        name = data.get("name", "이름 없음")
        category = data.get("mainCategory", "TOP")
        tags = [str(t) for t in data.get("tags", []) if t][:5]
        raw_idx = data.get("bestIndex", 0)
        try:
            best_index = int(raw_idx)
        except (TypeError, ValueError):
            best_index = 0
    except (json_mod.JSONDecodeError, AttributeError) as exc:
        logger.warning("Best JSON parse failed (%s), falling back to regex. json_str repr: %s", exc, repr(json_str[:300]))
        name_match = re.search(r'"name"\s*:\s*"([^"]*)"', json_str)
        cat_match = re.search(r'"mainCategory"\s*:\s*"([^"]*)"', json_str)
        tags_match = re.search(r'"tags"\s*:\s*\[([^\]]*)\]', json_str)
        idx_match = re.search(r'"bestIndex"\s*:\s*(\d+)', json_str)
        name = name_match.group(1) if name_match else "이름 없음"
        category = cat_match.group(1) if cat_match else "TOP"
        if tags_match:
            tags = [t for t in re.findall(r'"([^"]*)"', tags_match.group(1)) if t][:5]
        if idx_match:
            best_index = int(idx_match.group(1))

    main_category = category if category in VALID_CATEGORIES else "TOP"
    if image_count <= 0:
        best_index = 0
    else:
        best_index = max(0, min(best_index, image_count - 1))

    return ClassifyBestResponse(
        name=name,
        mainCategory=main_category,
        tags=tags,
        bestIndex=best_index,
    )


async def _call_gemini(parts: List[Dict[str, Any]]) -> str:
    """Gemini API 호출 → 응답 text 반환. 실패 시 HTTPException 발생."""
    import httpx

    api_key = _load_gemini_api_key()
    url = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent"
    payload: Dict[str, Any] = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, params={"key": api_key}, json=payload)

    if resp.status_code != 200:
        logger.error("Gemini API error: %s %s", resp.status_code, resp.text[:200])
        raise HTTPException(status_code=502, detail="AI 분류 서비스에 문제가 발생했어요")

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        logger.error("Gemini response parse error: %s", data)
        raise HTTPException(status_code=502, detail="AI 응답을 파싱할 수 없어요")


@router.post("", response_model=ClassifyResponse)
async def classify_clothes(
    image: UploadFile = File(...),
    token: dict = Depends(verify_firebase_token),
):
    """이미지를 받아 Gemini로 옷을 분류합니다."""
    import httpx

    image_bytes = await image.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="이미지 크기는 10MB 이하여야 합니다")

    mime_type = image.content_type or "image/jpeg"
    api_key = _load_gemini_api_key()
    url = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent"

    payload: Dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {"text": CLASSIFY_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            params={"key": api_key},
            json=payload,
        )

    if resp.status_code != 200:
        logger.error("Gemini API error: %s %s", resp.status_code, resp.text[:200])
        raise HTTPException(status_code=502, detail="AI 분류 서비스에 문제가 발생했어요")

    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        logger.error("Gemini response parse error: %s", data)
        raise HTTPException(status_code=502, detail="AI 응답을 파싱할 수 없어요")

    logger.info("Gemini raw response repr: %s", repr(text[:500]))
    result = _parse_gemini_response(text)
    logger.info("Parsed result: name=%s, category=%s, tags=%s", result.name, result.mainCategory, result.tags)
    return result


@router.post("/best", response_model=ClassifyBestResponse)
async def classify_best_image(
    images: List[UploadFile] = File(...),
    token: dict = Depends(verify_firebase_token),
):
    """여러 이미지를 받아 같은 상품으로 보고 분류 + 가장 단독 정면샷 1장의 인덱스 선택."""
    if not images:
        raise HTTPException(status_code=400, detail="이미지가 없습니다")
    if len(images) > MAX_BEST_IMAGES:
        raise HTTPException(status_code=400, detail=f"이미지는 최대 {MAX_BEST_IMAGES}장까지 가능합니다")

    parts: List[Dict[str, Any]] = [{"text": CLASSIFY_BEST_PROMPT}]
    accepted = 0
    for i, image in enumerate(images):
        raw = await image.read()
        if not raw:
            continue
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail=f"이미지 {i + 1}의 크기가 10MB를 초과합니다")
        mime_type = image.content_type or "image/jpeg"
        parts.append({"text": f"이미지 {accepted}:"})
        parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(raw).decode("utf-8"),
            }
        })
        accepted += 1

    if accepted == 0:
        raise HTTPException(status_code=400, detail="유효한 이미지가 없습니다")

    text = await _call_gemini(parts)
    logger.info("Gemini best raw repr: %s", repr(text[:500]))
    result = _parse_gemini_best_response(text, image_count=accepted)
    logger.info("Best result: name=%s, category=%s, bestIndex=%d/%d",
                result.name, result.mainCategory, result.bestIndex, accepted)
    return result


_HEADERS_HTML = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.6",
}

_HEADERS_IMG = {
    "User-Agent": _HEADERS_HTML["User-Agent"],
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def _absolute_url(url: str, base_url: str) -> str:
    if not url:
        return url
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        m = re.match(r"^(https?://[^/]+)", base_url)
        if m:
            return m.group(1) + url
    # 무신사: www.musinsa.com/images/ → image.msscdn.net/images/
    if "musinsa.com/images/" in url:
        url = re.sub(r"https?://[^/]*musinsa\.com/images/", "https://image.msscdn.net/images/", url)
    return url


def _extract_image_urls_from_html(html: str, base_url: str) -> List[str]:
    """og:image 우선, 그 다음 무신사/일반 img 태그 폴백. 상위 N개."""
    urls: List[str] = []

    def _push(u: Optional[str]) -> None:
        if not u:
            return
        fixed = _absolute_url(u.strip(), base_url)
        if fixed and fixed not in urls and "_125." not in fixed:
            urls.append(fixed)

    # og:image
    for prop in ("og:image:secure_url", "og:image"):
        m = re.search(
            rf'meta[^>]*property=["\']{re.escape(prop)}["\'][^>]*content=["\']([^"\']+)["\']',
            html, re.IGNORECASE,
        )
        if m:
            _push(m.group(1))
        m = re.search(
            rf'meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']{re.escape(prop)}["\']',
            html, re.IGNORECASE,
        )
        if m:
            _push(m.group(1))

    # 무신사 __NEXT_DATA__ 등 JSON 내 imageUrl
    for m in re.finditer(
        r'"thumbnailImageUrl"\s*:\s*"(/images/[^"]+\.(?:jpg|jpeg|png|webp))"',
        html, re.IGNORECASE,
    ):
        _push(m.group(1))
        if len(urls) >= MAX_BEST_IMAGES:
            return urls

    for m in re.finditer(
        r'"imageUrl"\s*:\s*"(/images/(?:goods_img|prd_img)/[^"]+\.(?:jpg|jpeg|png|webp))"',
        html, re.IGNORECASE,
    ):
        _push(m.group(1))
        if len(urls) >= MAX_BEST_IMAGES:
            return urls

    # img 태그 (상품 디렉토리만)
    for m in re.finditer(
        r"""<img[^>]+src=["']([^"']+(?:goods_img|prd_img|product)[^"']*?\.(?:jpg|jpeg|png|webp))["']""",
        html, re.IGNORECASE,
    ):
        url = m.group(1)
        if "logo" in url.lower() or "brand" in url.lower():
            continue
        _push(url)
        if len(urls) >= MAX_BEST_IMAGES:
            return urls

    return urls


async def _download_image(client: "httpx.AsyncClient", url: str, referer: str = "") -> Optional[bytes]:
    headers = dict(_HEADERS_IMG)
    if referer:
        headers["Referer"] = referer
        m = re.match(r"^(https?://[^/]+)", referer)
        if m:
            headers["Origin"] = m.group(1)
    try:
        resp = await client.get(url, headers=headers, follow_redirects=True, timeout=20.0)
    except Exception as e:
        logger.warning("download failed url=%s err=%s", url[:120], e)
        return None
    if resp.status_code != 200:
        logger.warning("download status=%s url=%s", resp.status_code, url[:120])
        return None
    if not resp.content:
        return None
    if len(resp.content) > MAX_IMAGE_BYTES:
        return None
    return resp.content


@router.post("/best-from-url", response_model=ClassifyBestFromUrlResponse)
async def classify_best_from_url(
    body: ClassifyBestFromUrlRequest,
    token: dict = Depends(verify_firebase_token),
):
    """productUrl HTML을 직접 가져와 이미지 후보를 추출·다운로드 → 멀티이미지 분류 → best 1장 반환."""
    import httpx

    product_url = body.productUrl.strip()
    og_image_url = body.ogImageUrl.strip()
    if not product_url:
        raise HTTPException(status_code=400, detail="productUrl이 필요합니다")

    async with httpx.AsyncClient(headers=_HEADERS_HTML, follow_redirects=True, timeout=30.0) as client:
        # 1. HTML 받아 이미지 후보 추출
        candidates: List[str] = []
        if og_image_url:
            candidates.append(og_image_url)

        resolved_url = product_url
        try:
            html_resp = await client.get(product_url)
            if html_resp.status_code < 400:
                resolved_url = str(html_resp.url)
                for u in _extract_image_urls_from_html(html_resp.text or "", resolved_url):
                    if u not in candidates:
                        candidates.append(u)
        except Exception as e:
            logger.warning("HTML fetch failed url=%s err=%s", product_url[:120], e)

        if not candidates:
            raise HTTPException(status_code=404, detail="상품 이미지를 찾지 못했어요")

        # 2. 이미지 다운로드 (최대 MAX_BEST_IMAGES장)
        downloaded: List[tuple[str, bytes, str]] = []  # (url, bytes, mime)
        for url in candidates[:MAX_BEST_IMAGES]:
            data = await _download_image(client, url, referer=resolved_url)
            if data:
                mime = "image/jpeg"
                lower = url.lower().split("?", 1)[0]
                if lower.endswith(".png"):
                    mime = "image/png"
                elif lower.endswith(".webp"):
                    mime = "image/webp"
                elif lower.endswith(".gif"):
                    mime = "image/gif"
                downloaded.append((url, data, mime))
            if len(downloaded) >= MAX_BEST_IMAGES:
                break

        if not downloaded:
            raise HTTPException(status_code=404, detail="이미지 다운로드에 실패했어요")

    # 3. Gemini 멀티이미지 분류 + bestIndex
    parts: List[Dict[str, Any]] = [{"text": CLASSIFY_BEST_PROMPT}]
    for i, (_, raw, mime) in enumerate(downloaded):
        parts.append({"text": f"이미지 {i}:"})
        parts.append({
            "inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(raw).decode("utf-8"),
            }
        })

    text = await _call_gemini(parts)
    logger.info("Best-from-url raw repr: %s", repr(text[:500]))
    parsed = _parse_gemini_best_response(text, image_count=len(downloaded))
    best_url = downloaded[parsed.bestIndex][0]
    logger.info("Best-from-url result: name=%s, category=%s, bestIndex=%d/%d, imageUrl=%s",
                parsed.name, parsed.mainCategory, parsed.bestIndex, len(downloaded), best_url[:120])

    return ClassifyBestFromUrlResponse(
        name=parsed.name,
        mainCategory=parsed.mainCategory,
        tags=parsed.tags,
        bestIndex=parsed.bestIndex,
        imageUrl=best_url,
    )
