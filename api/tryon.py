import asyncio
import base64
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, HttpUrl

from services.rate_limit import limiter
from services.url_validator import validate_url_for_fetch

logger = logging.getLogger("tryon")
router = APIRouter(prefix="/tryon", tags=["tryon"])

GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models")

SUPPORTED_MODELS = set(
    (os.getenv("TRYON_SUPPORTED_MODELS") or "gemini-2.5-flash-image,gemini-3-pro-image-preview").split(",")
)
SUPPORTED_MODELS = {m.strip() for m in SUPPORTED_MODELS if m.strip()}

ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}

_ASCII_CODE_LIST_RE = re.compile(r"^\s*\d+(?:\s+\d+)+\s*$")


def _field_min_1_list():
    # pydantic v1: min_items, v2: min_length
    try:
        return Field(min_length=1)
    except TypeError:
        return Field(min_items=1)


class TryOnUrlRequest(BaseModel):
    personImageUrl: HttpUrl
    clothesImageUrls: List[HttpUrl] = _field_min_1_list()

    model: str = "gemini-3-pro-image-preview"
    view: str = "auto"  # auto|front|back
    keepBackground: bool = True
    aspectRatio: str = "3:4"
    category: str = "auto"
    # clothesImageUrls와 같은 순서의 개별 옷 카테고리(TOP/BOTTOM/OUTER/SET...).
    # 상하의 동시 피팅 시 category=auto로 뭉개지는 문제를 해결 — 이미지별로
    # 어느 부위를 교체할지 결정적 지침을 만들 수 있게 함. 구버전 앱은 안 보냄(빈 리스트).
    clothesCategories: List[str] = []
    # 🔥 1. 안드로이드에서 보낸 마스터 프롬프트를 받는 필드 추가!
    prompt: str = ""


class TryOnResponse(BaseModel):
    mimeType: str
    imageBase64: str
    debug: Optional[Dict[str, Any]] = None


def _require_httpx():
    try:
        import httpx  # type: ignore
        return httpx
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"httpx not installed: {e}") from e


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
                raise HTTPException(status_code=500, detail="GEMINI_API_KEY malformed (ascii list decode failed)")
        else:
            raw = "".join(raw.split())

    raw = raw.strip()
    if not raw or any(ch in raw for ch in ("\r", "\n")):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY contains invalid characters")

    return raw


def _inline_part(img_bytes: bytes, mime: str) -> Dict[str, Any]:
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(img_bytes).decode("utf-8")}}


# 🔥 2. 파라미터에 user_prompt 추가 및 카테고리별 강력 방어 로직 적용
def _tryon_prompt(view: str, keep_background: bool, category: str = "auto", user_prompt: str = "", num_garments: int = 1, clothes_categories: Optional[List[str]] = None) -> str:
    """Build the Gemini try-on prompt.

    재작성(72c1d14) 이전의 검증된 단순 프롬프트로 복원. 그 버전이 "사람은
    그대로 두고 옷만 교체"가 잘 됐는데, 재작성이 복잡해지며 레퍼런스 모델의
    문신·신발·다른 옷까지 복사하는 regression을 만들었음. 유일한 추가는
    옷별 카테고리(clothes_categories)로 상하의 동시 피팅 시 이미지별 결정적
    매핑 — category=auto로 뭉개져 한쪽 옷이 누락되던 문제 해결.
    """
    v = (view or "auto").strip().lower()
    view_line = ""
    if v == "back":
        view_line = "The person photo is a back view. Keep it a back view. Do not rotate the person."
    elif v == "front":
        view_line = "The person photo is a front view. Keep it a front view. Do not rotate the person."

    bg_line = (
        "Preserve the original background and pose as much as possible."
        if keep_background
        else "Use a clean studio background."
    )

    prompt_lines = [
        "Generate a realistic full-body fashion photo in high resolution with sharp details.",
        "Use the first images as clothing references.",
        "The clothing should fit the person's body naturally, showing proper draping and fabric tension.",
        "Do NOT alter the person's face in any way.",
        "Preserve natural hand and finger details.",
    ]

    cat_lower = (category or "auto").strip().lower()
    cats = [c.strip().upper() for c in (clothes_categories or []) if c and c.strip()]
    _region = {
        "TOP": "the upper body (replace the top/shirt only)",
        "BOTTOM": "the lower body (replace the pants/shorts/skirt only)",
        "OUTER": "the outer layer (add/replace the jacket/coat only)",
        "SHOES": "the feet (replace footwear only)",
        "BAG": "carried as a bag",
        "ACC": "as an accessory",
        "SET": "the entire outfit (full-body one-piece/set)",
    }

    if len(cats) >= 2 and len(cats) == num_garments:
        # 다중 옷(상하의 등): 이미지 순서대로 결정적 매핑 — 한쪽 누락 방지.
        lines = ["[CRITICAL]: Each clothing reference image is a separate garment:"]
        for i, c in enumerate(cats, start=1):
            desc = _region.get(c, "the matching body region")
            lines.append(f" - Reference image #{i} is a {c}: put it on {desc}.")
        lines.append(
            "Apply EVERY listed garment; do not skip any. Keep each garment's "
            "length and silhouette exactly as in its own reference image."
        )
        prompt_lines.append("\n".join(lines))
    elif cat_lower != "auto":
        cat_hint = f"The clothing reference is a '{category}'."
        if any(kw in cat_lower for kw in ["원피스", "one-piece", "dress"]):
            cat_hint += " Replace the entire outfit (both upper and lower body) with this dress."
        elif any(kw in cat_lower for kw in ["수영복", "swimsuit", "bikini", "하이레그", "high-leg"]):
            cat_hint += " Replace the outfit with this swimsuit. Naturally blend the exposed skin with the person's original skin tone and body line."
        elif "bottom" in cat_lower:
            cat_hint += (
                " \n[CRITICAL]: Look at the person's ORIGINAL upper garment in the photo. "
                "You MUST EXACTLY reproduce the original upper garment (same color, same shape, same sleeves). "
                "ONLY change the lower body (pants/skirt) to match the clothing reference."
            )
        elif "top" in cat_lower:
            cat_hint += (
                " \n[CRITICAL]: Look at the person's ORIGINAL lower garment (pants/skirt) in the photo. "
                "You MUST EXACTLY reproduce the original lower garment (same color, same shape, same length). "
                "ONLY change the upper body to match the clothing reference."
            )
        elif "outer" in cat_lower:
            cat_hint += (
                " Add or replace ONLY the outer layer (jacket/coat). "
                "Keep the person's original inner top and bottom exactly as they are."
            )
        prompt_lines.append(cat_hint)

    prompt_lines.extend([
        "Put those clothes on the person in the last image.",
        "Preserve the person's identity (face), hairstyle, body shape, and skin tone.",
        bg_line,
        "Replace only the relevant garment areas; keep non-target garments unchanged.",
        "Match lighting and shadows naturally so the clothes look truly worn.",
        view_line,
    ])

    if user_prompt and user_prompt.strip():
        prompt_lines.append("\n[Additional Styling & Constraints from App]")
        prompt_lines.append(user_prompt.strip())

    return "\n".join([line for line in prompt_lines if line]).strip()


def _validate_url_for_fetch(url: str) -> None:
    """Thin wrapper that converts ``ValueError`` from the shared
    validator into an ``HTTPException`` for the API layer."""
    try:
        validate_url_for_fetch(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _sniff_mime(data: bytes, fallback: str) -> str:
    if len(data) >= 12:
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
    return fallback if fallback in ALLOWED_MIMES else "image/jpeg"


async def _download_image(url: str, *, max_bytes: int = 6_000_000) -> Tuple[bytes, str]:
    _validate_url_for_fetch(url)

    httpx = _require_httpx()
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(url)

    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Image download failed: {url} ({r.status_code})")

    data = r.content
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty image: {url}")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image too large (> {max_bytes} bytes): {url}")

    header_mime = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
    if header_mime not in ALLOWED_MIMES:
        header_mime = "image/jpeg"

    mime = _sniff_mime(data, header_mime)
    return data, mime


async def _call_gemini_image(*, model: str, parts: List[Dict[str, Any]], aspect_ratio: str) -> TryOnResponse:
    httpx = _require_httpx()
    api_key = _load_gemini_api_key()

    url = f"{GEMINI_BASE_URL}/{model}:generateContent"
    payload: Dict[str, Any] = {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": aspect_ratio}},
    }

    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(
            url,
            headers={"x-goog-api-key": api_key, "content-type": "application/json"},
            json=payload,
        )

    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {resp.status_code} {resp.text}")

    data = resp.json()
    try:
        out_parts = data["candidates"][0]["content"]["parts"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini response parse error: {e}")

    for p in out_parts:
        blob = p.get("inlineData") or p.get("inline_data")
        if not blob:
            continue
        b64 = blob.get("data")
        mime = blob.get("mimeType") or blob.get("mime_type") or "image/png"
        if b64:
            return TryOnResponse(mimeType=mime, imageBase64=b64, debug={"model": model, "partsIn": len(parts)})

    raise HTTPException(status_code=502, detail="No image returned from Gemini")


@router.post("/url", response_model=TryOnResponse)
@limiter.limit("10/minute")
async def tryon_url(request: Request, req: TryOnUrlRequest) -> TryOnResponse:
    # 🔥 4. 로그에 prompt가 잘 넘어오는지 찍히도록 업데이트
    logger.info(
        "[TRYON_URL] model=%s view=%s keepBg=%s category=%s clothesCategories=%s clothes=%d prompt='%s'",
        req.model,
        req.view,
        req.keepBackground,
        req.category,
        req.clothesCategories,
        len(req.clothesImageUrls),
        req.prompt,
    )

    if req.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model")

    clothes_urls = [str(u) for u in req.clothesImageUrls]
    clothes_dl = await asyncio.gather(*[_download_image(u) for u in clothes_urls])
    clothes_parts: List[Dict[str, Any]] = [_inline_part(b, mime) for (b, mime) in clothes_dl]

    person_bytes, person_mime = await _download_image(str(req.personImageUrl))
    
    # 🔥 5. _tryon_prompt 에 req.prompt 안전하게 넘겨주기
    prompt = _tryon_prompt(
        view=req.view,
        keep_background=req.keepBackground,
        category=req.category,
        user_prompt=req.prompt,
        num_garments=len(clothes_urls),
        clothes_categories=req.clothesCategories,
    )

    parts = [*clothes_parts, _inline_part(person_bytes, person_mime), {"text": prompt}]
    return await _call_gemini_image(model=req.model, parts=parts, aspect_ratio=req.aspectRatio)