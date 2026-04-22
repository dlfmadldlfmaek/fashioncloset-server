# api/classify.py
from __future__ import annotations

import base64
import logging
import os
import re
from typing import Any, Dict, List

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

CLASSIFY_PROMPT = """이 옷 이미지를 정확하게 분석해서 JSON으로 답변해주세요:
{
  "name": "옷 이름 (예: 흰색 와플 폴로 티셔츠)",
  "mainCategory": "카테고리 (TOP, BOTTOM, OUTER, SET, SHOES, BAG, ACC 중 하나)",
  "tags": ["색상", "소재", "스타일", "핏", "시즌"]
}

규칙:
- name: 색상 + 소재/직물 + 넥라인/디테일 + 종류 조합 (한국어)
  - 소재를 정확히 구분: 와플(격자 울퉁불퉁), 골지(세로 골), 니트(편직), 면(평직), 저지(신축성 편직), 데님, 코듀로이(굵은 세로골) 등
  - 넥라인을 정확히 구분: 폴로(카라+단추), 브이넥(V자), 라운드넥(둥근), 크루넥(높은 둥근), 헨리넥(단추만), 터틀넥 등
  - 로고나 텍스트가 보이면 브랜드명 포함 (예: 아디다스 와플 폴로 티)
- mainCategory: 반드시 TOP, BOTTOM, OUTER, SET, SHOES, BAG, ACC 중 하나
  - TOP: 티셔츠, 셔츠, 블라우스, 니트, 맨투맨, 후드, 폴로 등 상의
  - BOTTOM: 바지, 치마, 청바지, 레깅스 등 하의
  - OUTER: 자켓, 코트, 점퍼, 가디건 등 겉옷
  - SET: 원피스, 점프수트, 래시가드, 올인원, 드레스, 슈트세트 등 상하의 일체형
  - SHOES: 신발, 운동화, 슬리퍼 등
  - BAG: 가방, 백팩, 클러치 등
  - ACC: 모자, 벨트, 시계, 안경, 목걸이 등 액세서리
- tags: 최대 5개, 한국어 (색상, 소재, 넥라인, 핏, 시즌 등)
- JSON만 출력, 다른 텍스트 없이"""

VALID_CATEGORIES = {"TOP", "BOTTOM", "OUTER", "SET", "SHOES", "BAG", "ACC"}


class ClassifyResponse(BaseModel):
    name: str
    mainCategory: str
    tags: list[str]


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


def _parse_gemini_response(text: str) -> ClassifyResponse:
    import json as json_mod

    # JSON 부분만 추출 — 코드블록, 앞뒤 텍스트 모두 제거
    json_str = text.strip()

    # 코드블록에서 JSON 추출
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str)
    if code_block:
        json_str = code_block.group(1).strip()
    else:
        # { } 사이만 추출
        brace_match = re.search(r"\{[\s\S]*\}", json_str)
        if brace_match:
            json_str = brace_match.group(0)

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
