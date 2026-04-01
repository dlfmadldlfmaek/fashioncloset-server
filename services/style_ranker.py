# services/style_ranker.py
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

import numpy as np

from services.outfit_encoder import (
    encode_outfit_image_from_url,
    encode_text,
    cosine_similarity,
)

# 스타일별 텍스트 앵커 (가볍고 안정적: 서버에 이미지 폴더 없어도 됨)
_STYLE_PROMPTS: Dict[str, str] = {
    "casual": "a casual outfit, hoodie, sweatshirt, jeans, sneakers",
    "formal": "a formal outfit, blazer, coat, dress shirt, slacks, loafers",
    "minimal": "a minimal outfit, simple, solid colors, clean silhouette",
    "street": "a streetwear outfit, oversized, cargo pants, hip hop style",
    "vintage": "a vintage outfit, retro classic style",
}


@lru_cache(maxsize=64)
def get_style_text_anchors() -> Dict[str, np.ndarray]:
    anchors: Dict[str, np.ndarray] = {}
    for k, prompt in _STYLE_PROMPTS.items():
        anchors[k] = encode_text(prompt)
    return anchors


def style_similarity_from_image(image_url: Optional[str], style: str) -> float:
    """
    return cosine similarity [-1, 1]
    """
    if not image_url:
        return 0.0

    anchors = get_style_text_anchors()
    style_key = (style or "").strip().lower()
    if style_key not in anchors:
        return 0.0

    img_vec = encode_outfit_image_from_url(image_url)
    txt_vec = anchors[style_key]
    return float(cosine_similarity(img_vec, txt_vec))


def style_multiplier(sim: float) -> float:
    """
    similarity를 점수 배수로 변환
    - sim<=0은 영향 거의 없음
    - sim>0일수록 최대 1.25까지
    """
    # sim [-1,1] -> pos [0,1]
    pos = max(0.0, min(1.0, sim))
    return 1.0 + 0.25 * pos  # 1.0 ~ 1.25
