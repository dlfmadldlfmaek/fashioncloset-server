# services/style_anchor.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from services.outfit_encoder import encode_outfit_image, encode_text

logger = logging.getLogger(__name__)

__all__ = ["build_style_anchors"]  # ✅ 이 모듈에서 외부로 공개할 함수

STYLE_PROMPTS: Dict[str, List[str]] = {
    "casual": ["casual clothes", "comfortable fashion", "daily look", "hoodie and jeans"],
    "formal": ["formal suit", "business attire", "office wear", "dress shirt"],
    "minimal": ["minimalist fashion", "simple clothing", "clean look", "solid color"],
    "street": ["streetwear", "hiphop fashion", "trendy outfit", "oversized clothes"],
    "vintage": ["vintage clothing", "retro style", "classic look", "old school fashion"],
}


@dataclass(frozen=True)
class BuildAnchorsConfig:
    base_dir: str = "data/style_anchor"
    text_weight: float = 3.0
    allowed_ext: tuple[str, ...] = (".png", ".jpg", ".jpeg")
    min_norm: float = 1e-6


def _as_vector(x: object) -> Optional[np.ndarray]:
    try:
        v = np.asarray(x, dtype=np.float32)
    except Exception:
        return None
    if v.ndim != 1 or v.size == 0:
        return None
    if not np.isfinite(v).all():
        return None
    return v


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 0.0:
        return v.astype(np.float32, copy=False)
    return (v / norm).astype(np.float32)


def _is_near_zero(v: np.ndarray, min_norm: float) -> bool:
    return float(np.linalg.norm(v)) < float(min_norm)


def _collect_image_files(folder: Path, allowed_ext: tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for ext in allowed_ext:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(p for p in files if p.is_file()))


def build_style_anchors(
    base_dir: str = "data/style_anchor",
    *,
    text_weight: float = 3.0,
) -> Dict[str, List[float]]:
    """
    ✅ Text + Image style anchors
    - folder별 이미지 임베딩 centroid + 텍스트 프롬프트 임베딩을 text_weight로 가중합
    - 결과는 L2 normalize된 list[float]
    """
    cfg = BuildAnchorsConfig(base_dir=base_dir, text_weight=float(text_weight))
    base_path = Path(cfg.base_dir)

    if not base_path.exists():
        logger.warning("[ANCHOR] base_dir not found: %s", base_path)
        return {}

    style_folders = sorted([p for p in base_path.iterdir() if p.is_dir()])
    logger.info("[ANCHOR] detected style folders: %s", [p.name for p in style_folders])

    anchors: Dict[str, List[float]] = {}

    for folder in style_folders:
        style_name = folder.name
        style_key = style_name.strip().lower()

        img_vecs: List[np.ndarray] = []
        expected_dim: Optional[int] = None

        files = _collect_image_files(folder, cfg.allowed_ext)

        for img_path in files:
            # ✅ 파일명이 폴더명 prefix로 시작해야 함 (원치 않으면 아래 if를 주석처리)
            if not img_path.name.lower().startswith(style_key):
                logger.warning("[ANCHOR][SKIP] folder=%s mismatched file=%s", style_name, img_path.name)
                continue

            try:
                vec_raw = encode_outfit_image(str(img_path))
            except Exception as e:
                logger.warning("[ANCHOR][SKIP] encode failed file=%s err=%s", img_path.name, e)
                continue

            vec = _as_vector(vec_raw)
            if vec is None:
                logger.warning("[ANCHOR][SKIP] invalid vector file=%s", img_path.name)
                continue
            if _is_near_zero(vec, cfg.min_norm):
                logger.warning("[ANCHOR][SKIP] near-zero vector file=%s", img_path.name)
                continue

            if expected_dim is None:
                expected_dim = int(vec.shape[0])
            elif int(vec.shape[0]) != expected_dim:
                logger.warning(
                    "[ANCHOR][SKIP] dim mismatch file=%s expected=%s got=%s",
                    img_path.name, expected_dim, vec.shape[0]
                )
                continue

            img_vecs.append(vec)

        # ✅ 텍스트 프롬프트 임베딩
        prompts = STYLE_PROMPTS.get(style_key, [f"{style_name} fashion style"])
        text_vec: Optional[np.ndarray] = None
        try:
            text_raw = encode_text(prompts)
            tv = _as_vector(text_raw)
            if tv is not None and not _is_near_zero(tv, cfg.min_norm):
                text_vec = tv
        except Exception as e:
            logger.exception("[ANCHOR] text encoding failed style=%s err=%s", style_name, e)

        if expected_dim is None and text_vec is not None:
            expected_dim = int(text_vec.shape[0])

        if text_vec is not None and expected_dim is not None and int(text_vec.shape[0]) != expected_dim:
            logger.warning("[ANCHOR][SKIP] text dim mismatch style=%s", style_name)
            text_vec = None

        if not img_vecs and text_vec is None:
            logger.warning("[ANCHOR] [%s] insufficient data (0 vectors).", style_key)
            continue

        if expected_dim is None:
            logger.warning("[ANCHOR][SKIP] expected_dim unresolved style=%s", style_key)
            continue

        # ✅ weighted centroid
        sum_vec = np.zeros((expected_dim,), dtype=np.float32)
        n = 0.0

        if img_vecs:
            sum_vec += np.sum(np.stack(img_vecs, axis=0), axis=0)
            n += float(len(img_vecs))

        if text_vec is not None and cfg.text_weight > 0:
            sum_vec += text_vec * float(cfg.text_weight)
            n += float(cfg.text_weight)

        mean = sum_vec / max(n, 1.0)
        mean = _normalize(mean)

        anchors[style_key] = mean.tolist()
        logger.info(
            "[ANCHOR] [%s] trained images=%d text_weight=%.2f dim=%s",
            style_key.upper(), len(img_vecs), cfg.text_weight, expected_dim
        )

    return anchors
