# services/outfit_encoder.py
from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from io import BytesIO
from typing import Iterable, Sequence, Union

import numpy as np
import requests
from PIL import Image

from services.url_validator import validate_url_for_fetch

logger = logging.getLogger(__name__)

# requests 재사용(커넥션 풀)
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "fashioncloset-server/1.0",
        "Accept": "image/*,*/*;q=0.8",
    }
)

# 너무 큰 이미지로 메모리 터지는 것 방지
_MAX_IMAGE_PIXELS = 20_000_000  # 20MP
Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS

# FashionCLIP 모델 (프로세스당 1회 지연 로딩)
_MODEL = None
_PROCESSOR = None
_DEVICE = None
_LOAD_LOCK = threading.Lock()

_MODEL_NAME = os.getenv("FASHIONCLIP_MODEL", "patrickjohncyh/fashion-clip")


def _open_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(BytesIO(data))
    return img.convert("RGB")


def _as_1d_vector(x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("vector must be 1D and non-empty")
    if not np.isfinite(v).all():
        raise ValueError("vector contains non-finite values")
    return v


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return v
    return v / n


def _get_fashionclip():
    """
    Lazily load FashionCLIP model once per process.
    patrickjohncyh/fashion-clip: 패션 이미지 80만장으로 fine-tune된 CLIP ViT-B/32.
    """
    global _MODEL, _PROCESSOR, _DEVICE

    if _MODEL is not None and _PROCESSOR is not None and _DEVICE is not None:
        return _MODEL, _PROCESSOR, _DEVICE

    with _LOAD_LOCK:
        if _MODEL is not None and _PROCESSOR is not None and _DEVICE is not None:
            return _MODEL, _PROCESSOR, _DEVICE

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as e:
            logger.exception("[FASHIONCLIP] import failed. err=%s", e)
            raise

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            _PROCESSOR = CLIPProcessor.from_pretrained(_MODEL_NAME)
            _MODEL = CLIPModel.from_pretrained(_MODEL_NAME).to(_DEVICE)
            _MODEL.eval()
            logger.info("[FASHIONCLIP] loaded model=%s device=%s", _MODEL_NAME, _DEVICE)
        except Exception as e:
            logger.exception("[FASHIONCLIP] model load failed. err=%s", e)
            raise

    return _MODEL, _PROCESSOR, _DEVICE


def encode_outfit_image(image_path: str) -> np.ndarray:
    """Local image path -> FashionCLIP image embedding (512,) float32 numpy."""
    import torch

    model, processor, device = _get_fashionclip()
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        vec = model.get_image_features(**inputs)

    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec[0].detach().cpu().numpy().astype(np.float32)


@lru_cache(maxsize=512)
def encode_outfit_image_from_url(url: str) -> np.ndarray:
    """URL image -> FashionCLIP image embedding (512,) float32 numpy. Cached by URL."""
    import torch

    model, processor, device = _get_fashionclip()

    validate_url_for_fetch(url)

    try:
        res = _SESSION.get(url, timeout=7.0)
        res.raise_for_status()
    except Exception as e:
        logger.warning("[FASHIONCLIP] image fetch failed url=%s err=%s", url, e)
        raise

    try:
        img = _open_image_from_bytes(res.content)
    except Exception as e:
        logger.warning("[FASHIONCLIP] image decode failed url=%s err=%s", url, e)
        raise

    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        vec = model.get_image_features(**inputs)

    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec[0].detach().cpu().numpy().astype(np.float32)


def encode_text(texts: Union[str, Iterable[str]]) -> np.ndarray:
    """Text(s) -> mean FashionCLIP text embedding (512,) float32 numpy."""
    if isinstance(texts, str):
        texts_list = [texts]
    else:
        texts_list = [str(t) for t in texts if str(t).strip()]

    if not texts_list:
        raise ValueError("texts must not be empty")

    return _encode_text_cached(tuple(texts_list))


@lru_cache(maxsize=256)
def _encode_text_cached(texts_tuple: tuple[str, ...]) -> np.ndarray:
    import torch

    model, processor, device = _get_fashionclip()

    inputs = processor(text=list(texts_tuple), return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        vecs = model.get_text_features(**inputs)

    vecs = vecs / vecs.norm(dim=-1, keepdim=True)

    v = vecs.mean(dim=0)
    v = v / v.norm(dim=-1, keepdim=True)

    return v.detach().cpu().numpy().astype(np.float32)


def cosine_similarity(
    a: Union[np.ndarray, Sequence[float]],
    b: Union[np.ndarray, Sequence[float]],
) -> float:
    """Safe cosine similarity. Returns 0.0 on dimension mismatch."""
    va = _l2_normalize(_as_1d_vector(a))
    vb = _l2_normalize(_as_1d_vector(b))

    if va.shape[0] != vb.shape[0]:
        logger.warning("[FASHIONCLIP] cosine_similarity dim mismatch: %s vs %s", va.shape, vb.shape)
        return 0.0

    return float(np.dot(va, vb))
