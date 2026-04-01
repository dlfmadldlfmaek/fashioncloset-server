# services/outfit_encoder.py
from __future__ import annotations

import logging
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

# 너무 큰 이미지로 메모리 터지는 것 방지(원하면 조절)
_MAX_IMAGE_PIXELS = 20_000_000  # 20MP
Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS

# 프로세스당 1회 로딩 (지연 로딩)
_MODEL = None
_PREPROCESS = None
_DEVICE = None


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


def _get_clip():
    """
    Lazily load CLIP model once per process.

    Why:
    - Avoid crashing the whole service at import-time if torch/clip deps are missing.
    - Improve Cloud Run readiness by deferring heavy init until actually used.
    """
    global _MODEL, _PREPROCESS, _DEVICE

    if _MODEL is not None and _PREPROCESS is not None and _DEVICE is not None:
        return _MODEL, _PREPROCESS, _DEVICE

    try:
        import torch  # lazy
        import clip  # lazy
    except Exception as e:
        logger.exception("[CLIP] import failed (torch/clip). Check requirements.txt pins. err=%s", e)
        raise

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        _MODEL, _PREPROCESS = clip.load("ViT-B/32", device=_DEVICE)
        _MODEL.eval()
        logger.info("[CLIP] loaded model=ViT-B/32 device=%s", _DEVICE)
    except Exception as e:
        logger.exception("[CLIP] model load failed. err=%s", e)
        raise

    return _MODEL, _PREPROCESS, _DEVICE


def encode_outfit_image(image_path: str) -> np.ndarray:
    """
    Local image path -> CLIP image embedding (512,) float32 numpy.
    """
    model, preprocess, device = _get_clip()

    try:
        import torch
    except Exception as e:
        logger.exception("[CLIP] torch import failed at encode_outfit_image. err=%s", e)
        raise

    img = Image.open(image_path).convert("RGB")
    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        vec = model.encode_image(image)

    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.detach().cpu().numpy()[0].astype(np.float32)


@lru_cache(maxsize=512)
def encode_outfit_image_from_url(url: str) -> np.ndarray:
    """
    URL image -> CLIP image embedding (512,) float32 numpy.
    Cached by URL to reduce Cloud Run cost and latency.
    """
    model, preprocess, device = _get_clip()

    try:
        import torch
    except Exception as e:
        logger.exception("[CLIP] torch import failed at encode_outfit_image_from_url. err=%s", e)
        raise

    validate_url_for_fetch(url)

    try:
        res = _SESSION.get(url, timeout=7.0)
        res.raise_for_status()
    except Exception as e:
        logger.warning("[CLIP] image fetch failed url=%s err=%s", url, e)
        raise

    try:
        img = _open_image_from_bytes(res.content)
    except Exception as e:
        logger.warning("[CLIP] image decode failed url=%s err=%s", url, e)
        raise

    image = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        vec = model.encode_image(image)

    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.detach().cpu().numpy()[0].astype(np.float32)


def encode_text(texts: Union[str, Iterable[str]]) -> np.ndarray:
    """
    Text(s) -> mean CLIP text embedding (512,) float32 numpy.
    """
    if isinstance(texts, str):
        texts_list = [texts]
    else:
        texts_list = [str(t) for t in texts if str(t).strip()]

    if not texts_list:
        raise ValueError("texts must not be empty")

    return _encode_text_cached(tuple(texts_list))


@lru_cache(maxsize=256)
def _encode_text_cached(texts_tuple: tuple[str, ...]) -> np.ndarray:
    model, _, device = _get_clip()

    try:
        import torch
        import clip
    except Exception as e:
        logger.exception("[CLIP] import failed at _encode_text_cached. err=%s", e)
        raise

    tokens = clip.tokenize(list(texts_tuple)).to(device)

    with torch.no_grad():
        vecs = model.encode_text(tokens)

    vecs = vecs / vecs.norm(dim=-1, keepdim=True)

    v = vecs.mean(dim=0)
    v = v / v.norm(dim=-1, keepdim=True)

    return v.detach().cpu().numpy().astype(np.float32)


def cosine_similarity(
    a: Union[np.ndarray, Sequence[float]],
    b: Union[np.ndarray, Sequence[float]],
) -> float:
    """
    Safe cosine similarity.
    Returns 0.0 on dimension mismatch.
    """
    va = _l2_normalize(_as_1d_vector(a))
    vb = _l2_normalize(_as_1d_vector(b))

    if va.shape[0] != vb.shape[0]:
        logger.warning("[CLIP] cosine_similarity dim mismatch: %s vs %s", va.shape, vb.shape)
        return 0.0

    return float(np.dot(va, vb))
