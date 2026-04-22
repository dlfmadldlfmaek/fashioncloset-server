# services/embedding.py
from __future__ import annotations

import logging
import os
import threading
from typing import Iterable, List, Optional

import torch
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

_DEVICE: Optional[torch.device] = None
_MODEL = None
_PROCESSOR = None
_LOAD_LOCK = threading.Lock()

# FashionCLIP 모델 (패션 이미지 80만장으로 fine-tune된 CLIP ViT-B/32)
_MODEL_NAME = os.getenv("FASHIONCLIP_MODEL", "patrickjohncyh/fashion-clip")


def get_device(prefer: Optional[str] = None) -> torch.device:
    global _DEVICE
    if _DEVICE is not None and prefer is None:
        return _DEVICE

    if prefer:
        d = prefer.lower()
        if d == "cuda" and torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
        elif d == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _DEVICE = torch.device("mps")
        else:
            _DEVICE = torch.device("cpu")
        return _DEVICE

    if torch.cuda.is_available():
        _DEVICE = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _DEVICE = torch.device("mps")
    else:
        _DEVICE = torch.device("cpu")
    return _DEVICE


def _load_model(*, device: Optional[torch.device] = None):
    """
    Lazily load FashionCLIP model once per process.
    FashionCLIP: patrickjohncyh/fashion-clip — 패션 도메인 특화 CLIP ViT-B/32.
    출력 512-dim, 기존 CLIP과 API 호환.
    """
    global _MODEL, _PROCESSOR

    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    with _LOAD_LOCK:
        if _MODEL is None or _PROCESSOR is None:
            from transformers import CLIPModel, CLIPProcessor

            dev = device or get_device()
            _PROCESSOR = CLIPProcessor.from_pretrained(_MODEL_NAME)
            _MODEL = CLIPModel.from_pretrained(_MODEL_NAME).to(dev)
            _MODEL.eval()
            logger.info("[EMBEDDING] loaded FashionCLIP model=%s device=%s", _MODEL_NAME, dev)

    return _MODEL, _PROCESSOR


def encode_image(image_path: str, *, device: Optional[str] = None) -> List[float]:
    """
    Encode a single image into a normalized FashionCLIP embedding.
    Returns List[float] of length 512.
    """
    if not image_path or not isinstance(image_path, str):
        raise ValueError("image_path must be a non-empty string")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    dev = get_device(device)
    model, processor = _load_model(device=dev)

    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"invalid image file: {image_path}") from e

    inputs = processor(images=image, return_tensors="pt").to(dev)

    with torch.inference_mode():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb[0].detach().cpu().tolist()


def encode_images(
    image_paths: Iterable[str],
    *,
    device: Optional[str] = None,
    batch_size: int = 32,
) -> List[List[float]]:
    """Encode multiple images in batches for higher throughput."""
    paths = list(image_paths or [])
    if not paths:
        return []

    dev = get_device(device)
    model, processor = _load_model(device=dev)

    out: List[List[float]] = []
    i = 0
    while i < len(paths):
        batch_paths = paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"image not found: {p}")
            try:
                with Image.open(p) as img:
                    images.append(img.convert("RGB"))
            except UnidentifiedImageError as e:
                raise ValueError(f"invalid image file: {p}") from e

        inputs = processor(images=images, return_tensors="pt", padding=True).to(dev)

        with torch.inference_mode():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        out.extend(emb.detach().cpu().tolist())
        i += batch_size

    return out
