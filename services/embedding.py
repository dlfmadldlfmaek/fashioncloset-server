# services/embedding.py
from __future__ import annotations

import os
import threading
from typing import Iterable, List, Optional

import torch
import clip
from PIL import Image, UnidentifiedImageError


_DEVICE: Optional[torch.device] = None
_MODEL = None
_PREPROCESS = None
_LOAD_LOCK = threading.Lock()


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


def _load_model(*, device: Optional[torch.device] = None, model_name: str = "ViT-B/32"):
    """Lazily load CLIP model once per process."""
    global _MODEL, _PREPROCESS

    if _MODEL is not None and _PREPROCESS is not None:
        return _MODEL, _PREPROCESS

    with _LOAD_LOCK:
        if _MODEL is None or _PREPROCESS is None:
            dev = device or get_device()
            model, preprocess = clip.load(model_name, device=dev)
            model.eval()
            _MODEL, _PREPROCESS = model, preprocess

    return _MODEL, _PREPROCESS


def encode_image(image_path: str, *, device: Optional[str] = None) -> List[float]:
    """Encode a single image into a normalized CLIP embedding (512-dim)."""
    if not image_path or not isinstance(image_path, str):
        raise ValueError("image_path must be a non-empty string")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    dev = get_device(device)
    model, preprocess = _load_model(device=dev)

    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"invalid image file: {image_path}") from e

    image_input = preprocess(image).unsqueeze(0).to(dev)

    with torch.inference_mode():
        emb = model.encode_image(image_input)
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
    model, preprocess = _load_model(device=dev)

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
                    images.append(preprocess(img.convert("RGB")))
            except UnidentifiedImageError as e:
                raise ValueError(f"invalid image file: {p}") from e

        image_input = torch.stack(images, dim=0).to(dev)

        with torch.inference_mode():
            emb = model.encode_image(image_input)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        out.extend(emb.detach().cpu().tolist())
        i += batch_size

    return out
