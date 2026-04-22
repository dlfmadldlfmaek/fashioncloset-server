# services/compatibility.py
"""
Phase 2: 아이템 호환성 모델.

경량 MLP (2x512 -> 256 -> 1, sigmoid) 로 두 아이템의
FashionCLIP 임베딩 쌍의 호환 확률을 예측.

Colab에서 학습한 compatibility_model.pt (TorchScript) 를 로드하여 추론.
모델이 없으면 gracefully 0.5(중립)를 반환하여 기존 규칙 기반만 사용.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

_MODEL = None
_DEVICE = None
_LOAD_LOCK = threading.Lock()
_LOAD_ATTEMPTED = False


def _resolve_model_path() -> Path:
    raw = (os.getenv("COMPATIBILITY_MODEL_PATH") or "").strip()
    if raw:
        return Path(raw)
    base = Path(os.getenv("APP_ROOT", "/app"))
    return base / "data" / "compatibility_model.pt"


MODEL_PATH: Path = _resolve_model_path()


def _load_model():
    global _MODEL, _DEVICE, _LOAD_ATTEMPTED

    if _LOAD_ATTEMPTED:
        return _MODEL, _DEVICE

    with _LOAD_LOCK:
        if _LOAD_ATTEMPTED:
            return _MODEL, _DEVICE

        _LOAD_ATTEMPTED = True

        if not MODEL_PATH.exists():
            logger.info("[COMPAT] model not found at %s — using neutral score (0.5)", MODEL_PATH)
            return None, None

        try:
            import torch

            _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            _MODEL = torch.jit.load(str(MODEL_PATH), map_location=_DEVICE)
            _MODEL.eval()
            logger.info("[COMPAT] loaded compatibility model from %s device=%s", MODEL_PATH, _DEVICE)
        except Exception as e:
            logger.exception("[COMPAT] model load failed err=%s", e)
            _MODEL = None

    return _MODEL, _DEVICE


def _as_vec(x: Union[np.ndarray, Sequence[float], None]) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        v = np.asarray(x, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if v.size == 0 or not np.isfinite(v).all():
        return None
    return v


def predict_compatibility(emb_a: Union[np.ndarray, Sequence[float], None],
                          emb_b: Union[np.ndarray, Sequence[float], None]) -> float:
    """
    두 아이템의 FashionCLIP 임베딩으로 호환 확률을 예측.

    Returns:
        float 0~1 (높을수록 호환).
        모델 미로드 시 0.5 (중립) 반환.
    """
    va = _as_vec(emb_a)
    vb = _as_vec(emb_b)
    if va is None or vb is None:
        return 0.5

    model, device = _load_model()
    if model is None:
        return 0.5

    try:
        import torch

        # concat: (1, 1024)
        concat = np.concatenate([va, vb], axis=0)
        tensor = torch.tensor(concat, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.inference_mode():
            score = model(tensor).item()

        return float(max(0.0, min(1.0, score)))
    except Exception as e:
        logger.warning("[COMPAT] predict failed err=%s", e)
        return 0.5
