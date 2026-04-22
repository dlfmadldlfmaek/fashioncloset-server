# services/style_encoder.py
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

_style_anchors: Optional[Dict[str, np.ndarray]] = None
_lock = threading.Lock()


def _env_flag(name: str, default: str = "0") -> bool:
    v = (os.getenv(name, default) or default).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def _as_vector(x: object) -> Optional[np.ndarray]:
    try:
        v = np.asarray(x, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if v.size == 0 or not np.isfinite(v).all():
        return None
    return v


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0.0:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32)


def _is_near_zero(v: np.ndarray, min_norm: float = 1e-6) -> bool:
    return float(np.linalg.norm(v)) < float(min_norm)


def _default_style_dir() -> Path:
    base = Path(os.getenv("APP_ROOT", "/app"))
    return base / "data" / "style_anchor"


def _resolve_style_dir() -> Path:
    raw = (os.getenv("STYLE_ANCHOR_DIR") or "").strip()
    if not raw:
        return _default_style_dir()

    p = Path(raw)
    if p.is_absolute():
        return p

    return Path(os.getenv("APP_ROOT", "/app")) / p


STYLE_DIR: Path = _resolve_style_dir()

# ✅ main.py에서 import하는 상수 (ImportError 방지)
def _resolve_anchor_json_path() -> Path:
    raw = (os.getenv("STYLE_ANCHOR_JSON") or "").strip()
    if raw:
        return Path(raw)
    # Phase 1: FashionCLIP 앵커 우선, 없으면 기존 앵커 fallback
    v2 = STYLE_DIR / "anchors_v2.json"
    if v2.exists():
        return v2
    return STYLE_DIR / "anchors.json"


ANCHOR_JSON_PATH: Path = _resolve_anchor_json_path()


def _load_from_json(json_path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not json_path.exists():
        return None

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        raw = payload.get("anchors") or {}
        anchors: Dict[str, np.ndarray] = {}

        for k, v in raw.items():
            key = str(k).strip().lower()
            vec = _as_vector(v)
            if vec is None or _is_near_zero(vec):
                continue
            anchors[key] = _l2_normalize(vec)

        logger.info("[STYLE_ANCHOR] loaded anchors.json=%s styles=%d", json_path, len(anchors))
        return anchors
    except Exception as e:
        logger.exception("[STYLE_ANCHOR] failed to read anchors.json err=%s", e)
        return None


def load_style_anchors(*, force_reload: bool = False) -> Dict[str, np.ndarray]:
    global _style_anchors

    if _env_flag("DISABLE_STYLE_ANCHOR", "0"):
        _style_anchors = {}
        return _style_anchors

    with _lock:
        if _style_anchors is not None and not force_reload:
            return _style_anchors

        if not STYLE_DIR.exists():
            logger.warning("[STYLE_ANCHOR] directory not found: %s", STYLE_DIR)
            _style_anchors = {}
            return _style_anchors

        from_json = _load_from_json(ANCHOR_JSON_PATH)
        if from_json:
            _style_anchors = from_json
            return _style_anchors

        from services.outfit_encoder import encode_outfit_image  # lazy import

        anchors: Dict[str, np.ndarray] = {}

        for style_path in sorted([p for p in STYLE_DIR.iterdir() if p.is_dir()]):
            style_key = style_path.name.strip().lower()
            vectors: list[np.ndarray] = []
            expected_dim: Optional[int] = None

            for img_path in sorted([p for p in style_path.iterdir() if p.is_file()]):
                if img_path.suffix.lower() not in _ALLOWED_EXT:
                    continue
                if not img_path.name.lower().startswith(style_key):
                    continue

                try:
                    raw = encode_outfit_image(str(img_path))
                except Exception as e:
                    logger.warning(
                        "[STYLE_ANCHOR][SKIP] encode failed style=%s file=%s err=%s",
                        style_key,
                        img_path.name,
                        e,
                    )
                    continue

                vec = _as_vector(raw)
                if vec is None or _is_near_zero(vec):
                    continue

                if expected_dim is None:
                    expected_dim = int(vec.shape[0])
                elif int(vec.shape[0]) != expected_dim:
                    logger.warning("[STYLE_ANCHOR][SKIP] dim mismatch style=%s file=%s", style_key, img_path.name)
                    continue

                vectors.append(vec)

            if not vectors:
                logger.info("[STYLE_ANCHOR] style=%s has no valid images", style_key)
                continue

            centroid = np.mean(np.stack(vectors, axis=0), axis=0)
            centroid = _l2_normalize(centroid)

            anchors[style_key] = centroid
            logger.info(
                "[STYLE_ANCHOR] fallback centroid style=%s images=%d dim=%d",
                style_key,
                len(vectors),
                int(centroid.shape[0]),
            )

        _style_anchors = anchors
        return _style_anchors


def calc_style_similarity(outfit_vec: object, style_anchor_vec: object) -> float:
    from services.outfit_encoder import cosine_similarity  # lazy import

    a = _as_vector(outfit_vec)
    b = _as_vector(style_anchor_vec)
    if a is None or b is None or a.shape[0] != b.shape[0]:
        return 0.0

    a_n = _l2_normalize(a)
    b_n = _l2_normalize(b)

    if _is_near_zero(a_n) or _is_near_zero(b_n):
        return 0.0

    return float(cosine_similarity(a_n, b_n))
