import os
import numpy as np
from typing import Dict

STYLE_DIR = "data/style_anchors"

# ⚠️ 절대 전역에서 encode_outfit_image 실행 금지
_style_anchors: Dict[str, np.ndarray] | None = None

if os.getenv("DISABLE_STYLE_ANCHOR", "1") == "1":
    def load_style_anchors():
        return {}

def load_style_anchors():
    global _style_anchors
    if _style_anchors is not None:
        return _style_anchors

    if not os.path.exists(STYLE_DIR):
        print(f"[WARN] STYLE_DIR not found: {STYLE_DIR}")
        _style_anchors = {}
        return _style_anchors


    from services.outfit_encoder import encode_outfit_image  # 🔥 지연 import

    anchors = {}

    for style in os.listdir(STYLE_DIR):
        style_path = os.path.join(STYLE_DIR, style)
        if not os.path.isdir(style_path):
            continue

        vectors = []
        for fname in os.listdir(style_path):
            if not fname.lower().endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(style_path, fname)

            try:
                vec = encode_outfit_image(img_path)
                vectors.append(vec)
            except Exception as e:
                print(f"[STYLE ENCODE FAIL] {img_path}: {e}")

        if vectors:
            anchors[style] = np.mean(vectors, axis=0)

    print("🔥 load_style_anchors DONE")
    _style_anchors = anchors
    return anchors


def calc_style_similarity(outfit_vec, style_anchor_vec) -> float:
    from services.outfit_encoder import cosine_similarity  # 🔥 지연 import
    return cosine_similarity(outfit_vec, style_anchor_vec)
