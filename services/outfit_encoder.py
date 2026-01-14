import torch
import clip
import numpy as np
from PIL import Image
import requests
from io import BytesIO
# ---------------------------
# 🔥 Lazy-loaded CLIP
# ---------------------------
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_preprocess = None


def _get_clip():
    """
    CLIP 모델을 최초 1회만 로딩
    """
    global _model, _preprocess

    if _model is None:
        import torch
        import clip

        _model, _preprocess = clip.load("ViT-B/32", device=_device)
        _model.eval()

        print("🔥 CLIP model loaded")

    return _model, _preprocess


# ---------------------------
# 📸 Local image → vector
# ---------------------------
def encode_outfit_image(image_path: str) -> np.ndarray:
    model, preprocess = _get_clip()

    image = preprocess(
        Image.open(image_path).convert("RGB")
    ).unsqueeze(0)

    with model.no_grad():
        vec = model.encode_image(image)

    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()[0]


# ---------------------------
# 🌐 URL image → vector
# ---------------------------
def encode_outfit_image_from_url(url: str) -> np.ndarray:
    model, preprocess = _get_clip()

    res = requests.get(url, timeout=5)
    image = Image.open(BytesIO(res.content)).convert("RGB")

    image = preprocess(image).unsqueeze(0)

    with model.no_grad():
        vec = model.encode_image(image)

    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()[0]


# ---------------------------
# 📐 Cosine similarity
# ---------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
