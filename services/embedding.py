# services/embedding.py
import torch
import clip
from PIL import Image

device = "cpu"
_model = None
_preprocess = None


def _load_model():
    global _model, _preprocess
    if _model is None:
        _model, _preprocess = clip.load("ViT-B/32", device=device)
    return _model, _preprocess


def encode_image(image_path: str) -> list:
    model, preprocess = _load_model()

    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding[0].cpu().tolist()
