# tools/build_anchors.py
from __future__ import annotations

import sys
from pathlib import Path

# ✅ 프로젝트 루트를 PYTHONPATH에 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
from services.style_anchor import build_style_anchors

BASE_DIR = Path("data/style_anchor")
OUT_PATH = BASE_DIR / "anchors.json"

def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    anchors = build_style_anchors(
        base_dir=str(BASE_DIR),
        text_weight=3.0,
    )

    payload = {
        "version": 1,
        "text_weight": 3.0,
        "anchors": anchors,
        "dim": len(next(iter(anchors.values()))) if anchors else None,
    }

    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"✅ saved: {OUT_PATH} (styles={len(anchors)})")

if __name__ == "__main__":
    main()
