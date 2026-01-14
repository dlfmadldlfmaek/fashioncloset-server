# test_style_similarity.py

print("🔥 test_style_similarity.py START")

from services.style_encoder import load_style_anchors
from services.outfit_encoder import encode_outfit_image, cosine_similarity

def main():
    print("✅ main() called")

    # 1️⃣ 스타일 앵커 로드
    style_anchors = load_style_anchors()
    print("Loaded styles:", style_anchors.keys())

    # 2️⃣ 테스트할 코디 이미지 경로
    outfit_path = "data/user_outfits/test_01.jpg"
    print("Using outfit:", outfit_path)

    outfit_vec = encode_outfit_image(outfit_path)
    print("Outfit vector shape:", len(outfit_vec))

    # 3️⃣ 스타일별 유사도 계산
    print("\n=== STYLE SIMILARITY ===")
    for style, anchor_vec in style_anchors.items():
        score = cosine_similarity(outfit_vec, anchor_vec)
        print(f"{style:10s} : {score:.3f}")

if __name__ == "__main__":
    main()

