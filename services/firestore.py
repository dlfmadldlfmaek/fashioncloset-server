# services/firestore.py
from google.cloud import firestore

# ❌ 전역 Client 생성 금지
# db = firestore.Client()


def get_db():
    """
    Cloud Run 안전 Firestore Client 생성
    (요청 시점에만 생성)
    """
    return firestore.Client()


def save_like_log(
    user_id: str,
    clothes_id: str,
    category: str,
    season: str,
    color: str,
):
    db = get_db()

    ref = (
        db.collection("users")
        .document(user_id)
        .collection("likes")
        .document()
    )

    ref.set(
        {
            "clothesId": clothes_id,
            "category": category,
            "season": season,
            "color": color,
            "createdAt": firestore.SERVER_TIMESTAMP,
        }
    )


def save_style_anchor(
    user_id: str,
    style_name: str,
    vector: list,
):
    db = get_db()

    ref = (
        db.collection("users")
        .document(user_id)
        .collection("styleAnchors")
        .document(style_name)
    )

    ref.set(
        {
            "vector": vector,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
    )
