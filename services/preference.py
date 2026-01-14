import logging
from collections import defaultdict
from datetime import datetime, timezone
from google.cloud import firestore

# -------------------------------------------------
# logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# 내부 util: Firestore Client (lazy)
# -------------------------------------------------
def _get_db():
    return firestore.Client()

# -------------------------------------------------
# 사용자 누적 선호도 로드 (🔥 핵심)
# -------------------------------------------------
def load_user_preference(user_id: str) -> dict:
    """
    누적 사용자 선호도 1문서 로드
    Firestore read 1회
    """
    db = _get_db()

    doc = (
        db.collection("users")
        .document(user_id)
        .collection("meta")
        .document("preference")
        .get()
    )

    if not doc.exists:
        return {
            "category": defaultdict(float),
            "season": defaultdict(float),
            "color": defaultdict(float),
        }

    data = doc.to_dict() or {}
    return {
        "category": defaultdict(float, data.get("category", {})),
        "season": defaultdict(float, data.get("season", {})),
        "color": defaultdict(float, data.get("color", {})),
    }

# -------------------------------------------------
# 좋아요 시 선호도 즉시 누적
# -------------------------------------------------
def update_preference_on_like(user_id: str, log: dict):
    db = _get_db()

    ref = (
        db.collection("users")
        .document(user_id)
        .collection("meta")
        .document("preference")
    )

    ref.set(
        {
            f"category.{log['mainCategory']}": firestore.Increment(1),
            f"season.{log['season']}": firestore.Increment(1),
            f"color.{log['color']}": firestore.Increment(1),
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )

# -------------------------------------------------
# 추천 히스토리 저장 (batch write)
# -------------------------------------------------
def save_recommendation_history_batch(user_id: str, clothes_ids: list[str]):
    db = _get_db()
    batch = db.batch()

    ref = (
        db.collection("users")
        .document(user_id)
        .collection("recommendations")
    )

    now = datetime.now(timezone.utc)

    for cid in clothes_ids:
        doc_ref = ref.document()
        batch.set(
            doc_ref,
            {
                "clothesId": cid,
                "recommendedAt": now,
            },
        )

    batch.commit()
