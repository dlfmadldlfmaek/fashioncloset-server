from flask import Flask, request, jsonify
from google.cloud import firestore
from collections import defaultdict

app = Flask(__name__)

def _get_db():
    # Functions 환경에서 안전한 lazy client
    return firestore.Client()

@app.route("/user-pref", methods=["GET"])
def get_user_pref():
    """
    GET /user-pref?userId=...
    반환: 사용자 누적 선호도(pref)
    """
    user_id = request.args.get("userId")
    if not user_id:
        return jsonify({
            "category": {},
            "season": {},
            "color": {},
        }), 200

    try:
        db = _get_db()
        doc = (
            db.collection("users")
            .document(user_id)
            .collection("meta")
            .document("preference")
            .get()
        )

        if not doc.exists:
            # 선호도 없으면 빈 값
            return jsonify({
                "category": {},
                "season": {},
                "color": {},
            }), 200

        data = doc.to_dict() or {}
        return jsonify({
            "category": data.get("category", {}),
            "season": data.get("season", {}),
            "color": data.get("color", {}),
        }), 200

    except Exception:
        # 🔥 어떤 오류든 추천 서버에 영향 주지 않음
        return jsonify({
            "category": {},
            "season": {},
            "color": {},
        }), 200
