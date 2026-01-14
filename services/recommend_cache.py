import hashlib
import json
from typing import List
from datetime import datetime, timedelta

# ===========================
# 🔑 옷 리스트 해시
# ===========================
def clothes_hash(clothes: List[dict]) -> str:
    normalized = sorted(
        [
            {
                "id": c["id"],
                "m": c.get("mainCategory"),
                "s": c.get("season"),
                "c": c.get("color"),
            }
            for c in clothes
        ],
        key=lambda x: x["id"],
    )

    raw = json.dumps(normalized, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ===========================
# 🧠 추천 결과 캐시 (TTL)
# ===========================
_recommend_cache = {}


def build_cache_key(
    user_id: str,
    nx: int,
    ny: int,
    time_slot: str,
    clothes_hash_key: str,
) -> str:
    return f"{user_id}:{nx}:{ny}:{time_slot}:{clothes_hash_key}"


def get_cached_recommend(key: str):
    cached = _recommend_cache.get(key)
    if not cached:
        return None

    result, expires_at = cached
    if datetime.now() > expires_at:
        del _recommend_cache[key]
        return None

    return result


def set_cached_recommend(
    key: str,
    result: dict,
    minutes: int = 20,
):
    _recommend_cache[key] = (
        result,
        datetime.now() + timedelta(minutes=minutes),
    )
