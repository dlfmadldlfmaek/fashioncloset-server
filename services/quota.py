# services/quota.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

KST = timezone(timedelta(hours=9))


@dataclass
class QuotaResult:
    allowed: bool
    reason: Optional[str] = None
    dayKey: Optional[str] = None
    freeUsedToday: int = 0
    freeLimit: int = 4
    adCredit: int = 0
    hardLimitToday: int = 12
    usedToday: int = 0
    consumed: Optional[str] = None  # "FREE" | "AD"


def _day_key_kst(now: Optional[datetime] = None) -> str:
    now = now or datetime.now(tz=KST)
    return now.strftime("%Y-%m-%d")


def _firestore():
    try:
        from google.cloud import firestore  # type: ignore
        return firestore
    except Exception as e:
        raise RuntimeError(f"google-cloud-firestore not available: {e}") from e


_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        firestore = _firestore()
        project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
        _CLIENT = firestore.Client(project=project) if project else firestore.Client()
    return _CLIENT


def _doc_ref(db, user_id: str, action: str):
    return db.collection("users").document(user_id).collection("quota").document(action)


def _normalize_state(doc: Dict[str, Any], day_key: str, free_limit: int, hard_limit: int) -> Dict[str, Any]:
    firestore = _firestore()
    cur_day = (doc or {}).get("dayKey")
    if cur_day != day_key:
        return {
            "dayKey": day_key,
            "freeUsedToday": 0,
            "freeLimit": free_limit,
            "adCredit": int((doc or {}).get("adCredit", 0) or 0),
            "hardLimitToday": hard_limit,
            "usedToday": 0,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }

    return {
        "dayKey": day_key,
        "freeUsedToday": int(doc.get("freeUsedToday", 0) or 0),
        "freeLimit": int(doc.get("freeLimit", free_limit) or free_limit),
        "adCredit": int(doc.get("adCredit", 0) or 0),
        "hardLimitToday": int(doc.get("hardLimitToday", hard_limit) or hard_limit),
        "usedToday": int(doc.get("usedToday", 0) or 0),
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }


def consume_quota(*, user_id: str, action: str = "recommend_outfits", free_limit: int = 4, hard_limit_today: int = 12) -> QuotaResult:
    firestore = _firestore()
    db = _client()
    ref = _doc_ref(db, user_id, action)
    day_key = _day_key_kst()

    @firestore.transactional
    def _tx(transaction) -> QuotaResult:
        snap = ref.get(transaction=transaction)
        doc = snap.to_dict() if snap.exists else {}
        st = _normalize_state(doc, day_key, free_limit, hard_limit_today)

        free_used = st["freeUsedToday"]
        free_lim = st["freeLimit"]
        ad_credit = st["adCredit"]
        hard_lim = st["hardLimitToday"]
        used_today = st["usedToday"]

        if used_today >= hard_lim:
            return QuotaResult(False, "HARD_LIMIT", day_key, free_used, free_lim, ad_credit, hard_lim, used_today)

        if ad_credit > 0:
            st["adCredit"] = ad_credit - 1
            st["usedToday"] = used_today + 1
            transaction.set(ref, st, merge=True)
            return QuotaResult(True, None, day_key, free_used, free_lim, ad_credit - 1, hard_lim, used_today + 1, "AD")

        if free_used < free_lim:
            st["freeUsedToday"] = free_used + 1
            st["usedToday"] = used_today + 1
            transaction.set(ref, st, merge=True)
            return QuotaResult(True, None, day_key, free_used + 1, free_lim, ad_credit, hard_lim, used_today + 1, "FREE")

        return QuotaResult(False, "NO_CREDIT", day_key, free_used, free_lim, ad_credit, hard_lim, used_today)

    return _tx(db.transaction())


def refund_quota(*, user_id: str, action: str = "recommend_outfits", consumed: Optional[str], free_limit: int = 4, hard_limit_today: int = 12) -> None:
    if consumed not in {"AD", "FREE"}:
        return

    firestore = _firestore()
    db = _client()
    ref = _doc_ref(db, user_id, action)
    day_key = _day_key_kst()

    @firestore.transactional
    def _tx(transaction) -> None:
        snap = ref.get(transaction=transaction)
        doc = snap.to_dict() if snap.exists else {}
        st = _normalize_state(doc, day_key, free_limit, hard_limit_today)

        if consumed == "AD":
            st["adCredit"] = int(st.get("adCredit", 0) or 0) + 1
        else:
            st["freeUsedToday"] = max(0, int(st.get("freeUsedToday", 0) or 0) - 1)

        st["usedToday"] = max(0, int(st.get("usedToday", 0) or 0) - 1)
        transaction.set(ref, st, merge=True)

    _tx(db.transaction())


def add_ad_credit(*, user_id: str, action: str = "recommend_outfits", amount: int = 1, free_limit: int = 4, hard_limit_today: int = 12) -> Dict[str, Any]:
    firestore = _firestore()
    db = _client()
    ref = _doc_ref(db, user_id, action)
    day_key = _day_key_kst()

    @firestore.transactional
    def _tx(transaction) -> Dict[str, Any]:
        snap = ref.get(transaction=transaction)
        doc = snap.to_dict() if snap.exists else {}
        st = _normalize_state(doc, day_key, free_limit, hard_limit_today)

        st["adCredit"] = int(st.get("adCredit", 0) or 0) + max(1, int(amount))
        transaction.set(ref, st, merge=True)

        return {
            "dayKey": st["dayKey"],
            "freeUsedToday": st["freeUsedToday"],
            "freeLimit": st["freeLimit"],
            "adCredit": st["adCredit"],
            "hardLimitToday": st["hardLimitToday"],
            "usedToday": st["usedToday"],
        }

    return _tx(db.transaction())
