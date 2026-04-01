# services/calendar_retention.py
from __future__ import annotations

import datetime as dt
import logging
from typing import Dict

from google.cloud.firestore import FieldPath

from services.firestore import get_db

logger = logging.getLogger(__name__)


def _today_utc_date() -> dt.date:
    # 서버 시간 기준. 필요하면 KST로 바꾸고 싶으면 여기만 바꾸면 됨.
    return dt.datetime.utcnow().date()


def cleanup_outfits_for_non_premium(retention_days_free: int = 90, dry_run: bool = True) -> Dict:
    """
    무료 유저(non-premium)의 outfits 중 'cutoffDate(YYYY-MM-DD) 이전' 문서 삭제.
    - 유료는 무제한 보존하므로 제외.
    """
    db = get_db()

    today = _today_utc_date()
    cutoff = today - dt.timedelta(days=int(retention_days_free))
    cutoff_id = cutoff.isoformat()  # "YYYY-MM-DD"

    users_ref = db.collection("users")
    users = list(users_ref.stream())

    users_scanned = 0
    users_processed = 0
    deleted = 0

    for u in users:
        users_scanned += 1
        uid = u.id
        data = u.to_dict() or {}
        is_premium = bool(data.get("isPremium", False))
        if is_premium:
            continue

        users_processed += 1

        outfits_ref = users_ref.document(uid).collection("outfits")

        # 문서ID가 YYYY-MM-DD 라는 전제에서:
        # docId < cutoff_id 인 문서들을 삭제
        # 파이썬 클라이언트가 end_before를 지원하면 그게 가장 정확.
        q = outfits_ref.order_by(FieldPath.document_id())
        try:
            q = q.end_before({FieldPath.document_id(): cutoff_id})
        except Exception:
            # end_before 미지원/버전 이슈 -> end_at(cutoff_id) 후 cutoff_id 자체는 유지시키도록 skip
            q = q.end_at({FieldPath.document_id(): cutoff_id})

        docs = list(q.stream())
        if not docs:
            continue

        # end_at fallback에서 cutoff_id 문서를 지우지 않도록 방어
        docs = [d for d in docs if d.id < cutoff_id]

        if dry_run:
            deleted += len(docs)
            continue

        # Firestore batch delete(500 제한)
        chunk = []
        for d in docs:
            chunk.append(d.reference)
            if len(chunk) >= 450:
                batch = db.batch()
                for ref in chunk:
                    batch.delete(ref)
                batch.commit()
                deleted += len(chunk)
                chunk.clear()

        if chunk:
            batch = db.batch()
            for ref in chunk:
                batch.delete(ref)
            batch.commit()
            deleted += len(chunk)

    logger.info(
        "[CLEANUP] dryRun=%s cutoff=%s scanned=%d processed=%d deleted=%d",
        dry_run, cutoff_id, users_scanned, users_processed, deleted
    )

    return {
        "dryRun": dry_run,
        "cutoffDate": cutoff_id,
        "usersScanned": users_scanned,
        "usersProcessed": users_processed,
        "outfitsDeleted": deleted,
    }
