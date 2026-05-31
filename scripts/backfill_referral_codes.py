"""기존 users 컬렉션의 referralCode를 referral_codes/{code} 미러 컬렉션에 백필.

배경: 클라이언트가 추천 코드를 redeem할 때 users 컬렉션을 직접 쿼리해서
추천인을 찾았는데, Rules가 본인 외 read를 막아서 항상 실패했다. 새 구조는
공개 read 가능한 referral_codes 미러를 두고 그것으로 lookup한다. 신규 가입자는
클라이언트가 가입 시 미러를 같이 작성하지만, 출시 후 누적된 기존 유저는
이 스크립트로 일괄 백필해야 한다.

실행 (Windows PowerShell):
    $env:GOOGLE_CLOUD_PROJECT = "fashioncloset-132d5"
    gcloud auth application-default login    # 처음 한 번만
    python -m scripts.backfill_referral_codes --dry-run    # 미리보기
    python -m scripts.backfill_referral_codes              # 실제 실행

멱등: 미러가 이미 존재하면 건너뜀. 안전하게 재실행 가능.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Iterable

import firebase_admin
from firebase_admin import firestore

logger = logging.getLogger("backfill_referral_codes")


def _init_admin() -> None:
    if not firebase_admin._apps:
        firebase_admin.initialize_app()


def _iter_users(db) -> Iterable:
    # stream()은 페이지네이션을 자동으로 처리하지만 대규모 컬렉션에서는
    # 메모리 안정성을 위해 limit + start_after 패턴을 권장한다. 핏토리
    # 출시 초기 규모에서는 stream()으로 충분.
    return db.collection("users").stream()


def backfill(*, dry_run: bool) -> dict:
    _init_admin()
    db = firestore.client()
    mirrors = db.collection("referral_codes")

    stats = {
        "scanned": 0,
        "no_code": 0,
        "already_mirrored": 0,
        "created": 0,
        "conflict": 0,
        "errors": 0,
    }

    for user_doc in _iter_users(db):
        stats["scanned"] += 1
        data = user_doc.to_dict() or {}
        code = (data.get("referralCode") or "").strip()
        uid = user_doc.id

        if not code:
            stats["no_code"] += 1
            continue

        try:
            mirror_ref = mirrors.document(code)
            existing = mirror_ref.get()
            if existing.exists:
                owner = (existing.to_dict() or {}).get("ownerUid")
                if owner == uid:
                    stats["already_mirrored"] += 1
                else:
                    # 같은 코드를 가진 유저가 둘 이상인 충돌 상태.
                    # 먼저 본 유저가 미러를 차지하고, 나머지는 redeem 불가능.
                    # 별도 정책 결정 전까지 로그만 남긴다.
                    stats["conflict"] += 1
                    logger.warning(
                        "code %s already owned by %s; current user %s skipped",
                        code, owner, uid,
                    )
                continue

            if dry_run:
                logger.info("[dry-run] would create referral_codes/%s -> %s",
                            code, uid)
            else:
                mirror_ref.create({
                    "ownerUid": uid,
                    "createdAt": int(time.time() * 1000),
                })
            stats["created"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["errors"] += 1
            logger.exception("failed to backfill code=%s uid=%s: %s",
                             code, uid, exc)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 쓰기 없이 결과만 출력")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    stats = backfill(dry_run=args.dry_run)
    logger.info("backfill complete: %s", stats)

    if stats["conflict"] > 0:
        logger.warning(
            "%d code collisions detected — those users cannot redeem until "
            "their referralCode is regenerated.", stats["conflict"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
