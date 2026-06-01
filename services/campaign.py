# services/campaign.py
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

from google.cloud import firestore

from services.firestore import get_db
from services.push import send_to_tokens

logger = logging.getLogger("campaign")

# 한국 고정 오프셋(DST 없음). 클라이언트가 디바이스 로컬(KST)로 스트릭을
# 계산하므로 서버도 KST 자정 기준을 맞춘다.
KST = timezone(timedelta(hours=9))

# 한 번에 스캔할 최대 유저 수 (Cloud Run 타임아웃/메모리 안전장치).
_DEFAULT_SCAN_LIMIT = 3000

_DAY_MS = 86_400_000

# 캠페인 메시지 타입: (title, body, route)
Message = Tuple[str, str, str]
# 세그먼트 판정 함수: (user_data, today_start_ms, now_ms) -> Message | None
Segment = Callable[[dict, int, int], Optional[Message]]


def _kst_today_start_ms() -> int:
    now = datetime.now(KST)
    midnight = datetime(now.year, now.month, now.day, tzinfo=KST)
    return int(midnight.timestamp() * 1000)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ts_to_ms(ts) -> Optional[int]:
    """Firestore 타임스탬프(DatetimeWithNanoseconds) 또는 int(ms) → ms."""
    if ts is None:
        return None
    try:
        return int(ts.timestamp() * 1000)  # datetime-like
    except Exception:
        try:
            return int(ts)  # 이미 ms
        except Exception:
            return None


# ───────────────────────── 세그먼트 정의 ─────────────────────────
#
# 모두 users/{uid} 문서 필드만으로 판정 (서브컬렉션 조회 없이 1-pass 스캔).
#   - currentStreak (int)        : 연속 착용 기록 일수
#   - lastWearLogDate (int, ms)  : 마지막 착용 기록 시각 (KST 자정 비교)
#   - fcmTokenUpdatedAt (ts)     : 마지막 앱 실행 프록시 (splash에서 매번 갱신)
#   - createdAt (int, ms)        : 가입 시각
#   - fcmToken (str)             : 발송 대상 토큰 (run_campaign에서 별도 확인)


def _streak_risk(data: dict, today_start_ms: int, now_ms: int) -> Optional[Message]:
    """스트릭이 쌓였는데 오늘 아직 기록 안 한 유저 — 손실회피 자극. (서비스성 알림)"""
    streak = int(data.get("currentStreak", 0) or 0)
    last = int(data.get("lastWearLogDate", 0) or 0)
    if streak < 3:
        return None
    if last >= today_start_ms:
        return None  # 오늘 이미 기록함 → 보낼 필요 없음
    near_bonus = (streak + 1) % 7 == 0  # 다음 기록이 7일 보너스
    if near_bonus:
        body = f"스트릭 {streak}일! 오늘 착장 기록하면 보너스 토큰 받아요 🔥"
    else:
        body = f"스트릭 {streak}일째 🔥 오늘 착장 기록하고 이어가세요"
    return ("오늘 뭐 입었어요?", body, "/closet")


def _dormant(data: dict, today_start_ms: int, now_ms: int) -> Optional[Message]:
    """7~60일 미접속 유저 복귀 유도. (광고성 — 마케팅 수신 동의자만, 주간 발송)"""
    # 정보통신망법: 광고성 정보는 수신 동의자에게만.
    if data.get("marketingConsent") is not True:
        return None
    last_active = _ts_to_ms(data.get("fcmTokenUpdatedAt"))
    if last_active is None:
        return None
    days_idle = (now_ms - last_active) / _DAY_MS
    if days_idle < 7 or days_idle > 60:
        return None  # 너무 짧으면 스팸, 너무 길면 비활성
    return (
        "새 코디가 기다려요",
        "옷장 속 옷으로 새 추천이 떴어요. 오늘의 룩 확인해보세요 👗",
        "/recommend",
    )


def _onboarding_d1(data: dict, today_start_ms: int, now_ms: int) -> Optional[Message]:
    """가입 후 1일차 유저에게 첫 가상 피팅 유도. (서비스성 온보딩)"""
    created = int(data.get("createdAt", 0) or 0)
    if created <= 0:
        return None
    age_days = (now_ms - created) / _DAY_MS
    if not (1 <= age_days < 2):
        return None
    return (
        "데모 토큰이 남아있어요",
        "옷 하나만 담으면 바로 입어볼 수 있어요. 무료로 가상 피팅 해보세요 ✨",
        "/closet",
    )


CAMPAIGNS: Dict[str, Segment] = {
    "streak_risk": _streak_risk,
    "dormant": _dormant,
    "onboarding_d1": _onboarding_d1,
}

# 세그먼트 조건을 무시하고 디바이스에서 동작 확인할 때 쓰는 대표 메시지.
_SAMPLE: Dict[str, Message] = {
    "streak_risk": ("오늘 뭐 입었어요?", "스트릭 3일째 🔥 오늘 착장 기록하고 이어가세요", "/closet"),
    "dormant": ("새 코디가 기다려요", "옷장 속 옷으로 새 추천이 떴어요. 오늘의 룩 확인해보세요 👗", "/recommend"),
    "onboarding_d1": ("데모 토큰이 남아있어요", "옷 하나만 담으면 바로 입어볼 수 있어요 ✨", "/closet"),
}


def list_campaigns() -> List[str]:
    return list(CAMPAIGNS.keys())


def run_campaign(
    name: str,
    *,
    dry_run: bool = False,
    scan_limit: int = _DEFAULT_SCAN_LIMIT,
) -> dict:
    """
    캠페인 세그먼트에 해당하는 유저에게 푸시 발송.
    dry_run=True면 스캔/타겟 수만 집계하고 실제 발송은 안 한다.
    """
    fn = CAMPAIGNS.get(name)
    if fn is None:
        raise ValueError(f"unknown campaign: {name}")

    db = get_db()
    today_start_ms = _kst_today_start_ms()
    now_ms = _now_ms()

    users = db.collection("users").limit(scan_limit).stream()

    scanned = 0
    no_token = 0
    targeted = 0
    # 동일 메시지끼리 토큰을 묶어 멀티캐스트.
    buckets: Dict[Message, List[str]] = {}

    for u in users:
        scanned += 1
        data = u.to_dict() or {}
        msg = fn(data, today_start_ms, now_ms)
        if msg is None:
            continue
        token = data.get("fcmToken")
        if not token:
            no_token += 1
            continue
        targeted += 1
        buckets.setdefault(msg, []).append(token)

    if scanned >= scan_limit:
        logger.warning(
            "campaign=%s scan_limit(%d) 도달 — 일부 유저 미스캔 가능", name, scan_limit
        )

    sent = 0
    failed = 0
    purged = 0

    if not dry_run:
        for (title, body, route), tokens in buckets.items():
            s, f, invalid = send_to_tokens(tokens, title, body, route, dry_run=False)
            sent += s
            failed += f
            if invalid:
                purged += _purge_invalid_tokens(db, invalid)

    return {
        "campaign": name,
        "mode": "dry_run" if dry_run else "send",
        "scanned": scanned,
        "targeted": targeted,
        "no_token": no_token,
        "sent": sent,
        "failed": failed,
        "purged_invalid": purged,
        "scan_truncated": scanned >= scan_limit,
    }


def send_test_to_uid(name: str, uid: str) -> dict:
    """세그먼트 조건과 무관하게 특정 유저 1명에게 대표 메시지 발송 (디바이스 검증용)."""
    if name not in CAMPAIGNS:
        raise ValueError(f"unknown campaign: {name}")
    db = get_db()
    doc = db.collection("users").document(uid).get()
    if not doc.exists:
        return {"ok": False, "reason": "user_not_found"}
    token = (doc.to_dict() or {}).get("fcmToken")
    if not token:
        return {"ok": False, "reason": "no_fcm_token"}
    title, body, route = _SAMPLE[name]
    s, f, _ = send_to_tokens([token], title, body, route, dry_run=False)
    return {"ok": s > 0, "campaign": name, "route": route, "sent": s, "failed": f}


def _purge_invalid_tokens(db, tokens: List[str]) -> int:
    """죽은 토큰을 가진 유저 문서에서 fcmToken 필드 제거 (best-effort)."""
    purged = 0
    for t in tokens:
        try:
            q = db.collection("users").where("fcmToken", "==", t).limit(1).get()
            for d in q:
                d.reference.update({"fcmToken": firestore.DELETE_FIELD})
                purged += 1
        except Exception as e:
            logger.warning("purge token failed: %s", e)
    return purged
