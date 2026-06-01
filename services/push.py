# services/push.py
from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from services.auth import _ensure_firebase_app

logger = logging.getLogger("push")

# send_each_for_multicast 한 번에 보낼 수 있는 토큰 수 한도 (FCM 500).
_MULTICAST_LIMIT = 500

# 죽은 토큰으로 간주해 정리할 예외 이름.
_DEAD_TOKEN_ERRORS = {"UnregisteredError", "SenderIdMismatchError"}


def _messaging():
    """firebase_admin app을 보장한 뒤 messaging 모듈 반환."""
    _ensure_firebase_app()
    from firebase_admin import messaging  # type: ignore

    return messaging


def send_to_tokens(
    tokens: Sequence[str],
    title: str,
    body: str,
    route: Optional[str] = None,
    *,
    dry_run: bool = False,
) -> Tuple[int, int, List[str]]:
    """
    FCM 멀티캐스트로 다수 토큰에 알림 발송.

    클라이언트(`fcm_service.dart`)는 data['route']를 읽어 go_router로 딥링크한다.
    따라서 route(예: '/closet', '/recommend')를 data 페이로드로 실어 보낸다.

    반환: (성공 수, 실패 수, 죽은 토큰 목록)
      - 죽은 토큰: UNREGISTERED 등 → 호출측에서 fcmToken 정리 권장.
      - dry_run=True면 FCM 검증만 하고 실제 전송 안 함.
    """
    msg = _messaging()
    tokens = [t for t in tokens if t]
    if not tokens:
        return (0, 0, [])

    data = {"route": route} if route else {}
    notification = msg.Notification(title=title, body=body)
    # 리인게이지먼트는 즉시성보다 도달이 중요 — high priority로 깨움.
    android = msg.AndroidConfig(priority="high")

    success = 0
    failure = 0
    invalid: List[str] = []

    for i in range(0, len(tokens), _MULTICAST_LIMIT):
        chunk = list(tokens[i : i + _MULTICAST_LIMIT])
        multicast = msg.MulticastMessage(
            tokens=chunk,
            notification=notification,
            data=data,
            android=android,
        )
        resp = msg.send_each_for_multicast(multicast, dry_run=dry_run)
        success += resp.success_count
        failure += resp.failure_count
        for idx, r in enumerate(resp.responses):
            if r.success:
                continue
            exc = getattr(r, "exception", None)
            if exc is not None and exc.__class__.__name__ in _DEAD_TOKEN_ERRORS:
                invalid.append(chunk[idx])
            else:
                logger.warning("FCM send failed: %s", exc)

    return (success, failure, invalid)
