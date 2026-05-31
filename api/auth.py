# api/auth.py
"""
Anti-abuse server endpoints for account lifecycle.

Endpoints:
  POST /auth/grant-welcome-tokens
    Idempotently grants the welcome bonus (`AppConstants.initialFreeTokens`)
    if the caller's identifiers do not match any recently-deleted account.

  POST /auth/delete-account
    Records identity hashes for the caller into `deleted_users/{hash}` with a
    90-day TTL, deletes their Firestore data, then deletes their Firebase Auth
    account.

Why server-side:
  Token grants and account deletion used to happen entirely on the client, so
  a malicious user could create → delete → recreate the same account and
  collect the welcome bonus repeatedly. Moving the grant + deletion recording
  to the server lets us track identifiers across re-signups without exposing
  raw PII (we store SHA-256 hashes only).

  Firestore TTL must be configured in the Firebase Console with the field
  `expiresAt` on the `deleted_users` collection to auto-purge after 90 days.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from services.auth import verify_firebase_token
from services.identity import identity_hashes_from_user
from services.rate_limit import limiter

logger = logging.getLogger("auth_api")

router = APIRouter(prefix="/auth", tags=["auth"])

# Keep in sync with AppConstants.initialFreeTokens (client).
WELCOME_TOKEN_COUNT = 3

# How long a deleted account's identifiers stay on the blocklist.
DELETION_TTL_DAYS = 90


def _get_db():
    """Lazy Firestore client (Admin SDK)."""
    import firebase_admin  # type: ignore
    from firebase_admin import firestore  # type: ignore

    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return firestore.client()


def _get_admin_auth():
    """Lazy firebase_admin.auth module."""
    import firebase_admin  # type: ignore
    from firebase_admin import auth as admin_auth  # type: ignore

    if not firebase_admin._apps:
        firebase_admin.initialize_app()
    return admin_auth


def _now_ts() -> float:
    return time.time()


def _expires_at_dt() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=DELETION_TTL_DAYS)


def _any_hash_blocked(db, hashes: list[str]) -> bool:
    """
    Return True if any of the given identity hashes has a non-expired record
    in `deleted_users`. Hashes with expired TTL are treated as absent (TTL
    cleanup may lag a bit).
    """
    if not hashes:
        return False
    now = datetime.now(timezone.utc)
    coll = db.collection("deleted_users")
    for h in hashes:
        snap = coll.document(h).get()
        if not snap.exists:
            continue
        data = snap.to_dict() or {}
        expires_at = data.get("expiresAt")
        # If expiresAt missing, treat as still-blocked (defensive).
        if expires_at is None:
            return True
        # Firestore returns a DatetimeWithNanoseconds (timezone-aware).
        try:
            if expires_at > now:
                return True
        except TypeError:
            # Mismatched tz — treat as blocked.
            return True
    return False


# =========================================================
# Schemas
# =========================================================
class GrantResponse(BaseModel):
    granted: bool
    tokens: int
    reason: str | None = None


class DeleteResponse(BaseModel):
    ok: bool


# =========================================================
# Routes
# =========================================================
@router.post("/grant-welcome-tokens", response_model=GrantResponse)
@limiter.limit("6/minute")
def grant_welcome_tokens(
    request: Request,
    token: dict = Depends(verify_firebase_token),
):
    """
    Grant the welcome bonus once per identity.

    Decision order:
      1. If `welcome_tokens_granted/{uid}` already exists → already granted.
      2. If any of the caller's identity hashes is in `deleted_users` (and not
         expired) → previously deleted, refuse to grant. Also write the
         idempotency key so retries return the same answer cheaply.
      3. Otherwise transactionally increment `users/{uid}.tokens` by
         WELCOME_TOKEN_COUNT and create the idempotency key.
    """
    uid = token.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="uid missing from token")

    db = _get_db()
    granted_ref = db.collection("welcome_tokens_granted").document(uid)
    user_ref = db.collection("users").document(uid)

    # 1. Idempotency check.
    if granted_ref.get().exists:
        return GrantResponse(granted=False, tokens=0, reason="already_granted")

    # 2. Identity blocklist check.
    try:
        admin_auth = _get_admin_auth()
        user_record = admin_auth.get_user(uid)
    except Exception as e:
        logger.warning("[GRANT] get_user(%s) failed: %s", uid, e)
        raise HTTPException(status_code=500, detail="user lookup failed")

    hashes = identity_hashes_from_user(user_record)
    if _any_hash_blocked(db, hashes):
        # Mark as decided so subsequent calls return cheaply.
        granted_ref.set({
            "decidedAt": _now_ts(),
            "outcome": "blocked_previously_deleted",
            "tokensGranted": 0,
        })
        logger.info("[GRANT] uid=%s blocked: previously deleted identity", uid)
        return GrantResponse(
            granted=False,
            tokens=0,
            reason="previously_deleted",
        )

    # 3. Grant + idempotency in a transaction.
    from google.cloud import firestore as gcf  # type: ignore

    transaction = db.transaction()

    @gcf.transactional
    def _grant(txn) -> None:
        # Re-read inside the txn to guard against races.
        if granted_ref.get(transaction=txn).exists:
            return
        txn.set(user_ref, {"tokens": gcf.Increment(WELCOME_TOKEN_COUNT)}, merge=True)
        txn.set(granted_ref, {
            "decidedAt": _now_ts(),
            "outcome": "granted",
            "tokensGranted": WELCOME_TOKEN_COUNT,
        })

    try:
        _grant(transaction)
    except Exception as e:
        logger.exception("[GRANT] uid=%s txn failed: %s", uid, e)
        raise HTTPException(status_code=500, detail="grant failed")

    logger.info("[GRANT] uid=%s granted %d tokens", uid, WELCOME_TOKEN_COUNT)
    return GrantResponse(granted=True, tokens=WELCOME_TOKEN_COUNT)


@router.post("/delete-account", response_model=DeleteResponse)
@limiter.limit("3/minute")
def delete_account(
    request: Request,
    token: dict = Depends(verify_firebase_token),
):
    """
    Server-side hard delete with identity hash recording.

    Order matters: we must record identity hashes BEFORE deleting the auth
    account, because after deletion `get_user(uid)` can no longer recover the
    providerData. Storage cleanup stays on the client (its existing path-list
    + delete loop already works and we'd rather not duplicate it server-side).
    """
    uid = token.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="uid missing from token")

    db = _get_db()
    admin_auth = _get_admin_auth()

    # Step 1: read identifiers and record blocklist hashes.
    try:
        user_record = admin_auth.get_user(uid)
    except Exception as e:
        logger.warning("[DELETE] get_user(%s) failed: %s", uid, e)
        raise HTTPException(status_code=404, detail="user not found")

    hashes = identity_hashes_from_user(user_record)
    expires_at = _expires_at_dt()
    deleted_coll = db.collection("deleted_users")
    for h in hashes:
        try:
            deleted_coll.document(h).set({
                "deletedAt": _now_ts(),
                "expiresAt": expires_at,
            })
        except Exception as e:
            # Don't fail the whole deletion over one hash — log and continue.
            logger.warning("[DELETE] write deleted_users/%s failed: %s", h[:8], e)

    # Step 2: delete Firestore subcollections and the user doc.
    user_ref = db.collection("users").document(uid)
    for sub in ("clothes", "wearLogs", "outfits", "quota"):
        try:
            docs = user_ref.collection(sub).stream()
            for d in docs:
                d.reference.delete()
        except Exception as e:
            logger.warning("[DELETE] subcollection %s for uid=%s failed: %s", sub, uid, e)
    try:
        user_ref.delete()
    except Exception as e:
        logger.warning("[DELETE] users/%s delete failed: %s", uid, e)

    # Drop the welcome-grant idempotency marker too, so it doesn't leak the
    # uid forever (the deleted_users blocklist is the source of truth now).
    try:
        db.collection("welcome_tokens_granted").document(uid).delete()
    except Exception:
        pass

    # Step 3: delete the Firebase Auth account.
    try:
        admin_auth.delete_user(uid)
    except Exception as e:
        logger.exception("[DELETE] delete_user(%s) failed: %s", uid, e)
        raise HTTPException(status_code=500, detail="auth account delete failed")

    logger.info("[DELETE] uid=%s deleted; recorded %d identity hashes", uid, len(hashes))
    return DeleteResponse(ok=True)
