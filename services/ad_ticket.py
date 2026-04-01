# services/ad_ticket.py
"""
Ad ticket service: nonce-based ad credit system.

Flow:
1. Client calls POST /ad/request-token -> server generates a one-time nonce
2. Client shows rewarded ad, passing nonce as custom_data
3. AdMob sends SSV callback to POST /ad/ssv-callback -> server verifies & marks nonce as verified
4. Client calls POST /ad/redeem with the nonce -> server checks nonce is verified, grants ad credit
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Optional, Tuple

from services.firestore import get_db

logger = logging.getLogger("ad_ticket")

# Nonce expires after this many seconds (default 5 minutes)
AD_NONCE_TTL_SEC = int(os.getenv("AD_NONCE_TTL_SEC", "300"))

# Maximum ad credits a user can earn per day
AD_CREDIT_DAILY_LIMIT = int(os.getenv("AD_CREDIT_DAILY_LIMIT", "10"))


def _nonce_collection(user_id: str):
    """Return Firestore collection ref: users/{userId}/adNonces."""
    db = get_db()
    return db.collection("users").document(user_id).collection("adNonces")


def issue_ad_nonce(user_id: str) -> Tuple[str, int]:
    """
    Generate a one-time nonce for ad verification.

    Returns (nonce, expires_at_epoch).
    The nonce is stored in Firestore with status='pending'.
    """
    if not user_id:
        raise ValueError("user_id is required")

    nonce = secrets.token_urlsafe(32)
    now = int(time.time())
    expires_at = now + AD_NONCE_TTL_SEC

    ref = _nonce_collection(user_id).document(nonce)
    ref.set({
        "nonce": nonce,
        "userId": user_id,
        "status": "pending",       # pending -> verified -> redeemed
        "createdAt": now,
        "expiresAt": expires_at,
        "transactionId": None,     # filled by SSV callback
    })

    logger.info("Issued ad nonce for user=%s expires_at=%d", user_id, expires_at)
    return nonce, expires_at


def mark_nonce_verified(
    user_id: str,
    nonce: str,
    transaction_id: Optional[str] = None,
) -> bool:
    """
    Mark a nonce as verified (called after SSV callback validation).

    Returns True if the nonce was successfully marked.
    Returns False if nonce not found, expired, or already used.
    """
    if not user_id or not nonce:
        return False

    ref = _nonce_collection(user_id).document(nonce)
    snap = ref.get()

    if not snap.exists:
        logger.warning("Nonce not found: user=%s nonce=%s", user_id, nonce)
        return False

    data = snap.to_dict() or {}
    now = int(time.time())

    if data.get("status") != "pending":
        logger.warning(
            "Nonce not pending: user=%s nonce=%s status=%s",
            user_id, nonce, data.get("status"),
        )
        return False

    if now > data.get("expiresAt", 0):
        logger.warning("Nonce expired: user=%s nonce=%s", user_id, nonce)
        ref.update({"status": "expired"})
        return False

    ref.update({
        "status": "verified",
        "transactionId": transaction_id,
        "verifiedAt": now,
    })

    logger.info("Nonce verified: user=%s nonce=%s txn=%s", user_id, nonce, transaction_id)
    return True


def redeem_nonce(user_id: str, nonce: str) -> Tuple[bool, str]:
    """
    Redeem a verified nonce to grant ad credit.

    Returns (success, message).
    Only nonces with status='verified' can be redeemed.
    """
    if not user_id or not nonce:
        return False, "userId and nonce are required"

    db = get_db()
    ref = _nonce_collection(user_id).document(nonce)

    @db.transactional
    def _txn(transaction):
        snap = ref.get(transaction=transaction)

        if not snap.exists:
            return False, "nonce not found"

        data = snap.to_dict() or {}
        now = int(time.time())

        status = data.get("status")
        if status == "redeemed":
            return False, "nonce already redeemed"
        if status != "verified":
            return False, f"nonce not verified (status={status})"
        if now > data.get("expiresAt", 0):
            transaction.update(ref, {"status": "expired"})
            return False, "nonce expired"

        # Mark as redeemed
        transaction.update(ref, {
            "status": "redeemed",
            "redeemedAt": now,
        })

        # Grant ad credit via quota service
        from services.quota import add_ad_credit
        add_ad_credit(user_id=user_id, action="recommend_outfits", amount=1)

        return True, "ad credit granted"

    transaction = db.transaction()
    return _txn(transaction)


def verify_ad_ticket(user_id: str, nonce: str) -> Tuple[bool, int]:
    """
    Check if a nonce is valid and return its expiry.

    Returns (is_valid, expires_at).
    """
    if not user_id or not nonce:
        return False, 0

    ref = _nonce_collection(user_id).document(nonce)
    snap = ref.get()

    if not snap.exists:
        return False, 0

    data = snap.to_dict() or {}
    expires_at = int(data.get("expiresAt", 0))
    now = int(time.time())

    is_valid = (
        data.get("status") in ("pending", "verified")
        and now <= expires_at
    )
    return is_valid, expires_at
