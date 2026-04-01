# api/ad.py
"""
Ad endpoints for rewarded ad verification and credit redemption.

Flow:
1. POST /ad/request-token  -> Client gets a one-time nonce before showing ad
2. GET  /ad/ssv-callback   -> Google AdMob calls this after ad completion (SSV)
3. POST /ad/redeem          -> Client redeems verified nonce for ad credit
4. POST /ad/verify          -> (debug) Check nonce status
"""


import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from services.ad_ticket import (
    issue_ad_nonce,
    mark_nonce_verified,
    redeem_nonce,
    verify_ad_ticket,
)
from services.admob_ssv import verify_ssv_callback
from services.auth import verify_firebase_token
from services.premium import is_premium_user

router = APIRouter(prefix="/ad", tags=["ad"])
logger = logging.getLogger("ad")


# ─── Request / Response Schemas ───────────────────────────

class AdTokenRequest(BaseModel):
    userId: str = Field(..., min_length=1)


class AdTokenResponse(BaseModel):
    nonce: str
    expiresAt: int


class AdRedeemRequest(BaseModel):
    userId: str = Field(..., min_length=1)
    nonce: str = Field(..., min_length=1)


class AdRedeemResponse(BaseModel):
    success: bool
    message: str


class AdVerifyRequest(BaseModel):
    userId: str = Field(..., min_length=1)
    nonce: str = Field(..., min_length=1)


class AdVerifyResponse(BaseModel):
    valid: bool
    expiresAt: int


# ─── Endpoints ────────────────────────────────────────────

@router.post("/request-token", response_model=AdTokenResponse)
def request_token(
    body: AdTokenRequest,
    token: dict = Depends(verify_firebase_token),
):
    """
    Step 1: Client requests a one-time nonce before showing a rewarded ad.

    The nonce should be passed to AdMob as custom_data so the SSV callback
    can reference it. Nonce expires after 5 minutes.
    """
    user_id = body.userId.strip()

    if token["uid"] != user_id:
        raise HTTPException(status_code=403, detail="forbidden: userId mismatch")

    if is_premium_user(user_id):
        raise HTTPException(
            status_code=400,
            detail="premium user does not need ad ticket",
        )

    nonce, expires_at = issue_ad_nonce(user_id)
    return AdTokenResponse(nonce=nonce, expiresAt=expires_at)


@router.get("/ssv-callback")
def ssv_callback(request: Request):
    """
    Step 2: Google AdMob SSV callback (server-to-server).

    Google calls this URL after the user watches a rewarded ad.
    The query string contains the ad details and a cryptographic signature.
    We verify the signature, then mark the nonce as verified.

    NOTE: This endpoint has NO auth (no Firebase token) because Google calls it.
    Security comes from the ECDSA signature verification.
    """
    query_string = str(request.url.query or "")
    if not query_string:
        logger.warning("SSV callback with empty query string")
        raise HTTPException(status_code=400, detail="missing query parameters")

    # Verify the cryptographic signature
    is_valid, params = verify_ssv_callback(query_string)
    if not is_valid:
        logger.warning("SSV signature verification failed: %s", query_string[:200])
        raise HTTPException(status_code=403, detail="invalid signature")

    # Extract custom_data (contains our nonce) and user_id
    custom_data = params.get("custom_data", "")
    ad_user_id = params.get("user_id", "")
    transaction_id = params.get("transaction_id", "")

    if not custom_data or not ad_user_id:
        logger.warning(
            "SSV callback missing custom_data or user_id: params=%s",
            {k: v for k, v in params.items() if k != "signature"},
        )
        raise HTTPException(
            status_code=400,
            detail="missing custom_data or user_id",
        )

    # custom_data format: "{userId}:{nonce}"
    parts = custom_data.split(":", 1)
    if len(parts) != 2:
        logger.warning("SSV callback invalid custom_data format: %s", custom_data)
        raise HTTPException(status_code=400, detail="invalid custom_data format")

    nonce_user_id, nonce = parts

    # Mark the nonce as verified
    verified = mark_nonce_verified(
        user_id=nonce_user_id,
        nonce=nonce,
        transaction_id=transaction_id,
    )

    if not verified:
        logger.warning(
            "SSV nonce verification failed: user=%s nonce=%s",
            nonce_user_id, nonce,
        )
        # Still return 200 to Google (they don't retry on 4xx)
        return {"status": "nonce_invalid"}

    logger.info(
        "SSV verified: user=%s nonce=%s txn=%s",
        nonce_user_id, nonce, transaction_id,
    )
    return {"status": "ok"}


@router.post("/redeem", response_model=AdRedeemResponse)
def redeem(
    body: AdRedeemRequest,
    token: dict = Depends(verify_firebase_token),
):
    """
    Step 3: Client redeems a verified nonce to receive ad credit.

    Only nonces that have been verified via SSV callback can be redeemed.
    Each nonce can only be redeemed once.
    """
    user_id = body.userId.strip()
    nonce = body.nonce.strip()

    if token["uid"] != user_id:
        raise HTTPException(status_code=403, detail="forbidden: userId mismatch")

    success, message = redeem_nonce(user_id, nonce)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return AdRedeemResponse(success=True, message=message)


@router.post("/verify", response_model=AdVerifyResponse)
def verify_ticket(body: AdVerifyRequest):
    """
    (Debug/test) Check if a nonce is valid and its expiry time.
    """
    user_id = body.userId.strip()
    nonce = body.nonce.strip()

    ok, exp = verify_ad_ticket(user_id, nonce)
    return AdVerifyResponse(valid=ok, expiresAt=exp)
