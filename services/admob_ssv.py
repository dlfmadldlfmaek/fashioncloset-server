# services/admob_ssv.py
"""
AdMob Server-Side Verification (SSV) for rewarded ads.

Google sends a callback to your server with query params including a signature.
This module verifies the signature using Google's public ECDSA keys.

Reference: https://developers.google.com/admob/android/ssv
"""
from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Dict, Optional, Tuple
from urllib.parse import unquote

import httpx
from cryptography.hazmat.primitives.asymmetric import ec, utils as ec_utils
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.x509 import load_pem_x509_certificate

logger = logging.getLogger("admob_ssv")

KEYS_URL = "https://www.gstatic.com/admob/reward/verifier-keys.json"
_KEYS_CACHE: Dict[str, ec.EllipticCurvePublicKey] = {}
_KEYS_FETCHED_AT: float = 0.0
_KEYS_TTL: float = 3600.0  # Re-fetch keys every hour
_KEYS_LOCK = threading.Lock()


def _fetch_keys() -> Dict[str, ec.EllipticCurvePublicKey]:
    """
    Fetch and cache Google's AdMob SSV public keys.

    Keys are ECDSA public keys served as PEM-encoded X.509 certificates.
    """
    global _KEYS_CACHE, _KEYS_FETCHED_AT

    now = time.time()
    if _KEYS_CACHE and (now - _KEYS_FETCHED_AT) < _KEYS_TTL:
        return _KEYS_CACHE

    with _KEYS_LOCK:
        # Double-check after acquiring lock
        now = time.time()
        if _KEYS_CACHE and (now - _KEYS_FETCHED_AT) < _KEYS_TTL:
            return _KEYS_CACHE

        try:
            resp = httpx.get(KEYS_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            new_cache: Dict[str, ec.EllipticCurvePublicKey] = {}
            for key_entry in data.get("keys", []):
                key_id = str(key_entry.get("keyId", ""))
                pem = key_entry.get("pem", "")
                if not key_id or not pem:
                    continue

                try:
                    cert = load_pem_x509_certificate(pem.encode("utf-8"))
                    pub_key = cert.public_key()
                    if isinstance(pub_key, ec.EllipticCurvePublicKey):
                        new_cache[key_id] = pub_key
                    else:
                        logger.warning("Key %s is not ECDSA, skipping", key_id)
                except Exception:
                    logger.exception("Failed to parse key %s", key_id)

            _KEYS_CACHE = new_cache
            _KEYS_FETCHED_AT = now
            logger.info("Fetched %d AdMob SSV keys", len(new_cache))
            return _KEYS_CACHE

        except Exception:
            logger.exception("Failed to fetch AdMob SSV keys from %s", KEYS_URL)
            # Return stale cache if available
            return _KEYS_CACHE


def _clear_key_cache() -> None:
    """Force re-fetch on next verification (useful after key rotation)."""
    global _KEYS_CACHE, _KEYS_FETCHED_AT
    with _KEYS_LOCK:
        _KEYS_CACHE = {}
        _KEYS_FETCHED_AT = 0.0


def parse_ssv_params(query_string: str) -> Dict[str, str]:
    """
    Parse SSV callback query string into a dict.

    AdMob sends params like:
      ad_network=...&ad_unit=...&custom_data=...&reward_amount=1
      &reward_item=credit&timestamp=...&transaction_id=...
      &user_id=...&signature=...&key_id=...
    """
    params: Dict[str, str] = {}
    for pair in query_string.split("&"):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        params[key] = unquote(value)
    return params


def verify_ssv_callback(query_string: str) -> Tuple[bool, Dict[str, str]]:
    """
    Verify an AdMob SSV callback.

    The query_string is the raw query string from the callback URL.
    Google signs the message (everything before &signature=) with ECDSA/SHA-256.

    Returns:
        (is_valid, parsed_params) - is_valid is True if signature is valid.
    """
    params = parse_ssv_params(query_string)
    signature_b64 = params.get("signature", "")
    key_id = params.get("key_id", "")

    if not signature_b64 or not key_id:
        logger.warning("SSV callback missing signature or key_id")
        return False, params

    # The message to verify is the query string up to (but not including) &signature=
    sig_marker = "&signature="
    sig_idx = query_string.find(sig_marker)
    if sig_idx == -1:
        # Maybe signature is the first param (unlikely but handle it)
        sig_marker = "signature="
        sig_idx = query_string.find(sig_marker)
        if sig_idx == -1:
            logger.warning("SSV callback: cannot find signature in query string")
            return False, params

    message = query_string[:sig_idx]

    # Fetch public keys
    keys = _fetch_keys()
    pub_key = keys.get(key_id)

    if pub_key is None:
        # Key not found - try re-fetching in case of key rotation
        logger.info("Key %s not in cache, re-fetching", key_id)
        _clear_key_cache()
        keys = _fetch_keys()
        pub_key = keys.get(key_id)

        if pub_key is None:
            logger.warning("Unknown key_id after re-fetch: %s", key_id)
            return False, params

    try:
        # Decode the base64url signature
        # AdMob uses URL-safe base64 encoding
        signature_b64_padded = signature_b64 + "=" * (4 - len(signature_b64) % 4)
        signature_bytes = base64.urlsafe_b64decode(signature_b64_padded)

        # Verify ECDSA signature with SHA-256
        pub_key.verify(
            signature_bytes,
            message.encode("utf-8"),
            ec.ECDSA(SHA256()),
        )

        logger.info("SSV signature verified: key_id=%s", key_id)
        return True, params

    except Exception:
        logger.warning("SSV signature verification failed: key_id=%s", key_id)
        return False, params
