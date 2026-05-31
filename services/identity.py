# services/identity.py
"""
Identity hashing for the re-signup abuse defense.

Stores SHA-256 hashes (not plaintext) of normalized email + provider UIDs in
`deleted_users/{hash}` so a deleted account's identifiers can be matched on
re-signup without persisting raw PII.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List


def normalize_email(email: str | None) -> str:
    """
    Lowercase + Gmail-aware normalization.

    Gmail (and googlemail.com) ignore dots and `+tag` in the local part, so
    `Foo.Bar+anything@gmail.com` and `foobar@gmail.com` are the same mailbox.
    Other providers vary, so we only strip the +tag suffix conservatively.
    """
    if not email:
        return ""
    s = email.strip().lower()
    if "@" not in s:
        return s
    local, _, domain = s.partition("@")
    # Strip +tag suffix everywhere (common convention)
    if "+" in local:
        local = local.split("+", 1)[0]
    # Remove dots only for Gmail-family domains
    if domain in ("gmail.com", "googlemail.com"):
        local = local.replace(".", "")
        domain = "gmail.com"
    return f"{local}@{domain}"


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_email(email: str | None) -> str | None:
    """Hash a normalized email. Returns None for empty input."""
    norm = normalize_email(email)
    if not norm:
        return None
    return _sha256(f"email:{norm}")


def hash_provider_uid(provider_id: str, uid: str) -> str | None:
    """Hash a provider-specific UID (e.g. google.com sub, apple.com sub)."""
    if not provider_id or not uid:
        return None
    return _sha256(f"{provider_id}:{uid}")


def identity_hashes_from_user(user) -> List[str]:
    """
    Build the full set of identity hashes for a Firebase Admin UserRecord.

    Includes:
      - Normalized email hash (top-level user.email)
      - For each providerData entry: hash(provider_id:uid)
      - For each providerData entry with email: hash of its normalized email
        (catches Apple "Hide My Email" relay addresses linked to other logins)
    """
    hashes: List[str] = []

    seen: set[str] = set()

    def _add(h: str | None) -> None:
        if h and h not in seen:
            seen.add(h)
            hashes.append(h)

    email = getattr(user, "email", None)
    _add(hash_email(email))

    provider_data: Iterable = getattr(user, "provider_data", []) or []
    for p in provider_data:
        pid = getattr(p, "provider_id", None) or ""
        puid = getattr(p, "uid", None) or ""
        _add(hash_provider_uid(pid, puid))
        p_email = getattr(p, "email", None)
        _add(hash_email(p_email))

    return hashes
