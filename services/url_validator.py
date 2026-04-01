# services/url_validator.py
"""Shared SSRF-safe URL validation for outbound HTTP fetches."""
from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _is_private_ip(addr: str) -> bool:
    """Return True if *addr* is a private, loopback, link-local,
    multicast, reserved, or unspecified IP address."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _is_private_host(host: str) -> bool:
    """Return True if the hostname itself (before DNS resolution) looks
    internal/private."""
    if not host:
        return True

    h = host.strip().lower()
    if h in {"localhost"}:
        return True

    # Direct IP literal check
    if _is_private_ip(h):
        return True

    # Common internal TLDs
    if h.endswith(".local") or h.endswith(".localhost"):
        return True

    return False


def validate_url_for_fetch(url: str) -> None:
    """Validate that *url* is safe for server-side fetching.

    Raises ``ValueError`` if the URL uses a non-HTTP(S) scheme, points
    to a private/internal host, or resolves to a private IP address.
    """
    parsed = urlparse(url)

    # --- scheme check ---
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Only http/https URLs are allowed, got '{parsed.scheme}'")

    hostname = parsed.hostname or ""

    # --- static hostname check ---
    if _is_private_host(hostname):
        raise ValueError(f"Disallowed URL host: {hostname}")

    # --- DNS resolution check ---
    # Even if the hostname looks fine, the resolved IP might be internal
    # (e.g. DNS rebinding, attacker-controlled DNS).
    try:
        resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve hostname: {hostname}") from exc

    for family, _type, _proto, _canonname, sockaddr in resolved:
        ip_str = sockaddr[0]
        if _is_private_ip(ip_str):
            raise ValueError(
                f"Hostname '{hostname}' resolves to private IP {ip_str}"
            )
