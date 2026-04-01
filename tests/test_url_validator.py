# tests/test_url_validator.py
"""Tests for SSRF-safe URL validation (services/url_validator.py)."""
from __future__ import annotations

import pytest

from services.url_validator import _is_private_ip, _is_private_host, validate_url_for_fetch


# ---------------------------------------------------------------------------
# _is_private_ip
# ---------------------------------------------------------------------------
class TestIsPrivateIp:
    def test_loopback_ipv4(self):
        assert _is_private_ip("127.0.0.1") is True

    def test_loopback_ipv6(self):
        assert _is_private_ip("::1") is True

    def test_private_10(self):
        assert _is_private_ip("10.0.0.1") is True

    def test_private_172(self):
        assert _is_private_ip("172.16.0.1") is True

    def test_private_192(self):
        assert _is_private_ip("192.168.1.1") is True

    def test_link_local(self):
        assert _is_private_ip("169.254.1.1") is True

    def test_multicast(self):
        assert _is_private_ip("224.0.0.1") is True

    def test_unspecified(self):
        assert _is_private_ip("0.0.0.0") is True

    def test_public_ip(self):
        assert _is_private_ip("8.8.8.8") is False

    def test_public_ip_2(self):
        assert _is_private_ip("1.1.1.1") is False

    def test_invalid_returns_false(self):
        assert _is_private_ip("not-an-ip") is False


# ---------------------------------------------------------------------------
# _is_private_host
# ---------------------------------------------------------------------------
class TestIsPrivateHost:
    def test_localhost(self):
        assert _is_private_host("localhost") is True

    def test_empty_string(self):
        assert _is_private_host("") is True

    def test_dot_local(self):
        assert _is_private_host("myserver.local") is True

    def test_dot_localhost(self):
        assert _is_private_host("app.localhost") is True

    def test_ip_literal_private(self):
        assert _is_private_host("127.0.0.1") is True

    def test_public_domain(self):
        assert _is_private_host("example.com") is False

    def test_public_ip(self):
        assert _is_private_host("8.8.8.8") is False


# ---------------------------------------------------------------------------
# validate_url_for_fetch
# ---------------------------------------------------------------------------
class TestValidateUrlForFetch:
    def test_valid_https(self):
        """Public HTTPS URL should pass without raising."""
        validate_url_for_fetch("https://example.com/image.jpg")

    def test_valid_http(self):
        validate_url_for_fetch("http://example.com/path")

    def test_file_scheme_rejected(self):
        with pytest.raises(ValueError, match="Only http/https"):
            validate_url_for_fetch("file:///etc/passwd")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="Only http/https"):
            validate_url_for_fetch("ftp://example.com/file")

    def test_data_scheme_rejected(self):
        with pytest.raises(ValueError, match="Only http/https"):
            validate_url_for_fetch("data:text/html,<h1>hi</h1>")

    def test_localhost_rejected(self):
        with pytest.raises(ValueError, match="Disallowed URL host"):
            validate_url_for_fetch("http://localhost/admin")

    def test_localhost_with_port_rejected(self):
        with pytest.raises(ValueError, match="Disallowed URL host"):
            validate_url_for_fetch("http://localhost:8080/api")

    def test_private_ip_127(self):
        with pytest.raises(ValueError, match="Disallowed URL host"):
            validate_url_for_fetch("http://127.0.0.1/secret")

    def test_private_ip_10(self):
        with pytest.raises(ValueError, match="Disallowed URL host"):
            validate_url_for_fetch("http://10.0.0.1/internal")

    def test_private_ip_192(self):
        with pytest.raises(ValueError, match="Disallowed URL host"):
            validate_url_for_fetch("http://192.168.1.1/router")

    def test_link_local_169(self):
        with pytest.raises(ValueError, match="Disallowed URL host"):
            validate_url_for_fetch("http://169.254.169.254/metadata")

    def test_dot_local_rejected(self):
        with pytest.raises(ValueError, match="Disallowed URL host"):
            validate_url_for_fetch("http://myserver.local/api")
