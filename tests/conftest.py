import pytest
from fastapi.testclient import TestClient


# Note: We can't easily test with real Firebase in unit tests.
# These tests focus on schema validation, utility functions, and API contract.


@pytest.fixture
def client():
    """Test client - import app lazily to avoid Firebase init issues."""
    # For unit tests, we test utilities directly.
    pass
