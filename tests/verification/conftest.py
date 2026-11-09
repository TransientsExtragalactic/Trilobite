"""Fixtures for the verification suite."""

import pytest
from verification.compare import Verifier


@pytest.fixture
def verify(request):
    """Verifier: Assertion helper that records each comparison for the verification report."""
    return Verifier(request.node)
