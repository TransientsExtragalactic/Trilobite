"""Shared pytest fixtures for the synchrotron test suite."""

import pytest


@pytest.fixture
def typical_PL_params():
    """Typical astrophysical power-law parameters."""
    return {"p": 3.0, "gamma_min": 10.0, "gamma_max": 1e6}


@pytest.fixture
def typical_BPL_params():
    """Typical astrophysical broken power-law parameters."""
    return {"a1": -2.5, "a2": -3.5, "gamma_b": 1e3, "gamma_min": 10.0, "gamma_max": 1e6}


@pytest.fixture
def typical_MJD_params():
    """Typical dimensionless temperature for the Maxwell-Jüttner distribution."""
    return {"Theta": 10.0}
