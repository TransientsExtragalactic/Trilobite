"""
Tests for synchrotron utility functions.

These tests focus on the ``trilobite.radiation.synchrotron.utils`` module, which largely contains constants
relevant to the synchrotron theory.
"""

import numpy as np

# ======================================================== #
# Constant Testing                                         #
# ======================================================== #
# These tests are used to ensure that the values of relevant constants are correct.


def test_c5():
    """
    Test that the Pacholczyk constant c5 is correct as implemented.

    We use the known value for p = 3 of 7.52e-24 in cgs units against the value computed by the function.
    """
    from trilobite.radiation.synchrotron.utils import compute_c5_parameter

    # Compute the c5 parameter for p = 3
    p = 3.0
    c5_computed = compute_c5_parameter(p)
    c5_expected = 7.52e-24

    # Compare the computed value to the expected value
    assert np.isclose(c5_computed, c5_expected, rtol=1e-3), f"Computed c5: {c5_computed}, Expected c5: {c5_expected}"


def test_c6():
    """
    Test that the Pacholczyk constant c6 is correct as implemented.

    We use the known value for p = 3 of 7.96e-41 in cgs units against the value computed by the function.
    """
    from trilobite.radiation.synchrotron.utils import compute_c6_parameter

    # Compute the c6 parameter for p = 3
    p = 3.0
    c6_computed = compute_c6_parameter(p)
    c6_expected = 7.96e-41

    # Compare the computed value to the expected value
    assert np.isclose(c6_computed, c6_expected, rtol=1e-3), f"Computed c6: {c6_computed}, Expected c6: {c6_expected}"
