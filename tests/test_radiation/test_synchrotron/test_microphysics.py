import numpy as np
import pytest
from scipy.integrate import quad

from triceratops.radiation.synchrotron.microphysics import (
    _opt_compute_BPL_moment,
    _opt_compute_BPL_n_total,
    _opt_compute_PL_moment,
    _opt_compute_PL_n_total,
    _opt_normalize_BPL_from_magnetic_field,
    _opt_normalize_PL_from_magnetic_field,
    electron_rest_energy_cgs,
)


# ======================================================== #
# Helper Functions
# ======================================================== #
def numeric_PL_moment(p, gamma_min, gamma_max, k):
    """
    Numerically compute the k-th moment of a power-law distribution
    using adaptive quadrature.

    Computes

        ∫_{γ_min}^{γ_max} γ^k γ^{-p} dγ
    """

    def integrand(gamma):
        return gamma ** (k - p)

    result, _ = quad(
        integrand,
        gamma_min,
        gamma_max,
        epsabs=0,
        epsrel=1e-10,
        limit=200,
    )

    return result


def numeric_BPL_moment(a1, a2, x_min, x_max, order):
    """
    Numerically compute the moment of a broken power-law distribution

        ∫ x^order N(x) dx

    where

        N(x) ∝ x^{a1}   for x < 1
        N(x) ∝ x^{a2}   for x > 1
    """

    def integrand_low(x):
        return x ** (order + a1)

    def integrand_high(x):
        return x ** (order + a2)

    result = 0.0

    if x_min < 1:
        upper = min(1.0, x_max)
        result += quad(integrand_low, x_min, upper, epsrel=1e-10)[0]

    if x_max > 1:
        lower = max(1.0, x_min)
        result += quad(integrand_high, lower, x_max, epsrel=1e-10)[0]

    return result


# ======================================================== #
# Check moment calculations
# ======================================================= #
@pytest.mark.parametrize(
    "p,gamma_min,gamma_max,order",
    [
        (3.0, 1.0, 1e6, 0),
        (3.0, 1.0, 1e6, 1),
        (3.0, 1.0, 1e6, 2),
        (2.5, 10.0, 1e5, 0),
        (2.5, 10.0, 1e5, 1),
        (2.5, 10.0, 1e5, 2),
        (2.0, 100.0, 1e8, 0),
        (2.0, 100.0, 1e8, 1),
        (2.0, 100.0, 1e8, 2),
    ],
)
def test_PL_moment_matches_quadrature(p, gamma_min, gamma_max, order):
    """
    Verify that the analytic power-law moment matches numerical quadrature.

    This checks the optimized analytic implementation of

        ∫ γ^k N(γ) dγ

    against direct numerical integration.
    """
    # Analytic moment
    analytic = _opt_compute_PL_moment(
        p,
        gamma_min,
        gamma_max,
        order=order,
    )

    # Numerical quadrature
    numeric = numeric_PL_moment(
        p,
        gamma_min,
        gamma_max,
        order,
    )

    assert np.isclose(analytic, numeric, rtol=1e-3)


@pytest.mark.parametrize(
    "a1,a2,x_min,x_max,order",
    [
        (-2.0, -3.0, 1e-3, 1e3, 0),
        (-2.0, -3.0, 1e-3, 1e3, 1),
        (-2.0, -3.0, 1e-3, 1e3, 2),
        (-1.5, -2.5, 1e-2, 1e2, 0),
        (-1.5, -2.5, 1e-2, 1e2, 1),
        (-1.5, -2.5, 1e-2, 1e2, 2),
        (-2.5, -3.5, 1e-1, 1e4, 0),
        (-2.5, -3.5, 1e-1, 1e4, 1),
        (-2.5, -3.5, 1e-1, 1e4, 2),
    ],
)
def test_BPL_moment_matches_quadrature(a1, a2, x_min, x_max, order):
    """
    Verify that the analytic BPL moment matches numerical quadrature.
    """

    analytic = _opt_compute_BPL_moment(
        a1,
        a2,
        x_min,
        x_max,
        order=order,
    )

    numeric = numeric_BPL_moment(
        a1,
        a2,
        x_min,
        x_max,
        order,
    )

    assert np.isclose(analytic, numeric, rtol=1e-3)


@pytest.mark.parametrize(
    "a1,a2,gamma_b",
    [
        (-2.5, -3.5, 1e3),
        (-2.0, -3.0, 1e4),
        (-3.0, -4.0, 1e5),
    ],
)
def test_PL_and_BPL_normalizations_produce_same_energy_density(a1, a2, gamma_b):
    B = 1.0
    epsilon_E = 0.1
    epsilon_B = 0.1

    gamma_min = 1.0
    gamma_max = 1e6
    p = 3.0

    # Target magnetic energy density
    u_B = B**2 / (8 * np.pi)
    u_target = (epsilon_E / epsilon_B) * u_B

    # -------------------------------------------------
    # PL normalization
    # -------------------------------------------------

    N0_pl = _opt_normalize_PL_from_magnetic_field(
        B,
        p,
        epsilon_B,
        epsilon_E,
        gamma_min,
        gamma_max,
    )

    moment_pl = _opt_compute_PL_moment(
        p,
        gamma_min,
        gamma_max,
        order=1,
    )

    u_e_pl = electron_rest_energy_cgs * N0_pl * moment_pl

    # -------------------------------------------------
    # BPL normalization
    # -------------------------------------------------

    N0_bpl = _opt_normalize_BPL_from_magnetic_field(
        B,
        a1,
        a2,
        gamma_b,
        epsilon_B,
        epsilon_E,
        gamma_min,
        gamma_max,
    )

    moment_bpl = _opt_compute_BPL_moment(
        a1,
        a2,
        gamma_min / gamma_b,
        gamma_max / gamma_b,
        order=1,
    )

    u_e_bpl = electron_rest_energy_cgs * N0_bpl * gamma_b**2 * moment_bpl

    # -------------------------------------------------
    # Assertions
    # -------------------------------------------------

    assert np.isclose(u_e_pl, u_target, rtol=1e-10)
    assert np.isclose(u_e_bpl, u_target, rtol=1e-10)
    assert np.isclose(u_e_pl, u_e_bpl, rtol=1e-10)
