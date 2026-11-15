"""The numerical thermal SEDs use the exact Maxwell-Juettner distribution, on grids that resolve it at any Theta."""

import numpy as np
import pytest
from astropy import units as u

from trilobite.physics_utils.maxwell_juettner import _opt_mj_log_pdf, compute_mj_mean_kinetic_energy
from trilobite.radiation.synchrotron.electron_distributions import MaxwellJuettner
from trilobite.radiation.synchrotron.SEDs import (
    Numerical_Thermal_PowerLaw_SSA_SynchrotronSED,
    Numerical_Thermal_SSA_SynchrotronSED,
)
from trilobite.radiation.synchrotron.SEDs.one_zone.seds import _thermal_gamma_grid

NU = np.logspace(7, 18, 60) * u.Hz
SED_KWARGS = {"epsilon_B": 0.1, "epsilon_T": 0.1, "luminosity_distance": 100 * u.Mpc}
B, R = 1.0 * u.G, 1e16 * u.cm
FLUX_UNIT = u.erg / u.cm**2 / u.s / u.Hz

# Dense reference grid: log-spaced in gamma and in gamma - 1. Doubling it changes the SEDs by ~1e-5.
DENSE = np.unique(np.concatenate([np.geomspace(1.0, 1e8, 3000), 1.0 + np.geomspace(1e-9, 1e8, 3000)]))


@pytest.fixture(scope="module")
def thermal():
    """Pure thermal SED on its default grid."""
    return Numerical_Thermal_SSA_SynchrotronSED()


@pytest.fixture(scope="module")
def thermal_dense():
    """Pure thermal SED on a dense reference grid."""
    return Numerical_Thermal_SSA_SynchrotronSED(gamma=DENSE)


@pytest.fixture(scope="module")
def mixed():
    """Mixed SED on its default grid."""
    return Numerical_Thermal_PowerLaw_SSA_SynchrotronSED()


@pytest.fixture(scope="module")
def mixed_dense():
    """Mixed SED on a dense reference grid."""
    return Numerical_Thermal_PowerLaw_SSA_SynchrotronSED(gamma=DENSE)


def _flux(sed, Theta, **extra):
    return sed.sed(NU, B, R, Theta, **{**SED_KWARGS, **extra}).to_value(FLUX_UNIT)


def _max_relative_error(flux, reference):
    significant = reference > 1e-3 * reference.max()  # the exponential tails are noise-limited
    return np.max(np.abs(flux[significant] / reference[significant] - 1.0))


class TestDistribution:
    """The thermal electron distribution handed to the engine."""

    def test_exact_shape_reduces_to_the_ultra_relativistic_formula(self):
        # For Theta >> 1, beta -> 1 and Theta K_2(1/Theta) -> 2 Theta^3, so f -> gamma^2 e^(-gamma/Theta) / (2 Theta^3).
        Theta = 300.0
        gamma = np.geomspace(Theta, 10.0 * Theta, 30)
        exact = MaxwellJuettner.pdf(gamma, 1.0, Theta=Theta)
        np.testing.assert_allclose(exact, gamma**2 * np.exp(-gamma / Theta) / (2.0 * Theta**3), rtol=1e-2)

    @pytest.mark.parametrize("Theta", [0.03, 0.3, 1.0])
    def test_ultra_relativistic_shape_undercounts_electrons_at_low_theta(self, Theta):
        # The shape the thermal SEDs used for every Theta: it holds 0, 0.35 and 0.92 of the electrons at these Theta.
        gamma = np.geomspace(1.0, 1.0 + 60.0 * Theta, 4000)
        old_shape = gamma**2 * np.exp(-gamma / Theta) / (2.0 * Theta**3)
        assert np.sum(old_shape * np.gradient(gamma)) < 0.95

    @pytest.mark.parametrize("Theta", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0])
    @pytest.mark.parametrize("fixture", ["thermal", "mixed"])
    def test_default_grid_carries_the_number_and_energy_budget(self, request, fixture, Theta):
        sed = request.getfixturevalue(fixture)
        log_gamma, weights = sed._log_gamma, np.exp(sed._log_weights)
        gamma = np.exp(log_gamma)
        N = np.exp(_opt_mj_log_pdf(gamma, Theta))
        assert np.sum(N * weights) == pytest.approx(1.0, abs=1e-2)
        assert np.sum(N * (gamma - 1.0) * weights) == pytest.approx(compute_mj_mean_kinetic_energy(Theta), rel=1e-2)


class TestGrids:
    """The default Lorentz-factor grids."""

    def test_thermal_grid_is_log_spaced_in_gamma_minus_one(self):
        grid = _thermal_gamma_grid(1.0, 1e8, 200)
        assert len(grid) == 200
        assert grid[0] - 1.0 == pytest.approx(1e-6)
        assert grid[-1] == pytest.approx(1e8)
        ratios = np.diff(np.log(grid - 1.0))
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-6)

    def test_thermal_grid_respects_a_larger_gamma_min(self):
        assert _thermal_gamma_grid(5.0, 1e4, 50)[0] == pytest.approx(5.0)

    def test_mixed_grid_contains_the_log_gamma_grid(self):
        grid = _thermal_gamma_grid(1.0, 1e8, 200, mixed=True)
        assert np.all(np.diff(grid) > 0.0)
        assert len(grid) == 300
        assert np.all(np.isin(np.geomspace(1.0, 1e8, 200), grid))

    def test_mixed_grid_with_small_gamma_max(self):
        grid = _thermal_gamma_grid(1.0, 50.0, 200, mixed=True)
        assert grid[-1] == pytest.approx(50.0)
        assert np.all(np.diff(grid) > 0.0)

    def test_power_law_gamma_min_above_one_leaves_the_thermal_grid_intact(self, mixed, mixed_dense):
        """The power law's ``gamma_min`` (given at evaluation) does not cut the thermal electrons below it."""
        assert np.exp(mixed._log_gamma).min() == pytest.approx(1.0)
        nu = np.geomspace(1e8, 1e12, 5)
        kwargs = {"p": 3.0, "epsilon_B": 0.1, "epsilon_T": 1.0, "epsilon_E": 0.1, "gamma_min": 3.0, "gamma_max": 1e6}
        a = np.exp(mixed._log_opt_sed(nu, 1.0, 1e16, 1e27, 0.3, **kwargs))
        b = np.exp(mixed_dense._log_opt_sed(nu, 1.0, 1e16, 1e27, 0.3, **kwargs))
        np.testing.assert_allclose(a, b, rtol=5e-3)

    def test_explicit_grid_is_used_unchanged(self):
        gamma = np.geomspace(1.0 + 1e-3, 1e3, 77)
        sed = Numerical_Thermal_SSA_SynchrotronSED(gamma=gamma)
        np.testing.assert_allclose(np.exp(sed._log_gamma), gamma, rtol=1e-12)


class TestSpectra:
    """The SEDs themselves."""

    @pytest.mark.parametrize("Theta", [0.05, 0.5, 5.0, 30.0])
    def test_thermal_sed_converges_to_a_dense_grid(self, thermal, thermal_dense, Theta):
        error = _max_relative_error(_flux(thermal, Theta), _flux(thermal_dense, Theta))
        assert error < 1e-2

    @pytest.mark.parametrize("Theta", [0.1, 3.0])
    @pytest.mark.parametrize(("epsilon_T", "epsilon_E"), [(0.0, 0.1), (0.1, 0.1), (0.1, 0.0)])
    def test_mixed_sed_converges_to_a_dense_grid(self, mixed, mixed_dense, Theta, epsilon_T, epsilon_E):
        extra = {"epsilon_T": epsilon_T, "epsilon_E": epsilon_E, "p": 3.0, "gamma_min": 3.0, "gamma_max": 1e6}
        error = _max_relative_error(_flux(mixed, Theta, **extra), _flux(mixed_dense, Theta, **extra))
        assert error < 1e-2

    @pytest.mark.parametrize("Theta", [0.1, 1.0, 10.0])
    def test_mixed_sed_without_non_thermal_electrons_is_the_thermal_sed(self, Theta):
        # Both integrate the same thermal distribution; the mixed SED only adds nodes at the edges of its (empty)
        # power law, so they agree to the accuracy of the quadrature.
        gamma = np.geomspace(1.0 + 1e-6, 1e8, 150)
        thermal = Numerical_Thermal_SSA_SynchrotronSED(gamma=gamma)
        mixed = Numerical_Thermal_PowerLaw_SSA_SynchrotronSED(gamma=gamma)
        flux_mixed, flux_thermal = _flux(mixed, Theta, epsilon_E=0.0, p=3.0), _flux(thermal, Theta)
        assert _max_relative_error(flux_mixed, flux_thermal) < 5e-3
