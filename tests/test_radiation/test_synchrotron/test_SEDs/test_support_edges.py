"""The numerical SEDs do not depend on where the edges of the power law fall between the nodes of their grid."""

import astropy.units as u
import numpy as np
import pytest

from trilobite.radiation.synchrotron.electron_distributions import BrokenPowerLaw, PowerLaw
from trilobite.radiation.synchrotron.SEDs import (
    Numerical_PowerLaw_Cooling_SSA_SynchrotronSED,
    Numerical_PowerLaw_SSA_SynchrotronSED,
    Numerical_Thermal_PowerLaw_SSA_SynchrotronSED,
)
from trilobite.radiation.synchrotron.SEDs.one_zone.seds import _step_grid, _support_grid

NU = np.logspace(7, 13, 60) * u.Hz
FLUX_UNIT = u.erg / u.cm**2 / u.s / u.Hz
COMMON = {"B": 1.0 * u.G, "R": 1e16 * u.cm, "epsilon_B": 0.1, "luminosity_distance": 100 * u.Mpc}
P = 3.0


def _max_relative_error(flux, reference):
    significant = reference > 1e-3 * reference.max()
    return np.max(np.abs(flux[significant] / reference[significant] - 1.0))


def _offsets(sed, index=60):
    """Return the Lorentz factors of a grid node and of three points between it and the next node."""
    gamma = np.exp(sed._log_gamma)
    return [gamma[index] * (gamma[index + 1] / gamma[index]) ** fraction for fraction in (0.0, 0.25, 0.5, 0.75)]


class TestQuadratureWeights:
    """The grids integrate a truncated power law to the same accuracy for any position of its edges."""

    @pytest.mark.parametrize("fraction", [0.0, 0.1, 0.5, 0.9])
    def test_support_grid_integrates_a_power_law(self, fraction):
        log_gamma = np.linspace(0.0, np.log(1e8), 200)
        spacing = log_gamma[1] - log_gamma[0]
        lower = log_gamma[60] + fraction * spacing
        upper = log_gamma[150] + fraction * spacing
        nodes, log_weights = _support_grid(log_gamma, lower, upper)
        got = np.sum(np.exp(-P * nodes + log_weights))  # the weights are gamma dln(gamma), so this is sum(gamma^-P w)
        exact = (np.exp(-(P - 1.0) * lower) - np.exp(-(P - 1.0) * upper)) / (P - 1.0)
        assert got == pytest.approx(exact, rel=1e-2)  # smooth O(spacing^2) error; edge rounding gives 5% to 9%

    @pytest.mark.parametrize("fraction", [0.0, 0.1, 0.5, 0.9])
    def test_step_grid_integrates_a_truncated_power_law(self, fraction):
        log_gamma = np.linspace(0.0, np.log(1e8), 200)
        spacing = log_gamma[1] - log_gamma[0]
        lower = log_gamma[60] + fraction * spacing
        upper = log_gamma[150] + fraction * spacing
        nodes, log_weights = _step_grid(log_gamma, lower, upper)
        inside = (nodes >= lower) & (nodes <= upper)
        got = np.sum(np.where(inside, np.exp(-P * nodes + log_weights), 0.0))
        exact = (np.exp(-(P - 1.0) * lower) - np.exp(-(P - 1.0) * upper)) / (P - 1.0)
        assert got == pytest.approx(exact, rel=1e-2)  # smooth O(spacing^2) error; edge rounding gives 5% to 9%

    def test_support_beyond_the_grid_is_truncated_at_the_grid(self):
        log_gamma = np.linspace(0.0, np.log(1e4), 100)
        nodes, _ = _support_grid(log_gamma, -1.0, np.log(1e9))
        assert nodes[0] == log_gamma[0]
        assert nodes[-1] == log_gamma[-1]

    def test_support_outside_the_grid_raises(self):
        with pytest.raises(ValueError, match="does not overlap"):
            _support_grid(np.linspace(0.0, 5.0, 50), 6.0, 7.0)

    def test_step_grid_keeps_the_grid_when_the_edges_are_outside_it(self):
        log_gamma = np.linspace(0.0, 5.0, 50)
        nodes, _ = _step_grid(log_gamma, -1.0, 9.0)
        np.testing.assert_array_equal(nodes, log_gamma)


class TestSpectra:
    """The SEDs agree with a dense reference grid whatever the position of gamma_min and gamma_max on the grid."""

    def test_power_law_sed_does_not_depend_on_gamma_min_alignment(self):
        sed = Numerical_PowerLaw_SSA_SynchrotronSED()
        for gamma_min in _offsets(sed):
            kwargs = {**COMMON, "p": P, "epsilon_E": 0.1, "gamma_min": gamma_min, "gamma_max": 1e6}
            reference = Numerical_PowerLaw_SSA_SynchrotronSED(gamma=np.geomspace(gamma_min, 1e6, 20001))
            error = _max_relative_error(
                sed.sed(NU, **kwargs).to_value(FLUX_UNIT), reference.sed(NU, **kwargs).to_value(FLUX_UNIT)
            )
            assert error < 1e-2  # a grid-node-rounded edge gave 6% to 11% here

    def test_power_law_sed_does_not_depend_on_gamma_max_alignment(self):
        sed = Numerical_PowerLaw_SSA_SynchrotronSED()
        for gamma_max in _offsets(sed, index=150):
            kwargs = {**COMMON, "p": 1.8, "epsilon_E": 0.1, "gamma_min": 3.0, "gamma_max": gamma_max}
            reference = Numerical_PowerLaw_SSA_SynchrotronSED(gamma=np.geomspace(3.0, gamma_max, 20001))
            error = _max_relative_error(
                sed.sed(NU, **kwargs).to_value(FLUX_UNIT), reference.sed(NU, **kwargs).to_value(FLUX_UNIT)
            )
            assert error < 1e-2

    def test_mixed_sed_does_not_depend_on_gamma_min_alignment(self):
        sed = Numerical_Thermal_PowerLaw_SSA_SynchrotronSED()
        dense = np.unique(np.concatenate([np.geomspace(1.0, 1e8, 6000), 1.0 + np.geomspace(1e-9, 1e8, 6000)]))
        reference = Numerical_Thermal_PowerLaw_SSA_SynchrotronSED(gamma=dense)
        for gamma_min in _offsets(sed, index=150):
            kwargs = {**COMMON, "Theta": 1.0, "p": P, "epsilon_T": 0.05, "epsilon_E": 0.05, "gamma_min": gamma_min}
            kwargs["gamma_max"] = 1e6
            error = _max_relative_error(
                sed.sed(NU, **kwargs).to_value(FLUX_UNIT), reference.sed(NU, **kwargs).to_value(FLUX_UNIT)
            )
            assert error < 1e-2


B_CGS, R_CGS, D_A_CGS = 1.0, 1e16, 3e26
EPSILON_B, EPSILON_E, GAMMA_MAX = 0.1, 0.1, 1e6
# Times that put the cooling Lorentz factor (1.2e6, 1.2e3, 1.2 for B = 1 G) above gamma_max, between gamma_min and
# gamma_max, and below gamma_min = 10.
REGIME_TIMES = {"uncooled": 1e3, "slow": 1e6, "fast": 1e9}


@pytest.fixture
def cooling_distribution(monkeypatch):
    """Return a function giving the grid and electron distribution the cooling SED hands to the engine."""
    sed = Numerical_PowerLaw_Cooling_SSA_SynchrotronSED()
    captured = {}

    def spy(**kwargs):
        captured.update(kwargs)
        return np.zeros_like(kwargs["log_nu"])

    monkeypatch.setattr(sed._engine, "_compute_log_pa_flux_density", spy)

    def distribution(regime, p, gamma_min=10.0):
        sed._log_opt_sed(
            np.array([1e9]),
            REGIME_TIMES[regime],
            B_CGS,
            R_CGS,
            D_A_CGS,
            p=p,
            epsilon_B=EPSILON_B,
            epsilon_E=EPSILON_E,
            gamma_min=gamma_min,
            gamma_max=GAMMA_MAX,
        )
        gamma_c = float(sed._cooling_engine.compute_cooling_gamma(B=B_CGS, t=REGIME_TIMES[regime], sin_alpha=None))
        return (
            np.exp(captured["log_gamma"]),
            np.exp(captured["log_weights"]),
            np.exp(np.atleast_1d(captured["log_N"])),
            gamma_c,
        )

    return distribution


def _class_distribution(regime, p, gamma_min, gamma_c):
    """Return the distribution class and parameters of a cooling regime."""
    if regime == "fast":
        return BrokenPowerLaw, {"p1": 2.0, "p2": p + 1.0, "gamma_min": gamma_c, "gamma_c": gamma_min}
    if regime == "slow":
        return BrokenPowerLaw, {"p1": p, "p2": p + 1.0, "gamma_min": gamma_min, "gamma_c": gamma_c}
    return PowerLaw, {"p": p, "gamma_min": gamma_min}


@pytest.mark.parametrize("p", [2.5, 3.0, 3.5])
@pytest.mark.parametrize("regime", ["uncooled", "slow", "fast"])
class TestCoolingDistribution:
    """The cooling SED integrates the broken power law of its regime, and nothing outside its support."""

    def test_distribution_is_the_class_distribution(self, cooling_distribution, regime, p):
        gamma, _, N, gamma_c = cooling_distribution(regime, p)
        distribution, params = _class_distribution(regime, p, 10.0, gamma_c)
        params["gamma_max"] = GAMMA_MAX
        N0 = distribution.normalize_from_magnetic_field(B_CGS, EPSILON_B, EPSILON_E, **params).value
        inner = slice(1, -1)  # the end nodes sit on the support edges, where rounding decides in or out
        np.testing.assert_allclose(N[inner], distribution.pdf(gamma[inner], N0, **params), rtol=1e-10)

    def test_electrons_are_only_inside_the_support(self, cooling_distribution, regime, p):
        gamma, _, _, gamma_c = cooling_distribution(regime, p)
        lower = gamma_c if regime == "fast" else 10.0
        assert gamma[0] == pytest.approx(lower, rel=1e-12)
        assert gamma[-1] == pytest.approx(GAMMA_MAX, rel=1e-12)

    def test_energy_budget_is_the_kinetic_energy_budget(self, cooling_distribution, regime, p):
        from trilobite.radiation.constants import electron_rest_energy_cgs as mc2

        gamma, weights, N, _ = cooling_distribution(regime, p)
        energy = mc2 * np.sum(N * (gamma - 1.0) * weights)
        assert energy == pytest.approx(EPSILON_E / EPSILON_B * B_CGS**2 / (8.0 * np.pi), rel=1e-2)


def test_cooling_sed_does_not_depend_on_gamma_min_alignment():
    """The slow-cooling SED agrees with a dense reference wherever gamma_min falls on the grid."""
    sed = Numerical_PowerLaw_Cooling_SSA_SynchrotronSED()
    for gamma_min in _offsets(sed, index=60):
        kwargs = {**COMMON, "t": 1e6 * u.s, "p": P, "epsilon_E": 0.1, "gamma_min": gamma_min, "gamma_max": 1e6}
        reference = Numerical_PowerLaw_Cooling_SSA_SynchrotronSED(gamma=np.geomspace(gamma_min, 1e6, 20001))
        error = _max_relative_error(
            sed.sed(NU, **kwargs).to_value(FLUX_UNIT), reference.sed(NU, **kwargs).to_value(FLUX_UNIT)
        )
        assert error < 1e-2
