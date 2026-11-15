"""The numerical SEDs do not depend on where the edges of the power law fall between the nodes of their grid."""

import astropy.units as u
import numpy as np
import pytest

from trilobite.radiation.synchrotron.SEDs import (
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
