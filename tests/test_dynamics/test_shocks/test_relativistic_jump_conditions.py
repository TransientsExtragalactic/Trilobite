"""Tests for the post-shock thermal energy of :mod:`trilobite.dynamics.shocks.core.relativistic_jump_conditions`."""

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from trilobite.dynamics.shocks.core import (
    RelativisticColdShockConditions,
    StrongColdShockConditions,
    UltraRelativisticColdShockConditions,
)
from trilobite.physics_utils import compute_ion_electron_theta

RHO = 1e-24 * u.g / u.cm**3
C = const.c


def _xi(result):
    """Thermal energy density in units of the rest-mass energy density, u_th / (rho_2 c^2)."""
    return (result.post_shock_thermal_energy_density / (result.post_shock_density * C**2)).decompose().value


def _relative_lorentz_factor(beta_shock, result):
    # Static upstream, so the upstream speed in the shock frame is beta_shock.
    # Gamma_rel = Gamma_1 Gamma_2 (1 - beta_1 beta_2).
    gamma1 = 1.0 / np.sqrt(1.0 - beta_shock**2)
    return gamma1 * result.post_shock_lorentz_factor * (1.0 - beta_shock * result.post_shock_beta)


class TestThermalEnergyDensity:
    """The reported post-shock thermal energy density."""

    @pytest.mark.parametrize("gamma_hat", [4.0 / 3.0, 1.5, 5.0 / 3.0])
    @pytest.mark.parametrize("beta", [0.05, 0.3, 0.7, 0.99])
    def test_cold_shock_energy_per_rest_mass_is_the_relative_lorentz_factor_minus_one(self, beta, gamma_hat):
        # Energy-flux conservation across a shock into cold gas gives
        # u_th / (rho_2 c^2) = Gamma_rel - 1, for any gamma_hat.
        result = RelativisticColdShockConditions.solve(beta * C, RHO, gamma_hat=gamma_hat)
        assert _xi(result) == pytest.approx(_relative_lorentz_factor(beta, result) - 1.0, rel=1e-8)

    @pytest.mark.parametrize("beta", [0.9999, 0.99999, 0.999999])
    def test_ultra_relativistic_solver_converges_at_order_one_over_gamma(self, beta):
        # The analytic solver fixes beta_2 = gamma_hat - 1, which is exact only as Gamma_1 -> infinity.
        gamma1 = 1.0 / np.sqrt(1.0 - beta**2)
        result = UltraRelativisticColdShockConditions.solve(beta * C, RHO, gamma_hat=4.0 / 3.0)
        error = abs(_xi(result) / (_relative_lorentz_factor(beta, result) - 1.0) - 1.0)
        assert error < 0.5 / gamma1

    def test_newtonian_limit_matches_the_strong_shock_solver(self):
        gamma = 5.0 / 3.0
        v = 1e-3 * C
        relativistic = RelativisticColdShockConditions.solve(v, RHO, gamma_hat=gamma)
        newtonian = StrongColdShockConditions.solve(v, RHO, gamma=gamma)
        np.testing.assert_allclose(
            relativistic.post_shock_thermal_energy_density.to_value(u.erg / u.cm**3),
            newtonian.post_shock_thermal_energy_density.to_value(u.erg / u.cm**3),
            rtol=2e-3,
        )

    def test_units(self):
        result = RelativisticColdShockConditions.solve(0.5 * C, RHO)
        assert result.post_shock_thermal_energy_density.unit == u.erg / u.cm**3


class TestOutputFields:
    """The set of fields in the relativistic results."""

    @pytest.mark.parametrize("solver", [RelativisticColdShockConditions, UltraRelativisticColdShockConditions])
    def test_the_combined_species_temperature_is_not_reported(self, solver):
        result = solver.solve(0.9 * C, RHO)
        assert "post_shock_temperature" not in result._fields
        assert "post_shock_thermal_energy_density" in result._fields

    def test_feeds_the_ion_electron_helper(self):
        result = RelativisticColdShockConditions.solve(0.3 * C, RHO, gamma_hat=4.0 / 3.0)
        Theta_i, Theta_e = compute_ion_electron_theta(
            result.post_shock_thermal_energy_density, result.post_shock_density, 1.17, epsilon_T=0.1
        )
        assert Theta_i > 0.0
        assert Theta_e > Theta_i
