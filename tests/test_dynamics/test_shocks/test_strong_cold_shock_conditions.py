"""Regression tests for the strong cold-shock Rankine-Hugoniot conditions."""

import numpy as np
import pytest
from astropy import units as u

from trilobite.dynamics.shocks.core import StrongColdShockConditions

RHO = 1.0e-20 * u.g / u.cm**3
V_SH = 1.0e9 * u.cm / u.s
RHO_V2 = (RHO * V_SH**2).to_value(u.erg / u.cm**3)


def test_post_shock_pressure_is_three_quarters_rho_v2():
    """For gamma = 5/3, P_2 = 2 rho_1 v^2 / (gamma + 1) = (3/4) rho_1 v^2."""
    P2 = StrongColdShockConditions.compute_post_shock_pressure(shock_velocity=V_SH, upstream_density=RHO)
    assert P2.to_value(u.dyn / u.cm**2) == pytest.approx(0.75 * RHO_V2)


def test_post_shock_thermal_energy_density_is_nine_eighths_rho_v2():
    """For gamma = 5/3, e_2 = P_2 / (gamma - 1) = (9/8) rho_1 v^2."""
    e2 = StrongColdShockConditions.compute_post_shock_thermal_energy_density(shock_velocity=V_SH, upstream_density=RHO)
    assert e2.to_value(u.erg / u.cm**3) == pytest.approx(9.0 / 8.0 * RHO_V2)


@pytest.mark.parametrize("gamma", [4.0 / 3.0, 5.0 / 3.0, 2.0])
def test_thermal_energy_density_matches_pressure_over_gamma_minus_one(gamma):
    kwargs = dict(shock_velocity=V_SH, upstream_density=RHO, gamma=gamma)
    P2 = StrongColdShockConditions.compute_post_shock_pressure(**kwargs)
    e2 = StrongColdShockConditions.compute_post_shock_thermal_energy_density(**kwargs)
    assert np.isclose(e2.to_value(u.erg / u.cm**3), P2.to_value(u.dyn / u.cm**2) / (gamma - 1.0))
