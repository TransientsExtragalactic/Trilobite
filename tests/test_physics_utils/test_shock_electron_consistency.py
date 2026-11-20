"""End-to-end consistency of the shock solver, the ion/electron partition and the thermal electron distributions."""

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from trilobite.dynamics.shocks.core import RelativisticColdShockConditions
from trilobite.physics_utils import compute_ion_electron_theta
from trilobite.physics_utils.constants import c_cgs, m_e_cgs, m_p_cgs
from trilobite.physics_utils.maxwell_juettner import compute_mj_mean_kinetic_energy
from trilobite.radiation.synchrotron.electron_distributions import MaxwellJuettner

RHO = 1e-22 * u.g / u.cm**3
MU_E = 1.17
BETAS = [0.03, 0.1, 0.3, 0.7, 0.95]
BUDGETS = [
    {"epsilon_T": 0.05, "epsilon_B": 0.1, "epsilon_E": 0.0},
    {"epsilon_T": 0.5, "epsilon_B": 0.1, "epsilon_E": 0.1},
    {"epsilon_T": 5.0, "epsilon_B": 0.01, "epsilon_E": 0.2},
]


def _shock(beta):
    result = RelativisticColdShockConditions.solve(beta * const.c, RHO, gamma_hat=4.0 / 3.0)
    return result.post_shock_thermal_energy_density.to_value(u.erg / u.cm**3), result.post_shock_density.to_value(
        u.g / u.cm**3
    )


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("budget", BUDGETS)
class TestShockToElectronDistribution:
    """The thermal electrons' energy budget, with the helper's Theta_e, returns the electron density of the shock."""

    def test_energy_normalization_returns_the_electron_density(self, beta, budget):
        u_th, rho = _shock(beta)
        _, Theta_e = compute_ion_electron_theta(u_th, rho, MU_E, **budget)
        N0 = MaxwellJuettner.normalize_from_energy_density(u_th, Theta=Theta_e, **budget)
        assert N0.value == pytest.approx(rho / (MU_E * m_p_cgs), rel=1e-9)

    def test_magnetic_field_normalization_returns_the_electron_density(self, beta, budget):
        u_th, rho = _shock(beta)
        _, Theta_e = compute_ion_electron_theta(u_th, rho, MU_E, **budget)
        B = np.sqrt(8.0 * np.pi * budget["epsilon_B"] * u_th)  # u_B = epsilon_B u_th
        epsilon_B = budget["epsilon_B"]
        N0 = MaxwellJuettner.normalize_from_magnetic_field(
            B, epsilon_B, epsilon_T=budget["epsilon_T"], epsilon_E=budget["epsilon_E"], Theta=Theta_e
        )
        assert N0.value == pytest.approx(rho / (MU_E * m_p_cgs), rel=1e-9)

    def test_every_part_of_the_thermal_energy_is_accounted_for(self, beta, budget):
        u_th, rho = _shock(beta)
        Theta_i, Theta_e = compute_ion_electron_theta(u_th, rho, MU_E, **budget)
        u_i = rho * c_cgs**2 * compute_mj_mean_kinetic_energy(Theta_i)  # n_i m_i = rho
        u_e = rho / (MU_E * m_p_cgs) * m_e_cgs * c_cgs**2 * compute_mj_mean_kinetic_energy(Theta_e)
        accounted = u_i + u_e + (budget["epsilon_B"] + budget["epsilon_E"]) * u_th
        assert accounted == pytest.approx(u_th, rel=1e-9)
        assert u_e == pytest.approx(budget["epsilon_T"] * u_i, rel=1e-9)


class TestRegimes:
    """The partition spans the non-relativistic and relativistic regimes of both species."""

    def test_ions_become_relativistic_in_a_relativistic_shock(self):
        u_th, rho = _shock(0.95)
        Theta_i, Theta_e = compute_ion_electron_theta(u_th, rho, MU_E, epsilon_T=0.1, epsilon_B=0.1)
        assert Theta_i > 0.3  # relativistic ions, which the single-species inversion cannot represent
        assert Theta_e > Theta_i

    def test_ions_stay_non_relativistic_in_a_slow_shock(self):
        u_th, rho = _shock(0.01)
        Theta_i, _ = compute_ion_electron_theta(u_th, rho, MU_E, epsilon_T=0.1, epsilon_B=0.1)
        assert Theta_i < 1e-3

    def test_electrons_are_mildly_relativistic_at_a_tenth_of_the_speed_of_light(self):
        u_th, rho = _shock(0.1)
        _, Theta_e = compute_ion_electron_theta(u_th, rho, MU_E, epsilon_T=1.0, epsilon_B=0.0)
        assert 0.3 < Theta_e < 10.0
