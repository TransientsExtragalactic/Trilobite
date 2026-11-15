"""Tests for :mod:`trilobite.physics_utils.thermal_partition`."""

import astropy.units as u
import numpy as np
import pytest

from trilobite.physics_utils.constants import c_cgs, m_e_cgs, m_p_cgs
from trilobite.physics_utils.maxwell_juettner import compute_mj_mean_kinetic_energy
from trilobite.physics_utils.thermal_partition import (
    compute_ion_electron_theta,
    compute_thermal_electron_energy_fraction,
)

RHO = 3e-18  # g cm^-3
MU_E = 1.17


def _u_th(beta, rho=RHO):
    """Strong-shock thermal energy density, 9/8 rho v^2."""
    return 9.0 / 8.0 * rho * (beta * c_cgs) ** 2


class TestEnergyFraction:
    def test_known_values(self):
        assert compute_thermal_electron_energy_fraction(1.0) == pytest.approx(0.5)
        assert compute_thermal_electron_energy_fraction(0.25) == pytest.approx(0.2)
        assert compute_thermal_electron_energy_fraction(1.0, 0.1, 0.2) == pytest.approx(0.5 * 0.7)

    def test_no_thermal_electrons(self):
        assert compute_thermal_electron_energy_fraction(0.0, 0.1, 0.1) == 0.0

    def test_electrons_take_everything_that_is_left_as_epsilon_T_grows(self):
        assert compute_thermal_electron_energy_fraction(1e9, 0.1, 0.2) == pytest.approx(0.7, rel=1e-8)

    def test_broadcasts(self):
        out = compute_thermal_electron_energy_fraction(np.array([[0.1], [1.0]]), np.array([0.0, 0.1, 0.2]))
        assert out.shape == (2, 3)

    @pytest.mark.parametrize("kwargs", [{"epsilon_T": -0.1}, {"epsilon_T": 1.0, "epsilon_B": 0.6, "epsilon_E": 0.4}])
    def test_invalid_inputs_raise(self, kwargs):
        with pytest.raises(ValueError):
            compute_thermal_electron_energy_fraction(**kwargs)


class TestIonElectronTheta:
    @pytest.mark.parametrize("beta", [0.003, 0.03, 0.1, 0.3, 0.9])
    @pytest.mark.parametrize("epsilon_T", [0.01, 0.2, 1.0, 30.0])
    @pytest.mark.parametrize(("eps_B", "eps_E"), [(0.0, 0.0), (0.1, 0.05)])
    def test_energy_is_conserved(self, beta, epsilon_T, eps_B, eps_E):
        # The two Maxwell-Juettner species, plus the magnetic field and non-thermal electrons, add up to u_th.
        u_th = _u_th(beta)
        Theta_i, Theta_e = compute_ion_electron_theta(u_th, RHO, MU_E, epsilon_T, eps_B, eps_E)
        n_e = RHO / (MU_E * m_p_cgs)
        u_i = RHO * c_cgs**2 * compute_mj_mean_kinetic_energy(Theta_i)  # n_i m_i = rho
        u_e = n_e * m_e_cgs * c_cgs**2 * compute_mj_mean_kinetic_energy(Theta_e)
        assert u_i + u_e + (eps_B + eps_E) * u_th == pytest.approx(u_th, rel=1e-9)
        assert u_e == pytest.approx(epsilon_T * u_i, rel=1e-9)

    def test_non_relativistic_ions(self):
        # <gamma - 1> -> 3 Theta / 2, so Theta_i = (2/3) u_i / (rho c^2) with u_i = u_th / (1 + epsilon_T).
        beta, epsilon_T = 1e-3, 0.4
        Theta_i, _ = compute_ion_electron_theta(_u_th(beta), RHO, MU_E, epsilon_T)
        assert Theta_i == pytest.approx(2.0 / 3.0 * (9.0 / 8.0) * beta**2 / (1.0 + epsilon_T), rel=1e-5)

    def test_ultra_relativistic_electrons(self):
        # <gamma - 1> -> 3 Theta, so Theta_e = xi_e / 3 once xi_e >> 1.
        _, Theta_e = compute_ion_electron_theta(_u_th(0.9) * 1e4, RHO, MU_E, 1.0)
        xi_e = 0.5 * _u_th(0.9) * 1e4 * MU_E * m_p_cgs / (RHO * m_e_cgs * c_cgs**2)
        assert Theta_e == pytest.approx(xi_e / 3.0, rel=1e-3)

    @pytest.mark.parametrize("mu_i", [1.0, 3.97])
    def test_equal_temperatures_in_the_non_relativistic_limit(self, mu_i):
        # In equilibrium u_e / u_i = n_e / n_i = mu_i / mu_e, and T_e = T_i means Theta_e m_e = Theta_i m_i = Theta_i mu_i m_p.
        mu_e = 1.0 if mu_i == 1.0 else 2.0
        Theta_i, Theta_e = compute_ion_electron_theta(_u_th(1e-4), RHO, mu_e, epsilon_T=mu_i / mu_e)
        assert Theta_e * m_e_cgs == pytest.approx(Theta_i * mu_i * m_p_cgs, rel=1e-4)

    def test_few_thermal_electrons_leave_the_ion_temperature_unchanged(self):
        Theta_i_cold, _ = compute_ion_electron_theta(_u_th(0.1), RHO, MU_E, 1e-8)
        Theta_i_none, _ = compute_ion_electron_theta(_u_th(0.1), RHO, MU_E, 1e-8, 0.0, 0.0)
        assert Theta_i_cold == Theta_i_none
        _, Theta_e = compute_ion_electron_theta(_u_th(0.1), RHO, MU_E, 1e-8)
        assert Theta_e < 1e-3 * Theta_i_cold * m_p_cgs / m_e_cgs

    def test_electron_temperature_grows_with_epsilon_T(self):
        epsilon_T = np.logspace(-3, 3, 30)
        Theta_i, Theta_e = compute_ion_electron_theta(_u_th(0.2), RHO, MU_E, epsilon_T)
        assert np.all(np.diff(Theta_e) > 0.0)
        assert np.all(np.diff(Theta_i) < 0.0)

    def test_quantity_and_bare_inputs_agree(self):
        u_th = _u_th(0.1)
        bare = compute_ion_electron_theta(u_th, RHO, MU_E, 0.3)
        with_units = compute_ion_electron_theta(u_th * u.erg / u.cm**3, RHO * u.g / u.cm**3, MU_E, 0.3)
        np.testing.assert_allclose(bare, with_units, rtol=1e-14)

    def test_broadcasts_over_inputs(self):
        Theta_i, Theta_e = compute_ion_electron_theta(
            _u_th(np.array([[0.01], [0.1]])), RHO, MU_E, np.array([0.1, 1.0, 10.0])
        )
        assert Theta_i.shape == Theta_e.shape == (2, 3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"epsilon_T": 0.0},
            {"epsilon_T": -1.0},
            {"epsilon_T": 1.0, "epsilon_B": 0.7, "epsilon_E": 0.3},
            {"epsilon_T": np.nan},
            {"epsilon_T": 1.0, "mu_e": 0.0},
            {"epsilon_T": 1.0, "rho": -1.0},
            {"epsilon_T": 1.0, "u_th": 0.0},
        ],
    )
    def test_invalid_inputs_raise(self, kwargs):
        args = {"u_th": _u_th(0.1), "rho": RHO, "mu_e": MU_E, **kwargs}
        with pytest.raises(ValueError):
            compute_ion_electron_theta(**args)
