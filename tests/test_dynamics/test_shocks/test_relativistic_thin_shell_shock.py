r"""
Tests for :class:`~trilobite.dynamics.shocks.numerical.RelPressureDrivenThinShellShockEngine`.

Three properties are checked:

1. **Internal consistency**: the shell-frame energy :math:`E'=\sqrt{E^2-\Pi^2c^2}` may only change through mass
   loading, :math:`dE' = c^2(\eta_2 dM_2 + \eta_3 dM_3)`. A pressure force changes the lab-frame momentum *and*
   (through the work :math:`\beta c F`) the lab-frame energy, but cannot change :math:`E'`.
2. **Ultra-relativistic deceleration**: a cold shell sweeping up stationary gas decelerates in line with an
   energy-conserving adiabatic snowplow.
3. **Non-relativistic limit**: with matching adiabatic indices the engine converges to
   :class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine` as :math:`\beta^2`, and its
   reported post-shock pressures and loadings are those of the closure used by the ODE.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from trilobite.dynamics.shocks.chevalier import ChevalierSelfSimilarShockEngine
from trilobite.dynamics.shocks.numerical import (
    PressureDrivenThinShellShockEngine,
    RelPressureDrivenThinShellShockEngine,
    _rel_shock_face_closure,
)
from trilobite.physics_utils.constants import c_cgs

# A 1 Msun cold shell sweeping up a uniform, stationary medium. Starting at 1e-3 of the deceleration radius keeps the
# swept mass negligible initially.
_M0 = 2.0e33
_RHO0 = 1.0e-24
_R_DEC = (3 * _M0 / (4 * np.pi * _RHO0)) ** (1 / 3)


def _zero(r, t):
    return 0.0


def _uniform_csm(r, t):
    return _RHO0


def _sweep_uniform_csm(beta_0, n_times=1500, r_max=10.0, **engine_kwargs):
    """Run a cold shell with ``beta_0`` into the uniform CSM out to ``r_max`` deceleration radii."""
    engine = RelPressureDrivenThinShellShockEngine(**engine_kwargs)
    R_0 = 1e-3 * _R_DEC
    t_0 = R_0 / (beta_0 * c_cgs)
    times = np.geomspace(t_0, r_max * _R_DEC / (beta_0 * c_cgs), n_times)
    return engine._compute_shock_properties_cgs(
        times,
        rho_1=_zero,
        rho_4=_uniform_csm,
        beta_1=_zero,
        beta_4=_zero,
        R_0=R_0,
        beta_0=beta_0,
        M_0=_M0,
        t_0=t_0,
        rtol=1e-10,
    )


class TestRelativisticThinShellConsistency:
    @pytest.mark.parametrize("beta_0", [0.1, 0.9, 1 - 1e-6])
    def test_shell_frame_energy_changes_only_by_mass_loading(self, beta_0):
        r"""
        :math:`dE'/dt = c^2 \eta_3 \dot M_3` for a shell with no ejecta behind it.

        This fails without the pressure-work term :math:`\beta c\,4\pi R^2 (P_2 - P_3)` in :math:`dE/dt`.
        """
        sol = _sweep_uniform_csm(beta_0)

        E_shell = np.sqrt(sol.energy**2 - (sol.momentum * c_cgs) ** 2)
        loaded = c_cgs**2 * np.concatenate(
            [[0.0], np.cumsum(0.5 * (sol.eta_3[1:] + sol.eta_3[:-1]) * np.diff(sol.mass))]
        )

        # Compare the increments once the shell energy has changed measurably.
        grown = np.abs(E_shell - E_shell[0]) > 1e-3 * E_shell[0]
        assert grown.any()
        assert_allclose((E_shell - E_shell[0])[grown], loaded[grown], rtol=1e-4)

    def test_ultra_relativistic_shell_follows_adiabatic_snowplow(self):
        r"""
        Energy conservation of a cold shell sweeping up stationary gas gives
        :math:`\Gamma_0 M_0 = \Gamma M_0 + \Gamma^2 M_{\rm sw}` for :math:`\Gamma_0 \gg \Gamma \gg 1`.
        """
        beta_0 = 1 - 1e-6
        Gamma_0 = 1 / np.sqrt(1 - beta_0**2)
        sol = _sweep_uniform_csm(beta_0)

        for swept_fraction in (1e-4, 1e-3, 1e-2, 1e-1):
            i = np.argmin(np.abs((sol.mass - _M0) / _M0 - swept_fraction))
            x = (sol.mass[i] - _M0) / _M0
            Gamma_expected = (-1 + np.sqrt(1 + 4 * x * Gamma_0)) / (2 * x)
            # The pressure-driven closure is not exactly energy conserving, so allow ~10%.
            assert_allclose(sol.Gamma_sh[i], Gamma_expected, rtol=0.10)

    def test_inactive_faces_report_no_post_shock_gas(self):
        """With no ejecta (rho_1 = 0) the reverse-shock diagnostics are the neutral values."""
        sol = _sweep_uniform_csm(0.5, n_times=50, r_max=1.0)

        assert np.all(sol.pressure_2 == 0.0)
        assert np.all(sol.eta_2 == 1.0)
        assert np.all(sol.pressure_3 > 0.0)


def test_per_call_gamma_is_rejected():
    """Adiabatic indices are fixed at instantiation (pre-tabulated shock tables); `gamma=` must not be ignored."""
    engine = RelPressureDrivenThinShellShockEngine()
    with pytest.raises(TypeError, match="gamma_1"):
        engine._compute_shock_properties_cgs(
            np.array([1.0, 2.0]), rho_1=_zero, rho_4=_uniform_csm, beta_1=_zero, beta_4=_zero, gamma=5 / 3
        )


class TestRelativisticThinShellClosure:
    @pytest.mark.parametrize("gamma_hat", [4 / 3, 5 / 3])
    def test_reported_diagnostics_are_the_ode_closure(self, gamma_hat):
        """pressure_3 and eta_3 must be what the closure gives for the reported shell state."""
        engine = RelPressureDrivenThinShellShockEngine(gamma_1=gamma_hat, gamma_4=gamma_hat)
        sol = _sweep_uniform_csm(0.9, n_times=30, r_max=1.0, gamma_1=gamma_hat, gamma_4=gamma_hat)

        for i in range(len(sol.radius)):
            P, eta, _ = _rel_shock_face_closure(
                _RHO0, sol.beta_rel_fs[i], gamma_hat, engine._beta_grid, engine._beta_3_grid
            )
            assert_allclose(sol.pressure_3[i], P, rtol=1e-12)
            assert_allclose(sol.eta_3[i], eta, rtol=1e-12)

    @pytest.mark.parametrize("gamma_hat", [4 / 3, 5 / 3])
    def test_closure_nonrelativistic_limit(self, gamma_hat):
        """P -> (1 - 1/chi) rho v^2 and eta -> 1 as beta -> 0."""
        engine = RelPressureDrivenThinShellShockEngine(gamma_1=gamma_hat, gamma_4=gamma_hat)
        beta = 1e-3
        chi = (gamma_hat + 1) / (gamma_hat - 1)

        P, eta, compression = _rel_shock_face_closure(_RHO0, beta, gamma_hat, engine._beta_grid, engine._beta_3_grid)

        assert_allclose(P, (1 - 1 / chi) * _RHO0 * (beta * c_cgs) ** 2, rtol=1e-5)
        assert_allclose(compression, chi, rtol=1e-5)
        assert_allclose(eta, 1.0, atol=1e-5)

    @pytest.mark.parametrize("beta", [0.1, 0.9, 0.999, 1 - 1e-6])
    def test_loading_never_exceeds_the_upstream_energy_per_baryon(self, beta):
        """The shell cannot be loaded with more energy per rest mass than the upstream carries: eta <= Gamma_rel."""
        engine = RelPressureDrivenThinShellShockEngine()
        _, eta, _ = _rel_shock_face_closure(_RHO0, beta, 4 / 3, engine._beta_grid, engine._beta_3_grid)

        assert 1.0 <= eta <= 1 / np.sqrt(1 - beta**2) * (1 + 1e-9)


class TestRelativisticThinShellNonRelativisticLimit:
    @pytest.mark.parametrize("velocity_scale, tolerance", [(1e8, 1e-5), (1e9, 2e-3)])
    def test_converges_to_the_nonrelativistic_engine(self, velocity_scale, tolerance):
        """
        Homologous n=10 ejecta driving a shell into an s=2 wind, started on the Chevalier self-similar solution.

        With matching adiabatic indices the two engines differ only at order :math:`\\beta^2`.
        """
        n, s = 10, 2
        _lambda = (n - 3) / (n - s)
        zeta = ChevalierSelfSimilarShockEngine.compute_scale_parameter(n, s)

        K_ej = 1e-13 * velocity_scale**n * (86400.0) ** 3
        K_csm = 5e-16 * (1e13) ** s
        t_0 = 1e5
        R_0 = (zeta * K_csm / K_ej) ** (1 / (s - n)) * t_0**_lambda
        v_0 = _lambda * R_0 / t_0
        M_0 = 4 * np.pi * (K_csm * R_0 ** (3 - s) / (3 - s) + K_ej * t_0 ** (n - 3) * R_0 ** (3 - n) / (n - 3))

        def rho_1(r, t):
            return K_ej * t ** (n - 3) * r**-n

        def rho_4(r, t):
            return K_csm * r**-s

        times = np.geomspace(t_0, 1e3 * t_0, 30)

        nonrel = PressureDrivenThinShellShockEngine()._compute_shock_properties_cgs(
            times,
            rho_1=rho_1,
            rho_4=rho_4,
            u_1=lambda r, t: r / t,
            u_4=_zero,
            R_0=R_0,
            v_0=v_0,
            M_0=M_0,
            t_0=t_0,
        )
        rel = RelPressureDrivenThinShellShockEngine(gamma_1=5 / 3, gamma_4=5 / 3)._compute_shock_properties_cgs(
            times,
            rho_1=rho_1,
            rho_4=rho_4,
            beta_1=lambda r, t: r / t / c_cgs,
            beta_4=_zero,
            R_0=R_0,
            beta_0=v_0 / c_cgs,
            M_0=M_0,
            t_0=t_0,
        )

        assert_allclose(rel.radius, nonrel.radius, rtol=tolerance)
        assert_allclose(rel.beta_sh * c_cgs, nonrel.velocity, rtol=tolerance)
        assert_allclose(rel.mass, nonrel.mass, rtol=5 * tolerance)
