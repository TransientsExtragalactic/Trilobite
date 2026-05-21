"""Unit tests for the free-free optical depth module (absorption.py).

Organisation
------------
TestOutputContract       — shape, dtype, positivity for all five profile backends
TestUnitCoercion         — bare-float == astropy.Quantity inputs
TestShellAnalytic        — tau = alpha_ff * (r_max - r_min) cross-check
TestQuadratureVsShell    — constant-profile quadrature matches analytic shell
TestArraysVsShell        — uniform-density arrays match analytic shell
TestWindVsPowerlaw       — wind matches powerlaw(p=2)
TestPowerlawSpecialCases — p=0 matches shell; k=0 finite; k<0 converges
TestEMFunction           — EM backend matches shell when EM = n_e*n_i*L
TestFrequencyScaling     — tau ∝ nu^{-2} in RJ limit
TestRmaxEffect           — larger r_max increases tau
TestComposition          — Xs=None single-species vs Xs composition
TestThomson              — thomson=True returns eff >= tau_ff
"""

import numpy as np
import pytest
from astropy import units as u

from trilobite.radiation.free_free.absorption import (
    compute_ff_RJ_optical_depth_EM,
    compute_ff_RJ_optical_depth_from_arrays,
    compute_ff_RJ_optical_depth_from_quadrature,
    compute_ff_RJ_optical_depth_powerlaw,
    compute_ff_RJ_optical_depth_shell,
    compute_ff_RJ_optical_depth_wind,
)

# ------------------------------------------------------------------ #
# Shared fixtures                                                    #
# ------------------------------------------------------------------ #
_NU = np.array([1e9, 5e9, 1e10]) * u.Hz
_R_MIN = 1.0e15 * u.cm
_R_0 = 1.0e16 * u.cm
_R_MAX = 2.0e15 * u.cm
_T = 1.0e4 * u.K
_NE = 1.0e4 / u.cm**3
_NI = 1.0e4 / u.cm**3


# ================================================================== #
# Output contract                                                    #
# ================================================================== #
class TestOutputContract:
    """Every backend returns a finite, positive ndarray of the right shape."""

    def test_shell_shape(self):
        tau = compute_ff_RJ_optical_depth_shell(_NU, _R_MIN, _NE, _NI, r_max=_R_MAX, temperature=_T)
        assert tau.shape == (3,)

    def test_wind_shape(self):
        tau = compute_ff_RJ_optical_depth_wind(_NU, _R_MIN, _R_0, _NE, _NI, temperature=_T)
        assert tau.shape == (3,)

    def test_powerlaw_shape(self):
        tau = compute_ff_RJ_optical_depth_powerlaw(_NU, _R_MIN, _R_0, _NE, _NI, 2.0, temperature=_T)
        assert tau.shape == (3,)

    def test_arrays_shape(self):
        r_arr = np.geomspace(1e15, 2e15, 50) * u.cm
        ne_arr = _NE * np.ones(50)
        tau = compute_ff_RJ_optical_depth_from_arrays(_NU, r_arr, ne_arr, ne_arr, temperature=_T)
        assert tau.shape == (3,)

    def test_quadrature_shape(self):
        tau = compute_ff_RJ_optical_depth_from_quadrature(
            _NU,
            _R_MIN,
            n_e=lambda r: 1e4,
            n_i=lambda r: 1e4,
            temperature=lambda r: 1e4,
            r_max=_R_MAX,
        )
        assert tau.shape == (3,)

    def test_em_shape(self):
        EM = (_NE * _NI * (_R_MAX - _R_MIN)).to(u.cm**-5)
        tau = compute_ff_RJ_optical_depth_EM(_NU, EM, temperature=_T)
        assert tau.shape == (3,)

    @pytest.mark.parametrize(
        "backend,kwargs",
        [
            ("shell", dict(r_min=_R_MIN, n_e_0=_NE, n_i_0=_NI, r_max=_R_MAX, temperature=_T)),
            ("wind", dict(r_min=_R_MIN, r_0=_R_0, n_e_0=_NE, n_i_0=_NI, temperature=_T)),
            ("powerlaw", dict(r_min=_R_MIN, r_0=_R_0, n_e_0=_NE, n_i_0=_NI, p=2.0, temperature=_T)),
        ],
    )
    def test_all_positive(self, backend, kwargs):
        fn = {
            "shell": compute_ff_RJ_optical_depth_shell,
            "wind": compute_ff_RJ_optical_depth_wind,
            "powerlaw": compute_ff_RJ_optical_depth_powerlaw,
        }[backend]
        tau = fn(_NU, **kwargs)
        assert np.all(tau > 0)

    @pytest.mark.parametrize(
        "backend,kwargs",
        [
            ("shell", dict(r_min=_R_MIN, n_e_0=_NE, n_i_0=_NI, r_max=_R_MAX, temperature=_T)),
            ("wind", dict(r_min=_R_MIN, r_0=_R_0, n_e_0=_NE, n_i_0=_NI, temperature=_T)),
            ("powerlaw", dict(r_min=_R_MIN, r_0=_R_0, n_e_0=_NE, n_i_0=_NI, p=2.0, temperature=_T)),
        ],
    )
    def test_all_finite(self, backend, kwargs):
        fn = {
            "shell": compute_ff_RJ_optical_depth_shell,
            "wind": compute_ff_RJ_optical_depth_wind,
            "powerlaw": compute_ff_RJ_optical_depth_powerlaw,
        }[backend]
        tau = fn(_NU, **kwargs)
        assert np.all(np.isfinite(tau))


# ================================================================== #
# Unit coercion                                                      #
# ================================================================== #
class TestUnitCoercion:
    """Bare floats (in CGS) and astropy Quantities give identical results."""

    def test_shell_bare_vs_quantity(self):
        tau_qty = compute_ff_RJ_optical_depth_shell(_NU, _R_MIN, _NE, _NI, r_max=_R_MAX, temperature=_T)
        tau_bare = compute_ff_RJ_optical_depth_shell(
            _NU.to_value(u.Hz),
            _R_MIN.to_value(u.cm),
            _NE.to_value(u.cm**-3),
            _NI.to_value(u.cm**-3),
            r_max=_R_MAX.to_value(u.cm),
            temperature=_T.to_value(u.K),
        )
        assert np.allclose(tau_qty, tau_bare, rtol=1e-10)

    def test_wind_bare_vs_quantity(self):
        tau_qty = compute_ff_RJ_optical_depth_wind(_NU, _R_MIN, _R_0, _NE, _NI, temperature=_T)
        tau_bare = compute_ff_RJ_optical_depth_wind(
            _NU.to_value(u.Hz),
            _R_MIN.to_value(u.cm),
            _R_0.to_value(u.cm),
            _NE.to_value(u.cm**-3),
            _NI.to_value(u.cm**-3),
            temperature=_T.to_value(u.K),
        )
        assert np.allclose(tau_qty, tau_bare, rtol=1e-10)

    def test_em_bare_vs_quantity(self):
        EM = (_NE * _NI * (_R_MAX - _R_MIN)).to(u.cm**-5)
        tau_qty = compute_ff_RJ_optical_depth_EM(_NU, EM, temperature=_T)
        tau_bare = compute_ff_RJ_optical_depth_EM(
            _NU.to_value(u.Hz),
            EM.to_value(u.cm**-5),
            temperature=_T.to_value(u.K),
        )
        assert np.allclose(tau_qty, tau_bare, rtol=1e-10)


# ================================================================== #
# Shell analytic cross-check                                         #
# ================================================================== #
class TestShellAnalytic:
    """tau_shell = alpha_ff * L, where alpha_ff comes from core.py."""

    def test_shell_matches_core_absorption(self):
        from trilobite.radiation.free_free.core import compute_ff_RJ_absorption

        # Fix g_ff=1.0 in both so the Gaunt-factor defaults don't differ.
        G_FF = 1.0
        L = (_R_MAX - _R_MIN).to(u.cm)
        alpha = compute_ff_RJ_absorption(_NU, _NE, _NI, Z=1, T=_T, g_ff=G_FF)
        tau_expected = (alpha * L).decompose().value

        tau_shell = compute_ff_RJ_optical_depth_shell(_NU, _R_MIN, _NE, _NI, r_max=_R_MAX, temperature=_T, g_ff=G_FF)
        assert np.allclose(tau_shell, tau_expected, rtol=1e-6)


# ================================================================== #
# Quadrature vs shell                                                #
# ================================================================== #
class TestQuadratureVsShell:
    """Constant-profile quadrature must agree with analytic shell."""

    def test_quadrature_matches_shell(self):
        r_min, r_max = 1e15, 2e15  # cm
        ne = ni = 1e4  # cm^{-3}
        T = 1e4  # K
        G_FF = 1.0  # fix g_ff so both backends use the same value

        tau_quad = compute_ff_RJ_optical_depth_from_quadrature(
            _NU,
            r_min * u.cm,
            n_e=lambda r: ne,
            n_i=lambda r: ni,
            temperature=lambda r: T,
            r_max=r_max * u.cm,
            g_ff=G_FF,
        )
        tau_shell = compute_ff_RJ_optical_depth_shell(
            _NU,
            r_min * u.cm,
            ne / u.cm**3,
            ni / u.cm**3,
            r_max=r_max * u.cm,
            temperature=T * u.K,
            g_ff=G_FF,
        )
        assert np.allclose(tau_quad, tau_shell, rtol=1e-4)


# ================================================================== #
# Arrays vs shell                                                    #
# ================================================================== #
class TestArraysVsShell:
    """Uniform trapezoidal grid must converge to the analytic shell result."""

    def test_arrays_match_shell(self):
        r_min, r_max = 1e15, 2e15
        ne = ni = 1e4
        T = 1e4

        r_arr = np.linspace(r_min, r_max, 500) * u.cm
        ne_arr = np.full(500, ne) / u.cm**3

        tau_arr = compute_ff_RJ_optical_depth_from_arrays(_NU, r_arr, ne_arr, ne_arr, temperature=T * u.K)
        tau_shell = compute_ff_RJ_optical_depth_shell(
            _NU,
            r_min * u.cm,
            ne / u.cm**3,
            ni / u.cm**3,
            r_max=r_max * u.cm,
            temperature=T * u.K,
        )
        assert np.allclose(tau_arr, tau_shell, rtol=1e-3)


# ================================================================== #
# Wind vs powerlaw(p=2)                                              #
# ================================================================== #
class TestWindVsPowerlaw:
    def test_wind_equals_powerlaw_p2(self):
        tau_wind = compute_ff_RJ_optical_depth_wind(_NU, _R_MIN, _R_0, _NE, _NI, temperature=_T)
        tau_pl = compute_ff_RJ_optical_depth_powerlaw(_NU, _R_MIN, _R_0, _NE, _NI, 2.0, temperature=_T)
        assert np.allclose(tau_wind, tau_pl, rtol=1e-10)


# ================================================================== #
# Powerlaw special cases                                             #
# ================================================================== #
class TestPowerlawSpecialCases:
    def test_p0_matches_shell(self):
        """p=0 is a uniform-density slab; must agree with shell."""
        tau_pl = compute_ff_RJ_optical_depth_powerlaw(
            _NU,
            _R_MIN,
            _R_0,
            _NE,
            _NI,
            p=0.0,
            r_max=_R_MAX,
            temperature=_T,
        )
        # With p=0: n(r) = n_0*(r_0/r)^0 = n_0; use n_0 as uniform density
        tau_sh = compute_ff_RJ_optical_depth_shell(_NU, _R_MIN, _NE, _NI, r_max=_R_MAX, temperature=_T)
        assert np.allclose(tau_pl, tau_sh, rtol=1e-10)

    def test_k_zero_is_finite(self):
        """k = 1-2p = 0 at p=0.5; result must be finite and positive."""
        tau = compute_ff_RJ_optical_depth_powerlaw(
            _NU,
            _R_MIN,
            _R_0,
            _NE,
            _NI,
            p=0.5,
            r_max=_R_MAX,
            temperature=_T,
        )
        assert np.all(np.isfinite(tau))
        assert np.all(tau > 0)

    def test_k_negative_converges_at_infinity(self):
        """p > 0.5 (k < 0): integral converges as r_max → ∞."""
        tau_finite = compute_ff_RJ_optical_depth_powerlaw(
            _NU,
            _R_MIN,
            _R_0,
            _NE,
            _NI,
            p=2.0,
            r_max=1e20 * u.cm,
            temperature=_T,
        )
        tau_inf = compute_ff_RJ_optical_depth_powerlaw(
            _NU,
            _R_MIN,
            _R_0,
            _NE,
            _NI,
            p=2.0,
            r_max=np.inf * u.cm,
            temperature=_T,
        )
        assert np.allclose(tau_finite, tau_inf, rtol=1e-3)


# ================================================================== #
# EM function                                                        #
# ================================================================== #
class TestEMFunction:
    def test_em_matches_shell(self):
        """EM = n_e * n_i * L should reproduce the shell result."""
        L = (_R_MAX - _R_MIN).to(u.cm)
        EM = (_NE * _NI * L).to(u.cm**-5)

        tau_em = compute_ff_RJ_optical_depth_EM(_NU, EM, temperature=_T)
        tau_sh = compute_ff_RJ_optical_depth_shell(_NU, _R_MIN, _NE, _NI, r_max=_R_MAX, temperature=_T)
        assert np.allclose(tau_em, tau_sh, rtol=1e-6)

    def test_em_scales_linearly(self):
        EM = 1e24 * u.cm**-5
        tau1 = compute_ff_RJ_optical_depth_EM(_NU, EM, temperature=_T)
        tau2 = compute_ff_RJ_optical_depth_EM(_NU, 2 * EM, temperature=_T)
        assert np.allclose(tau2 / tau1, 2.0, rtol=1e-10)

    def test_em_positive(self):
        tau = compute_ff_RJ_optical_depth_EM(_NU, 1e24 * u.cm**-5, temperature=_T)
        assert np.all(tau > 0)


# ================================================================== #
# Frequency scaling                                                  #
# ================================================================== #
class TestFrequencyScaling:
    """In the RJ limit tau ∝ nu^{-2}."""

    @pytest.mark.parametrize(
        "backend,kwargs",
        [
            ("shell", dict(r_min=_R_MIN, n_e_0=_NE, n_i_0=_NI, r_max=_R_MAX, temperature=_T, g_ff=1.0)),
            ("wind", dict(r_min=_R_MIN, r_0=_R_0, n_e_0=_NE, n_i_0=_NI, temperature=_T, g_ff=1.0)),
            ("powerlaw", dict(r_min=_R_MIN, r_0=_R_0, n_e_0=_NE, n_i_0=_NI, p=2.0, temperature=_T, g_ff=1.0)),
        ],
    )
    def test_nu_minus_2_scaling(self, backend, kwargs):
        """tau ∝ nu^{-2} exactly when g_ff is held fixed."""
        fn = {
            "shell": compute_ff_RJ_optical_depth_shell,
            "wind": compute_ff_RJ_optical_depth_wind,
            "powerlaw": compute_ff_RJ_optical_depth_powerlaw,
        }[backend]
        nu1, nu2 = np.array([1e9]) * u.Hz, np.array([2e9]) * u.Hz
        tau1 = fn(nu1, **kwargs)
        tau2 = fn(nu2, **kwargs)
        assert tau1[0] / tau2[0] == pytest.approx(4.0, rel=1e-6)

    def test_em_nu_minus_2_scaling(self):
        """tau ∝ nu^{-2} exactly when g_ff is held fixed."""
        EM = 1e24 * u.cm**-5
        tau1 = compute_ff_RJ_optical_depth_EM(np.array([1e9]) * u.Hz, EM, temperature=_T, g_ff=1.0)
        tau2 = compute_ff_RJ_optical_depth_EM(np.array([2e9]) * u.Hz, EM, temperature=_T, g_ff=1.0)
        assert tau1[0] / tau2[0] == pytest.approx(4.0, rel=1e-6)


# ================================================================== #
# r_max effect                                                       #
# ================================================================== #
class TestRmaxEffect:
    def test_larger_rmax_increases_tau_shell(self):
        nu = np.array([1e9]) * u.Hz
        tau_small = compute_ff_RJ_optical_depth_shell(nu, _R_MIN, _NE, _NI, r_max=2e15 * u.cm, temperature=_T)
        tau_large = compute_ff_RJ_optical_depth_shell(nu, _R_MIN, _NE, _NI, r_max=5e15 * u.cm, temperature=_T)
        assert tau_large[0] > tau_small[0]

    def test_larger_rmax_increases_tau_wind(self):
        nu = np.array([1e9]) * u.Hz
        tau_small = compute_ff_RJ_optical_depth_wind(nu, _R_MIN, _R_0, _NE, _NI, r_max=2e15 * u.cm, temperature=_T)
        tau_large = compute_ff_RJ_optical_depth_wind(nu, _R_MIN, _R_0, _NE, _NI, r_max=5e15 * u.cm, temperature=_T)
        assert tau_large[0] > tau_small[0]


# ================================================================== #
# Composition                                                        #
# ================================================================== #
class TestComposition:
    """Xs=None (single-species) and Xs-specified (composition) are consistent."""

    Zs = np.array([1.0])
    Xs = np.array([1.0])

    def test_single_species_array_matches_scalar_Z(self):
        tau_scalar = compute_ff_RJ_optical_depth_shell(_NU, _R_MIN, _NE, _NI, r_max=_R_MAX, temperature=_T, Z=1.0)
        tau_comp = compute_ff_RJ_optical_depth_shell(
            _NU,
            _R_MIN,
            _NE,
            _NI,
            r_max=_R_MAX,
            temperature=_T,
            Z=self.Zs,
            X=self.Xs,
        )
        assert np.allclose(tau_scalar, tau_comp, rtol=1e-6)

    def test_helium_mix_increases_tau(self):
        """He contributes Z^2=4 so mixed plasma has larger tau than pure H."""
        tau_pure_H = compute_ff_RJ_optical_depth_shell(_NU, _R_MIN, _NE, _NI, r_max=_R_MAX, temperature=_T, Z=1.0)
        tau_mixed = compute_ff_RJ_optical_depth_shell(
            _NU,
            _R_MIN,
            _NE,
            _NI,
            r_max=_R_MAX,
            temperature=_T,
            Z=np.array([1.0, 2.0]),
            X=np.array([0.9, 0.1]),
        )
        assert np.all(tau_mixed > tau_pure_H)

    def test_composition_wind_positive(self):
        tau = compute_ff_RJ_optical_depth_wind(
            _NU,
            _R_MIN,
            _R_0,
            _NE,
            _NI,
            temperature=_T,
            Z=np.array([1.0, 2.0]),
            X=np.array([0.9, 0.1]),
        )
        assert np.all(tau > 0)


# ================================================================== #
# Thomson flag                                                       #
# ================================================================== #
class TestThomson:
    """With thomson=True, tau_eff >= tau_ff always."""

    @pytest.mark.parametrize(
        "backend,kwargs",
        [
            ("shell", dict(r_min=_R_MIN, n_e_0=_NE, n_i_0=_NI, r_max=_R_MAX, temperature=_T)),
            ("wind", dict(r_min=_R_MIN, r_0=_R_0, n_e_0=_NE, n_i_0=_NI, temperature=_T)),
            ("powerlaw", dict(r_min=_R_MIN, r_0=_R_0, n_e_0=_NE, n_i_0=_NI, p=2.0, temperature=_T)),
        ],
    )
    def test_eff_ge_ff(self, backend, kwargs):
        fn = {
            "shell": compute_ff_RJ_optical_depth_shell,
            "wind": compute_ff_RJ_optical_depth_wind,
            "powerlaw": compute_ff_RJ_optical_depth_powerlaw,
        }[backend]
        tau_ff = fn(_NU, **kwargs, thomson=False)
        tau_eff = fn(_NU, **kwargs, thomson=True)
        assert np.all(tau_eff >= tau_ff)

    def test_quadrature_eff_ge_ff(self):
        r_min, r_max, ne, T = 1e15, 2e15, 1e4, 1e4
        tau_ff = compute_ff_RJ_optical_depth_from_quadrature(
            _NU,
            r_min * u.cm,
            n_e=lambda r: ne,
            n_i=lambda r: ne,
            temperature=lambda r: T,
            r_max=r_max * u.cm,
            thomson=False,
        )
        tau_eff = compute_ff_RJ_optical_depth_from_quadrature(
            _NU,
            r_min * u.cm,
            n_e=lambda r: ne,
            n_i=lambda r: ne,
            temperature=lambda r: T,
            r_max=r_max * u.cm,
            thomson=True,
        )
        assert np.all(tau_eff >= tau_ff)

    def test_arrays_eff_ge_ff(self):
        r_arr = np.linspace(1e15, 2e15, 100) * u.cm
        ne_arr = np.full(100, 1e4) / u.cm**3
        tau_ff = compute_ff_RJ_optical_depth_from_arrays(_NU, r_arr, ne_arr, ne_arr, temperature=_T, thomson=False)
        tau_eff = compute_ff_RJ_optical_depth_from_arrays(_NU, r_arr, ne_arr, ne_arr, temperature=_T, thomson=True)
        assert np.all(tau_eff >= tau_ff)

    def test_thomson_positive(self):
        tau = compute_ff_RJ_optical_depth_shell(_NU, _R_MIN, _NE, _NI, r_max=_R_MAX, temperature=_T, thomson=True)
        assert np.all(tau > 0)
