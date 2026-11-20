"""
Tests for OffAxisAsymmetricSynchrotronEngine.

Covers:

  - Lifecycle: construction, property access, repr/str, guard rails on
    inapplicable on-axis properties.
  - Non-relativistic isotropy: at beta=0 and optically thin conditions the
    projected-area factor and the LOS-path-length correction cancel exactly,
    so F_nu must be independent of theta_obs.
  - On-axis limit: at theta_obs=0 the azimuthal integral evaluates to pi, which
    should reproduce the on-axis engine's result to within quadrature tolerance.
"""

import numpy as np
import pytest
from astropy import units as u

from trilobite.radiation.synchrotron.SEDs.numerical import (
    OffAxisAsymmetricSynchrotronEngine,
    OnAxisAsymmetricSynchrotronEngine,
)
from trilobite.radiation.synchrotron.electron_distributions import PowerLaw

# ============================================ #
# TEST CONFIGURATION                           #
# ============================================ #

_N_0 = 1e4  # cm^{-3}  (low enough to stay optically thin)
_B = 0.1  # G
_GAMMA_MIN = 1e2
_GAMMA_MAX = 1e7
_P = 2.5
_N_GAMMA = 300

# Frequencies: well above SSA turnover so we are in the optically thin regime
_NU_TEST = np.array([5e11, 2e12, 1e13])  # Hz

_D_A = (100 * u.Mpc).to(u.cm).value  # cm
_R = 1e17  # cm
_SHELL = 1e12  # cm  (thin shell, ensures optically thin)


# ============================================ #
# Helpers                                      #
# ============================================ #


def _make_N_arr(n_theta, n_gamma, gamma_grid):
    """
    Build a uniform (n_theta, n_gamma) electron distribution array.

    All zones have the same power-law distribution so that spatial gradients
    do not mask any quadrature bugs.
    """
    N_callable = PowerLaw.as_callable(norm=_N_0, p=_P, gamma_min=_GAMMA_MIN, gamma_max=_GAMMA_MAX)
    N_1d = N_callable(gamma_grid)  # (n_gamma,)
    return np.broadcast_to(N_1d, (n_theta, n_gamma)).copy()


def _gamma_grid(engine):
    """Return the gamma grid built by engine._build_gamma_grid with module defaults."""
    log_gamma, _ = engine._build_gamma_grid(None, _GAMMA_MIN, _GAMMA_MAX, _N_GAMMA)
    return np.exp(log_gamma)


# ============================================ #
# Fixtures                                     #
# ============================================ #


@pytest.fixture(scope="module")
def engine():
    """Module-scoped off-axis engine with both kernels loaded."""
    eng = OffAxisAsymmetricSynchrotronEngine(n_theta=20, n_phi=20)
    eng.load_first_kernel(x_min=1e-6, x_max=1e3, num_points=1000)
    eng.load_avg_first_kernel(x_min=1e-6, x_max=1e3, num_points=1000)
    return eng


@pytest.fixture(scope="module")
def on_axis_engine():
    """Module-scoped on-axis engine for on-axis limit comparison."""
    eng = OnAxisAsymmetricSynchrotronEngine(n_theta=40)
    eng.load_avg_first_kernel(x_min=1e-6, x_max=1e3, num_points=1000)
    return eng


@pytest.fixture(scope="module")
def high_res_off_axis_engine():
    """High-resolution off-axis engine for the on-axis limit comparison."""
    eng = OffAxisAsymmetricSynchrotronEngine(n_theta=40, n_phi=40)
    eng.load_avg_first_kernel(x_min=1e-6, x_max=1e3, num_points=1000)
    return eng


# ============================================ #
# Section 1: Lifecycle / Guard Rails           #
# ============================================ #


class TestLifecycle:
    """Construction, properties, repr/str, and on-axis attribute guard rails."""

    def test_construction_default(self):
        eng = OffAxisAsymmetricSynchrotronEngine()
        assert eng.n_theta == 20
        assert eng.n_phi == 20

    def test_construction_custom(self):
        eng = OffAxisAsymmetricSynchrotronEngine(n_theta=15, n_phi=10)
        assert eng.n_theta == 15
        assert eng.n_phi == 10

    def test_cos_theta_range(self, engine):
        ct = engine.cos_theta
        assert ct.min() >= -1.0 and ct.max() <= 1.0
        assert ct.shape == (engine.n_theta,)

    def test_sin_theta_nonneg(self, engine):
        assert np.all(engine.sin_theta >= 0.0)

    def test_phi_range(self, engine):
        phi = engine.phi
        assert phi.min() >= 0.0 and phi.max() <= np.pi
        assert phi.shape == (engine.n_phi,)

    def test_gl_weights_theta_sum(self, engine):
        # GL on [-1, 1] weights sum to 2
        assert np.isclose(engine.gl_weights_theta.sum(), 2.0, rtol=1e-12)

    def test_gl_weights_phi_sum(self, engine):
        # Weights for [0, pi] should sum to pi
        assert np.isclose(engine.gl_weights_phi.sum(), np.pi, rtol=1e-12)

    def test_gl_weights_raises(self, engine):
        with pytest.raises(AttributeError, match="gl_weights_theta"):
            _ = engine.gl_weights

    def test_quad_weights_raises(self, engine):
        with pytest.raises(AttributeError, match="gl_weights_theta"):
            _ = engine.quad_weights

    def test_repr(self, engine):
        r = repr(engine)
        assert "OffAxisAsymmetricSynchrotronEngine" in r
        assert "n_theta=20" in r
        assert "n_phi=20" in r

    def test_str(self, engine):
        s = str(engine)
        assert "OffAxisAsymmetricSynchrotronEngine" in s


# ============================================ #
# Section 2: Output shapes                     #
# ============================================ #


class TestOutputShapes:
    """Verify that all public methods return the documented output shapes."""

    @pytest.fixture(autouse=True)
    def _setup(self, engine):
        self.eng = engine
        gamma = _gamma_grid(engine)
        self.N_arr = _make_N_arr(engine.n_theta, len(gamma), gamma)
        self.gamma = gamma
        self.n_nu = len(_NU_TEST)

    def _common_kw(self):
        return dict(
            nu=_NU_TEST * u.Hz,
            B=np.full(self.eng.n_theta, _B) * u.G,
            N=self.N_arr,
            shell_thickness=np.full(self.eng.n_theta, _SHELL) * u.cm,
            beta=np.zeros(self.eng.n_theta),
            theta_obs=0.0,
            gamma=self.gamma,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        )

    def test_rest_frame_intensity_shape(self):
        I = self.eng.compute_rest_frame_specific_intensity(**self._common_kw())
        assert I.shape == (self.n_nu, self.eng.n_theta, self.eng.n_phi)

    def test_specific_intensity_shape(self):
        I = self.eng.compute_specific_intensity(**self._common_kw())
        assert I.shape == (self.n_nu, self.eng.n_theta, self.eng.n_phi)

    def test_rest_frame_tb_shape(self):
        T = self.eng.compute_rest_frame_brightness_temperature(**self._common_kw())
        assert T.shape == (self.n_nu, self.eng.n_theta, self.eng.n_phi)

    def test_brightness_temperature_shape(self):
        T = self.eng.compute_brightness_temperature(**self._common_kw())
        assert T.shape == (self.n_nu, self.eng.n_theta, self.eng.n_phi)

    def test_flux_density_shape(self):
        F = self.eng.compute_flux_density(
            angular_diameter_distance=_D_A * u.cm,
            R=np.full(self.eng.n_theta, _R) * u.cm,
            **self._common_kw(),
        )
        assert F.shape == (_NU_TEST.shape[0],)

    def test_sightline_flux_density_shape(self):
        F = self.eng.compute_sightline_flux_density(
            angular_diameter_distance=_D_A * u.cm,
            R=np.full(self.eng.n_theta, _R) * u.cm,
            **self._common_kw(),
        )
        assert F.shape == (self.n_nu, self.eng.n_theta, self.eng.n_phi)

    def test_sightline_sums_to_flux(self):
        kw = self._common_kw()
        geo = dict(angular_diameter_distance=_D_A * u.cm, R=np.full(self.eng.n_theta, _R) * u.cm)
        F_full = self.eng.compute_flux_density(**kw, **geo)
        F_zones = self.eng.compute_sightline_flux_density(**kw, **geo)
        np.testing.assert_allclose(
            F_zones.sum(axis=(-2, -1)).to_value(u.erg / u.s / u.cm**2 / u.Hz),
            F_full.to_value(u.erg / u.s / u.cm**2 / u.Hz),
            rtol=1e-10,
        )

    def test_isotropic_luminosity_shape(self):
        L = self.eng.compute_isotropic_luminosity(
            angular_diameter_distance=_D_A * u.cm,
            R=np.full(self.eng.n_theta, _R) * u.cm,
            **self._common_kw(),
        )
        assert L.shape == (_NU_TEST.shape[0],)


# ============================================ #
# Section 3: Non-relativistic isotropy         #
# ============================================ #


class TestNonRelativisticIsotropy:
    r"""
    At beta=0 and in the optically thin regime, the LOS path correction
    ``1/max(mu_obs, mu_min)`` and the projected-area factor ``max(mu_obs, 0)``
    cancel in the flux integrand, making F_nu independent of theta_obs.

    We verify that F_nu at theta_obs = 0, pi/4, pi/2 agree within 5%, which
    allows for quadrature error from the finite grid (n_theta=n_phi=20).
    """

    @pytest.fixture(autouse=True)
    def _setup(self, engine):
        self.eng = engine
        gamma = _gamma_grid(engine)
        self.N_arr = _make_N_arr(engine.n_theta, len(gamma), gamma)
        self.gamma = gamma

    def _flux(self, theta_obs):
        return self.eng.compute_flux_density(
            nu=_NU_TEST * u.Hz,
            B=np.full(self.eng.n_theta, _B) * u.G,
            N=self.N_arr,
            shell_thickness=np.full(self.eng.n_theta, _SHELL) * u.cm,
            R=np.full(self.eng.n_theta, _R) * u.cm,
            beta=np.zeros(self.eng.n_theta),
            theta_obs=theta_obs,
            angular_diameter_distance=_D_A * u.cm,
            gamma=self.gamma,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        ).to_value(u.erg / u.s / u.cm**2 / u.Hz)

    def test_on_axis_vs_forty_five_degrees(self):
        F0 = self._flux(0.0)
        F45 = self._flux(np.pi / 4)
        np.testing.assert_allclose(F45, F0, rtol=0.05)

    def test_on_axis_vs_edge_on(self):
        F0 = self._flux(0.0)
        F90 = self._flux(np.pi / 2)
        np.testing.assert_allclose(F90, F0, rtol=0.05)

    def test_forty_five_vs_edge_on(self):
        F45 = self._flux(np.pi / 4)
        F90 = self._flux(np.pi / 2)
        np.testing.assert_allclose(F90, F45, rtol=0.05)

    def test_flux_finite_and_positive(self):
        for theta in [0.0, np.pi / 4, np.pi / 2]:
            F = self._flux(theta)
            assert np.all(np.isfinite(F))
            assert np.all(F > 0)


# ============================================ #
# Section 4: On-axis limit                     #
# ============================================ #


class TestOnAxisLimit:
    r"""
    At theta_obs=0, mu_obs = cos(theta_jet), so the off-axis formula reduces to

        F = (2/D_A^2) * integral_{-1}^{1} integral_0^{pi} max(mu, 0)
              * f_A * R^2 * D^3 * I'(ell')  dmu dphi

    where ell' = shell / max(mu_obs, mu_min).  The phi integral evaluates to pi
    for a phi-independent integrand, giving

        F = (2*pi/D_A^2) * integral_0^{1} mu * f_A * R^2 * D^3
              * I'(shell/mu)  dmu

    The on-axis engine takes slab_depth as the raw LOS depth per zone (no angular
    correction applied internally).  For the comparison to be equivalent, the
    on-axis engine must receive slab_depth[i] = shell / cos_theta_i, matching
    the ell' that the off-axis engine computes at theta_obs=0.

    We use n_theta=40 for both engines and expect agreement within 5%.  The
    residual error comes from GL node placement differences: the off-axis engine
    places nodes on [-1,1] (half are back-facing and wasted), while the on-axis
    engine places all nodes on [0,1].
    """

    @pytest.fixture(autouse=True)
    def _setup(self, on_axis_engine, high_res_off_axis_engine):
        self.on_ax = on_axis_engine
        self.off_ax = high_res_off_axis_engine

        gamma = _gamma_grid(self.off_ax)
        self.N_off = _make_N_arr(self.off_ax.n_theta, len(gamma), gamma)
        self.N_on = _make_N_arr(self.on_ax.n_theta, len(gamma), gamma)
        self.gamma = gamma

    def _flux_off_axis(self, beta, theta_obs=0.0):
        return self.off_ax.compute_flux_density(
            nu=_NU_TEST * u.Hz,
            B=np.full(self.off_ax.n_theta, _B) * u.G,
            N=self.N_off,
            shell_thickness=np.full(self.off_ax.n_theta, _SHELL) * u.cm,
            R=np.full(self.off_ax.n_theta, _R) * u.cm,
            beta=np.full(self.off_ax.n_theta, beta),
            theta_obs=theta_obs,
            angular_diameter_distance=_D_A * u.cm,
            gamma=self.gamma,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        ).to_value(u.erg / u.s / u.cm**2 / u.Hz)

    def _flux_on_axis(self, beta):
        # slab_depth[i] = shell / cos_theta[i]: apply the LOS correction that
        # the off-axis engine computes internally (at theta_obs=0, mu_obs = cos_theta_i).
        cos_theta_on = self.on_ax.cos_theta  # GL nodes on [0, 1]
        slab_on = _SHELL / cos_theta_on  # LOS path = perp. shell / cos(theta_jet)
        return self.on_ax.compute_flux_density(
            nu=_NU_TEST * u.Hz,
            B=np.full(self.on_ax.n_theta, _B) * u.G,
            N=self.N_on,
            slab_depth=slab_on * u.cm,
            R=np.full(self.on_ax.n_theta, _R) * u.cm,
            beta=np.full(self.on_ax.n_theta, beta),
            angular_diameter_distance=_D_A * u.cm,
            gamma=self.gamma,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        ).to_value(u.erg / u.s / u.cm**2 / u.Hz)

    def test_on_axis_limit_non_relativistic(self):
        """Off-axis at theta_obs=0 matches on-axis engine at beta=0 within 5%."""
        F_off = self._flux_off_axis(beta=0.0, theta_obs=0.0)
        F_on = self._flux_on_axis(beta=0.0)
        np.testing.assert_allclose(F_off, F_on, rtol=0.05)

    def test_on_axis_limit_mildly_relativistic(self):
        """Off-axis at theta_obs=0 matches on-axis engine at beta=0.3 within 5%."""
        F_off = self._flux_off_axis(beta=0.3, theta_obs=0.0)
        F_on = self._flux_on_axis(beta=0.3)
        np.testing.assert_allclose(F_off, F_on, rtol=0.05)

    def test_on_axis_limit_highly_relativistic(self):
        """Off-axis at theta_obs=0 matches on-axis engine at beta=0.9 within 5%."""
        F_off = self._flux_off_axis(beta=0.9, theta_obs=0.0)
        F_on = self._flux_on_axis(beta=0.9)
        np.testing.assert_allclose(F_off, F_on, rtol=0.05)
