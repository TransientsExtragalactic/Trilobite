"""
Testing suite for verifying the accuracy and correctness of NumericalSynchrotronEngine.

Covers:

  - Lifecycle: kernel loading, guard rails, state flags, clear, dunder methods,
    context manager, load options.
  - Physical fidelity (fixed alpha=pi/2): emissivity vs analytic c_5(p), absorption
    vs analytic c_6(p), source function, and SSA turnover frequency.
  - Radiative transfer limits: optically thin (linear scaling) and thick (saturation).
  - Doppler/redshift: z=0,beta=0 identity; D^3 intensity scaling law.
  - Broadcasting: multi-zone output shape and per-column consistency.
  - Pitch-angle-averaged path: smoke tests (runs, finite, correct units).
"""

import numpy as np
import pytest
from astropy import constants as consts
from astropy import units as u

from trilobite.radiation.synchrotron.SEDs.numerical.core import NumericalSynchrotronEngine
from trilobite.radiation.synchrotron.utils import c_1_cgs, compute_c5_parameter, compute_c6_parameter
from trilobite.radiation.synchrotron.electron_distributions import PowerLaw

# ============================================ #
# TEST CONFIGURATION                           #
# ============================================ #
_M_E_C2_CGS = consts.m_e.cgs.value * consts.c.cgs.value**2  # erg

# Reference parameters for power-law fidelity tests
_N_0 = 1e5  # cm^{-3}
_B_CGS = 0.1  # G
_ALPHA = np.pi / 2  # rad  (sin alpha = 1)
_GAMMA_MIN = 1e2
_GAMMA_MAX = 1e7
_N_GAMMA = 500

# Test frequencies: well within the power-law regime (at least a decade inside both
# cutoff frequencies), avoiding contributions from the gamma_min / gamma_max edges.
_NU_TEST = np.geomspace(1e12, 1e15, 20)  # Hz
_NU_PLOT = np.geomspace(1e10, 1e17, 200)  # Hz (broader range for diagnostic plots)

# ================================================== #
# Helpers                                            #
# ================================================== #
# These are analytic formulations of the emissivity and the absorption coefficient for
# a power-law distribution which we can use to verify the numerical engine's results.


def _analytic_emissivity(p, nu, pitch_average=False):
    """Pacholczyk (1970) analytic emissivity for a power-law electron distribution."""
    c5 = compute_c5_parameter(p, pitch_average=pitch_average)
    if pitch_average:
        return c5 * _N_0 * _M_E_C2_CGS ** (p - 1) * _B_CGS ** ((p + 1) / 2) * (nu / (2 * c_1_cgs)) ** (-(p - 1) / 2)
    else:
        sin_alpha = np.sin(_ALPHA)
        return (
            c5
            * _N_0
            * _M_E_C2_CGS ** (p - 1)
            * (_B_CGS * sin_alpha) ** ((p + 1) / 2)
            * (nu / (2 * c_1_cgs)) ** (-(p - 1) / 2)
        )


def _analytic_absorption(p, nu, pitch_average=False):
    """Pacholczyk (1970) analytic SSA coefficient for a power-law electron distribution."""
    c6 = compute_c6_parameter(p, pitch_average=pitch_average)
    if pitch_average:
        return c6 * _N_0 * _M_E_C2_CGS ** (p - 1) * _B_CGS ** ((p + 2) / 2) * (nu / (2 * c_1_cgs)) ** (-(p + 4) / 2)
    else:
        sin_alpha = np.sin(_ALPHA)
        return (
            c6
            * _N_0
            * _M_E_C2_CGS ** (p - 1)
            * (_B_CGS * sin_alpha) ** ((p + 2) / 2)
            * (nu / (2 * c_1_cgs)) ** (-(p + 4) / 2)
        )


def _analytic_source_function(p, nu, pitch_average=False):
    """Analytic source function S_nu = j_nu / alpha_nu ~ B^{-1/2} (nu/2c1)^{5/2}."""
    c5 = compute_c5_parameter(p, pitch_average=pitch_average)
    c6 = compute_c6_parameter(p, pitch_average=pitch_average)
    if pitch_average:
        return (c5 / c6) * _B_CGS ** (-0.5) * (nu / (2 * c_1_cgs)) ** (5 / 2)
    else:
        sin_alpha = np.sin(_ALPHA)
        return (c5 / c6) * (_B_CGS * sin_alpha) ** (-0.5) * (nu / (2 * c_1_cgs)) ** (5 / 2)


def _doppler_factor(beta, cos_theta):
    """Relativistic Doppler factor D = 1 / (Gamma * (1 - beta*cos_theta))."""
    Gamma = 1.0 / np.sqrt(1 - beta**2)
    return 1.0 / (Gamma * (1 - beta * cos_theta))


def _make_power_law(p):
    """Return a power-law electron distribution callable with the module-level reference parameters."""
    return PowerLaw.as_callable(norm=_N_0, p=p, gamma_min=_GAMMA_MIN, gamma_max=_GAMMA_MAX)


# ================================================== #
# Fixtures                                           #
# ================================================== #
# In the fixtures, we provide a common setup for the NumericalSynchrotronEngine and the test parameters, so that
# individual test functions can focus on specific aspects without worrying about the setup. We also provide
# a fresh instance of the engine to test operations requiring a fresh engine.


@pytest.fixture(scope="session")
def engine():
    """Session-scoped engine with both kernels loaded at high resolution."""
    eng = NumericalSynchrotronEngine()
    eng.load_first_kernel(x_min=1e-6, x_max=1e3, num_points=1000)
    eng.load_avg_first_kernel(x_min=1e-6, x_max=1e3, num_points=1000)
    return eng


@pytest.fixture
def fresh_engine():
    """Function-scoped engine with no kernels loaded."""
    return NumericalSynchrotronEngine()


# ================================================== #
# Section 1: Lifecycle / Programmatic Tests          #
# ================================================== #


class TestLifecycle:
    """Engine state management, guard rails, dunder methods, and load options."""

    # --- State flags and __len__ ---

    def test_fresh_engine_has_no_kernels(self, fresh_engine):
        assert not fresh_engine.is_first_kernel_loaded
        assert not fresh_engine.is_avg_first_kernel_loaded
        assert len(fresh_engine) == 0

    def test_load_first_kernel_sets_state(self, fresh_engine):
        fresh_engine.load_first_kernel()
        assert fresh_engine.is_first_kernel_loaded
        assert not fresh_engine.is_avg_first_kernel_loaded
        assert len(fresh_engine) == 1

    def test_load_avg_first_kernel_sets_state(self, fresh_engine):
        fresh_engine.load_avg_first_kernel()
        assert not fresh_engine.is_first_kernel_loaded
        assert fresh_engine.is_avg_first_kernel_loaded
        assert len(fresh_engine) == 1

    def test_load_both_kernels(self, fresh_engine):
        fresh_engine.load_first_kernel()
        fresh_engine.load_avg_first_kernel()
        assert fresh_engine.is_first_kernel_loaded
        assert fresh_engine.is_avg_first_kernel_loaded
        assert len(fresh_engine) == 2

    def test_clear_resets_state(self, fresh_engine):
        fresh_engine.load_first_kernel()
        fresh_engine.load_avg_first_kernel()
        fresh_engine.clear()
        assert not fresh_engine.is_first_kernel_loaded
        assert not fresh_engine.is_avg_first_kernel_loaded
        assert len(fresh_engine) == 0

    # --- Guard rails ---

    def test_guard_rail_first_kernel_raises_before_load(self, fresh_engine):
        with pytest.raises(RuntimeError, match="load_first_kernel"):
            fresh_engine.ensure_first_kernel_loaded()

    def test_guard_rail_avg_first_kernel_raises_before_load(self, fresh_engine):
        with pytest.raises(RuntimeError, match="load_avg_first_kernel"):
            fresh_engine.ensure_avg_first_kernel_loaded()

    def test_compute_emissivity_requires_first_kernel_for_fixed_alpha(self, fresh_engine):
        N = PowerLaw.as_callable(norm=_N_0, p=3.0, gamma_min=_GAMMA_MIN, gamma_max=_GAMMA_MAX)
        with pytest.raises(RuntimeError):
            fresh_engine.compute_emissivity(1e10 * u.Hz, 0.1 * u.G, N, alpha=np.pi / 2)

    def test_compute_emissivity_requires_avg_kernel_for_pa_path(self, fresh_engine):
        N = PowerLaw.as_callable(norm=_N_0, p=3.0, gamma_min=_GAMMA_MIN, gamma_max=_GAMMA_MAX)
        with pytest.raises(RuntimeError):
            fresh_engine.compute_emissivity(1e10 * u.Hz, 0.1 * u.G, N, alpha=None)

    # --- Dunder methods ---

    def test_repr_is_string_and_contains_class_name(self, fresh_engine):
        r = repr(fresh_engine)
        assert isinstance(r, str)
        assert "NumericalSynchrotronEngine" in r

    def test_str_is_string_and_contains_class_name(self, fresh_engine):
        s = str(fresh_engine)
        assert isinstance(s, str)
        assert "NumericalSynchrotronEngine" in s

    def test_context_manager_clears_on_exit(self):
        with NumericalSynchrotronEngine() as eng:
            eng.load_first_kernel()
            eng.load_avg_first_kernel()
            assert len(eng) == 2
        assert not eng.is_first_kernel_loaded
        assert not eng.is_avg_first_kernel_loaded

    # --- load_first_kernel options ---

    def test_load_first_kernel_log_spacing(self, fresh_engine):
        fresh_engine.load_first_kernel(spacing="log")
        assert fresh_engine.is_first_kernel_loaded

    def test_load_first_kernel_linear_spacing(self, fresh_engine):
        fresh_engine.load_first_kernel(spacing="linear")
        assert fresh_engine.is_first_kernel_loaded

    def test_load_first_kernel_explicit_x_array(self, fresh_engine):
        x = np.geomspace(1e-5, 1e2, 200)
        fresh_engine.load_first_kernel(x=x)
        assert fresh_engine.is_first_kernel_loaded
        assert fresh_engine.first_kernel_size == 200

    def test_load_first_kernel_invalid_spacing_raises(self, fresh_engine):
        with pytest.raises(ValueError, match="spacing"):
            fresh_engine.load_first_kernel(spacing="invalid")

    # --- load_avg_first_kernel options ---

    def test_load_avg_first_kernel_linear_spacing(self, fresh_engine):
        fresh_engine.load_avg_first_kernel(spacing="linear")
        assert fresh_engine.is_avg_first_kernel_loaded

    def test_load_avg_first_kernel_invalid_spacing_raises(self, fresh_engine):
        with pytest.raises(ValueError, match="spacing"):
            fresh_engine.load_avg_first_kernel(spacing="invalid")


# ================================================== #
# PHYSICS TESTS                                      #
# ================================================== #
# These tests verify that the engine's physics is actually correct by comparing to known analytical
# cases.
class TestPhysics:
    """
    Performs tests of physical accuracy on the emissivity and absorption coefficient calculations,
    comparing to analytical results in the literature.

    In these tests, we use the known solutions from the literature for power-law distributions of electrons
    to ensure that we are correctly computing the synchrotron emissivity and absorption coefficients.
    We test both the fixed pitch angle case (using the first kernel) and the
    pitch-angle averaged case (using the average kernel).
    """

    @pytest.mark.parametrize("p", [2.5, 3.0, 3.5])
    @pytest.mark.parametrize("pitch_average", [False, True])
    def test_emissivity(self, engine, p, pitch_average, diagnostic_plots, diagnostic_plots_dir):
        """Numerical emissivity must match the Pacholczyk (1970) analytic result within 1%.

        The analytic formula is:

            j_nu = c_5(p) * N_0 * (m_e c^2)^(p-1) * (B sin α)^((p+1)/2) * (nu / 2c_1)^(-(p-1)/2)

        For the pitch-angle-averaged case (pitch_average=True), sin α is absorbed into c_5 and
        (B sin α) is replaced by B alone.

        The test frequencies (_NU_TEST) are chosen well within the power-law regime — at least
        one decade inside both the gamma_min and gamma_max cutoff frequencies — so the analytic
        formula is valid and the comparison is clean.
        """
        N = _make_power_law(p)
        alpha = None if pitch_average else _ALPHA

        j_num = engine.compute_emissivity(
            _NU_TEST * u.Hz,
            _B_CGS * u.G,
            N,
            alpha=alpha,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        )
        j_analytic = _analytic_emissivity(p, _NU_TEST, pitch_average=pitch_average)

        np.testing.assert_allclose(j_num.value, j_analytic, rtol=1e-2)

        if diagnostic_plots:
            import matplotlib.pyplot as plt

            j_plot = engine.compute_emissivity(
                _NU_PLOT * u.Hz,
                _B_CGS * u.G,
                N,
                alpha=alpha,
                gamma_min=_GAMMA_MIN,
                gamma_max=_GAMMA_MAX,
                n_gamma=_N_GAMMA,
            )
            j_plot_analytic = _analytic_emissivity(p, _NU_PLOT, pitch_average=pitch_average)

            label_suffix = "avg" if pitch_average else "fixed"
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.loglog(_NU_PLOT, j_plot.value, label="Numerical")
            ax1.loglog(_NU_PLOT, j_plot_analytic, "--", label=r"Analytic $c_5(p)$")
            ax1.axvspan(_NU_TEST.min(), _NU_TEST.max(), color="k", alpha=0.1, label="Test region")
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel(r"$j_\nu$ (erg s$^{-1}$ cm$^{-3}$ Hz$^{-1}$ sr$^{-1}$)")
            ax1.set_title(f"Emissivity (p={p}, {label_suffix})")
            ax1.legend()

            ax2.semilogx(_NU_PLOT, (j_plot.value - j_plot_analytic) / j_plot_analytic)
            ax2.axvspan(_NU_TEST.min(), _NU_TEST.max(), color="k", alpha=0.1, label="Test region")
            ax2.axhline(0, color="k", ls="--")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Fractional residual")
            ax2.set_title(f"Residuals (p={p}, {label_suffix})")
            ax2.legend()

            plt.tight_layout()
            plt.savefig(diagnostic_plots_dir / f"emissivity_vs_analytic_p{p}_{label_suffix}.png")
            plt.close()

    @pytest.mark.parametrize("p", [2.5, 3.0, 3.5])
    @pytest.mark.parametrize("pitch_average", [False, True])
    def test_absorption(self, engine, p, pitch_average, diagnostic_plots, diagnostic_plots_dir):
        """Numerical SSA absorption coefficient must match the Pacholczyk (1970) analytic result within 1%.

        The analytic formula is:

            alpha_nu = c_6(p) * N_0 * (m_e c^2)^(p-1) * (B sin α)^((p+2)/2) * (nu / 2c_1)^(-(p+4)/2)

        For the pitch-angle-averaged case (pitch_average=True), sin α is absorbed into c_6 and
        (B sin α) is replaced by B alone.

        The test frequencies (_NU_TEST) are chosen well within the power-law regime — at least
        one decade inside both the gamma_min and gamma_max cutoff frequencies — so the analytic
        formula is valid and the comparison is clean.
        """
        N = _make_power_law(p)
        alpha = None if pitch_average else _ALPHA

        alpha_num = engine.compute_absorption(
            _NU_TEST * u.Hz,
            _B_CGS * u.G,
            N,
            alpha=alpha,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        )
        alpha_analytic = _analytic_absorption(p, _NU_TEST, pitch_average=pitch_average)

        np.testing.assert_allclose(alpha_num.value, alpha_analytic, rtol=1e-2)

        if diagnostic_plots:
            import matplotlib.pyplot as plt

            a_plot = engine.compute_absorption(
                _NU_PLOT * u.Hz,
                _B_CGS * u.G,
                N,
                alpha=alpha,
                gamma_min=_GAMMA_MIN,
                gamma_max=_GAMMA_MAX,
                n_gamma=_N_GAMMA,
            )
            a_plot_analytic = _analytic_absorption(p, _NU_PLOT, pitch_average=pitch_average)

            label_suffix = "avg" if pitch_average else "fixed"
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.loglog(_NU_PLOT, a_plot.value, label="Numerical")
            ax1.loglog(_NU_PLOT, a_plot_analytic, "--", label=r"Analytic $c_6(p)$")
            ax1.axvspan(_NU_TEST.min(), _NU_TEST.max(), color="k", alpha=0.1, label="Test region")
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel(r"$\alpha_\nu$ (cm$^{-1}$)")
            ax1.set_title(f"Absorption coefficient (p={p}, {label_suffix})")
            ax1.legend()

            ax2.semilogx(_NU_PLOT, (a_plot.value - a_plot_analytic) / a_plot_analytic)
            ax2.axvspan(_NU_TEST.min(), _NU_TEST.max(), color="k", alpha=0.1, label="Test region")
            ax2.axhline(0, color="k", ls="--")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Fractional residual")
            ax2.set_title(f"Residuals (p={p}, {label_suffix})")
            ax2.legend()

            plt.tight_layout()
            plt.savefig(diagnostic_plots_dir / f"absorption_vs_analytic_p{p}_{label_suffix}.png")
            plt.close()

    @pytest.mark.parametrize("p", [2.5, 3.0, 3.5])
    @pytest.mark.parametrize("pitch_average", [False, True])
    def test_source_function(self, engine, p, pitch_average, diagnostic_plots, diagnostic_plots_dir):
        """Numerical source function S_nu = j_nu / alpha_nu must match the analytic ratio within 1%.

        For a power-law distribution, c_5 and c_6 cancel their p-dependence in the ratio, giving:

            S_nu = (c_5 / c_6) * (B sin α)^(-1/2) * (nu / 2c_1)^(5/2)

        For the pitch-angle-averaged case (pitch_average=True), sin α factors are absorbed into c_5
        and c_6 and B replaces (B sin α).

        This is the Rayleigh-Jeans source function for synchrotron radiation and is independent
        of N_0 and p, providing a consistency check that is orthogonal to the emissivity and
        absorption tests above.
        """
        N = _make_power_law(p)
        alpha = None if pitch_average else _ALPHA

        S_num = engine.compute_source_function(
            _NU_TEST * u.Hz,
            _B_CGS * u.G,
            N,
            alpha=alpha,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        )
        S_analytic = _analytic_source_function(p, _NU_TEST, pitch_average=pitch_average)

        np.testing.assert_allclose(S_num.value, S_analytic, rtol=1e-2)

        if diagnostic_plots:
            import matplotlib.pyplot as plt

            S_plot = engine.compute_source_function(
                _NU_PLOT * u.Hz,
                _B_CGS * u.G,
                N,
                alpha=alpha,
                gamma_min=_GAMMA_MIN,
                gamma_max=_GAMMA_MAX,
                n_gamma=_N_GAMMA,
            )
            S_plot_analytic = _analytic_source_function(p, _NU_PLOT, pitch_average=pitch_average)

            label_suffix = "avg" if pitch_average else "fixed"
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.loglog(_NU_PLOT, S_plot.value, label="Numerical")
            ax1.loglog(_NU_PLOT, S_plot_analytic, "--", label=r"Analytic $\nu^{5/2}$")
            ax1.axvspan(_NU_TEST.min(), _NU_TEST.max(), color="k", alpha=0.1, label="Test region")
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel(r"$S_\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$)")
            ax1.set_title(f"Source function (p={p}, {label_suffix})")
            ax1.legend()

            ax2.semilogx(_NU_PLOT, (S_plot.value - S_plot_analytic) / S_plot_analytic)
            ax2.axvspan(_NU_TEST.min(), _NU_TEST.max(), color="k", alpha=0.1, label="Test region")
            ax2.axhline(0, color="k", ls="--")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Fractional residual")
            ax2.set_title(f"Residuals (p={p}, {label_suffix})")
            ax2.legend()

            plt.tight_layout()
            plt.savefig(diagnostic_plots_dir / f"source_function_vs_analytic_p{p}_{label_suffix}.png")
            plt.close()

    @pytest.mark.parametrize("p", [2.5, 3.0, 3.5])
    @pytest.mark.parametrize("pitch_average", [False, True])
    def test_ssa_turnover_frequency(self, engine, p, pitch_average, diagnostic_plots, diagnostic_plots_dir):
        """The absorption coefficient at the analytic nu_a must give optical depth tau = 1."""

        # Configure the test parameters for this case.
        N_0_ssa = 1e4  # cm^{-3}, elevated density to shift nu_a into a testable range
        ell_ssa = 1e16  # cm
        gamma_min_ssa = 1.0
        gamma_max_ssa = 1e8
        alpha = None if pitch_average else _ALPHA

        # Compute the EXPECTED nu_a from the analytic formula for the given parameters.
        c6 = compute_c6_parameter(p, pitch_average=pitch_average)
        if pitch_average:
            nu_a = (
                2
                * c_1_cgs
                * (c6 * N_0_ssa * _M_E_C2_CGS ** (p - 1) * _B_CGS ** ((p + 2) / 2) * ell_ssa) ** (2 / (p + 4))
            )
        else:
            nu_a = (
                2
                * c_1_cgs
                * (c6 * N_0_ssa * _M_E_C2_CGS ** (p - 1) * (_B_CGS * np.sin(_ALPHA)) ** ((p + 2) / 2) * ell_ssa)
                ** (2 / (p + 4))
            )

        # Compute the engine optical depth at that frequency.
        N = PowerLaw.as_callable(norm=N_0_ssa, p=p, gamma_min=gamma_min_ssa, gamma_max=gamma_max_ssa)
        absorption_coefficient = engine.compute_absorption(
            nu_a * u.Hz,
            _B_CGS * u.G,
            N,
            alpha=alpha,
            gamma_min=gamma_min_ssa,
            gamma_max=gamma_max_ssa,
            n_gamma=_N_GAMMA,
        )
        tau = absorption_coefficient.cgs.value * ell_ssa

        # Verify that tau is accurate.
        np.testing.assert_allclose(tau, 1.0, rtol=1e-2)

        if diagnostic_plots:
            import matplotlib.pyplot as plt

            label_suffix = "avg" if pitch_average else "fixed"
            nu_plot = np.geomspace(nu_a / 100, nu_a * 100, 300)
            alpha_plot = engine.compute_absorption(
                nu_plot * u.Hz,
                _B_CGS * u.G,
                N,
                alpha=alpha,
                gamma_min=gamma_min_ssa,
                gamma_max=_GAMMA_MAX,
                n_gamma=_N_GAMMA,
            )
            tau_plot = (alpha_plot * ell_ssa * u.cm).to(u.dimensionless_unscaled).value

            plt.figure()
            plt.loglog(nu_plot, tau_plot, label=r"$\tau_\nu$ (numerical)")
            plt.axvline(nu_a, color="r", ls="--", label=r"Analytic $\nu_a$")
            plt.axhline(1.0, color="k", ls=":", label=r"$\tau = 1$")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel(r"$\tau_\nu = \alpha_\nu \ell$")
            plt.title(f"SSA optical depth (p={p}, {label_suffix})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(diagnostic_plots_dir / f"ssa_turnover_p{p}_{label_suffix}.png")
            plt.close()

    def test_optically_thin_intensity_scales_linearly(self, engine):
        r"""
        Check that the transfer solution reduces to :math:`I_\nu \simeq j_\nu \ell`
        in the optically thin limit.

        For a uniform emitting and absorbing slab, the formal solution is

        .. math::

            I_\nu = S_\nu \left(1 - e^{-\tau_\nu}\right),

        where :math:`S_\nu = j_\nu / \alpha_\nu` and
        :math:`\tau_\nu = \alpha_\nu \ell`. In the optically thin limit,
        :math:`\tau_\nu \ll 1`, this becomes

        .. math::

            I_\nu \simeq S_\nu \tau_\nu = j_\nu \ell.

        Therefore, holding the emissivity fixed and doubling the path length should
        double the emergent specific intensity.
        """
        electron_distribution = _make_power_law(3.0)

        # Choose a short path length and high observing frequency so that the
        # synchrotron self-absorption optical depth is safely in the thin regime.
        nu = np.array([1.0e13]) * u.Hz
        ell_thin = 1.0e8 * u.cm

        intensity_1 = engine.compute_rest_frame_specific_intensity(
            nu,
            ell_thin,
            _B_CGS * u.G,
            electron_distribution,
            alpha=_ALPHA,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        )

        intensity_2 = engine.compute_rest_frame_specific_intensity(
            nu,
            2.0 * ell_thin,
            _B_CGS * u.G,
            electron_distribution,
            alpha=_ALPHA,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        )

        # In the optically thin limit, I_nu is linear in ell.
        np.testing.assert_allclose(
            intensity_2.value,
            2.0 * intensity_1.value,
            rtol=1.0e-3,
        )

    def test_optically_thick_intensity_saturates(self, engine):
        r"""
        Check that the transfer solution saturates to :math:`I_\nu \simeq S_\nu`
        in the optically thick limit.

        For a uniform emitting and absorbing slab, the formal solution is

        .. math::

            I_\nu = S_\nu \left(1 - e^{-\tau_\nu}\right),

        where :math:`S_\nu = j_\nu / \alpha_\nu` and
        :math:`\tau_\nu = \alpha_\nu \ell`. In the optically thick limit,
        :math:`\tau_\nu \gg 1`, the exponential term vanishes and

        .. math::

            I_\nu \rightarrow S_\nu.

        Therefore, once the slab is sufficiently optically thick, increasing the
        path length should no longer change the emergent specific intensity.
        """

        def dense_electron_distribution(gamma):
            """Dense power-law electron distribution used to force large SSA optical depth."""
            return 1.0e10 * gamma ** (-3.0)

        # Choose a low observing frequency, high electron density, and long path
        # length so that synchrotron self-absorption places the slab deep in the
        # optically thick regime.
        nu = np.array([1.0e9]) * u.Hz
        ell_thick = 1.0e17 * u.cm

        intensity_1 = engine.compute_rest_frame_specific_intensity(
            nu,
            ell_thick,
            _B_CGS * u.G,
            dense_electron_distribution,
            alpha=_ALPHA,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        )

        intensity_10 = engine.compute_rest_frame_specific_intensity(
            nu,
            10.0 * ell_thick,
            _B_CGS * u.G,
            dense_electron_distribution,
            alpha=_ALPHA,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        )

        # In the optically thick limit, I_nu is controlled by the source function
        # rather than by the physical path length through the slab.
        np.testing.assert_allclose(
            intensity_10.value,
            intensity_1.value,
            rtol=1.0e-3,
        )


# ================================================== #
# Section 3: Doppler / Redshift Tests                #
# ================================================== #
class TestDopplerRedshift:
    """Frame transformation tests."""

    def test_beta0_z0_observer_equals_rest_frame(self, engine):
        """At z=0, beta=0, compute_specific_intensity must equal compute_rest_frame_specific_intensity."""
        N = _make_power_law(3.0)
        nu = _NU_TEST * u.Hz
        ell = 1e15 * u.cm

        I_rf = engine.compute_rest_frame_specific_intensity(
            nu,
            ell,
            _B_CGS * u.G,
            N,
            alpha=_ALPHA,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        )
        I_obs = engine.compute_specific_intensity(
            nu,
            ell,
            _B_CGS * u.G,
            N,
            z=0,
            beta=0,
            theta=0,
            alpha=_ALPHA,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        )

        np.testing.assert_allclose(I_obs.value, I_rf.value, rtol=1e-10)

    def test_doppler_boosting_scales_as_D_cubed(self, engine):
        """I_obs(nu) = D^3 * I_rf(nu/D) for a relativistically moving source at z=0."""
        N = _make_power_law(3.0)
        ell = 1e15 * u.cm
        beta = 0.5
        theta = 0.0  # observer along the direction of motion
        D = _doppler_factor(beta, np.cos(theta))

        nu_obs = _NU_TEST * u.Hz
        nu_rf = nu_obs / D  # comoving-frame frequency corresponding to nu_obs

        I_obs = engine.compute_specific_intensity(
            nu_obs,
            ell,
            _B_CGS * u.G,
            N,
            z=0,
            beta=beta,
            theta=theta,
            alpha=_ALPHA,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        )
        I_rf_at_nu_prime = engine.compute_rest_frame_specific_intensity(
            nu_rf,
            ell,
            _B_CGS * u.G,
            N,
            alpha=_ALPHA,
            gamma_min=_GAMMA_MIN,
            gamma_max=_GAMMA_MAX,
            n_gamma=_N_GAMMA,
        )

        np.testing.assert_allclose(I_obs.value, D**3 * I_rf_at_nu_prime.value, rtol=1e-10)


# ================================================== #
# Section 5: Broadcasting / Multi-Zone Shape Tests   #
# ================================================== #


class TestBroadcasting:
    """Multi-zone vectorized broadcasting over the zone axis."""

    def test_multi_zone_output_shape(self, engine):
        """N of shape (n_zones, n_gamma) with B array must produce output (n_nu, n_zones)."""
        n_zones = 5
        n_nu = 10
        nu = np.geomspace(1e11, 1e15, n_nu) * u.Hz
        B_arr = np.array([0.05, 0.08, 0.10, 0.15, 0.20]) * u.G

        gamma = np.geomspace(_GAMMA_MIN, _GAMMA_MAX, _N_GAMMA)
        N_batch = np.outer(np.ones(n_zones), _N_0 * gamma ** (-3.0))  # (n_zones, n_gamma)

        j = engine.compute_emissivity(nu, B_arr, N_batch, gamma=gamma, alpha=_ALPHA)

        assert j.shape == (n_nu, n_zones)

    def test_multi_zone_matches_scalar_calls(self, engine):
        """Each zone column in the batched result must match the corresponding scalar call."""
        n_zones = 3
        n_nu = 8
        nu = np.geomspace(1e12, 1e15, n_nu) * u.Hz
        B_values = [0.05, 0.10, 0.20]
        B_arr = np.array(B_values) * u.G

        gamma = np.geomspace(_GAMMA_MIN, _GAMMA_MAX, _N_GAMMA)
        N_batch = np.outer(np.ones(n_zones), _N_0 * gamma ** (-3.0))  # (n_zones, n_gamma)

        j_batch = engine.compute_emissivity(nu, B_arr, N_batch, gamma=gamma, alpha=_ALPHA)

        for i, B_scalar in enumerate(B_values):
            j_scalar = engine.compute_emissivity(
                nu,
                B_scalar * u.G,
                _N_0 * gamma ** (-3.0),
                gamma=gamma,
                alpha=_ALPHA,
            )
            np.testing.assert_allclose(j_batch[:, i].value, j_scalar.value, rtol=1e-12)


class TestPairedFrequencies:
    """``paired=True`` evaluates each zone at its own frequency, instead of every frequency in every zone."""

    @staticmethod
    def _zones(n_zones=7):
        rng = np.random.default_rng(3)
        gamma = np.geomspace(_GAMMA_MIN, _GAMMA_MAX, 60)
        log_gamma = np.log(gamma)
        log_weights = np.log(np.gradient(log_gamma) * np.ones_like(log_gamma))
        log_B = np.log(rng.uniform(0.05, 0.3, n_zones))
        log_ell = np.log(rng.uniform(1e15, 1e16, n_zones))
        log_A = np.log(rng.uniform(1e31, 1e33, n_zones))
        # a different spectrum in every zone, with a hard cutoff so that log N = -inf occurs
        log_N = np.log(np.outer(rng.uniform(1e-3, 1.0, n_zones), gamma ** (-3.0)) * (gamma < 3e3))
        log_nu = np.log(rng.uniform(1e9, 1e12, n_zones))
        return log_nu, log_ell, log_B, log_N, log_gamma, log_weights, log_A

    def test_flux_density_matches_the_diagonal_of_the_outer_product(self, engine):
        log_nu, log_ell, log_B, log_N, log_gamma, log_weights, log_A = self._zones()
        log_D_A = np.log(1e26)
        with np.errstate(divide="ignore"):
            paired = engine._compute_log_pa_flux_density(
                log_nu, log_ell, log_B, log_N, log_gamma, log_weights, log_A, log_D_A, z=0.1, paired=True
            )
            full = engine._compute_log_pa_flux_density(
                log_nu, log_ell, log_B, log_N, log_gamma, log_weights, log_A, log_D_A, z=0.1
            )
        assert paired.shape == log_nu.shape
        assert full.shape == (log_nu.size, log_nu.size)  # (n_nu, n_zones)
        assert np.array_equal(paired, np.diagonal(full))

    def test_scalar_frequency_is_broadcast_over_the_zones(self, engine):
        log_nu, log_ell, log_B, log_N, log_gamma, log_weights, log_A = self._zones()
        log_D_A = np.log(1e26)
        with np.errstate(divide="ignore"):
            paired = engine._compute_log_pa_flux_density(
                np.log(5e10), log_ell, log_B, log_N, log_gamma, log_weights, log_A, log_D_A, paired=True
            )
            full = engine._compute_log_pa_flux_density(
                np.log(5e10), log_ell, log_B, log_N, log_gamma, log_weights, log_A, log_D_A
            )
        assert np.array_equal(paired, full)

    def test_emissivity_and_absorption_paired(self, engine):
        log_nu, _, log_B, log_N, log_gamma, log_weights, _ = self._zones()
        with np.errstate(divide="ignore"):
            j_p, a_p = engine._compute_log_pa_emissivity_and_absorption(
                log_nu, log_B, log_N, log_gamma, log_weights, paired=True
            )
            j, a = engine._compute_log_pa_emissivity_and_absorption(log_nu, log_B, log_N, log_gamma, log_weights)
        assert np.array_equal(j_p, np.diagonal(j)) and np.array_equal(a_p, np.diagonal(a))
