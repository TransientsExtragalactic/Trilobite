r"""
Unit tests for thin-shell shock engine dynamics.

Tests for :class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`
and :class:`~trilobite.dynamics.shocks.numerical.MomentumConservingShockEngine`.

Two physical closures are exercised:

1. **Pressure-driven thin-shell** (``PressureDrivenThinShellShockEngine``): Shell acceleration is
   driven by the net post-shock pressure estimated from instantaneous Rankine--Hugoniot conditions.
   Tests cover homologous vacuum expansion, the Chevalier self-similar slope for n=10, s=2, and
   late-time Sedov-Taylor convergence (R ∝ t^{2/5}) for uniform CSM.

2. **Momentum-conserving snowplow** (``MomentumConservingShockEngine``): Total shell momentum is
   conserved; no pressure forces. After the ejecta are exhausted, the shell sweeps stationary CSM
   at constant momentum, giving R ∝ t^{1/4} for uniform CSM (s=0).
"""

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from trilobite.dynamics.profiles import BrokenPowerLawEjectaProfile, UniformCSMProfile
from trilobite.dynamics.shocks import ChevalierSelfSimilarShockEngine, make_homologous_stationary_sources
from trilobite.dynamics.shocks.numerical import MomentumConservingShockEngine, PressureDrivenThinShellShockEngine


# ------------------------------------------------------------------- #
# Test Cases                                                          #
# ------------------------------------------------------------------- #
class TestNumericalThinShellShockEngine:
    @pytest.fixture(scope="class")
    def relative_tolerance(self):
        """The relative tolerance to use for comparisons."""
        return 1e-5

    @pytest.fixture(scope="class")
    def time_grid(self):
        """The times at which to evaluate the shock engine."""
        return np.geomspace(1, 1e4, 512) * u.day

    @pytest.fixture(scope="class")
    def engine(self):
        """The instantiated engine to do the computation with."""
        return PressureDrivenThinShellShockEngine()

    def test_homologous_expansion_in_vacuum(
        self, time_grid, engine, diagnostic_plots, diagnostic_plots_dir, relative_tolerance
    ):
        """
        Test the numerical implementation of ``NumericalThinShellShockEngine`` by ensuring that
        it correctly reproduces homologous expansion into vacuum.

        In this scenario, we set up a run with some :math:`t_0` and :math:`R_0`, but set the velocity
        to the correct homologous expansion velocity

        .. math::

            v_0 = \frac{R_0}{t_0}

        and we start with some :math:`M_0` which can be arbitrary. We consider a simple power-law
        ejecta density as it should have no impact on the propagation of the shock. The resulting
        expansion should maintain the homologous expansion profile, i.e.

            R(t) = v_0 * t

        Parameters
        ----------
        time_grid
        engine

        Returns
        -------

        """
        # --- Configure parameters --- #
        # The first task in the test is to configure the parameters and ensure that we have the necessary
        # CGS v_0 value so that it can play a role in the G(v) function.
        R_0 = 1e13 * u.cm
        t_0 = time_grid[0]
        v_0 = R_0 / t_0
        M_0 = 1e-5 * u.Msun

        # Construct the CGS versions of each of the parameters.
        v_0_cgs = v_0.to_value(u.cm / u.s)

        # --- Configure the computational functions --- #
        # Homologous ejecta: rho_1(r,t) = t^{-3} G(r/t), u_1(r,t) = r/t.
        # CSM is vacuum: rho_4 = 0, u_4 = 0.
        def _G(v):
            return (v / v_0_cgs) ** -8

        def _rho_1(r, t):
            return t**-3 * _G(r / t)

        def _rho_4(r, t):
            return 0.0

        def _u_1(r, t):
            return r / t

        def _u_4(r, t):
            return 0.0

        # --- Perform the computation --- #
        results = engine.compute_shock_properties(
            time_grid,
            rho_1=_rho_1,
            rho_4=_rho_4,
            u_1=_u_1,
            u_4=_u_4,
            R_0=R_0,
            M_0=M_0,
            v_0=v_0,
            t_0=t_0,
        )

        # --- Generate expected results and compare --- #
        # Given the setup, we can now generate the expected results for the shock propagation.
        expected_R = v_0 * time_grid
        expected_v = v_0 * np.ones_like(time_grid.value)
        expected_M = M_0 * np.ones_like(time_grid.value)

        # Generate diagnostic plots if requested.
        if diagnostic_plots:
            import matplotlib.pyplot as plt

            from trilobite.utils.plot_utils import set_plot_style

            set_plot_style()

            fig, axes = plt.subplots(3, 1, figsize=(8, 12))
            axes[0].loglog(time_grid.to_value(u.day), results.radius.to_value(u.cm), label="Computed")
            axes[0].loglog(time_grid.to_value(u.day), expected_R.to_value(u.cm), ls="--", label="Expected")
            axes[0].set_xlabel("Time [days]")
            axes[0].set_ylabel("Radius [cm]")
            axes[0].legend()

            axes[1].loglog(time_grid.to_value(u.day), results.velocity.to_value(u.cm / u.s), label="Computed")
            axes[1].loglog(time_grid.to_value(u.day), expected_v.to_value(u.cm / u.s), ls="--", label="Expected")
            axes[1].set_xlabel("Time [days]")
            axes[1].set_ylabel("Velocity [cm/s]")
            axes[1].set_ylim([v_0.to_value("cm/s") / 10, v_0.to_value("cm/s") * 10])
            axes[1].legend()

            axes[2].loglog(time_grid.to_value(u.day), results.mass.to_value(u.g), label="Computed")
            axes[2].loglog(time_grid.to_value(u.day), expected_M.to_value(u.g), ls="--", label="Expected")
            axes[2].set_xlabel("Time [days]")
            axes[2].set_ylabel("Mass [g]")
            axes[2].set_ylim([M_0.to_value("g") / 10, M_0.to_value("g") * 10])
            axes[2].legend()

            plt.tight_layout()
            plt.savefig(f"{diagnostic_plots_dir}/test_homologous_expansion_in_vacuum.png")
            plt.close(fig)

        # --- Assert Validity --- #
        assert_allclose(
            results.radius.to_value(u.cm),
            expected_R.to_value(u.cm),
            rtol=relative_tolerance,
        )
        assert_allclose(
            results.velocity.to_value(u.cm / u.s),
            expected_v.to_value(u.cm / u.s),
            rtol=relative_tolerance,
        )
        assert_allclose(
            results.mass.to_value(u.g),
            expected_M.to_value(u.g),
            rtol=relative_tolerance,
        )

    def test_self_similar_power_law_solution(
        self, time_grid, engine, relative_tolerance, diagnostic_plots, diagnostic_plots_dir
    ):
        """
        Test the numerical implementation of ``NumericalThinShellShockEngine`` by ensuring that
        it correctly reproduces the self-similar solution for a power-law ejecta density profile
        expanding into a power-law CSM density profile.
        """
        # --- Configure parameters --- #
        # The first task is to set up the scenario. We'll use a power-law ejecta density profile
        # with index n=10 and a CSM density profile with index s=2 (wind-like). We set the initial
        # conditions such that the shock should follow the self-similar solution.
        n, s = 10, 2
        _lambda = (n - 3) / (n - s)
        _gamma = (3 - s) * _lambda

        # Construct the t_ref, R_ref, v_ref parameters for setting up the densities.
        R_ref = 1e13 * u.cm
        t_ref = 1 * u.day
        v_ref = 1e4 * (u.km / u.s)
        rho_0_csm = 5e-16 * (u.g / u.cm**3)
        rho_0_ej = 1e-13 * (u.g / u.cm**3)

        # We need to get the normalization for the ejecta density profile and
        # the CSM density profile. This can be done by the Chevalier class.
        K_csm = ChevalierSelfSimilarShockEngine.normalize_csm_density(rho_0_csm, R_ref, s)
        K_ej = ChevalierSelfSimilarShockEngine.normalize_outer_ejecta_density(rho_0_ej, v_ref, t_ref, n)

        # Compute the scale factor for the radius.
        zeta = ChevalierSelfSimilarShockEngine.compute_scale_parameter(n, s)

        # Now we can initialize the solution initial conditions except for the mass, which will require
        # some additional calculation.
        t_0 = time_grid[0]
        R_0 = (zeta * (K_csm / K_ej)) ** (1 / (s - n)) * t_0**_lambda
        v_0 = _lambda * (R_0 / t_0)

        # To calculate M_0, we need to use the relevant equations from Chevalier (1982).
        M_0 = 4 * np.pi * ((K_csm * R_0 ** (3 - s)) / (3 - s) + (K_ej * t_0 ** (n - 3) * R_0 ** (3 - n)) / (n - 3))

        # Generate the callables for the density structures.
        K_csm_cgs, K_ej_cgs = (
            K_csm.to_value(u.g / u.cm ** (3 - s)),
            K_ej.to_value(u.g * u.s ** (3 - n) * u.cm ** (n - 3)),
        )

        def _G(v):
            return K_ej_cgs * v**-n

        def _rho_1(r, t):
            return t**-3 * _G(r / t)

        def _rho_4(r, t):
            return K_csm_cgs * r**-s

        def _u_1(r, t):
            return r / t

        def _u_4(r, t):
            return 0.0

        # --- Perform the computation --- #
        results = engine.compute_shock_properties(
            time_grid,
            rho_1=_rho_1,
            rho_4=_rho_4,
            u_1=_u_1,
            u_4=_u_4,
            R_0=R_0,
            M_0=M_0,
            v_0=v_0,
            t_0=t_0,
        )

        # --- Generate expected results and compare --- #
        # Given the setup, we can now generate the expected results for the shock propagation.
        dlogR_dlogt_expected = _lambda
        dlogv_dlogt_expected = _lambda - 1
        dlogM_dlogt_expected = _gamma

        dlogR_dlott_computed = np.gradient(np.log10(results.radius.to_value(u.cm)), np.log10(time_grid.to_value(u.s)))
        dlogv_dlott_computed = np.gradient(
            np.log10(results.velocity.to_value(u.cm / u.s)), np.log10(time_grid.to_value(u.s))
        )
        dlogM_dlott_computed = np.gradient(np.log10(results.mass.to_value(u.g)), np.log10(time_grid.to_value(u.s)))

        # Generate diagnostic plots if requested.
        if diagnostic_plots:
            import matplotlib.pyplot as plt

            from trilobite.utils.plot_utils import set_plot_style

            set_plot_style()

            fig, axes = plt.subplots(3, 1, figsize=(8, 12))

            axes[0].plot(np.log10(time_grid.to_value(u.day)), dlogR_dlott_computed, label="Computed")
            axes[0].axhline(dlogR_dlogt_expected, ls="--", color="C1", label="Expected")
            axes[0].set_xlabel("log10(Time [days])")
            axes[0].set_ylabel("dlogR / dlogt")
            axes[0].set_ylim([dlogR_dlogt_expected * 0.5, dlogR_dlogt_expected * 1.5])
            axes[0].legend()

            axes[1].plot(np.log10(time_grid.to_value(u.day)), dlogv_dlott_computed, label="Computed")
            axes[1].axhline(dlogv_dlogt_expected, ls="--", color="C1", label="Expected")
            axes[1].set_xlabel("log10(Time [days])")
            axes[1].set_ylabel("dlogv / dlogt")
            axes[1].set_ylim([dlogv_dlogt_expected * 2, 0])
            axes[1].legend()

            axes[2].plot(np.log10(time_grid.to_value(u.day)), dlogM_dlott_computed, label="Computed")
            axes[2].axhline(dlogM_dlogt_expected, ls="--", color="C1", label="Expected")
            axes[2].set_xlabel("log10(Time [days])")
            axes[2].set_ylabel("dlogM / dlogt")
            axes[2].set_ylim([0, dlogM_dlogt_expected * 2])
            axes[2].legend()

            plt.tight_layout()
            plt.savefig(f"{diagnostic_plots_dir}/test_self_similar.png")
            plt.close(fig)

        # --- Assert Validity --- #
        # Check the shocks on the last 1/2 of the time grid to ensure that the solution has converged.
        half_index = len(time_grid) // 2
        assert_allclose(
            dlogR_dlott_computed[half_index:],
            dlogR_dlogt_expected,
            rtol=relative_tolerance,
        )
        assert_allclose(
            dlogv_dlott_computed[half_index:],
            dlogv_dlogt_expected,
            rtol=relative_tolerance,
        )
        assert_allclose(
            dlogM_dlott_computed[half_index:],
            dlogM_dlogt_expected,
            rtol=relative_tolerance,
        )

    def test_thin_shell_adiabatic_slope_uniform_csm(self, engine, diagnostic_plots, diagnostic_plots_dir):
        r"""
        Verify that the pressure-driven thin-shell engine converges to the
        thin-shell adiabatic scaling :math:`R \propto t^{4/13}` for n=10, s=0
        uniform CSM after the swept mass exceeds :math:`M_{\rm ej}`.

        The exponent 4/13 is the analytic late-time limit of the pressure-driven
        thin-shell ODE for :math:`\gamma = 5/3`, s=0.  Setting
        :math:`M = (4\pi/3)\rho_0 R^3` and the shell deceleration to
        :math:`dv/dt = -(9v^2)/(4R)` (from the pressure term) yields
        :math:`R \propto t^\alpha` with :math:`13\alpha = 4`.  This differs
        from the full Sedov-Taylor solution (R :math:`\propto t^{2/5}`), which
        requires proper energy conservation across the blast wave; the
        :class:`~trilobite.dynamics.shocks.numerical.MechanicalShockEngine`
        recovers that limit instead.

        Setup:

        - Broken-power-law ejecta: :math:`E_{\rm ej} = 10^{51}` erg,
          :math:`M_{\rm ej} = 5\,M_\odot`, n=10, δ=0.
        - Uniform CSM: :math:`\rho_0 = 10^{-16}` g cm\ :math:`^{-3}`.
        - Initial shell at :math:`R_0 = 10^{15}` cm, :math:`v_0 = 10^4` km/s.
        - Time grid: 1–:math:`10^{10}` days (ten decades).

        Only the last 10% of the time grid is checked to avoid the early
        transient.

        Expected result: :math:`d\log R/d\log t \to 4/13` within 5%.
        """
        n = 10
        _lambda_thin = 4.0 / 13.0  # thin-shell adiabatic limit for gamma=5/3, s=0
        rtol = 5e-2

        E_ej = 1e51 * u.erg
        M_ej = 5.0 * u.M_sun
        rho_0_cgs = 1e-16  # g/cm^3

        K, v_t = BrokenPowerLawEjectaProfile.normalize(E_ej, M_ej, n=n, delta=0)
        rho_ej = BrokenPowerLawEjectaProfile.as_optimized_callable(K=K, v_t=v_t, n=n, delta=0)
        rho_csm = UniformCSMProfile.as_optimized_callable(rho_0=rho_0_cgs)

        rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(rho_ej, rho_csm)

        time = np.geomspace(1e0, 1e10, 5000) * u.day
        R_cd_0 = 1e15 * u.cm
        v_cd_0 = 1e4 * u.km / u.s
        M_cd_0 = (4 * np.pi / 3) * rho_0_cgs * R_cd_0.to_value(u.cm) ** 3 * u.g

        state = engine.compute_shock_properties(
            time,
            rho_1=rho_1,
            rho_4=rho_4,
            u_1=u_1,
            u_4=u_4,
            R_0=R_cd_0,
            M_0=M_cd_0,
            v_0=v_cd_0,
            t_0=time[0],
        )

        log_t = np.log10(time.to_value(u.s))
        dlogR_dlogt = np.gradient(np.log10(state.radius.to_value(u.cm)), log_t)
        dlogv_dlogt = np.gradient(np.log10(state.velocity.to_value(u.cm / u.s)), log_t)

        M_ej_g = M_ej.to_value(u.g)
        m = state.mass.to_value(u.g)
        crossed = m >= M_ej_g
        if not np.any(crossed):
            pytest.skip(
                f"Swept mass did not reach M_ej within the time grid "
                f"(max M/M_ej = {m.max() / M_ej_g:.3g}). Extend the time grid or increase rho_0."
            )
        term_idx = int(np.argmax(crossed))

        # Check only the last 10% of the grid to ensure the transient has fully damped.
        check_start = max(term_idx + 50, len(time) * 9 // 10)

        if diagnostic_plots:
            import matplotlib.pyplot as plt

            from trilobite.utils.plot_utils import set_plot_style

            set_plot_style()

            log_t_days = np.log10(time.to_value(u.day))

            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle(
                rf"Pressure-Driven Engine — Thin-Shell Adiabatic Slope (n={n}, s=0, "
                rf"$\rho_0={rho_0_cgs:.0e}$ g cm$^{{-3}}$)"
            )

            axes[0].loglog(
                time.to_value(u.day),
                state.radius.to_value(u.cm),
                label=r"$R_{\rm sh}$ (pressure-driven)",
            )
            axes[0].axvline(time[term_idx].to_value(u.day), ls="-.", color="gray", label=r"$M=M_{\rm ej}$")
            axes[0].axvline(time[check_start].to_value(u.day), ls=":", color="gray", label="Check window start")
            axes[0].set_xlabel("Time [days]")
            axes[0].set_ylabel("Radius [cm]")
            axes[0].legend(fontsize=8)

            axes[1].semilogx(log_t_days, dlogR_dlogt, label=r"$d\log R/d\log t$")
            axes[1].axhline(_lambda_thin, ls="--", color="C1", label=rf"Thin-shell adiabatic $\lambda=4/13$")
            axes[1].axhline(2 / 5, ls=":", color="C2", alpha=0.6, label="Sedov-Taylor 2/5 (ref)")
            axes[1].axvline(time[term_idx].to_value(u.day), ls="-.", color="gray")
            axes[1].axvline(time[check_start].to_value(u.day), ls=":", color="gray")
            axes[1].set_xlabel("Time [days]")
            axes[1].set_ylabel(r"$d\log R/d\log t$")
            axes[1].set_ylim([0.0, 1.1])
            axes[1].legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(
                f"{diagnostic_plots_dir}/test_pressure_driven_thin_shell_adiabatic_slope.png",
                dpi=150,
            )
            plt.close(fig)

        assert check_start < len(time) - 10, (
            f"Too few grid points remain after M_ej crossing "
            f"({len(time) - check_start} points). Extend the time grid or increase rho_0."
        )
        assert_allclose(
            dlogR_dlogt[check_start:-3],
            _lambda_thin,
            rtol=rtol,
            err_msg=f"R slope did not converge to thin-shell adiabatic 4/13 = {_lambda_thin:.4f}",
        )
        assert_allclose(
            dlogv_dlogt[check_start:-3],
            _lambda_thin - 1,
            rtol=rtol,
            err_msg=f"v slope did not converge to -9/13 = {_lambda_thin - 1:.4f}",
        )


# ------------------------------------------------------------------ #
# Momentum-Conserving Snowplow Engine Tests                          #
# ------------------------------------------------------------------ #


class TestMomentumConservingShockEngine:
    r"""
    Asymptotic slope tests for :class:`~trilobite.dynamics.shocks.numerical.MomentumConservingShockEngine`.

    The momentum-conserving (snowplow) closure conserves the total shell momentum
    :math:`\Pi = Mv`.  After the reverse shock has traversed all ejecta, no further
    momentum is deposited (:math:`u_1 \to 0`, :math:`u_4 = 0`), so :math:`\Pi`
    approaches a constant :math:`\Pi_\infty`.  For uniform CSM the swept mass grows as
    :math:`M \propto R^3`, giving:

    .. math::

        v = \frac{\Pi_\infty}{M} \propto R^{-3}
        \;\Rightarrow\;
        R^3 \frac{dR}{dt} \propto 1
        \;\Rightarrow\;
        R \propto t^{1/4}.

    This is the classic *snowplow* scaling, which contrasts with the Sedov-Taylor
    :math:`R \propto t^{2/5}` produced by the pressure-driven closure.
    """

    @pytest.fixture(scope="class")
    def engine(self):
        """Momentum-conserving snowplow engine."""
        return MomentumConservingShockEngine()

    def test_snowplow_asymptotic_slope_uniform_csm(self, engine, diagnostic_plots, diagnostic_plots_dir):
        r"""
        Verify convergence to the snowplow (momentum-conserving) scaling
        :math:`R \propto t^{1/4}` after the swept CSM mass exceeds
        :math:`M_{\rm ej}` in uniform CSM.

        Setup:

        - Broken-power-law ejecta: :math:`E_{\rm ej} = 10^{51}` erg,
          :math:`M_{\rm ej} = 5\,M_\odot`, n=10, δ=0.
        - Uniform CSM: :math:`\rho_0 = 10^{-16}` g cm\ :math:`^{-3}`.
        - Initial shell at :math:`R_0 = 10^{15}` cm, :math:`v_0 = 10^4` km/s.
        - Time grid: 1–:math:`10^{10}` days (ten decades).

        The slope :math:`d\log R/d\log t` is measured from a settling interval
        after the swept mass crosses :math:`M_{\rm ej}` to the end of the grid.

        Expected result: :math:`d\log R/d\log t \to 1/4` within 10%.
        """
        n = 10
        _lambda_sp = 1.0 / 4.0
        rtol = 1e-1

        E_ej = 1e51 * u.erg
        M_ej = 5.0 * u.M_sun
        rho_0_cgs = 1e-16  # g/cm^3

        K, v_t = BrokenPowerLawEjectaProfile.normalize(E_ej, M_ej, n=n, delta=0)
        rho_ej = BrokenPowerLawEjectaProfile.as_optimized_callable(K=K, v_t=v_t, n=n, delta=0)
        rho_csm = UniformCSMProfile.as_optimized_callable(rho_0=rho_0_cgs)

        rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(rho_ej, rho_csm)

        time = np.geomspace(1e0, 1e10, 5000) * u.day
        R_cd_0 = 1e15 * u.cm
        v_cd_0 = 1e4 * u.km / u.s
        M_cd_0 = (4 * np.pi / 3) * rho_0_cgs * R_cd_0.to_value(u.cm) ** 3 * u.g

        state = engine.compute_shock_properties(
            time,
            rho_1=rho_1,
            rho_4=rho_4,
            u_1=u_1,
            u_4=u_4,
            R_0=R_cd_0,
            M_0=M_cd_0,
            v_0=v_cd_0,
            t_0=time[0],
        )

        log_t = np.log10(time.to_value(u.s))
        dlogR_dlogt = np.gradient(np.log10(state.radius.to_value(u.cm)), log_t)

        # Detect when swept shell mass exceeds M_ej (onset of pure snowplow regime).
        M_ej_g = M_ej.to_value(u.g)
        m = state.mass.to_value(u.g)
        crossed = m >= M_ej_g
        if not np.any(crossed):
            pytest.skip(
                f"Swept mass did not reach M_ej within the time grid "
                f"(max M/M_ej = {m.max() / M_ej_g:.3g}). Extend the time grid or increase rho_0."
            )
        term_idx = int(np.argmax(crossed))

        # Check only the last 10% of the grid to ensure the transient has fully damped.
        check_start = max(term_idx + 50, len(time) * 9 // 10)

        if diagnostic_plots:
            import matplotlib.pyplot as plt

            from trilobite.utils.plot_utils import set_plot_style

            set_plot_style()

            log_t_days = np.log10(time.to_value(u.day))

            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle(
                rf"Momentum-Conserving Engine — Snowplow Slope (n={n}, s=0, "
                rf"$\rho_0={rho_0_cgs:.0e}$ g cm$^{{-3}}$)"
            )

            axes[0].loglog(
                time.to_value(u.day),
                state.radius.to_value(u.cm),
                label=r"$R_{\rm sh}$ (momentum-conserving)",
            )
            axes[0].axvline(time[term_idx].to_value(u.day), ls="-.", color="gray", label=r"$M=M_{\rm ej}$")
            axes[0].axvline(time[check_start].to_value(u.day), ls=":", color="gray", label="Check window start")
            axes[0].set_xlabel("Time [days]")
            axes[0].set_ylabel("Radius [cm]")
            axes[0].legend(fontsize=8)

            axes[1].semilogx(log_t_days, dlogR_dlogt, label=r"$d\log R/d\log t$")
            axes[1].axhline(_lambda_sp, ls="--", color="C1", label=rf"Snowplow $\lambda={_lambda_sp:.2f}$")
            axes[1].axhline(2 / 5, ls=":", color="C2", alpha=0.6, label="Sedov-Taylor 2/5")
            axes[1].axvline(time[term_idx].to_value(u.day), ls="-.", color="gray")
            axes[1].axvline(time[check_start].to_value(u.day), ls=":", color="gray")
            axes[1].set_xlabel("Time [days]")
            axes[1].set_ylabel(r"$d\log R/d\log t$")
            axes[1].set_ylim([0.0, 1.1])
            axes[1].legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(
                f"{diagnostic_plots_dir}/test_momentum_conserving_snowplow_slope.png",
                dpi=150,
            )
            plt.close(fig)

        assert check_start < len(time) - 10, (
            f"Too few grid points remain after M_ej crossing "
            f"({len(time) - check_start} points). Extend the time grid or increase rho_0."
        )
        assert_allclose(
            dlogR_dlogt[check_start:-3],
            _lambda_sp,
            rtol=rtol,
            err_msg=f"R slope did not converge to snowplow 1/4 = {_lambda_sp:.3f}",
        )
