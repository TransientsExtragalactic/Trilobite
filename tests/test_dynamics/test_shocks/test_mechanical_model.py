r"""
Tests for :class:`~trilobite.dynamics.shocks.numerical.MechanicalShockEngine`
asymptotic power-law convergence.

Two fundamental limits are tested:

1. **Chevalier self-similar limit** (n=10, s=2 wind CSM): In the early regime where
   both shocks are active, the forward shock, contact discontinuity, and reverse shock
   should all evolve as R ∝ t^λ with λ = (n-3)/(n-s) = 7/8, consistent with the
   analytic Chevalier (1982) self-similar solution.

2. **Sedov-Taylor late-time limit** (n=10, s=0 uniform CSM): Once the reverse shock
   has traversed all ejecta (M2 → M_ej), the forward shock transitions to Sedov-Taylor
   blast-wave scaling R_fs ∝ t^(2/5).

Both tests seed self-consistent initial conditions from
:class:`~trilobite.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarEngine`,
eliminating the relaxation transient that arises when the initial state is
inconsistent with the shock closure.

The log-log time slopes are computed via ``np.gradient`` over the converged portion
of each time grid and compared to the analytic exponents within the stated tolerance.
"""

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from trilobite.dynamics.profiles import BrokenPowerLawEjectaProfile, UniformCSMProfile, WindCSMProfile
from trilobite.dynamics.shocks import (
    ChevalierTwoShockSelfSimilarEngine,
    MechanicalShockEngine,
    SedovTaylorShockEngine,
    make_homologous_stationary_sources,
)


class TestMechanicalShockEngine:
    """
    Asymptotic slope tests for :class:`~trilobite.dynamics.shocks.numerical.MechanicalShockEngine`.

    The two tests cover the full lifecycle of a mechanical shock:

    - *Chevalier wind test*: n=10, s=2, E_ej=10^51 erg, M_ej=10 M_sun.  The wind CSM
      (ρ ∝ r^-2) and broken-power-law ejecta reproduce the Chevalier self-similar
      two-shock structure.  ICs are seeded from
      :class:`~trilobite.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarEngine`
      at t=1 day.

    - *Sedov-Taylor transition test*: n=10, s=0, E_ej=10^51 erg, M_ej=5 M_sun,
      ρ_0=10^-16 g cm^-3.  The dense uniform CSM causes the reverse shock to terminate
      within the six-decade time grid.  After termination the forward shock converges
      to Sedov-Taylor scaling.
    """

    @pytest.fixture(scope="class")
    def engine(self):
        """Non-relativistic mechanical shock engine with γ=5/3, μ=0.6."""
        return MechanicalShockEngine(gamma_2=5 / 3, gamma_3=5 / 3, mu_2=0.6, mu_3=0.6)

    @pytest.fixture(scope="class")
    def analytic_engine(self):
        """Pre-tabulated Chevalier two-shock engine used for IC generation."""
        return ChevalierTwoShockSelfSimilarEngine(gamma=5 / 3, mu=0.6, show_progress=False)

    # ------------------------------------------------------------------ #
    # Test 1: Chevalier wind (n=10, s=2) asymptotic slopes               #
    # ------------------------------------------------------------------ #

    def test_chevalier_wind_asymptotic_slopes(self, engine, analytic_engine, diagnostic_plots, diagnostic_plots_dir):
        r"""
        Verify that the mechanical engine converges to the Chevalier (1982)
        self-similar power-law slopes for n=10, s=2 (wind-like CSM).

        The expected asymptotic log-log time derivatives (last half of the grid,
        where transients have damped) are:

        - :math:`d\log R_{\rm cd}/d\log t \to \lambda = 7/8`
        - :math:`d\log v_{\rm cd}/d\log t \to \lambda - 1 = -1/8`
        - :math:`d\log v_{\rm fs}/d\log t \to \lambda - 1 = -1/8`
        - :math:`d\log v_{\rm rs}/d\log t \to \lambda - 1 = -1/8`

        Tolerance: 5% (``rtol=5e-2``).
        """
        n, s = 10.0, 2
        _lambda = (n - 3.0) / (n - s)  # = 7/8 = 0.875
        rtol = 5e-2

        E_ej = 1e51 * u.erg
        M_ej = 10.0 * u.M_sun
        M_dot = 1e-5 * u.M_sun / u.yr
        v_wind = 100.0 * u.km / u.s

        K_csm = (M_dot / (4 * np.pi * v_wind)).to(u.g / u.cm)

        K, v_t = BrokenPowerLawEjectaProfile.normalize(E_ej, M_ej, n=n, delta=0)
        rho_ej = BrokenPowerLawEjectaProfile.as_optimized_callable(K=K, v_t=v_t, n=n, delta=0)
        rho_csm = WindCSMProfile.as_optimized_callable(mass_loss_rate=M_dot, wind_velocity=v_wind)

        rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(rho_ej, rho_csm)

        # Four decades from 1 to 1e4 days.
        time = np.geomspace(1, 1e4, 512) * u.day

        # Seed ICs from the analytic Chevalier two-shock engine at t=time[0].
        ic_state = analytic_engine.compute_shock_properties(
            time=time[0:1], E_ej=E_ej, M_ej=M_ej, n=n, s=s, K_csm=K_csm, delta=0
        )
        R_cd_0 = ic_state.radius_cd[0]
        v_cd_0 = ic_state.velocity_cd[0]

        ic = engine.infer_initial_conditions(
            R_cd_0=R_cd_0,
            v_cd_0=v_cd_0,
            t_0=time[0],
            rho_1=rho_1,
            rho_4=rho_4,
            u_1=u_1,
            u_4=u_4,
        )

        state = engine.compute_shock_properties(
            time=time,
            rho_1=rho_1,
            rho_4=rho_4,
            u_1=u_1,
            u_4=u_4,
            initial_conditions=ic,
            t_0=time[0],
        )

        log_t = np.log10(time.to_value(u.s))
        dlogR_cd = np.gradient(np.log10(state.radius.to_value(u.cm)), log_t)
        dlogv_cd = np.gradient(np.log10(state.velocity.to_value(u.cm / u.s)), log_t)
        dlogv_fs = np.gradient(np.log10(state.velocity_fs.to_value(u.cm / u.s)), log_t)
        dlogv_rs = np.gradient(np.log10(state.velocity_rs.to_value(u.cm / u.s)), log_t)

        if diagnostic_plots:
            import matplotlib.pyplot as plt

            from trilobite.utils.plot_utils import set_plot_style

            set_plot_style()

            log_t_days = np.log10(time.to_value(u.day))
            half = len(time) // 2

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Mechanical Engine — Chevalier Wind Slopes (n={n:.0f}, s={s})")

            for ax, slope, ylabel, expected in [
                (axes[0, 0], dlogR_cd, r"$d\log R_{\rm cd}/d\log t$", _lambda),
                (axes[0, 1], dlogv_cd, r"$d\log v_{\rm cd}/d\log t$", _lambda - 1),
                (axes[1, 0], dlogv_fs, r"$d\log v_{\rm fs}/d\log t$", _lambda - 1),
                (axes[1, 1], dlogv_rs, r"$d\log v_{\rm rs}/d\log t$", _lambda - 1),
            ]:
                ax.plot(log_t_days, slope, label="Mechanical engine")
                ax.axhline(expected, ls="--", color="C1", label=rf"Chevalier $\lambda={expected:.3f}$")
                ax.axvline(log_t_days[half], ls=":", color="gray", alpha=0.6, label="Check window start")
                ax.set_xlabel(r"$\log_{10}$(Time [days])")
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(
                f"{diagnostic_plots_dir}/test_mechanical_chevalier_wind_slopes.png",
                dpi=150,
            )
            plt.close(fig)

        # Check last half of the time grid; exclude last 3 points for endpoint effects.
        half = len(time) // 2
        assert_allclose(
            dlogR_cd[half:-3],
            _lambda,
            rtol=rtol,
            err_msg=f"R_cd slope did not converge to Chevalier λ = {_lambda:.4f}",
        )
        assert_allclose(
            dlogv_cd[half:-3],
            _lambda - 1,
            rtol=rtol,
            err_msg=f"v_cd slope did not converge to Chevalier λ-1 = {_lambda - 1:.4f}",
        )
        assert_allclose(
            dlogv_fs[half:-3],
            _lambda - 1,
            rtol=rtol,
            err_msg=f"v_fs slope did not converge to Chevalier λ-1 = {_lambda - 1:.4f}",
        )
        assert_allclose(
            dlogv_rs[half:-3],
            _lambda - 1,
            rtol=rtol,
            err_msg=f"v_rs slope did not converge to Chevalier λ-1 = {_lambda - 1:.4f}",
        )

    # ------------------------------------------------------------------ #
    # Test 2: Sedov-Taylor transition (n=10, s=0) asymptotic slopes      #
    # ------------------------------------------------------------------ #

    def test_sedov_taylor_asymptotic_slopes(self, engine, analytic_engine, diagnostic_plots, diagnostic_plots_dir):
        r"""
        Verify that the forward shock transitions to Sedov-Taylor blast-wave
        scaling after the reverse shock has traversed all ejecta (n=10, s=0).

        Setup uses a dense uniform CSM (ρ_0 = 10^-16 g cm^-3) so that the swept
        ejecta mass M2 reaches M_ej within the six-decade time grid.  The
        ``M1_total=M_ej`` argument triggers the ODE event that quenches RS
        mass-loading at that crossing.

        The reverse-shock termination epoch is detected dynamically from the
        evolved ``mass_2`` field; slopes are checked from a settling interval
        after termination to the end of the grid.

        Expected late-time slopes (after RS termination):

        - :math:`d\log R_{\rm fs}/d\log t \to 2/5`
        - :math:`d\log v_{\rm fs}/d\log t \to -3/5`

        Tolerance: 10% (``rtol=1e-1``).
        """
        n, s = 10.0, 0
        _lambda_chev = (n - 3.0) / n  # = 7/10 (early phase)
        _lambda_st = 2.0 / 5.0  # Sedov-Taylor (late phase)
        rtol_st = 1e-1

        E_ej = 1e51 * u.erg
        M_ej = 5.0 * u.M_sun
        # Dense uniform CSM — reverse shock terminates within ~1e4–1e5 days
        # for these ejecta parameters (estimated from 4π/3 ρ_0 R_ST^3 = M_ej).
        rho_0 = 1e-16  # g/cm^3

        K_csm = rho_0 * u.g / u.cm**3

        K, v_t = BrokenPowerLawEjectaProfile.normalize(E_ej, M_ej, n=n, delta=0)
        rho_ej = BrokenPowerLawEjectaProfile.as_optimized_callable(K=K, v_t=v_t, n=n, delta=0)
        rho_csm = UniformCSMProfile.as_optimized_callable(rho_0=rho_0)

        rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(rho_ej, rho_csm)

        # Six decades to capture both the Chevalier early phase and the ST late phase.
        time = np.geomspace(1, 1e6, 600) * u.day

        # Seed ICs from the analytic Chevalier two-shock engine.
        ic_state = analytic_engine.compute_shock_properties(
            time=time[0:1], E_ej=E_ej, M_ej=M_ej, n=n, s=s, K_csm=K_csm, delta=0
        )
        R_cd_0 = ic_state.radius_cd[0]
        v_cd_0 = ic_state.velocity_cd[0]

        ic = engine.infer_initial_conditions(
            R_cd_0=R_cd_0,
            v_cd_0=v_cd_0,
            t_0=time[0],
            rho_1=rho_1,
            rho_4=rho_4,
            u_1=u_1,
            u_4=u_4,
        )

        state = engine.compute_shock_properties(
            time=time,
            rho_1=rho_1,
            rho_4=rho_4,
            u_1=u_1,
            u_4=u_4,
            initial_conditions=ic,
            t_0=time[0],
            M1_total=M_ej,
        )

        log_t = np.log10(time.to_value(u.s))
        dlogR_fs = np.gradient(np.log10(state.radius_fs.to_value(u.cm)), log_t)
        dlogv_fs = np.gradient(np.log10(state.velocity_fs.to_value(u.cm / u.s)), log_t)

        # Detect RS termination: first index where M2 >= 0.99 × M_ej.
        M_ej_g = M_ej.to_value(u.g)
        m2 = state.mass_2.to_value(u.g)
        terminated = m2 >= 0.99 * M_ej_g
        if not np.any(terminated):
            pytest.skip(
                f"Reverse shock did not terminate within the time grid "
                f"(max M2/M_ej = {m2.max() / M_ej_g:.3g}). "
                "Increase rho_0 or extend the time grid."
            )
        term_idx = int(np.argmax(terminated))

        # Allow the forward shock an extra settling interval before checking slopes.
        settle = max(15, (len(time) - term_idx) // 10)
        check_start = min(term_idx + settle, len(time) - 20)

        if diagnostic_plots:
            import matplotlib.pyplot as plt

            from trilobite.utils.plot_utils import set_plot_style

            set_plot_style()

            log_t_days = np.log10(time.to_value(u.day))

            st_engine = SedovTaylorShockEngine(gamma=5 / 3, mu=0.6)
            st_state = st_engine.compute_shock_properties(time=time, E=E_ej, rho_0=rho_0 * u.g / u.cm**3)

            fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)
            fig.suptitle(
                rf"Mechanical Engine — Sedov-Taylor Transition "
                rf"(n={n:.0f}, s={s}, $\rho_0={rho_0:.0e}$ g cm$^{{-3}}$)"
            )

            # Panel 1: forward-shock radius vs Sedov-Taylor.
            axes[0].loglog(
                time.to_value(u.day),
                state.radius_fs.to_value(u.cm),
                label=r"$R_{\rm fs}$ (mechanical)",
            )
            axes[0].loglog(
                time.to_value(u.day),
                st_state.radius.to_value(u.cm),
                ls="--",
                label="Sedov-Taylor",
            )
            axes[0].axvline(
                time[term_idx].to_value(u.day),
                ls="-.",
                color="gray",
                label="RS termination",
            )
            axes[0].axvline(
                time[check_start].to_value(u.day),
                ls=":",
                color="gray",
                label="Check window start",
            )
            axes[0].set_xlabel("Time [days]")
            axes[0].set_ylabel("Radius [cm]")
            axes[0].legend(fontsize=8)

            # Panel 2: d log R_fs / d log t slope.
            axes[1].semilogx(time.to_value(u.day), dlogR_fs, label=r"$d\log R_{\rm fs}/d\log t$")
            axes[1].axhline(_lambda_chev, ls="--", color="C1", label=rf"Chevalier $\lambda={_lambda_chev:.2f}$")
            axes[1].axhline(_lambda_st, ls=":", color="C2", label=rf"Sedov-Taylor $\lambda={_lambda_st:.2f}$")
            axes[1].axvline(time[term_idx].to_value(u.day), ls="-.", color="gray")
            axes[1].axvline(time[check_start].to_value(u.day), ls=":", color="gray")
            axes[1].set_xlabel("Time [days]")
            axes[1].set_ylabel(r"$d\log R_{\rm fs}/d\log t$")
            axes[1].set_ylim([0.0, 1.0])
            axes[1].legend(fontsize=8)

            # Panel 3: d log v_fs / d log t slope.
            axes[2].semilogx(time.to_value(u.day), dlogv_fs, label=r"$d\log v_{\rm fs}/d\log t$")
            axes[2].axhline(
                _lambda_chev - 1, ls="--", color="C1", label=rf"Chevalier $\lambda-1={_lambda_chev - 1:.2f}$"
            )
            axes[2].axhline(_lambda_st - 1, ls=":", color="C2", label=rf"Sedov-Taylor $\lambda-1={_lambda_st - 1:.2f}$")
            axes[2].axvline(time[term_idx].to_value(u.day), ls="-.", color="gray")
            axes[2].axvline(time[check_start].to_value(u.day), ls=":", color="gray")
            axes[2].set_xlabel("Time [days]")
            axes[2].set_ylabel(r"$d\log v_{\rm fs}/d\log t$")
            axes[2].set_ylim([-1.0, 0.2])
            axes[2].legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(
                f"{diagnostic_plots_dir}/test_mechanical_sedov_taylor_transition.png",
                dpi=150,
            )
            plt.close(fig)

        assert check_start < len(time) - 5, (
            f"Too few grid points remain after RS termination "
            f"({len(time) - check_start} points). "
            "Extend the time grid or increase rho_0."
        )
        assert_allclose(
            dlogR_fs[check_start:-3],
            _lambda_st,
            rtol=rtol_st,
            err_msg=f"R_fs slope did not converge to Sedov-Taylor 2/5 = {_lambda_st:.3f}",
        )
        assert_allclose(
            dlogv_fs[check_start:-3],
            _lambda_st - 1,
            rtol=rtol_st,
            err_msg=f"v_fs slope did not converge to Sedov-Taylor -3/5 = {_lambda_st - 1:.3f}",
        )
