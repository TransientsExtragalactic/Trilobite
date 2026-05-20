r"""
Tests for self-similar shock engines.

This module validates analytic and semi-analytic shock engines whose solutions
are expected to follow self-similar power-law behavior. The tests focus on
relationships that should hold independently of the detailed normalization of
the solution, such as logarithmic time slopes, internal consistency between
radius and velocity, density scalings, and agreement between specialized
wrappers and their corresponding general engines.

The common structure tested throughout this module is a shock or contact
discontinuity radius of the form

.. math::

    R(t) \propto t^\lambda,

which implies

.. math::

    v(t) = \frac{dR}{dt} = \lambda \frac{R}{t}
    \propto t^{\lambda - 1}.

For models that also return swept-up or shocked mass, the expected mass scaling
is checked against the model-specific density profile. For example, a shock
expanding through a circumstellar medium with

.. math::

    \rho_{\rm csm}(r) \propto r^{-s}

has swept-up CSM mass scaling as

.. math::

    M_{\rm csm}(<R) \propto R^{3-s}
    \propto t^{(3-s)\lambda}.

The Chevalier ejecta-CSM interaction solution is one important case covered by
these tests. In that model, outer power-law ejecta with index ``n`` interact
with a power-law CSM with index ``s``, giving

.. math::

    \lambda = \frac{n - 3}{n - s}.

Additional self-similar shock engines can be tested in this module by checking
their model-specific value of ``lambda`` and any derived power-law scalings
against the returned numerical or analytic solution.

The tests are designed to emphasize robust self-similar behavior rather than
fragile absolute normalizations. Where appropriate, logarithmic derivatives are
used so that short startup transients, finite-difference edge effects, or
model-specific scale choices do not dominate the validation.
"""

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from triceratops.dynamics.shocks import ChevalierSelfSimilarShockEngine, ChevalierSelfSimilarWindShockEngine
from triceratops.dynamics.shocks.utils import _normalize_BPL_ejecta

# -------------------------------------------------------------------- #
# Test Fixtures / Data                                                 #
# -------------------------------------------------------------------- #
_CHEVALIER_TABLE_1 = [
    (0, 6, 2.4),
    (0, 7, 1.2),
    (0, 8, 0.71),
    (0, 9, 0.47),
    (0, 10, 0.33),
    (0, 12, 0.19),
    (0, 14, 0.12),
    (2, 6, 0.62),
    (2, 7, 0.27),
    (2, 8, 0.15),
    (2, 9, 0.096),
    (2, 10, 0.067),
    (2, 12, 0.038),
    (2, 14, 0.025),
]
""" Combinations of (n, s, A) from Chevalier 1982 Table 1 for verifying the normalization of the
self-similar solution. A is the normalization constant in the formula R_c = (A · K_EJ / K_csm)^{1/(n-s)} · t^lambda.
"""


# ------------------------------------------------------------------ #
# Chevalier (1982) self-similar shock engine tests                   #
# ------------------------------------------------------------------ #
class TestChevalierSelfSimilarShockEngine:
    """
    Tests for the analytic Chevalier (1982) self-similar shock engine.

    The engine solves for a shock propagating into a power-law CSM
    (rho_csm ∝ r^-s) driven by power-law ejecta (rho_ej ∝ v^-n). Under these
    assumptions the contact discontinuity evolves as R(t) ∝ t^λ, where the
    self-similar index is λ = (n-3)/(n-s).

    The fiducial case throughout is n=10, s=2 (→ λ=7/8), which is the
    canonical Chevalier wind-interaction scenario.
    """

    @pytest.fixture(scope="class")
    def relative_tolerance(self):
        """5% relative tolerance for power-law slope comparisons."""
        return 5e-2

    @pytest.fixture(scope="class")
    def time_grid(self):
        """512 logarithmically spaced times from 1 to 1e4 days."""
        return np.geomspace(1, 1e4, 512) * u.day

    @pytest.fixture(scope="class")
    def engine(self):
        """Stateless Chevalier shock engine with default gamma=5/3, mu=0.5."""
        return ChevalierSelfSimilarShockEngine()

    @pytest.fixture(scope="class")
    def fiducial_solution(self, time_grid, engine):
        """
        Pre-computed solution for n=10, s=2, delta=0, reused across multiple tests.

        K_csm = (5e-16 g/cm^3) * (1e15 cm)^2 places the shock at roughly AU-scale
        radii, which is physically reasonable for typical SN–CSM interaction.
        """
        E_ej = 1e51 * u.erg
        M_ej = 10 * u.Msun
        K_csm = (5e-16 * u.g / u.cm**3) * (1e15 * u.cm) ** 2
        return engine.compute_shock_properties(time_grid, E_ej, M_ej, n=10, s=2, K_csm=K_csm, delta=0)

    # ------------------------------------------------------------------ #
    # Time-slope tests                                                     #
    # ------------------------------------------------------------------ #

    def test_time_slope(self, time_grid, engine, diagnostic_plots, diagnostic_plots_dir, relative_tolerance):
        """
        Verify R(t) ∝ t^λ and v(t) ∝ t^(λ-1) for the canonical n=10, s=2 case.

        Method: compute log-log slopes numerically via np.gradient and compare to
        the analytic expectation λ = (n-3)/(n-s). For a pure power law the
        numerical gradient is exact at every interior point; the 5% tolerance
        accommodates first-order finite-difference error at the endpoints.
        """
        n, s = 10, 2
        _lambda = (n - 3) / (n - s)

        E_ej = 1e51 * u.erg
        M_ej = 10 * u.Msun
        K_csm = (5e-16 * u.g / u.cm**3) * (1e15 * u.cm) ** s

        solution = engine.compute_shock_properties(time_grid, E_ej, M_ej, n=n, s=s, K_csm=K_csm, delta=0)

        log_t = np.log10(time_grid.to_value(u.s))
        dlogR_dlogt = np.gradient(np.log10(solution.radius.to_value(u.cm)), log_t)
        dlogv_dlogt = np.gradient(np.log10(solution.velocity.to_value(u.cm / u.s)), log_t)

        if diagnostic_plots:
            import matplotlib.pyplot as plt

            from triceratops.utils.plot_utils import set_plot_style

            set_plot_style()
            fig, axes = plt.subplots(2, 1, figsize=(8, 8))
            log_t_days = np.log10(time_grid.to_value(u.day))

            axes[0].plot(log_t_days, dlogR_dlogt, label="Computed")
            axes[0].axhline(_lambda, ls="--", color="C1", label="Expected")
            axes[0].set_xlabel(r"$\log_{10}$(Time [days])")
            axes[0].set_ylabel(r"$d\log R\,/\,d\log t$")
            axes[0].set_ylim([_lambda * 0.5, _lambda * 1.5])
            axes[0].legend()

            axes[1].plot(log_t_days, dlogv_dlogt, label="Computed")
            axes[1].axhline(_lambda - 1, ls="--", color="C1", label="Expected")
            axes[1].set_xlabel(r"$\log_{10}$(Time [days])")
            axes[1].set_ylabel(r"$d\log v\,/\,d\log t$")
            axes[1].set_ylim([(_lambda - 1) * 2, 0])
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(f"{diagnostic_plots_dir}/test_chevalier_self_similar_time_slope.png")
            plt.close(fig)

        assert_allclose(dlogR_dlogt, _lambda, rtol=relative_tolerance)
        assert_allclose(dlogv_dlogt, _lambda - 1, rtol=relative_tolerance)

    # ------------------------------------------------------------------ #
    # Analytic quantity tests                                              #
    # ------------------------------------------------------------------ #

    def test_scale_parameter(self, engine):
        """
        Verify compute_scale_parameter against the analytically known value ζ=17
        for n=10, s=2.

        For n=10, s=2 the self-similar index is λ=7/8. Substituting into
        ζ = (3λ² + 4λ(λ−1)/(3−s)) / (3(1−λ)² − 4λ(λ−1)/(n−3))
        gives ζ = (119/64) / (7/64) = 17 exactly.
        """
        assert_allclose(engine.compute_scale_parameter(n=10, s=2), 17.0, rtol=1e-12)

    def test_velocity_radius_consistency(self, time_grid, fiducial_solution):
        """
        Verify v_cd = λ R / t at every time step.

        The implementation defines the velocity exactly this way (it is not
        independently integrated), so this checks for refactoring regressions
        that would break the identity.
        """
        n, s = 10, 2
        _lambda = (n - 3) / (n - s)
        expected_velocity = _lambda * fiducial_solution.radius / time_grid.to(u.s)

        assert_allclose(
            fiducial_solution.velocity.to_value(u.cm / u.s),
            expected_velocity.to_value(u.cm / u.s),
            rtol=1e-12,
        )

    def test_post_shock_density_time_slope(self, time_grid, fiducial_solution, relative_tolerance):
        """
        Verify the post-shock density time slope is −s·λ.

        The upstream CSM density at the shock is ρ_csm(R(t)) = K_csm · R(t)^(−s) ∝ t^(−sλ).
        The strong-shock compression ratio (γ+1)/(γ−1) is constant, so the
        post-shock density inherits the same slope. For n=10, s=2, λ=7/8:
        expected slope = −7/4.
        """
        n, s = 10, 2
        _lambda = (n - 3) / (n - s)
        expected_slope = -s * _lambda

        log_t = np.log10(time_grid.to_value(u.s))
        slope = np.gradient(np.log10(fiducial_solution.post_shock_density.to_value(u.g / u.cm**3)), log_t)

        assert_allclose(slope, expected_slope, rtol=relative_tolerance)

    # ------------------------------------------------------------------ #
    # Input validation tests                                               #
    # ------------------------------------------------------------------ #

    def test_convergence_guards(self, engine, time_grid):
        """
        Verify that physically invalid index values raise ValueError.

        n > 5 is required for a finite ejecta kinetic-energy integral;
        delta < 3 is required for a finite ejecta mass integral.
        """
        E_ej = 1e51 * u.erg
        M_ej = 10 * u.Msun
        K_csm = 5e11 * u.g / u.cm

        with pytest.raises(ValueError, match="outer ejecta"):
            engine.compute_shock_properties(time_grid, E_ej, M_ej, n=5, s=2, K_csm=K_csm)

        with pytest.raises(ValueError, match="inner ejecta"):
            engine.compute_shock_properties(time_grid, E_ej, M_ej, n=10, s=2, K_csm=K_csm, delta=3)

    # ------------------------------------------------------------------ #
    # Chevalier (1982) Table 1 — parametrized normalization tests        #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("s, n, A_chevalier", _CHEVALIER_TABLE_1)
    def test_time_slope_table(self, engine, s, n, A_chevalier):
        """
        Verify R(t) ∝ t^λ for every (s, n) entry in Chevalier (1982) Table 1.

        The slope λ = (n−3)/(n−s) is an exact consequence of self-similarity and
        is independent of the ζ approximation.  Two-point ratio test avoids
        numerical-differentiation edge effects.
        """
        _lambda = (n - 3) / (n - s)
        E_ej = 1e51 * u.erg
        M_ej = 10 * u.Msun
        K_csm = 1.0 * u.g * u.cm ** (s - 3)

        t1 = 100 * u.day
        t2 = 1000 * u.day

        sol1 = engine.compute_shock_properties(t1, E_ej, M_ej, n=n, s=s, K_csm=K_csm, delta=0)
        sol2 = engine.compute_shock_properties(t2, E_ej, M_ej, n=n, s=s, K_csm=K_csm, delta=0)

        assert_allclose((sol2.radius / sol1.radius).decompose().value, (t2 / t1) ** _lambda, rtol=1e-10)

    @pytest.mark.parametrize("s, n, A_chevalier", _CHEVALIER_TABLE_1)
    def test_radius_normalization_table(self, engine, s, n, A_chevalier):
        r"""
        Compare the engine radius at a reference time to the Chevalier (1982) formula.

        The reference is R_c = (A · K_EJ / K_csm)^{1/(n-s)} · t^λ with A from Table 1.
        K_EJ is extracted from the BPL normalizer using the same (E_ej, M_ej) passed to
        the engine, so the comparison is purely about the scale parameter ζ ≈ 1/A.

        Tolerance is 10%, reflecting the momentum-conservation approximation in ζ.
        The worst-case discrepancy across the full table is ~7% (n=6, s=2), as
        verified numerically from ζ·A ≈ 1.33 raised to the power 1/(n−s) = 1/4.
        """
        _lambda = (n - 3) / (n - s)
        E_ej_cgs = 1e51
        M_ej_cgs = (10 * u.Msun).to_value(u.g)
        K_csm_cgs = 1.0

        t_ref_s = (100 * u.day).to_value(u.s)

        # K_EJ: outer ejecta normalization (= g^n in Chevalier notation),
        # computed identically to the engine's internal call.
        v_t, K_inner = _normalize_BPL_ejecta(E_ej_cgs, M_ej_cgs, n=n, delta=0)
        K_EJ = K_inner * v_t**n

        R_chevalier = (A_chevalier * K_EJ / K_csm_cgs) ** (1 / (n - s)) * t_ref_s**_lambda

        sol = engine.compute_shock_properties(
            100 * u.day,
            E_ej_cgs * u.erg,
            M_ej_cgs * u.g,
            n=n,
            s=s,
            K_csm=K_csm_cgs * u.g * u.cm ** (s - 3),
            delta=0,
        )
        assert_allclose(sol.radius.to_value(u.cm), R_chevalier, rtol=0.10)


class TestChevalierSelfSimilarWindShockEngine:
    """
    Tests for the steady-wind specialization of the Chevalier shock engine.

    ChevalierSelfSimilarWindShockEngine is a thin wrapper that converts
    (M_dot, v_wind) into K_csm = M_dot / (4π v_wind) and calls the parent
    engine with s=2.  The primary test verifies exact agreement with the
    general engine given the same physical parameters.
    """

    @pytest.fixture(scope="class")
    def time_grid(self):
        return np.geomspace(1, 1e4, 512) * u.day

    @pytest.fixture(scope="class")
    def wind_engine(self):
        return ChevalierSelfSimilarWindShockEngine()

    @pytest.fixture(scope="class")
    def general_engine(self):
        return ChevalierSelfSimilarShockEngine()

    def test_wind_general_engine_consistency(self, time_grid, wind_engine, general_engine):
        """
        Verify that the wind engine is exactly equivalent to the general engine
        with s=2 and K_csm = M_dot / (4π v_wind).

        Both engines share the same CGS backend; agreement to 1e-10 confirms
        that the wind-specific parameter conversion introduces no numerical drift.
        """
        E_ej = 1e51 * u.erg
        M_ej = 10 * u.Msun
        M_dot = 1e-5 * u.Msun / u.yr
        v_wind = 1000 * u.km / u.s
        n = 10

        K_csm = M_dot / (4 * np.pi * v_wind)

        wind_sol = wind_engine.compute_shock_properties(time_grid, E_ej, M_ej, M_dot=M_dot, v_wind=v_wind, n=n)
        gen_sol = general_engine.compute_shock_properties(time_grid, E_ej, M_ej, n=n, s=2, K_csm=K_csm)

        assert_allclose(wind_sol.radius.to_value(u.cm), gen_sol.radius.to_value(u.cm), rtol=1e-12)
        assert_allclose(wind_sol.velocity.to_value(u.cm / u.s), gen_sol.velocity.to_value(u.cm / u.s), rtol=1e-12)
        assert_allclose(
            wind_sol.post_shock_density.to_value(u.g / u.cm**3),
            gen_sol.post_shock_density.to_value(u.g / u.cm**3),
            rtol=1e-12,
        )
