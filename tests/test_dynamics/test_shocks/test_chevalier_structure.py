"""
Tests for ``compute_self_similar_functions`` (Chevalier 1982 shock structure).

Validates the dimensionless self-similar profiles against Table 1 of
Chevalier (1982) for the canonical s=0, n=7 case and checks geometric and
finiteness invariants that must hold for every valid (n, s) pair.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from triceratops.dynamics.shocks.chevalier import compute_self_similar_functions


class TestChevalierSelfSimilarFunctions:
    """Tests for :func:`~triceratops.dynamics.shocks.chevalier.compute_self_similar_functions`."""

    @pytest.fixture(scope="class")
    def solution_n7_s0(self):
        """Canonical s=0, n=7, γ=5/3 solution at high resolution."""
        return compute_self_similar_functions(
            n=7.0,
            s=0.0,
            gamma=5.0 / 3.0,
            n_points=512,
            contact_epsilon=1.0e-5,
            rtol=1.0e-9,
            atol=1.0e-11,
        )

    # ------------------------------------------------------------------ #
    # Chevalier (1982) Table 1 reference values: s=0, n=7                #
    #   R_1 / R_c = 1.181,  R_2 / R_c = 0.935,  A = 1.2                 #
    # ------------------------------------------------------------------ #

    def test_radius_fs_over_cd(self, solution_n7_s0):
        """Forward shock / contact discontinuity ratio matches Chevalier Table 1."""
        assert_allclose(solution_n7_s0.radius_fs_over_radius_cd, 1.181, rtol=5.0e-3)

    def test_radius_rs_over_cd(self, solution_n7_s0):
        """Reverse shock / contact discontinuity ratio matches Chevalier Table 1."""
        assert_allclose(solution_n7_s0.radius_rs_over_radius_cd, 0.935, rtol=5.0e-3)

    def test_normalization_A(self, solution_n7_s0):
        """Normalization constant A matches Chevalier Table 1 within 5%."""
        assert_allclose(solution_n7_s0.A, 1.2, rtol=5.0e-2)

    # ------------------------------------------------------------------ #
    # Geometric ordering                                                   #
    # ------------------------------------------------------------------ #

    def test_geometric_ordering(self, solution_n7_s0):
        """Reverse shock must lie inside the contact discontinuity, forward shock outside."""
        assert solution_n7_s0.radius_rs_over_radius_cd < 1.0
        assert solution_n7_s0.radius_fs_over_radius_cd > 1.0

    # ------------------------------------------------------------------ #
    # Profile finiteness                                                   #
    # ------------------------------------------------------------------ #

    def test_profiles_finite(self, solution_n7_s0):
        """All profile arrays must be finite on the global grid."""
        assert np.all(np.isfinite(solution_n7_s0.xi))
        assert np.all(np.isfinite(solution_n7_s0.density_hat))
        assert np.all(np.isfinite(solution_n7_s0.pressure_hat))
        assert np.all(np.isfinite(solution_n7_s0.velocity_hat))

    # ------------------------------------------------------------------ #
    # Diagnostic plot                                                      #
    # ------------------------------------------------------------------ #

    def test_diagnostic_plot(self, solution_n7_s0, diagnostic_plots, diagnostic_plots_dir):
        """Generate a normalized structure plot when diagnostic plots are enabled."""
        if not diagnostic_plots:
            pytest.skip("diagnostic plots not requested")

        import matplotlib.pyplot as plt

        from triceratops.utils.plot_utils import set_plot_style

        set_plot_style()
        fig, ax = plt.subplots(figsize=(7.0, 4.5))

        ax.plot(
            solution_n7_s0.xi,
            solution_n7_s0.density_hat / np.nanmax(solution_n7_s0.density_hat),
            label=r"$\rho$",
        )
        ax.plot(
            solution_n7_s0.xi,
            solution_n7_s0.pressure_hat / np.nanmax(solution_n7_s0.pressure_hat),
            label=r"$p$",
        )
        ax.plot(
            solution_n7_s0.xi,
            solution_n7_s0.velocity_hat / np.nanmax(solution_n7_s0.velocity_hat),
            label=r"$u$",
        )

        ax.axvline(1.0, linestyle="--", linewidth=1.0, label=r"$R_c$")
        ax.axvline(
            solution_n7_s0.radius_rs_over_radius_cd,
            linestyle=":",
            linewidth=1.0,
            label=r"$R_{\rm rs}$",
        )
        ax.axvline(
            solution_n7_s0.radius_fs_over_radius_cd,
            linestyle=":",
            linewidth=1.0,
            label=r"$R_{\rm fs}$",
        )

        ax.set_xlabel(r"$r/R_c$")
        ax.set_ylabel("normalized structure")
        ax.set_title(r"Chevalier self-similar structure: $s=0,\ n=7$")
        ax.legend(loc="best")
        fig.tight_layout()

        plt.savefig(diagnostic_plots_dir / "test_chevalier_self_similar_structure.png")
        plt.close(fig)
