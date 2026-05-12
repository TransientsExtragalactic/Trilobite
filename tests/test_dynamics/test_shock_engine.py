"""
Unit tests for shock engine dynamics.

This module contains unit tests for the shock engine dynamics in the Triceratops project.
It uses the pytest framework to define and run tests that verify the behavior of the shock engine
under various conditions.
"""

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from triceratops.dynamics.shocks.numerical import NumericalThinShellShockEngine
from triceratops.dynamics.shocks import ChevalierSelfSimilarShockEngine


class TestChevalierSelfSimilarShockEngine:
    @pytest.fixture(scope="class")
    def relative_tolerance(self):
        """The relative tolerance to use for comparisons."""
        return 5e-2

    @pytest.fixture(scope="class")
    def time_grid(self):
        """The times at which to evaluate the shock engine."""
        return np.geomspace(1, 1e4, 512) * u.day

    @pytest.fixture(scope="class")
    def engine(self):
        """The instantiated engine to do the computation with."""
        return ChevalierSelfSimilarShockEngine()

    def test_time_slope(self, time_grid, engine, diagnostic_plots, diagnostic_plots_dir, relative_tolerance):
        """
        Perform the standard Chevalier self-similar computation and ensure that the time
        slope is lambda = (n-3)/(n-s) as expected.
        """
        # --- Configure parameters --- #
        n, s = 10, 2
        _lambda = (n - 3) / (n - s)

        # Set up the E_ej, M_ej.
        E_ej = 1e51 * u.erg
        M_ej = 10 * u.Msun
        K_csm = (5e-16 * u.g / u.cm**3) * (1e15 * u.cm) ** s

        solution = engine.compute_shock_properties(time_grid, E_ej, M_ej, n=n, s=s, K_csm=K_csm, delta=0)

        # --- Generate expected results and compare --- #
        dlogR_dlogt_expected = _lambda
        dlogv_dlogt_expected = _lambda - 1

        dlogR_dlott_computed = np.gradient(
            np.log10(solution["radius"].to_value(u.cm)), np.log10(time_grid.to_value(u.s))
        )
        dlogv_dlott_computed = np.gradient(
            np.log10(solution["velocity"].to_value(u.cm / u.s)), np.log10(time_grid.to_value(u.s))
        )

        # --- Diagnostic Plots --- #
        if diagnostic_plots:
            import matplotlib.pyplot as plt

            from triceratops.utils.plot_utils import set_plot_style

            set_plot_style()

            fig, axes = plt.subplots(2, 1, figsize=(8, 8))

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

            plt.tight_layout()
            plt.savefig(f"{diagnostic_plots_dir}/test_chevalier_self_similar_time_slope.png")
            plt.close(fig)

        # --- Assert Validity --- #
        assert_allclose(
            dlogR_dlott_computed,
            dlogR_dlogt_expected,
            rtol=relative_tolerance,
        )
        assert_allclose(
            dlogv_dlott_computed,
            dlogv_dlogt_expected,
            rtol=relative_tolerance,
        )


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
        return NumericalThinShellShockEngine()

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
        # We now generate the CSM and ejecta density structures. For the ejecta, we use a simple power-law
        # density structure that should have no impact on the homologous expansion. For the CSM, we use a vacuum
        # density structure.
        def _G(v):
            return (v / v_0_cgs) ** -8

        # Construct the CSM density (vacuum).
        def _rho_csm(r):
            return 0

        # --- Perform the computation --- #
        # Now we can perform the computation using the shock engine.
        results = engine.compute_shock_properties(
            time_grid,
            _rho_csm,
            _G,
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

            from triceratops.utils.plot_utils import set_plot_style

            set_plot_style()

            fig, axes = plt.subplots(3, 1, figsize=(8, 12))
            axes[0].loglog(time_grid.to_value(u.day), results["radius"].to_value(u.cm), label="Computed")
            axes[0].loglog(time_grid.to_value(u.day), expected_R.to_value(u.cm), ls="--", label="Expected")
            axes[0].set_xlabel("Time [days]")
            axes[0].set_ylabel("Radius [cm]")
            axes[0].legend()

            axes[1].loglog(time_grid.to_value(u.day), results["velocity"].to_value(u.cm / u.s), label="Computed")
            axes[1].loglog(time_grid.to_value(u.day), expected_v.to_value(u.cm / u.s), ls="--", label="Expected")
            axes[1].set_xlabel("Time [days]")
            axes[1].set_ylabel("Velocity [cm/s]")
            axes[1].set_ylim([v_0.to_value("cm/s") / 10, v_0.to_value("cm/s") * 10])
            axes[1].legend()

            axes[2].loglog(time_grid.to_value(u.day), results["mass"].to_value(u.g), label="Computed")
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
            results["radius"].to_value(u.cm),
            expected_R.to_value(u.cm),
            rtol=relative_tolerance,
        )
        assert_allclose(
            results["velocity"].to_value(u.cm / u.s),
            expected_v.to_value(u.cm / u.s),
            rtol=relative_tolerance,
        )
        assert_allclose(
            results["mass"].to_value(u.g),
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

        def _rho_csm(r):
            return K_csm_cgs * r**-s

        def _G(v):
            return K_ej_cgs * v**-n

        # --- Perform the computation --- #
        # Now we can perform the computation using the shock engine.
        results = engine.compute_shock_properties(
            time_grid,
            _rho_csm,
            _G,
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

        dlogR_dlott_computed = np.gradient(
            np.log10(results["radius"].to_value(u.cm)), np.log10(time_grid.to_value(u.s))
        )
        dlogv_dlott_computed = np.gradient(
            np.log10(results["velocity"].to_value(u.cm / u.s)), np.log10(time_grid.to_value(u.s))
        )
        dlogM_dlott_computed = np.gradient(np.log10(results["mass"].to_value(u.g)), np.log10(time_grid.to_value(u.s)))

        # Generate diagnostic plots if requested.
        if diagnostic_plots:
            import matplotlib.pyplot as plt

            from triceratops.utils.plot_utils import set_plot_style

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
            plt.savefig(f"{diagnostic_plots_dir}/test_homologous_expansion_in_vacuum.png")
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
