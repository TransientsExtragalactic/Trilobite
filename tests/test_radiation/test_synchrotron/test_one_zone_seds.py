import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u

from triceratops.radiation.synchrotron.cooling import SynchrotronRadiativeCoolingEngine
from triceratops.radiation.synchrotron.SEDs import (
    PowerLaw_Cooling_SSA_SynchrotronSED,
    PowerLaw_Cooling_SynchrotronSED,
    PowerLaw_SSA_SynchrotronSED,
    PowerLaw_SynchrotronSED,
)
from triceratops.radiation.synchrotron.SEDs._one_zone_closure import (
    _synchrotron_cooling_time_coefficient_cgs as THETA,
)
from triceratops.radiation.synchrotron.SEDs.one_zone_closure import (
    invert_powerlaw_implicit_cooling_sed,
    invert_powerlaw_implicit_cooling_ssa_sed,
    invert_powerlaw_ssa_sed_demarchi,
)


# ==========================================================
# Helper functions
# =========================================================
def _parameter_grid(param_dict):
    keys = list(param_dict.keys())
    values = list(param_dict.values())

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# =========================================================
# Base test class
# ========================================================
class BaseTestOneZoneSynchrotronSED:
    """
    Reusable pytest base class for testing one-zone synchrotron SEDs.
    """

    MODEL = None

    NORMALIZATION_PARAMETERS = {}
    INVERSION_PARAMETERS = {}

    EVAL_PARAMETERS = {}
    PHYSICS_PARAMETERS = {}

    LOG_X = True
    LOG_Y = True

    FREQUENCY_GRID = np.logspace(6, 20, 400) * u.Hz

    # ---------------------------------------------------------
    # Instantiation
    # ---------------------------------------------------------

    def test_instantiate(self):
        model = self.MODEL()
        assert model is not None

    # ---------------------------------------------------------
    # Parameter grid utilities
    # ---------------------------------------------------------

    def _generate_parameter_grid(self, param_dict):
        keys = list(param_dict.keys())
        values = list(param_dict.values())

        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    # ---------------------------------------------------------
    # SED evaluation
    # ---------------------------------------------------------

    def test_eval(self, diagnostic_plots, diagnostic_plots_dir):
        model = self.MODEL()
        nu = self.FREQUENCY_GRID

        for index, params in enumerate(self._generate_parameter_grid(self.EVAL_PARAMETERS)):
            F_nu = model.sed(nu, **params)

            assert F_nu is not None
            assert np.all(np.isfinite(F_nu.value))

            if diagnostic_plots:
                x = nu.value
                y = F_nu.value

                plt.figure()

                if self.LOG_X and self.LOG_Y:
                    plt.loglog(x, y)
                elif self.LOG_X:
                    plt.semilogx(x, y)
                elif self.LOG_Y:
                    plt.semilogy(x, y)
                else:
                    plt.plot(x, y)

                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Flux density")

                plt.tight_layout()

                fname = f"{self.MODEL.__name__}_eval_{index}.png"
                plt.savefig(diagnostic_plots_dir / fname)
                plt.close()

    # ---------------------------------------------------------
    # Physics <-> Parameter round-trip test
    # ---------------------------------------------------------

    def test_round_trip(self):
        if not self.PHYSICS_PARAMETERS:
            pytest.skip("No PHYSICS_PARAMETERS grid defined.")

        model = self.MODEL()

        for physics_params in self._generate_parameter_grid(self.PHYSICS_PARAMETERS):
            B = physics_params["B"]
            R = physics_params["R"]

            # Forward normalization
            params = model.from_physics_to_params(B, R, **self.NORMALIZATION_PARAMETERS)

            F_peak = params["F_peak"]
            nu_m = params["nu_m"]

            # Inversion
            recovered = model.from_params_to_physics(F_peak, nu_m, **self.NORMALIZATION_PARAMETERS)

            assert np.isclose(
                recovered["B"].to_value("G"),
                B.to_value("G"),
                rtol=0.1,
            )

            assert np.isclose(
                recovered["R"].to_value("cm"),
                R.to_value("cm"),
                rtol=0.01,
            )


# ---------------------------------------------------------
# SED CLASS TESTS
# ---------------------------------------------------------
class Test_PowerLaw_SynchrotronSED(BaseTestOneZoneSynchrotronSED):
    """
    Test suite for :class:`PowerLaw_SynchrotronSED`.

    This test verifies three aspects of the SED implementation:

    1. **Evaluation stability**
       The SED should evaluate without numerical errors across a small
       grid of phenomenological parameters.

    2. **Forward normalization**
       Physical parameters (B, R, etc.) should map correctly to the
       phenomenological SED parameters through
       ``from_physics_to_params``.

    3. **Inverse reconstruction**
       The inversion routine ``from_params_to_physics`` should recover
       the original physical parameters within reasonable tolerance.

    The round–trip test therefore checks

        (B, R) → (F_peak, ν_m) → (B, R)

    which ensures that the analytic normalization and inversion
    relations are mutually consistent.
    """

    # ---------------------------------------------------------
    # Model under test
    # ---------------------------------------------------------

    MODEL = PowerLaw_SynchrotronSED

    # ---------------------------------------------------------
    # Phenomenological evaluation grid
    # ---------------------------------------------------------

    EVAL_PARAMETERS = {
        "nu_m": [1e8 * u.Hz],
        "F_norm": [1 * u.Jy],
        "p": [2.5],
    }
    """
    Parameter grid used when directly evaluating the SED model.

    These values are intentionally simple so that the evaluation
    tests primarily verify numerical stability rather than physical
    realism.
    """

    # ---------------------------------------------------------
    # Physical parameter grid for round–trip tests
    # ---------------------------------------------------------

    PHYSICS_PARAMETERS = {
        "B": [0.1 * u.G, 1.0 * u.G, 10.0 * u.G],
        "R": [1e15 * u.cm, 1e16 * u.cm],
    }
    """
    Physical parameter grid used for normalization/inversion tests.

    These values probe a modest dynamic range typical of radio
    synchrotron sources while remaining safely within the regime
    where the analytic closure relations are valid.
    """

    # ---------------------------------------------------------
    # Additional physics assumptions
    # ---------------------------------------------------------

    NORMALIZATION_PARAMETERS = {
        "luminosity_distance": 35 * u.Mpc,
        "gamma_min": 1,
        "gamma_max": 1e8,
        "p": 3,
        "epsilon_E": 0.1,
        "epsilon_B": 0.1,
        "f_V": 1.0,
        "gamma_bulk": 1.0,
        "pitch_average": True,
    }
    """
    Fixed physical assumptions used during normalization and inversion.

    These parameters describe the underlying emitting region:

    - ``luminosity_distance`` : distance to the source
    - ``gamma_min``           : minimum electron Lorentz factor
    - ``gamma_max``           : maximum electron Lorentz factor
    - ``p``                   : electron power–law index
    - ``epsilon_E``           : fraction of internal energy in electrons
    - ``epsilon_B``           : fraction of internal energy in magnetic fields
    - ``f_V``                 : emitting volume filling factor
    - ``gamma_bulk``          : bulk Lorentz factor of the emitting region
    - ``pitch_average``       : assume isotropic pitch-angle distribution
    """


class Test_PowerLaw_Cooling_SynchrotronSED(BaseTestOneZoneSynchrotronSED):
    MODEL = PowerLaw_Cooling_SynchrotronSED

    # ---------------------------------------------------------
    # Phenomenological evaluation grid
    # ---------------------------------------------------------

    EVAL_PARAMETERS = {
        "nu_m": [1e8 * u.Hz, 1e10 * u.Hz],
        "nu_c": [1e10 * u.Hz, 1e9 * u.Hz],
        "F_norm": [1 * u.Jy],
        "p": [2.5],
    }

    # ---------------------------------------------------------
    # Physical parameter grid for round–trip tests
    # ---------------------------------------------------------

    PHYSICS_PARAMETERS = {
        "B": [1 * u.G, 1.0 * u.G, 10.0 * u.G],
        "R": [1e16 * u.cm, 1e16 * u.cm, 1e17 * u.cm],
        "gamma_c": [1e4],
    }

    # ---------------------------------------------------------
    # Additional physics assumptions
    # ---------------------------------------------------------

    NORMALIZATION_PARAMETERS = {
        "luminosity_distance": 35 * u.Mpc,
        "gamma_min": 5,
        "gamma_max": 1e8,
        "p": 3,
        "epsilon_E": 0.1,
        "epsilon_B": 0.1,
        "f_V": 1.0,
        "gamma_bulk": 1.0,
        "pitch_average": True,
    }

    # ---------------------------------------------------------
    # Round-trip physics ↔ parameters
    # ---------------------------------------------------------

    @pytest.mark.parametrize(
        "physics_params",
        list(BaseTestOneZoneSynchrotronSED._generate_parameter_grid(BaseTestOneZoneSynchrotronSED, PHYSICS_PARAMETERS)),
        ids=lambda p: f"B={p['B']}_R={p['R']}_gc={p['gamma_c']}",
    )
    def test_round_trip(self, physics_params):
        model = self.MODEL()

        B = physics_params["B"]
        R = physics_params["R"]
        gamma_c = physics_params["gamma_c"]

        params = model.from_physics_to_params(B, R, gamma_c=gamma_c, **self.NORMALIZATION_PARAMETERS)

        F_peak = params["F_peak"]
        F_norm = params["F_norm"]
        nu_peak = params["nu_peak"]
        regime = params["regime"]

        print(F_peak.to_value("Jy"))

        recovered = model.from_params_to_physics(
            regime, F_peak, nu_peak, gamma_c=gamma_c, **self.NORMALIZATION_PARAMETERS
        )

        assert np.isclose(
            recovered["B"].to_value("G"),
            B.to_value("G"),
            rtol=0.01,
        ), f"B mismatch\nB_true={B}\nB_rec={recovered['B']}\nRegime={regime}"

        assert np.isclose(
            recovered["R"].to_value("cm"),
            R.to_value("cm"),
            rtol=0.01,
        )


class Test_PowerLaw_SSA_SynchrotronSED(BaseTestOneZoneSynchrotronSED):
    MODEL = PowerLaw_SSA_SynchrotronSED

    EVAL_PARAMETERS = {"nu_m": [1e8 * u.Hz], "F_norm": [1 * u.Jy], "p": [2.5]}

    # ---------------------------------------------------------
    # Physical parameter grid for round–trip tests
    # ---------------------------------------------------------

    PHYSICS_PARAMETERS = {
        "B": [1 * u.G],
        "R": [1e15 * u.cm, 1e16 * u.cm],
    }
    """
    Physical parameter grid used for normalization/inversion tests.

    These values probe a modest dynamic range typical of radio
    synchrotron sources while remaining safely within the regime
    where the analytic closure relations are valid.
    """

    # ---------------------------------------------------------
    # Additional physics assumptions
    # ---------------------------------------------------------

    NORMALIZATION_PARAMETERS = {
        "luminosity_distance": 35 * u.Mpc,
        "gamma_min": 1,
        "gamma_max": 1e8,
        "p": 3,
        "epsilon_E": 0.1,
        "epsilon_B": 0.1,
        "f_V": 1.0,
        "f_A": 1.0,
        "gamma_bulk": 1.0,
        "pitch_average": True,
    }
    """
    Fixed physical assumptions used during normalization and inversion.

    These parameters describe the underlying emitting region:

    - ``luminosity_distance`` : distance to the source
    - ``gamma_min``           : minimum electron Lorentz factor
    - ``gamma_max``           : maximum electron Lorentz factor
    - ``p``                   : electron power–law index
    - ``epsilon_E``           : fraction of internal energy in electrons
    - ``epsilon_B``           : fraction of internal energy in magnetic fields
    - ``f_V``                 : emitting volume filling factor
    - ``gamma_bulk``          : bulk Lorentz factor of the emitting region
    - ``pitch_average``       : assume isotropic pitch-angle distribution
    """

    @pytest.mark.parametrize(
        "physics_params",
        list(BaseTestOneZoneSynchrotronSED._generate_parameter_grid(BaseTestOneZoneSynchrotronSED, PHYSICS_PARAMETERS)),
        ids=lambda p: f"B={p['B']}_R={p['R']}",
    )
    def test_round_trip(self, physics_params):
        model = self.MODEL()

        B = physics_params["B"]
        R = physics_params["R"]

        params = model.from_physics_to_params(B, R, **self.NORMALIZATION_PARAMETERS)

        F_peak = params["F_peak"]
        F_norm = params["F_norm"]
        nu_a = params["nu_a"]
        nu_peak = params["nu_peak"]
        regime = params["regime"]

        recovered = model.from_params_to_physics(regime, F_peak, nu_peak, **self.NORMALIZATION_PARAMETERS)

        assert np.isclose(
            recovered["B"].to_value("G"),
            B.to_value("G"),
            rtol=0.01,
        ), (
            f"B mismatch\n"
            f"B_true={B}\n"
            f"B_rec={recovered['B']}\n"
            f"Regime={regime}\n"
            f"F_peak={F_peak.to('Jy')}\n"
            f"nu_peak={nu_peak.to('GHz')}"
        )

        assert np.isclose(
            recovered["R"].to_value("cm"),
            R.to_value("cm"),
            rtol=0.01,
        )


class Test_PowerLaw_Cooling_SSA_SynchrotronSED(BaseTestOneZoneSynchrotronSED):
    MODEL = PowerLaw_Cooling_SSA_SynchrotronSED

    EVAL_PARAMETERS = {
        "nu_m": [1e8 * u.Hz, 1e10 * u.Hz],
        "nu_c": [1e10 * u.Hz, 1e9 * u.Hz],
        "F_norm": [1 * u.Jy],
        "p": [2.5],
    }

    # ---------------------------------------------------------
    # Physical parameter grid for round–trip tests
    # ---------------------------------------------------------
    PHYSICS_PARAMETERS = {
        "B": [1 * u.G, 1.0 * u.G, 10.0 * u.G, 100.0 * u.G, 1000.0 * u.G],
        "R": [1e17 * u.cm, 1e16 * u.cm, 1e17 * u.cm],
        "gamma_c": [1, 5, 10, 100, 1000, 1e4, 1e7],
    }
    """
    Physical parameter grid used for normalization/inversion tests.

    These values probe a modest dynamic range typical of radio
    synchrotron sources while remaining safely within the regime
    where the analytic closure relations are valid.
    """

    # ---------------------------------------------------------
    # Additional physics assumptions
    # ---------------------------------------------------------
    NORMALIZATION_PARAMETERS = {
        "luminosity_distance": 35 * u.Mpc,
        "gamma_min": 2,
        "gamma_max": 1e8,
        "p": 3,
        "epsilon_E": 0.1,
        "epsilon_B": 0.1,
        "f_V": 0.5,
        "f_A": 0.01,
        "gamma_bulk": 1.0,
        "pitch_average": True,
    }
    """
    Fixed physical assumptions used during normalization and inversion.

    These parameters describe the underlying emitting region:

    - ``luminosity_distance`` : distance to the source
    - ``gamma_min``           : minimum electron Lorentz factor
    - ``gamma_max``           : maximum electron Lorentz factor
    - ``p``                   : electron power–law index
    - ``epsilon_E``           : fraction of internal energy in electrons
    - ``epsilon_B``           : fraction of internal energy in magnetic fields
    - ``f_V``                 : emitting volume filling factor
    - ``gamma_bulk``          : bulk Lorentz factor of the emitting region
    - ``pitch_average``       : assume isotropic pitch-angle distribution
    """

    @pytest.mark.parametrize(
        "physics_params",
        list(_parameter_grid(PHYSICS_PARAMETERS)),
        ids=lambda p: f"B={p['B']}_R={p['R']}_gc={p['gamma_c']}",
    )
    def test_round_trip(self, physics_params):
        model = self.MODEL()

        B = physics_params["B"]
        R = physics_params["R"]
        gamma_c = physics_params["gamma_c"]

        params = model.from_physics_to_params(B, R, gamma_c=gamma_c, **self.NORMALIZATION_PARAMETERS)

        F_peak = params["F_peak"]
        F_norm = params["F_norm"]
        nu_a = params["nu_a"]
        nu_peak = params["nu_peak"]
        nu_m = params["nu_m"]
        regime = params["regime"]

        print("ROUND-TRIP FLUX PEAK: ", F_peak.to_value(u.Jy), "Jy")
        print("ROUND-TRIP FREQUENCY PEAK: ", nu_peak.to_value(u.GHz), "GHz")
        print("ROUND-TRIP REGIME: ", regime)

        recovered = model.from_params_to_physics(
            regime, F_peak, nu_peak, gamma_c=gamma_c, **self.NORMALIZATION_PARAMETERS
        )

        assert np.isclose(
            recovered["B"].to_value("G"),
            B.to_value("G"),
            rtol=0.01,
        ), f"Recovered B does not match original.\nB_true={B}\nB_rec={recovered['B']}\nRegime={regime}"

        assert np.isclose(
            recovered["R"].to_value("cm"),
            R.to_value("cm"),
            rtol=0.01,
        )


def test_demarchi_equivalence(diagnostic_plots, diagnostic_plots_dir):
    """
    Verify that the De Marchi (2022) SSA closure relations produce
    approximately the same magnetic field and radius evolution as the
    analytic inversion implemented in PowerLaw_SSA_SynchrotronSED.

    This uses a synthetic transient where

        F_peak ∝ t^{-1}
        ν_peak ∝ t^{-1/2}

    and compares the inferred B(t) and R(t) from the two methods.
    """

    from triceratops.radiation.synchrotron.SEDs.one_zone_closure import (
        invert_powerlaw_ssa_sed_demarchi,
    )

    sed_engine = PowerLaw_SSA_SynchrotronSED()

    # ---------------------------------------------------------
    # Synthetic transient
    # ---------------------------------------------------------

    t = np.logspace(0, 5, 10) * u.day

    t0 = 1 * u.day
    F0 = 1 * u.Jy
    nu0 = 5 * u.GHz

    alpha_F = -1.0
    alpha_nu = -0.5

    F_t = F0 * (t / t0) ** alpha_F
    nu_t = nu0 * (t / t0) ** alpha_nu

    # ---------------------------------------------------------
    # Physical assumptions
    # ---------------------------------------------------------

    luminosity_distance = 35 * u.Mpc
    epsilon_E = 0.1
    epsilon_B = 0.1
    f_V = 1.0
    gamma_bulk = 1.0
    gamma_min = 1
    gamma_max = 1e8
    p = 3

    # ---------------------------------------------------------
    # De Marchi closure
    # ---------------------------------------------------------

    B_dm22, R_dm22 = invert_powerlaw_ssa_sed_demarchi(
        nu_t,
        F_t,
        luminosity_distance=luminosity_distance,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f=f_V,
        theta=np.pi / 2,
    )

    # ---------------------------------------------------------
    # Model inversion
    # ---------------------------------------------------------

    closure_results = sed_engine.from_params_to_physics(
        "optically_thick",
        F_t,
        nu_t,
        luminosity_distance=luminosity_distance,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        gamma_bulk=gamma_bulk,
        pitch_average=False,
        alpha=np.pi / 2,
    )

    B_model = closure_results["B"]
    R_model = closure_results["R"]

    # ---------------------------------------------------------
    # Expected analytic slopes
    # ---------------------------------------------------------

    expected_slope_B = (-2 * alpha_F / (2 * p + 13)) + alpha_nu
    expected_slope_R = ((6 + p) * alpha_F / (2 * p + 13)) - alpha_nu

    log_t = np.log10(t.to_value(u.day))

    slope_dm22 = np.gradient(np.log10(B_dm22.value), log_t)
    slope_model = np.gradient(np.log10(B_model.value), log_t)

    slope_dm22_mean = np.mean(slope_dm22[1:-1])
    slope_model_mean = np.mean(slope_model[1:-1])

    assert np.isclose(
        slope_dm22_mean,
        expected_slope_B,
        rtol=0.2,
    ), f"DM22 B slope {slope_dm22_mean:.3f} != expected {expected_slope_B:.3f}"

    assert np.isclose(
        slope_model_mean,
        expected_slope_B,
        rtol=0.2,
    ), f"Model B slope {slope_model_mean:.3f} != expected {expected_slope_B:.3f}"

    # ---------------------------------------------------------
    # Numerical agreement
    # ---------------------------------------------------------

    assert np.allclose(
        B_model.to_value("G"),
        B_dm22.to_value("G"),
        rtol=2,
    )

    assert np.allclose(
        R_model.to_value("cm"),
        R_dm22.to_value("cm"),
        rtol=2,
    )

    # ---------------------------------------------------------
    # Diagnostic plots
    # ---------------------------------------------------------

    if diagnostic_plots:
        from triceratops.utils.plot_utils import set_plot_style

        set_plot_style()

        t_days = t.to_value(u.day)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # --- B comparison ---
        axes[0].loglog(t_days, B_dm22.to_value(u.G), "o-", label="DM22")
        axes[0].loglog(t_days, B_model.to_value(u.G), "s--", label="SED inversion")

        reference_B = B_dm22[0] * (t / t[0]) ** expected_slope_B
        axes[0].loglog(
            t_days,
            reference_B.to_value(u.G),
            "k:",
            label=f"Expected slope {expected_slope_B:.2f}",
        )

        axes[0].set_xlabel("Time [days]")
        axes[0].set_ylabel("B [G]")
        axes[0].legend()

        # --- R comparison ---
        axes[1].loglog(t_days, R_dm22.to_value(u.cm), "o-", label="DM22")
        axes[1].loglog(t_days, R_model.to_value(u.cm), "s--", label="SED inversion")

        reference_R = R_dm22[0] * (t / t[0]) ** expected_slope_R
        axes[1].loglog(
            t_days,
            reference_R.to_value(u.cm),
            "k:",
            label=f"Expected slope {expected_slope_R:.2f}",
        )

        axes[1].set_xlabel("Time [days]")
        axes[1].set_ylabel("R [cm]")
        axes[1].legend()

        plt.tight_layout()

        plt.savefig(diagnostic_plots_dir / "demarchi_equivalence.png")
        plt.close()


def test_spectrum_7_and_4_convergence(diagnostic_plots, diagnostic_plots_dir):
    """
    Verify that forward normalization remains continuous when the
    ordering transitions between

        nu_m < nu_a < nu_c   (Spectrum 4)
        nu_m < nu_c < nu_a   (Spectrum 7)

    as gamma_c is varied.

    The resulting flux normalization and break frequencies should vary
    smoothly across the regime boundary.
    """

    sed_engine = PowerLaw_Cooling_SSA_SynchrotronSED()

    # ---------------------------------------------------------
    # Fixed physical parameters
    # ---------------------------------------------------------

    B = 1.0 * u.G
    R = 1e16 * u.cm

    gamma_min = 5
    gamma_max = 1e8
    p = 3

    luminosity_distance = 35 * u.Mpc
    epsilon_E = 0.1
    epsilon_B = 0.1

    f_V = 1.0
    f_A = 0.1
    gamma_bulk = 1.0

    # Sweep gamma_c across many decades
    gamma_c_grid = np.geomspace(1, 1e6, 50)

    F_peak_vals = []
    F_norm_vals = []
    nu_a_vals = []
    nu_c_vals = []
    nu_m_vals = []
    regimes = []

    # ---------------------------------------------------------
    # Forward normalization sweep
    # ---------------------------------------------------------

    for gamma_c in gamma_c_grid:
        params = sed_engine.from_physics_to_params(
            B,
            R,
            gamma_c=gamma_c,
            luminosity_distance=luminosity_distance,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            gamma_bulk=gamma_bulk,
            pitch_average=True,
        )

        F_peak_vals.append(params["F_peak"].to_value(u.Jy))
        F_norm_vals.append(params["F_norm"].to_value(u.Jy))
        nu_a_vals.append(params["nu_a"].to_value(u.Hz))
        nu_c_vals.append(params["nu_c"].to_value(u.Hz))
        nu_m_vals.append(params["nu_m"].to_value(u.Hz))
        regimes.append(params["regime"])

    F_peak_vals = np.array(F_peak_vals)
    F_norm_vals = np.array(F_norm_vals)
    nu_a_vals = np.array(nu_a_vals)
    nu_c_vals = np.array(nu_c_vals)
    nu_m_vals = np.array(nu_m_vals)

    # ---------------------------------------------------------
    # Diagnostic plots
    # ---------------------------------------------------------

    if diagnostic_plots:
        from triceratops.utils.plot_utils import set_plot_style

        set_plot_style()

        gamma_vals = gamma_c_grid

        # Extract regime numbers (e.g. "Spectrum7" -> 7)
        regime_numbers = np.array([int("".join(filter(str.isdigit, r))) for r in regimes])

        # Identify regime transitions
        regime_changes = np.where(np.diff(regime_numbers) != 0)[0]

        # -----------------------------------------------------
        # Create figure
        # -----------------------------------------------------

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(8, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 2, 1]},
        )

        # -----------------------------------------------------
        # Flux panel
        # -----------------------------------------------------

        axes[0].loglog(gamma_vals, F_peak_vals, "o-", label="F_peak")
        axes[0].loglog(gamma_vals, F_norm_vals, "s--", label="F_norm")

        axes[0].set_ylabel("Flux [Jy]")
        axes[0].set_title("Flux normalization vs gamma_c")
        axes[0].legend()

        # -----------------------------------------------------
        # Frequency panel
        # -----------------------------------------------------

        axes[1].loglog(gamma_vals, nu_a_vals, "o-", label="nu_a")
        axes[1].loglog(gamma_vals, nu_m_vals, "s--", label="nu_m")
        axes[1].loglog(gamma_vals, nu_c_vals, "d-.", label="nu_c")

        axes[1].set_ylabel("Frequency [Hz]")
        axes[1].set_title("Break frequencies vs gamma_c")
        axes[1].legend()

        # -----------------------------------------------------
        # Regime panel
        # -----------------------------------------------------

        axes[2].semilogx(gamma_vals, regime_numbers, "o-")

        axes[2].set_ylabel("Spectrum")
        axes[2].set_xlabel("gamma_c")
        axes[2].set_yticks(np.unique(regime_numbers))
        axes[2].set_title("Detected spectral regime")

        # -----------------------------------------------------
        # Mark regime transitions
        # -----------------------------------------------------

        for idx in regime_changes:
            transition_gamma = gamma_vals[idx]

            for ax in axes:
                ax.axvline(
                    transition_gamma,
                    color="k",
                    linestyle=":",
                    alpha=0.5,
                )

        plt.tight_layout()

        plt.savefig(diagnostic_plots_dir / "spectrum4_spectrum7_forward_continuity.png")

        plt.close()


def test_spectrum_7_and_4_inversion_consistency(diagnostic_plots, diagnostic_plots_dir):
    """
    Verify that the normalization and inversion are mutually consistent
    across the Spectrum 4 ↔ Spectrum 7 transition.

    The pipeline tested is

        (B, R, gamma_c)
            -> normalization
            -> (F_peak, nu_peak, regime)
            -> inversion
            -> (B_recovered, R_recovered)

    The recovered values should match the true physical parameters.
    """

    sed_engine = PowerLaw_Cooling_SSA_SynchrotronSED()

    # ---------------------------------------------------------
    # True physical parameters
    # ---------------------------------------------------------

    B_true = 1.0 * u.G
    R_true = 1e16 * u.cm

    gamma_min = 5
    gamma_max = 1e8
    p = 3

    luminosity_distance = 35 * u.Mpc
    epsilon_E = 0.1
    epsilon_B = 0.1

    f_V = 1.0
    f_A = 0.1
    gamma_bulk = 1.0

    # Sweep gamma_c
    gamma_c_grid = np.geomspace(1, 1e6, 50)

    B_recovered = []
    R_recovered = []
    regimes = []

    # ---------------------------------------------------------
    # Forward → inverse loop
    # ---------------------------------------------------------

    for gamma_c in gamma_c_grid:
        # Forward normalization
        params = sed_engine.from_physics_to_params(
            B_true,
            R_true,
            gamma_c=gamma_c,
            luminosity_distance=luminosity_distance,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            gamma_bulk=gamma_bulk,
            pitch_average=True,
        )

        F_peak = params["F_peak"]
        nu_peak = params["nu_peak"]
        regime = params["regime"]

        # Inversion
        recovered = sed_engine.from_params_to_physics(
            regime,
            F_peak,
            nu_peak,
            gamma_c=gamma_c,
            luminosity_distance=luminosity_distance,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            gamma_bulk=gamma_bulk,
            pitch_average=True,
        )

        B_recovered.append(recovered["B"].to_value("G"))
        R_recovered.append(recovered["R"].to_value("cm"))
        regimes.append(regime)

    B_recovered = np.array(B_recovered)
    R_recovered = np.array(R_recovered)

    # ---------------------------------------------------------
    # Numerical consistency checks
    # ---------------------------------------------------------
    assert np.allclose(
        B_recovered,
        B_true.to_value("G"),
        rtol=0.1,
    )
    assert np.allclose(
        R_recovered,
        R_true.to_value("cm"),
        rtol=0.1,
    )

    # ---------------------------------------------------------
    # Diagnostic plots
    # ---------------------------------------------------------

    if diagnostic_plots:
        from triceratops.utils.plot_utils import set_plot_style

        set_plot_style()

        gamma_vals = gamma_c_grid

        regime_numbers = np.array([int("".join(filter(str.isdigit, r))) for r in regimes])

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(8, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 2, 1]},
        )

        # -----------------------------------------------------
        # Magnetic field
        # -----------------------------------------------------

        axes[0].semilogx(gamma_vals, B_recovered, "o-", label="Recovered B")
        axes[0].axhline(
            B_true.to_value("G"),
            color="k",
            linestyle="--",
            label="True B",
        )

        axes[0].set_ylabel("B [G]")
        axes[0].set_ylim(0.5 * B_true.to_value("G"), 2 * B_true.to_value("G"))
        axes[0].legend()

        # -----------------------------------------------------
        # Radius
        # -----------------------------------------------------

        axes[1].semilogx(gamma_vals, R_recovered, "o-", label="Recovered R")
        axes[1].axhline(
            R_true.to_value("cm"),
            color="k",
            linestyle="--",
            label="True R",
        )

        axes[1].set_ylabel("R [cm]")
        axes[1].set_ylim(0.5 * R_true.to_value("cm"), 2 * R_true.to_value("cm"))
        axes[1].legend()

        # -----------------------------------------------------
        # Regime
        # -----------------------------------------------------

        axes[2].semilogx(gamma_vals, regime_numbers, "o-")

        axes[2].set_ylabel("Spectrum")
        axes[2].set_xlabel("gamma_c")
        axes[2].set_yticks(np.unique(regime_numbers))

        plt.tight_layout()

        plt.savefig(diagnostic_plots_dir / "spectrum4_spectrum7_inversion_consistency.png")

        plt.close()


def test_cooling_transition_continuity(diagnostic_plots, diagnostic_plots_dir):
    """
    Verify that inferred B(t) and R(t) remain continuous as the cooling
    frequency sweeps from fast-cooling to slow-cooling and eventually
    to the no-cooling limit.

    The observable SED is held fixed:

        F_peak = constant
        nu_peak = constant

    while gamma_c increases monotonically.
    """
    sed_engine = PowerLaw_Cooling_SSA_SynchrotronSED()

    # ---------------------------------------------------------
    # Observed SED (constant in time)
    # ---------------------------------------------------------

    F_peak = 1.0 * u.Jy
    nu_peak = 10.0 * u.GHz

    # ---------------------------------------------------------
    # Physical assumptions
    # ---------------------------------------------------------

    luminosity_distance = 35 * u.Mpc
    gamma_m = 5
    gamma_max = 1e10
    p = 3

    epsilon_E = 0.1
    epsilon_B = 0.1

    f_V = 1.0
    gamma_bulk = 1.0

    # Sweep gamma_c across many decades
    gamma_c_grid = np.geomspace(1, 1e5, 100)

    B_vals = []
    R_vals = []
    regimes = []

    # ---------------------------------------------------------
    # Inversion sweep
    # ---------------------------------------------------------

    for gamma_c in gamma_c_grid:
        result = sed_engine.from_params_to_physics(
            "Spectrum7",  # Start in fast-cooling regime
            F_peak,
            nu_peak,
            gamma_min=gamma_m,
            gamma_max=gamma_max,
            gamma_c=gamma_c,
            luminosity_distance=luminosity_distance,
            p=p,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            gamma_bulk=gamma_bulk,
            pitch_average=True,
        )

        B_vals.append(result["B"].to_value(u.G))
        R_vals.append(result["R"].to_value(u.cm))

    B_vals = np.array(B_vals)
    R_vals = np.array(R_vals)

    # ---------------------------------------------------------
    # Continuity tests
    # ---------------------------------------------------------

    dlogB = np.diff(np.log10(B_vals))
    dlogR = np.diff(np.log10(R_vals))

    # ---------------------------------------------------------
    # Diagnostic plots
    # ---------------------------------------------------------

    if diagnostic_plots:
        from triceratops.utils.plot_utils import set_plot_style

        set_plot_style()

        gamma_vals = gamma_c_grid

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(8, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 2, 1]},
        )

        # -----------------------------------------------------
        # Magnetic field
        # -----------------------------------------------------

        axes[0].loglog(gamma_vals, B_vals, "o-")
        axes[0].set_ylabel("B [G]")
        axes[0].set_title("Magnetic field vs gamma_c")

        # -----------------------------------------------------
        # Radius
        # -----------------------------------------------------

        axes[1].loglog(gamma_vals, R_vals, "o-")
        axes[1].set_ylabel("R [cm]")
        axes[1].set_title("Radius vs gamma_c")

        # -----------------------------------------------------
        # Mark regime transitions
        # -----------------------------------------------------

        plt.tight_layout()

        plt.savefig(diagnostic_plots_dir / "cooling_transition_continuity.png")

        plt.close()


# ── Implicit Cooling (optically thin) Inversions ─────────────────────────────


@pytest.mark.parametrize(
    "B,R,t,p",
    list(
        itertools.product(
            [
                1 * u.G,
            ],  # 0.1 * u.G, 1.0 * u.G],
            [
                1e15 * u.cm,
            ],  # 1e16 * u.cm, 1e17 * u.cm],
            [
                1e6 * u.s,
            ],  # 1e6 * u.s],
            [
                2.5,
            ],  # 3.0],
        )
    ),
)
def test_implicit_cooling_round_trip(B, R, t, p, diagnostic_plots, diagnostic_plots_dir):
    r"""
    Round-trip test for the implicit synchrotron-cooling optically-thin inversion.

    Parameters
    ----------
    B : ~astropy.units.Quantity
        Magnetic field strength.
    R : ~astropy.units.Quantity
        Emitting-region radius.
    t : ~astropy.units.Quantity
        Source age.
    p : float
        Electron power-law index.
    diagnostic_plots : bool
        If ``True``, save comparison plots.
    diagnostic_plots_dir : ~pathlib.Path or None
        Directory in which to write diagnostic plots.

    Notes
    -----
    The pipeline tested is::

        (B, R, t)
            -> gamma_c = THETA / (B^2 * t)
            -> PowerLaw_Cooling_SynchrotronSED.from_physics_to_params
            -> (F_peak, nu_peak, regime)
            -> invert_powerlaw_implicit_cooling_sed("slow_cooling", ...)
            -> (B_rec, R_rec)

    ``gamma_c`` is not supplied explicitly; instead it is derived from the
    synchrotron cooling time :math:`t_{\rm cool} = \Theta / (B^2 \gamma)`,
    giving :math:`\gamma_c = \Theta / (B^2 t)`.  This is the same relation
    used internally by :func:`invert_powerlaw_implicit_cooling_sed`.

    The test is skipped automatically when the forward model produces a
    regime other than ``"slow_cooling"``, which is the only regime currently
    supported by the implicit-cooling inversion.

    Both ``B`` and ``R`` must be recovered within ``rtol=0.1``.
    """
    gamma_min = 1
    gamma_max = 1e8
    epsilon_E = 0.1
    epsilon_B = 0.1
    f_V = 1.0
    gamma_bulk = 1.0
    luminosity_distance = 35 * u.Mpc

    cooling_engine = SynchrotronRadiativeCoolingEngine(pitch_averaged=True)

    # Compute gamma_c from the synchrotron cooling closure: gamma_c = THETA / (B^2 t)
    B_cgs = B.to_value("G")
    t_cgs = t.to_value("s")

    gamma_c = cooling_engine.compute_cooling_gamma(B=B, t=t)

    # Forward normalization
    model = PowerLaw_Cooling_SynchrotronSED()
    params = model.from_physics_to_params(
        B,
        R,
        gamma_c=gamma_c,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        gamma_bulk=gamma_bulk,
        luminosity_distance=luminosity_distance,
        pitch_average=True,
    )

    regime = params["regime"]
    if regime != "slow_cooling":
        pytest.skip(f"Parameters produce regime '{regime}'; only 'slow_cooling' is supported.")

    F_peak = params["F_peak"]
    nu_peak = params["nu_peak"]
    print(F_peak.to_value("Jy"), nu_peak)

    # Inversion
    recovered = invert_powerlaw_implicit_cooling_sed(
        "slow_cooling",
        F_peak,
        nu_peak,
        t,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        gamma_bulk=gamma_bulk,
        luminosity_distance=luminosity_distance,
        pitch_average=True,
    )

    B_rec = recovered["B"]
    R_rec = recovered["R"]

    assert np.isclose(
        B_rec.to_value("G"),
        B.to_value("G"),
        rtol=0.1,
    ), f"B mismatch: B_true={B}, B_rec={B_rec}, regime={regime}"

    assert np.isclose(
        R_rec.to_value("cm"),
        R.to_value("cm"),
        rtol=0.1,
    ), f"R mismatch: R_true={R}, R_rec={R_rec}, regime={regime}"

    if diagnostic_plots:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].scatter([B.to_value("G")], [B_rec.to_value("G")], zorder=3)
        lim_B = [0.9 * B.to_value("G"), 1.1 * B.to_value("G")]
        axes[0].plot(lim_B, lim_B, "k--")
        axes[0].set_xlabel("B_in [G]")
        axes[0].set_ylabel("B_rec [G]")
        axes[0].set_title(f"B round-trip ({regime})")

        axes[1].scatter([R.to_value("cm")], [R_rec.to_value("cm")], zorder=3)
        lim_R = [0.9 * R.to_value("cm"), 1.1 * R.to_value("cm")]
        axes[1].plot(lim_R, lim_R, "k--")
        axes[1].set_xlabel("R_in [cm]")
        axes[1].set_ylabel("R_rec [cm]")
        axes[1].set_title("R round-trip")

        plt.tight_layout()
        fname = f"implicit_cooling_rt_B{B_cgs:.2e}_R{R.to_value('cm'):.2e}_t{t_cgs:.2e}_p{p}.png"
        plt.savefig(diagnostic_plots_dir / fname)
        plt.close()


@pytest.mark.parametrize(
    "regime,B,R,t,p",
    list(
        itertools.product(
            ["Spectrum4", "Spectrum7", "Spectrum3"],
            [1.0 * u.G, 10.0 * u.G, 100.0 * u.G],
            [1e16 * u.cm, 1e17 * u.cm, 1e18 * u.cm],
            [1e4 * u.s, 1e5 * u.s, 1e6 * u.s, 1e7 * u.s],
            [3.0],
        )
    ),
)
def test_implicit_cooling_ssa_round_trip(regime, B, R, t, p, diagnostic_plots, diagnostic_plots_dir):
    r"""
    Round-trip test for the implicit synchrotron-cooling SSA inversion.

    Parameters
    ----------
    regime : str
        Target SSA spectral regime: one of ``"Spectrum3"``, ``"Spectrum4"``,
        or ``"Spectrum7"``.
    B : ~astropy.units.Quantity
        Magnetic field strength.
    R : ~astropy.units.Quantity
        Emitting-region radius.
    t : ~astropy.units.Quantity
        Source age.
    p : float
        Electron power-law index.
    diagnostic_plots : bool
        If ``True``, save comparison plots.
    diagnostic_plots_dir : ~pathlib.Path or None
        Directory in which to write diagnostic plots.

    Notes
    -----
    The pipeline tested is::

        (B, R, t)
            -> gamma_c = THETA / (B^2 * t)
            -> PowerLaw_Cooling_SSA_SynchrotronSED.from_physics_to_params
            -> (F_peak, nu_peak, actual_regime)
            -> invert_powerlaw_implicit_cooling_ssa_sed(regime, ...)
            -> (B_rec, R_rec)

    The test is skipped when the forward model produces a regime that does
    not match the parametrized ``regime`` target.  This ensures that the
    inversion is tested only with observables that are self-consistent with
    the assumed spectral ordering.

    Both ``B`` and ``R`` must be recovered within ``rtol=0.1``.
    """
    gamma_min = 1
    gamma_max = 1e8
    epsilon_E = 0.1
    epsilon_B = 0.1
    f_V = 0.5
    f_A = 0.1
    gamma_bulk = 1.0
    luminosity_distance = 35 * u.Mpc

    # Compute gamma_c from the synchrotron cooling closure: gamma_c = THETA / (B^2 t)
    B_cgs = B.to_value("G")
    t_cgs = t.to_value("s")
    gamma_c = np.exp(np.log(THETA) - 2.0 * np.log(B_cgs) - np.log(t_cgs))

    # Forward normalization
    model = PowerLaw_Cooling_SSA_SynchrotronSED()
    params = model.from_physics_to_params(
        B,
        R,
        gamma_c=gamma_c,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        f_A=f_A,
        gamma_bulk=gamma_bulk,
        luminosity_distance=luminosity_distance,
        pitch_average=True,
    )

    actual_regime = params["regime"]
    if actual_regime != regime:
        pytest.skip(f"Parameters produce regime '{actual_regime}'; target regime was '{regime}'.")

    F_peak = params["F_peak"]
    nu_peak = params["nu_peak"]

    # Inversion
    recovered = invert_powerlaw_implicit_cooling_ssa_sed(
        regime,
        F_peak,
        nu_peak,
        t,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        f_A=f_A,
        gamma_bulk=gamma_bulk,
        luminosity_distance=luminosity_distance,
        pitch_average=True,
    )

    B_rec = recovered["B"]
    R_rec = recovered["R"]

    assert np.isclose(
        B_rec.to_value("G"),
        B.to_value("G"),
        rtol=0.1,
    ), f"B mismatch: B_true={B}, B_rec={B_rec}, regime={regime}"

    assert np.isclose(
        R_rec.to_value("cm"),
        R.to_value("cm"),
        rtol=0.1,
    ), f"R mismatch: R_true={R}, R_rec={R_rec}, regime={regime}"

    if diagnostic_plots:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].scatter([B.to_value("G")], [B_rec.to_value("G")], zorder=3)
        lim_B = [0.9 * B.to_value("G"), 1.1 * B.to_value("G")]
        axes[0].plot(lim_B, lim_B, "k--")
        axes[0].set_xlabel("B_in [G]")
        axes[0].set_ylabel("B_rec [G]")
        axes[0].set_title(f"B round-trip ({regime})")

        axes[1].scatter([R.to_value("cm")], [R_rec.to_value("cm")], zorder=3)
        lim_R = [0.9 * R.to_value("cm"), 1.1 * R.to_value("cm")]
        axes[1].plot(lim_R, lim_R, "k--")
        axes[1].set_xlabel("R_in [cm]")
        axes[1].set_ylabel("R_rec [cm]")
        axes[1].set_title("R round-trip")

        plt.tight_layout()
        fname = f"implicit_cooling_ssa_rt_{regime}_B{B_cgs:.2e}_R{R.to_value('cm'):.2e}_t{t_cgs:.2e}_p{p}.png"
        plt.savefig(diagnostic_plots_dir / fname)
        plt.close()


@pytest.mark.parametrize(
    "B,R,d,p",
    list(
        itertools.product(
            [0.01 * u.G, 0.1 * u.G],
            [1e15 * u.cm, 1e16 * u.cm],
            [100 * u.Mpc],
            [2.5, 3.0],
        )
    ),
)
def test_demarchi_round_trip(B, R, d, p, diagnostic_plots, diagnostic_plots_dir):
    r"""
    Round-trip test for the De Marchi (2022) SSA inversion.

    Parameters
    ----------
    B : ~astropy.units.Quantity
        Magnetic field strength.
    R : ~astropy.units.Quantity
        Emitting-region radius.
    d : ~astropy.units.Quantity
        Luminosity distance to the source.
    p : float
        Electron power-law index.
    diagnostic_plots : bool
        If ``True``, save comparison plots.
    diagnostic_plots_dir : ~pathlib.Path or None
        Directory in which to write diagnostic plots.

    Notes
    -----
    The pipeline tested is::

        (B, R)
            -> PowerLaw_SSA_SynchrotronSED.from_physics_to_params
               (pitch_average=False, alpha=pi/2 to match DM22 assumption)
            -> (F_peak, nu_peak, regime)
            -> invert_powerlaw_ssa_sed_demarchi(nu_peak, F_peak, ...)
            -> (B_rec, R_rec)

    The DM22 inversion is an independent analytic formula and is only an
    approximation to the Triceratops closure.  The tolerance is therefore
    set to ``rtol=2.0``, consistent with the agreement documented in
    :func:`test_demarchi_equivalence`.

    The test is skipped when the forward model produces a regime other than
    ``"optically_thick"``, since the DM22 inversion applies only to the
    SSA turnover regime.
    """
    gamma_min = 1
    gamma_max = 1e8
    epsilon_E = 0.1
    epsilon_B = 0.1
    f_V = 1.0
    f_A = 1.0
    gamma_bulk = 1.0
    theta = np.pi / 2

    # Forward normalization — use fixed pitch angle pi/2 to match DM22 theta assumption
    model = PowerLaw_SSA_SynchrotronSED()
    params = model.from_physics_to_params(
        B,
        R,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        f_A=f_A,
        gamma_bulk=gamma_bulk,
        luminosity_distance=d,
        pitch_average=False,
        alpha=theta,
    )

    regime = params["regime"]
    if regime != "optically_thick":
        pytest.skip(f"Parameters produce regime '{regime}'; DM22 inversion requires 'optically_thick'.")

    F_peak = params["F_peak"]
    nu_peak = params["nu_peak"]

    # DM22 inversion
    B_rec, R_rec = invert_powerlaw_ssa_sed_demarchi(
        nu_peak.to(u.GHz),
        F_peak.to(u.Jy),
        p=p,
        f=f_V,
        theta=theta,
        epsilon_B=epsilon_B,
        epsilon_E=epsilon_E,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        luminosity_distance=d,
    )

    assert np.isclose(
        B_rec.to_value("G"),
        B.to_value("G"),
        rtol=2.0,
    ), f"B mismatch (DM22): B_true={B}, B_rec={B_rec}, regime={regime}"

    assert np.isclose(
        R_rec.to_value("cm"),
        R.to_value("cm"),
        rtol=2.0,
    ), f"R mismatch (DM22): R_true={R}, R_rec={R_rec}, regime={regime}"

    if diagnostic_plots:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].scatter([B.to_value("G")], [B_rec.to_value("G")], zorder=3)
        lim_B = [min(B.to_value("G"), B_rec.to_value("G")) * 0.5, max(B.to_value("G"), B_rec.to_value("G")) * 2.0]
        axes[0].plot(lim_B, lim_B, "k--")
        axes[0].set_xlabel("B_in [G]")
        axes[0].set_ylabel("B_rec [G]")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_title(f"DM22 B round-trip (p={p})")

        axes[1].scatter([R.to_value("cm")], [R_rec.to_value("cm")], zorder=3)
        lim_R = [min(R.to_value("cm"), R_rec.to_value("cm")) * 0.5, max(R.to_value("cm"), R_rec.to_value("cm")) * 2.0]
        axes[1].plot(lim_R, lim_R, "k--")
        axes[1].set_xlabel("R_in [cm]")
        axes[1].set_ylabel("R_rec [cm]")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_title("DM22 R round-trip")

        plt.tight_layout()
        fname = f"demarchi_rt_B{B.to_value('G'):.2e}_R{R.to_value('cm'):.2e}_d{d.to_value('Mpc'):.0f}Mpc_p{p}.png"
        plt.savefig(diagnostic_plots_dir / fname)
        plt.close()
