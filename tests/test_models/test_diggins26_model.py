"""Tests for :class:`~trilobite.models.LFBOTs.Diggins26_TwoComponentModel`."""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from scipy.integrate import simpson

from trilobite.data.core import InferenceData, Observable
from trilobite.dynamics.profiles.csm import AxisymmetricCSMDensityProfile
from trilobite.inference import InferenceProblem, LogGaussianCensoredLikelihood
from trilobite.models.LFBOTs import (
    Diggins26_TwoComponentModel,
    Diggins26Dynamics,
    Diggins26Tabulation,
)

DAY = 86400.0
M_P = const.m_p.cgs.value
M_E = const.m_e.cgs.value
C = const.c.cgs.value
SIGMA_T = const.sigma_T.cgs.value


@pytest.fixture(scope="module")
def model():
    return Diggins26_TwoComponentModel(luminosity_distance=60 * u.Mpc)


# A modest grid keeps the tests quick: 4 epochs and 5 frequencies.
TIMES = np.array([5.0, 20.0, 80.0, 300.0]) * DAY
FREQS = np.array([1e9, 5e9, 2e10, 1e11, 3e11])


# =========================================================================== #
# Construction                                                                #
# =========================================================================== #
class TestConstruction:
    def test_requires_exactly_one_distance(self):
        with pytest.raises(ValueError, match="Exactly one"):
            Diggins26_TwoComponentModel()
        with pytest.raises(ValueError, match="Exactly one"):
            Diggins26_TwoComponentModel(redshift=0.01, luminosity_distance=60 * u.Mpc)
        with pytest.raises(ValueError, match="Exactly one"):
            Diggins26_TwoComponentModel(luminosity_distance=60 * u.Mpc, angular_diameter_distance=55 * u.Mpc)

    def test_angular_diameter_distance_constructor(self, model):
        m = Diggins26_TwoComponentModel(angular_diameter_distance=model.angular_diameter_distance)
        assert m.redshift == pytest.approx(model.redshift, rel=1e-6)
        assert m.luminosity_distance.to_value(u.cm) == pytest.approx(model.luminosity_distance.to_value(u.cm), rel=1e-6)
        bare = Diggins26_TwoComponentModel(angular_diameter_distance=model.angular_diameter_distance.to_value(u.Mpc))
        assert bare.redshift == pytest.approx(model.redshift, rel=1e-6)

    def test_distances_are_consistent(self, model):
        assert model.luminosity_distance.to_value(u.Mpc) == pytest.approx(60.0)
        z = model.redshift
        assert 0.012 < z < 0.015
        ratio = (model.luminosity_distance / model.angular_diameter_distance).to_value(u.dimensionless_unscaled)
        assert ratio == pytest.approx((1 + z) ** 2, rel=1e-8)

    def test_redshift_constructor(self):
        m = Diggins26_TwoComponentModel(redshift=0.1)
        assert m.redshift == 0.1

    def test_composition_sets_mean_molecular_weights(self):
        m = Diggins26_TwoComponentModel(redshift=0.01, hydrogen_mass_fraction=0.7, helium_mass_fraction=0.28)
        assert m.mean_molecular_weight == pytest.approx(1 / (2 * 0.7 + 0.75 * 0.28 + 0.5 * 0.02))
        assert m.mean_molecular_weight_per_electron == pytest.approx(1 / (0.7 + 0.5 * 0.28 + 0.5 * 0.02))
        # The shock engine must use the same mean molecular weight.
        assert m._shock_engine._mu == pytest.approx(m.mean_molecular_weight)

    def test_helium_defaults_to_the_remainder(self):
        m = Diggins26_TwoComponentModel(redshift=0.01, hydrogen_mass_fraction=0.6)
        assert m._Y == pytest.approx(0.4)
        assert m._Z_metals == 0.0

    def test_integrator_kw_defaults_and_overrides(self):
        assert Diggins26_TwoComponentModel(redshift=0.01)._integrator_kw == {"method": "LSODA", "rtol": 1e-7}
        m = Diggins26_TwoComponentModel(redshift=0.01, integrator_kw={"method": "Radau", "rtol": 1e-8})
        assert m._integrator_kw == {"method": "Radau", "rtol": 1e-8}

    def test_integrator_kw_reaches_the_solver(self, monkeypatch):
        m = Diggins26_TwoComponentModel(redshift=0.01, integrator_kw={"atol": 1e-3}, adiabatic_index=1.4)
        seen = {}
        original = m._shock_engine._compute_shock_properties_cgs

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(m._shock_engine, "_compute_shock_properties_cgs", spy)
        m.forward_model({"time": 20 * DAY, "frequency": 5e9}, {})
        assert seen["atol"] == 1e-3 and seen["rtol"] == 1e-7 and seen["method"] == "LSODA" and seen["gamma"] == 1.4

    def test_seed_luminosity_must_be_callable(self):
        with pytest.raises(TypeError):
            Diggins26_TwoComponentModel(redshift=0.01, seed_photon_luminosity=1e43)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_initial_beta": 1.5},
            {"shock_start_time": 0.0},
            {"reference_radius": -1.0},
            {"inner_radius": -1.0},
            {"gamma_points_per_decade": 0},
            {"integrator_kw": {"rtol": 0.0}},
            {"integrator_kw": {"t_eval": [1.0]}},
            {"adiabatic_index": 1.0},
            {"hydrogen_mass_fraction": 1.2},
            {"hydrogen_mass_fraction": 0.7, "helium_mass_fraction": 0.5},
        ],
    )
    def test_invalid_configuration_raises(self, kwargs):
        with pytest.raises(ValueError):
            Diggins26_TwoComponentModel(redshift=0.01, **kwargs)

    def test_registered_for_serialization(self, model):
        kwargs = model._model_init_kwargs
        assert kwargs["inner_radius"] == model._r_min
        assert kwargs["redshift"] == model.redshift
        rebuilt = Diggins26_TwoComponentModel(**kwargs)
        assert rebuilt.mean_molecular_weight == pytest.approx(model.mean_molecular_weight)
        assert rebuilt._integrator_kw == model._integrator_kw
        assert rebuilt.angular_diameter_distance.to_value(u.cm) == pytest.approx(
            model.angular_diameter_distance.to_value(u.cm), rel=1e-12
        )


# =========================================================================== #
# Forward model                                                               #
# =========================================================================== #
class TestForwardModel:
    def test_output_units_and_shape(self, model):
        out = model.forward_model({"time": TIMES, "frequency": np.full_like(TIMES, 5e9)}, {})
        assert out.flux_density.unit == u.Jy
        assert out.flux_density.shape == TIMES.shape
        assert np.all(np.isfinite(out.flux_density.value)) and np.all(out.flux_density.value >= 0)

    def test_units_in_inputs(self, model):
        a = model.forward_model({"time": TIMES * u.s, "frequency": np.full_like(TIMES, 5e9) * u.Hz}, {})
        b = model.forward_model({"time": TIMES / DAY * u.day, "frequency": np.full_like(TIMES, 5) * u.GHz}, {})
        assert np.allclose(a.flux_density.value, b.flux_density.value, rtol=1e-12)

    def test_scalar_inputs(self, model):
        out = model.forward_model({"time": 20 * DAY, "frequency": 5e9}, {})
        assert np.ndim(out.flux_density.value) == 0
        assert np.isfinite(out.flux_density.value)

    def test_broadcasting(self, model):
        out = model.forward_model({"time": TIMES[:, None], "frequency": FREQS[None, :]}, {})
        assert out.flux_density.shape == (TIMES.size, FREQS.size)

    def test_order_independent(self, model):
        """Paired evaluation must not depend on the ordering or repetition of the epochs."""
        t = np.array([80.0, 5.0, 80.0, 20.0, 5.0]) * DAY
        nu = np.array([2e10, 1e9, 5e9, 5e9, 1e11])
        out = model.forward_model({"time": t, "frequency": nu}, {}).flux_density.value
        for i in range(t.size):
            single = model.forward_model({"time": t[i], "frequency": nu[i]}, {}).flux_density.value
            # The adaptive ODE step history depends on the end of the integration span, so single-point and
            # grouped evaluations agree only to the solver tolerance (rtol = 1e-7).
            assert out[i] == pytest.approx(single, rel=1e-5)

    def test_paired_matches_tabulate(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        T, N = np.meshgrid(TIMES, FREQS, indexing="ij")
        paired = model.forward_model({"time": T.ravel(), "frequency": N.ravel()}, {}).flux_density.value
        assert np.allclose(paired.reshape(T.shape), tab.flux_density.value, rtol=1e-10)

    def test_epoch_before_t0_raises(self, model):
        with pytest.raises(ValueError, match="shock_start_time"):
            model.forward_model({"time": 0.1 * DAY, "frequency": 5e9}, {})

    def test_parameter_bounds_raise(self, model):
        with pytest.raises(ValueError):
            model.forward_model({"time": 20 * DAY, "frequency": 5e9}, {"epsilon_B": 2.0})

    def test_flux_scales_with_epsilon_e(self, model):
        """At high frequency the emission is optically thin, so the flux scales linearly with the electron energy."""
        lo = model.forward_model({"time": 80 * DAY, "frequency": 3e11}, {"epsilon_e": 0.05}).flux_density.value
        hi = model.forward_model({"time": 80 * DAY, "frequency": 3e11}, {"epsilon_e": 0.1}).flux_density.value
        assert hi > lo


# =========================================================================== #
# Redshift                                                                    #
# =========================================================================== #
class TestRedshift:
    def test_dynamics_are_evaluated_in_the_source_frame(self):
        z = 0.2
        m_z = Diggins26_TwoComponentModel(redshift=z)
        m_0 = Diggins26_TwoComponentModel(redshift=1e-9)
        t_obs = np.array([10.0, 50.0, 200.0]) * DAY
        d_z = m_z.compute_dynamics({}, t_obs)
        d_0 = m_0.compute_dynamics({}, t_obs / (1 + z))
        assert np.allclose(d_z.time_rest.to_value(u.s), t_obs / (1 + z))
        for sector in ("equatorial", "polar"):
            for field in ("radius", "velocity", "thermal_energy_density"):
                a = getattr(getattr(d_z, sector), field).value
                b = getattr(getattr(d_0, sector), field).value
                assert np.allclose(a, b, rtol=1e-7), (sector, field)

    def test_free_free_is_evaluated_at_the_source_frame_frequency(self):
        """tau_ff(nu_obs) at redshift z equals tau_ff at nu_obs * (1 + z) with the source-frame optical depth."""
        z = 0.2
        m_z = Diggins26_TwoComponentModel(redshift=z)
        m_0 = Diggins26_TwoComponentModel(redshift=1e-9)
        t_obs = np.array([20.0, 80.0]) * DAY
        nu = np.array([1e9, 4e9])
        tau_z = m_z.tabulate(t_obs, nu, {}).equatorial.tau_ff
        tau_0 = m_0.tabulate(t_obs / (1 + z), nu * (1 + z), {}).equatorial.tau_ff
        assert np.allclose(tau_z, tau_0, rtol=1e-6)

    def test_distance_dimming(self):
        """At fixed physics, the flux scales as the engine's (1 + z)-corrected 1 / D_A^2 transformation."""
        near = Diggins26_TwoComponentModel(luminosity_distance=60 * u.Mpc)
        far = Diggins26_TwoComponentModel(luminosity_distance=120 * u.Mpc)
        f_near = near.forward_model({"time": 80 * DAY, "frequency": 3e11}, {}).flux_density.value
        f_far = far.forward_model({"time": 80 * DAY, "frequency": 3e11}, {}).flux_density.value
        assert 2.0 < f_near / f_far < 6.0  # roughly D^-2 with a modest redshift correction


# =========================================================================== #
# Free-free emission measure                                                  #
# =========================================================================== #
class TestLineOfSightEmissionMeasure:
    _P = {
        "rho_int": 2e-16,
        "rho_ext": 1e-18,
        "alpha_int": 1.5,
        "alpha_ext": 2.0,
        "R_trans": 5e15,
        "delta_trans": 0.3,
        "f_pole": 0.2,
        "aspect": 2.0,
        "R_ion": 1e19,
    }

    @staticmethod
    def _geometry(inc, p_hat):
        """``(cos_psi, z_hat)`` of a sightline leaving along ``p_hat`` toward an observer at inclination ``inc``."""
        n_hat = np.array([np.sin(inc), 0.0, np.cos(inc)])
        return np.array([p_hat @ n_hat]), np.array([p_hat[2]])

    def _em(self, model, p, radius, inc, p_hat):
        p = model._coerce({**p, "inclination": inc})
        cos_psi, z_hat = self._geometry(inc, np.asarray(p_hat, dtype=float))
        return model._line_of_sight_emission_measure(p, radius, cos_psi, z_hat, model._csm_profile_parameters(p))[0]

    def _direct(self, model, p, radius, inc, p_hat, composition=None, n=300001):
        """
        Reference int rho^2 ds from three-dimensional vectors by dense Simpson integration, independent of the model's
        geometry and quadrature. The path starts where the sightline leaves the sphere of radius R and ends at R_ion.
        """
        p = model._coerce({**p, "inclination": inc})
        csm = model._csm_profile_parameters(p)
        n_hat = np.array([np.sin(inc), 0.0, np.cos(inc)])
        p_hat = np.asarray(p_hat, dtype=float)
        cos_psi = p_hat @ n_hat
        out = []
        for R in radius:
            s0 = max(0.0, -2.0 * R * cos_psi)
            s_end = -R * cos_psi + np.sqrt(p["R_ion"] ** 2 - R**2 * (1.0 - cos_psi**2))
            eps = 1e-4
            x = np.linspace(np.log(eps), np.log((s_end - s0) / R + eps), n)
            s = s0 + R * (np.exp(x) - eps)
            pos = R * p_hat[None, :] + s[:, None] * n_hat[None, :]
            r = np.linalg.norm(pos, axis=1)
            theta = np.arccos(np.clip(pos[:, 2] / r, -1.0, 1.0))
            rho = AxisymmetricCSMDensityProfile._opt_eval(r, None, **{**csm, "theta": theta})
            out.append(simpson(rho**2 * R * np.exp(x), x=x))
        out = np.array(out)
        return out * (composition if composition is not None else 1.0 / (model._mu_e * model._mu_i)) / M_P**2

    @pytest.mark.parametrize("inc", [0.0, 0.7, np.pi / 2])
    @pytest.mark.parametrize(
        "p_hat",
        [(1, 0, 0), (0, 1, 0), (0.6, -0.8, 0), (0, 0, 1), (0, 0, -1)],
        ids=["x", "y", "mixed", "north", "south"],
    )
    def test_matches_vector_reference(self, model, inc, p_hat):
        radius = np.geomspace(3e14, 3e18, 7)
        em = self._em(model, self._P, radius, inc, p_hat)
        assert np.allclose(em, self._direct(model, self._P, radius, inc, p_hat), rtol=1e-3)

    @pytest.mark.parametrize("X", [1.0, 0.7, 0.0])
    def test_composition_enters_as_n_e_n_i(self, X):
        """For fully ionized H and He, n_e n_i = (1 + X)(1 + 3X) / 8 * rho^2 / m_p^2."""
        model = Diggins26_TwoComponentModel(redshift=0.01, hydrogen_mass_fraction=X)
        radius = np.array([1e15, 4e16])
        em = self._em(model, self._P, radius, 0.9, (1, 0, 0))
        composition = (1 + X) * (1 + 3 * X) / 8
        assert np.allclose(em, self._direct(model, self._P, radius, 0.9, (1, 0, 0), composition), rtol=1e-3)

    def test_pole_on_near_lobe_is_the_radial_integral(self, model):
        """Looking down the axis from the near lobe, the sightline is the radius along theta = 0."""
        radius = np.geomspace(1e14, 3e18, 7)
        csm = AxisymmetricCSMDensityProfile.as_optimized_callable(theta=0.0, **model._csm_parameters(self._P))
        radial = []
        for R in radius:
            x = np.linspace(np.log(R), np.log(self._P["R_ion"]), 200001)
            radial.append(simpson(csm(np.exp(x), 0.0) ** 2 * np.exp(x), x=x))
        radial = np.array(radial) / (model._mu_e * model._mu_i * M_P**2)
        assert np.allclose(self._em(model, self._P, radius, 0.0, (0, 0, 1)), radial, rtol=1e-3)

    def test_pole_on_far_lobe_matches_near_lobe(self, model):
        """Only the CSM exterior to the shock counts, so down the axis the far lobe sees what the near lobe sees."""
        radius = np.geomspace(1e14, 3e18, 7)
        near = self._em(model, self._P, radius, 0.0, (0, 0, 1))
        far = self._em(model, self._P, radius, 0.0, (0, 0, -1))
        assert np.allclose(far, near, rtol=1e-12)

    def test_spherical_csm_depends_only_on_the_angle_to_the_observer(self, model):
        """With rho_int = 0 the integral depends only on the angle between the sightline and the observer."""
        p = {**self._P, "rho_int": 0.0}
        radius = np.array([1e15, 1e17])
        a = self._em(model, p, radius, 0.0, (1, 0, 0))
        b = self._em(model, p, radius, 0.0, (0, 1, 0))
        assert np.allclose(a, b, rtol=1e-12)  # both perpendicular to the line of sight
        assert np.allclose(a, self._em(model, p, radius, np.pi / 2, (0, 1, 0)), rtol=1e-12)

    def test_shape_and_zero_beyond_the_ionized_radius(self, model):
        p = model._coerce({**self._P, "R_ion": 1e16, "inclination": 0.8})
        cos_psi, z_hat = np.array([0.1, -0.4, 0.8]), np.array([0.0, 1.0, -1.0])
        em = model._line_of_sight_emission_measure(
            p, np.array([1e15, 2e16, 5e17]), cos_psi, z_hat, model._csm_profile_parameters(p)
        )
        assert em.shape == (3, 3)
        assert np.all(em[:, 0] > 0) and np.all(em[:, 1:] == 0)

    def test_decreases_outward_along_the_axis(self, model):
        em = self._em(model, self._P, np.geomspace(1e14, 1e18, 20), 0.0, (0, 0, 1))
        assert np.all(np.diff(em) < 0)

    def test_sharp_transition_is_resolved(self, model):
        p = {**self._P, "delta_trans": 0.02}
        radius = np.geomspace(1e14, 1e18, 6)
        for p_hat in ((1, 0, 0), (0, 0, 1)):
            em = self._em(model, p, radius, 0.9, p_hat)
            assert np.allclose(em, self._direct(model, p, radius, 0.9, p_hat), rtol=1e-3)

    def test_chunking_is_identical(self, model, monkeypatch):
        import trilobite.models.LFBOTs.diggins26 as module

        radius = np.geomspace(1e14, 1e18, 12)
        whole = self._em(model, self._P, radius, 0.9, (1, 0, 0))
        monkeypatch.setattr(module, "_LOS_MAX_NODES", 1)  # one epoch per chunk
        assert np.allclose(self._em(model, self._P, radius, 0.9, (1, 0, 0)), whole, rtol=1e-12)


class TestSightlinesAndTransmission:
    def test_polar_lobes_are_symmetric(self, model):
        _, (cos_psi, z_hat) = model._sightlines(0.6)
        assert np.allclose(cos_psi, [np.cos(0.6), -np.cos(0.6)]) and np.allclose(z_hat, [1.0, -1.0])

    @pytest.mark.parametrize("n_azimuth", [1, 4, 7])
    def test_ring_sightlines_lie_in_the_plane(self, n_azimuth):
        m = Diggins26_TwoComponentModel(redshift=0.01, n_azimuth=n_azimuth)
        (cos_psi, z_hat), _ = m._sightlines(0.6)
        assert cos_psi.shape == z_hat.shape == (n_azimuth,)
        assert np.all(z_hat == 0.0) and np.all(np.abs(cos_psi) <= np.sin(0.6))
        # the half-circle sample of cos(phi) is balanced about the ring's mid-azimuth
        assert np.mean(cos_psi) == pytest.approx(0.0, abs=1e-12)

    def test_pole_on_ring_is_perpendicular_to_the_line_of_sight(self, model):
        (cos_psi, _), _ = model._sightlines(0.0)
        assert np.all(cos_psi == 0.0)

    def test_transmission_averages_e_to_the_minus_tau_not_tau(self, model):
        """A transparent front and an opaque back transmit half the flux, not e^{-mean tau}."""
        EM = np.array([[0.0], [1e40]])
        (T,) = model._transmission(EM, [np.array([1.0])])
        assert T[0] == pytest.approx(0.5)

    def test_transmission_of_equal_sightlines_is_exponential(self, model):
        EM = np.full((3, 2), 2.0)
        tau_per_EM = [np.array([0.1, 0.5]), np.array([1.0, 2.0])]
        for T, t in zip(model._transmission(EM, tau_per_EM), tau_per_EM):
            assert np.allclose(T, np.exp(-2.0 * t))

    def test_one_azimuth_is_the_single_sightline(self):
        m = Diggins26_TwoComponentModel(redshift=0.01, n_azimuth=1)
        (cos_psi, _), _ = m._sightlines(0.9)
        assert cos_psi[0] == pytest.approx(np.sin(0.9) * np.cos(np.pi / 2))

    def test_invalid_n_azimuth_raises(self):
        with pytest.raises(ValueError, match="n_azimuth"):
            Diggins26_TwoComponentModel(redshift=0.01, n_azimuth=0)


# =========================================================================== #
# Geometry and the area-fraction table                                        #
# =========================================================================== #
class TestGeometry:
    def test_table_is_built_once_and_shared(self, model):
        other = Diggins26_TwoComponentModel(redshift=0.02)
        assert other.filling_table is model.filling_table
        assert model.build_filling_table() is model.filling_table
        aspect, inc, f, table = model.filling_table
        assert table.shape == (2, aspect.size, inc.size, f.size)

    def test_volume_fractions_partition_the_filling_factor(self, model):
        eq, pol = model._sector_geometry(model._coerce({"f_V": 0.4, "aspect": 1.5}))
        assert eq["f_V"] + pol["f_V"] == pytest.approx(0.4)

    def test_path_length_is_volume_over_projected_area(self, model):
        for geo in model._sector_geometry(model._coerce({"inclination": 0.5})):
            assert geo["ell_over_R"] == pytest.approx(4.0 / 3.0 * geo["f_V"] / geo["f_A"])

    def test_geometry_depends_on_inclination_not_dynamics(self, model):
        a = model._sector_geometry(model._coerce({"inclination": 0.2}))
        b = model._sector_geometry(model._coerce({"inclination": 1.3}))
        assert a[0]["f_A"] != pytest.approx(b[0]["f_A"], rel=1e-3)
        assert a[0]["f_V"] == pytest.approx(b[0]["f_V"], rel=1e-12)

    def test_table_agrees_with_direct_computation(self, model):
        p = model._coerce({"inclination": 0.8, "aspect": 1.4, "f_V": 0.35})
        direct = AxisymmetricCSMDensityProfile.compute_filling_parameters(0.35, 0.8, 1.4)
        for geo, kind in zip(model._sector_geometry(p), ("equatorial", "polar")):
            assert geo["f_A"] == pytest.approx(float(direct[kind]["f_A"]), abs=0.03)

    @pytest.mark.parametrize("f_V", [0.0, 1.0])
    def test_filling_factor_limits_do_not_raise(self, model, f_V):
        eq, pol = model._sector_geometry(model._coerce({"f_V": f_V}))
        assert np.isfinite(eq["f_A"]) and np.isfinite(pol["f_A"])

    def test_inclination_accepts_degrees(self, model):
        a = model.forward_model({"time": 20 * DAY, "frequency": 1e10}, {"inclination": 30 * u.deg})
        b = model.forward_model({"time": 20 * DAY, "frequency": 1e10}, {"inclination": np.pi / 6})
        assert a.flux_density.value == pytest.approx(b.flux_density.value, rel=1e-12)

    def test_inclination_bounds(self, model):
        with pytest.raises(ValueError):
            model.forward_model({"time": 20 * DAY, "frequency": 1e10}, {"inclination": 2.0})


# =========================================================================== #
# Tabulation and dynamics reuse                                               #
# =========================================================================== #
class TestTabulate:
    def test_structure(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        assert isinstance(tab, Diggins26Tabulation)
        assert isinstance(tab.dynamics, Diggins26Dynamics)
        assert tab.flux_density.shape == (TIMES.size, FREQS.size)
        for sector in (tab.equatorial, tab.polar):
            assert sector.flux_density.shape == tab.flux_density.shape
            assert sector.tau_ff.shape == tab.flux_density.shape
            assert sector.emission_measure.shape == TIMES.shape
            assert sector.magnetic_field.shape == TIMES.shape
        assert tab.equatorial.volume_fraction + tab.polar.volume_fraction == pytest.approx(0.5)  # default f_V
        for sector in (tab.equatorial, tab.polar):
            assert 0.0 < sector.area_fraction <= 1.0
            assert sector.path_length_ratio == pytest.approx(4.0 / 3.0 * sector.volume_fraction / sector.area_fraction)

    def test_total_is_sum_of_attenuated_sectors(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        total = tab.equatorial.flux_density_attenuated + tab.polar.flux_density_attenuated
        assert np.allclose(total.value, tab.flux_density.value, rtol=1e-12)

    def test_attenuation_never_increases_flux(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        for sector in (tab.equatorial, tab.polar):
            assert np.all(sector.flux_density_attenuated.value <= sector.flux_density.value * (1 + 1e-12))
            assert np.all(sector.tau_ff >= 0)

    def test_free_free_absorbs_low_frequencies_more(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        assert np.all(np.diff(tab.equatorial.tau_ff, axis=1) <= 0)

    def test_reused_dynamics_gives_identical_result(self, model):
        dyn = model.compute_dynamics({}, TIMES)
        a = model.tabulate(TIMES, FREQS, {}, dynamics=dyn)
        b = model.tabulate(TIMES, FREQS, {})
        assert np.allclose(a.flux_density.value, b.flux_density.value, rtol=1e-12)

    def test_reused_dynamics_allows_changed_microphysics(self, model):
        dyn = model.compute_dynamics({}, TIMES)
        a = model.tabulate(TIMES, FREQS, {"epsilon_B": 0.05}, dynamics=dyn)
        b = model.tabulate(TIMES, FREQS, {"epsilon_B": 0.05})
        assert np.allclose(a.flux_density.value, b.flux_density.value, rtol=1e-12)

    def test_mismatched_dynamics_raises(self, model):
        dyn = model.compute_dynamics({}, TIMES)
        with pytest.raises(ValueError, match="rho_int"):
            model.tabulate(TIMES, FREQS, {"rho_int": 1e-15}, dynamics=dyn)
        with pytest.raises(ValueError, match="different epochs"):
            model.tabulate(TIMES[:-1], FREQS, {}, dynamics=dyn)

    def test_unsorted_times_raise(self, model):
        with pytest.raises(ValueError, match="increasing"):
            model.tabulate(TIMES[::-1], FREQS, {})

    def test_no_interior_makes_the_dynamics_identical(self, model):
        """With rho_int = 0 the CSM is spherical, so the shock is the same along both sightlines."""
        dyn = model.compute_dynamics({"rho_int": 0.0}, TIMES)
        assert np.allclose(dyn.equatorial.radius.value, dyn.polar.radius.value, rtol=1e-6)

    def test_dynamics_are_reusable_across_inclination(self, model):
        """The inclination changes only the geometry and the sightlines, so one shock solution serves all of it."""
        dyn = model.compute_dynamics({}, TIMES)
        for inc in (0.2, 1.2):
            reused = model.tabulate(TIMES, FREQS, {"inclination": inc}, dynamics=dyn)
            fresh = model.tabulate(TIMES, FREQS, {"inclination": inc})
            assert np.array_equal(reused.flux_density.value, fresh.flux_density.value)

    def test_effective_tau_is_minus_log_of_mean_transmission(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        for sector in (tab.equatorial, tab.polar):
            assert np.allclose(
                sector.flux_density_attenuated.value, sector.flux_density.value * np.exp(-sector.tau_ff), rtol=1e-12
            )

    def test_dense_equator_decelerates_the_shock(self, model):
        dyn = model.compute_dynamics({}, TIMES)
        assert np.all(dyn.equatorial.radius.value < dyn.polar.radius.value)


# =========================================================================== #
# Infeasible parameters                                                       #
# =========================================================================== #
class TestInfeasibility:
    def test_default_parameters_are_feasible(self, model):
        assert model.diagnose({}, TIMES) is None

    @pytest.mark.parametrize(
        "parameters,fragment",
        [
            ({"M_ej": 1e29}, "max_initial_beta"),  # v_0 ~ c
            ({"gamma_max": 1.005, "gamma_min": 1.0}, "cooling break"),
            ({"rho_int": 0.0, "rho_ext": 0.0}, "no CSM mass"),
        ],
    )
    def test_nan_flux_and_diagnosis(self, model, parameters, fragment):
        out = model.forward_model({"time": TIMES, "frequency": np.full_like(TIMES, 5e9)}, parameters)
        assert np.all(np.isnan(out.flux_density.value))
        reason = model.diagnose(parameters, TIMES)
        assert reason is not None and fragment.lower() in reason.lower()

    def test_inner_radius_beyond_start_radius_is_infeasible(self):
        """No CSM lies inside R_0 = v_0 t_0 when the CSM starts beyond it."""
        m = Diggins26_TwoComponentModel(redshift=0.01, inner_radius=1e18)
        reason = m.diagnose({}, TIMES)
        assert reason is not None and "no CSM mass" in reason
        out = m.forward_model({"time": TIMES, "frequency": np.full_like(TIMES, 5e9)}, {})
        assert np.all(np.isnan(out.flux_density.value))

    def test_inner_radius_passes_through_to_the_density(self):
        m = Diggins26_TwoComponentModel(redshift=0.01, inner_radius=2e14)
        rho = m._sector_density(m._coerce({}), np.pi / 2)
        assert rho(1e14, 0.0) == 0.0 and rho(3e14, 0.0) > 0.0

    def test_swept_mass_lower_limit(self):
        """The swept mass of a power law starts at the inner radius; a slope >= 3 only converges with one."""
        rho = lambda r, t: 1e-16 * (r / 1e15) ** -3.5  # noqa: E731
        M = Diggins26_TwoComponentModel._swept_mass
        r_min, R = 1e14, 1e15
        analytic = 4 * np.pi * 1e-16 * 1e15**3.5 * (r_min**-0.5 - R**-0.5) / 0.5
        assert M(rho, R, r_min) == pytest.approx(analytic, rel=1e-6)
        assert np.isnan(M(rho, R, 0.0))
        assert M(rho, R, R) == 0.0

    @pytest.mark.parametrize("which", ["alpha_int", "alpha_ext"])
    @pytest.mark.parametrize("slope", [3.0, 3.5])
    def test_steep_slope_needs_an_inner_radius(self, model, which, slope):
        """A slope >= 3 has divergent mass at small radius: infeasible without `inner_radius`, fine with one."""
        parameters = {which: slope}
        reason = model.diagnose(parameters, TIMES)
        assert reason is not None and "inner_radius" in reason
        out = model.forward_model({"time": TIMES, "frequency": np.full_like(TIMES, 5e9)}, parameters)
        assert np.all(np.isnan(out.flux_density.value))

        bounded = Diggins26_TwoComponentModel(redshift=0.01, inner_radius=1e13)
        assert bounded.diagnose(parameters, TIMES) is None
        out = bounded.forward_model({"time": TIMES, "frequency": np.full_like(TIMES, 5e9)}, parameters)
        assert np.all(np.isfinite(out.flux_density.value))

    def test_slope_of_zero_component_does_not_matter(self, model):
        """A steep slope on a component with zero amplitude contributes no mass."""
        assert model.diagnose({"rho_ext": 0.0, "alpha_ext": 4.0}, TIMES) is None

    def test_ejecta_normalization_failure_is_infeasible(self):
        m = Diggins26_TwoComponentModel(redshift=0.01, max_initial_beta=0.99)
        reason = m.diagnose({"M_ej": 1e29}, 20 * DAY)
        assert reason is not None
        out = m.forward_model({"time": 20 * DAY, "frequency": 5e9}, {"M_ej": 1e29})
        assert np.isnan(out.flux_density.value)

    def test_unnormalizable_ejecta_below_beta_max_is_infeasible(self, model):
        """v_0 = 0.79 c passes the max_initial_beta = 0.8 check, but a delta = 0 kernel truncated at c cannot reach it."""
        M_ej = 2e51 / (0.79 * C) ** 2
        parameters = {"M_ej": M_ej, "delta": 0.0}
        reason = model.diagnose(parameters, TIMES)
        assert reason is not None and "normalized" in reason
        out = model.forward_model({"time": TIMES, "frequency": np.full_like(TIMES, 5e9)}, parameters)
        assert np.all(np.isnan(out.flux_density.value))

    def test_tabulate_raises_with_reason(self, model):
        with pytest.raises(ValueError, match="infeasible"):
            model.tabulate(TIMES, FREQS, {"M_ej": 1e29})

    def test_unexpected_errors_propagate(self, model, monkeypatch):
        def broken(*args, **kwargs):
            raise KeyError("a genuine bug")

        monkeypatch.setattr(model, "_microphysics", broken)
        with pytest.raises(KeyError):
            model.forward_model({"time": 20 * DAY, "frequency": 5e9}, {})


# =========================================================================== #
# Vectorized radiation                                                        #
# =========================================================================== #
class TestVectorizedRadiation:
    """The batched microphysics and paired spectra must equal a plain per-epoch loop of the scalar library calls."""

    @pytest.fixture(scope="class")
    def setup(self, model):
        p = model._coerce({})
        t_rest = model._rest_times(np.unique(TIMES))
        states, reason = model._dynamics_cgs(p, t_rest)
        assert states is not None, reason
        gamma = model._gamma_grid(p, model._gamma_points_per_decade)
        return p, t_rest, states[0], gamma

    def test_microphysics_matches_the_scalar_distribution(self, model, setup):
        from trilobite.radiation.synchrotron.electron_distributions import BrokenPowerLaw

        p, t_rest, state, gamma = setup
        B, gamma_c, N0, log_N = model._microphysics(p, state, t_rest, gamma)

        for i in range(B.size):
            shape = {
                "p1": p["p"],
                "p2": p["p"] + 1.0,
                "gamma_min": p["gamma_min"],
                "gamma_max": p["gamma_max"],
                "gamma_c": gamma_c[i],
            }
            N0_i = BrokenPowerLaw._normalize_from_magnetic_field(B[i], p["epsilon_B"], p["epsilon_e"], **shape)
            assert N0[i] == pytest.approx(N0_i, rel=1e-12)
            with np.errstate(divide="ignore"):
                expected = np.log(BrokenPowerLaw.pdf(gamma, norm=N0_i, **shape))
            assert np.allclose(log_N[i], expected, rtol=1e-12, atol=0.0, equal_nan=True)

    def test_paired_spectra_match_per_epoch_engine_calls(self, model, setup):
        p, t_rest, state, gamma = setup
        micro = model._microphysics(p, state, t_rest, gamma)
        B, _, _, log_N = micro
        geometry = model._sector_geometry(p)[0]

        # a different number of frequencies at every epoch, including a single one
        rng = np.random.default_rng(0)
        nu_groups = [np.sort(rng.uniform(1e9, 3e11, size=1 + (i % 4))) for i in range(t_rest.size)]
        flux = model._sector_spectra(p, state, micro, geometry, gamma, nu_groups)

        log_gamma, log_weights = model._synchrotron_engine._build_gamma_grid(gamma)
        ell = geometry["ell_over_R"] * state.radius
        area = np.pi * state.radius**2 * geometry["f_A"]
        for i, nu in enumerate(nu_groups):
            with np.errstate(divide="ignore"):
                expected = np.exp(
                    model._synchrotron_engine._compute_log_pa_flux_density(
                        np.log(nu),
                        np.log(ell[i]),
                        np.log(B[i]),
                        log_N[i],
                        log_gamma,
                        log_weights,
                        np.log(area[i]),
                        model._log_D_A,
                        z=model._z,
                    )
                )
            assert np.allclose(flux[i], expected, rtol=1e-12, atol=0.0)

    def test_chunked_evaluation_is_identical(self, model, setup, monkeypatch):
        import trilobite.models.LFBOTs.diggins26 as module

        p, t_rest, state, gamma = setup
        micro = model._microphysics(p, state, t_rest, gamma)
        nu_groups = [np.array([1e9, 5e10, 2e11]) for _ in t_rest]
        args = (p, state, micro, model._sector_geometry(p)[0], gamma, nu_groups)

        whole = model._sector_spectra(*args)
        monkeypatch.setattr(module, "_SPECTRUM_CHUNK", 5)  # forces several chunks that split epochs mid-way
        chunked = model._sector_spectra(*args)
        assert all(np.array_equal(a, b) for a, b in zip(whole, chunked))


# =========================================================================== #
# Compton cooling                                                             #
# =========================================================================== #
class TestComptonCooling:
    @staticmethod
    def _constant(L):
        return lambda t: np.full_like(t, L)

    def test_no_seed_photons_is_synchrotron_cooling(self, model):
        B = np.array([0.1, 1.0])
        radius = np.array([1e16, 1e17])
        t = np.array([1e6, 1e7])
        assert np.allclose(model._cooling_gamma(B, radius, t), 9 * np.pi * M_E * C / (SIGMA_T * B**2 * t), rtol=1e-12)

    def test_inverse_compton_loss_rates_add(self):
        L = 1e44
        model = Diggins26_TwoComponentModel(redshift=0.01, seed_photon_luminosity=self._constant(L))
        B, radius, t = np.array([0.3]), np.array([3e16]), np.array([2e6])
        gamma_syn = 9 * np.pi * M_E * C / (SIGMA_T * B**2 * t)
        gamma_ic = 3 * np.pi * M_E * C**2 * radius**2 / (SIGMA_T * L * t)
        assert np.allclose(model._cooling_gamma(B, radius, t), 1 / (1 / gamma_syn + 1 / gamma_ic), rtol=1e-12)

    def test_zero_luminosity_recovers_synchrotron_only(self, model):
        dark = Diggins26_TwoComponentModel(luminosity_distance=60 * u.Mpc, seed_photon_luminosity=self._constant(0.0))
        assert np.allclose(
            dark.tabulate(TIMES, FREQS, {}).flux_density.value,
            model.tabulate(TIMES, FREQS, {}).flux_density.value,
            rtol=1e-12,
            atol=0.0,
        )

    def test_seed_photons_lower_the_cooling_break_and_leave_the_dynamics_alone(self, model):
        bright = Diggins26_TwoComponentModel(
            luminosity_distance=60 * u.Mpc, seed_photon_luminosity=self._constant(1e45)
        )
        plain, cooled = model.tabulate(TIMES, FREQS, {}), bright.tabulate(TIMES, FREQS, {})
        assert np.all(cooled.equatorial.gamma_c <= plain.equatorial.gamma_c)
        assert np.any(cooled.equatorial.gamma_c < plain.equatorial.gamma_c)
        assert np.allclose(cooled.dynamics.equatorial.radius.value, plain.dynamics.equatorial.radius.value, rtol=1e-10)

    def test_callable_receives_source_frame_seconds_and_may_return_a_quantity(self):
        seen = []

        def luminosity(t):
            seen.append(np.array(t))
            return 1e43 * (t / 1e6) ** -1.0 * u.erg / u.s

        m = Diggins26_TwoComponentModel(redshift=0.5, seed_photon_luminosity=luminosity)
        out = m.forward_model({"time": 30 * DAY, "frequency": 5e9}, {})
        assert np.isfinite(out.flux_density.value)
        assert seen[0][0] == pytest.approx(30 * DAY / 1.5)

    @pytest.mark.parametrize("bad", [-1.0, np.nan])
    def test_invalid_luminosity_raises(self, bad):
        m = Diggins26_TwoComponentModel(redshift=0.01, seed_photon_luminosity=self._constant(bad))
        with pytest.raises(ValueError, match="seed_photon_luminosity"):
            m.forward_model({"time": 20 * DAY, "frequency": 5e9}, {})


# =========================================================================== #
# Inference integration                                                       #
# =========================================================================== #
class TestInference:
    @pytest.fixture(scope="class")
    def problem(self):
        model = Diggins26_TwoComponentModel(luminosity_distance=60 * u.Mpc)
        t = np.repeat(np.array([10.0, 40.0, 160.0]) * DAY, 3)
        nu = np.tile(np.array([2e9, 1e10, 1e11]), 3)
        truth = model.forward_model({"time": t, "frequency": nu}, {}).flux_density.to_value(u.Jy)
        data = InferenceData(
            x={"time": t, "frequency": nu},
            observables={"flux_density": Observable(value=truth, error=0.1 * truth)},
        )
        like = LogGaussianCensoredLikelihood(model, data, sys_dex=0.1)
        problem = InferenceProblem(likelihood=like)
        problem.set_prior("rho_int", "uniform", lower=1e-17, upper=1e-14)
        problem.set_prior("epsilon_B", "uniform", lower=1e-4, upper=0.5)
        for name in problem.parameter_order:
            if name not in ("rho_int", "epsilon_B"):
                problem.parameters[name].freeze = True
        problem.parameters["rho_int"].initial_value = 2e-16
        problem.parameters["epsilon_B"].initial_value = 0.01
        return problem

    def test_posterior_is_finite_and_peaks_at_truth(self, problem):
        theta = problem.initial_theta
        at_truth = problem._log_free_posterior(theta)
        assert np.isfinite(at_truth)
        # Perturbing either free parameter lowers the posterior.
        for i in range(theta.size):
            moved = theta.copy()
            moved[i] = moved[i] * 3.0
            assert problem._log_free_posterior(moved) < at_truth

    def test_infeasible_draw_is_rejected_cleanly(self, problem):
        problem.parameters["M_ej"].freeze = True
        problem.parameters["M_ej"].initial_value = 1e29
        try:
            assert problem._log_free_posterior(problem.initial_theta) == -np.inf
        finally:
            problem.parameters["M_ej"].initial_value = 0.1 * const.M_sun.cgs.value
