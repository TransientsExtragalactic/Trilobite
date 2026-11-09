"""Tests for the Arnett-type optical models in :mod:`trilobite.models.supernovae.arnett`."""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from scipy.integrate import trapezoid

from trilobite.models.supernovae.arnett import (
    ArnettOpticalModel,
    MagnetarArnettCutoffModel,
    MagnetarArnettModel,
    MagnetarNickelArnettCutoffModel,
    MagnetarNickelArnettModel,
    NickelCobaltArnettCutoffModel,
    NickelCobaltArnettModel,
)
from trilobite.utils.phot_utils import FilterBundle, PhotometryFilter

_DAY = 86400.0
_PLAIN_MODELS = [NickelCobaltArnettModel, MagnetarArnettModel, MagnetarNickelArnettModel]
_CUTOFF_MODELS = [NickelCobaltArnettCutoffModel, MagnetarArnettCutoffModel, MagnetarNickelArnettCutoffModel]
_MODELS = _PLAIN_MODELS + _CUTOFF_MODELS


def _box_filter(lam_center_cm: float, width_cm: float, name: str) -> PhotometryFilter:
    lam = np.linspace(lam_center_cm - width_cm / 2, lam_center_cm + width_cm / 2, 50)
    return PhotometryFilter(lam, np.ones(50), name=name)


@pytest.fixture
def bundle() -> FilterBundle:
    return FilterBundle({"V": _box_filter(550e-7, 80e-7, "V"), "R": _box_filter(700e-7, 100e-7, "R")})


def _bolometric_flux(model, parameters, t_obs):
    """Integrate the model's SED over a dense observer-frame frequency grid."""
    nu = np.logspace(11.0, 17.5, 200_000)
    raw = model.coerce_model_parameters(parameters)
    sed = model._compute_sed(nu, np.atleast_1d(np.asarray(t_obs, dtype=float)), raw)
    return trapezoid(sed, nu, axis=-1)


# --------------------------------------------------------------------------------------------------------------- #
# Construction                                                                                                    #
# --------------------------------------------------------------------------------------------------------------- #
class TestConstruction:
    def test_exported_from_models(self):
        import trilobite.models as models

        for cls in _MODELS:
            assert getattr(models, cls.__name__) is cls
        assert issubclass(NickelCobaltArnettModel, ArnettOpticalModel)

    def test_distance_required(self, bundle):
        with pytest.raises(ValueError, match="Exactly one"):
            NickelCobaltArnettModel(bundle)

    def test_redshift_and_distance_exclusive(self, bundle):
        with pytest.raises(ValueError):
            NickelCobaltArnettModel(bundle, redshift=0.1, luminosity_distance=100 * u.Mpc)

    def test_zero_redshift_needs_a_distance(self, bundle):
        with pytest.raises(ValueError, match="must be positive"):
            NickelCobaltArnettModel(bundle, redshift=0.0)
        model = NickelCobaltArnettModel(bundle, luminosity_distance=10 * u.Mpc)
        assert model.luminosity_distance.to_value(u.Mpc) == pytest.approx(10.0)

    def test_distance_from_redshift(self, bundle):
        model = MagnetarArnettModel(bundle, redshift=0.1)
        assert model.redshift == 0.1
        assert model.luminosity_distance.to_value(u.Mpc) > 400

    def test_bare_distance_is_mpc(self, bundle):
        model = NickelCobaltArnettModel(bundle, luminosity_distance=50.0)
        assert model.luminosity_distance.to_value(u.Mpc) == pytest.approx(50.0)

    def test_engine_options(self, bundle):
        model = NickelCobaltArnettModel(bundle, redshift=0.05, positron_trapping=True, beta=12.0)
        assert model.engine.positron_trapping is True
        assert model.engine.beta == 12.0

    def test_magnetar_has_no_positron_switch(self, bundle):
        with pytest.raises(TypeError):
            MagnetarArnettModel(bundle, redshift=0.05, positron_trapping=True)

    def test_base_class_requires_an_engine(self, bundle):
        with pytest.raises(TypeError, match="ENGINE_CLASS"):
            ArnettOpticalModel(bundle, redshift=0.05)

    def test_parameters_cover_engine_parameters(self):
        for cls in _MODELS:
            names = {p.name for p in cls.PARAMETERS}
            assert set(cls._SOURCE_PARAMETERS) <= names
            extra = {"lambda_cut"} if cls in _CUTOFF_MODELS else set()
            assert set(cls._SOURCE_PARAMETERS) | {"M_ej", "E_ej", "kappa", "T_floor", "t_exp"} | extra == names
            assert set(cls._SOURCE_PARAMETERS) == set(cls.ENGINE_CLASS._PARAMETER_UNITS) - {
                "M_ej",
                "E_ej",
                "kappa",
                "R_0",
                "E_th_0",
            }


# --------------------------------------------------------------------------------------------------------------- #
# Forward model                                                                                                   #
# --------------------------------------------------------------------------------------------------------------- #
class TestForwardModel:
    @pytest.mark.parametrize("cls", _MODELS)
    def test_defaults_give_positive_finite_flux(self, bundle, cls):
        model = cls(bundle, redshift=0.05)
        variables = {"band_index": ["V", "R", "V", "R"], "time": [10, 10, 40, 40] * u.day}
        result = model.forward_model(variables, {})
        assert result.flux.shape == (4,)
        assert result.flux.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.Hz)
        assert np.all(np.isfinite(result.flux.value))
        assert np.all(result.flux.value > 0)

    def test_nickel_parameters_matter(self, bundle):
        model = NickelCobaltArnettModel(bundle, redshift=0.05)
        variables = {"band_index": [0], "time": [20] * u.day}
        low = model.forward_model(variables, {"M_Ni": 0.05 * u.Msun}).flux
        high = model.forward_model(variables, {"M_Ni": 0.5 * u.Msun}).flux
        assert high > low

    def test_magnetar_parameters_matter(self, bundle):
        model = MagnetarArnettModel(bundle, redshift=0.05)
        variables = {"band_index": [0], "time": [30] * u.day}
        slow = model.forward_model(variables, {"P_spin": 10 * u.ms}).flux
        fast = model.forward_model(variables, {"P_spin": 2 * u.ms}).flux
        assert fast > slow

    def test_no_light_before_explosion(self, bundle):
        model = NickelCobaltArnettModel(bundle, redshift=0.05)
        result = model.forward_model({"band_index": [0, 1, 0], "time": [-5, 0, 10] * u.day}, {})
        assert result.flux.value[0] == 0.0
        assert result.flux.value[1] == 0.0
        assert result.flux.value[2] > 0.0

    def test_band_names_and_indices_agree(self, bundle):
        model = NickelCobaltArnettModel(bundle, redshift=0.05)
        by_name = model.forward_model({"band_index": ["V", "R"], "time": [15, 15] * u.day}, {}).flux
        by_index = model.forward_model({"band_index": [0, 1], "time": [15, 15] * u.day}, {}).flux
        np.testing.assert_allclose(by_name.value, by_index.value)

    def test_bare_and_quantity_parameters_agree(self, bundle):
        model = NickelCobaltArnettModel(bundle, redshift=0.05)
        variables = {"band_index": [0, 1], "time": [15, 25] * u.day}
        with_units = model.forward_model(variables, {"M_ej": 2 * u.Msun, "M_Ni": 0.2 * u.Msun}).flux
        bare = model.forward_model(
            variables, {"M_ej": (2 * u.Msun).to_value(u.g), "M_Ni": (0.2 * u.Msun).to_value(u.g)}
        )
        np.testing.assert_allclose(with_units.value, bare.flux.value)

    def test_bounds_are_enforced(self, bundle):
        model = NickelCobaltArnettModel(bundle, redshift=0.05)
        with pytest.raises(ValueError, match="M_ej"):
            model.forward_model({"band_index": [0], "time": [20] * u.day}, {"M_ej": -1 * u.Msun})
        magnetar = MagnetarArnettModel(bundle, redshift=0.05)
        with pytest.raises(ValueError, match="braking_index"):
            magnetar.forward_model({"band_index": [0], "time": [20] * u.day}, {"braking_index": 0.5})


# --------------------------------------------------------------------------------------------------------------- #
# Physics                                                                                                         #
# --------------------------------------------------------------------------------------------------------------- #
class TestPhysics:
    @pytest.mark.parametrize("redshift", [0.0, 0.5])
    @pytest.mark.parametrize("cls", _MODELS)
    def test_bolometric_flux_is_conserved(self, bundle, cls, redshift):
        """The integrated spectrum is L / 4 pi D_L^2, at any redshift (T_floor low so the BB carries all of L)."""
        if redshift == 0.0:
            model = cls(bundle, luminosity_distance=100 * u.Mpc)
        else:
            model = cls(bundle, redshift=redshift)
        parameters = {"T_floor": 1.0 * u.K}
        t_obs = np.array([5.0, 20.0, 60.0]) * _DAY * (1 + redshift)

        flux = _bolometric_flux(model, parameters, t_obs)
        L = model.compute_photosphere(t_obs * u.s, parameters).luminosity.to_value(u.erg / u.s)
        D_L = model.luminosity_distance.to_value(u.cm)
        np.testing.assert_allclose(flux, L / (4 * np.pi * D_L**2), rtol=2e-3)

    @pytest.mark.parametrize("cls", _MODELS)
    def test_floor_conserves_luminosity(self, bundle, cls):
        model = cls(bundle, redshift=0.05)
        t = np.array([5.0, 20.0, 60.0, 150.0]) * u.day
        free = model.compute_photosphere(t, {"T_floor": 1.0 * u.K})
        floored = model.compute_photosphere(t, {"T_floor": 2.0e4 * u.K})

        np.testing.assert_allclose(floored.luminosity.value, free.luminosity.value)
        assert np.all(floored.temperature.value >= 2.0e4 - 1e-6)
        assert np.all(floored.radius.value <= free.radius.value * (1 + 1e-12))
        np.testing.assert_allclose(
            4 * np.pi * floored.radius.value**2 * 5.670374419e-5 * floored.temperature.value**4,
            floored.luminosity.value,
            rtol=1e-6,
        )
        # And the spectrum still carries it.
        flux = _bolometric_flux(model, {"T_floor": 2.0e4 * u.K}, t.to_value(u.s))
        D_L = model.luminosity_distance.to_value(u.cm)
        np.testing.assert_allclose(flux, floored.luminosity.value / (4 * np.pi * D_L**2), rtol=2e-3)

    def test_floor_only_binds_at_late_times(self, bundle):
        model = NickelCobaltArnettModel(bundle, redshift=0.05)
        photosphere = model.compute_photosphere([5, 150] * u.day, {"T_floor": 6000 * u.K})
        assert photosphere.temperature[0].value > 6000.0
        assert photosphere.temperature[1].value == pytest.approx(6000.0)

    def test_surface_radius_before_the_floor_binds(self, bundle):
        model = NickelCobaltArnettModel(bundle, redshift=0.05)
        parameters = {"M_ej": 2 * u.Msun, "E_ej": 1e51 * u.erg, "T_floor": 1.0 * u.K}
        v_sc = np.sqrt(10 * (1e51) / (3 * (2 * u.Msun).to_value(u.g)))
        t_rest = 10 * _DAY
        photosphere = model.compute_photosphere(t_rest * 1.05 * u.s, parameters)
        assert photosphere.radius[0].value == pytest.approx(v_sc * t_rest, rel=1e-9)

    def test_redshift_dilates_time(self, bundle):
        """The luminosity at observer time t is the engine's luminosity at t / (1 + z)."""
        model = NickelCobaltArnettModel(bundle, redshift=0.5)
        parameters = {"M_ej": 2 * u.Msun, "E_ej": 1e51 * u.erg, "kappa": 0.1 * u.cm**2 / u.g, "M_Ni": 0.3 * u.Msun}
        t_obs = np.array([15.0, 30.0, 60.0]) * u.day
        L_model = model.compute_photosphere(t_obs, parameters).luminosity
        L_engine = model.engine.compute_luminosity(t_obs / 1.5, **{k: v for k, v in parameters.items()})
        np.testing.assert_allclose(L_model.value, L_engine.value, rtol=1e-12)

    def test_flux_falls_with_distance(self, bundle):
        # (A different distance also implies a different redshift, so this is a monotonicity check only.)
        near = NickelCobaltArnettModel(bundle, luminosity_distance=50 * u.Mpc)
        far = NickelCobaltArnettModel(bundle, luminosity_distance=100 * u.Mpc)
        variables = {"band_index": [0], "time": [20] * u.day}
        assert near.forward_model(variables, {}).flux > far.forward_model(variables, {}).flux

    def test_magnetar_nickel_is_the_sum(self, bundle):
        """The problem is linear in the power, so the combined luminosity is the sum of the two engines."""
        both = MagnetarNickelArnettModel(bundle, redshift=0.05)
        ni = NickelCobaltArnettModel(bundle, redshift=0.05)
        mag = MagnetarArnettModel(bundle, redshift=0.05)
        t = np.array([10.0, 30.0, 80.0]) * u.day
        L_both = both.compute_photosphere(t).luminosity.value
        L_sum = ni.compute_photosphere(t).luminosity.value + mag.compute_photosphere(t).luminosity.value
        np.testing.assert_allclose(L_both, L_sum, rtol=2e-3)

    def test_explosion_epoch_default_is_zero(self, bundle):
        model = NickelCobaltArnettModel(bundle, redshift=0.05)
        assert model.coerce_model_parameters({})["t_exp"] == 0.0

    @pytest.mark.parametrize("cls", _MODELS)
    def test_explosion_epoch_shifts_the_time_axis(self, bundle, cls):
        """Moving the explosion earlier by dt is the same as observing dt later: the radio-style convention."""
        model = cls(bundle, redshift=0.05)
        shift = 12.0 * u.day
        variables = {"band_index": ["V", "R", "V", "R"], "time": [10, 10, 40, 40] * u.day}
        shifted = {"band_index": variables["band_index"], "time": variables["time"] + shift}
        at_explosion = model.forward_model(shifted, {}).flux
        with_offset = model.forward_model(shifted, {"t_exp": shift}).flux
        reference = model.forward_model(variables, {}).flux
        np.testing.assert_allclose(with_offset.value, reference.value)
        assert not np.allclose(at_explosion.value, reference.value, rtol=1e-3, atol=0.0)

    def test_nothing_shines_before_the_explosion_epoch(self, bundle):
        model = NickelCobaltArnettModel(bundle, redshift=0.05)
        result = model.forward_model({"band_index": [0, 0, 0], "time": [5, 20, 40] * u.day}, {"t_exp": 20 * u.day})
        assert result.flux.value[0] == 0.0
        assert result.flux.value[1] == 0.0
        assert result.flux.value[2] > 0.0

    def test_photosphere_follows_the_explosion_epoch(self, bundle):
        model = MagnetarArnettModel(bundle, redshift=0.05)
        shift = 7.0 * u.day
        base = model.compute_photosphere([10, 30, 80] * u.day)
        shifted = model.compute_photosphere([10, 30, 80] * u.day + shift, {"t_exp": shift})
        np.testing.assert_allclose(shifted.luminosity.value, base.luminosity.value)
        np.testing.assert_allclose(shifted.temperature.value, base.temperature.value)
        np.testing.assert_allclose(shifted.radius.value, base.radius.value)
        np.testing.assert_allclose(shifted.time.to_value(u.day), [17, 37, 87])

    def test_positron_trapping_switch(self, bundle):
        t = [200] * u.day
        parameters = {"kappa_positron": 0.01 * u.cm**2 / u.g}  # low enough that positrons leak at 200 d
        free = NickelCobaltArnettModel(bundle, redshift=0.05, positron_trapping=False)
        trapped = NickelCobaltArnettModel(bundle, redshift=0.05, positron_trapping=True)
        L_free = free.compute_photosphere(t, parameters).luminosity.value
        L_trapped = trapped.compute_photosphere(t, parameters).luminosity.value
        assert L_trapped < L_free

    def test_gamma_trapping_switch(self, bundle):
        t = [200] * u.day
        leaky = NickelCobaltArnettModel(bundle, redshift=0.05, gamma_trapping=True)
        full = NickelCobaltArnettModel(bundle, redshift=0.05, gamma_trapping=False)
        assert leaky.compute_photosphere(t).luminosity.value < full.compute_photosphere(t).luminosity.value


# --------------------------------------------------------------------------------------------------------------- #
# Serialization                                                                                                   #
# --------------------------------------------------------------------------------------------------------------- #
class TestSerialization:
    @pytest.mark.parametrize("cls", _MODELS)
    def test_round_trip(self, bundle, cls):
        model = cls(bundle, redshift=0.05, gamma_trapping=False)
        restored = cls.from_model_spec(model.to_model_spec())
        assert isinstance(restored, cls)
        assert restored.redshift == model.redshift
        assert restored.engine.gamma_trapping is False

        variables = {"band_index": ["V", "R"], "time": [20, 35] * u.day}
        np.testing.assert_allclose(
            restored.forward_model(variables, {}).flux.value, model.forward_model(variables, {}).flux.value
        )

    def test_round_trip_preserves_a_given_distance(self, bundle):
        model = NickelCobaltArnettModel(bundle, luminosity_distance=37.5 * u.Mpc)
        restored = NickelCobaltArnettModel.from_model_spec(model.to_model_spec())
        assert restored.luminosity_distance.to_value(u.Mpc) == pytest.approx(37.5)


# --------------------------------------------------------------------------------------------------------------- #
# Photosphere, against the equations of Nicholl et al. (2017)                                                    #
# --------------------------------------------------------------------------------------------------------------- #
def _nicholl_photosphere(L, v, t, T_f):
    """Sec. III.6 of Nicholl et al. (2017) (arXiv:1706.00825), implemented from the published piecewise equations."""
    sigma = const.sigma_sb.cgs.value
    T_eff = (L / (4.0 * np.pi * sigma * v**2 * t**2)) ** 0.25
    hot = T_eff > T_f
    T = np.where(hot, T_eff, T_f)
    R = np.where(hot, v * t, (L / (4.0 * np.pi * sigma * T_f**4)) ** 0.5)
    return T, R


class TestPhotosphereMatchesMOSFiT:
    @pytest.mark.parametrize("cls", _MODELS)
    @pytest.mark.parametrize("T_floor", [1.0e3, 6.0e3, 1.5e4, 4.0e4])
    def test_temperature_and_radius(self, bundle, cls, T_floor):
        """Both branches (above and at the floor) agree with the published piecewise equations."""
        z = 0.1
        model = cls(bundle, redshift=z)
        parameters = {"M_ej": 8 * u.Msun, "E_ej": 2e51 * u.erg, "T_floor": T_floor * u.K}
        t_obs = np.array([3.0, 12.0, 30.0, 60.0, 120.0, 250.0]) * u.day

        photosphere = model.compute_photosphere(t_obs, parameters)
        t_rest = (t_obs / (1 + z)).to_value(u.s)
        v = np.sqrt(10 * (2e51) / (3 * (8 * u.Msun).to_value(u.g)))
        T, R = _nicholl_photosphere(photosphere.luminosity.to_value(u.erg / u.s), v, t_rest, T_floor)

        np.testing.assert_allclose(photosphere.temperature.to_value(u.K), T, rtol=1e-12)
        np.testing.assert_allclose(photosphere.radius.to_value(u.cm), R, rtol=1e-12)

    def test_both_regimes_are_exercised(self, bundle):
        model = MagnetarArnettModel(bundle, redshift=0.1)
        photosphere = model.compute_photosphere([3, 250] * u.day, {"T_floor": 6000 * u.K})
        assert photosphere.temperature[0].value > 6000.0  # surface of the ejecta
        assert photosphere.temperature[1].value == pytest.approx(6000.0)  # receded to the floor

    def test_diffusion_time_and_leakage_match_the_paper(self, bundle):
        """t_diff = (2 kappa M / beta c v)^(1/2) and A = 3 kappa_gamma M / (4 pi v^2), with v = v_sc."""
        model = NickelCobaltArnettModel(bundle, redshift=0.05, beta=13.8)
        M, E, kappa, kappa_gamma = (5 * u.Msun).to_value(u.g), 1.5e51, 0.1, 0.03
        v = np.sqrt(10 * E / (3 * M))
        t_diff = model.engine.compute_diffusion_time(M, E, kappa).to_value(u.s)
        assert t_diff == pytest.approx(np.sqrt(2 * kappa * M / (13.8 * 2.99792458e10 * v)), rel=1e-6)
        A = model.engine._compute_trapping_parameter_cgs(M, E, kappa_gamma)
        assert A == pytest.approx(3 * kappa_gamma * M / (4 * np.pi * v**2), rel=1e-12)


# --------------------------------------------------------------------------------------------------------------- #
# Where the trapping is applied                                                                                   #
# --------------------------------------------------------------------------------------------------------------- #
class TestTrappingMode:
    @pytest.mark.parametrize("cls", _MODELS)
    def test_default_and_option_reach_the_engine(self, bundle, cls):
        assert cls(bundle, redshift=0.05).engine.trapping_mode == "deposition"
        assert cls(bundle, redshift=0.05, trapping_mode="emergence").engine.trapping_mode == "emergence"

    def test_unknown_mode_is_rejected(self, bundle):
        with pytest.raises(ValueError, match="trapping_mode"):
            MagnetarArnettModel(bundle, redshift=0.05, trapping_mode="late")

    @pytest.mark.parametrize("cls", [MagnetarArnettModel, MagnetarArnettCutoffModel])
    def test_emergence_is_the_mosfit_leakage(self, bundle, cls):
        """The luminosity is (1 - exp(-A/t^2)) times that of the model with no leakage (Nicholl et al. 2017, Eq. 5)."""
        z, kappa_magnetar = 0.1, 0.05 * u.cm**2 / u.g
        parameters = {"M_ej": 6 * u.Msun, "E_ej": 3e51 * u.erg, "kappa_magnetar": kappa_magnetar}
        t = np.array([15.0, 50.0, 120.0, 300.0]) * u.day
        emergence = cls(bundle, redshift=z, trapping_mode="emergence").compute_photosphere(t, parameters)
        unleaky = cls(bundle, redshift=z, gamma_trapping=False).compute_photosphere(t, parameters)
        A = 9 * kappa_magnetar.value * (6 * u.Msun).to_value(u.g) ** 2 / (40 * np.pi * 3e51)
        t_rest = (t / (1 + z)).to_value(u.s)
        np.testing.assert_allclose(
            emergence.luminosity.value, -np.expm1(-A / t_rest**2) * unleaky.luminosity.value, rtol=1e-12
        )

    def test_the_modes_give_different_light_curves(self, bundle):
        variables = {"band_index": ["V", "R", "V", "R"], "time": [30, 30, 150, 150] * u.day}
        parameters = {"kappa_magnetar": 0.02 * u.cm**2 / u.g}
        deposition = MagnetarArnettModel(bundle, redshift=0.1)
        emergence = MagnetarArnettModel(bundle, redshift=0.1, trapping_mode="emergence")
        a = deposition.forward_model(variables, parameters).flux.value
        b = emergence.forward_model(variables, parameters).flux.value
        assert not np.allclose(a, b, rtol=1e-2, atol=0.0)

    @pytest.mark.parametrize("cls", _MODELS)
    def test_round_trip_keeps_the_mode(self, bundle, cls):
        model = cls(bundle, redshift=0.05, trapping_mode="emergence")
        restored = cls.from_model_spec(model.to_model_spec())
        assert restored.engine.trapping_mode == "emergence"


# --------------------------------------------------------------------------------------------------------------- #
# UV-suppressed (modified) blackbody variants                                                                     #
# --------------------------------------------------------------------------------------------------------------- #
@pytest.fixture
def uv_bundle() -> FilterBundle:
    """A far-UV (1300 Angstrom, 1180 Angstrom in the rest frame at z = 0.1) and a V filter."""
    return FilterBundle({"UV": _box_filter(1300e-8, 200e-8, "UV"), "V": _box_filter(550e-7, 80e-7, "V")})


class TestCutoffModels:
    @pytest.mark.parametrize("cutoff,plain", list(zip(_CUTOFF_MODELS, _PLAIN_MODELS)))
    def test_parameters(self, bundle, cutoff, plain):
        assert [p.name for p in cutoff.PARAMETERS] == [p.name for p in plain.PARAMETERS] + ["lambda_cut"]
        model = cutoff(bundle, redshift=0.05)
        assert model.coerce_model_parameters({})["lambda_cut"] == 3000.0
        assert cutoff._SOURCE_PARAMETERS == plain._SOURCE_PARAMETERS
        assert cutoff.ENGINE_CLASS is plain.ENGINE_CLASS

    @pytest.mark.parametrize("cutoff,plain", list(zip(_CUTOFF_MODELS, _PLAIN_MODELS)))
    def test_the_photosphere_is_unchanged(self, bundle, cutoff, plain):
        t = [5, 20, 60, 150] * u.day
        a = plain(bundle, redshift=0.1).compute_photosphere(t)
        b = cutoff(bundle, redshift=0.1).compute_photosphere(t)
        for field in ("luminosity", "temperature", "radius"):
            np.testing.assert_allclose(getattr(a, field).value, getattr(b, field).value)

    @pytest.mark.parametrize("cutoff,plain", list(zip(_CUTOFF_MODELS, _PLAIN_MODELS)))
    def test_no_cutoff_is_the_plain_model(self, bundle, cutoff, plain):
        variables = {"band_index": ["V", "R", "V", "R"], "time": [10, 10, 40, 40] * u.day}
        expected = plain(bundle, redshift=0.1).forward_model(variables, {}).flux
        result = cutoff(bundle, redshift=0.1).forward_model(variables, {"lambda_cut": 0.0}).flux
        np.testing.assert_allclose(result.value, expected.value, rtol=1e-12)

    @pytest.mark.parametrize("cls", _CUTOFF_MODELS)
    def test_the_uv_is_suppressed_and_the_optical_boosted(self, uv_bundle, cls):
        """At a hot epoch the far UV is dimmer than the blackbody, and the optical brighter.

        (Only the far UV: the spectrum is renormalised by 1/f(T), so it is brighter than the blackbody wherever
        lambda / lambda_cut > f(T), which is about 1450 Angstrom for a 33000 K photosphere.)
        """
        plain_cls = _PLAIN_MODELS[_CUTOFF_MODELS.index(cls)]
        variables = {"band_index": ["UV", "V"], "time": [12, 12] * u.day}
        parameters = {"T_floor": 1000 * u.K}  # let the photosphere cool freely, so it is hot early on
        plain = plain_cls(uv_bundle, redshift=0.1).forward_model(variables, parameters).flux.value
        cutoff = cls(uv_bundle, redshift=0.1).forward_model(variables, parameters).flux.value
        assert cutoff[0] < plain[0]
        assert cutoff[1] > plain[1]

    @pytest.mark.parametrize("redshift", [0.0, 0.5])
    @pytest.mark.parametrize("cls", _CUTOFF_MODELS)
    def test_bolometric_flux_is_conserved(self, bundle, cls, redshift):
        """The renormalised spectrum carries the whole luminosity, at any redshift."""
        model = cls(bundle, luminosity_distance=100 * u.Mpc) if redshift == 0.0 else cls(bundle, redshift=redshift)
        parameters = {"T_floor": 1.0 * u.K}
        t_obs = np.array([5.0, 20.0, 60.0]) * _DAY * (1 + redshift)

        flux = _bolometric_flux(model, parameters, t_obs)
        L = model.compute_photosphere(t_obs * u.s, parameters).luminosity.to_value(u.erg / u.s)
        D_L = model.luminosity_distance.to_value(u.cm)
        np.testing.assert_allclose(flux, L / (4 * np.pi * D_L**2), rtol=2e-3)

    @pytest.mark.parametrize("cls", _CUTOFF_MODELS)
    def test_round_trip(self, bundle, cls):
        model = cls(bundle, redshift=0.05)
        restored = cls.from_model_spec(model.to_model_spec())
        assert isinstance(restored, cls)
        variables = {"band_index": ["V", "R"], "time": [20, 35] * u.day}
        parameters = {"lambda_cut": 3500.0}
        np.testing.assert_allclose(
            restored.forward_model(variables, parameters).flux.value,
            model.forward_model(variables, parameters).flux.value,
        )

    def test_a_larger_cutoff_suppresses_more(self, uv_bundle):
        model = MagnetarArnettCutoffModel(uv_bundle, redshift=0.1)
        variables = {"band_index": ["UV"], "time": [12] * u.day}
        low = model.forward_model(variables, {"lambda_cut": 2500.0, "T_floor": 1000 * u.K}).flux.value
        high = model.forward_model(variables, {"lambda_cut": 4500.0, "T_floor": 1000 * u.K}).flux.value
        assert high < low
