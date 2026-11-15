"""Tests for the N-component Diggins26 models."""

import warnings

import numpy as np
import pytest
from astropy import units as u

import trilobite.models.LFBOTs.diggins26_n_component as n_component_module
from trilobite.models.LFBOTs import (
    Diggins26_NComponentModel,
    Diggins26_RelativisticNComponentModel,
    Diggins26_RelativisticTwoComponentModel,
    Diggins26_TwoComponentModel,
    Diggins26NDynamics,
    Diggins26NTabulation,
)

DAY = 86400.0
TIMES = np.array([3.0, 10.0, 30.0, 100.0]) * DAY
FREQS = np.array([1e9, 5e9, 2e10, 1e11])


# =========================================================================== #
# N = 2 is the two-component model                                            #
# =========================================================================== #
@pytest.fixture(scope="module")
def two():
    return Diggins26_TwoComponentModel(luminosity_distance=100 * u.Mpc)


@pytest.fixture(scope="module")
def n2(two):
    return Diggins26_NComponentModel(luminosity_distance=100 * u.Mpc, n_sectors=2, area_mode="hit")


@pytest.fixture(scope="module")
def rel_two():
    return Diggins26_RelativisticTwoComponentModel(luminosity_distance=100 * u.Mpc)


@pytest.fixture(scope="module")
def rel_n2():
    return Diggins26_RelativisticNComponentModel(luminosity_distance=100 * u.Mpc, n_sectors=2, area_mode="hit")


class TestTwoSegmentLimit:
    @pytest.mark.parametrize("pair", [("two", "n2"), ("rel_two", "rel_n2")])
    def test_forward_model_is_identical(self, pair, request):
        parent, model = (request.getfixturevalue(name) for name in pair)
        p = {"aspect": 1.3, "inclination": 0.8, "f_V": 0.35}
        t, nu = np.meshgrid(TIMES, FREQS)
        a = parent.forward_model({"time": t * u.s, "frequency": nu * u.Hz}, p).flux_density
        b = model.forward_model({"time": t * u.s, "frequency": nu * u.Hz}, p).flux_density
        np.testing.assert_allclose(b.value, a.value, rtol=1e-12, atol=0.0)
        assert np.all(a.value > 0)

    @pytest.mark.parametrize("pair", [("two", "n2"), ("rel_two", "rel_n2")])
    def test_tabulation_is_identical_sector_by_sector(self, pair, request):
        parent, model = (request.getfixturevalue(name) for name in pair)
        a, b = parent.tabulate(TIMES, FREQS, {}), model.tabulate(TIMES, FREQS, {})
        np.testing.assert_allclose(b.flux_density.value, a.flux_density.value, rtol=1e-12, atol=0.0)
        # The segments run from the pole to the equator; the parent lists the equator first.
        for mine, theirs in ((b.sectors[0], a.polar), (b.sectors[1], a.equatorial)):
            for field in ("flux_density", "flux_density_attenuated", "tau_ff", "emission_measure", "magnetic_field"):
                np.testing.assert_allclose(
                    np.asarray(getattr(mine, field)), np.asarray(getattr(theirs, field)), rtol=1e-12, atol=0.0
                )
            for field in ("area_fraction", "volume_fraction", "path_length_ratio"):
                assert getattr(mine, field) == pytest.approx(getattr(theirs, field), rel=1e-12)

    def test_dynamics_are_identical(self, two, n2):
        a, b = two.compute_dynamics({}, TIMES), n2.compute_dynamics({}, TIMES)
        for mine, theirs in ((b.sectors[0], a.polar), (b.sectors[1], a.equatorial)):
            np.testing.assert_allclose(mine.radius.value, theirs.radius.value, rtol=1e-12)
            np.testing.assert_allclose(
                mine.thermal_energy_density.value, theirs.thermal_energy_density.value, rtol=1e-12
            )


# =========================================================================== #
# Construction                                                                #
# =========================================================================== #
class TestConstruction:
    def test_defaults(self):
        m = Diggins26_NComponentModel(redshift=0.01)
        assert m.n_sectors == 8 and m.area_mode == "chord"
        assert Diggins26_RelativisticNComponentModel(redshift=0.01).n_sectors == 8

    @pytest.mark.parametrize("n", [1, 0, -2])
    def test_rejects_too_few_segments(self, n):
        with pytest.raises(ValueError, match="at least 2"):
            Diggins26_NComponentModel(redshift=0.01, n_sectors=n)

    def test_rejects_unknown_area_mode(self):
        with pytest.raises(ValueError, match="area_mode"):
            Diggins26_NComponentModel(redshift=0.01, area_mode="sum")

    def test_hit_mode_warns_only_beyond_two_segments(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Diggins26_NComponentModel(redshift=0.01, n_sectors=2, area_mode="hit")
        with pytest.warns(UserWarning, match="overcounts"):
            Diggins26_NComponentModel(redshift=0.01, n_sectors=4, area_mode="hit")

    def test_the_parents_arguments_still_apply(self):
        m = Diggins26_NComponentModel(redshift=0.01, hydrogen_mass_fraction=0.6, n_azimuth=6, n_sectors=3)
        assert m.mean_molecular_weight == pytest.approx(
            Diggins26_TwoComponentModel(redshift=0.01, hydrogen_mass_fraction=0.6).mean_molecular_weight
        )
        assert m._n_azimuth == 6

    def test_relativistic_model_uses_the_relativistic_engine_and_thermal_electrons(self):
        m = Diggins26_RelativisticNComponentModel(redshift=0.01, n_sectors=3)
        assert type(m._shock_engine).__name__ == "RelPressureDrivenThinShellShockEngine"
        names = [parameter.name for parameter in Diggins26_RelativisticNComponentModel.PARAMETERS]
        assert "epsilon_T" in names
        assert "epsilon_T" not in [parameter.name for parameter in Diggins26_NComponentModel.PARAMETERS]

    def test_status_is_declared(self):
        for cls in (Diggins26_NComponentModel, Diggins26_RelativisticNComponentModel):
            assert "STATUS" in cls.__dict__ and cls.STATUS_NOTE

    @pytest.mark.parametrize("cls", [Diggins26_NComponentModel, Diggins26_RelativisticNComponentModel])
    def test_round_trips_through_the_model_spec(self, cls):
        m = cls(redshift=0.02, n_sectors=5, area_mode="chord", n_azimuth=3)
        spec = m.to_model_spec()
        assert spec.kwargs["n_sectors"] == 5 and spec.kwargs["area_mode"] == "chord"
        rebuilt = cls.from_model_spec(spec)
        assert type(rebuilt) is cls
        assert (rebuilt.n_sectors, rebuilt.area_mode, rebuilt._n_azimuth) == (5, "chord", 3)
        assert rebuilt.redshift == pytest.approx(m.redshift)


# =========================================================================== #
# Geometry and structure (coarse area table: the tests do not need accuracy)  #
# =========================================================================== #
@pytest.fixture(scope="module")
def coarse_tables():
    patch = pytest.MonkeyPatch()
    patch.setattr(
        n_component_module,
        "_FILLING_GRID",
        (np.geomspace(0.2, 25.0, 6), np.linspace(0, np.pi / 2, 5), np.linspace(0.05, 0.95, 4)),
    )
    patch.setattr(n_component_module, "_FILLING_N_PIX", 100)
    yield
    patch.undo()


@pytest.fixture(scope="module")
def chord8(coarse_tables):
    return Diggins26_NComponentModel(luminosity_distance=100 * u.Mpc, n_sectors=8)


class TestStructure:
    def test_geometry_conserves_the_volume(self, chord8):
        p = chord8._coerce({"aspect": 1.7, "f_V": 0.4, "inclination": 0.9})
        geometry = chord8._sector_geometry(p)
        assert len(geometry) == 8
        assert sum(g["f_V"] for g in geometry) == pytest.approx(0.4, rel=1e-12)
        for g in geometry:
            assert g["f_A"] > 0 and g["ell_over_R"] == pytest.approx(4 / 3 * g["f_V"] / g["f_A"])
        # The areas partition the shell's silhouette (about the unit disk) in chord mode.
        assert sum(g["f_A"] for g in geometry) == pytest.approx(1.0, abs=0.05)

    def test_sightline_sampling(self, chord8):
        p = chord8._coerce({"inclination": 1.0})
        sightlines = chord8._sector_sightlines(p)
        assert len(sightlines) == 8
        (cos_pole, z_pole), (cos_eq, z_eq) = sightlines[0], sightlines[-1]
        np.testing.assert_allclose(cos_pole, [np.cos(1.0), -np.cos(1.0)])
        np.testing.assert_array_equal(z_pole, [1.0, -1.0])
        assert cos_eq.size == chord8._n_azimuth and np.all(z_eq == 0.0)
        for cos_psi, z_hat in sightlines:
            assert np.all(np.abs(cos_psi) <= 1.0 + 1e-12) and np.all(np.abs(z_hat) <= 1.0)

    def test_dynamics_container(self, chord8):
        dyn = chord8.compute_dynamics({"aspect": 3.0}, TIMES)
        assert isinstance(dyn, Diggins26NDynamics)
        seg = chord8.segmentation(3.0)
        assert len(dyn.sectors) == 8 and dyn.n_polar == 4 == seg.n_polar
        np.testing.assert_allclose(dyn.colatitude, seg.colatitude)
        np.testing.assert_allclose(dyn.colatitude_edges, seg.edges)
        # The disk is densest at the equator, so the shock there is slowest and the polar one fastest.
        radii = np.array([s.radius.value[-1] for s in dyn.sectors])
        assert np.all(np.diff(radii) < 0)

    def test_tabulation_is_the_sum_of_its_segments(self, chord8):
        tab = chord8.tabulate(TIMES, FREQS, {"aspect": 1.0})
        assert isinstance(tab, Diggins26NTabulation) and len(tab.sectors) == 8
        total = sum(s.flux_density_attenuated.to_value(u.Jy) for s in tab.sectors)
        np.testing.assert_allclose(tab.flux_density.to_value(u.Jy), total, rtol=1e-12)
        np.testing.assert_allclose(sum(s.volume_fraction for s in tab.sectors), 0.5, rtol=1e-12)

    def test_forward_model_matches_the_tabulation(self, chord8):
        p = {"aspect": 1.5, "inclination": 0.6}
        tab = chord8.tabulate(TIMES, FREQS, p).flux_density.to_value(u.Jy)
        paired = chord8.forward_model({"time": TIMES * u.s, "frequency": FREQS * u.Hz}, p).flux_density
        np.testing.assert_allclose(paired.to_value(u.Jy), np.diag(tab), rtol=1e-9)

    def test_dynamics_can_be_reused_and_must_match(self, chord8, coarse_tables):
        p = {"aspect": 1.5}
        dyn = chord8.compute_dynamics(p, TIMES)
        np.testing.assert_allclose(
            chord8.tabulate(TIMES, FREQS, p, dynamics=dyn).flux_density.value,
            chord8.tabulate(TIMES, FREQS, p).flux_density.value,
            rtol=1e-12,
        )
        with pytest.raises(ValueError, match="different `aspect`"):
            chord8.tabulate(TIMES, FREQS, {"aspect": 2.5}, dynamics=dyn)
        other = Diggins26_NComponentModel(luminosity_distance=100 * u.Mpc, n_sectors=4)
        with pytest.raises(ValueError, match="n_sectors"):
            other.tabulate(TIMES, FREQS, p, dynamics=dyn)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_polar_cap_is_kept_for_a_thick_disk(self, coarse_tables, n):
        """At b = 20 the polar cap is only about 4 degrees wide, but its segment is still there and radiates."""
        m = Diggins26_NComponentModel(luminosity_distance=100 * u.Mpc, n_sectors=n)
        tab = m.tabulate(TIMES, FREQS, {"aspect": 20.0, "inclination": 0.4, "f_pole": 0.05})
        polar = tab.sectors[0]
        assert tab.dynamics.colatitude[0] == 0.0
        assert tab.dynamics.colatitude_edges[tab.dynamics.n_polar] == pytest.approx(
            0.5 * np.pi - np.arctan(20 * np.log(2))
        )
        assert polar.volume_fraction > 0 and polar.area_fraction > 0
        assert np.all(polar.flux_density.value >= 0) and np.any(polar.flux_density.value > 0)

    def test_infeasible_parameters_give_nan(self, chord8):
        t = np.array([5.0, 20.0]) * DAY * u.s
        out = chord8.forward_model({"time": t, "frequency": np.array([5e9, 5e9]) * u.Hz}, {"E_ej": 1e55}).flux_density
        assert np.all(np.isnan(out.value))
        assert chord8.diagnose({"E_ej": 1e55}) is not None


# =========================================================================== #
# Convergence in N, and the area modes                                        #
# =========================================================================== #
class TestConvergence:
    @pytest.fixture(scope="class")
    def fluxes(self, coarse_tables):
        t = np.array([3.0, 10.0, 30.0, 100.0]) * DAY * u.s
        nu = np.array([1e9, 5e9, 2e10, 1e11]) * u.Hz
        out = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for mode in ("chord", "hit"):
                for n in (2, 4, 8, 16):
                    m = Diggins26_NComponentModel(luminosity_distance=100 * u.Mpc, n_sectors=n, area_mode=mode)
                    out[mode, n] = m.tabulate(t, nu, {}).flux_density.to_value(u.Jy)
        return out

    def test_chord_flux_converges_with_the_number_of_segments(self, fluxes):
        reference = fluxes["chord", 16]
        measurable = reference > 1e-6  # leave the exponentially absorbed corner out
        error = {n: np.abs(fluxes["chord", n][measurable] / reference[measurable] - 1) for n in (2, 4, 8)}
        assert error[2].max() > error[4].max() > error[8].max()
        assert np.median(error[8]) < 0.03 and error[8].max() < 0.15

    def test_hit_flux_does_not_converge(self, fluxes):
        """Counting every touched pixel in every segment keeps raising the optically thick flux with N."""
        thick = fluxes["chord", 8] > 1e-6
        ratio_8 = fluxes["hit", 8][thick] / fluxes["chord", 8][thick]
        ratio_16 = fluxes["hit", 16][thick] / fluxes["chord", 16][thick]
        assert np.median(ratio_8) > 1.5
        assert np.median(ratio_16) > np.median(ratio_8)

    def test_modes_agree_where_the_source_is_optically_thin(self, fluxes):
        """At late times and high frequency the path through the shell is transparent and both definitions agree."""
        for n in (4, 8, 16):
            assert fluxes["hit", n][-1, -1] == pytest.approx(fluxes["chord", n][-1, -1], rel=0.02)
