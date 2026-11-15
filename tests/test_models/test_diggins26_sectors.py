"""Tests for the segmentation and area geometry of the N-component Diggins26 models."""

import numpy as np
import pytest

from trilobite.dynamics.profiles.csm import AxisymmetricCSMDensityProfile
from trilobite.models.LFBOTs.sectors import (
    build_segment_filling_table,
    compute_segmentation,
    interpolate_segment_filling_table,
    projected_area_fractions,
    segment_volume_fractions,
    split_segments,
)

ASPECTS = (0.2, 0.5, 2.0, 8.0, 20.0)


def _phi_e(aspect):
    return np.arctan(aspect * np.log(2.0))


# =========================================================================== #
# Segmentation                                                                #
# =========================================================================== #
class TestSegmentation:
    @pytest.mark.parametrize(("n", "expected"), [(2, (1, 1)), (3, (1, 2)), (4, (2, 2)), (7, (3, 4)), (8, (4, 4))])
    def test_split_does_not_depend_on_parameters(self, n, expected):
        assert split_segments(n) == expected

    @pytest.mark.parametrize("n", [0, 1, -3])
    def test_requires_at_least_two_segments(self, n):
        with pytest.raises(ValueError, match="at least 2"):
            split_segments(n)
        with pytest.raises(ValueError, match="at least 2"):
            compute_segmentation(1.0, n)

    @pytest.mark.parametrize("aspect", [0.0, -1.0, np.inf, np.nan])
    def test_rejects_invalid_aspect(self, aspect):
        with pytest.raises(ValueError, match="aspect"):
            compute_segmentation(aspect, 4)

    @pytest.mark.parametrize("aspect", ASPECTS)
    def test_two_segments_are_the_two_component_split(self, aspect):
        seg = compute_segmentation(aspect, 2)
        np.testing.assert_allclose(seg.edges, [0.0, 0.5 * np.pi - _phi_e(aspect), 0.5 * np.pi], rtol=1e-14)
        np.testing.assert_array_equal(seg.colatitude, [0.0, 0.5 * np.pi])
        assert seg.n_polar == 1

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 16])
    @pytest.mark.parametrize("aspect", ASPECTS)
    def test_disk_edge_is_always_a_segment_edge(self, aspect, n):
        """The polar and equatorial regions stay distinct however coarse the segmentation."""
        seg = compute_segmentation(aspect, n)
        assert seg.edges[seg.n_polar] == pytest.approx(0.5 * np.pi - _phi_e(aspect), rel=1e-12)
        # The polar cap is kept: exactly n // 2 segments lie inside it, however narrow it is.
        assert seg.n_polar == n // 2
        assert np.all(seg.edges[: seg.n_polar + 1] <= 0.5 * np.pi - _phi_e(aspect) + 1e-12)

    @pytest.mark.parametrize("n", [2, 3, 6, 9])
    @pytest.mark.parametrize("aspect", ASPECTS)
    def test_edges_and_representatives_are_ordered(self, aspect, n):
        seg = compute_segmentation(aspect, n)
        assert seg.edges.shape == (n + 1,) and seg.colatitude.shape == (n,)
        assert seg.edges[0] == 0.0 and seg.edges[-1] == 0.5 * np.pi
        assert np.all(np.diff(seg.edges) > 0)
        assert np.all(seg.colatitude >= seg.edges[:-1]) and np.all(seg.colatitude <= seg.edges[1:])
        # The innermost segments are represented by the pole and the equator.
        assert seg.colatitude[0] == 0.0 and seg.colatitude[-1] == 0.5 * np.pi
        assert np.all(np.diff(seg.colatitude) > 0)

    def test_edges_are_evenly_spaced_in_the_density_variable(self):
        aspect, n = 2.0, 8
        seg = compute_segmentation(aspect, n)
        equatorial = seg.edges[seg.n_polar :]
        u = 1.0 / (aspect * np.tan(equatorial[:-1]))  # |cot(theta)| / b; the equator itself is u = 0
        np.testing.assert_allclose(np.diff(np.append(u, 0.0)), -np.log(2.0) / 4, rtol=1e-12)
        polar = seg.edges[: seg.n_polar + 1][1:]
        s = np.exp(-1.0 / (aspect * np.tan(polar)))
        np.testing.assert_allclose(np.diff(np.insert(s, 0, 0.0)), 0.5 / 4, rtol=1e-12)

    def test_refining_resolves_the_density_transition(self):
        """More segments place more edges where the disk density changes."""
        coarse, fine = compute_segmentation(1.0, 4).edges, compute_segmentation(1.0, 16).edges
        assert np.max(np.diff(fine)) < np.max(np.diff(coarse))


# =========================================================================== #
# Volume fractions                                                            #
# =========================================================================== #
class TestVolumeFractions:
    @pytest.mark.parametrize("n", [2, 5, 12])
    @pytest.mark.parametrize("aspect", ASPECTS)
    @pytest.mark.parametrize("f", [0.05, 0.5, 0.95])
    def test_sum_to_the_filling_factor(self, aspect, n, f):
        fv = segment_volume_fractions(compute_segmentation(aspect, n).edges, f)
        assert np.all(fv > 0)
        assert fv.sum() == pytest.approx(f, rel=1e-14)

    @pytest.mark.parametrize("aspect", ASPECTS)
    def test_two_segments_match_the_two_component_split(self, aspect):
        f = 0.4
        polar, equatorial = segment_volume_fractions(compute_segmentation(aspect, 2).edges, f)
        assert equatorial == pytest.approx(f * AxisymmetricCSMDensityProfile.equatorial_solid_angle_fraction(aspect))
        assert polar == pytest.approx(f * AxisymmetricCSMDensityProfile.polar_solid_angle_fraction(aspect))


# =========================================================================== #
# Projected areas                                                             #
# =========================================================================== #
def _eta(f):
    return 1.0 - (1.0 - f) ** (1.0 / 3.0)


class TestProjectedAreas:
    @pytest.mark.parametrize("aspect", [0.3, 2.0, 12.0])
    @pytest.mark.parametrize("inclination", [0.0, 0.7, 0.5 * np.pi])
    @pytest.mark.parametrize("f", [0.1, 0.6])
    def test_two_segment_hit_area_is_the_two_component_area(self, aspect, inclination, f):
        phi_e = _phi_e(aspect)
        new = projected_area_fractions(compute_segmentation(aspect, 2).edges, inclination, _eta(f), mode="hit")
        old = [
            AxisymmetricCSMDensityProfile._projected_area_fraction("polar", 0.5 * np.pi - phi_e, inclination, _eta(f)),
            AxisymmetricCSMDensityProfile._projected_area_fraction("equatorial", phi_e, inclination, _eta(f)),
        ]
        np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("n", [2, 3, 6, 9])
    @pytest.mark.parametrize("inclination", [0.0, 1.0, 0.5 * np.pi])
    def test_chord_areas_partition_the_shells_footprint(self, n, inclination):
        """Summing the chord areas gives the area of the whole shell's silhouette, however many segments."""
        eta = _eta(0.5)
        whole = projected_area_fractions(np.array([0.0, 0.5 * np.pi]), inclination, eta, mode="hit")[0]
        chord = projected_area_fractions(compute_segmentation(2.0, n).edges, inclination, eta, mode="chord")
        assert np.all(chord >= 0)
        assert chord.sum() == pytest.approx(whole, rel=1e-12)

    def test_whole_shell_footprint_is_the_projected_disk(self):
        """A shell of finite thickness seen from outside has the area of its outer sphere's disk."""
        whole = projected_area_fractions(np.array([0.0, 0.5 * np.pi]), 0.9, _eta(0.5), mode="chord")[0]
        assert whole == pytest.approx(1.0, abs=2e-3)

    @pytest.mark.parametrize("n", [3, 6])
    def test_hit_areas_overlap_and_chord_areas_do_not_exceed_them(self, n):
        edges = compute_segmentation(2.0, n).edges
        hit = projected_area_fractions(edges, 1.0, _eta(0.5), mode="hit")
        chord = projected_area_fractions(edges, 1.0, _eta(0.5), mode="chord")
        assert hit.sum() > 1.3  # the sightlines of several segments touch the same pixels
        assert np.all(chord <= hit + 1e-12)

    def test_pole_on_polar_area_is_the_silhouette_of_the_cap(self):
        """Seen pole-on, the polar cap's silhouette is the disk of radius sin(theta_p) around the axis."""
        theta_p = 0.4
        edges = np.array([0.0, theta_p, 0.5 * np.pi])
        # Thin enough that the shell's inner sphere does not hide the cap's footprint.
        hit = projected_area_fractions(edges, 0.0, 0.3, mode="hit", n_pix=400)
        assert hit[0] == pytest.approx(np.sin(theta_p) ** 2, rel=2e-2)

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            projected_area_fractions(np.array([0.0, 1.0, 1.5]), 0.5, 0.2, mode="nope")


# =========================================================================== #
# Table                                                                       #
# =========================================================================== #
class TestTable:
    GRIDS = (np.geomspace(0.3, 6.0, 4), np.linspace(0.0, 0.5 * np.pi, 3), np.array([0.2, 0.6]))

    @pytest.fixture(scope="class")
    def table(self):
        return build_segment_filling_table(*self.GRIDS, 4, mode="chord", n_pix=80)

    def test_shape_and_grid_values(self, table):
        assert table.shape == (4, 4, 3, 2)
        edges = compute_segmentation(self.GRIDS[0][2], 4).edges
        direct = projected_area_fractions(edges, self.GRIDS[1][1], _eta(0.6), mode="chord", n_pix=80)
        np.testing.assert_array_equal(table[:, 2, 1, 1], direct)

    def test_interpolation_reproduces_grid_points_and_clamps(self, table):
        full = (*self.GRIDS, table)
        np.testing.assert_allclose(
            interpolate_segment_filling_table(full, 0.6, self.GRIDS[1][1], self.GRIDS[0][2]),
            table[:, 2, 1, 1],
            rtol=1e-12,
        )
        # Beyond the grid the nearest edge value is used.
        np.testing.assert_allclose(
            interpolate_segment_filling_table(full, 0.9, 0.5 * np.pi, 100.0), table[:, -1, -1, -1], rtol=1e-12
        )
        # The inclination is folded about pi / 2.
        np.testing.assert_allclose(
            interpolate_segment_filling_table(full, 0.4, np.pi - 0.3, 1.0),
            interpolate_segment_filling_table(full, 0.4, 0.3, 1.0),
            rtol=1e-12,
        )

    def test_rejects_bad_grids(self):
        with pytest.raises(ValueError, match="filling_factor"):
            build_segment_filling_table([1.0], [0.5], [1.0], 2)
        with pytest.raises(ValueError, match="aspect"):
            build_segment_filling_table([0.0], [0.5], [0.5], 2)

    def test_shared_interpolation_matches_the_two_component_table(self):
        """The two-component profile interpolates through the same helper, so its results are unchanged."""
        aspect, inc, f = np.geomspace(0.3, 6.0, 4), np.linspace(0.0, 0.5 * np.pi, 3), np.array([0.2, 0.6])
        built = AxisymmetricCSMDensityProfile.build_filling_parameter_table(aspect, inc, f, n_pix=60)
        got = AxisymmetricCSMDensityProfile.compute_filling_parameters(
            0.6, inc[1], aspect[2], table=(aspect, inc, f, built)
        )
        assert got["polar"]["f_A"] == pytest.approx(built[0, 2, 1, 1], rel=1e-12)
        assert got["equatorial"]["f_A"] == pytest.approx(built[1, 2, 1, 1], rel=1e-12)
