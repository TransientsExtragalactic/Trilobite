"""Tests for the segmentation and area geometry of the N-component Diggins26 models."""

import numpy as np
import pytest

from trilobite.dynamics.profiles.csm import AxisymmetricCSMDensityProfile
from trilobite.models.LFBOTs.sectors import (
    compute_segmentation,
    projected_area_fractions,
    segment_geometry,
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
def _quadrature_areas(edges, inclination, n=1500):
    """Brute-force midpoint integral of max(n . o, 0) over each segment's surface, both hemispheres, over pi."""
    cos_t = (np.arange(n) + 0.5) / n  # north hemisphere, cos(theta) in (0, 1)
    phi = (np.arange(n) + 0.5) * 2.0 * np.pi / n
    c, ph = np.meshgrid(cos_t, phi, indexing="ij")
    s_t = np.sqrt(1.0 - c**2)
    weight = (1.0 / n) * (2.0 * np.pi / n)
    out = np.zeros(edges.size - 1)
    for sign in (1.0, -1.0):  # the north zone and its mirror image
        mu = sign * c * np.cos(inclination) + s_t * np.sin(inclination) * np.cos(ph)
        visible = np.maximum(mu, 0.0) * weight
        theta = np.arccos(c)
        for k in range(out.size):
            out[k] += visible[(theta >= edges[k]) & (theta < edges[k + 1])].sum()
    return out / np.pi


class TestProjectedAreas:
    @pytest.mark.parametrize("aspect", [0.3, 2.0, 12.0])
    @pytest.mark.parametrize("n", [2, 3, 6])
    @pytest.mark.parametrize("inclination", [0.0, 0.4, 1.0, 0.5 * np.pi])
    def test_matches_a_brute_force_surface_integral(self, aspect, n, inclination):
        edges = compute_segmentation(aspect, n).edges
        np.testing.assert_allclose(
            projected_area_fractions(edges, inclination), _quadrature_areas(edges, inclination), atol=2e-3
        )

    @pytest.mark.parametrize("n", [2, 3, 6, 9])
    @pytest.mark.parametrize("inclination", [0.0, 1e-9, 0.3, 1.0, 1.5, 0.5 * np.pi, 2.2, np.pi, -0.7])
    def test_areas_partition_the_disk(self, n, inclination):
        """The visible surfaces of a sphere do not overlap on the sky, so the segments' areas sum to pi R^2."""
        f_A = projected_area_fractions(compute_segmentation(2.0, n).edges, inclination)
        assert np.all(f_A >= 0)
        assert f_A.sum() == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("theta_b", [0.4, 1.0])
    def test_pole_on_areas_are_the_flat_silhouettes(self, theta_b):
        """Seen pole-on, only the near hemisphere is visible and a zone is the annulus sin^2(theta_a) to sin^2(theta_b)."""
        edges = np.array([0.0, theta_b, 0.5 * np.pi])
        np.testing.assert_allclose(
            projected_area_fractions(edges, 0.0), [np.sin(theta_b) ** 2, 1.0 - np.sin(theta_b) ** 2], rtol=1e-12
        )

    def test_edge_on_areas_have_a_closed_form(self):
        """At i = pi / 2 the cap area is (theta - sin(theta) cos(theta)), giving (2 d(theta) - d sin(2 theta)) / pi."""
        edges = compute_segmentation(1.3, 5).edges
        expected = (2.0 * np.diff(edges) - np.diff(np.sin(2.0 * edges))) / np.pi
        np.testing.assert_allclose(projected_area_fractions(edges, 0.5 * np.pi), expected, rtol=1e-12)

    def test_single_segment_is_the_whole_disk(self):
        for inclination in (0.0, 0.8, 0.5 * np.pi):
            assert projected_area_fractions(np.array([0.0, 0.5 * np.pi]), inclination)[0] == pytest.approx(1.0)

    def test_is_continuous_in_inclination(self):
        edges = compute_segmentation(0.7, 4).edges
        np.testing.assert_allclose(
            projected_area_fractions(edges, 1e-9), projected_area_fractions(edges, 1e-4), atol=1e-6
        )
        np.testing.assert_allclose(
            projected_area_fractions(edges, 0.5 * np.pi), projected_area_fractions(edges, 0.5 * np.pi - 1e-6), atol=1e-5
        )

    def test_polar_area_shrinks_and_equatorial_area_grows_toward_edge_on(self):
        edges = compute_segmentation(1.0, 2).edges
        inclinations = np.linspace(0.0, 0.5 * np.pi, 12)
        areas = np.array([projected_area_fractions(edges, i) for i in inclinations])
        assert np.all(np.diff(areas[:, 0]) < 0) and np.all(np.diff(areas[:, 1]) > 0)

    @pytest.mark.parametrize("inclination", [0.3, 1.2])
    def test_symmetric_under_inclination_reflection(self, inclination):
        edges = compute_segmentation(1.0, 4).edges
        np.testing.assert_allclose(
            projected_area_fractions(edges, np.pi - inclination),
            projected_area_fractions(edges, inclination),
            rtol=1e-12,
        )

    def test_rejects_non_finite_inclination(self):
        with pytest.raises(ValueError, match="inclination"):
            projected_area_fractions(np.array([0.0, 1.0, 1.5]), np.nan)


class TestSegmentGeometry:
    def test_combines_volume_area_and_path_length(self):
        edges = compute_segmentation(1.5, 5).edges
        geometry = segment_geometry(edges, 0.4, 0.9)
        assert len(geometry) == 5
        np.testing.assert_allclose([g["f_V"] for g in geometry], segment_volume_fractions(edges, 0.4))
        np.testing.assert_allclose([g["f_A"] for g in geometry], projected_area_fractions(edges, 0.9))
        for g in geometry:
            assert g["ell_over_R"] == pytest.approx(4.0 / 3.0 * g["f_V"] / g["f_A"])

    @pytest.mark.parametrize("f", [0.0, 1.0])
    def test_filling_factor_limits_are_clipped(self, f):
        for g in segment_geometry(compute_segmentation(1.0, 3).edges, f, 0.5):
            assert np.isfinite(g["f_V"]) and np.isfinite(g["ell_over_R"])
