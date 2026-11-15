r"""
Angular segmentation and emitting geometry of the N-component Diggins26 models.

The N-component models split the sphere into ``N`` axisymmetric, north-south symmetric segments, each a range of
colatitude :math:`[\theta_a, \theta_b]` (mirrored about the equator) with a shock solved along one representative
sightline. This module holds the geometry shared by the models:

- :func:`compute_segmentation` places the segment edges and representative sightlines,
- :func:`segment_volume_fractions` gives each segment's analytic share of the shell volume, and
- :func:`projected_area_fractions` and :func:`build_segment_filling_table` give each segment's sky-projected area
  fraction, in either of two definitions (see below).

.. rubric:: Segmentation

The disk half-maximum latitude :math:`\phi_e = \arctan(b\ln 2)` (see
:meth:`~trilobite.dynamics.profiles.csm.AxisymmetricCSMDensityProfile.disk_half_opening_angle`) is always a segment
edge, so the equatorial (disk) and polar regions stay distinct at any ``N``. The segments are split between them by a
rule that does not depend on the model parameters, ``n_polar = N // 2`` and ``n_equatorial = N - n_polar``, so that the
sampled likelihood is continuous. The disk density is :math:`G(\theta) = f_{\rm pole} + (1 - f_{\rm pole})e^{-u}`
with :math:`u = |\cot\theta|/b`; edges are spaced evenly in :math:`u` over the equatorial region
(:math:`u \in [0, \ln 2]`) and evenly in :math:`s = e^{-u}` over the polar region (:math:`s \in [0, 1/2]`), so that
segments concentrate where the density changes. For ``N = 2`` the only interior edge is :math:`\phi_e`, which is the
split of :class:`~trilobite.models.LFBOTs.diggins26.Diggins26_TwoComponentModel`.

.. rubric:: Projected areas

``mode="hit"`` is the area of the sky pixels whose line of sight touches the segment's shell, which is the definition
of the two-component model. It is correct for a segment on its own, but the segments overlap on the sky: summing the
fluxes of independent slabs counts a sightline that crosses several segments once per segment, and the
optically thick flux overcounts roughly in proportion to the sum of the areas, which grows with ``N``.
``mode="chord"`` instead gives each segment, for every pixel, the share of the pixel's area equal to the segment's
share of that sightline's path through the shell. The segments' areas then partition the shell's projected area, and
:math:`\ell_k = V_k/A_k` remains the mean path length that the one-zone synchrotron engine needs.

Both are computed in units of each segment's own shock radius (every segment is a unit-radius shell
:math:`1 - \eta < r < 1` with :math:`\eta = 1 - (1 - f)^{1/3}`), as in
:meth:`~trilobite.dynamics.profiles.csm.AxisymmetricCSMDensityProfile.build_filling_parameter_table`. They do not
capture the depth ordering of segments whose radii differ.
"""

from typing import NamedTuple

import numpy as np

from trilobite.dynamics.profiles.csm import AxisymmetricCSMDensityProfile

__all__ = [
    "Segmentation",
    "compute_segmentation",
    "segment_volume_fractions",
    "projected_area_fractions",
    "build_segment_filling_table",
    "interpolate_segment_filling_table",
]

AREA_MODES = ("chord", "hit")


class Segmentation(NamedTuple):
    """The segments of the sphere, ordered from the pole (index 0) to the equator."""

    edges: np.ndarray
    """Colatitudes of the ``N + 1`` segment edges in radians, from ``0`` (pole) to ``pi / 2`` (equator)."""
    colatitude: np.ndarray
    """Colatitude of each segment's representative sightline in radians, shape ``(N,)``."""
    n_polar: int
    """Number of polar segments; segments ``0 .. n_polar - 1`` are polar and the rest equatorial."""


def split_segments(n_sectors: int) -> tuple[int, int]:
    """
    Split ``n_sectors`` segments between the polar and equatorial regions.

    The split does not depend on any model parameter, so that the sampled likelihood is continuous.

    Parameters
    ----------
    n_sectors : int
        Total number of segments, at least 2.

    Returns
    -------
    tuple of int
        ``(n_polar, n_equatorial)``, with ``n_polar = n_sectors // 2``.

    Raises
    ------
    ValueError
        If ``n_sectors`` is less than 2.
    """
    n_sectors = int(n_sectors)
    if n_sectors < 2:
        raise ValueError("`n_sectors` must be at least 2 (one polar and one equatorial segment).")
    n_polar = n_sectors // 2
    return n_polar, n_sectors - n_polar


def _colatitude(u: np.ndarray, aspect: float) -> np.ndarray:
    r"""Colatitude at which the density variable :math:`u = |\cot\theta|/b` takes the value ``u``."""
    with np.errstate(divide="ignore"):
        return np.arctan2(1.0, aspect * np.asarray(u, dtype=float))


def compute_segmentation(aspect: float, n_sectors: int) -> Segmentation:
    r"""
    Segment edges and representative sightlines for a disk of angular thickness ``aspect``.

    Parameters
    ----------
    aspect : float
        Angular thickness :math:`b` of the CSM disk. Must be positive.
    n_sectors : int
        Total number of segments, at least 2. ``n_sectors // 2`` are polar and the rest equatorial.

    Returns
    -------
    Segmentation
        The edges and representative colatitudes. The innermost polar segment is represented by the pole, the
        innermost equatorial segment by the equator, and every other segment by its midpoint in the density variable
        (:math:`u` in the equatorial region, :math:`s = e^{-u}` in the polar region).
    """
    aspect = float(aspect)
    if not (np.isfinite(aspect) and aspect > 0):
        raise ValueError("`aspect` must be positive and finite.")
    n_polar, n_equatorial = split_segments(n_sectors)
    ln2 = np.log(2.0)

    # Equatorial region, from its polar edge (u = ln 2) to the equator (u = 0), evenly spaced in u.
    u_eq = ln2 * (1.0 - np.arange(n_equatorial + 1) / n_equatorial)
    # Polar region, from the pole (s = 0) to its equatorial edge (s = 1/2), evenly spaced in s = exp(-u).
    s_pol = 0.5 * np.arange(n_polar + 1) / n_polar

    with np.errstate(divide="ignore"):
        u_pol = -np.log(s_pol)  # inf at the pole
    edges = np.concatenate([_colatitude(u_pol, aspect), _colatitude(u_eq[1:], aspect)])
    edges[0], edges[-1] = 0.0, 0.5 * np.pi

    # Representative sightlines at the midpoints of the density variable.
    s_mid = 0.5 * (s_pol[1:] + s_pol[:-1])
    u_mid_eq = 0.5 * (u_eq[1:] + u_eq[:-1])
    with np.errstate(divide="ignore"):
        colatitude = np.concatenate([_colatitude(-np.log(s_mid), aspect), _colatitude(u_mid_eq, aspect)])
    colatitude[0], colatitude[-1] = 0.0, 0.5 * np.pi
    return Segmentation(edges=edges, colatitude=colatitude, n_polar=n_polar)


def segment_volume_fractions(edges: np.ndarray, filling_factor: float) -> np.ndarray:
    r"""
    Fraction of the full-sphere volume held by each segment, :math:`f_{V,k} = f\,(\cos\theta_a - \cos\theta_b)`.

    This is exact, and sums to the shell's filling factor ``f``.

    Parameters
    ----------
    edges : numpy.ndarray
        Colatitudes of the segment edges from the pole to the equator, as in :class:`Segmentation`.
    filling_factor : float
        Shell volume filling factor :math:`f`.

    Returns
    -------
    numpy.ndarray
        The volume fraction of each segment, shape ``(N,)``.
    """
    edges = np.asarray(edges, dtype=float)
    return float(filling_factor) * (np.cos(edges[:-1]) - np.cos(edges[1:]))


def projected_area_fractions(
    edges: np.ndarray, inclination: float, eta: float, *, mode: str = "chord", n_pix: int = 200
) -> np.ndarray:
    r"""
    Sky-projected area fraction :math:`f_A = A_\perp/\pi R^2` of every segment of a unit-radius shell.

    Parameters
    ----------
    edges : numpy.ndarray
        Colatitudes of the segment edges from the pole to the equator, as in :class:`Segmentation`.
    inclination : float
        Angle between the symmetry axis and the line of sight, in radians.
    eta : float
        Fractional thickness of the shell, :math:`\eta = 1 - (1 - f)^{1/3}`.
    mode : {"chord", "hit"}, optional
        ``"hit"`` counts every pixel whose line of sight meets the segment's shell. ``"chord"`` splits each pixel
        between the segments in proportion to their path lengths through the shell. See the module description.
    n_pix : int, optional
        Sky pixels across the diameter. Default 200.

    Returns
    -------
    numpy.ndarray
        The area fraction of each segment, shape ``(N,)``, ordered from the pole to the equator.
    """
    if mode not in AREA_MODES:
        raise ValueError(f"`mode` must be one of {AREA_MODES}, not {mode!r}.")
    edges = np.asarray(edges, dtype=float)
    n_seg = edges.size - 1

    # Pixel centres across [-1, 1]; both the segments and the shell are mirror-symmetric under x -> -x, so only
    # x > 0 is kept and the area is doubled at the end.
    pixel_size = 2.0 / n_pix
    centres = -1.0 + (np.arange(n_pix) + 0.5) * pixel_size
    x_sky, y_sky = np.meshgrid(centres[centres > 0], centres)
    rho_sq = x_sky.ravel() ** 2 + y_sky.ravel() ** 2
    y_sky = y_sky.ravel()
    on_disk = rho_sq < 1.0  # only lines of sight inside the unit disk can meet the shell
    rho_sq, y_sky = rho_sq[on_disk], y_sky[on_disk]

    # Along a line of sight r^2 = rho^2 + z^2, so the shell occupies z_in < |z| < z_out.
    z_out = np.sqrt(1.0 - rho_sq)
    z_in = np.sqrt(np.clip((1.0 - eta) ** 2 - rho_sq, 0.0, None))

    # A segment boundary at colatitude theta_e is where cos^2(theta) = c_e^2, with cos(theta) = (y sin i + z cos i)/r.
    # Multiplying by r^2 avoids square roots: g_e(z) = (y sin i + z cos i)^2 - c_e^2 (rho^2 + z^2) = qa z^2 + qb z + qc,
    # positive where |cos(theta)| exceeds c_e (nearer the pole). Its roots are where the line of sight crosses the
    # boundary.
    sin_inc, cos_inc = np.sin(inclination), np.cos(inclination)
    thresholds_sq = np.cos(edges[1:-1]) ** 2  # interior edges, from the pole to the equator
    qa = cos_inc**2 - thresholds_sq  # (n_edges,)
    qb = 2.0 * y_sky * sin_inc * cos_inc  # (n_pix,)
    qc = (y_sky * sin_inc)[:, None] ** 2 - thresholds_sq[None, :] * rho_sq[:, None]  # (n_pix, n_edges)

    with np.errstate(invalid="ignore", divide="ignore"):
        regular = np.abs(qa) > 1e-14
        disc = qb[:, None] ** 2 - 4.0 * qa[None, :] * qc
        sqrt_disc = np.sqrt(np.where(disc > 0.0, disc, np.nan))  # NaN: no crossing
        safe_qa = np.where(regular, qa, 1.0)[None, :]
        root_lo = np.where(regular[None, :], (-qb[:, None] - sqrt_disc) / (2.0 * safe_qa), -qc / qb[:, None])
        # A line of sight parallel to the boundary has a linear g_e(z): a single crossing and no second root.
        root_hi = np.where(regular[None, :], (-qb[:, None] + sqrt_disc) / (2.0 * safe_qa), np.nan)

    # Missing roots (NaN) and roots outside the sphere move onto [-z_out, z_out]; a missing root lands on -z_out and
    # only creates a zero-length piece, so every line of sight has the same number of breakpoints.
    cone_roots = [np.clip(np.nan_to_num(r, nan=-2.0), -z_out[:, None], z_out[:, None]) for r in (root_lo, root_hi)]
    breakpoints = np.sort(
        np.concatenate([np.stack([-z_out, -z_in, z_in, z_out], axis=1), *cone_roots], axis=1),
        axis=1,
    )
    z_lo, z_hi = breakpoints[:, :-1], breakpoints[:, 1:]
    z_mid = 0.5 * (z_lo + z_hi)
    length = z_hi - z_lo

    # Between consecutive breakpoints nothing changes sign, so the midpoint decides which segment a piece belongs
    # to and whether it lies in the shell.
    r_mid_sq = rho_sq[:, None] + z_mid**2
    in_shell = (r_mid_sq > (1.0 - eta) ** 2) & (r_mid_sq < 1.0)
    cos_sq_mid = (y_sky[:, None] * sin_inc + z_mid * cos_inc) ** 2 / r_mid_sq
    # Number of interior boundaries whose cone (|cos| above the boundary's) contains the piece: n_seg - 1 is the
    # polar segment and 0 the equatorial-most one.
    nearer_pole = np.searchsorted(thresholds_sq[::-1], cos_sq_mid)
    segment = (n_seg - 1) - nearer_pole  # 0 is the polar segment

    # Accumulate the path length (or the number of pieces) of every segment along every line of sight.
    n_los = rho_sq.size
    slot = (np.arange(n_los)[:, None] * n_seg + segment)[in_shell]
    path = np.bincount(slot, weights=length[in_shell], minlength=n_los * n_seg).reshape(n_los, n_seg)

    area_scale = 2.0 * pixel_size**2 / np.pi  # pixel area x 2 (mirror half-plane) / pi (unit disk area)
    if mode == "hit":
        # A line of sight meets a segment if some piece of non-zero length lies in the segment's shell.
        return (path > 0.0).sum(axis=0) * area_scale
    # Each line of sight splits its pixel between the segments in proportion to their path lengths.
    return (path / path.sum(axis=1, keepdims=True)).sum(axis=0) * area_scale


def build_segment_filling_table(
    aspect, inclination, filling_factor, n_sectors: int, *, mode: str = "chord", n_pix: int = 200, progress=False
) -> np.ndarray:
    r"""
    Tabulate the projected area fraction of every segment on a grid of aspect, inclination and filling factor.

    This is the N-segment counterpart of
    :meth:`~trilobite.dynamics.profiles.csm.AxisymmetricCSMDensityProfile.build_filling_parameter_table`. The
    segment edges depend on the aspect (see :func:`compute_segmentation`), so they are recomputed for every grid
    aspect.

    Parameters
    ----------
    aspect : array-like
        Grid of disk angular thicknesses. Must be positive.
    inclination : array-like
        Grid of inclinations in radians. Both hemispheres are symmetric, so :math:`[0, \pi/2]` suffices.
    filling_factor : array-like
        Grid of shell volume filling factors, within :math:`(0, 1)`.
    n_sectors : int
        Number of segments.
    mode : {"chord", "hit"}, optional
        Area definition; see :func:`projected_area_fractions`.
    n_pix : int, optional
        Sky pixels across the diameter. Default 200.
    progress : bool, optional
        Show a ``tqdm`` progress bar. Default ``False``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sectors, len(aspect), len(inclination), len(filling_factor))``, ordered from the pole to
        the equator along the first axis.
    """
    aspect = np.atleast_1d(np.asarray(aspect, dtype=float))
    inclination = np.atleast_1d(np.asarray(inclination, dtype=float))
    filling_factor = np.atleast_1d(np.asarray(filling_factor, dtype=float))
    if np.any(aspect <= 0) or not np.all(np.isfinite(aspect)):
        raise ValueError("`aspect` must be positive and finite.")
    if np.any((filling_factor <= 0) | (filling_factor >= 1)):
        raise ValueError("`filling_factor` must lie in (0, 1).")
    if not np.all(np.isfinite(inclination)):
        raise ValueError("`inclination` must be finite.")

    eta = 1.0 - (1.0 - filling_factor) ** (1.0 / 3.0)
    table = np.empty((n_sectors, aspect.size, inclination.size, filling_factor.size))
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(total=aspect.size * inclination.size, desc="Building filling-factor table")
        except ImportError:
            pass
    for i, b in enumerate(aspect):
        edges = compute_segmentation(b, n_sectors).edges
        for j, inc in enumerate(inclination):
            for k, et in enumerate(eta):
                table[:, i, j, k] = projected_area_fractions(edges, inc, et, mode=mode, n_pix=n_pix)
            if bar is not None:
                bar.update(1)
    if bar is not None:
        bar.close()
    return table


def interpolate_segment_filling_table(table: tuple, filling_factor, inclination, aspect) -> np.ndarray:
    r"""
    Interpolate the area fractions of every segment from a precomputed table.

    Parameters
    ----------
    table : tuple
        ``(aspect_grid, inclination_grid, filling_factor_grid, f_A_table)`` with ``f_A_table`` from
        :func:`build_segment_filling_table`.
    filling_factor, inclination, aspect : float
        The point to evaluate. The inclination is folded into :math:`[0, \pi/2]`.

    Returns
    -------
    numpy.ndarray
        The area fractions, shape ``(N,)``. The interpolation is trilinear in :math:`(\ln b, i, f)` and points
        beyond the grid take the value at its nearest edge.
    """
    aspect_grid, inc_grid, f_grid, f_A_table = table
    f_A_table = np.asarray(f_A_table, dtype=float)
    return AxisymmetricCSMDensityProfile._interpolate_filling_table(
        f_A_table,
        (
            np.log(np.asarray(aspect_grid, dtype=float)),
            np.asarray(inc_grid, dtype=float),
            np.asarray(f_grid, dtype=float),
        ),
        (
            np.log(np.atleast_1d(float(aspect))),
            np.atleast_1d(_fold_inclination(inclination)),
            np.atleast_1d(float(filling_factor)),
        ),
    )[:, 0]


def _fold_inclination(inclination: float) -> float:
    r"""Fold an inclination into :math:`[0, \pi/2]`, using the symmetry under :math:`i \to \pi - i`."""
    inc = float(np.mod(inclination, np.pi))
    return np.pi - inc if inc > 0.5 * np.pi else inc
