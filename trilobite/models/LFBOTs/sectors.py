r"""
Angular segmentation and emitting geometry of the N-component Diggins26 models.

The N-component models split the sphere into ``N`` axisymmetric, north-south symmetric segments, each a range of
colatitude :math:`[\theta_a, \theta_b]` (mirrored about the equator) with a shock solved along one representative
sightline. This module holds the geometry shared by the models:

- :func:`compute_segmentation` places the segment edges and representative sightlines,
- :func:`segment_volume_fractions` gives each segment's analytic share of the shell volume, and
- :func:`projected_area_fractions` gives each segment's sky-projected area fraction, in closed form, and
- :func:`segment_geometry` combines the two into the volume fraction, area fraction and mean path length that the
  synchrotron calculation needs.

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

The projected area of a segment is the area of the sky covered by the part of its spherical surface that faces the
observer: for a surface element of normal :math:`\hat n` and area :math:`dS` it is
:math:`\max(\hat n\cdot\hat o, 0)\,dS`. The sphere is convex, so the visible elements never overlap on the sky, and
the projected areas of the segments partition the disk :math:`\pi R^2` exactly, :math:`\sum_k f_{A,k} = 1`. This is what
makes the segments' fluxes safe to add: a sightline is assigned to the one segment whose surface it meets first,
rather than being counted once per segment it crosses. The areas are elementary functions of the inclination, so no
table or sky grid is needed (see :func:`projected_area_fractions`).

The surface is that of a sphere of the segment's own shock radius; the finite thickness of the shell and the depth
ordering of segments whose radii differ are not captured. The mean path length :math:`\ell_k = V_k/A_k` that the
one-zone synchrotron engine needs is built from the volume fraction of the whole segment (both hemispheres) over this
projected area.
"""

from typing import NamedTuple

import numpy as np

__all__ = [
    "Segmentation",
    "compute_segmentation",
    "segment_volume_fractions",
    "projected_area_fractions",
    "segment_geometry",
]


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


def _visible_cap_area(theta: np.ndarray, inclination: float) -> np.ndarray:
    r"""
    Visible projected area of the polar cap :math:`[0, \theta]` of a unit sphere, as a fraction of :math:`R^2`.

    Parameters
    ----------
    theta : numpy.ndarray
        Colatitudes of the cap edge, in :math:`[0, \pi]`.
    inclination : float
        Angle between the symmetry axis and the line of sight, in :math:`[0, \pi/2]`.

    Notes
    -----
    The edge of the cap is a circle whose projection is the ellipse with semi-axes :math:`\sin\theta` (across the
    projected axis) and :math:`\sin\theta\cos i` (along it), centred at :math:`\cos\theta\sin i` along the
    projected axis. With :math:`\kappa = \cot\theta\cot i`, the circle is wholly visible for
    :math:`\kappa \ge 1` and wholly hidden for :math:`\kappa \le -1`. In between it crosses the limb at the sky
    height :math:`y_0 = \cos\theta/\sin i`, and the visible part of the cap is the segment of the unit disk above
    :math:`y_0` plus the part of the ellipse below it.
    """
    theta = np.asarray(theta, dtype=float)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_i, cos_i = np.sin(inclination), np.cos(inclination)

    if sin_i < 1e-12:  # axis along the line of sight: the near cap is seen face-on, the far cap is hidden
        return np.where(theta < 0.5 * np.pi, np.pi * sin_t**2, np.pi)

    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = cos_t * cos_i / (sin_t * sin_i)  # +-inf at the poles
    whole = kappa >= 1.0  # cap entirely on the near side: a flat ellipse
    hidden = kappa <= -1.0  # edge entirely on the far side: the cap covers the whole near hemisphere

    kappa_c = np.clip(np.nan_to_num(kappa), -1.0, 1.0)
    y0 = np.clip(cos_t / sin_i, -1.0, 1.0)
    phi_0 = np.arccos(kappa_c)
    unit_disk_segment = np.arccos(y0) - y0 * np.sqrt(1.0 - y0**2)
    ellipse_part = sin_t**2 * cos_i * (np.pi - phi_0 + kappa_c * np.sin(phi_0))
    partial = unit_disk_segment + ellipse_part
    return np.where(whole, np.pi * sin_t**2 * cos_i, np.where(hidden, np.pi, partial))


def projected_area_fractions(edges: np.ndarray, inclination: float) -> np.ndarray:
    r"""
    Sky-projected area fraction :math:`f_A = A_\perp/\pi R^2` of every segment of a spherical shell.

    A segment is the zone :math:`[\theta_a, \theta_b]` together with its mirror image about the equator, and its
    projected area is that of the part of the zone's surface that faces the observer,

    .. math::

        A_\perp = \int_{\rm zone}\max(\hat n\cdot\hat o,\,0)\,dS.

    Writing :math:`F(\theta)` for the visible projected area of the cap :math:`[0, \theta]` (see below), the segment's
    area is :math:`G(\theta_b) - G(\theta_a)` with :math:`G(\theta) = [F(\theta) - F(\pi - \theta)]/\pi` in units of
    :math:`\pi R^2`. :math:`F` is elementary. With :math:`\kappa = \cot\theta\cot i`:

    .. math::

        F/R^2 = \begin{cases}
            \pi\sin^2\theta\cos i & \kappa \ge 1, \\
            \pi & \kappa \le -1, \\
            \arccos y_0 - y_0\sqrt{1 - y_0^2}
                + \sin^2\theta\cos i\,\left(\pi - \arccos\kappa + \kappa\sqrt{1 - \kappa^2}\right) & {\rm otherwise,}
        \end{cases}

    where :math:`y_0 = \cos\theta/\sin i`. The areas sum to 1 for any inclination.

    Parameters
    ----------
    edges : numpy.ndarray
        Colatitudes of the segment edges from the pole to the equator, as in :class:`Segmentation`.
    inclination : float
        Angle between the symmetry axis and the line of sight, in radians. The areas are symmetric under
        :math:`i \to \pi - i`, and the inclination is folded into :math:`[0, \pi/2]`.

    Returns
    -------
    numpy.ndarray
        The area fraction of each segment, shape ``(N,)``, ordered from the pole to the equator.
    """
    edges = np.asarray(edges, dtype=float)
    inclination = _fold_inclination(inclination)
    cap_near = _visible_cap_area(edges, inclination)
    cap_far = _visible_cap_area(np.pi - edges, inclination)
    return np.diff(cap_near - cap_far) / np.pi


def segment_geometry(edges: np.ndarray, filling_factor: float, inclination: float) -> tuple[dict, ...]:
    r"""
    Volume fraction, area fraction and mean path length of every segment, from the pole to the equator.

    Parameters
    ----------
    edges : numpy.ndarray
        Colatitudes of the segment edges from the pole to the equator, as in :class:`Segmentation`.
    filling_factor : float
        Shell volume filling factor :math:`f`. It is clipped just inside :math:`(0, 1)`.
    inclination : float
        Angle between the symmetry axis and the line of sight, in radians.

    Returns
    -------
    tuple of dict
        One dict per segment with the float entries ``f_V`` (:func:`segment_volume_fractions`), ``f_A``
        (:func:`projected_area_fractions`) and ``ell_over_R = (4/3) f_V / f_A``, the mean path length in units of the
        segment's shock radius (0 for a segment with no projected area).
    """
    f = float(np.clip(filling_factor, 1e-6, 1.0 - 1e-6))
    f_V = segment_volume_fractions(edges, f)
    f_A = projected_area_fractions(edges, inclination)
    return tuple(
        {"f_V": float(v), "f_A": float(a), "ell_over_R": (4.0 / 3.0) * float(v) / float(a) if a > 0.0 else 0.0}
        for v, a in zip(f_V, f_A)
    )


def _fold_inclination(inclination: float) -> float:
    r"""Fold an inclination into :math:`[0, \pi/2]`, using the symmetry under :math:`i \to \pi - i`."""
    if not np.isfinite(inclination):
        raise ValueError("`inclination` must be finite.")
    inc = float(np.mod(inclination, np.pi))
    return np.pi - inc if inc > 0.5 * np.pi else inc
