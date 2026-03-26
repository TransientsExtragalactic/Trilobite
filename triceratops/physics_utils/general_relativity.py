r"""
General relativistic utility functions for Triceratops models.

Provides radius, spin, angular-momentum, and orbital precession quantities
for Schwarzschild and Kerr black holes.

Two-level API
-------------
Private ``_log_compute_*`` / ``_precession_per_orbit_log`` functions take
bare log-CGS floats and return ``ln(quantity)`` (or a dimensionless float
for spin).  Public ``compute_*`` wrappers accept
:class:`~astropy.units.Quantity` inputs and return unit-bearing outputs.

+---------------------------------------+-------------------------------------------+
| Private (log-CGS)                     | Public (:class:`~astropy.units.Quantity`) |
+=======================================+===========================================+
| :func:`_log_compute_r_g`              |     :func:`compute_gravitational_radius`  |
+---------------------------------------+-------------------------------------------+
| :func:`_log_compute_r_schwarzschild`  |     :func:`compute_schwarzschild_radius`  |
+---------------------------------------+-------------------------------------------+
| :func:`_log_compute_isco`             |     :func:`compute_ISCO`                  |
+---------------------------------------+-------------------------------------------+
| :func:`_log_compute_kerr_horizon_     |     :func:`compute_kerr_horizon_radius`   |
| radius`                               |                                           |
+---------------------------------------+-------------------------------------------+
| :func:`_log_compute_kerr_spin`        |     :func:`compute_kerr_spin`             |
+---------------------------------------+-------------------------------------------+
| :func:`_log_compute_kerr_angular_     |     :func:`compute_kerr_angular_momentum` |
| momentum`                             |                                           |
+---------------------------------------+-------------------------------------------+
| :func:`_precession_per_orbit`         |     :func:`compute_precession_per_orbit`  |
+---------------------------------------+-------------------------------------------+

Notes
-----
All log-space functions operate in CGS units (:math:`\text{cm}`,
:math:`\text{g}`, :math:`\text{s}`).  The Kerr ISCO formula follows
Bardeen, Press & Teukolsky (1972).

See Also
--------
:mod:`.constants` : CGS constants consumed internally.
"""

from typing import Optional

import numpy as np
from astropy import units as u

from triceratops._typing import _UnitBearingArrayLike
from triceratops.physics_utils.constants import _log_c_cgs, _log_G_cgs
from triceratops.utils.misc_utils import ensure_in_units

# ================================================================= #
# Private log-space helpers                                         #
# ================================================================= #


def _log_compute_r_g(log_M: float) -> float:
    r"""
    Compute :math:`\ln r_g` in CGS units.

    The gravitational radius (half the Schwarzschild radius) is

    .. math::

        r_g = \frac{G M}{c^2}
        \implies
        \ln r_g = \ln G + \ln M - 2\,\ln c.

    Parameters
    ----------
    log_M : float
        :math:`\ln M` [:math:`\ln(\text{g})`].

    Returns
    -------
    float
        :math:`\ln r_g` [:math:`\ln(\text{cm})`].
    """
    return _log_G_cgs + log_M - 2.0 * _log_c_cgs


def _log_compute_r_schwarzschild(log_M: float) -> float:
    r"""
    Compute :math:`\ln r_s` in CGS units.

    The Schwarzschild radius is twice the gravitational radius:

    .. math::

        r_s = \frac{2 G M}{c^2}
        \implies
        \ln r_s = \ln 2 + \ln G + \ln M - 2\,\ln c.

    Parameters
    ----------
    log_M : float
        :math:`\ln M` [:math:`\ln(\text{g})`].

    Returns
    -------
    float
        :math:`\ln r_s` [:math:`\ln(\text{cm})`].
    """
    return np.log(2.0) + _log_compute_r_g(log_M)


def _log_compute_isco(log_M: float, spin: float, prograde: bool = True) -> float:
    r"""
    Compute :math:`\ln r_{\rm ISCO}` in CGS units (Bardeen et al. 1972).

    .. math::

        r_{\rm ISCO} = r_g \left[ 3 + Z_2 \mp \sqrt{(3 - Z_1)(3 + Z_1 + 2 Z_2)} \right]

    The minus sign is for prograde orbits, the plus sign for retrograde.

    Parameters
    ----------
    log_M : float
        :math:`\ln M` [:math:`\ln(\text{g})`].
    spin : float
        Dimensionless Kerr spin parameter :math:`a_\star \in [-1, 1]`.
    prograde : bool, optional
        If ``True`` (default), prograde orbit; otherwise retrograde.

    Returns
    -------
    float
        :math:`\ln r_{\rm ISCO}` [:math:`\ln(\text{cm})`].
    """
    log_r_g = _log_compute_r_g(log_M)

    Z1 = 1.0 + (1.0 - spin**2) ** (1.0 / 3.0) * ((1.0 + spin) ** (1.0 / 3.0) + (1.0 - spin) ** (1.0 / 3.0))
    Z2 = np.sqrt(3.0 * spin**2 + Z1**2)

    if prograde:
        r_hat = 3.0 + Z2 - np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
    else:
        r_hat = 3.0 + Z2 + np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))

    return log_r_g + np.log(r_hat)


def _log_compute_kerr_horizon_radius(log_M: float, spin: float) -> float:
    r"""
    Compute :math:`\ln r_+` in CGS units.

    The outer Kerr event horizon is

    .. math::

        r_+ = r_g \left(1 + \sqrt{1 - a_\star^2}\right)
        \implies
        \ln r_+ = \ln r_g + \ln\!\left(1 + \sqrt{1 - a_\star^2}\right).

    At :math:`a_\star = 0` this reduces to :math:`r_+ = 2 r_g = r_s`; at
    :math:`|a_\star| = 1` the horizon shrinks to :math:`r_+ = r_g`.

    Parameters
    ----------
    log_M : float
        :math:`\ln M` [:math:`\ln(\text{g})`].
    spin : float
        Dimensionless spin parameter :math:`a_\star \in [-1, 1]`.

    Returns
    -------
    float
        :math:`\ln r_+` [:math:`\ln(\text{cm})`].
    """
    log_r_g = _log_compute_r_g(log_M)
    return log_r_g + np.log(1.0 + np.sqrt(1.0 - spin**2))


def _log_compute_kerr_spin(log_M: float, log_J: float) -> float:
    r"""
    Compute :math:`\ln|a_\star|` from mass and angular momentum.

    .. math::

        a_\star = \frac{c\,J}{G\,M^2}
        \implies
        \ln|a_\star| = \ln c + \ln J - \ln G - 2\,\ln M.

    Parameters
    ----------
    log_M : float
        :math:`\ln M` [:math:`\ln(\text{g})`].
    log_J : float
        :math:`\ln J` [:math:`\ln(\text{g cm}^2\,\text{s}^{-1})`].

    Returns
    -------
    float
        :math:`\ln|a_\star|` (dimensionless).
    """
    return _log_c_cgs + log_J - _log_G_cgs - 2.0 * log_M


def _log_compute_kerr_angular_momentum(log_M: float, spin: float) -> float:
    r"""
    Compute :math:`\ln J` in CGS units from mass and dimensionless spin.

    .. math::

        J = a_\star \frac{G\,M^2}{c}
        \implies
        \ln J = \ln|a_\star| + \ln G + 2\,\ln M - \ln c.

    Parameters
    ----------
    log_M : float
        :math:`\ln M` [:math:`\ln(\text{g})`].
    spin : float
        Dimensionless Kerr spin parameter :math:`a_\star` (may be negative
        for retrograde).

    Returns
    -------
    float
        :math:`\ln J` [:math:`\ln(\text{g cm}^2\,\text{s}^{-1})`].
    """
    return np.log(abs(spin)) + _log_G_cgs + 2.0 * log_M - _log_c_cgs


def _precession_per_orbit(log_M: float, log_r: float) -> float:
    r"""
    Compute :math:`\ln \Delta\phi` — periapsis precession per orbit in CGS.

    For a nearly circular orbit in the Schwarzschild metric the apsidal
    angle advance per orbit is

    .. math::

        \Delta\phi = \frac{6\pi\,G\,M}{r\,c^2} = 6\pi\,\frac{r_g}{r}
        \implies
        \ln\Delta\phi = \ln(6\pi) + \ln r_g - \ln r.

    Parameters
    ----------
    log_M : float
        :math:`\ln M` [:math:`\ln(\text{g})`].
    log_r : float
        :math:`\ln r` [:math:`\ln(\text{cm})`], orbital radius.

    Returns
    -------
    float
        :math:`\ln\Delta\phi` [:math:`\ln(\text{rad per orbit})`].
    """
    log_r_g = _log_compute_r_g(log_M)
    return np.log(6.0 * np.pi) + log_r_g - log_r


# ================================================================= #
# Public unit-bearing functions                                     #
# ================================================================= #


def compute_gravitational_radius(mass: "_UnitBearingArrayLike") -> u.Quantity:
    r"""
    Compute the gravitational radius :math:`r_g = GM/c^2`.

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        Black hole mass.

    Returns
    -------
    `~astropy.units.Quantity`
        Gravitational radius [:math:`\text{cm}`].

    Examples
    --------
    .. code-block:: python

        from astropy import units as u
        from triceratops.physics_utils.general_relativity import (
            compute_gravitational_radius,
        )

        compute_gravitational_radius(10 * u.Msun)
    """
    log_M = np.log(ensure_in_units(mass, u.g))
    return np.exp(_log_compute_r_g(log_M)) * u.cm


def compute_schwarzschild_radius(mass: "_UnitBearingArrayLike") -> u.Quantity:
    r"""
    Compute the Schwarzschild radius :math:`r_s = 2GM/c^2`.

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        Black hole mass.

    Returns
    -------
    `~astropy.units.Quantity`
        Schwarzschild radius [:math:`\text{cm}`].

    Examples
    --------
    .. code-block:: python

        from astropy import units as u
        from triceratops.physics_utils.general_relativity import (
            compute_schwarzschild_radius,
        )

        compute_schwarzschild_radius(10 * u.Msun)
    """
    log_M = np.log(ensure_in_units(mass, u.g))
    return np.exp(_log_compute_r_schwarzschild(log_M)) * u.cm


def compute_ISCO(
    mass: "_UnitBearingArrayLike",
    spin: Optional[float] = None,
    J: Optional["_UnitBearingArrayLike"] = None,
    prograde: bool = True,
) -> u.Quantity:
    r"""
    Compute the innermost stable circular orbit (ISCO) radius.

    Uses the Kerr metric formula (Bardeen, Press & Teukolsky 1972):

    .. math::

        r_{\rm ISCO} = r_g \left[ 3 + Z_2 \mp \sqrt{(3 - Z_1)(3 + Z_1 + 2Z_2)} \right]

    where :math:`r_g = GM/c^2` and

    .. math::

        Z_1 &= 1 + (1 - a_\star^2)^{1/3}
              \left[ (1 + a_\star)^{1/3} + (1 - a_\star)^{1/3} \right] \\
        Z_2 &= \sqrt{3 a_\star^2 + Z_1^2}.

    The minus (plus) sign is for prograde (retrograde) orbits.

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        Black hole mass.
    spin : float, optional
        Dimensionless spin parameter :math:`a_\star \in [-1, 1]`.
        Defaults to ``0`` (Schwarzschild) if neither ``spin`` nor ``J``
        is provided.
    J : `~astropy.units.Quantity`, optional
        Black hole angular momentum; used to compute ``spin`` when
        ``spin`` is not given.
    prograde : bool, optional
        ``True`` (default) for prograde orbits; ``False`` for retrograde.

    Returns
    -------
    `~astropy.units.Quantity`
        ISCO radius [:math:`\text{cm}`].

    Raises
    ------
    ValueError
        If both ``spin`` and ``J`` are supplied, or if the spin lies
        outside :math:`[-1, 1]`.

    Notes
    -----
    Special cases:

    * :math:`a_\star = 0` (Schwarzschild): :math:`r_{\rm ISCO} = 6\,r_g`.
    * :math:`a_\star = 1`, prograde: :math:`r_{\rm ISCO} = r_g`.
    * :math:`a_\star = 1`, retrograde: :math:`r_{\rm ISCO} = 9\,r_g`.

    Examples
    --------
    .. code-block:: python

        from astropy import units as u
        from triceratops.physics_utils.general_relativity import compute_ISCO

        compute_ISCO(10 * u.Msun)                    # Schwarzschild
        compute_ISCO(10 * u.Msun, spin=0.9)          # prograde Kerr
        compute_ISCO(10 * u.Msun, spin=0.9, prograde=False)  # retrograde
    """
    if spin is not None and J is not None:
        raise ValueError("Provide either 'spin' or 'J', not both.")

    M_cgs = ensure_in_units(mass, u.g)
    log_M = np.log(M_cgs)

    if spin is None:
        if J is not None:
            log_J = np.log(ensure_in_units(J, u.g * u.cm**2 / u.s))
            spin = float(np.exp(_log_compute_kerr_spin(log_M, log_J)))
        else:
            spin = 0.0

    spin = float(spin)
    if not (-1.0 <= spin <= 1.0):
        raise ValueError(f"Spin must be in [-1, 1], got {spin}.")

    return np.exp(_log_compute_isco(log_M, spin, prograde)) * u.cm


def compute_kerr_horizon_radius(
    mass: "_UnitBearingArrayLike",
    spin: Optional[float] = None,
    J: Optional["_UnitBearingArrayLike"] = None,
) -> u.Quantity:
    r"""
    Compute the outer Kerr event horizon radius :math:`r_+ = r_g(1 + \sqrt{1-a_\star^2})`.

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        Black hole mass.
    spin : float, optional
        Dimensionless spin parameter :math:`a_\star \in [-1, 1]`.
        Defaults to ``0`` (Schwarzschild) if neither ``spin`` nor ``J``
        is provided.
    J : `~astropy.units.Quantity`, optional
        Black hole angular momentum; used to derive ``spin`` when not given.

    Returns
    -------
    `~astropy.units.Quantity`
        Outer horizon radius [:math:`\text{cm}`].

    Raises
    ------
    ValueError
        If both ``spin`` and ``J`` are supplied, or if the spin lies
        outside :math:`[-1, 1]`.

    Examples
    --------
    .. code-block:: python

        from astropy import units as u
        from triceratops.physics_utils.general_relativity import (
            compute_kerr_horizon_radius,
        )

        compute_kerr_horizon_radius(10 * u.Msun, spin=0.9)
    """
    if spin is not None and J is not None:
        raise ValueError("Provide either 'spin' or 'J', not both.")

    M_cgs = ensure_in_units(mass, u.g)
    log_M = np.log(M_cgs)

    if spin is None:
        if J is not None:
            log_J = np.log(ensure_in_units(J, u.g * u.cm**2 / u.s))
            spin = float(np.exp(_log_compute_kerr_spin(log_M, log_J)))
        else:
            spin = 0.0

    spin = float(spin)
    if not (-1.0 <= spin <= 1.0):
        raise ValueError(f"Spin must be in [-1, 1], got {spin}.")

    return np.exp(_log_compute_kerr_horizon_radius(log_M, spin)) * u.cm


def compute_kerr_spin(
    mass: "_UnitBearingArrayLike",
    J: "_UnitBearingArrayLike",
) -> float:
    r"""
    Compute the dimensionless Kerr spin parameter :math:`a_\star = cJ/(GM^2)`.

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        Black hole mass.
    J : `~astropy.units.Quantity`
        Black hole angular momentum.

    Returns
    -------
    float
        Dimensionless spin :math:`a_\star` (positive for prograde).

    Examples
    --------
    .. code-block:: python

        from astropy import constants as const, units as u
        from triceratops.physics_utils.general_relativity import (
            compute_kerr_spin,
        )

        M = 10 * u.Msun
        compute_kerr_spin(
            M, J=0.9 * const.G * M**2 / const.c
        )
    """
    log_M = np.log(ensure_in_units(mass, u.g))
    log_J = np.log(ensure_in_units(J, u.g * u.cm**2 / u.s))
    return float(np.exp(_log_compute_kerr_spin(log_M, log_J)))


def compute_kerr_angular_momentum(
    mass: "_UnitBearingArrayLike",
    spin: float,
) -> u.Quantity:
    r"""
    Compute the black hole angular momentum :math:`J = a_\star GM^2/c`.

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        Black hole mass.
    spin : float
        Dimensionless spin parameter :math:`a_\star \in [-1, 1]`.

    Returns
    -------
    `~astropy.units.Quantity`
        Angular momentum [:math:`\text{g cm}^2\,\text{s}^{-1}`].

    Raises
    ------
    ValueError
        If spin is zero (log is undefined) or outside :math:`[-1, 1]`.

    Examples
    --------
    .. code-block:: python

        from astropy import units as u
        from triceratops.physics_utils.general_relativity import (
            compute_kerr_angular_momentum,
        )

        compute_kerr_angular_momentum(
            10 * u.Msun, spin=0.9
        )
    """
    if spin == 0.0:
        raise ValueError("spin=0 corresponds to J=0; angular momentum is exactly zero.")
    if not (-1.0 <= spin <= 1.0):
        raise ValueError(f"Spin must be in [-1, 1], got {spin}.")

    log_M = np.log(ensure_in_units(mass, u.g))
    return np.exp(_log_compute_kerr_angular_momentum(log_M, spin)) * u.g * u.cm**2 / u.s


def compute_precession_per_orbit(
    mass: "_UnitBearingArrayLike",
    r: "_UnitBearingArrayLike",
) -> u.Quantity:
    r"""
    Compute the Schwarzschild periapsis precession per orbit.

    For a nearly circular orbit at radius :math:`r` around a mass :math:`M`:

    .. math::

        \Delta\phi = \frac{6\pi G M}{r c^2} = 6\pi\,\frac{r_g}{r}.

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        Central mass.
    r : `~astropy.units.Quantity`
        Orbital radius.

    Returns
    -------
    `~astropy.units.Quantity`
        Precession angle per orbit [:math:`\text{rad}`].

    Examples
    --------
    .. code-block:: python

        from astropy import units as u
        from triceratops.physics_utils.general_relativity import (
            compute_precession_per_orbit,
        )

        compute_precession_per_orbit(
            1.4 * u.Msun, 1e6 * u.cm
        )
    """
    log_M = np.log(ensure_in_units(mass, u.g))
    log_r = np.log(ensure_in_units(r, u.cm))
    return np.exp(_precession_per_orbit(log_M, log_r)) * u.rad
