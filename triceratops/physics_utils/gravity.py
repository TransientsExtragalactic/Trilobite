r"""
Newtonian gravitational physics utilities for Triceratops models.

Provides characteristic radii for tidal disruption, binary mass transfer,
accretion, and gravitational capture commonly encountered in astrophysical
transient modelling.

Notes
-----
All log-space functions operate in CGS units (:math:`\text{cm}`,
:math:`\text{g}`, :math:`\text{s}`).  The Roche-lobe formula follows
Eggleton (1983); the Hill-sphere formula assumes a circular, restricted
three-body configuration.

References
----------
Eggleton (1983), ApJ, 268, 368.
"""

import numpy as np
from astropy import units as u

from triceratops._typing import _UnitBearingArrayLike
from triceratops.physics_utils.constants import _log_G_cgs
from triceratops.utils.misc_utils import ensure_in_units

# ================================================================= #
# Private log-space helpers                                         #
# ================================================================= #


def _compute_tidal_radius(log_R_star: float, log_M_primary: float, log_M_secondary: float) -> float:
    r"""
    Compute :math:`\ln R_T` in CGS units.

    The tidal disruption radius is the orbital distance at which the primary's
    self-gravity is overcome by tidal forces from a more massive body:

    .. math::

        R_T = R_\star \left(\frac{M_{\rm primary}}{M_{\rm secondary}}\right)^{1/3}
        \implies
        \ln R_T = \ln R_\star + \tfrac{1}{3}(\ln M_{\rm primary} - \ln M_{\rm secondary}).

    Parameters
    ----------
    log_R_star : float
        :math:`\ln R_\star` [:math:`\ln(\text{cm})`], stellar radius.
    log_M_primary : float
        :math:`\ln M_{\rm primary}` [:math:`\ln(\text{g})`], mass of the
        disrupting body (e.g. black hole).
    log_M_secondary : float
        :math:`\ln M_{\rm secondary}` [:math:`\ln(\text{g})`], mass of the
        disrupted body (e.g. star).

    Returns
    -------
    float
        :math:`\ln R_T` [:math:`\ln(\text{cm})`].
    """
    return log_R_star + (log_M_primary - log_M_secondary) / 3.0


def _compute_hill_radius(log_a: float, log_m_secondary: float, log_M_primary: float) -> float:
    r"""
    Compute :math:`\ln R_H` in CGS units.

    The Hill (Roche) sphere radius sets the region within which a secondary
    body dominates the gravitational attraction of a test particle:

    .. math::

        R_H = a \left(\frac{m_{\rm secondary}}{3 M_{\rm primary}}\right)^{1/3}
        \implies
        \ln R_H = \ln a + \tfrac{1}{3}\left(\ln m_{\rm secondary} - \ln M_{\rm primary} - \ln 3\right).

    Parameters
    ----------
    log_a : float
        :math:`\ln a` [:math:`\ln(\text{cm})`], orbital semi-major axis.
    log_m_secondary : float
        :math:`\ln m_{\rm secondary}` [:math:`\ln(\text{g})`], mass of the
        secondary (the body whose sphere of influence is computed).
    log_M_primary : float
        :math:`\ln M_{\rm primary}` [:math:`\ln(\text{g})`], mass of the
        primary (the dominant central body).

    Returns
    -------
    float
        :math:`\ln R_H` [:math:`\ln(\text{cm})`].
    """
    return log_a + (log_m_secondary - log_M_primary - np.log(3.0)) / 3.0


def _compute_roche_lobe_radius(log_m_donor: float, log_m_accretor: float, log_a: float) -> float:
    r"""
    Compute :math:`\ln r_L` in CGS units via the Eggleton (1983) approximation.

    The Roche lobe radius of the donor star is

    .. math::

        \frac{r_L}{a} = \frac{0.49\,q^{2/3}}{0.6\,q^{2/3} + \ln(1 + q^{1/3})},
        \quad q = \frac{M_{\rm donor}}{M_{\rm accretor}},

    accurate to better than 1 % for all mass ratios :math:`0 < q < \infty`.

    Parameters
    ----------
    log_m_donor : float
        :math:`\ln M_{\rm donor}` [:math:`\ln(\text{g})`], donor mass.
    log_m_accretor : float
        :math:`\ln M_{\rm accretor}` [:math:`\ln(\text{g})`], accretor mass.
    log_a : float
        :math:`\ln a` [:math:`\ln(\text{cm})`], orbital semi-major axis.

    Returns
    -------
    float
        :math:`\ln r_L` [:math:`\ln(\text{cm})`].
    """
    q = np.exp(log_m_donor - log_m_accretor)
    q_13 = q ** (1.0 / 3.0)
    q_23 = q ** (2.0 / 3.0)
    roche_factor = 0.49 * q_23 / (0.6 * q_23 + np.log(1.0 + q_13))
    return log_a + np.log(roche_factor)


def _compute_bondi_radius(log_M: float, log_c_s: float) -> float:
    r"""
    Compute :math:`\ln r_B` in CGS units.

    The Bondi (gravitational capture) radius is the distance at which the
    escape speed equals the sound speed of the ambient medium:

    .. math::

        r_B = \frac{G M}{c_s^2}
        \implies
        \ln r_B = \ln G + \ln M - 2\,\ln c_s.

    Parameters
    ----------
    log_M : float
        :math:`\ln M` [:math:`\ln(\text{g})`], accreting mass.
    log_c_s : float
        :math:`\ln c_s` [:math:`\ln(\text{cm s}^{-1})`], ambient sound speed.

    Returns
    -------
    float
        :math:`\ln r_B` [:math:`\ln(\text{cm})`].
    """
    return _log_G_cgs + log_M - 2.0 * log_c_s


# ================================================================= #
# Public unit-bearing functions                                     #
# ================================================================= #


def compute_tidal_radius(
    R_star: "_UnitBearingArrayLike",
    M_primary: "_UnitBearingArrayLike",
    M_secondary: "_UnitBearingArrayLike",
) -> u.Quantity:
    r"""
    Compute the tidal disruption radius :math:`R_T`.

    The tidal radius is the orbital separation at which the tidal force of the
    primary on the secondary equals the secondary's self-gravity:

    .. math::

        R_T = R_\star \left(\frac{M_{\rm primary}}{M_{\rm secondary}}\right)^{1/3}.

    For a tidal disruption event (TDE) this gives the periapsis inside which a
    star of radius :math:`R_\star` and mass :math:`M_\star` is disrupted by a
    black hole of mass :math:`M_{\rm BH}`.

    Parameters
    ----------
    R_star : `~astropy.units.Quantity`
        Radius of the secondary (disrupted) body.
    M_primary : `~astropy.units.Quantity`
        Mass of the primary (disrupting) body, e.g. a black hole.
    M_secondary : `~astropy.units.Quantity`
        Mass of the secondary (disrupted) body, e.g. a star.

    Returns
    -------
    `~astropy.units.Quantity`
        Tidal disruption radius [:math:`\text{cm}`].

    Examples
    --------
    .. code-block:: python

        from astropy import units as u
        from triceratops.physics_utils.gravity import (
            compute_tidal_radius,
        )

        # Solar-type star disrupted by a 10^6 Msun SMBH
        R_T = compute_tidal_radius(
            R_star=1.0 * u.Rsun,
            M_primary=1e6 * u.Msun,
            M_secondary=1.0 * u.Msun,
        )
    """
    log_R_star = np.log(ensure_in_units(R_star, u.cm))
    log_M_primary = np.log(ensure_in_units(M_primary, u.g))
    log_M_secondary = np.log(ensure_in_units(M_secondary, u.g))
    return np.exp(_compute_tidal_radius(log_R_star, log_M_primary, log_M_secondary)) * u.cm


def compute_hill_radius(
    a: "_UnitBearingArrayLike",
    m_secondary: "_UnitBearingArrayLike",
    M_primary: "_UnitBearingArrayLike",
) -> u.Quantity:
    r"""
    Compute the Hill sphere radius :math:`R_H`.

    The Hill sphere is the region around a secondary body within which it
    dominates the gravitational attraction over the primary.  For a circular
    orbit of semi-major axis :math:`a`:

    .. math::

        R_H = a \left(\frac{m_{\rm secondary}}{3 M_{\rm primary}}\right)^{1/3}.

    This radius sets the size of a planet's sphere of gravitational influence
    relative to its host star, or the zone within which a satellite can
    maintain a stable orbit around a planet.

    Parameters
    ----------
    a : `~astropy.units.Quantity`
        Orbital semi-major axis of the secondary around the primary.
    m_secondary : `~astropy.units.Quantity`
        Mass of the secondary (the body whose sphere of influence is computed).
    M_primary : `~astropy.units.Quantity`
        Mass of the primary (the dominant central body).

    Returns
    -------
    `~astropy.units.Quantity`
        Hill sphere radius [:math:`\text{cm}`].

    Examples
    --------
    .. code-block:: python

        from astropy import units as u
        from triceratops.physics_utils.gravity import (
            compute_hill_radius,
        )

        # Earth's Hill sphere at 1 AU from the Sun
        R_H = compute_hill_radius(
            a=1.0 * u.au,
            m_secondary=1.0 * u.Mearth,
            M_primary=1.0 * u.Msun,
        )
    """
    log_a = np.log(ensure_in_units(a, u.cm))
    log_m_secondary = np.log(ensure_in_units(m_secondary, u.g))
    log_M_primary = np.log(ensure_in_units(M_primary, u.g))
    return np.exp(_compute_hill_radius(log_a, log_m_secondary, log_M_primary)) * u.cm


def compute_roche_lobe_radius(
    m_donor: "_UnitBearingArrayLike",
    m_accretor: "_UnitBearingArrayLike",
    a: "_UnitBearingArrayLike",
) -> u.Quantity:
    r"""
    Compute the donor Roche lobe radius via the Eggleton (1983) formula.

    The Eggleton approximation gives the Roche lobe radius of the donor star
    as a fraction of the binary separation:

    .. math::

        \frac{r_L}{a} = \frac{0.49\,q^{2/3}}{0.6\,q^{2/3} + \ln(1 + q^{1/3})},
        \quad q = \frac{M_{\rm donor}}{M_{\rm accretor}}.

    This fit is accurate to better than 1 % for all mass ratios
    :math:`0 < q < \infty`.

    Parameters
    ----------
    m_donor : `~astropy.units.Quantity`
        Mass of the Roche-lobe-filling star (the donor).
    m_accretor : `~astropy.units.Quantity`
        Mass of the accreting companion.
    a : `~astropy.units.Quantity`
        Orbital semi-major axis.

    Returns
    -------
    `~astropy.units.Quantity`
        Roche lobe radius of the donor [:math:`\text{cm}`].

    Notes
    -----
    Mass transfer begins when the donor's stellar radius equals :math:`r_L`.
    The formula is symmetric in the sense that swapping donor and accretor
    (and updating :math:`q` accordingly) gives the Roche lobe of the other star.

    References
    ----------
    Eggleton (1983), ApJ, 268, 368.

    Examples
    --------
    .. code-block:: python

        from astropy import units as u
        from triceratops.physics_utils.gravity import (
            compute_roche_lobe_radius,
        )

        # Equal-mass binary at 10 Rsun separation
        r_L = compute_roche_lobe_radius(
            m_donor=1.0 * u.Msun,
            m_accretor=1.4 * u.Msun,
            a=10.0 * u.Rsun,
        )
    """
    log_m_donor = np.log(ensure_in_units(m_donor, u.g))
    log_m_accretor = np.log(ensure_in_units(m_accretor, u.g))
    log_a = np.log(ensure_in_units(a, u.cm))
    return np.exp(_compute_roche_lobe_radius(log_m_donor, log_m_accretor, log_a)) * u.cm


def compute_bondi_radius(
    M: "_UnitBearingArrayLike",
    c_s: "_UnitBearingArrayLike",
) -> u.Quantity:
    r"""
    Compute the Bondi accretion radius :math:`r_B = GM / c_s^2`.

    The Bondi radius is the characteristic length scale within which an
    accreting body can gravitationally capture material from the ambient
    medium.  At :math:`r \lesssim r_B` the escape speed exceeds the
    sound speed and material falls inward; at :math:`r \gg r_B` the
    ambient pressure resists capture.

    .. math::

        r_B = \frac{G M}{c_s^2}.

    Parameters
    ----------
    M : `~astropy.units.Quantity`
        Mass of the accreting body.
    c_s : `~astropy.units.Quantity`
        Sound speed of the ambient medium.

    Returns
    -------
    `~astropy.units.Quantity`
        Bondi accretion radius [:math:`\text{cm}`].

    Notes
    -----
    The Bondi accretion rate follows as
    :math:`\dot{M}_B \approx \pi r_B^2 \rho c_s`, where :math:`\rho` is the
    ambient density.  For a hot medium (large :math:`c_s`) the Bondi radius
    shrinks and accretion becomes inefficient.

    Examples
    --------
    .. code-block:: python

        from astropy import units as u
        from triceratops.physics_utils.gravity import (
            compute_bondi_radius,
        )

        # Neutron star accreting from a warm ISM
        r_B = compute_bondi_radius(
            M=1.4 * u.Msun,
            c_s=10.0 * u.km / u.s,
        )
    """
    log_M = np.log(ensure_in_units(M, u.g))
    log_c_s = np.log(ensure_in_units(c_s, u.cm / u.s))
    return np.exp(_compute_bondi_radius(log_M, log_c_s)) * u.cm
