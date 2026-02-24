"""
Relativistic and Cosmological Corrections for Radiative Quantities.

This module provides functions to apply relativistic Doppler corrections and cosmological redshift corrections
to radiative quantities such as frequencies, flux densities, specific intensities, absorption coefficients,
emissivities, and solid angles. These corrections are essential for transforming spectral energy distributions
(SEDs) between the emitter rest frame and the observer frame in astrophysical contexts, particularly for
relativistic jets and cosmologically distant sources.
"""

import numpy as np
from astropy import units as u
from astropy.cosmology import FLRW, Planck18, z_at_value


# ============================================================= #
# Relativistic Corrections for Radiative Quantities             #
# ============================================================= #
# In many contexts, we construct SEDs in the rest frame of the emitting region. In simple cases
# where light-travel time corrections are not important, we can apply relativistic corrections
# post-hoc to transform the SED into the observer frame. The following functions apply these
# corrections to flux densities and frequencies.
def compute_doppler_factor(lorentz_factor: float, viewing_angle: float = 0.0):
    r"""
    Compute the doppler factor :math:`\delta(\Gamma, \theta)` for a given Lorentz factor and viewing angle.

    The Doppler factor provides the conversion between rest-frame frequency and observer frequency accounting
    for beaming and aberration effects. Formally, we consider a frame :math:`S` moving with velocity
    :math:`\beta c \hat{\bf z}` relative to the observer frame :math:`S'`. A photon emitted in the
    rest frame :math:`S` at an angle :math:`\theta` with respect to the direction of motion will be observed
    in frame :math:`S'` with frequency

    .. math::

        \nu' = \nu \Gamma(1+\beta \cos \theta) = \frac{\nu}{\Gamma(1-\beta \cos \theta'} = \nu \delta(\Gamma,
         \theta).

    This function computes this doppler factor for a given Lorentz factor :math:`\Gamma` and viewing
    angle :math:`\theta'`.

    Parameters
    ----------
    lorentz_factor: float
        The Lorentz factor :math:`\Gamma` of the emitting region.
    viewing_angle: float, optional
        The **observer frame** angle :math:`\theta'` between photon's direction of motion and the line of sight.
        Default is 0 (on-axis).

    Returns
    -------
    float
        The Doppler factor :math:`\delta(\Gamma, \theta)`.
    """
    beta = (1 - 1 / lorentz_factor**2) ** 0.5
    doppler_factor = 1 / (lorentz_factor * (1 - beta * np.cos(viewing_angle)))
    return doppler_factor


def compute_doppler_factor_rest_frame(
    lorentz_factor: float,
    rest_frame_angle: float = 0.0,
):
    r"""
    Compute the Doppler factor using the **emitter rest-frame angle**.

    This is the less commonly used form of the Doppler factor, appropriate
    when the photon emission angle is known in the *comoving (emitter) frame*.

    Consider an emitting region moving with velocity βc along +z relative
    to the observer (lab) frame. A photon emitted in the emitter rest frame
    at angle θ (measured relative to the direction of motion) will be observed
    in the lab frame with frequency

    .. math::

        \nu_{\rm obs}
        = \Gamma (1 + \beta \cos\theta) \, \nu'

    where:

    - :math:`\nu'` is the rest-frame frequency,
    - :math:`\theta` is the **rest-frame angle**,
    - :math:`\Gamma` is the Lorentz factor,
    - :math:`\beta = v/c`.

    The Doppler factor in this case is therefore

    .. math::

        \delta(\Gamma, \theta)
        = \Gamma (1 + \beta \cos\theta).

    .. note::

        This form is used only when the emission angle is known in the
        emitter rest frame. In most astrophysical jet applications,
        the observer-frame angle is known instead, and the appropriate
        expression is

        .. math::

            \delta = \frac{1}{\Gamma(1 - \beta \cos\theta')}.

    Parameters
    ----------
    lorentz_factor : float
        Lorentz factor :math:`\Gamma` of the emitting region.

    rest_frame_angle : float, optional
        Angle :math:`\theta` (radians) measured in the emitter rest frame,
        relative to the direction of motion. Default is 0 (forward emission).

    Returns
    -------
    float
        Doppler factor :math:`\delta`.
    """
    beta = (1.0 - 1.0 / lorentz_factor**2) ** 0.5
    return lorentz_factor * (1.0 + beta * np.cos(rest_frame_angle))


def transform_frequency_from_rest_to_observer_frame(
    rest_frame_frequency: float,
    lorentz_factor: float,
    viewing_angle: float = 0.0,
):
    """
    Transform a frequency from the emitter rest frame to the observer frame using the Doppler factor.

    Parameters
    ----------
    rest_frame_frequency : float
        The frequency in the emitter rest frame.
    lorentz_factor : float
        The Lorentz factor of the emitting region.
    viewing_angle : float, optional
        The observer-frame angle (radians) between the photon's direction of motion and the line of sight.
        Default is 0 (on-axis).

    Returns
    -------
    float
        The frequency in the observer frame.
    """
    doppler_factor = compute_doppler_factor(lorentz_factor, viewing_angle)
    return rest_frame_frequency * doppler_factor


def transform_frequency_from_observer_to_rest_frame(
    observer_frame_frequency: float,
    lorentz_factor: float,
    viewing_angle: float = 0.0,
):
    """
    Transform a frequency from the observer frame to the emitter rest frame using the Doppler factor.

    Parameters
    ----------
    observer_frame_frequency : float
        The frequency in the observer frame.
    lorentz_factor : float
        The Lorentz factor of the emitting region.
    viewing_angle : float, optional
        The observer-frame angle (radians) between the photon's direction of motion and the line of sight.
        Default is 0 (on-axis).

    Returns
    -------
    float
        The frequency in the emitter rest frame.
    """
    doppler_factor = compute_doppler_factor(lorentz_factor, viewing_angle)
    return observer_frame_frequency / doppler_factor


def transform_spec_intensity_from_rest_to_observer_frame(
    rest_frame_intensity: float,
    lorentz_factor: float,
    viewing_angle: float = 0.0,
):
    r"""
    Transform specific intensity from the emitter rest frame to the observer frame.

    The Lorentz invariant

    .. math::

        \frac{I_\nu}{\nu^3}

    implies

    .. math::

        I_{\nu,\mathrm{obs}} = \delta^3 I'_\nu.

    Parameters
    ----------
    rest_frame_intensity : float
        Specific intensity :math:`I'_\nu` in the emitter rest frame.
    lorentz_factor : float
        Lorentz factor :math:`\Gamma` of the emitting region.
    viewing_angle : float, optional
        Observer-frame angle :math:`\theta'` in radians.
        Default is 0 (on-axis).

    Returns
    -------
    float
        Observer-frame specific intensity :math:`I_{\nu,\mathrm{obs}}`.
    """
    delta = compute_doppler_factor(lorentz_factor, viewing_angle)
    return rest_frame_intensity * delta**3


def transform_spec_intensity_from_observer_to_rest_frame(
    observer_frame_intensity: float,
    lorentz_factor: float,
    viewing_angle: float = 0.0,
):
    r"""
    Transform specific intensity from the observer frame to the emitter rest frame.

    Using

    .. math::

        I_{\nu,\mathrm{obs}} = \delta^3 I'_\nu,

    we obtain

    .. math::

        I'_\nu = \frac{I_{\nu,\mathrm{obs}}}{\delta^3}.

    Parameters
    ----------
    observer_frame_intensity : float
        Observer-frame specific intensity :math:`I_{\nu,\mathrm{obs}}`.
    lorentz_factor : float
        Lorentz factor :math:`\Gamma`.
    viewing_angle : float, optional
        Observer-frame angle :math:`\theta'` in radians.

    Returns
    -------
    float
        Rest-frame specific intensity :math:`I'_\nu`.
    """
    delta = compute_doppler_factor(lorentz_factor, viewing_angle)
    return observer_frame_intensity / delta**3


def transform_flux_density_from_rest_to_observer_frame(
    rest_frame_flux_density: float,
    lorentz_factor: float,
    viewing_angle: float = 0.0,
):
    """
    Transform a flux density from the emitter rest frame to the observer frame using the Doppler factor.

    Parameters
    ----------
    rest_frame_flux_density : float
        The flux density in the emitter rest frame.
    lorentz_factor : float
        The Lorentz factor of the emitting region.
    viewing_angle : float, optional
        The observer-frame angle (radians) between the photon's direction of motion and the line of sight.
        Default is 0 (on-axis).

    Returns
    -------
    float
        The flux density in the observer frame.
    """
    doppler_factor = compute_doppler_factor(lorentz_factor, viewing_angle)
    return rest_frame_flux_density * doppler_factor**3


def transform_flux_density_from_observer_to_rest_frame(
    observer_frame_flux_density: float,
    lorentz_factor: float,
    viewing_angle: float = 0.0,
):
    """
    Transform a flux density from the observer frame to the emitter rest frame using the Doppler factor.

    Parameters
    ----------
    observer_frame_flux_density : float
        The flux density in the observer frame.
    lorentz_factor : float
        The Lorentz factor of the emitting region.
    viewing_angle : float, optional
        The observer-frame angle (radians) between the photon's direction of motion and the line of sight.
        Default is 0 (on-axis).

    Returns
    -------
    float
        The flux density in the emitter rest frame.
    """
    doppler_factor = compute_doppler_factor(lorentz_factor, viewing_angle)
    return observer_frame_flux_density / doppler_factor**3


def transform_absorption_coefficient_from_rest_to_observer_frame(
    rest_frame_alpha: float,
    lorentz_factor: float,
    viewing_angle: float = 0.0,
):
    r"""
    Transform absorption coefficient from the rest frame to the observer frame.

    The Lorentz invariant

    .. math::

        \alpha_\nu \nu

    implies

    .. math::

        \alpha_{\nu,\mathrm{obs}} = \frac{\alpha'_\nu}{\delta}.

    Parameters
    ----------
    rest_frame_alpha : float
        Rest-frame absorption coefficient :math:`\alpha'_\nu`.
    lorentz_factor : float
        Lorentz factor :math:`\Gamma`.
    viewing_angle : float, optional
        Observer-frame angle :math:`\theta'`.

    Returns
    -------
    float
        Observer-frame absorption coefficient :math:`\alpha_{\nu,\mathrm{obs}}`.
    """
    delta = compute_doppler_factor(lorentz_factor, viewing_angle)
    return rest_frame_alpha / delta


def transform_absorption_coefficient_from_observer_to_rest_frame(
    observer_frame_alpha: float,
    lorentz_factor: float,
    viewing_angle: float = 0.0,
):
    r"""
    Transform absorption coefficient from the observer frame to the rest frame.

    Using

    .. math::

        \alpha_{\nu,\mathrm{obs}} = \frac{\alpha'_\nu}{\delta},

    we obtain

    .. math::

        \alpha'_\nu = \delta \alpha_{\nu,\mathrm{obs}}.

    Parameters
    ----------
    observer_frame_alpha : float
        Observer-frame absorption coefficient :math:`\alpha_{\nu,\mathrm{obs}}`.
    lorentz_factor : float
        Lorentz factor :math:`\Gamma`.
    viewing_angle : float, optional
        Observer-frame angle :math:`\theta'`.

    Returns
    -------
    float
        Rest-frame absorption coefficient :math:`\alpha'_\nu`.
    """
    delta = compute_doppler_factor(lorentz_factor, viewing_angle)
    return observer_frame_alpha * delta


def transform_emissivity_from_rest_to_observer_frame(
    rest_frame_emissivity: float,
    lorentz_factor: float,
    viewing_angle: float = 0.0,
):
    r"""
    Transform emissivity from the rest frame to the observer frame.

    The Lorentz invariant

    .. math::

        \frac{j_\nu}{\nu^2}

    implies

    .. math::

        j_{\nu,\mathrm{obs}} = \delta^2 j'_\nu.

    Parameters
    ----------
    rest_frame_emissivity : float
        Rest-frame emissivity :math:`j'_\nu`.
    lorentz_factor : float
        Lorentz factor :math:`\Gamma`.
    viewing_angle : float, optional
        Observer-frame angle :math:`\theta'` in radians.

    Returns
    -------
    float
        Observer-frame emissivity :math:`j_{\nu,\mathrm{obs}}`.
    """
    delta = compute_doppler_factor(lorentz_factor, viewing_angle)
    return rest_frame_emissivity * delta**2


def transform_emissivity_from_observer_to_rest_frame(
    observer_frame_emissivity: float,
    lorentz_factor: float,
    viewing_angle: float = 0.0,
):
    r"""
    Transform emissivity from the observer frame to the rest frame.

    Using

    .. math::

        j_{\nu,\mathrm{obs}} = \delta^2 j'_\nu,

    we obtain

    .. math::

        j'_\nu = \frac{j_{\nu,\mathrm{obs}}}{\delta^2}.

    Parameters
    ----------
    observer_frame_emissivity : float
        Observer-frame emissivity :math:`j_{\nu,\mathrm{obs}}`.
    lorentz_factor : float
        Lorentz factor :math:`\Gamma`.
    viewing_angle : float, optional
        Observer-frame angle :math:`\theta'`.

    Returns
    -------
    float
        Rest-frame emissivity :math:`j'_\nu`.
    """
    delta = compute_doppler_factor(lorentz_factor, viewing_angle)
    return observer_frame_emissivity / delta**2


def transform_solid_angle_from_rest_to_observer_frame(
    rest_frame_solid_angle: float,
    lorentz_factor: float,
    viewing_angle: float = 0.0,
):
    r"""
    Transform differential solid angle from rest frame to observer frame.

    Relativistic aberration gives

    .. math::

        d\Omega_{\mathrm{obs}} = \delta^{-2} d\Omega'.

    Parameters
    ----------
    rest_frame_solid_angle : float
        Rest-frame differential solid angle :math:`d\Omega'`.
    lorentz_factor : float
        Lorentz factor :math:`\Gamma`.
    viewing_angle : float, optional
        Observer-frame angle :math:`\theta'`.

    Returns
    -------
    float
        Observer-frame solid angle :math:`d\Omega_{\mathrm{obs}}`.
    """
    delta = compute_doppler_factor(lorentz_factor, viewing_angle)
    return rest_frame_solid_angle / delta**2


def transform_solid_angle_from_observer_to_rest_frame(
    observer_frame_solid_angle: float,
    lorentz_factor: float,
    viewing_angle: float = 0.0,
):
    r"""
    Transform differential solid angle from observer frame to rest frame.

    Using

    .. math::

        d\Omega_{\mathrm{obs}} = \delta^{-2} d\Omega',

    we obtain

    .. math::

        d\Omega' = \delta^{2} d\Omega_{\mathrm{obs}}.

    Parameters
    ----------
    observer_frame_solid_angle : float
        Observer-frame differential solid angle :math:`d\Omega_{\mathrm{obs}}`.
    lorentz_factor : float
        Lorentz factor :math:`\Gamma`.
    viewing_angle : float, optional
        Observer-frame angle :math:`\theta'`.

    Returns
    -------
    float
        Rest-frame differential solid angle :math:`d\Omega'`.
    """
    delta = compute_doppler_factor(lorentz_factor, viewing_angle)
    return observer_frame_solid_angle * delta**2


# ============================================================= #
# Cosmological Corrections                                      #
# ============================================================= #
# In many contexts, we construct SEDs in the rest frame of the emitting region. In simple cases
# where light-travel time corrections are not important, we can apply relativistic corrections
# post-hoc to transform the SED into the observer frame. The following functions apply these
# corrections to flux densities and frequencies.
def redshift_correct_frequency_from_rest_to_observer_frame(
    rest_frame_frequency: float,
    redshift: float,
):
    r"""
    Transform frequency from emitter rest frame to observer frame via cosmological redshift.

    Cosmological expansion stretches photon wavelengths such that

    .. math::

        \nu_{\rm obs} = \frac{\nu_{\rm em}}{1 + z}.

    Parameters
    ----------
    rest_frame_frequency : float
        Emitted (rest-frame) frequency :math:`\nu_{\rm em}`.
    redshift : float
        Cosmological redshift :math:`z`.

    Returns
    -------
    float
        Observed frequency :math:`\nu_{\rm obs}`.
    """
    return rest_frame_frequency / (1.0 + redshift)


def redshift_correct_frequency_from_observer_to_rest_frame(
    observer_frame_frequency: float,
    redshift: float,
):
    r"""
    Transform frequency from observer frame to emitter rest frame.

    Using

    .. math::

        \nu_{\rm obs} = \frac{\nu_{\rm em}}{1 + z},

    we obtain

    .. math::

        \nu_{\rm em} = (1 + z) \nu_{\rm obs}.

    Parameters
    ----------
    observer_frame_frequency : float
        Observed frequency :math:`\nu_{\rm obs}`.
    redshift : float
        Cosmological redshift :math:`z`.

    Returns
    -------
    float
        Rest-frame frequency :math:`\nu_{\rm em}`.
    """
    return observer_frame_frequency * (1.0 + redshift)


def redshift_correct_spec_intensity_from_rest_to_observer_frame(
    rest_frame_intensity: float,
    redshift: float,
):
    r"""
    Transform specific intensity from rest frame to observer frame under cosmological redshift.

    The Lorentz invariant

    .. math::

        \frac{I_\nu}{\nu^3}

    implies

    .. math::

        I_{\nu,\rm obs}
        =
        \frac{I_{\nu,\rm em}}{(1+z)^3}.

    Parameters
    ----------
    rest_frame_intensity : float
        Rest-frame specific intensity :math:`I_{\nu,\rm em}`.
    redshift : float
        Cosmological redshift :math:`z`.

    Returns
    -------
    float
        Observer-frame specific intensity :math:`I_{\nu,\rm obs}`.
    """
    return rest_frame_intensity / (1.0 + redshift) ** 3


def redshift_correct_spec_intensity_from_observer_to_rest_frame(
    observer_frame_intensity: float,
    redshift: float,
):
    r"""
    Transform specific intensity from observer frame to rest frame.

    .. math::

        I_{\nu,\rm em}
        =
        (1+z)^3 I_{\nu,\rm obs}.

    Parameters
    ----------
    observer_frame_intensity : float
        Observer-frame specific intensity.
    redshift : float
        Cosmological redshift.

    Returns
    -------
    float
        Rest-frame specific intensity.
    """
    return observer_frame_intensity * (1.0 + redshift) ** 3


def redshift_correct_flux_density_from_rest_to_observer_frame(
    rest_frame_flux_density: float,
    redshift: float,
):
    r"""
    Apply cosmological redshift correction to a spectral flux density.

    This function applies the cosmological redshift correction to a spectral flux density
    using

    .. math::

        F_{\nu,\mathrm{obs}}
        =
        \frac{F_{\nu,\mathrm{em}}}{1+z},

    where

    - :math:`z` is the cosmological redshift.

    .. important::

        This function **does not include luminosity distance dilution**.
        It assumes geometric effects and source relativistic effects have already been handled
        or are being handled elsewhere.

    Parameters
    ----------
    rest_frame_flux_density : float
        Rest-frame spectral flux density :math:`F_{\nu,\mathrm{em}}`.
    redshift : float
        Cosmological redshift :math:`z`.

    Returns
    -------
    float
        Observer-frame spectral flux density :math:`F_{\nu,\mathrm{obs}}`.
    """
    return rest_frame_flux_density / (1.0 + redshift)


def redshift_correct_flux_density_from_observer_frame_to_rest_frame(
    observer_frame_flux_density: float,
    redshift: float,
):
    r"""
    Transform observer-frame spectral flux density back to rest frame.

    Inverting

    .. math::

        F_{\nu,\mathrm{obs}}
        =
        \frac{F_{\nu,\mathrm{em}}}{1+z},

    gives

    .. math::

        F_{\nu,\mathrm{em}}
        =
        (1+z)\,F_{\nu,\mathrm{obs}}.

    .. important::

        This does **not** undo luminosity distance effects.

    Parameters
    ----------
    observer_frame_flux_density : float
        Observer-frame spectral flux density.
    redshift : float
        Cosmological redshift.

    Returns
    -------
    float
        Rest-frame spectral flux density.
    """
    return observer_frame_flux_density * (1.0 + redshift)


# ========================================================== #
# Utility Functions                                          #
# ========================================================== #
def compute_flux_density_from_rest_frame_luminosity(
    rest_frame_luminosity_density: float,
    luminosity_distance: float,
    redshift: float,
):
    r"""
    Compute observed spectral flux density from rest-frame spectral luminosity.

    The correct cosmological relation is

    .. math::

        F_{\nu,\mathrm{obs}}
        =
        \frac{(1+z)\,L_{\nu,\mathrm{em}}}
        {4\pi D_L^2},

    where

    - :math:`L_{\nu,\mathrm{em}}` is the rest-frame spectral luminosity,
    - :math:`D_L` is the luminosity distance,
    - :math:`z` is the cosmological redshift.

    .. important::

        This formula already includes the appropriate redshift effects.
        Do **not** apply additional :math:`(1+z)` corrections afterward.

    Parameters
    ----------
    rest_frame_luminosity_density : float
        Spectral luminosity :math:`L_{\nu,\mathrm{em}}`.
    luminosity_distance : float
        Luminosity distance :math:`D_L` (must be in consistent units).
    redshift : float
        Cosmological redshift :math:`z`.

    Returns
    -------
    float
        Observer-frame spectral flux density :math:`F_{\nu,\mathrm{obs}}`.
    """
    return (1.0 + redshift) * rest_frame_luminosity_density / (4.0 * np.pi * luminosity_distance**2)


def compute_luminosity_from_rest_frame_flux_density(
    observer_frame_flux_density: float,
    luminosity_distance: float,
    redshift: float,
):
    r"""
    Compute rest-frame spectral luminosity from observed spectral flux density.

    Inverting

    .. math::

        F_{\nu,\mathrm{obs}}
        =
        \frac{(1+z)\,L_{\nu,\mathrm{em}}}
        {4\pi D_L^2},

    gives

    .. math::

        L_{\nu,\mathrm{em}}
        =
        \frac{4\pi D_L^2}{1+z}
        F_{\nu,\mathrm{obs}}.

    Parameters
    ----------
    observer_frame_flux_density : float
        Observer-frame spectral flux density.
    luminosity_distance : float
        Luminosity distance :math:`D_L`.
    redshift : float
        Cosmological redshift.

    Returns
    -------
    float
        Rest-frame spectral luminosity :math:`L_{\nu,\mathrm{em}}`.
    """
    return 4.0 * np.pi * luminosity_distance**2 * observer_frame_flux_density / (1.0 + redshift)


# ========================================================== #
# Utility Functions For Cosmological Distance Measures       #
# ========================================================== #
def compute_luminosity_distance_from_redshift(
    redshift: float,
    cosmology: FLRW = Planck18,
):
    r"""
    Compute luminosity distance from redshift.

    The luminosity distance is defined by

    .. math::

        F = \frac{L}{4\pi D_L^2},

    and relates to angular diameter distance via

    .. math::

        D_L = (1+z)^2 D_A.

    Parameters
    ----------
    redshift : float
        Cosmological redshift :math:`z`.
    cosmology : astropy.cosmology.FLRW, optional
        FLRW model to use. Default is :class:`~astropy.cosmology.Planck18`.

    Returns
    -------
    astropy.units.Quantity
        Luminosity distance :math:`D_L`.
    """
    return cosmology.luminosity_distance(redshift)


def angular_diameter_distance_from_redshift(
    redshift: float,
    cosmology: FLRW = Planck18,
):
    r"""
    Compute angular diameter distance from redshift.

    The angular diameter distance satisfies

    .. math::

        \theta = \frac{\ell}{D_A},

    and is related to luminosity distance by

    .. math::

        D_L = (1+z)^2 D_A.

    Parameters
    ----------
    redshift : float
        Cosmological redshift.
    cosmology : astropy.cosmology.FLRW, optional
        FLRW model to use.

    Returns
    -------
    astropy.units.Quantity
        Angular diameter distance :math:`D_A`.
    """
    return cosmology.angular_diameter_distance(redshift)


def comoving_distance_from_redshift(
    redshift: float,
    cosmology: FLRW = Planck18,
):
    r"""
    Compute comoving distance from redshift.

    The comoving distance is related to luminosity distance via

    .. math::

        D_L = (1+z) D_C.

    Parameters
    ----------
    redshift : float
        Cosmological redshift.
    cosmology : astropy.cosmology.FLRW, optional
        FLRW model to use.

    Returns
    -------
    astropy.units.Quantity
        Comoving distance :math:`D_C`.
    """
    return cosmology.comoving_distance(redshift)


def compute_redshift_from_luminosity_distance(
    luminosity_distance: u.Quantity,
    cosmology: FLRW = Planck18,
):
    r"""
    Compute redshift from luminosity distance.

    This function numerically inverts

    .. math::

        D_L(z)

    using :func:`astropy.cosmology.z_at_value`.

    Parameters
    ----------
    luminosity_distance : astropy.units.Quantity
        Luminosity distance :math:`D_L`.
    cosmology : astropy.cosmology.FLRW, optional
        FLRW model to use.

    Returns
    -------
    float
        Cosmological redshift :math:`z`.
    """
    return z_at_value(cosmology.luminosity_distance, luminosity_distance)


def compute_redshift_from_angular_diameter_distance(
    angular_diameter_distance: u.Quantity,
    cosmology: FLRW = Planck18,
    z_min: float = 0.0,
    z_max: float = 10.0,
):
    r"""
    Compute redshift from angular diameter distance.

    .. warning::

        The angular diameter distance is not monotonic with redshift.
        It peaks at :math:`z \sim 1.5` in standard cosmologies.
        Therefore, this inversion requires specifying bounds.

    Parameters
    ----------
    angular_diameter_distance : astropy.units.Quantity
        Angular diameter distance :math:`D_A`.
    cosmology : astropy.cosmology.FLRW, optional
        FLRW model to use.
    z_min : float, optional
        Lower bound for redshift search.
    z_max : float, optional
        Upper bound for redshift search.

    Returns
    -------
    float
        Cosmological redshift.
    """
    return z_at_value(
        cosmology.angular_diameter_distance,
        angular_diameter_distance,
        zmin=z_min,
        zmax=z_max,
    )


def compute_redshift_from_comoving_distance(
    comoving_distance: u.Quantity,
    cosmology: FLRW = Planck18,
):
    r"""
    Compute redshift from comoving distance.

    Numerically inverts

    .. math::

        D_C(z)

    using :func:`astropy.cosmology.z_at_value`.

    Parameters
    ----------
    comoving_distance : astropy.units.Quantity
        Comoving distance :math:`D_C`.
    cosmology : astropy.cosmology.FLRW, optional
        FLRW model to use.

    Returns
    -------
    float
        Cosmological redshift.
    """
    return z_at_value(cosmology.comoving_distance, comoving_distance)
