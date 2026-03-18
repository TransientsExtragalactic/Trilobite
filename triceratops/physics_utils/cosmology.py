"""
Cosmological utility functions.

This module provides helper utilities used throughout the Triceratops
framework for resolving cosmological quantities such as redshift and
distances. These functions ensure a consistent interface for interacting
with Astropy cosmologies and for converting between commonly used
cosmological distance measures.
"""

from astropy.cosmology import z_at_value

from triceratops.utils.config import triceratops_config


def get_cosmology(cosmology=None):
    """
    Return the cosmology to be used for calculations.

    If no cosmology is provided, the default cosmology defined in the
    Triceratops configuration is returned.

    Parameters
    ----------
    cosmology : astropy.cosmology.FLRW, optional
        Cosmology to use. If ``None``, the cosmology specified in
        ``triceratops_config['physics.default_cosmology']`` is returned.

    Returns
    -------
    astropy.cosmology.FLRW
        Cosmology object used for subsequent calculations.

    Notes
    -----
    This helper ensures that all parts of the framework consistently
    use the configured cosmology unless explicitly overridden.
    """
    if cosmology is None:
        return triceratops_config["physics.default_cosmology"]
    return cosmology


def resolve_cosmological_distances(
    redshift=None,
    luminosity_distance=None,
    angular_diameter_distance=None,
    proper_distance=None,
    cosmology=None,
):
    """
    Resolve cosmological distance measures and redshift.

    Given **exactly one** of redshift or a cosmological distance,
    compute all other distance measures consistently using the
    specified cosmology.

    Parameters
    ----------
    redshift : float or array-like, optional
        Cosmological redshift.

    luminosity_distance : `~astropy.units.Quantity`, optional
        Luminosity distance :math:`D_L`.

    angular_diameter_distance : `~astropy.units.Quantity`, optional
        Angular diameter distance :math:`D_A`.

    proper_distance : `~astropy.units.Quantity`, optional
        Proper (comoving line-of-sight) distance :math:`D`.

    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used to compute the relations between redshift
        and distances. If ``None``, the configured default cosmology
        is used.

    Returns
    -------
    dict
        Dictionary containing

        - ``redshift`` : float or ndarray
        - ``luminosity_distance`` : `~astropy.units.Quantity`
        - ``angular_diameter_distance`` : `~astropy.units.Quantity`
        - ``proper_distance`` : `~astropy.units.Quantity`

    Raises
    ------
    ValueError
        If none or more than one of the inputs is provided.

    Notes
    -----
    The relations between cosmological distances are

    .. math::

        D_L = (1+z)^2 D_A

    and

    .. math::

        D = D_C

    where :math:`D_C` is the line-of-sight comoving distance.

    Astropy internally handles these relations via the cosmology object.

    Examples
    --------
    Resolve distances from redshift

    >>> resolve_cosmological_distances(redshift=0.5)

    Resolve redshift from luminosity distance

    >>> resolve_cosmological_distances(
    ...     luminosity_distance=3 * u.Gpc
    ... )
    """
    cosmo = get_cosmology(cosmology)

    provided = [
        redshift is not None,
        luminosity_distance is not None,
        angular_diameter_distance is not None,
        proper_distance is not None,
    ]

    if sum(provided) != 1:
        raise ValueError(
            "Exactly one of redshift, luminosity_distance, "
            "angular_diameter_distance, or proper_distance must be provided."
        )

    # ---------------------------------------------------
    # Determine redshift
    # ---------------------------------------------------
    if redshift is None:
        if luminosity_distance is not None:
            redshift = z_at_value(cosmo.luminosity_distance, luminosity_distance)

        elif angular_diameter_distance is not None:
            redshift = z_at_value(cosmo.angular_diameter_distance, angular_diameter_distance)

        elif proper_distance is not None:
            redshift = z_at_value(cosmo.comoving_distance, proper_distance)

    # ---------------------------------------------------
    # Compute distances from redshift
    # ---------------------------------------------------
    luminosity_distance = cosmo.luminosity_distance(redshift)
    angular_diameter_distance = cosmo.angular_diameter_distance(redshift)
    proper_distance = cosmo.comoving_distance(redshift)

    return {
        "redshift": redshift,
        "luminosity_distance": luminosity_distance,
        "angular_diameter_distance": angular_diameter_distance,
        "proper_distance": proper_distance,
    }


def redshift_to_age(z, cosmology=None):
    """
    Compute the age of the universe at a given redshift.

    Parameters
    ----------
    z : float or array-like
        Cosmological redshift.

    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used to evaluate the relation between redshift and cosmic
        age. If ``None``, the configured default cosmology is used.

    Returns
    -------
    `~astropy.units.Quantity`
        Age of the universe at redshift ``z`` with units of time.

    Notes
    -----
    This is equivalent to calling

    .. code-block:: python

        cosmology.age(z)

    from the Astropy cosmology API.

    Examples
    --------
    >>> redshift_to_age(1.0)
    <Quantity ... Gyr>
    """
    cosmo = get_cosmology(cosmology)
    return cosmo.age(z)


def redshift_to_lookback_time(z, cosmology=None):
    r"""
    Compute the lookback time corresponding to a given redshift.

    Parameters
    ----------
    z : float or array-like
        Cosmological redshift.

    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used to compute the lookback time. If ``None``, the
        configured default cosmology is used.

    Returns
    -------
    `~astropy.units.Quantity`
        Lookback time to redshift ``z`` with units of time.

    Notes
    -----
    The lookback time is the difference between the current age of the
    universe and the age of the universe at redshift ``z``:

    .. math::

        t_{\mathrm{lookback}} = t_0 - t(z)

    Examples
    --------
    >>> redshift_to_lookback_time(0.5)
    <Quantity ... Gyr>
    """
    cosmo = get_cosmology(cosmology)
    return cosmo.lookback_time(z)


def age_to_redshift(age, cosmology=None):
    """
    Compute the redshift corresponding to a given cosmic age.

    Parameters
    ----------
    age : `~astropy.units.Quantity`
        Age of the universe.

    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used to perform the inversion. If ``None``, the
        configured default cosmology is used.

    Returns
    -------
    float or ndarray
        Redshift corresponding to the provided cosmic age.

    Notes
    -----
    This function numerically inverts the cosmological age relation using
    :func:`astropy.cosmology.z_at_value`.

    Examples
    --------
    >>> import astropy.units as u
    >>> age_to_redshift(5 * u.Gyr)
    1.2
    """
    cosmo = get_cosmology(cosmology)
    return z_at_value(cosmo.age, age)


def angular_to_physical(theta, redshift=None, cosmology=None):
    r"""
    Convert an angular size to a physical transverse size.

    Parameters
    ----------
    theta : `~astropy.units.Quantity`
        Angular size (e.g., arcsec or radians).

    redshift : float
        Redshift of the object.

    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used to compute the angular diameter distance.
        If ``None``, the configured default cosmology is used.

    Returns
    -------
    `~astropy.units.Quantity`
        Physical transverse size corresponding to the angular extent.

    Notes
    -----
    The physical transverse size is given by

    .. math::

        s = \theta \, D_A

    where :math:`D_A` is the angular diameter distance.

    Examples
    --------
    >>> import astropy.units as u
    >>> angular_to_physical(1 * u.arcsec, redshift=1.0)
    <Quantity ... kpc>
    """
    distances = resolve_cosmological_distances(redshift=redshift, cosmology=cosmology)
    DA = distances["angular_diameter_distance"]
    return theta * DA


def physical_to_angular(size, redshift=None, cosmology=None):
    r"""
    Convert a physical transverse size to an angular size.

    Parameters
    ----------
    size : `~astropy.units.Quantity`
        Physical transverse size.

    redshift : float
        Redshift of the object.

    cosmology : `~astropy.cosmology.FLRW`, optional
        Cosmology used to compute the angular diameter distance.
        If ``None``, the configured default cosmology is used.

    Returns
    -------
    `~astropy.units.Quantity`
        Angular size corresponding to the physical extent.

    Notes
    -----
    The angular size is given by

    .. math::

        \theta = \frac{s}{D_A}

    where :math:`D_A` is the angular diameter distance.

    Examples
    --------
    >>> import astropy.units as u
    >>> physical_to_angular(10 * u.kpc, redshift=0.5)
    <Quantity ... arcsec>
    """
    distances = resolve_cosmological_distances(redshift=redshift, cosmology=cosmology)
    DA = distances["angular_diameter_distance"]
    return size / DA
