"""
Helper module containing various routines for CSM / ejecta profile calculations relevant to supernovae.

In the :mod:`triceratops.dynamics.supernovae.shock_dynamics` module, various shock engines require knowledge of the
ejecta and circumstellar medium (CSM) density profiles to compute shock dynamics. This module provides
routines to compute these profiles based on standard models in the literature.
"""

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from triceratops._typing import _ArrayLike, _UnitBearingArrayLike


# ============================================ #
# Ejecta Profiles and Routines                 #
# ============================================ #
def _normalize_inner_BPL_ejecta_cgs(
    E_ej: "_ArrayLike",
    M_ej: "_ArrayLike",
    n: "_ArrayLike" = 10,
    delta: "_ArrayLike" = 0,
) -> tuple["_ArrayLike", "_ArrayLike"]:
    r"""
    Optimized computation of a BPL normalization parameters in CGS for performance.

    See the public-facing function :func:`normalize_inner_BPL_ejecta` for details.

    Parameters
    ----------
    E_ej: float or array-like
        The total kinetic energy of the ejecta in CGS units (erg).
    M_ej: float or array-like
        The total mass of the ejecta in CGS units (g).
    n: float or array-like
        The outer ejecta density profile power-law index. By default, this is set to ``10``.
    delta: float or array-like, optional
        The inner ejecta density profile power-law index. By default, this is set to ``0``.

    Returns
    -------
    v_t : float or array-like
        The transition velocity between the inner and outer ejecta profiles in CGS units (cm/s).
    K: float or array-like
        The normalization constant of the ejecta density profile in CGS units. The units are
        ``g * cm^(2*delta-3) * s^(3-delta)``.
    """
    # Compute the energy per unit mass for derivation of v_t.
    E_per_M = 2.0 * E_ej / M_ej  # factor of 2 for v^2

    # Compute transition velocity v_t
    v_t = np.sqrt(E_per_M * ((5 - delta) * (n - 5)) / ((3 - delta) * (n - 3)))

    # Compute normalization constant K
    K = M_ej * v_t ** (delta - 3) * (1 / (4.0 * np.pi)) * ((3 - delta) * (n - 3)) / (n - delta)

    return v_t, K


def normalize_inner_BPL_ejecta(
    E_ej: "_UnitBearingArrayLike",
    M_ej: "_UnitBearingArrayLike",
    n: float = 10,
    delta: float = 0,
) -> tuple[u.Quantity, u.Quantity]:
    r"""
    Compute the transition velocity and normalization constant for a Chevalier-style ejecta profile.

    This function computes the transition velocity :math:`v_t` and normalization constant :math:`K` of a
    broken power-law ejecta density profile as described in
    :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`.
    See the notes for a detailed description of the theory.

    Parameters
    ----------
    E_ej: astropy.units.Quantity or array-like
        The total kinetic energy of the ejecta. If units are specified, then they will be taken into
        account. Otherwise, CGS units are assumed.
    M_ej: astropy.units.Quantity or array-like
        The total mass of the ejecta. If units are specified, then they will be taken into
        account. Otherwise, CGS units are assumed.
    n: float
        The outer ejecta density profile power-law index. By default, this is set to ``10``.
    delta: float, optional
        The inner ejecta density profile power-law index. By default, this is set to ``0``.

    Returns
    -------
    v_t : astropy.units.Quantity
        The transition velocity between the inner and outer ejecta profiles.
    K: astropy.units.Quantity
        The normalization constant of the ejecta density profile.

    Notes
    -----
    As described in :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`, the ejecta velocity profile
    is well described by a broken power-law. During homologous expansion, :math:`r = vt` implies that the density
    of the ejecta is likewise a broken power-law in velocity space:

    .. math::

        \rho(r,t) = Kt^{-3} \begin{cases}
            v^{-\delta}, & v < v_t \\
            v_t^{n-\delta} v^{-n}, & v \geq v_t,
        \end{cases}

    where :math:`v_t` is the transition velocity between the inner and outer ejecta profiles. The total mass of
    the ejecta is given by :math:`M_{\rm ej}` and must be conserved:

    .. math::

        M_{\rm ej} = \int_0^{\infty} 4\pi r^2 \rho(r,t) dr = 4\pi K v_t^{3-\delta} \frac{n-\delta}{(3-\delta)(n-3)}.

    Similarly, the total kinetic energy of the ejecta is given by :math:`E_{\rm ej}` and must also be conserved:

    .. math::

        E_{\rm ej} = \int_0^{\infty} \frac{1}{2} 4\pi r^2 \rho(r,t) v^2 dr =
        2\pi K v_t^{5-\delta} \frac{n-\delta}{(5-\delta)(n-5)}.

    In terms of the energy per unit mass, :math:`E_{\rm ej}/M_{\rm ej}`, these two equations can be combined to
    solve for the transition velocity:

    .. math::

        v_t^2 = \frac{2(5-\delta)(n-5)}{(3-\delta)(n-3)} \frac{E_{\rm ej}}{M_{\rm ej}}.

    Finally, substituting this back into the mass equation allows us to solve for the normalization constant
    :math:`K`:

    .. math::

        K = \frac{1}{4\pi} \left(\frac{(3-\delta)(n-3)}{(n-\delta)}\right) \frac{M_{\rm ej}}{v_t^{3-\delta}}.

    """
    # Ensure that ``n`` and ``delta`` are valid values for convergence. This requires that delta < 3 and n > 5.
    if delta >= 3:
        raise ValueError("The inner ejecta density profile index `delta` must be less than 3 for convergence.")
    if n <= 5:
        raise ValueError("The outer ejecta density profile index `n` must be greater than 5 for convergence.")

    # Convert any unit-bearing inputs to CGS for internal calculations
    E_ej_cgs = ensure_in_units(E_ej, "erg")
    M_ej_cgs = ensure_in_units(M_ej, "g")

    # Call the optimized internal function for computation
    v_t_cgs, K_cgs = _normalize_inner_BPL_ejecta_cgs(
        E_ej=E_ej_cgs,
        M_ej=M_ej_cgs,
        n=n,
        delta=delta,
    )

    # Attach units to the outputs. For v_t, CGS velocity is cm/s. For K, the units are g * cm^(2*delta-3)
    # * s^(3-delta).
    v_t = v_t_cgs * (u.cm / u.s)
    K_units = u.g * u.cm ** (2 * delta - 3) * u.s ** (3 - delta)
    K = K_cgs * K_units

    return v_t, K


def compute_BKC_ejecta_parameters(
    E_ej: "_UnitBearingArrayLike",
    M_ej: "_UnitBearingArrayLike",
) -> tuple[u.Quantity, float]:
    r"""
    Compute empirical outer ejecta normalization parameters following Berger–Kulkarni–Chevalier (BKC) scalings.

    This model provides an empirical calibration of the *outer* ejecta
    density profile based on numerical supernova explosion models and
    radio supernova observations :footcite:p:`2002ApJ...577L...5B,1999ApJ...510..379M`.

    The ejecta density is assumed to take the Chevalier self-similar form

    .. math::

        \rho(r,t) = K\,t^{-3}\left(\frac{r}{t}\right)^{-n}

    with a fixed outer power-law index

    .. math::

        n = 10.18.

    The normalization :math:`K` is calibrated as a power-law function of
    ejecta kinetic energy and ejecta mass:

    .. math::

        K = 3 \times 10^{96}
        \left(\frac{E_{\rm ej}}{10^{51}\,\mathrm{erg}}\right)^{3.59}
        \left(\frac{M_{\rm ej}}{10\,M_\odot}\right)^{-2.59}

    Parameters
    ----------
    E_ej : astropy.units.Quantity
        Total kinetic energy of the supernova ejecta.
    M_ej : astropy.units.Quantity
        Total mass of the supernova ejecta.

    Returns
    -------
    K : astropy.units.Quantity
        Ejecta density normalization with units
        ``g * cm^(n-3) * s^(3-n)``.
    n : float
        Outer ejecta density power-law index.

    Notes
    -----
    This prescription is **empirical**, not derived from exact mass/energy
    conservation. It is calibrated for:

    - Red supergiant progenitors
    - Wind-like circumstellar media
    - Early-time radio supernova evolution

    For applications requiring strict mass/energy conservation or
    arbitrary ejecta profiles, use
    :func:`compute_chevalier_ejecta_parameters` instead.

    References
    ----------
    Chevalier, R. A. (1982), ApJ, 258, 790
    Matzner & McKee (1999), ApJ, 510, 379
    Berger et al. (2002), ApJ, 572, 503
    """
    # Convert inputs to CGS
    E_cgs = E_ej.to(u.erg).value
    M_cgs = M_ej.to(u.g).value

    K_cgs, n = _optimized_compute_BKC_ejecta_parameters(E_cgs, M_cgs)

    # Attach physical units
    K_units = u.g * u.cm ** (n - 3) * u.s ** (3 - n)
    K = K_cgs * K_units

    return K, n


def _optimized_compute_BKC_ejecta_parameters(
    E_ej: "_ArrayLike",
    M_ej: "_ArrayLike",
) -> tuple["_ArrayLike", float]:
    """
    Optimized CGS backend for Berger–Kulkarni–Chevalier ejecta scalings.

    Parameters
    ----------
    E_ej : float or array-like
        Ejecta kinetic energy in erg.
    M_ej : float or array-like
        Ejecta mass in g.

    Returns
    -------
    K : float or array-like
        Ejecta density normalization in CGS units.
    n : float
        Outer ejecta density power-law index.
    """
    n = 10.18

    K = 3.0e96 * (E_ej / 1.0e51) ** 3.59 * (M_ej / (10.0 * 1.989e33)) ** -2.59

    return K, n


# --- Ejecta Profile Functions for Numerics --- #
def get_broken_power_law_ejecta_kernel_func(
    E_ej: u.Quantity,
    M_ej: u.Quantity,
    n: float = 10,
    delta: float = 0,
):
    r"""
    Generate a callable broken power-law ejecta kernel function ``G(v)`` in CGS units.

    The returned function satisfies

    .. math::

        \rho_{\rm ej}(r,t) = t^{-3} G\!\left(\frac{r}{t}\right),

    with

    .. math::

        G(v) =
        \begin{cases}
            K\, v^{-\delta}, & v < v_t \\
            K\, v_t^{n-\delta} v^{-n}, & v \ge v_t
        \end{cases}

    where ``v_t`` and ``K`` are fixed by total ejecta mass and energy.

    Parameters
    ----------
    E_ej : astropy.units.Quantity
        Total kinetic energy of the ejecta.
    M_ej : astropy.units.Quantity
        Total mass of the ejecta.
    n : float, optional
        Outer ejecta density power-law index (must be > 5).
    delta : float, optional
        Inner ejecta density power-law index (must be < 3).

    Returns
    -------
    G_ej : callable
        Function ``G(v)`` returning the ejecta kernel in CGS units
        ``g * s^3 / cm^3`` for velocity ``v`` in ``cm/s``.
    """
    # --- validation ---
    if delta >= 3:
        raise ValueError("Inner ejecta index delta must be < 3.")
    if n <= 5:
        raise ValueError("Outer ejecta index n must be > 5.")
    if n <= delta:
        raise ValueError("Outer index n must exceed inner index delta.")

    # --- convert to CGS ---
    E_cgs = E_ej.to(u.erg).value
    M_cgs = M_ej.to(u.g).value

    # --- compute normalization ---
    v_t, K_inner = _normalize_inner_BPL_ejecta_cgs(
        E_ej=E_cgs,
        M_ej=M_cgs,
        n=n,
        delta=delta,
    )

    # Precompute outer normalization factor
    K_outer = K_inner * v_t ** (n - delta)

    # --- construct kernel ---
    def _G_ej(v):
        v = np.asarray(v)
        G = np.empty_like(v, dtype=float)

        inner = v < v_t
        outer = ~inner

        G[inner] = K_inner * v[inner] ** (-delta)
        G[outer] = K_outer * v[outer] ** (-n)

        return G

    return _G_ej


def get_wind_csm_density_func(
    mass_loss_rate: u.Quantity,
    wind_velocity: u.Quantity,
):
    r"""
    Generate a callable function for the wind-like CSM density profile in CGS units.

    This function returns a callable that computes the density of a wind-like circumstellar medium (CSM)
    at a given radius based on the mass-loss rate and wind velocity.

    Parameters
    ----------
    mass_loss_rate: astropy.units.Quantity
        The mass-loss rate of the progenitor star.
    wind_velocity: astropy.units.Quantity
        The velocity of the stellar wind.

    Returns
    -------
    csm_density_func: callable
        A function that takes a radius (with units) as input and returns the CSM density at that radius.
    """
    # Compute the normalization constant rho_0 using the provided parameters
    rho_0 = compute_wind_csm_parameters(
        mass_loss_rate=mass_loss_rate,
        wind_velocity=wind_velocity,
    ).to_value("g/cm**3")

    def csm_density_func(radius: float) -> float:
        """
        Compute the CSM density at a given radius.

        Parameters
        ----------
        radius: astropy.units.Quantity
            The radius at which to evaluate the CSM density.

        Returns
        -------
        rho_csm: astropy.units.Quantity
            The CSM density at the specified radius.
        """
        return rho_0 * radius**-2

    return csm_density_func


# ================================================== #
# CSM Profiles and Routines                          #
# ================================================== #
def compute_wind_csm_parameters(
    mass_loss_rate: "_UnitBearingArrayLike",
    wind_velocity: "_UnitBearingArrayLike",
) -> u.Quantity:
    r"""
    Compute the normalization constant for a wind-like circumstellar medium (CSM) density profile.

    For a steady wind :math:`\dot{M}` with velocity :math:`v_w`, the density profile of the CSM is given by

    .. math::

        \rho_{\rm CSM}(r) = \frac{\dot{M}}{4\pi r^2 v_w} = \rho_0 r^{-2}.

    This function computes the normalization constant :math:`\rho_0` given the mass-loss rate and wind velocity.

    Parameters
    ----------
    mass_loss_rate: astropy.units.Quantity or array-like
        The mass-loss rate of the progenitor star. If units are specified, they will be taken into account.
        Otherwise, CGS units are assumed (g/s).
    wind_velocity: astropy.units.Quantity or array-like
        The velocity of the stellar wind. If units are specified, they will be taken into account.
        Otherwise, CGS units are assumed (cm/s).

    Returns
    -------
    rho_0: astropy.units.Quantity
        The normalization constant of the wind-like CSM density profile.
    """
    # Convert inputs to CGS units for internal calculations
    mdot_cgs = mass_loss_rate.to(u.g / u.s).value
    v_w_cgs = wind_velocity.to(u.cm / u.s).value

    # Compute the normalization constant rho_0
    # NOTE: we skip the call to the optimized function here since this is a simple one-liner.
    rho_0_cgs = mdot_cgs / (4.0 * np.pi * v_w_cgs)

    # Attach units to the output. The units are g/cm.
    rho_0 = rho_0_cgs * (u.g / u.cm)

    return rho_0


def _optimized_compute_wind_csm_parameters(
    mass_loss_rate: "_ArrayLike",
    wind_velocity: "_ArrayLike",
) -> "_ArrayLike":
    """
    Optimized computation of wind CSM normalization constant in CGS for performance.

    See the public-facing function :func:`compute_wind_csm_parameters` for details.

    Parameters
    ----------
    mass_loss_rate: float or array-like
        The mass-loss rate of the progenitor star in CGS units (g/s).
    wind_velocity: float or array-like
        The velocity of the stellar wind in CGS units (cm/s).

    Returns
    -------
    rho_0: float or array-like
        The normalization constant of the wind-like CSM density profile in CGS units (g/cm).
    """
    # Compute the normalization constant rho_0
    rho_0 = mass_loss_rate / (4.0 * np.pi * wind_velocity)

    return rho_0
