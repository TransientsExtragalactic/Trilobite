"""
Various utilities for working with the Rankine-Hugoniot conditions in fluid dynamics.

This module contains a few (advanced) RH scenarios which are necessary in various parts of the
dynamics module. The intent here is not to exhaustively cover all possible RH conditions, but rather to
provide a few key implementations that are useful for astrophysical shock dynamics calculations and sufficiently
complex to be worth isolating into their own module.
"""

import astropy.constants as const
import astropy.units as u
import numpy as np

# --- Handle Necessary CGS constants for Low-Level API --- #
k_B_cgs = const.k_B.cgs.value  # Boltzmann constant in CGS
m_p_cgs = const.m_p.cgs.value  # Proton mass in CGS

# ========================================= #
# Strong Shock Utilities                    #
# ========================================= #

# --- Low-Level CGS Implementations --- #
# Implementations of the Low-Level strong shock functions in CGS units.
# These are used by the higher-level functions that handle units.
#
# CONVENTION:
# Naming should be _compute_<s/w>_<c/h>_<physical_quantity>_cgs
# where:
# s = strong shock
# w = weak shock
# c = cold upstream
# h = hot upstream


def _compute_s_c_shock_magnetic_field_cgs(
    shock_velocity: float,
    upstream_density: float,
    gamma: float = 5 / 3,
    epsilon_B: float = 0.1,
) -> float:
    r"""
    Compute the magnetic field strength immediately behind a strong shock assuming energy partition.

    This function estimates the post-shock magnetic field for classical, strong, cold shocks using
    an equipartition argument (e.g. :footcite:t:`Chevalier1998SynchrotronSelfAbsorption`). This function
    assumes a cold, strong, classical shock propagating into an unmagnetized medium. The magnetic field strength
    is computed by assuming that a fixed fraction of the post-shock internal energy density is converted into
    magnetic field energy.

    Parameters
    ----------
    shock_velocity: float
        The relative velocity of the shock front in cm/s. In cases where the pre-shock medium is
        not at rest, this should be the *relative* velocity of the shock in the comoving frame.
    upstream_density: float
        The density of the unshocked medium ahead of the shock in g/cm^3. Typically,
        this is the density of the CSM of ISM.
    gamma: float, optional
        The adiabatic index of the fluid. Default is 5/3 (ideal monatomic gas).
    epsilon_B: float, optional
        The fraction of the post-shock internal energy density that is converted into magnetic field energy.
        Default is 0.1.

    Returns
    -------
    float
        The magnetic field strength immediately behind the shock front in Gauss.

    Notes
    -----
    The post-shock thermal energy of a classical, strong, cold shock is given by

    .. math::

        U = \frac{3}{2} \frac{(R-1)}{R^2} \rho_1 v_{\rm sh}^2.

    Assuming :math:`\epsilon_B` fraction of this energy goes into magnetic fields, the magnetic field strength is
    given by

    .. math::

        B = \sqrt{8\pi \epsilon_B U} = \sqrt{8 \pi \epsilon_B \frac{3}{2} \frac{(R-1)}{R^2} \rho_1 v_{\rm sh}^2},
    """
    # Calculate the density jump across the shock using RH conditions in strong shock.
    R = (gamma + 1) / (gamma - 1)

    # Compute the internal energy density behind the shock
    U = (3 / 2) * ((R - 1) / R**2) * upstream_density * shock_velocity**2

    # Compute the magnetic field strength
    B = np.sqrt(8 * np.pi * epsilon_B * U)

    return B


def _compute_s_c_shock_pressure_cgs(
    shock_velocity: float,
    upstream_density: float,
    gamma: float = 5 / 3,
) -> float:
    """
    Compute the gas pressure immediately behind a classical strong shock propagating into a cold upstream medium.

    Parameters
    ----------
    shock_velocity: float
        The relative velocity of the shock front in cm/s.
    upstream_density: float
        The density of the unshocked medium ahead of the shock in g/cm^3.
    gamma: float, optional
        The adiabatic index of the fluid.

    Returns
    -------
    float
        The gas pressure immediately behind the shock front in dyne/cm^2.
    """
    # Strong-shock, cold-upstream pressure jump
    P2 = (2.0 / (gamma + 1.0)) * upstream_density * shock_velocity**2
    return P2


def _compute_s_c_shock_temperature_cgs(
    shock_velocity: float,
    gamma: float = 5 / 3,
    mu: float = 0.61,
) -> float:
    r"""
    Compute the temperature immediately behind a strong shock under canonical assumptions.

    Low-level CGS implementation of `compute_strong_shock_temperature`. Assumptions:

    - The upstream region is cold and the shock is strong (Mach >> 1).
    - The gas is ideal with adiabatic index gamma.
    - The shock is non-relativistic.
    - Temperature is inferred from the ideal-gas relation using a mean molecular weight mu.

    Parameters
    ----------
    shock_velocity: float
        Shock speed relative to the upstream medium in cm/s.
    gamma: float, optional
        Adiabatic index.
    mu: float, optional
        Mean molecular weight (dimensionless), in units of m_p.

    Returns
    -------
    float
        Post-shock temperature in Kelvin.
    """
    prefac = 2.0 * (gamma - 1.0) / (gamma + 1.0) ** 2
    T2 = prefac * (mu * m_p_cgs / k_B_cgs) * shock_velocity**2
    return T2


def _compute_s_shock_velocity_cgs(
    shock_velocity: float,
    gamma: float = 5 / 3,
) -> float:
    r"""
    Compute the downstream bulk velocity for a strong shock into a cold upstream medium.

    Low-level CGS implementation of `compute_strong_shock_velocity`. Assumptions:

    - Strong shock (Mach >> 1), cold upstream.
    - Non-relativistic, ideal gas.
    - Returned velocity is the *lab-frame* downstream velocity assuming the upstream is at rest.

    Parameters
    ----------
    shock_velocity: float
        Shock speed relative to the upstream medium in cm/s.
    gamma: float, optional
        Adiabatic index.

    Returns
    -------
    float
        Downstream bulk velocity in cm/s.
    """
    R = (gamma + 1.0) / (gamma - 1.0)
    v2 = shock_velocity * (R - 1.0) / R  # == 2/(gamma+1) * v_sh
    return v2


def _compute_s_density_cgs(
    upstream_density: float,
    gamma: float = 5 / 3,
) -> float:
    r"""
    Compute the downstream density for a strong shock into a cold upstream medium.

    Low-level CGS implementation of `compute_strong_shock_density`. Assumptions:

    - Strong shock (Mach >> 1), cold upstream.
    - Non-relativistic, ideal gas.

    Parameters
    ----------
    upstream_density: float
        Upstream density in g/cm^3.
    gamma: float, optional
        Adiabatic index.

    Returns
    -------
    float
        Downstream density in g/cm^3.
    """
    R = (gamma + 1.0) / (gamma - 1.0)
    return R * upstream_density


def compute_strong_cold_shock_magnetic_field(
    shock_velocity,
    upstream_density,
    gamma: float = 5 / 3,
    epsilon_B: float = 0.1,
):
    r"""
    Compute the magnetic field strength immediately behind a strong shock assuming energy partition.

    Utilizing the canonical assumptions for strong shocks and the Rankine-Hugoniot jump conditions, this function
    estimates the magnetic field strength just behind the shock front. It assumes that a fixed fraction of the
    post-shock internal energy density is converted into magnetic field energy. The resulting magnetic field strength
    is

    .. math::

        B^2 = 8\pi \left(\frac{3}{2}\right) \epsilon_B \rho_{\rm 0} v_{\rm sh}^2 \frac{(R-1)}{R^2},

    where :math:`R` is the density jump across the shock given by the Rankine-Hugoniot conditions for a strong shock:

    .. math::

        R = \frac{\gamma + 1}{\gamma - 1}.

    Parameters
    ----------
    shock_velocity: ~astropy.units.Quantity, float, or np.ndarray
        The relative velocity of the shock front (in CGS units if no units are provided). This should be the velocity
        of the shock front in the comoving frame of the upstream medium. Alternatively, if the upstream medium is
        at rest, this is the shock velocity in the lab frame.
    upstream_density: ~astropy.units.Quantity, float, or np.ndarray
        The density of the unshocked medium ahead of the shock (in CGS units if no units are provided). Typically,
        this is the density of the CSM of ISM.
    gamma: float, optional
        The adiabatic index of the fluid. Default is 5/3 (ideal monatomic gas).
    epsilon_B: float, optional
        The fraction of the post-shock internal energy density that is converted into magnetic field energy.
        Default is 0.1.

    Returns
    -------
    ~astropy.units.Quantity
        The magnetic field strength immediately behind the shock front.
    """
    # Handle units
    if isinstance(shock_velocity, u.Quantity):
        shock_velocity_cgs = shock_velocity.to(u.cm / u.s).value
    else:
        shock_velocity_cgs = shock_velocity  # Assume CGS

    if isinstance(upstream_density, u.Quantity):
        upstream_density_cgs = upstream_density.to(u.g / u.cm**3).value
    else:
        upstream_density_cgs = upstream_density  # Assume CGS

    # Compute the magnetic field in CGS
    B_cgs = _compute_s_c_shock_magnetic_field_cgs(
        shock_velocity=shock_velocity_cgs,
        upstream_density=upstream_density_cgs,
        gamma=gamma,
        epsilon_B=epsilon_B,
    )

    return B_cgs * u.G  # Return with Gauss units


def compute_strong_cold_shock_pressure(
    shock_velocity,
    upstream_density,
    gamma: float = 5 / 3,
):
    r"""
    Compute the gas pressure immediately behind a strong shock propagating into a cold upstream medium.

    Under the canonical strong-shock Rankine-Hugoniot assumptions (cold upstream, Mach >> 1, ideal gas),
    the downstream (post-shock) pressure is

    .. math::

        P_2 = \frac{2}{\gamma + 1} \rho_1 v_{\rm sh}^2,

    where :math:`v_{\rm sh}` is the shock speed relative to the upstream medium and :math:`\rho_1`
    is the upstream density.

    Parameters
    ----------
    shock_velocity: ~astropy.units.Quantity, float, or np.ndarray
        The relative velocity of the shock front (in CGS units if no units are provided). This should be the velocity
        of the shock front in the comoving frame of the upstream medium. Alternatively, if the upstream medium is
        at rest, this is the shock velocity in the lab frame.
    upstream_density: ~astropy.units.Quantity, float, or np.ndarray
        The density of the unshocked medium ahead of the shock (in CGS units if no units are provided). Typically,
        this is the density of the CSM or ISM.
    gamma: float, optional
        The adiabatic index of the fluid. Default is 5/3 (ideal monatomic gas).

    Returns
    -------
    ~astropy.units.Quantity
        The gas pressure immediately behind the shock front.
    """
    # Handle units
    if isinstance(shock_velocity, u.Quantity):
        shock_velocity_cgs = shock_velocity.to(u.cm / u.s).value
    else:
        shock_velocity_cgs = shock_velocity  # Assume CGS

    if isinstance(upstream_density, u.Quantity):
        upstream_density_cgs = upstream_density.to(u.g / u.cm**3).value
    else:
        upstream_density_cgs = upstream_density  # Assume CGS

    P_cgs = _compute_s_c_shock_pressure_cgs(
        shock_velocity=shock_velocity_cgs,
        upstream_density=upstream_density_cgs,
        gamma=gamma,
    )

    return P_cgs * (u.dyne / u.cm**2)


def compute_strong_cold_shock_temperature(
    shock_velocity,
    gamma: float = 5 / 3,
    mu: float = 0.61,
):
    r"""
    Compute the temperature immediately behind a strong shock under canonical assumptions.

    Using the strong-shock Rankine-Hugoniot relations for a cold upstream medium, and assuming an ideal gas
    equation of state, the downstream temperature is

    .. math::

        T_2 = \frac{P_2}{n_2 k_B} = \frac{P_2 \mu m_p}{\rho_2 k_B},

    with :math:`\rho_2 = R\rho_1` and :math:`R = (\gamma+1)/(\gamma-1)`. Eliminating :math:`\rho_1` gives

    .. math::

        T_2 = \frac{2(\gamma-1)}{(\gamma+1)^2} \frac{\mu m_p}{k_B} v_{\rm sh}^2.

    Parameters
    ----------
    shock_velocity: ~astropy.units.Quantity, float, or np.ndarray
        The relative velocity of the shock front (in CGS units if no units are provided), in the upstream comoving
        frame.
    gamma: float, optional
        The adiabatic index of the fluid. Default is 5/3 (ideal monatomic gas).
    mu: float, optional
        Mean molecular weight in units of the proton mass. A typical value for fully ionized,
        solar-abundance gas is ~0.61. Default is 0.61.

    Returns
    -------
    ~astropy.units.Quantity
        The post-shock temperature immediately behind the shock front.
    """
    # Handle units
    if isinstance(shock_velocity, u.Quantity):
        shock_velocity_cgs = shock_velocity.to(u.cm / u.s).value
    else:
        shock_velocity_cgs = shock_velocity  # Assume CGS

    T_cgs = _compute_s_c_shock_temperature_cgs(
        shock_velocity=shock_velocity_cgs,
        gamma=gamma,
        mu=mu,
    )

    return T_cgs * u.K


def compute_strong_shock_velocity(
    shock_velocity,
    gamma: float = 5 / 3,
):
    r"""
    Compute the downstream (post-shock) fluid velocity for a strong shock into a cold upstream medium.

    This returns the *post-shock bulk velocity in the lab frame* under the canonical assumption that the
    upstream medium is at rest in the lab frame.

    For a strong shock, the density jump is :math:`R = (\gamma+1)/(\gamma-1)`. Mass flux conservation gives
    the downstream speed in the *shock frame* as :math:`u_2 = u_1/R`. Transforming back to the lab frame
    (upstream at rest, shock speed :math:`v_{\rm sh}`) yields

    .. math::

        v_2 = v_{\rm sh} - \frac{v_{\rm sh}}{R} = v_{\rm sh}\left(\frac{R-1}{R}\right)
            = \frac{2}{\gamma+1} v_{\rm sh}.

    Parameters
    ----------
    shock_velocity: ~astropy.units.Quantity, float, or np.ndarray
        The relative velocity of the shock front (in CGS units if no units are provided). This should be the velocity
        of the shock front in the comoving frame of the upstream medium. If the upstream is at rest, this is the
        lab-frame shock speed.
    gamma: float, optional
        The adiabatic index of the fluid.

    Returns
    -------
    ~astropy.units.Quantity
        The downstream bulk velocity in the lab frame (assuming upstream is at rest).
    """
    if isinstance(shock_velocity, u.Quantity):
        shock_velocity_cgs = shock_velocity.to(u.cm / u.s).value
    else:
        shock_velocity_cgs = shock_velocity  # Assume CGS

    v2_cgs = _compute_s_shock_velocity_cgs(
        shock_velocity=shock_velocity_cgs,
        gamma=gamma,
    )

    return v2_cgs * (u.cm / u.s)


def compute_strong_shock_density(
    upstream_density,
    gamma: float = 5 / 3,
):
    r"""
    Compute the downstream (post-shock) density for a strong shock into a cold upstream medium.

    For a strong shock, the Rankine-Hugoniot density jump is

    .. math::

        \rho_2 = R\rho_1, \quad R = \frac{\gamma+1}{\gamma-1}.

    Parameters
    ----------
    upstream_density: ~astropy.units.Quantity, float, or np.ndarray
        The density of the unshocked medium ahead of the shock (in CGS units if no units are provided).
    gamma: float, optional
        The adiabatic index of the fluid.

    Returns
    -------
    ~astropy.units.Quantity
        The downstream density immediately behind the shock front.
    """
    if isinstance(upstream_density, u.Quantity):
        upstream_density_cgs = upstream_density.to(u.g / u.cm**3).value
    else:
        upstream_density_cgs = upstream_density  # Assume CGS

    rho2_cgs = _compute_s_density_cgs(
        upstream_density=upstream_density_cgs,
        gamma=gamma,
    )

    return rho2_cgs * (u.g / u.cm**3)
