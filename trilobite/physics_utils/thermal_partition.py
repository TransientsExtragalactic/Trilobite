r"""
Partition of the post-shock thermal energy among ions, electrons and the magnetic field.

The shock jump conditions fix the post-shock thermal (internal) energy density :math:`u_{\rm th}`, which excludes rest
mass. How it is shared between species is microphysics. Trilobite splits it into disjoint parts,

.. math::

    u_{\rm th} = u_i + u_{e,\rm th} + u_{e,\rm nt} + u_B,
    \qquad u_B = \epsilon_B u_{\rm th},
    \quad u_{e,\rm nt} = \epsilon_E u_{\rm th},
    \quad u_{e,\rm th} = \epsilon_T u_i,

so that :math:`\epsilon_B` and :math:`\epsilon_E` keep their usual meaning as fractions of :math:`u_{\rm th}`, and
:math:`\epsilon_T` is the ratio of the thermal-electron to the ion energy. Then

.. math::

    u_i = \frac{1 - \epsilon_B - \epsilon_E}{1 + \epsilon_T}\, u_{\rm th},
    \qquad u_{e,\rm th} = \epsilon_T\, u_i .

Each thermal species follows a Maxwell-Juettner distribution at its own temperature, so its energy per particle fixes
its dimensionless temperature :math:`\Theta_s = k_B T_s / (m_s c^2)` through
:math:`\langle \gamma - 1 \rangle(\Theta_s) = u_s / (n_s m_s c^2)`.
"""

from typing import Union

import numpy as np
from astropy import units as u

from trilobite._typing import _ArrayLike, _ScalarLike, _UnitBearingArrayLike
from trilobite.physics_utils.constants import c_cgs, m_e_cgs, m_p_cgs
from trilobite.physics_utils.maxwell_juettner import (
    _as_cgs,
    _as_dimensionless,
    _check,
    _opt_mj_exact_theta_from_kinetic_energy,
)

__all__ = [
    "compute_thermal_electron_energy_fraction",
    "compute_ion_electron_theta",
]


# ============================================================================ #
# Backends                                                                     #
# ============================================================================ #
def _opt_ion_energy_fraction(epsilon_T, epsilon_B, epsilon_E):
    return (1.0 - epsilon_B - epsilon_E) / (1.0 + epsilon_T)


def _opt_thermal_electron_energy_fraction(epsilon_T, epsilon_B, epsilon_E):
    return epsilon_T * _opt_ion_energy_fraction(epsilon_T, epsilon_B, epsilon_E)


def _opt_ion_electron_theta(u_th, rho, mu_e, epsilon_T, epsilon_B, epsilon_E):
    u_i = _opt_ion_energy_fraction(epsilon_T, epsilon_B, epsilon_E) * u_th
    u_e = epsilon_T * u_i
    # The ions carry the rest-mass density, n_i m_i = rho. The electrons have n_e = rho / (mu_e m_p).
    xi_i = u_i / (rho * c_cgs**2)
    xi_e = u_e * mu_e * m_p_cgs / (rho * m_e_cgs * c_cgs**2)
    return _opt_mj_exact_theta_from_kinetic_energy(xi_i), _opt_mj_exact_theta_from_kinetic_energy(xi_e)


# ============================================================================ #
# Validation                                                                   #
# ============================================================================ #
def _validate_partition(epsilon_T, epsilon_B, epsilon_E, allow_zero_T):
    epsilon_T = _check(_as_dimensionless(epsilon_T, "epsilon_T"), "epsilon_T", allow_zero=allow_zero_T)
    epsilon_B = _check(_as_dimensionless(epsilon_B, "epsilon_B"), "epsilon_B", allow_zero=True)
    epsilon_E = _check(_as_dimensionless(epsilon_E, "epsilon_E"), "epsilon_E", allow_zero=True)
    if not np.all(epsilon_B + epsilon_E < 1.0):
        raise ValueError("`epsilon_B + epsilon_E` must be less than 1: the ions carry the remaining energy.")
    return epsilon_T, epsilon_B, epsilon_E


# ============================================================================ #
# Public interface                                                             #
# ============================================================================ #
def compute_thermal_electron_energy_fraction(
    epsilon_T: Union[_ArrayLike, _ScalarLike],
    epsilon_B: Union[_ArrayLike, _ScalarLike] = 0.0,
    epsilon_E: Union[_ArrayLike, _ScalarLike] = 0.0,
) -> Union[_ScalarLike, np.ndarray]:
    r"""
    Compute the fraction of the thermal energy density carried by the thermal electrons.

    With :math:`u_{e,\rm th} = \epsilon_T u_i` and the magnetic and non-thermal electron energies removed first
    (see :mod:`trilobite.physics_utils.thermal_partition`),

    .. math::

        \frac{u_{e,\rm th}}{u_{\rm th}} = \frac{\epsilon_T\,(1 - \epsilon_B - \epsilon_E)}{1 + \epsilon_T}.

    Parameters
    ----------
    epsilon_T : float or array-like
        Ratio of the thermal-electron energy to the ion energy, :math:`u_{e,\rm th} / u_i`. Must be non-negative.
    epsilon_B : float or array-like, optional
        Fraction of the thermal energy density in the magnetic field. Default is ``0``.
    epsilon_E : float or array-like, optional
        Fraction of the thermal energy density in the kinetic energy of non-thermal electrons. Default is ``0``.

    Returns
    -------
    float or ~numpy.ndarray
        :math:`u_{e,\rm th} / u_{\rm th}`.

    Raises
    ------
    ValueError
        If an input is negative or non-finite, or if ``epsilon_B + epsilon_E >= 1``.
    """
    return _opt_thermal_electron_energy_fraction(*_validate_partition(epsilon_T, epsilon_B, epsilon_E, True))


def compute_ion_electron_theta(
    u_th: _UnitBearingArrayLike,
    rho: _UnitBearingArrayLike,
    mu_e: Union[_ArrayLike, _ScalarLike],
    epsilon_T: Union[_ArrayLike, _ScalarLike],
    epsilon_B: Union[_ArrayLike, _ScalarLike] = 0.0,
    epsilon_E: Union[_ArrayLike, _ScalarLike] = 0.0,
) -> tuple[Union[_ScalarLike, np.ndarray], Union[_ScalarLike, np.ndarray]]:
    r"""
    Compute the dimensionless temperatures of the post-shock ions and thermal electrons.

    The thermal energy density ``u_th`` is split as described in :mod:`trilobite.physics_utils.thermal_partition`.
    Each species then follows a Maxwell-Juettner distribution whose energy per particle fixes its temperature,
    :math:`\langle \gamma - 1 \rangle(\Theta_s) = \xi_s`, with

    .. math::

        \xi_i = \frac{u_i}{\rho c^2},
        \qquad
        \xi_e = \frac{u_{e,\rm th}\,\mu_e m_p}{\rho\, m_e c^2}.

    Here :math:`\rho = n_i m_i = n_e \mu_e m_p` is the proper rest-mass density, so the ion mass :math:`m_i` is the
    total mass per ion, :math:`\rho / n_i = \mu_i m_p`. The mean molecular weight per ion :math:`\mu_i` cancels:
    :math:`\Theta_i = k_B T_i / (\mu_i m_p c^2)`, and the ion temperature is :math:`T_i = \Theta_i \mu_i m_p c^2 / k_B`.
    The inversion of :math:`\langle \gamma - 1 \rangle` is exact; see
    :func:`~trilobite.physics_utils.maxwell_juettner.compute_mj_theta_from_kinetic_energy`.

    Parameters
    ----------
    u_th : float or ~astropy.units.Quantity
        Post-shock thermal (internal) energy density, excluding rest mass. Bare floats are in
        :math:`\mathrm{erg\,cm^{-3}}`.
    rho : float or ~astropy.units.Quantity
        Post-shock proper rest-mass density. Bare floats are in :math:`\mathrm{g\,cm^{-3}}`.
    mu_e : float or array-like
        Mean molecular weight per electron, :math:`\rho / (n_e m_p)`; see
        :func:`~trilobite.physics_utils.composition.compute_mean_molecular_weight_per_electron`.
    epsilon_T : float or array-like
        Ratio of the thermal-electron to the ion energy, :math:`u_{e,\rm th} / u_i`. Must be positive.
    epsilon_B : float or array-like, optional
        Fraction of the thermal energy density in the magnetic field. Default is ``0``.
    epsilon_E : float or array-like, optional
        Fraction of the thermal energy density in the kinetic energy of non-thermal electrons. Default is ``0``.

    Returns
    -------
    Theta_i : float or ~numpy.ndarray
        Dimensionless ion temperature :math:`\Theta_i = k_B T_i / (\mu_i m_p c^2)`.
    Theta_e : float or ~numpy.ndarray
        Dimensionless thermal-electron temperature :math:`\Theta_e = k_B T_e / (m_e c^2)`.

    Raises
    ------
    ValueError
        If an input is non-positive or non-finite, or if ``epsilon_B + epsilon_E >= 1``.

    Notes
    -----
    The jump conditions are solved with a single adiabatic index, whereas a mixture of non-relativistic ions and
    partly relativistic electrons has an effective index that varies with :math:`\epsilon_T` and the shock speed. The
    resulting change in :math:`u_{\rm th}` is small and is not corrected for.

    Examples
    --------
    A :math:`0.1c` strong shock into hydrogen gas (:math:`\mu_e = 1`) gives ion and electron temperatures:

    >>> from trilobite.physics_utils.thermal_partition import (
    ...     compute_ion_electron_theta,
    ... )
    >>> rho = 1e-20  # g / cm^3
    >>> u_th = (
    ...     9 / 8 * rho * (0.1 * 2.998e10) ** 2
    ... )  # erg / cm^3
    >>> Theta_i, Theta_e = compute_ion_electron_theta(
    ...     u_th, rho, mu_e=1.0, epsilon_T=0.1
    ... )
    >>> bool(Theta_e > Theta_i)
    True
    """
    u_th = _as_cgs(u_th, u.erg / u.cm**3, "u_th")
    rho = _as_cgs(rho, u.g / u.cm**3, "rho")
    mu_e = _check(_as_dimensionless(mu_e, "mu_e"), "mu_e")
    epsilon_T, epsilon_B, epsilon_E = _validate_partition(epsilon_T, epsilon_B, epsilon_E, False)
    return _opt_ion_electron_theta(u_th, rho, mu_e, epsilon_T, epsilon_B, epsilon_E)
