"""
Planck (blackbody) radiation functions.

Provides both a high-performance CGS float implementation for use inside
MCMC hot loops and a unit-aware public wrapper for interactive use.
"""

import numpy as np
from astropy import units as u

from .constants import c_cgs, h_cgs, kB_cgs

__all__ = ["planck_fnu"]

# ---------------------------------------------------------------------------
# CGS constants (pre-extracted for MCMC performance)
# ---------------------------------------------------------------------------
#: 2h / c² in CGS
_PLANCK_PREFACTOR_CGS: float = 2.0 * h_cgs / c_cgs**2

#: h / k_B in CGS (K·s)
_H_OVER_KB_CGS: float = h_cgs / kB_cgs


# ---------------------------------------------------------------------------
# Low-level, unit-free implementation
# ---------------------------------------------------------------------------
def _planck_fnu_cgs(nu_hz: np.ndarray, T_K: float) -> np.ndarray:
    r"""
    Planck spectral radiance :math:`B_\nu` in CGS units.

    Evaluates the Planck function

    .. math::

        B_\nu(\nu, T) = \frac{2 h \nu^3}{c^2} \frac{1}{e^{h\nu / k_B T} - 1}

    Parameters
    ----------
    nu_hz : numpy.ndarray
        Frequency grid in Hz. Must be a plain NumPy array (no units).
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    numpy.ndarray
        Spectral radiance :math:`B_\nu` in erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹.

    Notes
    -----
    This function uses pre-extracted CGS constants and avoids all unit
    overhead. Use it inside MCMC hot loops. For an interactive, unit-aware
    wrapper, use :func:`planck_fnu`.
    """
    x = _H_OVER_KB_CGS * nu_hz / T_K
    return _PLANCK_PREFACTOR_CGS * nu_hz**3 / np.expm1(x)


# ---------------------------------------------------------------------------
# Public, unit-aware wrapper
# ---------------------------------------------------------------------------
def planck_fnu(nu: "u.Quantity", T: "u.Quantity") -> u.Quantity:
    r"""
    Planck spectral radiance :math:`B_\nu(\nu, T)`.

    Unit-aware wrapper around :func:`_planck_fnu_cgs`. Accepts frequency
    and temperature as :class:`~astropy.units.Quantity` objects and returns
    the result with the appropriate CGS unit.

    Parameters
    ----------
    nu : astropy.units.Quantity
        Frequency. Converted internally to Hz.
    T : astropy.units.Quantity
        Temperature. Converted internally to K.

    Returns
    -------
    astropy.units.Quantity
        Spectral radiance :math:`B_\nu` in erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹.

    Examples
    --------
    >>> from astropy import units as u
    >>> from triceratops.radiation.blackbody import (
    ...     planck_fnu,
    ... )
    >>> B = planck_fnu(5e14 * u.Hz, 6000.0 * u.K)
    >>> B.unit
    Unit("erg / (cm2 Hz s sr)")
    """
    nu_hz = nu.to(u.Hz).value if isinstance(nu, u.Quantity) else float(nu)
    T_K = T.to(u.K).value if isinstance(T, u.Quantity) else float(T)
    return _planck_fnu_cgs(nu_hz, T_K) * u.Unit("erg / (s cm2 Hz sr)")
