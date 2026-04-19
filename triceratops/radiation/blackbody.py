"""
Planck (blackbody) radiation functions.

Provides both high-performance CGS float implementations for use inside
MCMC hot loops and unit-aware public wrappers for interactive use.

The module follows the two-level API convention used throughout Triceratops:

- ``_*_cgs`` functions accept plain NumPy arrays / Python floats in CGS and
  return the same.  They carry no unit overhead and are safe to call inside
  MCMC loops.
- Public wrappers (no underscore prefix) accept :class:`~astropy.units.Quantity`
  objects and return a :class:`~astropy.units.Quantity` with the correct unit.
"""

import numpy as np
from astropy import units as u

from .constants import c_cgs, h_cgs, kB_cgs, sigma_sb_cgs

__all__ = [
    "planck_fnu",
    "planck_flambda",
    "wien_peak_frequency",
    "wien_peak_wavelength",
    "stefan_boltzmann_flux",
]

# ---------------------------------------------------------------------------
# Pre-computed CGS coefficients (extracted once for MCMC performance)
# ---------------------------------------------------------------------------
#: 2h / c²  [erg s³ cm⁻²]
_PLANCK_FNU_PREFACTOR_CGS: float = 2.0 * h_cgs / c_cgs**2

#: 2h c²  [erg cm² s⁻¹]
_PLANCK_FLAMBDA_PREFACTOR_CGS: float = 2.0 * h_cgs * c_cgs**2

#: h / k_B  [K s]
_H_OVER_KB_CGS: float = h_cgs / kB_cgs

#: h c / k_B  [K cm]
_HC_OVER_KB_CGS: float = h_cgs * c_cgs / kB_cgs

#: Wien displacement constant for frequency: ν_peak = _WIEN_B_FREQ * T
#: Value: x_peak * k_B / h  where x_peak ≈ 2.821439 solves  x = 3(1 − exp(−x))
_WIEN_B_FREQ_CGS: float = 2.821439372122079 * kB_cgs / h_cgs  # Hz / K

#: Wien displacement constant for wavelength: λ_peak = _WIEN_B_WAV / T
#: b = h c / (x_peak k_B)  in CGS (cm K)
_WIEN_B_WAV_CGS: float = _HC_OVER_KB_CGS / 4.965114231744276  # cm K


# ---------------------------------------------------------------------------
# Low-level, unit-free implementations
# ---------------------------------------------------------------------------
def _planck_fnu_cgs(nu_hz: np.ndarray, T_K: float) -> np.ndarray:
    r"""
    Planck spectral radiance :math:`B_\nu` per unit frequency in CGS.

    .. math::

        B_\nu(\nu, T) = \frac{2 h \nu^3}{c^2}
                        \frac{1}{e^{h\nu / k_{\rm B} T} - 1}

    Parameters
    ----------
    nu_hz : numpy.ndarray
        Frequency grid in Hz. Plain NumPy array; no units attached.
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    numpy.ndarray
        Spectral radiance :math:`B_\nu` in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.

    Notes
    -----
    Uses ``numpy.expm1`` for numerical stability at low frequencies.
    Non-finite values (arising from the :math:`T \to 0` boundary) are
    silently replaced with 0.

    For a unit-aware wrapper, see :func:`planck_fnu`.
    """
    x = _H_OVER_KB_CGS * nu_hz / T_K
    result = _PLANCK_FNU_PREFACTOR_CGS * nu_hz**3 / np.expm1(x)
    return np.where(np.isfinite(result), result, 0.0)


def _planck_flambda_cgs(lam_cm: np.ndarray, T_K: float) -> np.ndarray:
    r"""
    Planck spectral radiance :math:`B_\lambda` per unit wavelength in CGS.

    .. math::

        B_\lambda(\lambda, T) = \frac{2 h c^2}{\lambda^5}
                                \frac{1}{e^{h c / \lambda k_{\rm B} T} - 1}

    Parameters
    ----------
    lam_cm : numpy.ndarray
        Wavelength grid in cm. Plain NumPy array; no units attached.
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    numpy.ndarray
        Spectral radiance :math:`B_\lambda` in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-3}\,sr^{-1}}`.

    Notes
    -----
    Uses ``numpy.expm1`` for numerical stability. Non-finite values are
    silently replaced with 0.

    For a unit-aware wrapper, see :func:`planck_flambda`.
    """
    x = _HC_OVER_KB_CGS / (lam_cm * T_K)
    result = _PLANCK_FLAMBDA_PREFACTOR_CGS / (lam_cm**5 * np.expm1(x))
    return np.where(np.isfinite(result), result, 0.0)


def _wien_peak_frequency_cgs(T_K: float) -> float:
    r"""
    Wien peak frequency in CGS.

    .. math::

        \nu_{\rm peak} = \frac{x_{\rm peak}\,k_{\rm B}}{h}\,T,
        \qquad x_{\rm peak} \approx 2.8214

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Peak frequency :math:`\nu_{\rm peak}` in Hz.
    """
    return _WIEN_B_FREQ_CGS * T_K


def _wien_peak_wavelength_cgs(T_K: float) -> float:
    r"""
    Wien peak wavelength in CGS.

    .. math::

        \lambda_{\rm peak} = \frac{b}{T},
        \qquad b \approx 0.2898\;\mathrm{cm\,K}

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Peak wavelength :math:`\lambda_{\rm peak}` in cm.
    """
    return _WIEN_B_WAV_CGS / T_K


def _stefan_boltzmann_flux_cgs(T_K: float) -> float:
    r"""
    Bolometric radiative flux from a blackbody surface in CGS.

    Integrates the Planck function over all frequencies and over the
    outward hemisphere:

    .. math::

        F = \sigma_{\rm SB}\,T^4

    Parameters
    ----------
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Bolometric flux :math:`F` in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`.
    """
    return sigma_sb_cgs * T_K**4


# ---------------------------------------------------------------------------
# Public, unit-aware wrappers
# ---------------------------------------------------------------------------
def planck_fnu(nu: "u.Quantity", T: "u.Quantity") -> u.Quantity:
    r"""
    Planck spectral radiance :math:`B_\nu(\nu, T)` per unit frequency.

    Unit-aware wrapper around :func:`_planck_fnu_cgs`. Accepts frequency
    and temperature as :class:`~astropy.units.Quantity` objects (or plain
    floats assumed to be in CGS) and returns the result with the appropriate
    CGS unit.

    Parameters
    ----------
    nu : astropy.units.Quantity or float
        Frequency. :class:`~astropy.units.Quantity` is converted to Hz;
        a plain float is assumed to already be in Hz.
    T : astropy.units.Quantity or float
        Temperature. :class:`~astropy.units.Quantity` is converted to K;
        a plain float is assumed to already be in K.

    Returns
    -------
    astropy.units.Quantity
        Spectral radiance :math:`B_\nu` in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from triceratops.radiation.blackbody import (
            planck_fnu,
        )

        B = planck_fnu(5e14 * u.Hz, 6000.0 * u.K)
        print(B.unit)  # erg / (Hz s sr cm2)
    """
    nu_hz = nu.to(u.Hz).value if isinstance(nu, u.Quantity) else float(nu)
    T_K = T.to(u.K).value if isinstance(T, u.Quantity) else float(T)
    return _planck_fnu_cgs(nu_hz, T_K) * u.Unit("erg / (s cm**2 Hz sr)")


def planck_flambda(lam: "u.Quantity", T: "u.Quantity") -> u.Quantity:
    r"""
    Planck spectral radiance :math:`B_\lambda(\lambda, T)` per unit wavelength.

    Unit-aware wrapper around :func:`_planck_flambda_cgs`.

    Parameters
    ----------
    lam : astropy.units.Quantity or float
        Wavelength. :class:`~astropy.units.Quantity` is converted to cm;
        a plain float is assumed to already be in cm.
    T : astropy.units.Quantity or float
        Temperature. :class:`~astropy.units.Quantity` is converted to K;
        a plain float is assumed to already be in K.

    Returns
    -------
    astropy.units.Quantity
        Spectral radiance :math:`B_\lambda` in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-3}\,sr^{-1}}`.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from triceratops.radiation.blackbody import (
            planck_flambda,
        )

        B = planck_flambda(500e-7 * u.cm, 6000.0 * u.K)
        print(B.unit)  # erg / (s cm3 sr)
    """
    lam_cm = lam.to(u.cm).value if isinstance(lam, u.Quantity) else float(lam)
    T_K = T.to(u.K).value if isinstance(T, u.Quantity) else float(T)
    return _planck_flambda_cgs(lam_cm, T_K) * u.Unit("erg / (s cm**3 sr)")


def wien_peak_frequency(T: "u.Quantity") -> u.Quantity:
    r"""
    Wien peak frequency of a blackbody spectrum.

    Returns the frequency :math:`\nu_{\rm peak}` at which the Planck
    function :math:`B_\nu` is maximised:

    .. math::

        \nu_{\rm peak} = \frac{x_{\rm peak}\,k_{\rm B}}{h}\,T,
        \qquad x_{\rm peak} \approx 2.8214

    Parameters
    ----------
    T : astropy.units.Quantity or float
        Temperature. :class:`~astropy.units.Quantity` is converted to K;
        a plain float is assumed to already be in K.

    Returns
    -------
    astropy.units.Quantity
        Peak frequency :math:`\nu_{\rm peak}` in Hz.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from triceratops.radiation.blackbody import (
            wien_peak_frequency,
        )

        nu_peak = wien_peak_frequency(6000.0 * u.K)
        print(nu_peak.to(u.THz))  # ~352 THz
    """
    T_K = T.to(u.K).value if isinstance(T, u.Quantity) else float(T)
    return _wien_peak_frequency_cgs(T_K) * u.Hz


def wien_peak_wavelength(T: "u.Quantity") -> u.Quantity:
    r"""
    Wien peak wavelength of a blackbody spectrum.

    Returns the wavelength :math:`\lambda_{\rm peak}` at which the Planck
    function :math:`B_\lambda` is maximised:

    .. math::

        \lambda_{\rm peak} = \frac{b}{T},
        \qquad b \approx 0.2898\;\mathrm{cm\,K}

    Parameters
    ----------
    T : astropy.units.Quantity or float
        Temperature. :class:`~astropy.units.Quantity` is converted to K;
        a plain float is assumed to already be in K.

    Returns
    -------
    astropy.units.Quantity
        Peak wavelength :math:`\lambda_{\rm peak}` in cm.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from triceratops.radiation.blackbody import (
            wien_peak_wavelength,
        )

        lam_peak = wien_peak_wavelength(6000.0 * u.K)
        print(lam_peak.to(u.nm))  # ~483 nm
    """
    T_K = T.to(u.K).value if isinstance(T, u.Quantity) else float(T)
    return _wien_peak_wavelength_cgs(T_K) * u.cm


def stefan_boltzmann_flux(T: "u.Quantity") -> u.Quantity:
    r"""
    Bolometric radiative flux from a blackbody surface.

    Integrates the Planck function over all frequencies and the outward
    hemisphere:

    .. math::

        F = \sigma_{\rm SB}\,T^4

    Parameters
    ----------
    T : astropy.units.Quantity or float
        Temperature. :class:`~astropy.units.Quantity` is converted to K;
        a plain float is assumed to already be in K.

    Returns
    -------
    astropy.units.Quantity
        Bolometric flux :math:`F` in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from triceratops.radiation.blackbody import (
            stefan_boltzmann_flux,
        )

        F = stefan_boltzmann_flux(1e4 * u.K)
        print(F.to(u.L_sun / u.pc**2))
    """
    T_K = T.to(u.K).value if isinstance(T, u.Quantity) else float(T)
    return _stefan_boltzmann_flux_cgs(T_K) * u.Unit("erg / (s cm**2)")
