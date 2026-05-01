"""
Foundational components of synchrotron modeling for triceratops.

This module handles core synchrotron functionality which is not specialized either to SED modeling
or to specific distribution functions. This includes computations of relevant synchrotron frequencies,
kernels, and other low-level building blocks that are used across the codebase.

.. seealso::

    :mod:`~triceratops.radiation.synchrotron.SEDs`: SEDs for synchrotron emitting regions.

    :mod:`~triceratops.radiation.synchrotron.microphysics`: Microphysical distribution functions for
    synchrotron-emitting electrons.

    :mod:`~triceratops.radiation.synchrotron.cooling`: Synchrotron cooling calculations and timescales.

    :ref:`synchrotron_overview`: An overview of the synchrotron modeling framework in triceratops, including how
    the various components fit together.

    :ref:`synchrotron_seds`: Details on how synchrotron SEDs are computed,
    including the numerical integration methods and kernel approximations used.
"""

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from astropy import constants
from astropy import units as u
from scipy.integrate import quad
from scipy.special import kv

from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    pass

# NumPy compatibility: np.trapezoid added in 2.0, np.trapz removed in 2.0.
# The default arg to getattr is evaluated eagerly, so nest the fallback.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ============================================ #
# Frequency Calculations for Synchrotron       #
# ============================================ #

# --- CGS CONSTANTS --- #
# These are hard coded so we don't need to do unit conversions repeatedly.
_gyrofrequency_coefficient_cgs = (constants.e.esu / (constants.m_e * constants.c)).cgs.value
_log_gyrofrequency_coefficient_cgs = np.log(_gyrofrequency_coefficient_cgs)

# Prefactor for single-electron synchrotron power: sqrt(3) e^3 / (m_e c^2).
# Units: erg / G (i.e. erg s^{-1} Hz^{-1} per Gauss).  Multiply by B [G] and
# optionally sin(alpha) to obtain P(nu) [erg s^{-1} Hz^{-1}].
_sqrt3_e3_over_mec2_cgs = np.sqrt(3) * constants.e.esu.value**3 / (constants.m_e.cgs.value * constants.c.cgs.value**2)
_log_sqrt3_e3_over_mec2_cgs = np.log(_sqrt3_e3_over_mec2_cgs)


# --- Low-Level API --- #
def _optimized_compute_nu_gyro(
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray],
):
    r"""
    Compute the synchrotron gyrofrequency (CGS, optimized).

    Parameters
    ----------
    gamma : float or ~numpy.ndarray
        Electron Lorentz factor.

    B : float or ~numpy.ndarray
        Magnetic field strength in Gauss.

    Returns
    -------
    nu_g : float or ~numpy.ndarray
        Synchrotron gyrofrequency in Hz (CGS-equivalent).

    Notes
    -----
    Implements the gyrofrequency formula:

    .. math::

        \nu_g = \frac{e B}{m_e c \gamma}

    No unit validation is performed.
    """
    return _gyrofrequency_coefficient_cgs * B / gamma


def _opt_compute_log_nu_gyro(
    log_gamma: Union[float, np.ndarray],
    log_B: Union[float, np.ndarray],
):
    r"""
    Compute the logarithm of the synchrotron gyrofrequency (CGS, optimized).

    Parameters
    ----------
    log_gamma : float or ~numpy.ndarray
        Natural logarithm of the electron Lorentz factor.
    log_B : float or ~numpy.ndarray
        Natural logarithm of the magnetic field strength in Gauss.

    Returns
    -------
    log_nu_g : float or ~numpy.ndarray
        Natural logarithm of the synchrotron gyrofrequency in Hz.

    Notes
    -----
    Implements

    .. math::

        \nu_g = \frac{e B}{m_e c \gamma}

    in logarithmic form.
    """
    return _log_gyrofrequency_coefficient_cgs + log_B - log_gamma


def _optimized_compute_nu_critical(
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray],
    sin_alpha: Union[float, np.ndarray] = 1.0,
):
    r"""
    Compute the synchrotron critical frequency (CGS, optimized).

    Parameters
    ----------
    gamma : float or ~numpy.ndarray
        Electron Lorentz factor.

    B : float or ~numpy.ndarray
        Magnetic field strength in Gauss.
    sin_alpha : float or ~numpy.ndarray
        Sine of the pitch angle. Default is 1.0 (i.e., alpha
        = pi/2).

    Returns
    -------
    nu_critical : float or ~numpy.ndarray
        Synchrotron critical frequency in Hz (CGS-equivalent).

    Notes
    -----
    Implements the critical frequency formula:

    .. math::

        \nu_{critical} = \frac{3 e B \sin \alpha \gamma^2}{4 \pi m_e c}

    No unit validation is performed.
    """
    return (3 / (4 * np.pi)) * _gyrofrequency_coefficient_cgs * B * sin_alpha * gamma**2


def _opt_compute_log_nu_critical(
    log_gamma: Union[float, np.ndarray],
    log_B: Union[float, np.ndarray],
    sin_alpha: Union[float, np.ndarray] = 1.0,
):
    r"""
    Compute the logarithm of the synchrotron critical frequency (CGS, optimized).

    Parameters
    ----------
    log_gamma : float or ~numpy.ndarray
        Natural logarithm of the electron Lorentz factor.
    log_B : float or ~numpy.ndarray
        Natural logarithm of the magnetic field strength in Gauss.
    sin_alpha : float or ~numpy.ndarray
        Sine of the pitch angle.

    Returns
    -------
    log_nu_critical : float or ~numpy.ndarray
        Natural logarithm of the synchrotron critical frequency in Hz.
    """
    return np.log(3 / (4 * np.pi)) + _log_gyrofrequency_coefficient_cgs + log_B + np.log(sin_alpha) + (2.0 * log_gamma)


def _opt_compute_synch_frequency(
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray],
    sin_alpha: Union[float, np.ndarray] = 1.0,
    pitch_average: bool = True,
):
    r"""
    Compute the synchrotron injection frequency (CGS, optimized).

    Parameters
    ----------
    gamma : float or ~numpy.ndarray
        Electron Lorentz factor.

    B : float or ~numpy.ndarray
        Magnetic field strength in Gauss.
    sin_alpha : float or ~numpy.ndarray
        Sine of the pitch angle. Default is 1.0 (i.e., alpha
        = pi/2). This is only used if ``pitch_average`` is False.
    pitch_average : bool
        Whether to use pitch-angle averaged value. Default is True.

    Returns
    -------
    nu_injection : float or ~numpy.ndarray
        Synchrotron injection frequency in Hz (CGS-equivalent).

    Notes
    -----
    Implements the injection frequency formula:

    .. math::

        \nu_m = \gamma^2 \frac{3 eB\sin\alpha}{4 \pi m_e c},

    where, in the case of pitch-angle averaging, :math:`\langle \sin \alpha \rangle = 2/\pi`:

    .. math::

        \nu_m = \gamma^2 \frac{3 eB}{2 \pi^2 m_e c}

    No unit validation is performed.
    """
    if pitch_average:
        sin_alpha_factor = 2 / np.pi
    else:
        sin_alpha_factor = sin_alpha

    return (3 / (4 * np.pi)) * _gyrofrequency_coefficient_cgs * B * sin_alpha_factor * gamma**2


def _opt_compute_log_synch_frequency(
    log_gamma: Union[float, np.ndarray],
    log_B: Union[float, np.ndarray],
    sin_alpha: Union[float, np.ndarray] = 1.0,
    pitch_average: bool = True,
):
    r"""
    Compute the logarithm of the synchrotron injection frequency (CGS, optimized).

    Parameters
    ----------
    log_gamma : float or ~numpy.ndarray
        Natural logarithm of the electron Lorentz factor.
    log_B : float or ~numpy.ndarray
        Natural logarithm of the magnetic field strength in Gauss.
    sin_alpha : float or ~numpy.ndarray
        Sine of the pitch angle (used only if ``pitch_average=False``).
    pitch_average : bool
        Whether to use pitch-angle averaged value.

    Returns
    -------
    log_nu_injection : float or ~numpy.ndarray
        Natural logarithm of the synchrotron injection frequency in Hz.
    """
    if pitch_average:
        log_sin_alpha_factor = np.log(2 / np.pi)
    else:
        log_sin_alpha_factor = np.log(sin_alpha)

    return np.log(3 / (4 * np.pi)) + _log_gyrofrequency_coefficient_cgs + log_B + log_sin_alpha_factor + 2.0 * log_gamma


def _opt_compute_synch_gamma(
    nu: Union[float, np.ndarray],
    B: Union[float, np.ndarray],
    sin_alpha: Union[float, np.ndarray] = 1.0,
    pitch_average: bool = True,
):
    r"""
    Compute the electron Lorentz factor corresponding to a given synchrotron frequency (CGS, optimized).

    Parameters
    ----------
    nu : float or ~numpy.ndarray
        Synchrotron frequency in Hz.

    B : float or ~numpy.ndarray
        Magnetic field strength in Gauss.

    sin_alpha : float or ~numpy.ndarray
        Sine of the pitch angle. Only used if ``pitch_average=False``.

    pitch_average : bool
        Whether to use pitch-angle averaged value.
        Uses :math:`\left<\sin\alpha\right> = 2/\pi` if True.

    Returns
    -------
    gamma : float or ~numpy.ndarray
        Electron Lorentz factor.

    Notes
    -----
    Inverts the synchrotron characteristic frequency relation:

    .. math::

        \nu = \gamma^2 \frac{3 e B \sin\alpha}{4\pi m_e c}
    """
    if pitch_average:
        sin_alpha_factor = 2 / np.pi
    else:
        sin_alpha_factor = sin_alpha

    return np.sqrt((4 * np.pi / 3) * nu / (_gyrofrequency_coefficient_cgs * B * sin_alpha_factor))


def _opt_compute_single_electron_power(
    nu: Union[float, np.ndarray],
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray],
    sin_alpha: Union[float, np.ndarray] = 1.0,
) -> Union[float, np.ndarray]:
    r"""
    Compute the synchrotron power per unit frequency for a single electron at a fixed pitch angle (CGS, optimized).

    Parameters
    ----------
    nu : float or ~numpy.ndarray
        Frequency in Hz.
    gamma : float or ~numpy.ndarray
        Electron Lorentz factor.
    B : float or ~numpy.ndarray
        Magnetic field strength in Gauss.
    sin_alpha : float or ~numpy.ndarray
        Sine of the pitch angle. Default is ``1.0`` (perpendicular,
        :math:`\alpha = \pi/2`).

    Returns
    -------
    P_nu : float or ~numpy.ndarray
        Power per unit frequency in :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.

    Notes
    -----
    Implements :footcite:p:`RybickiLightman` eq. 6.31c:

    .. math::

        P(\nu,\alpha) = \frac{\sqrt{3}\,e^3 B\sin\alpha}{m_e c^2}
                        F\!\left(\frac{\nu}{\nu_c}\right),

    where :math:`\nu_c = \frac{3\,e\,B\sin\alpha\,\gamma^2}{4\pi\,m_e c}` and
    :math:`F(x) = x\int_x^\infty K_{5/3}(z)\,dz`.
    No unit validation is performed.

    References
    ----------
    .. footbibliography::
    """
    nu_c = _optimized_compute_nu_critical(gamma, B, sin_alpha)
    log_F, _ = _log_first_synchrotron_kernel(np.log(nu / nu_c), derivative=False)
    return _sqrt3_e3_over_mec2_cgs * B * sin_alpha * np.exp(log_F)


def _opt_compute_pa_averaged_single_electron_power(
    nu: Union[float, np.ndarray],
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""
    Compute the pitch-angle-averaged synchrotron power per unit frequency for a single electron (CGS, optimized).

    Parameters
    ----------
    nu : float or ~numpy.ndarray
        Frequency in Hz.
    gamma : float or ~numpy.ndarray
        Electron Lorentz factor.
    B : float or ~numpy.ndarray
        Magnetic field strength in Gauss.

    Returns
    -------
    P_nu_avg : float or ~numpy.ndarray
        Pitch-angle-averaged power per unit frequency in :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.

    Notes
    -----
    For an isotropic pitch-angle distribution, the averaged power is
    :footcite:p:`1986A&A...164L..16C,1988ApJ...334L...5G`:

    .. math::

        \bar{P}(\nu) = \frac{\sqrt{3}\,e^3 B}{m_e c^2}\,
                       \bar{F}\!\left(\frac{\nu}{\nu_{c,\perp}}\right),

    where :math:`\nu_{c,\perp} = \frac{3\,e\,B\,\gamma^2}{4\pi\,m_e c}` is the critical
    frequency at :math:`\sin\alpha = 1`, and :math:`\bar{F}(x)` is the Crusius--Schlickeiser
    pitch-angle-averaged kernel (see :func:`_log_averaged_first_synchrotron_kernel`).
    No unit validation is performed.

    References
    ----------
    .. footbibliography::
    """
    nu_c_perp = _optimized_compute_nu_critical(gamma, B, sin_alpha=1.0)
    log_F_avg, _ = _log_averaged_first_synchrotron_kernel(np.log(nu / nu_c_perp), derivative=False)
    return _sqrt3_e3_over_mec2_cgs * B * np.exp(log_F_avg)


# --- High-Level API --- #
def compute_gyrofrequency(
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray, u.Quantity],
) -> u.Quantity:
    r"""
    Compute the synchrotron gyrofrequency for relativistic electrons.

    The gyrofrequency corresponds to the frequency at which a charged particle
    orbits in a magnetic field. For relativistic electrons, this frequency is
    modified by the Lorentz factor of the electron.

    Parameters
    ----------
    gamma : float or ~numpy.ndarray
        Electron Lorentz factor.

    B : float, ~numpy.ndarray, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.

    Returns
    -------
    nu_g : astropy.units.Quantity
        Synchrotron gyrofrequency in Hz.

    Notes
    -----
    The gyrofrequency for a relativistic electron is given by :footcite:p:`RybickiLightman`

    .. math::

        \nu_g = \frac{e B}{m_e c \gamma}

    This function computes the gyrofrequency associated with
    synchrotron emission from electrons of Lorentz factor ``gamma`` in
    a magnetic field ``B``.

    References
    ----------
    .. footbibliography::
    """
    B = ensure_in_units(B, u.Gauss)

    return _optimized_compute_nu_gyro(gamma, B) * u.Hz


def compute_nu_critical(
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray, u.Quantity],
    alpha: float = np.pi / 2,
) -> u.Quantity:
    r"""
    Compute the synchrotron critical frequency for relativistic electrons.

    The critical frequency corresponds to the frequency at which the synchrotron
    emission from a relativistic electron peaks. This follows the formalism as described in
    :footcite:t:`RybickiLightman`.

    Parameters
    ----------
    gamma : float or ~numpy.ndarray
        Electron Lorentz factor.

    B : float, ~numpy.ndarray, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.
    alpha: float
        Pitch angle in radians. Default is ``pi/2``.

    Returns
    -------
    nu_critical : astropy.units.Quantity
        Synchrotron critical frequency in Hz.

    Notes
    -----
    The critical frequency for a relativistic electron is given by :footcite:p:`RybickiLightman`

    .. math::

        \nu_{critical} = \frac{3 e B\sin \alpha \gamma^2}{4 \pi m_e c}

    This function computes the critical frequency associated with
    synchrotron emission from electrons of Lorentz factor ``gamma`` in
    a magnetic field ``B``.

    References
    ----------
    .. footbibliography::
    """
    B = ensure_in_units(B, u.Gauss)

    # compute sin(alpha) factor
    sin_alpha = np.sin(alpha)

    return _optimized_compute_nu_critical(gamma, B, sin_alpha) * u.Hz


def compute_synchrotron_frequency(
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray, u.Quantity],
    alpha: float = np.pi / 2,
    pitch_average: bool = True,
) -> u.Quantity:
    r"""
    Compute the synchrotron characteristic (injection / critical) frequency for relativistic electrons.

    This function provides a high-level, unit-safe wrapper around the optimized
    CGS implementation, with optional pitch-angle averaging.

    Parameters
    ----------
    gamma : float or ~numpy.ndarray
        Electron Lorentz factor.

    B : float, ~numpy.ndarray, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.

    alpha : float
        Pitch angle in radians. Only used if ``pitch_average=False``.
        Default is ``pi/2``.

    pitch_average : bool
        Whether to use the pitch-angle averaged value
        :math:`\langle \sin \alpha \rangle = 2/\pi`.
        Default is True.

    Returns
    -------
    nu_synch : astropy.units.Quantity
        Synchrotron characteristic frequency in Hz.

    Notes
    -----
    The synchrotron characteristic frequency is given by
    :footcite:p:`RybickiLightman`

    .. math::

        \nu = \frac{3 e B \sin \alpha \gamma^2}{4 \pi m_e c}

    When pitch-angle averaging is enabled, this becomes

    .. math::

        \nu = \gamma^2 \frac{3 e B}{2 \pi^2 m_e c}

    References
    ----------
    .. footbibliography::
    """
    # Ensure magnetic field is in Gauss
    B = ensure_in_units(B, u.Gauss)

    # Compute sin(alpha) only if needed
    if pitch_average:
        sin_alpha = 1.0  # placeholder; ignored downstream
    else:
        sin_alpha = np.sin(alpha)

    return (
        _opt_compute_synch_frequency(
            gamma=gamma,
            B=B,
            sin_alpha=sin_alpha,
            pitch_average=pitch_average,
        )
        * u.Hz
    )


def compute_synchrotron_gamma(
    nu: Union[float, np.ndarray, u.Quantity],
    B: Union[float, np.ndarray, u.Quantity],
    alpha: float = np.pi / 2,
    pitch_average: bool = True,
) -> u.Quantity:
    r"""
    Compute the electron Lorentz factor corresponding to a given synchrotron characteristic frequency.

    Parameters
    ----------
    nu : float, ~numpy.ndarray, or astropy.units.Quantity
        Synchrotron frequency. Default units are Hz.

    B : float, ~numpy.ndarray, or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.

    alpha : float
        Pitch angle in radians. Only used if ``pitch_average=False``.

    pitch_average : bool
        Whether to use pitch-angle averaged value
        :math:`\left<\sin\alpha\right> = 2/\pi`. Default is True.

    Returns
    -------
    gamma : astropy.units.Quantity
        Electron Lorentz factor (dimensionless).

    Notes
    -----
    This function inverts the synchrotron characteristic frequency relation
    (see :footcite:p:`RybickiLightman`).

    References
    ----------
    .. footbibliography::
    """
    nu = ensure_in_units(nu, u.Hz)
    B = ensure_in_units(B, u.Gauss)

    if pitch_average:
        sin_alpha = 1.0  # ignored downstream
    else:
        sin_alpha = np.sin(alpha)

    return (
        _opt_compute_synch_gamma(
            nu=nu,
            B=B,
            sin_alpha=sin_alpha,
            pitch_average=pitch_average,
        )
        * u.dimensionless_unscaled
    )


def compute_single_electron_power(
    nu: Union[float, np.ndarray, u.Quantity],
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray, u.Quantity],
    alpha: Optional[float] = None,
) -> u.Quantity:
    r"""
    Compute the synchrotron power per unit frequency for a single relativistic electron.

    Parameters
    ----------
    nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Frequency. Bare values are interpreted as Hz.
    gamma : float or ~numpy.ndarray
        Electron Lorentz factor.
    B : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Magnetic field strength. Bare values are interpreted as Gauss.
    alpha : float, optional
        Pitch angle in radians. When ``None`` (default), the isotropic pitch-angle
        average is returned. Otherwise, the power at the specified pitch angle is
        returned.

    Returns
    -------
    P_nu : ~astropy.units.Quantity
        Power per unit frequency in :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.

    Notes
    -----
    For a fixed pitch angle :math:`\alpha`, the formula is :footcite:p:`RybickiLightman`:

    .. math::

        P(\nu,\alpha) = \frac{\sqrt{3}\,e^3 B\sin\alpha}{m_e c^2}
                        F\!\left(\frac{\nu}{\nu_c}\right),

    where :math:`\nu_c = \frac{3\,e\,B\sin\alpha\,\gamma^2}{4\pi\,m_e c}` and
    :math:`F(x) = x\int_x^\infty K_{5/3}(z)\,dz` is the first synchrotron kernel.

    When ``alpha=None``, the result averaged over an isotropic pitch-angle distribution is
    returned instead :footcite:p:`1986A&A...164L..16C,1988ApJ...334L...5G`:

    .. math::

        \bar{P}(\nu) = \frac{\sqrt{3}\,e^3 B}{m_e c^2}\,
                       \bar{F}\!\left(\frac{\nu}{\nu_{c,\perp}}\right),

    where :math:`\nu_{c,\perp} = \frac{3\,e\,B\,\gamma^2}{4\pi\,m_e c}` (the perpendicular
    critical frequency, :math:`\sin\alpha = 1`) and :math:`\bar{F}(x)` is the
    Crusius--Schlickeiser pitch-angle-averaged kernel
    (see :func:`compute_averaged_first_synchrotron_kernel`).

    References
    ----------
    .. footbibliography::
    """
    nu_hz = ensure_in_units(nu, u.Hz)
    B_gauss = ensure_in_units(B, u.Gauss)

    if alpha is None:
        result = _opt_compute_pa_averaged_single_electron_power(nu_hz, gamma, B_gauss)
    else:
        result = _opt_compute_single_electron_power(nu_hz, gamma, B_gauss, np.sin(alpha))

    return result * (u.erg / u.s / u.Hz)


# ============================================ #
# Synchrotron Kernels                          #
# ============================================ #
# These functions implement the synchrotron kernel functions F(x) and G(x) in
# various ways, including direct integration and interpolation-based approximations.
def _log_first_synchrotron_kernel(
    log_x: Union[np.ndarray, float],
    log_x_asymptotic_min: float = -8.0,
    log_x_asymptotic_max: float = 2.0,
    method: str = "exact",
    derivative: bool = True,
) -> tuple[Union[np.ndarray, float], Optional[Union[np.ndarray, float]]]:
    r"""
    Compute :math:`\ln F(x)` for the first synchrotron kernel, numerically stable across all :math:`x`.

    The synchrotron kernel is defined as

    .. math::

        F(x) = x \int_x^\infty K_{5/3}(z)\,dz.

    Working in log-space prevents underflow for :math:`x \gg 1` where :math:`F(x) \to 0`.

    Parameters
    ----------
    log_x : float or ~numpy.ndarray
        Natural log of the dimensionless frequency ratio :math:`x = \nu/\nu_c`.
    log_x_asymptotic_min : float, optional
        :math:`\ln x` below which the low-:math:`x` power-law asymptotic
        :math:`F(x) \approx C\,x^{1/3}` replaces the integral. Only used when
        ``method="exact"``. Default is ``-5.0``.
    log_x_asymptotic_max : float, optional
        :math:`\ln x` above which the high-:math:`x` exponential asymptotic
        :math:`F(x) \approx \sqrt{\pi x/2}\,e^{-x}` replaces the integral. Only used
        when ``method="exact"``. Default is ``2.0``.
    method : {"exact", "lu"}, optional
        Algorithm used to evaluate the interior region.

        ``"exact"``
            Splits the domain into three regions: power-law asymptotic for
            :math:`x \ll 1`, exponential asymptotic for :math:`x \gg 1`, and
            :func:`scipy.integrate.quad` over :math:`K_{5/3}` in the interior.
            Accurate to ``epsrel=1e-10`` but slow for large arrays.
        ``"lu"``
            Liu--Uhm (Wenbin) closed-form approximation applied to all :math:`x`
            without domain splitting. Fast and accurate to a few percent everywhere.

    Returns
    -------
    log_F : float or ~numpy.ndarray
        :math:`\ln F(x)`. Returns a scalar when ``log_x`` is scalar.

    See Also
    --------
    second_synchrotron_kernel : The companion kernel :math:`G(x) = x K_{2/3}(x)`.

    Notes
    -----
    The low-:math:`x` prefactor :math:`C \approx 2.1495` follows from the small-argument
    expansion of :math:`K_{5/3}` (see e.g. :footcite:p:`RybickiLightman`).

    References
    ----------
    .. footbibliography::
    """
    scalar = np.ndim(log_x) == 0
    log_x = np.atleast_1d(np.asarray(log_x, dtype=float))
    log_F = np.empty_like(log_x)
    dlog_F_dlog_x = np.empty_like(log_x) if derivative else None

    if method == "exact":
        lmsk = log_x < log_x_asymptotic_min
        rmsk = log_x > log_x_asymptotic_max
        imsk = ~(lmsk | rmsk)

        # Perform the asymptotic calculation at the lower edge to avoid any heavy
        # computation in the regime where the kernel is well approximated by a power law.
        # We precompute the logarithmic prefactor at super-precision to avoid
        # any loss of accuracy.
        if np.any(lmsk):
            log_F[lmsk] = 0.7652483955208204416206842 + (1.0 / 3.0) * log_x[lmsk]
            if derivative:
                dlog_F_dlog_x[lmsk] = 1.0 / 3.0

        # Now the high-x mask for the exponential cutoff.
        if np.any(rmsk):
            x_r = np.exp(log_x[rmsk])
            log_F[rmsk] = 0.5 * (np.log(np.pi) - np.log(2.0)) + 0.5 * log_x[rmsk] - x_r

            if derivative:
                dlog_F_dlog_x[rmsk] = 0.5 - x_r

        # Now perform the quadrature part of the tabulation. Here we start by computing the
        # raw integral int_x^inf K_5/3(z) dz and then we spin it into both F and d log F/d log x.
        if np.any(imsk):
            # Perform the quadrature.
            x = np.exp(log_x[imsk])
            Integral = np.empty_like(x)
            for i, xi in enumerate(x):
                Integral[i] = quad(lambda z: kv(5 / 3, z), xi, np.inf, epsrel=1e-10)[0]

            # Now generate log_F = log(x) + log(I).
            log_F[imsk] = log_x[imsk] + np.log(Integral)

            # Now the derivative if we are asked for it.
            if derivative:
                dlog_F_dlog_x[imsk] = 1 - (x / Integral) * kv(5 / 3, x)

    elif method == "lu":
        x = np.exp(log_x)

        A = (np.pi * x / 2.0) ** (1.430 / 2.0)
        B = (2.150 * x ** (1.0 / 3.0)) ** (-2.627)

        log_A = np.log1p(A)
        log_B = np.log1p(B)

        log_F = (1.0 / 1.430) * log_A - (1.0 / 2.627) * log_B - x

        if derivative:
            dlog_F_dlog_x = 0.5 * (A / (1.0 + A)) + (1.0 / 3.0) * (B / (1.0 + B)) - x

    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'exact' or 'lu'.")

    return log_F.item() if scalar else log_F, dlog_F_dlog_x.item() if derivative and scalar else dlog_F_dlog_x


def _log_averaged_first_synchrotron_kernel(
    log_x: Union[np.ndarray, float],
    log_x_asymptotic_min: float = -8.0,
    log_x_asymptotic_max: float = 2.0,
    method: str = "exact",
    derivative: bool = True,
) -> tuple[Union[np.ndarray, float], Optional[Union[np.ndarray, float]]]:
    scalar = np.ndim(log_x) == 0
    log_x = np.atleast_1d(np.asarray(log_x, dtype=float))
    log_F = np.empty_like(log_x)
    dlog_F_dlog_x = np.empty_like(log_x) if derivative else None

    if method == "exact":
        lmsk = log_x < log_x_asymptotic_min
        rmsk = log_x > log_x_asymptotic_max
        imsk = ~(lmsk | rmsk)

        # Perform the asymptotic calculation at the lower edge to avoid any heavy
        # computation in the regime where the kernel is well approximated by a power law.
        # We precompute the logarithmic prefactor at super-precision to avoid
        # any loss of accuracy.
        if np.any(lmsk):
            log_F[lmsk] = 0.59245244160808220024599944 + (1.0 / 3.0) * log_x[lmsk]
            if derivative:
                dlog_F_dlog_x[lmsk] = 1.0 / 3.0

        # Now the high-x mask for the exponential cutoff.
        if np.any(rmsk):
            x_r = np.exp(log_x[rmsk])
            eps = 99 / (162 * x_r)
            log_F[rmsk] = (np.log(np.pi) - np.log(2.0)) + np.log1p(-eps) - x_r

            if derivative:
                dlog_F_dlog_x[rmsk] = 0.5 - x_r

        # Now we perform the exact computation using the Bessel functions. This is taken from
        # Wenbin Lu's book equation 8.79.
        if np.any(imsk):
            # Declare the relevant u parameter that gets fed into everything.
            u = np.exp(log_x[imsk]) / 2.0

            # Compute the relevant Bessel function constituents.
            K_1_3 = kv(1 / 3, u)
            K_4_3 = kv(4 / 3, u)
            A = K_1_3 * K_4_3
            B = K_4_3**2 - K_1_3**2
            Q = A - (3 * u / 5) * B

            # Compute R directly.
            R = (2 * u**2) * Q
            log_F[imsk] = np.log(R)

            # If requested, we can now produce dlog_R dlog_x from this.
            if derivative:
                DK_1_3 = -0.5 * (kv(-2 / 3, u) + kv(4 / 3, u))
                DK_4_3 = -0.5 * (kv(1 / 3, u) + kv(7 / 3, u))
                DADU = DK_1_3 * K_4_3 + K_1_3 * DK_4_3
                DBDU = 2 * (K_4_3 * DK_4_3 - K_1_3 * DK_1_3)

                dlog_F_dlog_x[imsk] = 2 + (u / Q) * (DADU - (3 / 5) * B - (3 * u / 5) * DBDU)

    elif method == "lu":
        x = np.exp(log_x)

        A = np.log((np.pi * x**0.9 / 2.0) + 1)
        B = np.log(x**0.9 + 1)
        C = -2.443 * np.log(1.808 * x ** (1 / 3))
        D = -0.0263440

        log_F = A - B - (1 / 2.443) * np.logaddexp(C, D) - x

        if derivative:
            dlog_F_dlog_x = (
                ((0.9 * np.pi * x**0.9 / 2.0) / (1 + (np.pi * x**0.9 / 2.0)))  # term 1
                - (0.9 * x**0.9 / (1 + x**0.9))  # term 2
                + (1 / 3.0) * ((1.808 * x ** (1 / 3)) ** (-2.443) / ((1.808 * x ** (1 / 3)) ** (-2.443) + 0.974))
                - x
            )

    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'exact' or 'lu'.")

    return log_F.item() if scalar else log_F, dlog_F_dlog_x.item() if derivative and scalar else dlog_F_dlog_x


# ============================================ #
# Public Kernel API                            #
# ============================================ #
def compute_first_synchrotron_kernel(
    x: Union[np.ndarray, float],
    method: str = "exact",
) -> Union[np.ndarray, float]:
    r"""
    Evaluate the first synchrotron kernel :math:`F(x)`.

    Parameters
    ----------
    x : float or ~numpy.ndarray
        Dimensionless frequency ratio :math:`x = \nu / \nu_c`. Must be positive.
    method : {"exact", "lu"}, optional
        Evaluation algorithm.

        ``"exact"``
            Bessel-function quadrature in the interior, stitched to the
            power-law asymptotic :math:`F(x) \approx C\,x^{1/3}` at small
            :math:`x` and the exponential asymptotic
            :math:`F(x) \approx \sqrt{\pi x/2}\,e^{-x}` at large :math:`x`.
        ``"lu"``
            Closed-form approximation accurate to a few percent everywhere
            :footcite:p:`lu_2026_18603474`.

        Default is ``"exact"``.

    Returns
    -------
    F : float or ~numpy.ndarray
        :math:`F(x)` at each input point.

    Notes
    -----
    The kernel is defined by

    .. math::

        F(x) = x \int_x^\infty K_{5/3}(z)\,dz,

    where :math:`K_{5/3}` is the modified Bessel function of the second kind.
    It peaks near :math:`x \approx 0.29` and determines the spectral shape of
    single-electron synchrotron emission.

    For performance-critical applications (e.g. inference loops), prefer
    :class:`~triceratops.radiation.synchrotron.SEDs.numerical.NumericalSynchrotronEngine`,
    which pre-tabulates this kernel on a spline grid.

    References
    ----------
    .. footbibliography::
    """
    log_F, _ = _log_first_synchrotron_kernel(np.log(x), method=method, derivative=False)
    return np.exp(log_F)


def compute_averaged_first_synchrotron_kernel(
    x: Union[np.ndarray, float],
    method: str = "exact",
) -> Union[np.ndarray, float]:
    r"""
    Evaluate the pitch-angle-averaged first synchrotron kernel :math:`\bar{F}(x)`.

    Parameters
    ----------
    x : float or ~numpy.ndarray
        Dimensionless frequency ratio :math:`x = \nu / \nu_c`. Must be positive.
    method : {"exact", "lu"}, optional
        Evaluation algorithm.

        ``"exact"``
            Closed-form Bessel-function expression in the interior, stitched to
            power-law and exponential asymptotics at the domain edges.
        ``"lu"``
            Approximation of :footcite:t:`2007A&A...465..695Z`, accurate to
            better than one percent everywhere.

        Default is ``"exact"``.

    Returns
    -------
    F_avg : float or ~numpy.ndarray
        :math:`\bar{F}(x)` at each input point.

    Notes
    -----
    The pitch-angle-averaged kernel applies when electron pitch angles are
    distributed isotropically :footcite:p:`lu_2026_18603474`. It is defined by

    .. math::

        \bar{F}(x) = \int_0^{1} F\!\left(\frac{x}{\xi}\right)
                     \frac{\xi^2}{\sqrt{1-\xi^2}}\, d\xi,

    and admits the closed-form expression
    :footcite:p:`1986A&A...164L..16C,1988ApJ...334L...5G`

    .. math::

        \bar{F}(x) = \frac{x^2}{2}\!\left[
            K_{4/3}(x/2)\,K_{1/3}(x/2)
            - \frac{3x}{10}\bigl(K_{4/3}^2(x/2) - K_{1/3}^2(x/2)\bigr)
        \right].

    References
    ----------
    .. footbibliography::
    """
    log_F, _ = _log_averaged_first_synchrotron_kernel(np.log(x), method=method, derivative=False)
    return np.exp(log_F)
