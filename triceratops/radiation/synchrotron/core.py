"""
Foundational components of synchrotron modeling for triceratops.

This module handles core synchrotron functionality which is not specialized either to SED modeling
or to specific distribution functions. This includes the single-electron synchrotron power spectrum
and related calculations.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from astropy import constants
from astropy import units as u
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import kv
from tqdm.auto import tqdm

from triceratops.utils.config import triceratops_config
from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from triceratops._typing import _ArrayLike, _UnitBearingArrayLike

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


# --- Low-Level API --- #
def _optimized_compute_nu_gyro(
    gamma: Union[float, np.ndarray],
    B: Union[float, np.ndarray],
):
    r"""
    Compute the synchrotron gyrofrequency (CGS, optimized).

    Parameters
    ----------
    gamma : float or array-like
        Electron Lorentz factor.

    B : float or array-like
        Magnetic field strength in Gauss.

    Returns
    -------
    nu_g : float or array-like
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
    log_gamma : float or array-like
        Natural logarithm of the electron Lorentz factor.
    log_B : float or array-like
        Natural logarithm of the magnetic field strength in Gauss.

    Returns
    -------
    log_nu_g : float or array-like
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
    gamma : float or array-like
        Electron Lorentz factor.

    B : float or array-like
        Magnetic field strength in Gauss.
    sin_alpha : float or array-like
        Sine of the pitch angle. Default is 1.0 (i.e., alpha
        = pi/2).

    Returns
    -------
    nu_critical : float or array-like
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
    log_gamma : float or array-like
        Natural logarithm of the electron Lorentz factor.
    log_B : float or array-like
        Natural logarithm of the magnetic field strength in Gauss.
    sin_alpha : float or array-like
        Sine of the pitch angle.

    Returns
    -------
    log_nu_critical : float or array-like
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
    gamma : float or array-like
        Electron Lorentz factor.

    B : float or array-like
        Magnetic field strength in Gauss.
    sin_alpha : float or array-like
        Sine of the pitch angle. Default is 1.0 (i.e., alpha
        = pi/2). This is only used if ``pitch_average`` is False.
    pitch_average : bool
        Whether to use pitch-angle averaged value. Default is True.

    Returns
    -------
    nu_injection : float or array-like
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
    log_gamma : float or array-like
        Natural logarithm of the electron Lorentz factor.
    log_B : float or array-like
        Natural logarithm of the magnetic field strength in Gauss.
    sin_alpha : float or array-like
        Sine of the pitch angle (used only if ``pitch_average=False``).
    pitch_average : bool
        Whether to use pitch-angle averaged value.

    Returns
    -------
    log_nu_injection : float or array-like
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
    nu : float or array-like
        Synchrotron frequency in Hz.

    B : float or array-like
        Magnetic field strength in Gauss.

    sin_alpha : float or array-like
        Sine of the pitch angle. Only used if ``pitch_average=False``.

    pitch_average : bool
        Whether to use pitch-angle averaged value.
        Uses :math:`\left<\sin\alpha\right> = 2/\pi` if True.

    Returns
    -------
    gamma : float or array-like
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
    gamma : float or array-like
        Electron Lorentz factor.

    B : float, array-like, or astropy.units.Quantity
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
    gamma : float or array-like
        Electron Lorentz factor.

    B : float, array-like, or astropy.units.Quantity
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
    gamma : float or array-like
        Electron Lorentz factor.

    B : float, array-like, or astropy.units.Quantity
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
    nu : float, array-like, or astropy.units.Quantity
        Synchrotron frequency. Default units are Hz.

    B : float, array-like, or astropy.units.Quantity
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


# ============================================ #
# Synchrotron Kernels                          #
# ============================================ #
# These functions implement the synchrotron kernel functions F(x) and G(x) in
# various ways, including direct integration and interpolation-based approximations.
def first_synchrotron_kernel(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Compute the first synchrotron kernel function :math:`F(x)`.

    The function :math:`F(x)` describes the spectral shape of synchrotron
    radiation emitted by a single relativistic electron. It is defined as:

    .. math::

        F(x) = x \int_x^\infty K_{5/3}(z) \, dz

    .. warning::

        This is NOT an efficient option for large-scale synchrotron calculation. It uses
        the :func:`scipy.integrate.quad` function to compute the integral over calls to
        ``scipy.special.kv``, which can be slow for large arrays.

    Parameters
    ----------
    x : float or array-like
        Dimensionless frequency parameter, defined as the ratio of the observing
        frequency to the critical frequency, :math:`x = \nu / \nu_{critical}`.
    """
    # Coerce everything to numpy arrays for vectorized operations.
    x = np.atleast_1d(x).astype(float)
    F_x = np.empty_like(x)

    for i, xi in tqdm(
        enumerate(x),
        total=len(x),
        bar_format=triceratops_config["system.appearance.progress_bar_format"],
        desc="Integrating Synchrotron Kernel",
    ):
        integral = quad(
            lambda z: kv(5 / 3, z),
            xi,
            np.inf,
            epsrel=1e-10,
        )[0]
        F_x[i] = xi * integral

    return F_x if F_x.size == 1 else F_x


def second_synchrotron_kernel(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Compute the second synchrotron kernel function :math:`G(x)`.

    The function :math:`G(x)` describes another spectral shape of synchrotron
    radiation emitted by a single relativistic electron. It is defined as:

    .. math::

        G(x) = x K_{2/3}(x)

    Parameters
    ----------
    x : float or array-like
        Dimensionless frequency parameter, defined as the ratio of the observing
        frequency to the critical frequency, :math:`x = \nu / \nu_{critical}`.
    """
    x = np.atleast_1d(x).astype(float)
    G_x = x * kv(2 / 3, x)
    return G_x[0] if G_x.size == 1 else G_x


def _build_first_kernel_table(
    x_min: float = 1e-5,
    x_max: float = 1e2,
    num_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Build a lookup table for the first synchrotron kernel :math:`F(x)`.

    This function precomputes the values of the first synchrotron kernel
    :math:`F(x)` over a specified range of :math:`x` values. This can be used
    to speed up calculations that require repeated evaluations of the kernel.

    Parameters
    ----------
    x_min : float, optional
        Minimum value of :math:`x` to compute. Default is ``1e-5``.

    x_max : float, optional
        Maximum value of :math:`x` to compute. Default is ``1e2``.

    num_points : int, optional
        Number of points to compute between ``x_min`` and ``x_max``.
        Default is ``1000``.

    Returns
    -------
    x_values : numpy.ndarray
        Array of :math:`x` values.

    F_values : numpy.ndarray
        Corresponding array of :math:`F(x)` values.
    """
    x_values = np.geomspace(x_min, x_max, num_points)
    F_values = first_synchrotron_kernel(x_values)

    return x_values, F_values


def _build_second_kernel_table(
    x_min: float = 1e-5,
    x_max: float = 1e2,
    num_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Build a lookup table for the second synchrotron kernel :math:`G(x)`.

    This function precomputes the values of the second synchrotron kernel
    :math:`G(x)` over a specified range of :math:`x` values. This can be used
    to speed up calculations that require repeated evaluations of the kernel.

    Parameters
    ----------
    x_min : float, optional
        Minimum value of :math:`x` to compute. Default is ``1e-5``.

    x_max : float, optional
        Maximum value of :math:`x` to compute. Default is ``1e2``.

    num_points : int, optional
        Number of points to compute between ``x_min`` and ``x_max``.
        Default is ``1000``.

    Returns
    -------
    x_values : numpy.ndarray
        Array of :math:`x` values.

    G_values : numpy.ndarray
        Corresponding array of :math:`G(x)` values.
    """
    x_values = np.geomspace(x_min, x_max, num_points)
    G_values = second_synchrotron_kernel(x_values)

    return x_values, G_values


def compute_first_kernel_interp(x_values: np.ndarray, F_values: np.ndarray, **kwargs):
    r"""
    Get an interpolation function for the first synchrotron kernel :math:`F(x)`.

    This function creates an interpolation function based on precomputed
    :math:`x` and :math:`F(x)` values. The resulting function can be used to
    efficiently evaluate :math:`F(x)` at arbitrary :math:`x` values within
    the range of the provided data.

    Parameters
    ----------
    x_values : numpy.ndarray
        Array of :math:`x` values.

    F_values : numpy.ndarray
        Corresponding array of :math:`F(x)` values.

    Returns
    -------
    interp_func : callable
        Interpolation function that takes :math:`x` as input and returns
        the interpolated :math:`F(x)`.
    """
    # Extract the bounds
    x_min = x_values[0]
    x_max = x_values[-1]

    # Generate the interpolation function.
    interp_func = interp1d(x_values, F_values, **kwargs)

    # Write the guarded interpolator function using the asymptotic limits.
    def _interpolator(_x):
        # Coerce _x to a numpy array so we can mask.
        _x = np.atleast_1d(_x).astype(float)
        _y = np.empty_like(_x)

        # Handle the lower bound
        _lmsk = _x < x_min
        _y[_lmsk] = 2.14952824153447863671029 * _x[_lmsk] ** (1 / 3)

        # Handle the upper bound
        _rmsk = _x > x_max
        _y[_rmsk] = np.sqrt(np.pi * _x[_rmsk] / 2) * np.exp(-_x[_rmsk])

        # Handle the interpolation region
        _imsk = ~_lmsk & ~_rmsk
        _y[_imsk] = np.exp(interp_func(_x[_imsk]))

        return _y[0] if _y.size == 1 else _y

    return _interpolator


def compute_second_kernel_interp(x_values: np.ndarray, G_values: np.ndarray, **kwargs):
    r"""
    Get an interpolation function for the second synchrotron kernel :math:`G(x)`.

    This function creates an interpolation function based on precomputed
    :math:`x` and :math:`G(x)` values. The resulting function can be used to
    efficiently evaluate :math:`G(x)` at arbitrary :math:`x` values within
    the range of the provided data.

    Parameters
    ----------
    x_values : numpy.ndarray
        Array of :math:`x` values.

    G_values : numpy.ndarray
        Corresponding array of :math:`G(x)` values.

    Returns
    -------
    interp_func : callable
        Interpolation function that takes :math:`x` as input and returns
        the interpolated :math:`G(x)`.
    """
    # Extract the bounds
    x_min = x_values[0]
    x_max = x_values[-1]

    # Generate the interpolation function.
    interp_func = interp1d(x_values, G_values, **kwargs)

    # Write the guarded interpolator function using the asymptotic limits.
    def _interpolator(_x):
        # Coerce _x to a numpy array so we can mask.
        _x = np.atleast_1d(_x).astype(float)
        _y = np.empty_like(_x)

        # Handle the lower bound
        _lmsk = _x < x_min
        _y[_lmsk] = 1.808 * _x[_lmsk] ** (1 / 3)

        # Handle the upper bound
        _rmsk = _x > x_max
        _y[_rmsk] = np.sqrt(np.pi * _x[_rmsk] / 2) * np.exp(-_x[_rmsk])

        # Handle the interpolation region
        _imsk = ~_lmsk & ~_rmsk
        _y[_imsk] = np.exp(interp_func(_x[_imsk]))

        return _y[0] if _y.size == 1 else _y

    return _interpolator


def get_first_kernel_interpolator(x_min: float = 1e-5, x_max: float = 1e2, num_points: int = 1000, **kwargs):
    r"""
    Get a pre-built interpolation function for the first synchrotron kernel :math:`F(x)`.

    This function builds a lookup table for :math:`F(x)` and returns an
    interpolation function that can be used to efficiently evaluate
    :math:`F(x)` at arbitrary :math:`x` values within the specified range.

    Parameters
    ----------
    x_min : float, optional
        Minimum value of :math:`x` to compute. Default is ``1e-5``.

    x_max : float, optional
        Maximum value of :math:`x` to compute. Default is ``1e2``.

    num_points : int, optional
        Number of points to compute between ``x_min`` and ``x_max``.
        Default is ``1000``.

    Returns
    -------
    interp_func : callable
        Interpolation function that takes :math:`x` as input and returns
        the interpolated :math:`F(x)`.
    """
    x_values, F_values = _build_first_kernel_table(
        x_min=x_min,
        x_max=x_max,
        num_points=num_points,
    )

    return compute_first_kernel_interp(x_values, np.log(F_values), **kwargs)


def get_second_kernel_interpolator(x_min: float = 1e-5, x_max: float = 1e2, num_points: int = 1000, **kwargs):
    r"""
    Get a pre-built interpolation function for the second synchrotron kernel :math:`G(x)`.

    This function builds a lookup table for :math:`G(x)` and returns an
    interpolation function that can be used to efficiently evaluate
    :math:`G(x)` at arbitrary :math:`x` values within the specified range.

    Parameters
    ----------
    x_min : float, optional
        Minimum value of :math:`x` to compute. Default is ``1e-5``.

    x_max : float, optional
        Maximum value of :math:`x` to compute. Default is ``1e2``.

    num_points : int, optional
        Number of points to compute between ``x_min`` and ``x_max``.
        Default is ``1000``.

    Returns
    -------
    interp_func : callable
        Interpolation function that takes :math:`x` as input and returns
        the interpolated :math:`G(x)`.
    """
    x_values, G_values = _build_second_kernel_table(
        x_min=x_min,
        x_max=x_max,
        num_points=num_points,
    )

    return compute_second_kernel_interp(x_values, np.log(G_values), **kwargs)


# ================================================== #
# Single Electron Spectra                            #
# ================================================== #
_single_electron_power_coef_CGS = np.sqrt(3) * (constants.e.esu**3 / (constants.m_e * constants.c**2)).cgs.value


def _opt_compute_single_electron_power(
    nu: "_ArrayLike", gamma: float, B: float, alpha: float, kernel_function: Callable
):
    r"""
    Compute the spectral power of a single electron undergoing synchrotron radiation.

    Given a magnetic field strength, electron Lorentz factor, pitch angle, and observing frequency,
    compute the synchrotron power per unit frequency emitted by a single electron. Documentation can
    be found in the high-level API function :func:`compute_single_electron_power`.

    Parameters
    ----------
    nu : array-like
        Observing frequency in Hz.
    gamma : float
        Electron Lorentz factor.
    B : float
        Magnetic field strength in Gauss.
    alpha : float
        Pitch angle in radians.
    kernel_function : callable
        Synchrotron kernel function to use. For heavy use cases, it is recommended that a kernel approximation
        be used here instead of the :func:`first_synchrotron_kernel` function. Interpolations can be generated
        using :func:`get_first_kernel_interpolator`. There are also power-series implementations which
        may be faster for some applications.

    Returns
    -------
    P_nu : array-like
        Synchrotron power per unit frequency in erg/s/Hz.
    """
    # Enforce nu as a numpy array.
    nu = np.atleast_1d(nu).astype(float)

    # Compute the critical frequency nu_c using the optimized pathway. This is
    # the form in RL CH 6.
    nu_c = _optimized_compute_nu_critical(gamma, B)

    # Compute the dimensionless frequency parameter x.
    x = nu / nu_c

    # Compute the kernel values.
    F_x = kernel_function(x)

    # Compute the prefactor.
    prefactor = _single_electron_power_coef_CGS * B * np.sin(alpha)

    # Compute the power spectrum.
    P_nu = prefactor * F_x

    return P_nu if P_nu.size > 1 else P_nu[0]


def compute_single_electron_power(
    nu: "_UnitBearingArrayLike",
    gamma: float,
    B: "_UnitBearingArrayLike",
    alpha: float = np.pi / 2,
    kernel_function: Callable = None,
):
    r"""
    Compute the spectral power of a single electron undergoing synchrotron radiation.

    Given a magnetic field strength, electron Lorentz factor, pitch angle, and observing frequency,
    compute the synchrotron power per unit frequency emitted by a single electron. From :footcite:t:`RybickiLightman`,

    .. math::

        P(\nu) = \frac{\sqrt{3} e^3 B}{m_e c^2} \sin\alpha\, F\!\left(\frac{\nu}{\nu_c}\right),

    where :math:`F(x)` is the first synchrotron kernel function and :math:`\nu_c` is the critical frequency.

    Parameters
    ----------
    nu : array-like or astropy.units.Quantity
        Observing frequency. Default units are Hz. If a Quantity is provided, units will be validated. ``nu``
        may be vector or scalar.
    gamma : float
        Electron Lorentz factor.
    B : array-like or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.
    alpha : float
        Pitch angle in radians. By default, this is assumed to ``pi/2``.
    kernel_function : callable, optional
        Synchrotron kernel function to use. For heavy use cases, it is recommended that a kernel approximation
        be used here instead of the :func:`first_synchrotron_kernel` function. Interpolations can be generated
        using :func:`get_first_kernel_interpolator`. There are also power-series implementations which
        may be faster for some applications. If ``None``, the default :func:`first_synchrotron_kernel` function
        will be used.

    Returns
    -------
    P_nu : array-like or astropy.units.Quantity
        Synchrotron power per unit frequency. Default units are erg/s/Hz.

    Notes
    -----
    For heavy use cases, the low-level API function :func:`_opt_compute_single_electron_power` is recommended.

    For details on the relevant theory, see :ref:`synchrotron_theory`.
    """
    # Validate units and coerce to CGS values.
    nu = ensure_in_units(nu, u.Hz)
    B = ensure_in_units(B, u.Gauss)

    # Set the kernel function if not provided.
    if kernel_function is None:
        kernel_function = first_synchrotron_kernel

    # Call the optimized computation function.
    P_nu_cgs = _opt_compute_single_electron_power(nu=nu, gamma=gamma, B=B, alpha=alpha, kernel_function=kernel_function)
    return P_nu_cgs * u.erg / u.s / u.Hz


# ================================================= #
# Multi-Electron Spectra                            #
# ================================================= #
def _opt_compute_ME_spectrum_from_dist_grid(
    nu: np.ndarray,
    gamma: np.ndarray,
    N_gamma: np.ndarray,
    B: float,
    alpha: float,
    kernel_function: Callable,
):
    r"""
    Compute the synchrotron spectrum from a multi-electron distribution using a finite stencil of the dist. func.

    This function computes the synchrotron spectral power density emitted by a population of electrons
    with a given Lorentz factor distribution. The calculation integrates over the electron distribution
    using a finite grid of Lorentz factor values and corresponding number densities.

    For more detailed documentation, see the public API function :func:`compute_ME_spectrum_from_dist_grid`.

    Parameters
    ----------
    nu: array-like
        Observing frequency in Hz.
    gamma: array-like
        Electron Lorentz factor grid. This should be a 1D monotonic array of shape ``(N,)``. Each element should
        match the tabulated values in ``N_gamma``.
    N_gamma: array-like
        The electron number density distribution evaluated at each Lorentz factor in ``gamma``.
        This should be a 1D array of shape ``(N,)``.

        The implementation of this function computes :math:`\Delta \gamma` on the fly, so the values in
        ``N_gamma`` should represent the number density per unit Lorentz factor, i.e., :math:`N(\gamma) = dN/d\gamma`.
    B: float
        Magnetic field strength in Gauss.
    alpha: float
        The pitch angle in radians. By default, this is :math:`\pi/2`.
    kernel_function: callable
        Synchrotron kernel function to use. For heavy use cases, it is recommended that
        a kernel approximation be used here instead of the :func:`first_synchrotron_kernel` function.
        Interpolations can be generated using :func:`get_first_kernel_interpolator`. There are also power-series
        implementations which may be faster for some applications.

    Returns
    -------
    P_nu : array-like
        Synchrotron power per unit frequency per unit volume.
    """
    # Ensure inputs are numpy arrays.
    nu = np.atleast_1d(nu)
    gamma = np.asarray(gamma)
    N_gamma = np.asarray(N_gamma)

    # Compute the critical frequency nu_c and the corresponding dimensionless
    # parameter x. For (N,) shaped gamma and (M,) shaped nu, this results in
    # (M, N) shaped x.
    nu_c = _optimized_compute_nu_critical(gamma[None, :], B)
    x = nu[:, None] / nu_c

    # Compute the kernel values F(x).
    F_x = kernel_function(x)

    # Compute the prefactor.
    prefactor = _single_electron_power_coef_CGS * B * np.sin(alpha)

    # Prepare the integration. We use the trapezoidal rule. The integrand has
    # shape (M, N) where M is the number of nu values and N is the number of
    # gamma values.
    integrand = N_gamma[None, :] * F_x
    P_nu = prefactor * _trapz(integrand, gamma, axis=1)

    return P_nu


def _opt_compute_ME_spectrum_from_dist_function(
    nu: np.ndarray,
    N_gamma_func: Callable[[float], float],
    gamma_min: float,
    gamma_max: float,
    B: float,
    alpha: float,
    kernel_function: Callable,
    **kwargs: object,
) -> Any:
    r"""
    Compute the synchrotron spectrum from a multi-electron distribution defined as a continuous function N(gamma).

    Parameters
    ----------
    nu : array-like
        Observing frequencies in Hz.
    N_gamma_func : callable
        Function returning dN/dgamma at gamma.
    gamma_min, gamma_max : float
        Integration bounds in Lorentz factor.
    B : float
        Magnetic field strength in Gauss.
    alpha : float
        Pitch angle in radians.
    kernel_function : callable
        Synchrotron kernel function.
    **kwargs:
        Additional keyword arguments passed to :func:`scipy.integrate.quad`.

    Returns
    -------
    P_nu : ndarray
        Synchrotron power per unit frequency per unit volume.
    """
    nu = np.atleast_1d(nu).astype(float)

    prefactor = _single_electron_power_coef_CGS * B * np.sin(alpha)

    P_nu = np.empty_like(nu)

    for i, nui in enumerate(nu):

        def _integrand(gamma, _nui=nui):
            nu_c = _optimized_compute_nu_critical(gamma, B)
            x = _nui / nu_c
            return N_gamma_func(gamma) * kernel_function(x)

        integral = quad(_integrand, gamma_min, gamma_max, **kwargs)[0]

        P_nu[i] = prefactor * integral

    return P_nu


def compute_ME_spectrum_from_dist_function(
    nu: "_UnitBearingArrayLike",
    N_gamma_func: Callable[[float], float],
    gamma_min: float,
    gamma_max: float,
    B: "_UnitBearingArrayLike",
    alpha: float = np.pi / 2,
    kernel_function: Callable = None,
    **kwargs,
):
    r"""
    Compute the synchrotron spectrum from a multi-electron distribution defined as a function.

    This function computes the synchrotron power per unit frequency per unit volume emitted
    by a population of relativistic electrons whose distribution is specified as a continuous
    function :math:`dN/d\gamma`.

    Parameters
    ----------
    nu : array-like or astropy.units.Quantity
        Observing frequencies. Default units are Hz.
    N_gamma_func : callable
        Function returning :math:`dN/d\gamma` at a given Lorentz factor.
        The function must return values in units of ``cm⁻³``.
    gamma_min, gamma_max : float
        Integration bounds in Lorentz factor.
    B : array-like or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.
    alpha : float, optional
        Pitch angle in radians. Default is :math:`\pi/2`.
    kernel_function : callable, optional
        Synchrotron kernel function. If ``None``, the exact
        :func:`first_synchrotron_kernel` is used. For inference or repeated
        evaluations, an interpolated kernel is strongly recommended.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`scipy.integrate.quad` (e.g., ``epsrel``, ``limit``).

    Returns
    -------
    P_nu : astropy.units.Quantity
        Synchrotron power per unit frequency per unit volume
        in units of ``erg s⁻¹ Hz⁻¹ cm⁻³``.

    Notes
    -----
    This function performs adaptive quadrature over the electron Lorentz
    factor. It is best suited for analytic or injection distributions.
    For evolved or tabulated distributions, use
    :func:`compute_ME_spectrum_from_dist_grid` instead.
    """
    # --- Unit handling ---
    nu = ensure_in_units(nu, u.Hz)
    B = ensure_in_units(B, u.Gauss)

    if kernel_function is None:
        kernel_function = first_synchrotron_kernel

    # --- Low-level computation ---
    P_nu_cgs = _opt_compute_ME_spectrum_from_dist_function(
        nu=nu,
        N_gamma_func=N_gamma_func,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        B=B,
        alpha=alpha,
        kernel_function=kernel_function,
        **kwargs,
    )

    return P_nu_cgs * u.erg / u.s / u.Hz / u.cm**3


def compute_ME_spectrum_from_dist_grid(
    nu: "_UnitBearingArrayLike",
    gamma: np.ndarray,
    N_gamma: "_UnitBearingArrayLike",
    B: "_UnitBearingArrayLike",
    alpha: float = np.pi / 2,
    kernel_function: Callable = None,
):
    r"""
    Compute the synchrotron spectrum from a multi-electron distribution defined on a grid.

    This function computes the synchrotron power per unit frequency per unit volume emitted
    by a population of relativistic electrons whose Lorentz factor distribution is provided
    as a finite grid.

    Parameters
    ----------
    nu : array-like or astropy.units.Quantity
        Observing frequencies. Default units are Hz.
    gamma : array-like
        Electron Lorentz factor grid. Must be 1D, monotonic, and match ``N_gamma``.
    N_gamma : array-like or astropy.units.Quantity
        Electron distribution evaluated on ``gamma``, interpreted as
        :math:`dN/d\gamma`. Default units are ``cm⁻³``.
    B : array-like or astropy.units.Quantity
        Magnetic field strength. Default units are Gauss.
    alpha : float, optional
        Pitch angle in radians. Default is :math:`\pi/2`.
    kernel_function : callable, optional
        Synchrotron kernel function. If ``None``, the exact
        :func:`first_synchrotron_kernel` is used. For performance-critical
        applications, an interpolated kernel is strongly recommended.

    Returns
    -------
    P_nu : astropy.units.Quantity
        Synchrotron power per unit frequency per unit volume
        in units of ``erg s⁻¹ Hz⁻¹ cm⁻³``.

    Notes
    -----
    This function performs a trapezoidal integration over the electron
    Lorentz factor grid. For logarithmically spaced grids, this approach
    is numerically stable and converges reliably.
    """
    # --- Unit handling ---
    nu = ensure_in_units(nu, u.Hz)
    B = ensure_in_units(B, u.Gauss)
    N_gamma = ensure_in_units(N_gamma, 1 / u.cm**3)

    gamma = np.asarray(gamma, dtype=float)

    if kernel_function is None:
        kernel_function = first_synchrotron_kernel

    # --- Low-level computation ---
    P_nu_cgs = _opt_compute_ME_spectrum_from_dist_grid(
        nu=nu,
        gamma=gamma,
        N_gamma=N_gamma,
        B=B,
        alpha=alpha,
        kernel_function=kernel_function,
    )

    return P_nu_cgs * u.erg / u.s / u.Hz / u.cm**3
