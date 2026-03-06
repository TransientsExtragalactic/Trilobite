"""
Utilities for special relativity.

This module provides small, composable helpers for common special-relativistic
transformations used in radiative processes (Doppler boosting, frequency
transforms, and specific-intensity transforms).
"""

from typing import Union

import numpy as np

# ------------------------------ #
# Type Conventions               #
# ------------------------------ #
_VectorFloat = Union[float, np.ndarray]

# ================================================== #
# Lorentz Transformation Utilities                   #
# ================================================== #


def compute_lorentz_factor(beta: _VectorFloat) -> _VectorFloat:
    r"""
    Compute the Lorentz factor :math:`\Gamma` from dimensionless speed :math:`\beta`.

    This function computes the Lorentz factor :math:`\Gamma` given a specific dimensionless
    speed :math:`\beta = v/c`. The Lorentz factor is defined as:

    .. math::

        \Gamma = \frac{1}{\sqrt{1-\beta^2}}.

    Parameters
    ----------
    beta : float or array-like
        Dimensionless speed :math:`\beta = v/c`. Must satisfy :math:`|\beta| < 1`.

    Returns
    -------
    float or ndarray
        Lorentz factor :math:`\Gamma = (1-\beta^2)^{-1/2}`.

    Raises
    ------
    ValueError
        If any value of ``beta`` is unphysical (i.e., :math:`|\beta| \geq 1`).
    """
    beta = np.asarray(beta, dtype=float)
    if np.any(np.abs(beta) >= 1.0):
        raise ValueError("Invalid beta: must satisfy |beta| < 1.")

    gamma = 1.0 / np.sqrt(1.0 - beta**2)

    return gamma.item() if gamma.ndim == 0 else gamma


def compute_beta_from_gamma(gamma: _VectorFloat) -> _VectorFloat:
    r"""
    Compute the dimensionless speed :math:`\beta` from the Lorentz factor :math:`\Gamma`.

    This function computes the dimensionless speed :math:`\beta = v/c` given a specific Lorentz
    factor :math:`\Gamma`. The relationship is defined as:

    .. math::

        \beta = \sqrt{1 - \frac{1}{\Gamma^2}}.

    Parameters
    ----------
    gamma : float or array-like
        Lorentz factor :math:`\Gamma`. Must satisfy :math:`\Gamma \geq 1`.

    Returns
    -------
    float or ndarray
        Dimensionless speed :math:`\beta = \sqrt{1 - 1/\Gamma^2}`.

    Raises
    ------
    ValueError
        If any value of ``gamma`` is unphysical (i.e., :math:`\Gamma < 1`).
    """
    gamma = np.asarray(gamma, dtype=float)
    if np.any(gamma < 1.0):
        raise ValueError("Invalid gamma: must satisfy gamma >= 1.")

    beta = np.sqrt(1.0 - 1.0 / gamma**2)

    return beta.item() if beta.ndim == 0 else beta


# ================================================== #
# Doppler Factor Corrections                         #
# ================================================== #
# These are utilities for correcting various quantities for relativistic
# doppler boosting and aberration effects. The transformations are based on the Doppler factor
# :math:`\delta = [\Gamma(1-\beta\cos\theta)]^{-1}` and the Lorentz invariance of :math:`I_\nu/\nu^3`.
def compute_doppler_factor(gamma: _VectorFloat, theta: _VectorFloat) -> _VectorFloat:
    r"""
    Compute the Doppler factor :math:`\delta` from the Lorentz factor :math:`\Gamma` and viewing angle :math:`\theta`.

    The Doppler factor is defined as

    .. math::

        \delta = \frac{1}{\Gamma(1-\beta\cos\theta)},

    where :math:`\beta = v/c = \sqrt{1 - \Gamma^{-2}}`.

    Parameters
    ----------
    gamma : float or array-like
        Lorentz factor :math:`\Gamma`. Must satisfy :math:`\Gamma \ge 1`.

    theta : float or array-like
        Angle between the velocity vector and the line of sight in radians.
        Must satisfy :math:`0 \le \theta \le \pi`.

        The angle is measured in the **observer (lab) frame**, not the
        rest frame of the emitting plasma.

    Returns
    -------
    float or ndarray
        Doppler factor

        .. math::

            \delta = [\Gamma(1-\beta\cos\theta)]^{-1}.

    Raises
    ------
    ValueError
        If ``gamma < 1`` or ``theta`` lies outside the interval
        :math:`[0,\pi]`.
    """
    gamma = np.asarray(gamma, dtype=float)
    theta = np.asarray(theta, dtype=float)

    if np.any(gamma < 1):
        raise ValueError("gamma must satisfy Γ ≥ 1.")

    if np.any((theta < 0) | (theta > np.pi)):
        raise ValueError("theta must satisfy 0 ≤ θ ≤ π.")

    beta = compute_beta_from_gamma(gamma)
    cos_theta = np.cos(theta)

    delta = 1.0 / (gamma * (1.0 - beta * cos_theta))

    return delta.item() if delta.ndim == 0 else delta


def _resolve_doppler_factor(gamma=None, theta=None, doppler_factor=None):
    if doppler_factor is not None:
        return doppler_factor
    if gamma is None or theta is None:
        raise ValueError("Either doppler_factor or both gamma and theta must be provided.")
    return compute_doppler_factor(gamma, theta)


def compute_lab_frequency(
    rest_frequency: _VectorFloat,
    gamma: _VectorFloat = None,
    theta: _VectorFloat = None,
    doppler_factor: _VectorFloat = None,
) -> _VectorFloat:
    r"""
    Compute the observed (lab-frame) frequency from the rest-frame frequency.

    The relativistic frequency transformation is

    .. math::

        \nu_{\rm lab} = \delta \, \nu_{\rm rest},

    where :math:`\delta` is the Doppler factor.

    Parameters
    ----------
    rest_frequency : float or array-like
        Rest-frame frequency :math:`\nu_{\rm rest}`.

    gamma : float or array-like, optional
        Lorentz factor :math:`\Gamma`. Required if ``doppler_factor`` is not provided.

    theta : float or array-like, optional
        Angle between the velocity vector and the line of sight in radians.
        Required if ``doppler_factor`` is not provided.

    doppler_factor : float or array-like, optional
        Doppler factor :math:`\delta`. If provided, ``gamma`` and ``theta`` are ignored.

    Returns
    -------
    float or ndarray
        Observed frequency

        .. math::

            \nu_{\rm lab} = \delta \, \nu_{\rm rest}.

    Raises
    ------
    ValueError
        If neither ``doppler_factor`` nor the pair ``(gamma, theta)`` are provided.
    """
    rest_frequency = np.asarray(rest_frequency, dtype=float)
    doppler_factor = np.asarray(
        _resolve_doppler_factor(gamma=gamma, theta=theta, doppler_factor=doppler_factor), dtype=float
    )

    nu_lab = doppler_factor * rest_frequency

    return nu_lab.item() if nu_lab.ndim == 0 else nu_lab


def compute_rest_frequency(
    lab_frequency: _VectorFloat,
    gamma: _VectorFloat = None,
    theta: _VectorFloat = None,
    doppler_factor: _VectorFloat = None,
) -> _VectorFloat:
    r"""
    Compute the rest-frame frequency from the observed (lab-frame) frequency.

    The relativistic frequency transformation is

    .. math::

        \nu_{\rm rest} = \frac{\nu_{\rm lab}}{\delta},

    where :math:`\delta` is the Doppler factor.

    Parameters
    ----------
    lab_frequency : float or array-like
        Observed (lab-frame) frequency :math:`\nu_{\rm lab}`.

    gamma : float or array-like, optional
        Lorentz factor :math:`\Gamma`. Required if ``doppler_factor`` is not provided.

    theta : float or array-like, optional
        Angle between the velocity vector and the line of sight in radians.
        Required if ``doppler_factor`` is not provided.

    doppler_factor : float or array-like, optional
        Doppler factor :math:`\delta`. If provided, ``gamma`` and ``theta`` are ignored.

    Returns
    -------
    float or ndarray
        Rest-frame frequency :math:`\nu_{\rm rest}`.
    """
    lab_frequency = np.asarray(lab_frequency, dtype=float)

    d = np.asarray(_resolve_doppler_factor(gamma=gamma, theta=theta, doppler_factor=doppler_factor), dtype=float)

    nu_rest = lab_frequency / d

    return nu_rest.item() if nu_rest.ndim == 0 else nu_rest


def compute_lab_specific_intensity(
    rest_specific_intensity: _VectorFloat,
    gamma: _VectorFloat = None,
    theta: _VectorFloat = None,
    doppler_factor: _VectorFloat = None,
) -> _VectorFloat:
    r"""
    Transform rest-frame specific intensity to the observer (lab) frame.

    Using the Lorentz invariant :math:`I_\nu/\nu^3`, the transformation is

    .. math::

        I_{\nu,\rm lab} = \delta^3 I_{\nu,\rm rest}.

    Parameters
    ----------
    rest_specific_intensity : float or array-like
        Rest-frame specific intensity :math:`I_{\nu,\rm rest}`.

    gamma : float or array-like, optional
        Lorentz factor :math:`\Gamma`. Required if ``doppler_factor`` is not provided.

    theta : float or array-like, optional
        Viewing angle in radians. Required if ``doppler_factor`` is not provided.

    doppler_factor : float or array-like, optional
        Doppler factor :math:`\delta`.

    Returns
    -------
    float or ndarray
        Lab-frame specific intensity :math:`I_{\nu,\rm lab}`.
    """
    rest_specific_intensity = np.asarray(rest_specific_intensity, dtype=float)

    d = np.asarray(_resolve_doppler_factor(gamma=gamma, theta=theta, doppler_factor=doppler_factor), dtype=float)

    I_lab = d**3 * rest_specific_intensity

    return I_lab.item() if I_lab.ndim == 0 else I_lab


def compute_rest_specific_intensity(
    lab_specific_intensity: _VectorFloat,
    gamma: _VectorFloat = None,
    theta: _VectorFloat = None,
    doppler_factor: _VectorFloat = None,
) -> _VectorFloat:
    r"""
    Transform lab-frame specific intensity to the rest frame.

    .. math::

        I_{\nu,\rm rest} = \delta^{-3} I_{\nu,\rm lab}.

    Parameters
    ----------
    lab_specific_intensity : float or array-like
        Lab-frame specific intensity :math:`I_{\nu,\rm lab}`.

    gamma : float or array-like, optional
        Lorentz factor :math:`\Gamma`. Required if ``doppler_factor`` is not provided.

    theta : float or array-like, optional
        Viewing angle in radians. Required if ``doppler_factor`` is not provided.

    doppler_factor : float or array-like, optional
        Doppler factor :math:`\delta`.

    Returns
    -------
    float or ndarray
        Rest-frame specific intensity :math:`I_{\nu,\rm rest}`.
    """
    lab_specific_intensity = np.asarray(lab_specific_intensity, dtype=float)

    d = np.asarray(_resolve_doppler_factor(gamma=gamma, theta=theta, doppler_factor=doppler_factor), dtype=float)

    I_rest = lab_specific_intensity / d**3

    return I_rest.item() if I_rest.ndim == 0 else I_rest


def compute_lab_bolometric_intensity(
    rest_bolometric_intensity: _VectorFloat,
    gamma: _VectorFloat = None,
    theta: _VectorFloat = None,
    doppler_factor: _VectorFloat = None,
) -> _VectorFloat:
    r"""
    Transform rest-frame bolometric intensity to the lab frame.

    Because

    .. math::

        I_\nu \rightarrow \delta^3 I_\nu
        \quad\text{and}\quad
        d\nu \rightarrow \delta\,d\nu

    the integrated intensity transforms as

    .. math::

        I_{\rm lab} = \delta^4 I_{\rm rest}.

    Parameters
    ----------
    rest_bolometric_intensity : float or array-like
        Rest-frame bolometric intensity.

    gamma : float or array-like, optional
        Lorentz factor :math:`\Gamma`.

    theta : float or array-like, optional
        Viewing angle in radians.

    doppler_factor : float or array-like, optional
        Doppler factor :math:`\delta`.

    Returns
    -------
    float or ndarray
        Lab-frame bolometric intensity.
    """
    rest_bolometric_intensity = np.asarray(rest_bolometric_intensity, dtype=float)

    d = np.asarray(_resolve_doppler_factor(gamma=gamma, theta=theta, doppler_factor=doppler_factor), dtype=float)

    I_lab = d**4 * rest_bolometric_intensity

    return I_lab.item() if I_lab.ndim == 0 else I_lab


def compute_rest_bolometric_intensity(
    lab_bolometric_intensity: _VectorFloat,
    gamma: _VectorFloat = None,
    theta: _VectorFloat = None,
    doppler_factor: _VectorFloat = None,
) -> _VectorFloat:
    r"""
    Transform lab-frame bolometric intensity to the rest frame.

    .. math::

        I_{\rm rest} = \delta^{-4} I_{\rm lab}.

    Parameters
    ----------
    lab_bolometric_intensity : float or array-like
        Lab-frame bolometric intensity.

    gamma : float or array-like, optional
        Lorentz factor :math:`\Gamma`.

    theta : float or array-like, optional
        Viewing angle in radians.

    doppler_factor : float or array-like, optional
        Doppler factor :math:`\delta`.

    Returns
    -------
    float or ndarray
        Rest-frame bolometric intensity.
    """
    lab_bolometric_intensity = np.asarray(lab_bolometric_intensity, dtype=float)

    d = np.asarray(_resolve_doppler_factor(gamma=gamma, theta=theta, doppler_factor=doppler_factor), dtype=float)

    I_rest = lab_bolometric_intensity / d**4

    return I_rest.item() if I_rest.ndim == 0 else I_rest
