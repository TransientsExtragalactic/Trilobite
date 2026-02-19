"""
Core functions for computing Gaussian log-likelihood contributions.

This module provides **pure, stateless numerical utilities** for evaluating
Gaussian likelihood terms under a variety of observational constraints.
These functions implement only the *mathematical form* of the likelihood and
make **no assumptions** about:

- The origin of the model predictions,
- The structure or semantics of the data,
- Units, shapes, masking, or parameter validation.

They are therefore intended to be used as **low-level building blocks**
inside higher-level likelihood classes (e.g.
:class:`~inference.likelihood.base.Likelihood`), which are responsible for:

- Model–data compatibility validation,
- Unit coercion and consistency,
- Censoring semantics and masking logic,
- Efficient reuse of preprocessed data.

Notes
-----
For more details on the use of likelihood functions in Triceratops, see :ref:`likelihoods`.
"""

from typing import Optional

import numpy as np
from scipy.stats import norm


def censored_gaussian_loglikelihood(
    data_y: np.ndarray,
    model_y: np.ndarray,
    y_err: np.ndarray,
    *,
    y_upper: Optional[np.ndarray] = None,
    y_lower: Optional[np.ndarray] = None,
    y_upper_mask: Optional[np.ndarray] = None,
    y_lower_mask: Optional[np.ndarray] = None,
) -> float:
    r"""
    Compute the log-likelihood for Gaussian-distributed data with censoring.

    This function evaluates the total log-likelihood assuming normally
    distributed measurement errors, supporting a mixture of:

    - Fully detected data points,
    - Upper-censored observations (y < y_upper),
    - Lower-censored observations (y > y_lower).

    All inputs are assumed to be **numerically compatible and preprocessed**.
    In particular, this function performs **no** unit handling, validation,
    reshaping, broadcasting, or inference of censoring semantics.

    Parameters
    ----------
    data_y : np.ndarray
        The observed data array. ``data_y`` should have the same shape as
        ``model_y`` and ``y_err``. For censored data points, the values in
        ``data_y`` are **not used** and may therefore be set to arbitrary
        values (e.g. ``np.nan``) without affecting the likelihood.
    model_y : np.ndarray
        Model-predicted values corresponding to each data point. This array
        represents the mean of the Gaussian distribution evaluated at each
        location and must match the shape of ``data_y``.
    y_err : np.ndarray
        One-sigma measurement uncertainties associated with the dependent
        variable. Uncertainties must be defined for **all** data points,
        including censored observations.
    y_upper : np.ndarray, optional
        Upper limits for upper-censored observations. Must match the shape of
        ``data_y``. Values are only accessed for indices where
        ``y_upper_mask`` is ``True``; all other entries are ignored.
    y_lower : np.ndarray, optional
        Lower limits for lower-censored observations. Must match the shape of
        ``data_y``. Values are only accessed for indices where
        ``y_lower_mask`` is ``True``; all other entries are ignored.
    y_upper_mask : np.ndarray of bool, optional
        Boolean mask indicating which data points are upper-censored.
    y_lower_mask : np.ndarray of bool, optional
        Boolean mask indicating which data points are lower-censored.

    Returns
    -------
    float
        Total log-likelihood value.

    Notes
    -----
    Censoring is handled *exclusively* via the supplied boolean masks.
    No assumptions are made based on the numerical values of ``data_y``,
    ``y_upper``, or ``y_lower``.

    For each data point :math:`i`, the likelihood contribution is determined
    as follows:

    - **Detection**
      If ``y_upper_mask[i]`` and ``y_lower_mask[i]`` are both ``False``,
      the data point is treated as a standard Gaussian detection with
      likelihood

      .. math::

          \ln \mathcal{L}_i
          =
          -\frac{1}{2}
          \left[
              \frac{(y_i - \mu_i)^2}{\sigma_i^2}
              + \ln\!\left(2\pi\sigma_i^2\right)
          \right],

      where :math:`y_i` is the observed value, :math:`\mu_i` is the model
      prediction, and :math:`\sigma_i` is the measurement uncertainty.

    - **Upper-censored observation**
      If ``y_upper_mask[i]`` is ``True``, the data point contributes the
      probability that the true value lies *below* the reported upper limit:

      .. math::

          \ln \mathcal{L}_i
          =
          \ln \left[
              \Phi\!\left(
                  \frac{y_i^{\rm upper} - \mu_i}{\sigma_i}
              \right)
          \right],

      where :math:`\Phi` is the standard normal cumulative distribution
      function.

    - **Lower-censored observation**
      If ``y_lower_mask[i]`` is ``True``, the data point contributes the
      probability that the true value lies *above* the reported lower limit:

      .. math::

          \ln \mathcal{L}_i
          =
          \ln \left[
              1 - \Phi\!\left(
                  \frac{y_i^{\rm lower} - \mu_i}{\sigma_i}
              \right)
          \right].

    The total log-likelihood is obtained by summing over all data points:

    .. math::

        \ln \mathcal{L}
        =
        \sum_i \ln \mathcal{L}_i.

    Numerical underflow in the evaluation of tail probabilities is handled
    by assigning ``-np.inf`` to non-finite contributions.

    This function is intended to serve as a **low-level numerical backend**
    for higher-level likelihood classes, which are responsible for enforcing
    model–data compatibility, constructing the censoring masks, and managing
    units and data containers.
    """
    # Set the initial likelihood value.
    log_likelihood = 0.0

    # Default masks
    if y_upper_mask is None:
        y_upper_mask = np.zeros_like(data_y, dtype=bool)
    if y_lower_mask is None:
        y_lower_mask = np.zeros_like(data_y, dtype=bool)

    det_mask = ~y_upper_mask & ~y_lower_mask

    # Compute the likelihood for detections.
    if np.any(det_mask):
        resid = (data_y[det_mask] - model_y[det_mask]) / y_err[det_mask]
        log_likelihood += -0.5 * (np.sum(resid**2) + np.sum(np.log(2.0 * np.pi * y_err[det_mask] ** 2)))

    # Compute for non detections.
    if y_upper is not None and np.any(y_upper_mask):
        logcdf = norm.logcdf(
            y_upper[y_upper_mask],
            loc=model_y[y_upper_mask],
            scale=y_err[y_upper_mask],
        )
        log_likelihood += np.sum(np.where(np.isfinite(logcdf), logcdf, -np.inf))

    if y_lower is not None and np.any(y_lower_mask):
        logsf = norm.logsf(
            y_lower[y_lower_mask],
            loc=model_y[y_lower_mask],
            scale=y_err[y_lower_mask],
        )
        log_likelihood += np.sum(np.where(np.isfinite(logsf), logsf, -np.inf))

    return log_likelihood


def gaussian_loglikelihood(
    data_y: np.ndarray,
    model_y: np.ndarray,
    y_err: np.ndarray,
) -> float:
    r"""
    Compute the log-likelihood for Gaussian-distributed, uncensored data.

    This function evaluates the total log-likelihood under the assumption
    that all observations are fully detected and have normally distributed
    measurement errors.

    All inputs are assumed to be **numerically compatible and preprocessed**.
    In particular, this function performs **no** unit handling, validation,
    reshaping, broadcasting, or inference of data semantics.

    Parameters
    ----------
    data_y : np.ndarray
        The observed data array. Must have the same shape as ``model_y`` and
        ``y_err``.
    model_y : np.ndarray
        Model-predicted values corresponding to each data point. This array
        represents the mean of the Gaussian distribution evaluated at each
        location and must match the shape of ``data_y``.
    y_err : np.ndarray
        One-sigma measurement uncertainties associated with the dependent
        variable. Must be defined for all data points.

    Returns
    -------
    float
        Total log-likelihood value.

    Notes
    -----
    For each data point :math:`i`, the likelihood contribution is given by
    the standard Gaussian log-likelihood:

    .. math::

        \ln \mathcal{L}_i
        =
        -\frac{1}{2}
        \left[
            \frac{(y_i - \mu_i)^2}{\sigma_i^2}
            + \ln\!\left(2\pi\sigma_i^2\right)
        \right],

    where :math:`y_i` is the observed value, :math:`\mu_i` is the model
    prediction, and :math:`\sigma_i` is the measurement uncertainty.

    The total log-likelihood is obtained by summing over all data points:

    .. math::

        \ln \mathcal{L}
        =
        \sum_i \ln \mathcal{L}_i.

    This function is intended to serve as a **low-level numerical backend**
    for higher-level likelihood classes that handle model–data compatibility,
    unit management, and data container semantics.
    """
    resid = (data_y - model_y) / y_err

    log_likelihood = -0.5 * (np.sum(resid**2) + np.sum(np.log(2.0 * np.pi * y_err**2)))

    return log_likelihood
