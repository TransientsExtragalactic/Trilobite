"""
Sampling utilities and functions for Trilobite inference problems.

This module provides helper functions and utilities to facilitate sampling-based
inference on astrophysical models using the Trilobite library. It includes
functions for initializing samplers, processing sampling results, and visualizing
sampling outputs.
"""

from typing import Callable

import numpy as np
from scipy.optimize import minimize


# --------------------------------------- #
# Optimization Utilities                  #
# --------------------------------------- #
def find_posterior_maximum(
    log_posterior: Callable[[np.ndarray], float],
    theta0: np.ndarray,
    *,
    maxiter: int = 4000,
) -> np.ndarray:
    r"""
    Locally maximize a log-posterior starting from ``theta0``.

    This is used to move a poor initial guess into the bulk of the posterior before sampling. A
    derivative-free Nelder-Mead search is run in coordinates :math:`u_i = (\theta_i - \theta_{0,i})/|\theta_{0,i}|`
    (absolute units for components that start at zero), so parameters spanning many orders of magnitude are
    treated on equal footing. Points with a non-finite log-posterior are treated as infinitely bad.

    Parameters
    ----------
    log_posterior : callable
        Function mapping a parameter vector of shape ``(n_dim,)`` to the (unnormalized) log-posterior.
    theta0 : numpy.ndarray
        Starting parameter vector.
    maxiter : int, optional
        Maximum number of Nelder-Mead iterations.

    Returns
    -------
    numpy.ndarray
        The best parameter vector found. This is ``theta0`` itself if the search does not improve on it,
        so the result is never worse than the starting point.
    """
    theta0 = np.asarray(theta0, dtype=float)
    scale = np.where(theta0 != 0.0, np.abs(theta0), 1.0)

    def _objective(u: np.ndarray) -> float:
        value = log_posterior(theta0 + scale * u)
        return -value if np.isfinite(value) else 1e300

    result = minimize(
        _objective,
        np.zeros_like(theta0),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-8, "adaptive": True},
    )

    if result.fun < _objective(np.zeros_like(theta0)):
        return theta0 + scale * result.x
    return theta0


# --------------------------------------- #
# Mathematics / Statistics Utilities      #
# --------------------------------------- #
def compute_gelman_rubin_rhat(x: np.ndarray) -> np.ndarray:
    r"""
    Compute the Gelman-Rubin :math:`\hat{R}` statistic for MCMC convergence diagnostics.

    The Gelman-Rubin :math:`\hat{R}` statistic compares the variance between multiple
    chains to the variance within each chain to assess convergence. Values of
    :math:`\hat{R}` close to 1 indicate convergence, while values significantly
    greater than 1 suggest that the chains have not yet converged. For more details,
    see :footcite:t:`gelman1992inference`.

    Parameters
    ----------
    x: np.ndarray
        Array of shape (n, m, dim) containing samples from m chains of length n.

    Returns
    -------
    np.ndarray
        Array of shape (dim,) containing Rhat values.

    Notes
    -----
    Consider a set of :math:`M` Markov chains, each of length :math:`N`, sampling from the
    same target distribution. Let :math:`\theta_{ij}` denote the :math:`j`-th sample from the
    :math:`i`-th chain. The Gelman-Rubin :math:`\hat{R}` statistic is computed as follows:

    First, compute the within-chain variance :math:`W` and the between-chain variance :math:`B`:

    .. math::

        W = \frac{1}{M} \sum_{i=1}^{M} s_i^2,

    .. math::

        B = \frac{N}{M - 1} \sum_{i=1}^{M} (\bar{\theta}_i - \bar{\theta})^2,

    where :math:`s_i^2` is the sample variance of chain :math:`i`, :math:`\bar{\theta}_i` is the mean of chain
    :math:`i`, and :math:`\bar{\theta}` is the overall mean across all chains.

    With these two statistics, we take a weighted average to estimate the marginal posterior variance
    :math:`\hat{V}`:

    .. math::

        \hat{V} = \frac{N - 1}{N} W + \frac{1}{N} B.

    The Gelman-Rubin :math:`\hat{R}` statistic is then given by:

    .. math::

        \hat{R} = \sqrt{\frac{\hat{V}}{W}}.

    References
    ----------
    .. footbibliography::

    """
    # Extract the number of chains, samples and dimensions from
    # the provided array.
    nsamp, nchain, ndim = x.shape

    # Compute the chain means and variances.
    chain_mean, chain_var = np.mean(x, axis=0), np.var(x, axis=0, ddof=1)  # (m, dim)

    # Compute the average within-chain variance parameter W. This is the
    # first component of the Rhat calculation.
    W = np.mean(chain_var, axis=0)  # (dim,)

    # Compute the mean of the chain means. This will play a role in
    # the between chain variance calculation.
    mean_of_means = np.mean(chain_mean, axis=0)

    B = nsamp * np.sum((chain_mean - mean_of_means) ** 2, axis=0) / (nchain - 1)

    # Compute the var_hat parameter from B and W. This amounts to a weighted
    # average of the within-chain and between-chain variances based on the
    # number of samples per chain.
    var_hat = (nsamp - 1) / nsamp * W + B / nsamp
    return np.sqrt(var_hat / W)
