"""
Synchrotron Spectral Energy Distributions (SEDs).

This module implements a comprehensive library of **phenomenological synchrotron
spectral energy distributions (SEDs)** for power-law electron populations, following
the standard theoretical framework developed in the literature (e.g. :footcite:t:`GranotSari2002SpectralBreaks`)

The core design philosophy is to construct SEDs via **log-space SED surgery**:
complex spectra are assembled by multiplying (adding in log-space) a sequence of
*scale-free, smoothed broken power-law* factors. Each factor introduces a controlled
change in spectral slope at a characteristic frequency without altering the overall
normalization. This approach ensures:

- numerical stability over many decades in frequency,
- clean separation of spectral segments,
- correct asymptotic slopes,
- and composability of multiple spectral breaks.

The module supports:

- non-cooling, slow-cooling, and fast-cooling synchrotron regimes,
- synchrotron self-absorption (SSA), including stratified SSA cases,
- hidden or absorbed cooling breaks,
- smooth or sharp spectral transitions,
- automatic regime determination from physical parameters,
- and analytic closure relations linking phenomenological SED parameters to
  physical quantities (e.g. magnetic field strength, radius).

.. note::

    A complete theoretical discussion of SED construction, including derivations
    and physical interpretations, is provided in the :ref:`synch_sed_theory`. A
    user-guide description of this module can be found in :ref:`synchrotron_seds`, including
    usage examples.

"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from astropy import units as u

from triceratops.radiation.constants import (
    electron_rest_energy_cgs,
    electron_rest_mass_cgs,
)
from triceratops.utils.log import triceratops_logger
from triceratops.utils.misc_utils import ensure_in_units

from .closure import _compute_ssa_BR_from_spectrum_dm22
from .core import _opt_compute_log_synch_frequency
from .microphysics import (
    _opt_normalize_BPL_from_magnetic_field,
    _opt_normalize_PL_from_magnetic_field,
)
from .utils import (
    _log_chi_cgs,
    _log_chi_cgs_iso,
    c_1_cgs,
    compute_c5_parameter,
    compute_c6_parameter,
)

# Type checking imports
if TYPE_CHECKING:
    from triceratops._typing import (
        _ArrayLike,
        _UnitBearingArrayLike,
        _UnitBearingScalarLike,
    )


# ============================================================= #
# SED Functions                                                 #
# ============================================================= #
# This is where we implement all of the low-level SED functions that
# can be used to draw the correct SED. These functions should all be
# implemented in log-space for numerical stability.
#
# These can be used directly, but are typically wrapped in the corresponding
# SED class. PLEASE DOCUMENT ALL OF THESE FUNCTIONS THOROUGHLY. The corresponding
# theory is complex.
def log_smoothed_SFBPL(
    log_x: "_ArrayLike",
    a1: float,
    a2: float,
    smoothing: float,
):
    r"""
    Logarithm of a scale-free smoothed broken power law (SFBPL).

    This function returns the **logarithm** of a *scale-free*, multiplicative
    spectral break factor commonly used for constructing piecewise spectra via
    **log-space SED surgery** (i.e. multiplication in linear space, addition in
    log space).

    The factor is

    .. math::

        \tilde f(x)
        =
        \left[
            1 + x^{(a_2 - a_1)/s}
        \right]^s,

    where :math:`x` is a dimensionless ratio (e.g. :math:`x=\nu/\nu_{\rm brk}`),
    :math:`a_1` and :math:`a_2` are the asymptotic power-law indices on opposite
    sides of the break, and :math:`s` controls the smoothness.

    The returned quantity is

    .. math::

        \log \tilde f(x)
        =
        s \log\!\left(1 + x^{(a_2 - a_1)/s}\right).

    Parameters
    ----------
    log_x : array-like
        Logarithm of the dimensionless ratio :math:`x`, typically
        :math:`\log(\nu/\nu_{\rm brk})`.
    a1 : float
        Reference spectral slope used in the definition of the break.
        In typical usage, this is the **baseline slope** you already have on one
        side of the break.
    a2 : float
        Target spectral slope after applying the break on the other side.
        The slope change across the break is :math:`\Delta a = a_2 - a_1`.
    smoothing : float
        Smoothness parameter :math:`s`. The magnitude :math:`|s|` sets the
        sharpness of the transition: smaller :math:`|s|` gives a sharper break.
        The **sign** of :math:`s` controls *which* side of the break receives the
        slope change (i.e. whether the modification acts primarily at
        :math:`x \ll 1` or :math:`x \gg 1`).

    Returns
    -------
    array-like
        :math:`\log \tilde f(x)` evaluated at ``log_x``.

    Notes
    -----
    - **Scale-free:** :math:`\tilde f(1)=2^s` and no additional scale parameters
      are introduced besides the implicit break location in :math:`x`.
    - Intended to be **added** to an existing log-spectrum:
      if :math:`\log F(\nu)` is your baseline, then
      :math:`\log F'(\nu) = \log F(\nu) + \log \tilde f(\nu/\nu_{\rm brk})`.
    - Multiple factors may be composed, but you should ensure each break is
      applied with the correct baseline slopes and does not unintentionally
      re-modify the same frequency range.

    .. rubric:: Asymptotic behavior and slope surgery

    Let :math:`k=(a_2-a_1)/s`. Then:

    - If :math:`k>0` (the common case for "low side → a1, high side → a2"):

      - For :math:`x \ll 1`: :math:`\tilde f(x) \to 1`
      - For :math:`x \gg 1`: :math:`\tilde f(x) \propto x^{\,a_2-a_1}`

      Adding :math:`\log\tilde f` to a baseline power law with slope :math:`a_1`
      produces slope :math:`a_1` below the break and slope :math:`a_2` above it.

    - If :math:`k<0`, the roles of :math:`x\ll 1` and :math:`x\gg 1` swap:
      the slope modification acts on the **low-**:math:`x` side instead of the
      high-:math:`x` side.

    In all cases, the multiplicative factor produces a net slope change of
    :math:`\Delta a = a_2-a_1` across the transition.
    """
    return smoothing * (np.logaddexp(0, ((a2 - a1) / smoothing) * log_x) - np.log(2))


def log_exp_cutoff_sed(
    log_x: "_ArrayLike",
):
    r"""
    Logarithm of a smooth high-frequency exponential cutoff.

    This function implements a *scale-free*, phenomenological cutoff factor

    .. math::

        \Phi(x)
        =
        x^{1/2}\,\exp(1 - x),

    where :math:`x \equiv \nu / \nu_{\max}`. The logarithm of this factor is

    .. math::

        \log \Phi(x)
        =
        \frac{1}{2}\log x + (1 - x).

    The cutoff is normalized such that :math:`\Phi(1) = 1`, allowing it to be
    applied multiplicatively to an existing SED without altering its
    normalization below the cutoff frequency.

    Parameters
    ----------
    log_x : array-like
        Logarithm of the dimensionless frequency ratio
        :math:`\log(\nu / \nu_{\max})`.

    Returns
    -------
    array-like
        Logarithm of the exponential cutoff factor.

    Notes
    -----
    - For :math:`\nu \ll \nu_{\max}`, the cutoff approaches unity.
    - For :math:`\nu > \nu_{\max}`, the spectrum is exponentially suppressed.
    - The conditional expression ensures that the cutoff contributes only
      above :math:`\nu_{\max}`, preserving exact scale-freeness below the
      cutoff.
    - Implemented entirely in log-space for numerical stability.
    """
    return np.where(log_x > 0, 0.5 * log_x + (1.0 - np.exp(log_x)), 0.0)


def smoothed_BPL(nu, F_nu, nu_brk, alpha_1, alpha_2, smoothing):
    r"""
    Smoothed broken power-law (BPL) SED.

    This function constructs a smoothed broken power-law SED by combining
    two power-law segments with a smooth transition at a specified break
    frequency.

    Parameters
    ----------
    nu : array-like
        Frequencies at which to evaluate the SED.
    F_nu : float
        Normalization of the SED at the break frequency.
    nu_brk : float
        Break frequency where the spectral slope changes.
    alpha_1 : float
        Spectral slope for frequencies below the break.
    alpha_2 : float
        Spectral slope for frequencies above the break.
    smoothing : float
        Smoothness parameter controlling the width of the transition.

    Returns
    -------
    array-like
        The smoothed broken power-law SED evaluated at ``nu``.

    Notes
    -----
    - The SED is normalized such that :math:`F_\nu(\nu_{\rm brk}) = F_{\nu}`.
    - The smoothness parameter controls how gradual the transition is between
      the two power-law segments.
    """
    return F_nu * ((nu / nu_brk) ** (alpha_1 / smoothing) + (nu / nu_brk) ** (alpha_2 / smoothing)) ** smoothing


# --- Power-Law No-Cooling No-SSA SEDs --- #
def _log_powerlaw_sbpl_sed(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Logarithm of the synchrotron SED for a non-cooling power-law electron population.

    This function implements the **non-cooling**, optically thin synchrotron
    spectrum for a power-law electron distribution, with no synchrotron
    self-absorption (SSA) and including a high-frequency exponential cutoff.

    The optically thin spectral slopes are:

    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for :math:`\nu > \nu_m`.

    The spectrum is constructed using *scale-free smoothed broken power laws*
    (SFBPLs) via log-space SED surgery and is **normalized at the injection
    frequency** :math:`\nu_m`.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection (minimum electron) frequency :math:`\nu_m`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency :math:`\nu_{\max}`.
    p : float
        Power-law index of the electron energy distribution.
    s : float
        Smoothness parameter for the SFBPL transition. The magnitude controls
        the sharpness of the break, while the sign controls the direction of
        the slope change.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED evaluated at ``log_nu``.

    Notes
    -----
    - No synchrotron self-absorption (SSA) is included.
    - The high-frequency exponential cutoff is applied multiplicatively.
    """
    x_m = log_nu - log_nu_m
    x_max = log_nu - log_nu_max

    # Normalize at the injection frequency ν_m
    log_sed = (1.0 / 3.0) * x_m

    # Injection break
    log_sed += log_smoothed_SFBPL(x_m, 1.0 / 3.0, -(p - 1.0) / 2.0, s)

    # High-frequency cutoff
    log_sed += log_exp_cutoff_sed(x_max)

    return log_sed


# --- Power-Law Cooling SEDs --- #
def _log_powerlaw_sbpl_sed_cool_1(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_c: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Logarithm of the synchrotron SED in the **fast-cooling** regime.

    This function implements the fast-cooling synchrotron spectrum for a
    power-law electron distribution, assuming the frequency ordering

    .. math::

        \nu_c < \nu_m.

    The optically thin spectral slopes are:

    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu < \nu_c`,
    - :math:`F_\nu \propto \nu^{-1/2}` for :math:`\nu_c < \nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_m`.

    The spectrum is constructed via log-space SED surgery using SFBPL kernels
    and is **normalized at the cooling frequency** :math:`\nu_c`.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_c : float
        Logarithm of the cooling frequency :math:`\nu_c`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency :math:`\nu_{\max}`.
    p : float
        Power-law index of the electron energy distribution.
    s : float
        Smoothness parameter for the SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED evaluated at ``log_nu``.

    Notes
    -----
    - Assumes :math:`\nu_c < \nu_m`; no internal validation is performed.
    - No synchrotron self-absorption (SSA) is included.
    """
    x_c = log_nu - log_nu_c
    x_m = log_nu - log_nu_m
    x_max = log_nu - log_nu_max

    # Normalize at the cooling frequency ν_c
    log_sed = (1.0 / 3.0) * x_c

    # Cooling break
    log_sed += log_smoothed_SFBPL(x_c, 1.0 / 3.0, -1.0 / 2.0, s)

    # Injection break
    log_sed += log_smoothed_SFBPL(x_m, -1.0 / 2.0, -p / 2.0, s)

    # High-frequency cutoff
    log_sed += log_exp_cutoff_sed(x_max)

    return log_sed


def _log_powerlaw_sbpl_sed_cool_2(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_c: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Logarithm of the synchrotron SED in the **slow-cooling** regime.

    This function implements the slow-cooling synchrotron spectrum for a
    power-law electron distribution, assuming the frequency ordering

    .. math::

        \nu_m < \nu_c.

    The optically thin spectral slopes are:

    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for
      :math:`\nu_m < \nu < \nu_c`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_c`.

    The spectrum is constructed via log-space SED surgery using SFBPL kernels
    and is **normalized at the injection frequency** :math:`\nu_m`.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_c : float
        Logarithm of the cooling frequency :math:`\nu_c`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency :math:`\nu_{\max}`.
    p : float
        Power-law index of the electron energy distribution.
    s : float
        Smoothness parameter for the SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED evaluated at ``log_nu``.

    Notes
    -----
    - Assumes :math:`\nu_m < \nu_c`; no internal validation is performed.
    - No synchrotron self-absorption (SSA) is included.
    """
    x_m = log_nu - log_nu_m
    x_c = log_nu - log_nu_c
    x_max = log_nu - log_nu_max

    # Normalize at the injection frequency ν_m
    log_sed = (1.0 / 3.0) * x_m

    # Injection break
    log_sed += log_smoothed_SFBPL(x_m, 1.0 / 3.0, -(p - 1.0) / 2.0, s)

    # Cooling break
    log_sed += log_smoothed_SFBPL(x_c, -(p - 1.0) / 2.0, -p / 2.0, s)

    # High-frequency cutoff
    log_sed += log_exp_cutoff_sed(x_max)

    return log_sed


class _SynchrotronCoolingSEDFunctions(Enum):
    SPECTRUM_1 = _log_powerlaw_sbpl_sed_cool_1  # fast-cooling
    SPECTRUM_2 = _log_powerlaw_sbpl_sed_cool_2  # slow-cooling
    SPECTRUM_3 = _log_powerlaw_sbpl_sed  # non-cooling


# --- Power-Law SSA SEDs --- #
def _log_powerlaw_sbpl_sed_ssa_1(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_a: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Logarithm of the synchrotron SED with SSA, assuming the ordering :math:`\nu < \nu_a < \nu_m`.

    The resulting spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu_a < \nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for :math:`\nu > \nu_m`.

    In this configuration, the SSA break lies below the injection frequency.
    The spectrum is therefore **normalized at the injection break**
    :math:`\nu_m`, where the optically thin synchrotron slope is well defined.
    The SSA turnover is introduced at lower frequencies using a scale-free
    smoothed broken power-law (SFBPL) factor.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection (minimum electron) frequency
        :math:`\nu_m`.
    log_nu_a : float
        Logarithm of the synchrotron self-absorption frequency
        :math:`\nu_a`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency
        :math:`\nu_{\max}`.
    p : float
        Power-law index of the electron energy distribution.
    s : float
        Smoothness parameter for the SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED evaluated at ``log_nu``.

    Notes
    -----
    - Assumes :math:`\nu < \nu_a < \nu_m`; no internal validation is performed.
    - The spectrum is constructed via log-space SED surgery using scale-free
      SFBPL kernels.
    """
    x_a = log_nu - log_nu_a
    x_m = log_nu - log_nu_m
    x_max = log_nu - log_nu_max

    # Normalize at the injection frequency ν_m
    log_sed = 2.0 * x_a + (1.0 / 3.0) * (log_nu_a - log_nu_m)

    # SSA break (optically thick → optically thin)
    log_sed += log_smoothed_SFBPL(x_a, 2.0, 1.0 / 3.0, s)

    # Injection break
    log_sed += log_smoothed_SFBPL(x_m, 1.0 / 3.0, -(p - 1.0) / 2.0, s)

    # High-frequency cutoff
    log_sed += log_exp_cutoff_sed(x_max)

    return log_sed


def _log_powerlaw_sbpl_sed_ssa_2(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_a: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Logarithm of the synchrotron SED with SSA, assuming the ordering :math:`\nu < \nu_m < \nu_a`.

    The resulting spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{5/2}` for :math:`\nu_a < \nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for :math:`\nu > \nu_a`.

    In this case, the SSA break lies above the injection frequency and the
    intermediate :math:`\nu^{1/3}` segment is absent. The spectrum is therefore
    **normalized at the injection frequency** :math:`\nu_m` and propagated
    downward to the SSA break using the appropriate optically thick slope.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_a : float
        Logarithm of the synchrotron self-absorption frequency
        :math:`\nu_a`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency
        :math:`\nu_{\max}`.
    p : float
        Power-law index of the electron energy distribution.
    s : float
        Smoothness parameter for the SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED evaluated at ``log_nu``.

    Notes
    -----
    - Assumes :math:`\nu < \nu_m < \nu_a`; no internal validation is performed.
    - The increase in slope from :math:`\nu^2` to :math:`\nu^{5/2}` requires
      flipping the sign of the smoothing parameter due to the adopted SFBPL
      convention.
    """
    x_m = log_nu - log_nu_m
    x_a = log_nu - log_nu_a
    x_max = log_nu - log_nu_max

    # Normalize at the injection frequency ν_m
    log_sed = 2.0 * x_m + (5.0 / 2.0) * (log_nu_m - log_nu_a)

    # SSA break (slope increase requires -s in this convention)
    log_sed += log_smoothed_SFBPL(x_m, 2.0, 5.0 / 2.0, -s)

    # Injection break
    log_sed += log_smoothed_SFBPL(x_a, 5.0 / 2.0, -(p - 1.0) / 2.0, s)

    # High-frequency cutoff
    log_sed += log_exp_cutoff_sed(x_max)

    return log_sed


class _SynchrotronSSASEDFunctions(Enum):
    SPECTRUM_1 = _log_powerlaw_sbpl_sed_ssa_1
    SPECTRUM_2 = _log_powerlaw_sbpl_sed_ssa_2


# --- Power-Law SSA + Cooling SEDs --- #
# NOTE: Spectra 1 and 2 are identical to the non-cooling SSA cases because
# the cooling break lies above the maximum synchrotron frequency.
# They are therefore not implemented separately here.
def _log_powerlaw_sbpl_sed_ssa_cool_3(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_c: float,
    log_nu_a: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Synchrotron SED with SSA in the **slow-cooling** regime, assuming the ordering :math:`\nu < \nu_a < \nu_m < \nu_c`.

    The resulting spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu_a < \nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for
      :math:`\nu_m < \nu < \nu_c`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_c`.

    The spectrum is **normalized at the injection frequency** :math:`\nu_m`,
    which is the dominant optically thin break in the slow-cooling regime.
    Synchrotron self-absorption and cooling breaks are then applied via
    scale-free smoothed broken power-law (SFBPL) factors.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_c : float
        Logarithm of the cooling frequency :math:`\nu_c`.
    log_nu_a : float
        Logarithm of the synchrotron self-absorption frequency :math:`\nu_a`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency.
    p : float
        Electron energy distribution index.
    s : float
        Smoothness parameter for SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED.
    """
    x_a = log_nu - log_nu_a
    x_m = log_nu - log_nu_m
    x_c = log_nu - log_nu_c
    x_max = log_nu - log_nu_max

    # Normalize at ν_m
    log_sed = 2.0 * x_a  # +(1.0 / 3.0) * (log_nu_a - log_nu_m)

    # SSA break
    log_sed += log_smoothed_SFBPL(x_a, 2.0, 1.0 / 3.0, s)

    # Injection break
    log_sed += log_smoothed_SFBPL(x_m, 1.0 / 3.0, -(p - 1.0) / 2.0, s)

    # Cooling break
    log_sed += log_smoothed_SFBPL(x_c, -(p - 1.0) / 2.0, -p / 2.0, s)

    log_sed += log_exp_cutoff_sed(x_max)
    return log_sed


def _log_powerlaw_sbpl_sed_ssa_cool_4(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_c: float,
    log_nu_a: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Synchrotron SED with SSA in the **slow-cooling** regime, assuming the ordering :math:`\nu < \nu_m < \nu_a < \nu_c`.

    The spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{5/2}` for :math:`\nu_m < \nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for
      :math:`\nu_a < \nu < \nu_c`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_c`.

    The spectrum is **normalized at the injection frequency** :math:`\nu_m`.
    The increase in slope from :math:`\nu^2` to :math:`\nu^{5/2}` requires
    flipping the sign of the smoothing parameter due to the adopted SFBPL
    convention.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED.
    """
    x_m = log_nu - log_nu_m
    x_a = log_nu - log_nu_a
    x_c = log_nu - log_nu_c
    x_max = log_nu - log_nu_max

    log_sed = 2.0 * x_m

    # RJ → optically thick SSA (slope increase)
    log_sed += log_smoothed_SFBPL(x_m, 2.0, 5.0 / 2.0, -s)

    # SSA → optically thin uncooled
    log_sed += log_smoothed_SFBPL(x_a, 5.0 / 2.0, -(p - 1.0) / 2.0, s)
    log_sed -= (5.0 / 2.0) * (log_nu_a - log_nu_m)

    # Cooling break
    log_sed += log_smoothed_SFBPL(x_c, -(p - 1.0) / 2.0, -p / 2.0, s)

    log_sed += log_exp_cutoff_sed(x_max)
    return log_sed


def _log_powerlaw_sbpl_sed_ssa_cool_5(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_c: float,
    log_nu_a: float,
    log_nu_ac: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Synchrotron SED with synchrotron self-absorption (SSA) in the **fast-cooling** regime.

    Includes stratified absorption, assuming the ordering

    .. math::

        \nu < \nu_{ac} < \nu_a < \nu_c < \nu_m.

    In this regime, electrons cool efficiently below the injection energy, and
    synchrotron self-absorption occurs within a stratified, cooling electron
    population behind the shock. This produces an additional low-frequency break
    at :math:`\nu_{ac}`, separating optically thick emission from uncooled and
    cooled electron layers.

    The resulting spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_{ac}`,
    - :math:`F_\nu \propto \nu^{11/8}` for :math:`\nu_{ac} < \nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu_a < \nu < \nu_c`,
    - :math:`F_\nu \propto \nu^{-1/2}` for :math:`\nu_c < \nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_m`.

    The spectrum is **anchored at the cooling frequency** :math:`\nu_c`, which is
    the dominant physical break in the fast-cooling regime. Lower-frequency
    structure is constructed via scale-free smoothed broken power laws (SFBPLs)
    to ensure continuity and correct asymptotic behavior.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_c : float
        Logarithm of the cooling frequency :math:`\nu_c`.
    log_nu_a : float
        Logarithm of the synchrotron self-absorption frequency :math:`\nu_a`.
    log_nu_ac : float
        Logarithm of the stratified absorption break frequency :math:`\nu_{ac}`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency.
    p : float
        Electron energy distribution index.
    s : float
        Smoothness parameter for SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED.

    Notes
    -----
    - Assumes fast cooling (:math:`\nu_c \ll \nu_m`) and the stated frequency
      ordering; no internal validation is performed.
    - The :math:`11/8` slope arises from stratified synchrotron self-absorption in
      a cooling electron population.
    """
    # Determine the nu/nu_a, nu/nu_m, and nu/nu_c ratios
    x_a, _, x_m, x_ac, x_max = (
        log_nu - log_nu_a,
        log_nu - log_nu_c,
        log_nu - log_nu_m,
        log_nu - log_nu_ac,
        log_nu - log_nu_max,
    )

    # This is a fast-cooled spectrum, so the dominant break is at nu_c and that
    # is where we anchor the spectrum. The transition at the cooling break is from SPL
    # E to SPL F (1/3 to -1/2) corresponding from the transition from the uncooled RJ tail to
    # the cooled population.
    log_sed = 2 * x_ac + (11 / 8) * (log_nu_ac - log_nu_a) + (1 / 3) * (log_nu_a - log_nu_c)

    # We now add the nu_ac break which corresponds to the transition from SPL B to SPL C
    # (2 to 11/8). This is the transition from the RJ tail to the cooled SSA segment.
    log_sed += log_smoothed_SFBPL(x_ac, 2.0, 11 / 8, s)

    # Add the transition from the 11/8 nu_ac stratified segment to the RJ tail at the
    # absorption break nu_a (11/8 to 1/3; SPL C to SPL E).
    log_sed += log_smoothed_SFBPL(x_a, 11 / 8, 1 / 3, s)

    # Finally, we need to add in the injection break at nu_m. This is SPL F -> SPL H
    # (-1/2 to -p/2)
    log_sed += log_smoothed_SFBPL(x_m, -1 / 2, -p / 2, s)
    # Truncate
    log_sed += log_exp_cutoff_sed(x_max)
    return log_sed


def _log_powerlaw_sbpl_sed_ssa_cool_6(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_a: float,
    log_nu_ac: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Synchrotron SED with SSA assuming the ordering :math:`\nu < \nu_{ac} < \nu_a < \nu_m`.

    In this configuration, the cooling break lies *above* the self-absorption
    photosphere and is therefore not directly visible in the emergent spectrum.
    The observed emission is dominated by optically thick and marginally thin
    synchrotron radiation from a cooling electron population.

    The resulting spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_{ac}`,
    - :math:`F_\nu \propto \nu^{11/8}` for :math:`\nu_{ac} < \nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{-1/2}` for :math:`\nu_a < \nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_m`.

    Although the system is fast cooling, the cooling break itself does not appear
    explicitly because it is obscured by synchrotron self-absorption. The spectrum
    is therefore constructed by anchoring at the absorption break and propagating
    to higher frequencies using the fast-cooling optically thin slopes.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_a : float
        Logarithm of the synchrotron self-absorption frequency :math:`\nu_a`.
    log_nu_ac : float
        Logarithm of the stratified absorption break frequency :math:`\nu_{ac}`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency.
    p : float
        Electron energy distribution index.
    s : float
        Smoothness parameter for SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED.

    Notes
    -----
    - Assumes fast cooling with the cooling break hidden by SSA.
    - The :math:`11/8` segment reflects stratified absorption in a cooling flow.
    """
    # Determine the nu/nu_a, nu/nu_m, and nu/nu_c ratios.
    x_a, x_m, x_ac, x_max = log_nu - log_nu_a, log_nu - log_nu_m, log_nu - log_nu_ac, log_nu - log_nu_max

    # This is a fast-cooling spectrum, so we normalize at the cooling break, but use
    # the power-law propagation technique to actually place the anchor point at nu_a instead.
    # This is SPL C to SPL F (11/8 to -1/2).
    log_sed = 2 * x_ac + (11 / 8) * (log_nu_ac - log_nu_a)

    # We now add the nu_ac break which corresponds to the transition from SPL B to SPL C
    # (2 to 11/8). This is the transition from the RJ tail to the cooled SSA segment.
    log_sed += log_smoothed_SFBPL(x_ac, 2.0, 11 / 8, s)

    # Add the transition from 11/8 to -1/2 at the absorption break nu_a (SPL C to SPL F).
    log_sed += log_smoothed_SFBPL(x_a, 11 / 8, -1 / 2, s)

    # Now add on the injection break at nu_m. This is SPL F -> SPL H. Because we are
    # fast cooling, the high frequency slope is -p/2.
    log_sed += log_smoothed_SFBPL(x_m, -1 / 2, -p / 2, s)
    # Truncate
    log_sed += log_exp_cutoff_sed(x_max)
    return log_sed


def _log_powerlaw_sbpl_sed_ssa_cool_7(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_a: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Synchrotron SED with SSA in the **fast-cooling** regime, assuming the ordering :math:`\nu < \nu_m < \nu_a`.

    In this case, both the cooling break and any stratified absorption structure
    are hidden beneath the synchrotron self-absorption photosphere. The observed
    spectrum transitions directly from optically thick emission to optically thin
    fast-cooled synchrotron radiation.

    The resulting spectral segments are:

    - :math:`F_\nu \propto \nu^2` for :math:`\nu < \nu_m`,
    - :math:`F_\nu \propto \nu^{5/2}` for :math:`\nu_m < \nu < \nu_a`,
    - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_a`.

    The spectrum is **anchored at the injection frequency** :math:`\nu_m`, since
    all cooling-related breaks occur at higher frequencies and do not affect the
    low-frequency emission. The SSA break at :math:`\nu_a` directly connects the
    optically thick synchrotron emission to the fast-cooled optically thin regime.

    Parameters
    ----------
    log_nu : array-like
        Logarithm of the frequencies at which to evaluate the SED.
    log_nu_m : float
        Logarithm of the injection frequency :math:`\nu_m`.
    log_nu_a : float
        Logarithm of the synchrotron self-absorption frequency :math:`\nu_a`.
    log_nu_max : float
        Logarithm of the maximum synchrotron frequency.
    p : float
        Electron energy distribution index.
    s : float
        Smoothness parameter for SFBPL transitions.

    Returns
    -------
    array-like
        Logarithm of the synchrotron SED.

    Notes
    -----
    - This regime corresponds to extreme self-absorption in a fast-cooling system.
    - No explicit cooling break appears in the observable spectrum.
    """
    # Determine the nu/nu_a and nu/nu_m ratios
    x_a, x_m, x_max = log_nu - log_nu_a, log_nu - log_nu_m, log_nu - log_nu_max

    # We anchor at the injection break because the cooling break (if present) is not visible behind the
    # absorption photosphere at the shock. We therefore start with the SPL B -> SPL A transition (2 to 5/2).
    log_sed = 2 * x_m + (5 / 2) * (log_nu_m - log_nu_a)

    # Now add the SSA break at nu_a which correspond to the transition from optically
    # thick SSA (A) to optically thin cooled (H) (5/2 to -p/2).
    log_sed += log_smoothed_SFBPL(x_a, 5 / 2, -p / 2, s)

    # Now we need to add in the injection break at nu_m.
    log_sed += log_smoothed_SFBPL(x_m, 2, 5 / 2, -s)

    # Truncate
    log_sed += log_exp_cutoff_sed(x_max)
    return log_sed


class _SynchrotronSSACoolingSEDFunctions(Enum):
    SPECTRUM_1 = _log_powerlaw_sbpl_sed_ssa_1
    SPECTRUM_2 = _log_powerlaw_sbpl_sed_ssa_2
    SPECTRUM_3 = _log_powerlaw_sbpl_sed_ssa_cool_3
    SPECTRUM_4 = _log_powerlaw_sbpl_sed_ssa_cool_4
    SPECTRUM_5 = _log_powerlaw_sbpl_sed_ssa_cool_5
    SPECTRUM_6 = _log_powerlaw_sbpl_sed_ssa_cool_6
    SPECTRUM_7 = _log_powerlaw_sbpl_sed_ssa_cool_7


# =============================================================
# SED Base Class
# =============================================================
# To help compartmentalize the name space and prevent any issues
# with clarity, we provide a small base class for SEDs to serve as a
# guide both for SED implementation and for documentation.
class SynchrotronSED(ABC):
    """
    Base class for synchrotron SED implementation.

    The :class:`SynchrotronSED` is a simple compartment for defining the
    structure of a specific spectral energy distribution. For each SED,
    one needs to provide

    1. The :meth:`sed` method (vis-a-vis the low-level ``_log_opt_sed`` method), which
       simply provides the phenomenological SED shape as a function of frequency and parameters. For example
       a broken power-law SED would implement the appropriate power-law segments and breaks in this method.

    2. The :meth:`from_params_to_physics` (vis-a-vis the low-level ``_opt_from_params_to_physics`` method), which
       provides the mapping from phenomenological SED parameters (e.g., break frequencies, flux normalizations, etc.)
       to physical parameters (e.g., magnetic field strength, radius, etc.) based on closure relations.
    3. The :meth:`from_physics_to_params` (vis-a-vis the low-level ``_opt_from_physics_to_params`` method), which
       provides the mapping from physical parameters (e.g., magnetic field strength, radius, etc.) to phenomenological
       SED parameters (e.g., break frequencies, flux normalizations, etc.) based on closure relations.

    Components (2) and (3) are **NOT REQUIRED** for an SED to be functional; however, they are generally necessary
    for inference. The necessary extent of the infrastructure is left to the implementer.

    .. hint::

        All of the existing SEDs are implemented using this base class as a guide. If you're
        in need of check out the other classes in this module.

    Instantiation
    --------------
    Instantiation of the SED class may be used to pre-compute class-wide constants, evaluate or construct kernels,
    etc; however, it should **NOT** be used to store SED-specific parameters. All SED-specific parameters
    should be passed directly to the relevant methods. This ensures that, during high-load inference tasks,
    no unnecessary re-instantiation of SED objects is required.
    """

    # ============================================================ #
    # Instantiation and basic structure                            #
    # ============================================================ #
    def __init__(self, *args, **kwargs):
        r"""
        Instantiate the SED object.

        Parameters
        ----------
        args:
            Positional arguments for SED instantiation.
        kwargs:
            Keyword arguments for SED instantiation.

        Notes
        -----
        The SED instantiation should NOT be used to store SED-specific parameters. All SED-specific parameters
        should be passed directly to the relevant methods. This ensures that, during high-load inference tasks,
        no unnecessary re-instantiation of SED objects is required.
        """
        pass

    def __call__(self, nu, **parameters):
        r"""
        Evaluate the SED at the given frequency.

        This is a thin wrapper around :meth:`sed` and exists purely for
        convenience, allowing SED objects to be used as callables.
        """
        return self.sed(nu, **parameters)

    def __repr__(self):
        return f"<{self.__class__.__name__}()>"

    # ============================================================ #
    # SED Function Implementation                                  #
    # ============================================================ #
    # Here should be the implementation of the SED function itself,
    # which is a function of nu and some set of additional parameters.
    @abstractmethod
    def _log_opt_sed(self, nu: "_ArrayLike", **parameters):
        r"""
        Low-level optimized log-space SED evaluation.

        This method implements the core numerical kernel for the synchrotron
        spectral energy distribution (SED) in **logarithmic space**. It is intended
        for performance-critical use and therefore assumes that all inputs are
        already provided in a validated, unit-consistent form.

        No unit handling, type checking, or safety checks are performed.

        Parameters
        ----------
        nu : float or array-like
            Natural logarithm of the frequency at which to evaluate the SED,

            .. math::

                \nu \equiv \ln\!\left(\frac{\nu_{\rm phys}}{\mathrm{Hz}}\right),

            where :math:`\nu_{\rm phys}` is the physical frequency expressed in
            Hz-equivalent CGS units.
        **parameters
            Additional dimensionless or CGS-valued parameters required for the
            SED calculation. The exact set of required parameters is
            implementation-specific.

        Returns
        -------
        float or array-like
            Natural logarithm of the synchrotron SED evaluated at the specified
            frequency.
        """
        raise NotImplementedError

    @abstractmethod
    def sed(self, nu: "_UnitBearingArrayLike", **parameters):
        r"""
        User-facing synchrotron SED evaluation.

        This method provides a high-level, user-friendly interface for computing
        the synchrotron spectral energy distribution (SED). It is responsible for
        handling unit validation and coercion, basic shape checking, and any other
        user-facing conveniences before dispatching to the low-level optimized
        backend.

        Internally, this method should convert inputs into the dimensionless,
        log-space form expected by the optimized implementation
        (:meth:`_log_opt_sed`).

        Parameters
        ----------
        nu : float, array-like, or astropy.units.Quantity
            Frequency at which to evaluate the SED. If provided without units,
            frequencies are assumed to be in Hz. If provided as an
            :class:`astropy.units.Quantity`, the value will be converted to
            Hz-equivalent CGS units before evaluation.

            The frequency may be specified as a scalar (to evaluate a single
            spectrum) or as a one-dimensional array (to evaluate the SED over
            a frequency grid).
        **parameters
            Additional parameters required for the SED calculation. These may
            include phenomenological SED parameters (e.g. break frequencies,
            normalization constants) or physical model parameters, depending
            on the specific SED implementation.

        Returns
        -------
        float, array-like, or astropy.units.Quantity
            The synchrotron SED evaluated at the specified frequency. Implementations
            may return either plain numerical values or
            :class:`astropy.units.Quantity` objects with appropriate physical units.

        Notes
        -----
        - Subclasses must implement this method.
        - The returned SED should be consistent with the conventions and units
          adopted by the corresponding low-level implementation.
        """
        raise NotImplementedError

    # =========================================================== #
    # Closure Relations Implementation                            #
    # =========================================================== #
    # Here we implement the closure relations to go forward and backward
    # between the physics parameters and the phenomenological SED parameters.
    def from_params_to_physics(self, **parameters):
        r"""
        Convert phenomenological SED parameters into physical parameters.

        This method provides a **user-facing interface** for mapping phenomenological
        SED parameters—such as break frequencies, peak fluxes, or normalization
        constants—into underlying physical quantities like magnetic field strength,
        emitting radius, characteristic electron energies, or energy densities.

        The mapping implemented by this method is **model-dependent** and typically
        relies on analytic closure relations derived from synchrotron theory, often
        supplemented by additional microphysical assumptions. Examples include:

        - assumptions about particle acceleration efficiency,
        - equipartition or near-equipartition between fields and particles,
        - prescriptions for the minimum Lorentz factor :math:`\gamma_m`,
        - geometric assumptions encoded via solid angle or filling factor terms.

        As a result, the inferred physical parameters should be interpreted within
        the context of the specific microphysical model adopted by the implementing
        subclass.

        This method is optional: an SED implementation is fully functional without
        it. However, closure relations are generally required for inference workflows,
        parameter estimation, and for coupling SEDs to dynamical or microphysical
        models.

        Parameters
        ----------
        **parameters
            Keyword arguments specifying phenomenological SED parameters. The exact
            set of required parameters is model-dependent and determined by the
            implementing subclass.

        Returns
        -------
        dict
            Dictionary containing inferred physical parameters. The contents,
            naming conventions, and units are implementation-specific.

        Notes
        -----
        - This method may perform unit validation, coercion, or shape checking
          before dispatching to the low-level optimized implementation
          :meth:`_opt_from_params_to_physics`.
        - The mapping is not guaranteed to be unique; degeneracies may be present
          depending on the assumed microphysics.
        """
        raise NotImplementedError

    def _opt_from_params_to_physics(self, **parameters):
        r"""
        Low-level optimized conversion from SED parameters to physical parameters.

        This method implements the same phenomenological-to-physical parameter
        mapping as :meth:`from_params_to_physics`, but assumes that all inputs are
        provided as dimensionless scalars or NumPy arrays in consistent CGS units.

        The mapping may encode analytic closure relations and microphysical
        assumptions specific to the SED model, but **no validation or safety checks**
        are performed at this level.

        Parameters
        ----------
        **parameters
            Keyword arguments specifying phenomenological SED parameters in CGS or
            dimensionless form. The exact set of required parameters is
            implementation-specific.

        Returns
        -------
        dict
            Dictionary containing inferred physical parameters in CGS units.

        Notes
        -----
        - This method is intended for internal use in performance-critical contexts.
        - It should be called only after unit handling and basic validation have been
          performed by :meth:`from_params_to_physics`.
        """
        raise NotImplementedError

    def _opt_from_physics_to_params(self, **parameters):
        r"""
        Low-level optimized conversion from physical parameters to SED parameters.

        This method implements the inverse mapping of
        :meth:`_opt_from_params_to_physics`, converting physical quantities—such as
        magnetic field strength, system size, particle energy scales, or energy
        densities—into phenomenological SED parameters like break frequencies or
        peak fluxes.

        The mapping assumes a specific set of synchrotron closure relations and
        microphysical prescriptions adopted by the implementing subclass.

        All inputs are assumed to be provided in CGS units or as dimensionless
        scalars. No unit validation, consistency checks, or physical sanity checks
        are performed.

        Parameters
        ----------
        **parameters
            Keyword arguments specifying physical parameters in CGS units. The exact
            set of required parameters is implementation-specific.

        Returns
        -------
        dict
            Dictionary containing phenomenological SED parameters in CGS or
            dimensionless form.

        Notes
        -----
        - This method is intended for internal use in performance-critical contexts.
        - Inverse mappings may not exist or may not be unique for all SED models.
        """
        raise NotImplementedError

    def from_physics_to_params(self, **parameters):
        r"""
        Convert physical parameters into phenomenological SED parameters.

        This method provides a **user-facing interface** for mapping physical
        quantities—such as magnetic field strength, emitting radius, particle
        energy scales, or energy densities—into phenomenological SED parameters
        like break frequencies, normalization constants, or spectral amplitudes.

        The conversion is based on analytic closure relations from synchrotron
        theory and typically incorporates additional microphysical assumptions
        (e.g., acceleration efficiency, geometry, or equipartition conditions)
        defined by the implementing subclass.

        This functionality is primarily used in inference workflows, where
        physical model parameters are sampled and must be translated into
        observable SED quantities.

        Parameters
        ----------
        **parameters
            Keyword arguments specifying physical parameters. The exact set of
            required parameters is model-dependent and determined by the
            implementing subclass.

        Returns
        -------
        dict
            Dictionary containing phenomenological SED parameters.

        Notes
        -----
        - This method may perform unit validation, coercion, or shape checking
          before dispatching to the low-level optimized implementation
          :meth:`_opt_from_physics_to_params`.
        - Subclasses that do not support inversion of closure relations may leave
          this method unimplemented.
        """
        raise NotImplementedError


class MultiSpectrumSynchrotronSED(SynchrotronSED, ABC):
    r"""
    Base class for synchrotron SEDs with multiple discrete spectral regimes.

    Many synchrotron models admit multiple global spectral "regimes" defined by
    the ordering of characteristic frequencies (e.g. :math:`\nu_a, \nu_m, \nu_c`).
    This class provides a standard pattern:

    1. Determine a regime label from the (global) model parameters.
    2. Optionally compute *derived* parameters needed to evaluate that regime
       (e.g. :math:`\nu_a` inferred from :math:`F_{\nu,\mathrm{pk}}`, geometry, etc.).
    3. Dispatch to a regime-specific optimized kernel.

    Subclasses implement:

    - :meth:`_compute_sed_regime`
    - :meth:`determine_sed_regime`
    - :meth:`_log_opt_sed_from_regime`

    Notes
    -----
    All "opt" methods operate on **unitless CGS scalars / NumPy arrays** and
    should not perform validation. Concrete subclasses must define whether
    the optimized backend expects linear frequencies ``nu`` or log-frequencies
    ``log_nu`` (see :meth:`_expects_log_frequency`).
    """

    #: Optional mapping/enum describing available regime functions.
    SPECTRUM_FUNCTIONS = None
    r"""
    Optional registry of synchrotron spectral regime functions.

    This attribute, when defined, provides a mapping or enumeration that
    associates **spectral regime identifiers** with the corresponding
    callable objects responsible for evaluating or constructing the SED
    in that regime.

    Typical use cases include:

    - Mapping a regime enum (e.g. :class:`SynchrotronSEDRegime`) to a
      concrete SED implementation.
    - Providing a lookup table for regime-specific normalization or
      evaluation logic.
    - Enabling dynamic dispatch based on the ordering of characteristic
      frequencies (e.g. :math:`\nu_a`, :math:`\nu_m`, :math:`\nu_c`).

    If set to ``None``, the class is assumed to represent a **single,
    fixed spectral regime** and does not support internal regime dispatch.

    Notes
    -----
    - Subclasses implementing multiple spectral regimes should override
      this attribute.
    - The objects stored in this registry are expected to be **callable**
      and to follow a consistent interface for SED evaluation or
      construction.
    """

    # ============================================================ #
    # Regime Management                                            #
    # ============================================================ #
    @abstractmethod
    def _compute_sed_regime(self, **parameters) -> tuple[Any, dict[str, Any]]:
        r"""
        Determine the global SED regime and any derived parameters.

        This method encodes the logic used to classify the SED into a discrete
        physical regime based on the ordering of characteristic frequencies.
        It may additionally compute *derived* parameters required to evaluate
        the spectrum (e.g. an inferred SSA frequency :math:`\nu_a`).

        The returned regime applies **globally** and does not depend on the
        sampling frequency grid.

        Parameters
        ----------
        **parameters
            Model parameters required to determine the regime. Typical examples
            include characteristic frequencies (or their log-values), peak flux
            normalizations, geometric factors, and microphysical parameters.

        Returns
        -------
        (regime, derived)
            regime : int or Enum-like
                Regime identifier. The interpretation is defined by the subclass.
            derived : dict
                Derived parameters needed for evaluation in the selected regime.
                This may include values such as ``log_nu_a`` or other cached
                quantities used by the regime-specific kernel.

        Notes
        -----
        - This method should be *fast* and free of allocations when possible.
        - No unit checking or validation should occur here.
        """
        raise NotImplementedError

    @abstractmethod
    def determine_sed_regime(self, **parameters) -> Any:
        r"""
        Determine the physical synchrotron spectral regime (public API).

        This method is intended for diagnostic and introspection use. It should
        perform any necessary unit handling and validation, then delegate to the
        optimized implementation (:meth:`_compute_sed_regime`).

        Parameters
        ----------
        **parameters
            User-facing model parameters. Typical examples include quantities
            such as :math:`\nu_m`, :math:`\nu_c`, :math:`F_{\nu,\mathrm{pk}}`, etc.

        Returns
        -------
        regime : int or Enum-like
            Regime identifier defined by the subclass.

        Notes
        -----
        - The regime does not depend on the frequency grid used for evaluation.
        - Subclasses should ensure that the regime returned here is consistent
          with the behavior of :meth:`sed`.
        """
        raise NotImplementedError

    # ============================================================ #
    # Regime-Specific SED Kernel                                   #
    # ============================================================ #
    @abstractmethod
    def _log_opt_sed_from_regime(
        self,
        nu: "_ArrayLike",
        regime: Any,
        **parameters,
    ):
        r"""
        Evaluate the log-space SED for a pre-determined regime.

        This method computes the logarithm of the SED for a single regime.
        All branching on ``regime`` should occur here (or above), and this
        method should assume that any derived parameters required for the regime
        have already been computed.

        Parameters
        ----------
        nu : float or array-like
            Frequency grid in optimized form. If :meth:`_expects_log_frequency`
            returns True, this is ``log_nu = log(ν)``. Otherwise it is linear ν.
        regime : int or Enum-like
            Regime identifier returned by :meth:`_compute_sed_regime`.
        **parameters
            Parameters required by the kernel, including any derived values
            produced by :meth:`_compute_sed_regime`.

        Returns
        -------
        float or array-like
            Logarithm of the SED evaluated at the given frequencies.

        Notes
        -----
        This is the performance-critical kernel. Implementations should avoid
        allocations and unnecessary branching when possible.
        """
        raise NotImplementedError

    # ============================================================ #
    # Log-Space Orchestration                                      #
    # ============================================================ #
    def _log_opt_sed(self, nu: "_ArrayLike", **parameters):
        r"""
        Log-space optimized SED evaluation with regime dispatch.

        This method determines the appropriate SED regime from the input
        parameters and dispatches to the corresponding regime-specific kernel.

        Subclasses should generally **not override** this method. Instead,
        implement :meth:`_compute_sed_regime` and :meth:`_log_opt_sed_from_regime`.

        Parameters
        ----------
        nu : float or array-like
            Frequency grid in optimized form. If :meth:`_expects_log_frequency`
            returns True, this is ``log_nu = log(ν)``. Otherwise it is linear ν.
        **parameters
            Parameters required for both regime determination and SED evaluation.

        Returns
        -------
        float or array-like
            Logarithm of the SED evaluated at the given frequencies.
        """
        regime, derived = self._compute_sed_regime(**parameters)
        merged = dict(parameters)
        merged.update(derived)
        return self._log_opt_sed_from_regime(nu, regime, **merged)


# ============================================================ #
# SED Implementations                                          #
# ============================================================ #
# Now we can include concrete implementations of various SEDs. Not
# all of the SEDs we plan to implement are currently implemented in the
# codebase, but we provide a few examples here to illustrate the structure.
class PowerLaw_SynchrotronSED(SynchrotronSED):
    r"""
    Canonical optically thin power-law synchrotron SED.

    This class implements the standard synchrotron spectral energy distribution
    for an **uncooled, optically thin** population of relativistic electrons with
    a power-law energy distribution. See :ref:`synchrotron_theory` for a discussion
    of the relevant background physics and :ref:`synch_sed_theory` for a detailed
    derivation of this and related SEDs.

    This SED is applicable only in the simplest regimes where both radiative
    cooling and synchrotron self-absorption can be neglected. Typical examples
    include young systems whose cooling timescales are long compared to the
    dynamical time, or low-density environments where the emission remains
    optically thin across the observed band.

    .. rubric:: Spectral Structure

    This SED consists of a single optically thin synchrotron spectrum with one
    physical spectral break at the injection frequency :math:`\nu_m`, and an
    optional high-frequency exponential cutoff at :math:`\nu_{\max}`:

    - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu < \nu_m`, corresponding to the
      low-frequency optically thin synchrotron tail.
    - :math:`F_\nu \propto \nu^{-(p-1)/2}` for :math:`\nu_m < \nu < \nu_{\max}`,
      corresponding to emission from the accelerated power-law electron population.
    - An exponential suppression above :math:`\nu_{\max}`.

    .. hint::

        For a detailed derivation and discussion of this spectrum, see
        :ref:`synch_sed_theory`.

    .. rubric:: SED Parameters

    The parameters entering this SED fall into three conceptual categories.

    .. tab-set::

        .. tab-item:: Free parameters (phenomenological)

            These parameters define the observable structure of the SED and are
            typically inferred directly from broadband data.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Normalization flux density
                  - :math:`F_{\nu,\mathrm{norm}}`
                  - Flux density of the **dominant optically thin emitting
                    population** used to normalize the SED.
                    For this SED, the dominant optically thin emission occurs at
                    the injection frequency :math:`\nu_m`, so
                    :math:`F_{\nu,\mathrm{norm}} = F_\nu(\nu_m)`.
                * - Injection frequency
                  - :math:`\nu_m`
                  - Synchrotron characteristic frequency of the minimum-energy
                    electrons.
                * - Maximum frequency
                  - :math:`\nu_{\max}`
                  - High-frequency exponential cutoff.

        .. tab-item:: Hyper-parameters

            These parameters control the *shape* of the spectrum but are not
            usually directly constrained by broadband observations.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Electron power-law index
                  - :math:`p`
                  - Index of the injected electron energy distribution,
                    :math:`N(\gamma) \propto \gamma^{-p}`.
                * - Smoothing parameter
                  - :math:`s`
                  - Controls the sharpness of the spectral transition at
                    :math:`\nu_m`.

    .. rubric:: Normalization and Closure

    The SED is **always normalized using** :math:`F_{\nu,\mathrm{norm}}`, defined
    as the flux density of the dominant optically thin emitting electron
    population. The location of this normalization frequency depends on the
    physical regime.

    In the present optically thin, uncooled case, the dominant emission occurs
    at :math:`\nu_m`, so the normalization coincides with the injection break.
    In more complex SEDs (e.g. with cooling or self-absorption), the normalization
    may instead correspond to a different characteristic frequency.

    Optional closure relations are provided to map physical model parameters
    (e.g. magnetic field strength, emitting volume, electron energy fractions)
    onto the phenomenological SED parameters under assumed microphysical
    conditions such as equipartition.

    See Also
    --------
    :class:`SynchrotronSED`
        Base class for synchrotron SED implementations.
    :class:`MultiSpectrumSynchrotronSED`
        Base class for multi-regime synchrotron SEDs.
    :class:`PowerLaw_Cooling_SynchrotronSED`
        Synchrotron SED including a radiative cooling break.
    :class:`PowerLaw_SSA_SynchrotronSED`
        Synchrotron SED including synchrotron self-absorption.
    :class:`PowerLaw_Cooling_SSA_SynchrotronSED`
        Synchrotron SED including both cooling and self-absorption.

    Examples
    --------
    The simplest use case is to evaluate the SED directly from phenomenological
    parameters inferred from data.

    .. code-block:: python

        import numpy as np
        import astropy.units as u
        from triceratops.radiation.synchrotron import (
            PowerLaw_SynchrotronSED,
        )

        sed = PowerLaw_SynchrotronSED()

        nu = np.logspace(8, 18, 300) * u.Hz

        F_nu = sed.sed(
            nu=nu,
            F_norm=1e-26 * u.erg / (u.cm**2 * u.s * u.Hz),
            nu_m=1e12 * u.Hz,
            nu_max=1e18 * u.Hz,
            p=2.4,
        )

    This evaluates the optically thin synchrotron spectrum with a normalization
    at :math:`\nu_m = 10^{12}\,\mathrm{Hz}` and an electron index :math:`p=2.4`.

    More commonly, the SED may be constructed from physical model parameters
    using the equipartition-based closure relation.

    .. code-block:: python

        params = sed.from_physics_to_params(
            B=0.1 * u.G,
            V=1e48 * u.cm**3,
            D_L=100 * u.Mpc,
            gamma_min=100.0,
            gamma_max=1e6,
            p=2.3,
            epsilon_E=0.1,
            epsilon_B=0.1,
            pitch_average=True,
        )

        F_nu = sed.sed(
            nu=nu,
            **params,
            p=2.3,
        )
    """

    # ============================================================ #
    # SED Function Implementation                                  #
    # ============================================================ #
    # Here should be the implementation of the SED function itself,
    # which is a function of nu and some set of additional parameters.
    def _log_opt_sed(
        self,
        log_nu: "_ArrayLike",
        log_F_norm: float,
        log_nu_m: float,
        log_nu_max: float = np.inf,
        p: float = 2.5,
        s: float = 5.0,
    ):
        r"""
        Compute the logarithm of the optically thin power-law synchrotron SED.

        This method implements the **core spectral shape** of the canonical,
        optically thin, uncooled synchrotron SED entirely in log-space. It is the
        lowest-level representation of the SED used internally by this class and
        by inference routines.

        The spectrum is constructed as a smoothed broken power law with a single
        physical break at the injection frequency :math:`\nu_m` and an exponential
        cutoff at :math:`\nu_{\max}`. The overall normalization is applied
        multiplicatively via the peak flux density.

        Parameters
        ----------
        log_nu : array-like
            Natural logarithm of the observing frequency (in Hz).
            This may be a scalar or an array and is assumed to be dimensionless.
        log_F_norm : float
            The natural logarithm of the normalization flux density, corresponding to the
            corresponding flux density at the dominant optically thin frequency (in this case, :math:`\nu_m`).
            Because this SED is optically thin at peak, this is also the flux density at the injection break.
        log_nu_m : float
            Natural logarithm of the injection (peak) frequency
            :math:`\nu_m`.
        log_nu_max : float, optional
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`. Above this frequency, the spectrum is exponentially
            suppressed. The default is :math:`+\infty`, corresponding to no cutoff.
        p : float, optional
            Power-law index of the injected electron energy distribution,
            :math:`N(\gamma) \propto \gamma^{-p}`.
        s : float, optional
            Smoothing parameter controlling the sharpness of the spectral break
            at :math:`\nu_m`. Larger values correspond to sharper transitions.

        Returns
        -------
        log_F_nu : array-like
            Natural logarithm of the flux density :math:`F_\nu` evaluated at
            ``log_nu``.

        Notes
        -----
        - This method performs **no unit handling** and assumes all inputs are
          dimensionless and expressed in CGS-consistent logarithmic form.
        - This method does **not** perform any physical normalization; it merely
          applies the spectral shape and multiplies by the supplied peak flux.
        - All higher-level interfaces (e.g. :meth:`sed`) are thin wrappers around
          this method.

        See Also
        --------
        _log_powerlaw_sbpl_sed :
            Low-level implementation of the smoothed broken power-law shape.
        """
        return (
            _log_powerlaw_sbpl_sed(
                log_nu,
                log_nu_m,
                log_nu_max,
                p,
                s,
            )
            + log_F_norm
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        F_norm: "_UnitBearingScalarLike",
        nu_m: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf * u.Hz,
        p: float = 2.5,
        s: float = 5.0,
    ):
        r"""
        Evaluate the optically thin power-law synchrotron SED.

        This is the **public, unit-aware interface** for evaluating the canonical
        power-law synchrotron SED. It performs unit validation and conversion,
        dispatches to the internal log-space implementation, and returns the
        flux density with appropriate physical units.

        Parameters
        ----------
        nu : array-like or quantity
            Observing frequencies at which to evaluate the SED. If unit-bearing,
            must be convertible to Hz.
        F_norm : scalar or quantity
            The normalization flux density, corresponding to the
            corresponding flux density at the dominant optically thin frequency (in this case, :math:`\nu_m`).
            Because this SED is optically thin at peak, this is also the flux density at the injection break.
        nu_m : scalar or quantity
            Injection (peak) frequency :math:`\nu_m`. If unit-bearing, must be
            convertible to Hz.
        nu_max : scalar or quantity, optional
            Maximum synchrotron frequency :math:`\nu_{\max}`. Frequencies above
            this value are exponentially suppressed. The default corresponds to
            no cutoff.
        p : float, optional
            Electron power-law index.
        s : float, optional
            Smoothing parameter controlling the sharpness of the spectral break.

        Returns
        -------
        F_nu : array-like or quantity
            Flux density evaluated at ``nu``, returned with units of
            ``erg cm^-2 s^-1 Hz^-1``.

        Notes
        -----
        - This method is a thin wrapper around the internal log-space method
          :meth:`_log_opt_sed`.
        - All numerical evaluation is performed in logarithmic space for
          numerical stability.
        - This method does not enforce any physical closure; the parameters
          ``F_norm``, ``nu_m``, and ``nu_max`` are treated as phenomenological
          inputs.

        Examples
        --------
        Evaluate a simple power-law synchrotron SED:

        .. code-block:: python

            sed = PowerLaw_SynchrotronSED()
            nu = np.logspace(8, 18, 200) * u.Hz
            F_nu = sed.sed(
                nu=nu,
                F_norm=1e-26 * u.erg / (u.cm**2 * u.s * u.Hz),
                nu_m=1e12 * u.Hz,
                p=2.4,
            )

        """
        # Handle units
        nu = ensure_in_units(nu, "Hz")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")
        F_norm = ensure_in_units(F_norm, "erg cm^-2 s^-1 Hz^-1")

        # Convert to log-space dimensionless CGS
        log_nu = np.log(nu)
        log_nu_m = np.log(nu_m)
        log_nu_max = np.log(nu_max)
        log_F_norm = np.log(F_norm)

        # Dispatch to optimized implementation
        log_sed = self._log_opt_sed(
            log_nu=log_nu,
            log_F_norm=log_F_norm,
            log_nu_m=log_nu_m,
            log_nu_max=log_nu_max,
            p=p,
            s=s,
        )

        return np.exp(log_sed) * u.erg / (u.cm**2 * u.s * u.Hz)

    # =========================================================== #
    # Closure Relations Implementation                            #
    # =========================================================== #
    # Here we implement the closure relations to go forward and backward
    # between the physics parameters and the phenomenological SED parameters.
    def _opt_from_physics_to_params(
        self,
        log_B: float,
        log_V: float,
        log_D_L: float,
        log_gamma_min: float,
        log_gamma_max: float = np.inf,
        p: float = 2.5,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk=1.0,
        redshift=0.0,
        pitch_average: bool = False,
    ):
        r"""
        Map physical parameters to phenomenological SED parameters.

        This method implements a deterministic **closure relation** that connects
        physical model parameters (e.g. magnetic field strength, emitting volume,
        electron energy fractions) to the phenomenological parameters defining the
        canonical power-law synchrotron SED.

        The mapping is performed under the assumption of **equipartition** between
        relativistic electrons and magnetic fields and follows the normalization
        scheme described in :ref:`synch_sed_theory`.

        All computations are carried out in log-space to ensure numerical stability
        and compatibility with inference workflows.

        Parameters
        ----------
        log_B : float
            Natural logarithm of the magnetic field strength (Gauss).
        log_V : float
            Natural logarithm of the effective emitting volume (cm^3). This is often
            parameterized in terms of a filling factor :math:`f`.
        log_D_L : float
            Natural logarithm of the luminosity distance (cm).
        log_gamma_min : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\gamma_{\min}`.
        log_gamma_max : float, optional
            Natural logarithm of the maximum electron Lorentz factor
            :math:`\gamma_{\max}`. The default corresponds to no upper cutoff.
        p : float, optional
            Power-law index of the injected electron distribution.
        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic electrons.
        epsilon_B : float, optional
            Fraction of post-shock internal energy stored in magnetic fields.
        alpha : float, optional
            Electron pitch angle in radians. This parameter is ignored if
            ``pitch_average=True``.
        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region. This affects the Doppler
            boosting of the observed frequencies and fluxes.
        redshift: float, optional
            Cosmological redshift of the source. This affects the observed frequencies
            and fluxes via redshift corrections.
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.
            Otherwise, a fixed pitch angle specified by ``alpha`` is used.

        Returns
        -------
        params : dict
            Dictionary containing the phenomenological SED parameters:

            - ``log_F_norm`` : Natural logarithm of the peak flux density
              :math:`F_{\nu,\mathrm{pk}}`.
            - ``log_nu_m`` : Natural logarithm of the injection (peak) frequency
              :math:`\nu_m`.
            - ``log_nu_max`` : Natural logarithm of the maximum synchrotron
              frequency :math:`\nu_{\max}`.

        Notes
        -----
        - This closure assumes a **single-zone**, homogeneous emitting region.
        - Equipartition is an **assumption**, not a physical requirement; alternative
          closures may be implemented by overriding this method.
        - This method does not perform any consistency checks on the resulting SED
          (e.g. cooling or self-absorption); such checks are handled by more
          specialized SED subclasses.
        """
        # Handle pitch angle details and the relevant values of
        # the log_chi parameter in each case. This allows us to
        # permit both a fixed pitch angle and pitch-angle averaging.
        sin_alpha = np.sin(alpha)
        log_sin_alpha = np.log(sin_alpha) if not pitch_average else 0.0
        log_chi = _log_chi_cgs_iso if pitch_average else _log_chi_cgs

        # Compute the Doppler factors and redshift corrections for the
        # fluxes and the frequencies.
        beta_bulk = np.sqrt(1 - gamma_bulk**-2)
        delta_bulk = gamma_bulk * (1 + beta_bulk)
        log_delta_bulk = np.log(delta_bulk)
        log_nu_correction = log_delta_bulk - np.log1p(redshift)
        log_flux_correction = 3 * log_delta_bulk - np.log1p(redshift)

        # Characteristic synchrotron frequencies. These are all
        # OBSERVER FRAME quantities.
        log_nu_m = (
            _opt_compute_log_synch_frequency(
                log_gamma_min,
                log_B,
                sin_alpha=sin_alpha,
                pitch_average=pitch_average,
            )
            + log_nu_correction
        )

        log_nu_max = (
            _opt_compute_log_synch_frequency(
                log_gamma_max,
                log_B,
                sin_alpha=sin_alpha,
                pitch_average=pitch_average,
            )
            + log_nu_correction
        )

        # Electron distribution normalization via equipartition
        log_N0 = np.log(
            _opt_normalize_PL_from_magnetic_field(
                np.exp(log_B),
                p=p,
                epsilon_E=epsilon_E,
                epsilon_B=epsilon_B,
                gamma_min=np.exp(log_gamma_min),
                gamma_max=np.exp(log_gamma_max),
            )
        )

        # Peak flux normalization
        log_F_norm = (
            log_chi
            + log_B
            + log_sin_alpha  # Will be zero if pitch-averaged
            + log_N0
            + log_V
            - 2.0 * log_D_L
            + (1.0 - p) * log_gamma_min
            + log_flux_correction
        )

        return {
            "log_F_norm": log_F_norm,
            "log_nu_m": log_nu_m,
            "log_nu_max": log_nu_max,
        }

    def from_physics_to_params(
        self,
        B: "_UnitBearingScalarLike",
        V: "_UnitBearingScalarLike",
        D_L: "_UnitBearingScalarLike",
        gamma_min: float,
        gamma_max: float = np.inf,
        p: float = 2.5,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        bulk_gamma=1.0,
        redshift=0.0,
        pitch_average: bool = False,
    ):
        r"""
        Construct phenomenological SED parameters from physical model parameters.

        This is the **public, unit-aware interface** to the equipartition-based
        closure relation implemented by
        :meth:`_opt_from_physics_to_params`. It maps physical quantities describing
        the emitting region and electron population onto the parameters defining
        the canonical power-law synchrotron SED.

        Parameters
        ----------
        B : scalar or quantity
            Magnetic field strength in the emitting region. Must be convertible
            to Gauss.
        V : scalar or quantity
            Effective emitting volume. Must be convertible to cm³.
            This is often parameterized in terms of a filling factor.
        D_L : scalar or quantity
            Luminosity distance to the source. Must be convertible to cm.
        gamma_min : float
            Minimum electron Lorentz factor :math:`\gamma_{\min}`.
        gamma_max : float, optional
            Maximum electron Lorentz factor :math:`\gamma_{\max}`.
            The default corresponds to no upper cutoff.
        p : float, optional
            Power-law index of the injected electron energy distribution.
        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic electrons.
        epsilon_B : float, optional
            Fraction of post-shock internal energy stored in magnetic fields.
        alpha : float, optional
            Electron pitch angle in radians. Ignored if ``pitch_average=True``.
        bulk_gamma: float, optional
            Bulk Lorentz factor of the emitting region, affecting Doppler boosting.
        redshift: float, optional
            Cosmological redshift of the source, affecting observed frequencies and fluxes.
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.
            Otherwise, use a fixed pitch angle specified by ``alpha``.

        Returns
        -------
        params : dict
            Dictionary containing the phenomenological SED parameters:

            - ``F_norm`` : Peak flux density :math:`F_{\nu,\mathrm{pk}}`
              with units of ``erg cm^-2 s^-1 Hz^-1``.
            - ``nu_m`` : Injection (peak) frequency :math:`\nu_m` with units of Hz.
            - ``nu_max`` : Maximum synchrotron frequency :math:`\nu_{\max}` with
              units of Hz.

        Notes
        -----
        - This method assumes a single-zone, homogeneous emitting region.
        - Equipartition is an **assumption**, not a physical necessity.
        - This method does not check for radiative cooling or synchrotron
          self-absorption; those effects are handled by specialized SED subclasses.

        See Also
        --------
        _opt_from_physics_to_params :
            Log-space implementation of the closure relation.
        sed :
            Evaluate the synchrotron SED using the returned parameters.
        """
        # Enforce units
        B = ensure_in_units(B, "G")
        V = ensure_in_units(V, "cm^3")
        D_L = ensure_in_units(D_L, "cm")

        # Convert to log-space
        log_B = np.log(B)
        log_V = np.log(V)
        log_D_L = np.log(D_L)
        log_gamma_min = np.log(gamma_min)
        log_gamma_max = np.log(gamma_max)

        # Dispatch to optimized log-space closure
        params_log = self._opt_from_physics_to_params(
            log_B=log_B,
            log_V=log_V,
            log_D_L=log_D_L,
            log_gamma_min=log_gamma_min,
            log_gamma_max=log_gamma_max,
            p=p,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            alpha=alpha,
            pitch_average=pitch_average,
            gamma_bulk=bulk_gamma,
            redshift=redshift,
        )

        # Convert back to physical units
        return {
            "F_norm": np.exp(params_log["log_F_norm"]) * u.erg / (u.cm**2 * u.s * u.Hz),
            "nu_m": np.exp(params_log["log_nu_m"]) * u.Hz,
            "nu_max": np.exp(params_log["log_nu_max"]) * u.Hz,
        }


class PowerLaw_Cooling_SSA_SynchrotronSED(MultiSpectrumSynchrotronSED):
    r"""
    Synchrotron spectral energy distribution with cooling and self-absorption.

    This class implements the **full piecewise synchrotron spectrum** for a
    power-law electron population, including:

    - Radiative cooling (fast, slow, and non-cooling regimes),
    - Synchrotron self-absorption (SSA),
    - Stratified SSA corrections where applicable,
    - A high-frequency exponential cutoff.

    The implementation follows the standard GRB / supernova afterglow
    formalism (e.g. :footcite:t:`GranotSari2002SpectralBreaks`, :footcite:t:`2020MNRAS.493.3521B`,
    :footcite:t:`duran2013radius`, :footcite:t:`2025ApJ...992L..18S`, :footcite:t:`GaoSynchrotronReview2013`,
    etc.)

    Specifically we implement spectra using **log-space SED surgery** with scale-free smoothed broken
    power laws (SFBPLs). The spectrum is assembled by:

    1. Determining the **global spectral regime** from characteristic frequencies,
    2. Computing any **derived break frequencies** required by that regime,
    3. Dispatching to a **regime-specific optimized kernel**,
    4. Applying an overall normalization via the peak flux density.

    .. hint::

        For a detailed derivation of the spectral segments and break orderings, see
        :ref:`synchrotron_theory` and :ref:`synch_sed_theory`.

    .. rubric:: Spectral Structure

    The SED is globally classified into one of several discrete regimes based on
    the ordering of the characteristic frequencies:

    - Injection frequency :math:`\nu_m`,
    - Cooling frequency :math:`\nu_c`,
    - Self-absorption frequency :math:`\nu_a` (computed internally),
    - Maximum synchrotron frequency :math:`\nu_{\max}`.

    Each regime corresponds to a specific ordering (e.g.
    :math:`\nu_a < \nu_c < \nu_m < \nu_{\max}`) and therefore to a unique set of
    spectral slopes. The regime selection is **global** and does not depend on the
    frequency grid used for evaluation. As such, when the SED is called, the parameters
    are used once to determine the regime and then to evaluate the SED at all requested frequencies.

    .. rubric:: SED Parameters

    The parameters entering this SED fall into three conceptual categories.

    .. tab-set::

        .. tab-item:: Free parameters (phenomenological)

            These parameters define the observable structure of the SED and are
            typically inferred directly from data.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Peak flux density
                  - :math:`F_{\nu,\mathrm{norm}}`
                  - The equivalent flux at the dominant electron frequency in
                    an optically thin medium. This sets the SED peak. :meth:`from_physics_to_params` maps
                    physical parameters onto this quantity.
                * - Injection frequency
                  - :math:`\nu_m`
                  - Synchrotron frequency of minimum-energy electrons
                * - Cooling frequency
                  - :math:`\nu_c`
                  - Frequency corresponding to the cooling Lorentz factor
                * - Maximum frequency
                  - :math:`\nu_{\max}`
                  - High-energy cutoff frequency
                * - (Optional) stratified SSA frequency
                  - :math:`\nu_{ac}`
                  - Transition frequency for stratified SSA regimes

        .. tab-item:: Hyper-parameters

            These parameters control the *shape* and smoothness of the spectrum
            but are not usually directly inferred from broadband data.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Electron power-law index
                  - :math:`p`
                  - Index of the injected electron distribution
                * - Smoothing parameter
                  - :math:`s`
                  - Controls sharpness of spectral breaks
                * - Emission solid angle
                  - :math:`\Omega`
                  - Effective emitting area divided by distance squared
                * - Minimum Lorentz factor
                  - :math:`\gamma_m`
                  - Minimum electron Lorentz factor

        .. tab-item:: Derived parameters (internal)

            These quantities are **not user inputs**, but are computed internally
            from the free parameters and microphysical assumptions.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - SSA frequency
                  - :math:`\nu_a`
                  - Self-absorption break frequency
                * - Cooling regime
                  - —
                  - Fast, slow, or non-cooling classification
                * - Regime identifier
                  - —
                  - Discrete index selecting the SED kernel

    See Also
    --------
    :class:`SynchrotronSED` : Base class for synchrotron SED implementations.
    :class:`MultiSpectrumSynchrotronSED` : Base class for multi-regime synchrotron SEDs.
    :class:`PowerLaw_Cooling_SynchrotronSED` : Synchrotron SED with cooling break.
    :class:`PowerLaw_SSA_SynchrotronSED` : Synchrotron SED with synchrotron self-absorption.
    :class:`PowerLaw_SynchrotronSED` : Canonical optically-thin power-law synchrotron SED.

    References
    ----------
    .. footbibliography::
    """

    # ============================================================ #
    # Declare the spectrum functions mapping                       #
    # ============================================================ #
    # These are the synchrotron SED functions with SSA and cooling. We use
    # this enum class to trace through the runtime which regime we're in.
    SPECTRUM_FUNCTIONS = _SynchrotronSSACoolingSEDFunctions

    # ============================================================ #
    # Regime Management                                            #
    # ============================================================ #
    # Regime management in this class is non-trivial as the SSA frequency
    # must be self-consistently computed from the other parameters for each
    # regime. This requires some careful bookkeeping. We implement this in the
    # _compute_sed_regime method below.
    def _compute_ssa_frequencies_from_F_norm(
        self,
        log_F_norm: float,
        log_nu_m: float,
        log_nu_c: float,
        log_nu_max: float,
        log_omega: float,
        log_gamma_m: float,
        p: float = 3.0,
    ) -> tuple[dict[_SynchrotronSSACoolingSEDFunctions, float], int]:
        r"""
        Compute candidate self-absorption frequencies using anchored spectral normalizations.

        This method evaluates analytic expressions for the synchrotron self-absorption
        frequency :math:`\nu_a` for all spectral regimes compatible with the system’s
        cooling state, **using explicit spectral normalizations** rather than a single
        peak flux density.

        Unlike :meth:`_compute_ssa_frequencies_from_F_peak`, which assumes the supplied
        flux normalization corresponds to the *true spectral peak*, this method allows
        the spectrum to be anchored at a **specific characteristic frequency**, typically:

        - :math:`\nu_m` (slow-cooling or non-cooling regimes), or
        - :math:`\nu_c` (fast-cooling regimes).

        This distinction is essential when the dominant emitting electron population
        does *not* coincide with the observed spectral peak, such as in fast-cooling
        or heavily self-absorbed scenarios.

        Parameters
        ----------
        log_F_norm : float
            Natural logarithm of the flux normalization.
        log_nu_m : float
            Natural logarithm of the injection frequency :math:`\nu_m`.
        log_nu_c : float
            Natural logarithm of the cooling frequency :math:`\nu_c`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`.
        log_omega : float
            Natural logarithm of the effective emission solid angle
            :math:`\log \Omega`.
        log_gamma_m : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\log \gamma_m`.
        p : float, optional
            Power-law index of the injected electron distribution.

        Returns
        -------
        tuple
            A tuple ``(log_nu_a_dict, cooling_regime)`` where:

            - ``log_nu_a_dict`` maps each candidate spectral regime to its corresponding
              self-absorption frequency :math:`\log \nu_a`, computed under the assumption
              that *that regime applies*.
            - ``cooling_regime`` is an integer flag indicating the cooling state:
              ``0`` = fast cooling,
              ``1`` = slow cooling,
              ``2`` = no cooling.

        Notes
        -----
        - This method is required for **physically correct normalization** in regimes
          where the spectral peak does not coincide with :math:`\nu_m`.
        - Only regimes compatible with the inferred cooling state are included.
        - The returned self-absorption frequencies are **candidates**; the final regime
          selection and consistency check are performed by
          :meth:`_compute_sed_regime`.
        - All calculations are performed in **logarithmic CGS units** and assume
          isotropic pitch-angle distributions.

        See Also
        --------
        _compute_ssa_frequencies_from_F_peak :
            Simplified SSA computation assuming the supplied normalization corresponds
            to the true spectral peak.
        _compute_sed_regime :
            Final regime selection and SSA consistency enforcement.
        """
        # Pre-compute the common factor so that we do not need to recompute it.
        log_Q = log_F_norm - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m
        log_Q_c = log_Q + (1 / 2) * (log_nu_m - log_nu_c)

        # We'll want to quickly check if we are fast, slow, or non-cooling. This will
        # narrow down the scenarios we need to consider and the corresponding construction
        # of the relevant SSA frequencies.
        if log_nu_c < log_nu_m:
            # This is fast cooling. We have access to S5, S6, and S7 only.
            return {
                self.SPECTRUM_FUNCTIONS.SPECTRUM_5: (6 * log_Q_c / 13) + (3 * log_nu_m / 13) - (2 * log_nu_c / 13),
                self.SPECTRUM_FUNCTIONS.SPECTRUM_6: (log_Q_c / 3) + (log_nu_m / 6) + (log_nu_c / 6),
                self.SPECTRUM_FUNCTIONS.SPECTRUM_7: (2 * log_Q / (5 + p))
                + (log_nu_c / (5 + p))
                + (p * log_nu_m / (5 + p)),
            }, 0
        elif (log_nu_m < log_nu_c) and (log_nu_c < log_nu_max):
            # This is slow cooling with nu_c < nu_max. We have access to S3 and S4 and S7 only.
            return {
                self.SPECTRUM_FUNCTIONS.SPECTRUM_3: (3 * log_Q / 5) - (log_nu_m / 5),
                self.SPECTRUM_FUNCTIONS.SPECTRUM_4: (2 * log_Q / (p + 4)) + (p * log_nu_m / (p + 4)),
                self.SPECTRUM_FUNCTIONS.SPECTRUM_7: (2 * log_Q / (5 + p))
                + (log_nu_c / (5 + p))
                + (p * log_nu_m / (5 + p)),
            }, 1
        else:
            # This is the NO COOLING case. We have access to S1 and S2 only.
            return {
                self.SPECTRUM_FUNCTIONS.SPECTRUM_1: (6 * log_Q / 13) + (log_nu_m / 13),
                self.SPECTRUM_FUNCTIONS.SPECTRUM_2: (2 * log_Q / (p + 4)) + (p * log_nu_m / (p + 4)),
            }, 2

    def _compute_sed_regime_from_nu_a_dict(
        self,
        log_nu_ssa: dict[_SynchrotronSSACoolingSEDFunctions, float],
        cooling_regime: int,
        log_nu_m: float,
        log_nu_c: float,
        log_nu_max: float,
    ):
        r"""
        Select the unique synchrotron spectral regime from candidate SSA frequencies.

        This method determines the **single, globally applicable synchrotron spectral
        regime** by comparing candidate self-absorption frequencies against the
        characteristic synchrotron break frequencies.

        The input ``log_nu_ssa`` contains **candidate self-absorption frequencies**
        appropriate to each allowed spectral regime, computed analytically under the
        assumption that *that regime applies*. This method resolves that ambiguity by
        enforcing the correct ordering of frequencies.

        The selection logic depends on the cooling state:

        - **Fast cooling** (:math:`\nu_c < \nu_m`): choose among spectra S5, S6, S7
        - **Slow cooling** (:math:`\nu_m < \nu_c < \nu_{\max}`): choose among S3, S4, S7
        - **No cooling** (:math:`\nu_c > \nu_{\max}`): choose among S1, S2

        We implement this so that we do not need to replicate the entire regime determination
        workflow in each SED evaluation. Instead, we compute all candidate SSA frequencies
        once, then use this method to select the correct one based on the physical
        parameters.

        Parameters
        ----------
        log_nu_ssa : dict
            Dictionary mapping candidate spectral regimes to their corresponding
            self-absorption frequencies :math:`\log \nu_a`, computed under the
            assumption that each regime applies.
        cooling_regime : int
            Cooling classification flag:
            ``0`` = fast cooling,
            ``1`` = slow cooling,
            ``2`` = no cooling.
        log_nu_m : float
            Natural logarithm of the injection frequency :math:`\nu_m`.
        log_nu_c : float
            Natural logarithm of the cooling frequency :math:`\nu_c`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency :math:`\nu_{\max}`.

        Returns
        -------
        tuple
            ``(regime, log_nu_a)`` where:

            - ``regime`` is a member of
              :class:`~_SynchrotronSSACoolingSEDFunctions` identifying the selected
              spectral configuration.
            - ``log_nu_a`` is the self-absorption frequency consistent with that regime.

        Raises
        ------
        RuntimeError
            If no candidate regime satisfies the required frequency ordering.

        Notes
        -----
        - Exactly **one** regime must satisfy the ordering constraints.
        - This method performs no numerical approximations—only logical selection.
        - The selected regime applies **globally** to the SED.
        """
        # For debugging, we take a minute here to log some detailed information
        # about the candidate SSA frequencies and the characteristic frequencies. This can be
        # helpful for diagnosing regime selection issues and ensuring that the logic is working as intended.
        #
        # This should not interfere with performance in production as it is gated behind a debug-level log check,
        # but it provides a valuable window into the internal decision-making process when enabled.
        if triceratops_logger.isEnabledFor(logging.DEBUG):
            triceratops_logger.debug(
                "PowerLaw_Cooling_SSA_SynchrotronSED: performing regime determination step:",
                extra={
                    "cooling_regime": cooling_regime,
                    "log_nu_m": log_nu_m,
                    "log_nu_c": log_nu_c,
                    "log_nu_max": log_nu_max,
                    "log_nu_ssa": log_nu_ssa,
                },
            )

        if cooling_regime == 0:
            # Fast cooling: S5, S6, or S7
            if log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_5] < log_nu_c:
                # We get the 2, 11/8, 1/3, -1/2, -p/2 spectrum.
                regime, log_nu_a = self.SPECTRUM_FUNCTIONS.SPECTRUM_5, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_5]
            elif log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_7] > log_nu_m:
                # This is the moderate fast cooling case with extreme absorption nu_c < nu_m < nu_a < nu_max.
                regime, log_nu_a = self.SPECTRUM_FUNCTIONS.SPECTRUM_7, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_7]
            elif log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_6] < log_nu_m:
                # This is the intermediate fast cooling case with nu_c < nu_a < nu_m < nu_max.
                # NOTE: the catch on nu_c above allows us to avoid ambiguity with the extreme fast cooling case.
                regime, log_nu_a = self.SPECTRUM_FUNCTIONS.SPECTRUM_6, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_6]
            else:
                raise RuntimeError(
                    "Could not determine regime in fast cooling scenario:\n"
                    f"  log_nu_c          = {log_nu_c:0.3f}\n"
                    f"  log_nu_m          = {log_nu_m:0.3f}\n"
                    f"  log_nu_max        = {log_nu_max:0.3f}\n"
                    f"  log_nu_a (S5)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_5]:0.3f}\n"
                    f"  log_nu_a (S6)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_6]:0.3f}\n"
                    f"  log_nu_a (S7)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_7]:0.3f}\n"
                )

        # Slow cooling: S3 or S4
        elif cooling_regime == 1:
            if log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_3] < log_nu_m:
                # This is the optically thin at peak slow cooling case with nu_a < nu_m < nu_c < nu_max.
                regime, log_nu_a = self.SPECTRUM_FUNCTIONS.SPECTRUM_3, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_3]
            elif log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_4] < log_nu_c:
                # This is the case where nu_m < nu_a < nu_c < nu_max. Note that we catch nu_a < nu_m above.
                regime, log_nu_a = self.SPECTRUM_FUNCTIONS.SPECTRUM_4, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_4]
            elif log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_7] > log_nu_c:
                # This is the extreme case where nu_m < nu_c < nu_a < nu_max and the cooling break is hidden.
                regime, log_nu_a = self.SPECTRUM_FUNCTIONS.SPECTRUM_7, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_7]
            else:
                raise RuntimeError(
                    "Could not determine regime in slow cooling scenario:\n"
                    f"  log_nu_c          = {log_nu_c:0.3f}\n"
                    f"  log_nu_m          = {log_nu_m:0.3f}\n"
                    f"  log_nu_max        = {log_nu_max:0.3f}\n"
                    f"  log_nu_a (S3)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_3]:0.3f}\n"
                    f"  log_nu_a (S4)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_4]:0.3f}\n"
                    f"  log_nu_a (S7)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_7]:0.3f}\n"
                )

        # No cooling: S1 or S2
        elif cooling_regime == 2:
            if log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_1] < log_nu_m:
                # This is the optically thin at peak no cooling case with nu_a < nu_m < nu_max.
                regime, log_nu_a = self.SPECTRUM_FUNCTIONS.SPECTRUM_1, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_1]
            elif log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_2] > log_nu_m:
                # This is the extreme case where nu_m < nu_a < nu_max.
                regime, log_nu_a = self.SPECTRUM_FUNCTIONS.SPECTRUM_2, log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_2]
            else:
                raise RuntimeError(
                    "Could not determine regime in no cooling scenario:\n"
                    f"  log_nu_m          = {log_nu_m:0.3f}\n"
                    f"  log_nu_max        = {log_nu_max:0.3f}\n"
                    f"  log_nu_c          = {log_nu_c:0.3f}\n"
                    f"  log_nu_a (S1)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_1]:0.3f}\n"
                    f"  log_nu_a (S2)     = {log_nu_ssa[self.SPECTRUM_FUNCTIONS.SPECTRUM_2]:0.3f}\n"
                )
        else:
            raise RuntimeError("Unable to determine SSA spectrum regime: unrecognized cooling regime.")

        # Now complete, log that we have completed the step, and return.
        if triceratops_logger.isEnabledFor(logging.DEBUG):
            triceratops_logger.debug(
                "PowerLaw_Cooling_SSA_SynchrotronSED: selected regime.",
                extra={
                    "cooling_regime": cooling_regime,
                    "log_nu_m": log_nu_m,
                    "log_nu_c": log_nu_c,
                    "log_nu_max": log_nu_max,
                    "log_nu_ssa": log_nu_a,
                    "regime": regime.__name__,
                },
            )

        return regime, log_nu_a

    def _compute_sed_regime(
        self,
        log_F_norm: float,
        log_nu_m: float,
        log_nu_c: float,
        log_nu_max: float,
        log_omega: float,
        log_gamma_m: float,
        p: float = 3.0,
    ):
        r"""
        Determine the synchrotron spectral regime and SSA frequency.

        This low-level method classifies the synchrotron spectrum into a **single,
        global spectral regime** based on the ordering of the characteristic
        frequencies and returns both:

        1. A discrete **regime identifier**, and
        2. The corresponding **self-absorption frequency** :math:`\nu_a`
           appropriate to that regime.

        The classification proceeds in two stages:

        1. Determine whether the system is **fast-cooling**, **slow-cooling**, or
           **non-cooling** by comparing :math:`\nu_c` and :math:`\nu_m`.
        2. Within that cooling class, select the specific SSA spectrum by comparing
           the candidate self-absorption frequencies to :math:`\nu_m` and
           :math:`\nu_c`.

        All inputs and outputs are assumed to be in **natural logarithmic space**.

        Parameters
        ----------
        log_F_norm : float
            Natural logarithm of the normalization flux density corresponding to the *optically thin* equivalent
            emission at the dominant electron frequency. See
            :ref:`sed_normalization` for details on how to determine the appropriate normalization.
        log_nu_m : float
            Natural logarithm of the injection frequency
            :math:`\log \nu_m`.
        log_nu_c : float
            Natural logarithm of the cooling frequency
            :math:`\log \nu_c`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\log \nu_{\max}`.
        log_omega : float
            Natural logarithm of the effective emission solid angle
            :math:`\log \Omega`.
        log_gamma_m : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\log \gamma_m`.

        Returns
        -------
        tuple
            A two-element tuple ``(regime, log_nu_a)`` where:

            - ``regime`` is a member of
              :class:`~_SynchrotronSSACoolingSEDFunctions` identifying the spectral
              branch to evaluate.
            - ``log_nu_a`` is the natural logarithm of the self-absorption frequency
              appropriate to that regime.

        Notes
        -----
        - This method performs **no unit checking** and assumes all inputs are valid.
        - The returned regime applies **globally** to the spectrum.
        - Errors are raised if the frequency ordering is inconsistent with any
          supported spectral configuration.
        """
        # Begin by calculating the SSA frequency from the other parameters. This also
        # tells some regime information. 0=Fast cooling, 1=Slow cooling, 2=No cooling.
        log_nu_ssa, cooling_regime = self._compute_ssa_frequencies_from_F_norm(
            log_F_norm, log_nu_m, log_nu_c, log_nu_max, log_omega, log_gamma_m, p
        )
        return self._compute_sed_regime_from_nu_a_dict(
            log_nu_ssa,
            cooling_regime,
            log_nu_m,
            log_nu_c,
            log_nu_max,
        )

    def determine_sed_regime(
        self,
        F_norm: "_UnitBearingScalarLike",
        nu_m: "_UnitBearingScalarLike",
        nu_c: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf,
        omega: float = 1.0,
        gamma_m: float = 1.0,
    ):
        r"""
        Determine the synchrotron spectral regime from physical parameters.

        This is the **user-facing interface** for regime determination. It performs
        unit validation and coercion, converts all inputs to logarithmic CGS form,
        and dispatches to the optimized internal routine
        :meth:`_compute_sed_regime`.

        This method is intended for **diagnostic and introspection purposes** and
        does not compute the SED itself.

        Parameters
        ----------
        F_norm : ~astropy.units.Quantity or float
            Normalization flux density corresponding to the *optically thin* equivalent
            emission at the dominant electron frequency. This should be a :class:`~astropy.units.Quantity`
            with units convertible to :math:`\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1} \, \mathrm{Hz}^{-1}`
            or a float in CGS units. See :ref:`sed_normalization` for details on how to determine the appropriate
            normalization.
        nu_m : ~astropy.units.Quantity or float
            Injection frequency :math:`\nu_m`. May be either a :class:`~astropy.units.Quantity`
            with units convertible to Hz or a float in CGS units.
        nu_c : ~astropy.units.Quantity or float
            Cooling frequency :math:`\nu_c`. May be either a :class:`~astropy.units.Quantity`
            with units convertible to Hz or a float in CGS units.
        nu_max : ~astropy.units.Quantity or float
            Maximum synchrotron frequency :math:`\nu_{\max}`. May be either a :class:`~astropy.units.Quantity`
            with units convertible to Hz or a float in CGS units. The default (``np.inf``) corresponds to no cutoff.
        omega : float
            Effective emission solid angle. This should be a floating-point number
            representing the ratio of the emitting area to the square of the distance
            to the observer (i.e. :math:`\Omega = A / D^2`).
        gamma_m : float
            Minimum electron Lorentz factor. This should be a dimensionless floating-point number.

        Returns
        -------
        regime : callable
            A member of the :attr:`~PowerLaw_Cooling_SSA_SynchrotronSED.SPECTRUM_FUNCTIONS` enumeration identifying the
            applicable synchrotron spectral regime.

            Each enum member encodes a specific ordering of the characteristic
            synchrotron frequencies (e.g. :math:`\nu_a`, :math:`\nu_m`, :math:`\nu_c`,
            :math:`\nu_{\max}`) and uniquely determines the corresponding spectral
            structure.

            Enum members are **callable** and may be invoked to construct or dispatch
            the appropriate SED implementation, or to access regime-specific metadata
            such as break ordering and spectral slopes.

        Notes
        -----
        - The returned regime applies **globally** to the SED.
        - This method does not depend on the frequency array ``nu``.
        """
        # Handle units and then coerce to log space
        F_norm = ensure_in_units(F_norm, "erg / (cm**2 * s * Hz)")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_c = ensure_in_units(nu_c, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")

        # Coerce to log space
        log_F_peak = np.log(F_norm)
        log_nu_m = np.log(nu_m)
        log_nu_c = np.log(nu_c)
        log_nu_max = np.log(nu_max)
        log_omega = np.log(omega)
        log_gamma_m = np.log(gamma_m)

        # Dispatch to the low-level regime computation
        regime, _ = self._compute_sed_regime(
            log_F_norm=log_F_peak,
            log_nu_m=log_nu_m,
            log_nu_c=log_nu_c,
            log_nu_max=log_nu_max,
            log_omega=log_omega,
            log_gamma_m=log_gamma_m,
        )
        return regime

    # ============================================================ #
    # Regime-Specific SED Kernel                                   #
    # ============================================================ #
    # noinspection PyCallingNonCallable
    def _log_opt_sed_from_regime(
        self,
        regime: _SynchrotronSSACoolingSEDFunctions,
        log_nu: "_ArrayLike",
        log_nu_m: float,
        log_nu_a: float,
        log_nu_c: float,
        log_nu_max: float,
        log_F_norm: float,
        log_nu_ac: float = None,
        p: float = 3.0,
        s: float = -1.0,
    ):
        r"""
        Evaluate the log-space SED for a fixed spectral regime.

        This method dispatches to the **regime-specific numerical kernel**
        corresponding to the supplied regime identifier and applies the overall
        flux normalization.

        All branching logic on spectral shape is confined to this method.

        Parameters
        ----------
        regime : enum
            Spectral regime identifier from
            :class:`~_SynchrotronSSACoolingSEDFunctions`.
        log_nu : array-like
            Natural logarithm of the evaluation frequencies.
        log_nu_m, log_nu_a, log_nu_c, log_nu_max : float
            Natural logarithms of the characteristic frequencies.
        log_F_norm : float
            Natural logarithm of the flux normalization corresponding to the *optically thin* equivalent
            emission at the dominant electron frequency. See
            :ref:`sed_normalization` for details on how to determine the appropriate normalization.
        log_nu_ac : float, optional
            Natural logarithm of the stratified SSA transition frequency.
        p : float
            Electron power-law index.
        s : float
            Smoothing parameter for spectral breaks.

        Returns
        -------
        array-like
            Natural logarithm of the SED evaluated at ``log_nu``.

        Notes
        -----
        - This method assumes the regime is **already validated**.
        - All SED kernels are evaluated in log-space for numerical stability.
        """
        # Given that we have the regime, we can dispatch to the appropriate function.
        if regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_1:
            return regime(log_nu, log_nu_m, log_nu_a, log_nu_max, p, s) + log_F_norm
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_2:
            return (
                regime(log_nu, log_nu_m, log_nu_a, log_nu_max, p, s)
                + log_F_norm
                - ((p - 1) / 2) * (log_nu_a - log_nu_m)
            )
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_3:
            return regime(log_nu, log_nu_m, log_nu_c, log_nu_a, log_nu_max, p, s) + log_F_norm
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_4:
            return (
                regime(log_nu, log_nu_m, log_nu_c, log_nu_a, log_nu_max, p, s)
                + log_F_norm
                - ((p - 1) / 2) * (log_nu_a - log_nu_m)
            )
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_5:
            return regime(log_nu, log_nu_m, log_nu_c, log_nu_a, log_nu_ac or log_nu_a, log_nu_max, p, s) + log_F_norm
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_6:
            return (
                regime(log_nu, log_nu_m, log_nu_a, log_nu_ac or log_nu_a, log_nu_max, p, s)
                + log_F_norm
                - (1 / 2) * (log_nu_a - log_nu_c)
            )
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_7:
            return (
                regime(log_nu, log_nu_m, log_nu_a, log_nu_max, p, s)
                + log_F_norm
                + (-1 / 2) * (log_nu_m - log_nu_c)
                + (-p / 2) * (log_nu_a - log_nu_m)
            )
        else:
            raise RuntimeError("Unrecognized regime in _log_opt_sed_from_regime.")

    def _log_opt_sed(
        self,
        log_nu: "_ArrayLike",
        log_nu_m: float,
        log_nu_c: float,
        log_nu_max: float,
        log_F_norm: float,
        log_omega: float,
        log_gamma_m: float,
        log_nu_ac: float = None,
        p: float = 3.0,
        s: float = -1.0,
    ):
        r"""
        Optimized log-space SED evaluation with regime determination.

        This method orchestrates the full SED evaluation by:

        1. Determining the global spectral regime,
        2. Computing the corresponding self-absorption frequency,
        3. Dispatching to the appropriate regime-specific kernel.

        All inputs are assumed to be in **natural logarithmic CGS units**.

        Returns
        -------
        array-like
            Natural logarithm of the synchrotron SED evaluated at ``log_nu``.
        """
        # Determine the regime
        regime, log_nu_a = self._compute_sed_regime(
            log_F_norm,
            log_nu_m,
            log_nu_c,
            log_nu_max,
            log_omega,
            log_gamma_m,
        )
        # Dispatch to the appropriate regime function
        return self._log_opt_sed_from_regime(
            regime,
            log_nu,
            log_nu_m,
            log_nu_a,
            log_nu_c,
            log_nu_max,
            log_F_norm,
            log_nu_ac,
            p,
            s,
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        nu_m: "_UnitBearingScalarLike",
        nu_c: "_UnitBearingScalarLike",
        F_norm: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf,
        nu_ac: "_UnitBearingScalarLike" = None,
        omega: float = 4 * np.pi,
        gamma_m: float = 1,
        p: float = 3,
        s: float = -1,
    ):
        r"""
        Evaluate the synchrotron spectral energy distribution.

        This is the **primary user-facing interface** for computing the synchrotron
        SED. It performs unit validation, converts inputs to logarithmic CGS form,
        and dispatches to the optimized internal implementation.

        Parameters
        ----------
        nu : ~astropy.units.Quantity or float or array-like
            Frequencies at which to evaluate the SED.
        nu_m, nu_c, nu_max : ~astropy.units.Quantity or float
            Injection, cooling, and maximum synchrotron frequencies.
        F_norm : ~astropy.units.Quantity or float
            The normalization flux density corresponding to the *optically thin* equivalent
            emission at the dominant electron frequency. See :ref:`sed_normalization` for details on how
            to determine the appropriate normalization.
        nu_ac : ~astropy.units.Quantity or float, optional
            Stratified SSA transition frequency.
        omega : float
            Effective emission solid angle.
        gamma_m : float
            Minimum electron Lorentz factor.
        p : float
            Electron power-law index.
        s : float
            Spectral smoothing parameter.

        Returns
        -------
        astropy.units.Quantity
            Flux density :math:`F_\nu` evaluated at ``nu``.

        Notes
        -----
        - This is the **only method** that enforces units.
        - All internal calculations are performed in log-space.
        """
        # Enforce units on each of the inputs
        nu = ensure_in_units(nu, "Hz")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_c = ensure_in_units(nu_c, "Hz")
        F_norm = ensure_in_units(F_norm, "erg s^-1 cm^-2 Hz^-1")
        nu_max = ensure_in_units(nu_max, "Hz")
        nu_ac = ensure_in_units(nu_ac, "Hz") if nu_ac is not None else None

        # Dispatch to the optimized log-space SED
        log_sed = self._log_opt_sed(
            np.log(nu),
            np.log(nu_m),
            np.log(nu_c),
            np.log(nu_max),
            np.log(F_norm),
            np.log(omega),
            np.log(gamma_m),
            log_nu_ac=np.log(nu_ac) if nu_ac is not None else None,
            p=p,
            s=s,
        )

        return np.exp(log_sed) * u.erg / (u.s * u.cm**2 * u.Hz)

    # =========================================================== #
    # Closure Relations Implementation                            #
    # =========================================================== #
    # Here we implement the closure relations to go forward and backward
    # between the physics parameters and the phenomenological SED parameters.
    def _opt_from_physics_to_params(
        self,
        log_B: float,
        log_V: float,
        log_D_L: float,
        log_Omega: float,
        log_gamma_min: float,
        log_gamma_c: float,
        log_gamma_max: float = np.inf,
        p: float = 2.5,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        pitch_average: bool = False,
    ):
        r"""
        Low-level equipartition-based closure for synchrotron SED normalization.

        This method maps **physical model parameters** describing the emitting region
        and electron population to a **self-consistent set of phenomenological SED
        parameters** for synchrotron emission with cooling and self-absorption.

        It is the optimized, log-space implementation underlying
        :meth:`from_physics_to_params` and is intended **only for internal use**.
        All inputs and outputs are assumed to be expressed in **natural logarithmic
        CGS units**, and no unit validation is performed.

        The procedure implemented here follows a physically motivated decision tree:

        1. Compute characteristic synchrotron frequencies
           (:math:`\nu_m`, :math:`\nu_c`, :math:`\nu_{\max}`) from the supplied Lorentz
           factors and magnetic field.
        2. Determine whether the system is **fast-cooling** or **slow-/non-cooling**
           by comparing :math:`\gamma_c` and :math:`\gamma_m`.
        3. Select the appropriate electron energy distribution and normalization:
           - Fast cooling: broken power-law (BPL) anchored at :math:`\gamma_c`,
           - Slow / no cooling: power-law (PL) anchored at :math:`\gamma_m`.
        4. Compute a provisional spectral normalization via equipartition.
        5. Compute candidate self-absorption frequencies for all compatible spectral
           regimes.
        6. Select the unique, globally consistent synchrotron regime.
        7. Propagate the normalization to the **true spectral peak**, if required by
           the regime (i.e. optically thick cases).

        The output parameters are guaranteed to be **mutually consistent** with the
        selected synchrotron regime.

        Parameters
        ----------
        log_B : float
            Natural logarithm of the magnetic field strength (Gauss).
        log_V : float
            Natural logarithm of the effective emitting volume (cm³).
        log_D_L : float
            Natural logarithm of the luminosity distance (cm).
        log_Omega : float
            Natural logarithm of the effective emission solid angle
            :math:`\Omega = A / D^2`.
        log_gamma_min : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\gamma_m`.
        log_gamma_c : float
            Natural logarithm of the cooling Lorentz factor
            :math:`\gamma_c`.
        log_gamma_max : float, optional
            Natural logarithm of the maximum electron Lorentz factor.
            Defaults to :math:`+\infty`.
        p : float, optional
            Power-law index of the injected electron energy distribution.
        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic electrons.
        epsilon_B : float, optional
            Fraction of post-shock internal energy stored in magnetic fields.
        alpha : float, optional
            Electron pitch angle in radians. Ignored if ``pitch_average=True``.
        gamma_bulk: float, optional
            Bulk Lorentz factor of the emitting region. This can be used to compute the Doppler
            factor for transforming frequencies and fluxes to the observer frame. Defaults to 1
            (no relativistic beaming).
        redshift: float, optional
            Redshift of the source. This can be used to compute the luminosity distance and to
            apply cosmological corrections to the frequencies and fluxes. Defaults to 0 (no redshift).
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.
            Otherwise, a fixed pitch angle is assumed.

        Returns
        -------
        params : dict
            Dictionary containing **logarithmic** phenomenological SED parameters:

            - ``F_peak`` : :math:`\log F_{\nu,\mathrm{pk}}`
            - ``nu_m`` : :math:`\log \nu_m`
            - ``nu_c`` : :math:`\log \nu_c`
            - ``nu_a`` : :math:`\log \nu_a`
            - ``nu_max`` : :math:`\log \nu_{\max}`
            - ``regime`` : Enum identifying the selected synchrotron spectral regime

        Notes
        -----
        - This method performs **no unit checking** and assumes all inputs are valid.
        - All calculations are performed in **log-space** for numerical stability.
        - Equipartition is an **assumption**, not a physical requirement.
        - This method enforces **global regime consistency**; mixed-regime solutions
          are not permitted.
        - The returned normalization corresponds to the **true spectral peak**, which
          may differ from the initial anchoring frequency.

        See Also
        --------
        from_physics_to_params :
            Public, unit-aware interface to this closure relation.
        _compute_sed_regime :
            Global synchrotron regime determination logic.
        _compute_ssa_frequencies_from_F_norm :
            SSA frequency computation using anchored spectral normalizations.
        """
        # Handle pitch angle details and the relevant values of
        # the log_chi parameter in each case. This allows us to
        # permit both a fixed pitch angle and pitch-angle averaging.
        sin_alpha = np.sin(alpha)
        log_sin_alpha = np.log(sin_alpha) if not pitch_average else 0.0
        log_chi = _log_chi_cgs_iso if pitch_average else _log_chi_cgs

        # Compute the Doppler factors and redshift corrections for the
        # fluxes and the frequencies.
        beta_bulk = np.sqrt(1 - gamma_bulk**-2)
        delta_bulk = gamma_bulk * (1 + beta_bulk)
        log_delta_bulk = np.log(delta_bulk)
        log_nu_correction = log_delta_bulk - np.log1p(redshift)
        log_flux_correction = 3 * log_delta_bulk - np.log1p(redshift)

        # --- Compute the Frequencies --- #
        # We need to use the electron Lorentz factors to compute
        # the relevant synchrotron frequencies. This should be done for
        # gamma_c, gamma_min, and gamma_max.
        log_nu_m = (
            _opt_compute_log_synch_frequency(
                log_gamma_min,
                log_B,
                sin_alpha=sin_alpha,
                pitch_average=pitch_average,
            )
            + log_nu_correction
        )
        log_nu_max = (
            _opt_compute_log_synch_frequency(
                log_gamma_max,
                log_B,
                sin_alpha=sin_alpha,
                pitch_average=pitch_average,
            )
            + log_nu_correction
        )
        log_nu_c = (
            _opt_compute_log_synch_frequency(
                log_gamma_c,
                log_B,
                sin_alpha=sin_alpha,
                pitch_average=pitch_average,
            )
            + log_nu_correction
        )

        # Determine if we are fast cooling or not.
        _FAST_COOLING_FLAG = log_gamma_c < log_gamma_min

        # Log that we have computed the frequencies, as this is a critical step in the closure.
        if triceratops_logger.isEnabledFor(logging.DEBUG):
            triceratops_logger.debug(
                "PowerLaw_Cooling_SSA_SynchrotronSED: computed characteristic frequencies.",
                extra={
                    "log_gamma_min": log_gamma_min,
                    "log_gamma_c": log_gamma_c,
                    "log_gamma_max": log_gamma_max,
                    "log_nu_m": log_nu_m,
                    "log_nu_c": log_nu_c,
                    "log_nu_max": log_nu_max,
                    "fast_cooling": _FAST_COOLING_FLAG,
                },
            )

        # --- Determine the Normalizations for the Electrons --- #
        #  At this stage, we immediately know if we have fast or slow cooling and
        #  therefore which frequency we are going to anchor the electron distribution to.
        if _FAST_COOLING_FLAG:
            # The dominant electron population will occur at nu_c and require us to
            # normalize via equipartition over a BPL distribution at gamma_c.
            log_electron_norm = np.log(
                _opt_normalize_BPL_from_magnetic_field(
                    np.exp(log_B),
                    -(p + 1),  # TODO: THIS NEEDS FIXING!
                    -p,
                    gamma_b=np.exp(log_gamma_min),
                    gamma_min=np.exp(log_gamma_c),
                    gamma_max=np.exp(log_gamma_max),
                    epsilon_E=epsilon_E,
                    epsilon_B=epsilon_B,
                )
            )

            log_F_norm = (
                log_chi
                + log_B
                + log_sin_alpha  # Zero if pitch averaged
                + log_electron_norm
                + log_nu_m
                - log_nu_c
                + log_gamma_c
                + log_V
                - 2 * log_D_L
                + log_flux_correction
            )

            # Compute the SSA frequencies.
            regime, log_nu_a = self._compute_sed_regime(
                log_F_norm, log_nu_m, log_nu_c, log_nu_max, log_Omega, log_gamma_min
            )

        else:
            # The dominant electron population will occur at nu_m and require us to
            # normalize via equipartition over a PL distribution at gamma_min.
            log_electron_norm = np.log(
                _opt_normalize_PL_from_magnetic_field(
                    np.exp(log_B),
                    p,
                    gamma_min=np.exp(log_gamma_min),
                    gamma_max=np.exp(log_gamma_max),
                    epsilon_E=epsilon_E,
                    epsilon_B=epsilon_B,
                )
            )
            log_F_norm = (
                log_chi
                + log_B
                + log_sin_alpha  # Zero if pitch averaged
                + log_electron_norm
                + (1 - p) * log_gamma_min
                + log_V
                - 2 * log_D_L
                + log_flux_correction
            )

            regime, log_nu_a = self._compute_sed_regime(
                log_F_norm, log_nu_m, log_nu_c, log_nu_max, log_Omega, log_gamma_min
            )

        # --- LOG AND RETURN --- #
        if triceratops_logger.isEnabledFor(logging.DEBUG):
            triceratops_logger.debug(
                "PowerLaw_Cooling_SSA_SynchrotronSED: completed normalization.",
                extra={
                    "log_gamma_min": log_gamma_min,
                    "log_gamma_c": log_gamma_c,
                    "log_gamma_max": log_gamma_max,
                    "log_nu_m": log_nu_m,
                    "log_nu_c": log_nu_c,
                    "log_nu_max": log_nu_max,
                    "log_nu_a": log_nu_a,
                    "fast_cooling": _FAST_COOLING_FLAG,
                    "regime": regime.__name__,
                    "log_electron_norm": log_electron_norm,
                },
            )

        return {
            "F_norm": log_F_norm,
            "nu_m": log_nu_m,
            "nu_c": log_nu_c,
            "nu_a": log_nu_a,
            "nu_max": log_nu_max,
            "regime": regime,
        }

    def from_physics_to_params(
        self,
        B: "_UnitBearingScalarLike",
        V: "_UnitBearingScalarLike",
        D_L: "_UnitBearingScalarLike",
        Omega: float,
        gamma_min: float,
        gamma_c: float,
        gamma_max: float = np.inf,
        p: float = 2.5,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        pitch_average: bool = False,
    ):
        r"""
        Determine the equipartition parameters for this SED from physical inputs.

        This method utilizes the normalization scheme described in :ref:`synch_sed_theory`
        (specifically :meth:`sed_normalization`) to compute the phenomenological SED parameters
        (i.e. :math:`F_{\nu,\mathrm{pk}}`, :math:`\nu_m`, :math:`\nu_c`, :math:`\nu_a`, etc.)
        from the underlying physical parameters of the system.

        .. note::

            Users are **strongly encouraged** to read the notes in :ref:`synch_sed_theory`
            regarding the assumptions and limitations of this approach before applying
            it to real systems.

        Parameters
        ----------
        B : ~astropy.units.Quantity or float
            Magnetic field strength. Must be convertible to Gauss.
        V : ~astropy.units.Quantity or float
            Effective emitting volume. Must be convertible to ``cm^3``.
        D_L : ~astropy.units.Quantity or float
            Luminosity distance to the source. Must be convertible to ``cm``.
        Omega : float
            Effective emission solid angle :math:`\Omega = A / D^2`.
        gamma_min : float
            Minimum electron Lorentz factor :math:`\gamma_m`.
        gamma_c : float
            Cooling Lorentz factor :math:`\gamma_c`.
        gamma_max : float, optional
            Maximum electron Lorentz factor. Default corresponds to no cutoff.
        p : float, optional
            Power-law index of the injected electron energy distribution.
        epsilon_E : float, optional
            Fraction of post-shock internal energy in relativistic electrons.
        epsilon_B : float, optional
            Fraction of post-shock internal energy in magnetic fields.
        alpha : float, optional
            Electron pitch angle in radians. Ignored if ``pitch_average=True``.
        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting region. This can be used to compute the Doppler
            factor for transforming frequencies and fluxes to the observer frame. Defaults to 1
            (no relativistic beaming).
        redshift : float, optional
            Redshift of the source. This can be used to compute the luminosity distance and to
            apply cosmological corrections to the frequencies and fluxes. Defaults to 0 (no red
            shift).
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.

        Returns
        -------
        params : dict
            Dictionary containing the phenomenological SED parameters:

            - ``F_peak`` : Peak flux density (:math:`F_{\nu,\mathrm{pk}}`)
              with units ``erg cm^-2 s^-1 Hz^-1``.
            - ``nu_m`` : Injection frequency :math:`\nu_m` (Hz).
            - ``nu_c`` : Cooling frequency :math:`\nu_c` (Hz).
            - ``nu_a`` : Self-absorption frequency :math:`\nu_a` (Hz).
            - ``nu_max`` : Maximum synchrotron frequency :math:`\nu_{\max}` (Hz).
            - ``regime`` : Enum identifying the global synchrotron spectral regime.

        Notes
        -----
        - This method assumes a **single-zone, homogeneous emitting region**.
        - Equipartition is an **assumption**, not a physical necessity.
        - The returned parameters are guaranteed to be **self-consistent** with
          the selected synchrotron spectral regime.
        """
        # Coerce things down to unit carrying values.
        B = ensure_in_units(B, "G")
        V = ensure_in_units(V, "cm^3")
        D_L = ensure_in_units(D_L, "cm")

        # Get everything in log-space. The low-level implementation is
        # done natively in log-space for stability.
        log_B = np.log(B)
        log_V = np.log(V)
        log_D_L = np.log(D_L)
        log_Omega = np.log(Omega)

        log_gamma_min = np.log(gamma_min)
        log_gamma_c = np.log(gamma_c)
        log_gamma_max = np.log(gamma_max)

        # Dispatch to the optimized low-level implementation.
        params_log = self._opt_from_physics_to_params(
            log_B=log_B,
            log_V=log_V,
            log_D_L=log_D_L,
            log_Omega=log_Omega,
            log_gamma_min=log_gamma_min,
            log_gamma_c=log_gamma_c,
            log_gamma_max=log_gamma_max,
            p=p,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            alpha=alpha,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            pitch_average=pitch_average,
        )

        # Convert back to linear CGS units and return.
        return {
            "F_norm": np.exp(params_log["F_norm"]) * u.erg / (u.cm**2 * u.s * u.Hz),
            "nu_m": np.exp(params_log["nu_m"]) * u.Hz,
            "nu_c": np.exp(params_log["nu_c"]) * u.Hz,
            "nu_a": np.exp(params_log["nu_a"]) * u.Hz,
            "nu_max": np.exp(params_log["nu_max"]) * u.Hz,
            "regime": params_log["regime"],
        }


class PowerLaw_Cooling_SynchrotronSED(MultiSpectrumSynchrotronSED):
    r"""
    Synchrotron SED with cooling but no self-absorption.

    This class implements the **canonical optically thin synchrotron spectral
    energy distribution** produced by a power-law electron energy distribution
    subject to radiative cooling, **without synchrotron self-absorption (SSA)**.

    It is appropriate for emission regions where the synchrotron self-absorption
    turnover frequency :math:`\nu_a` lies well below the lowest frequency of
    interest, and the emitting plasma can be treated as homogeneous.

    The spectrum is constructed using **smoothed broken power laws (SBPLs)** and
    supports all standard cooling regimes encountered in synchrotron theory.
    Regime selection is **global** and based entirely on the ordering of the
    characteristic break frequencies.

    For a detailed theoretical derivation of the spectral slopes and normalization
    conventions, see :ref:`synchrotron_theory`.

    .. rubric:: Spectral Structure

    The synchrotron spectrum is classified into one of three cooling regimes
    based on the ordering of the characteristic frequencies:

    - Injection frequency :math:`\nu_m`,
    - Cooling frequency :math:`\nu_c`,
    - Maximum synchrotron frequency :math:`\nu_{\max}`.

    The supported regimes are:

    **Fast cooling** (:math:`\nu_c < \nu_m`)

    .. math::

        F_\nu \propto
        \begin{cases}
            \nu^{1/3}, & \nu < \nu_c \
            \nu^{-1/2}, & \nu_c < \nu < \nu_m \
            \nu^{-p/2}, & \nu > \nu_m
        \end{cases}

    **Slow cooling** (:math:`\nu_m < \nu_c < \nu_{\max}`)

    .. math::

        F_\nu \propto
        \begin{cases}
            \nu^{1/3}, & \nu < \nu_m \
            \nu^{-(p-1)/2}, & \nu_m < \nu < \nu_c \
            \nu^{-p/2}, & \nu > \nu_c
        \end{cases}

    **Effectively non-cooling** (:math:`\nu_c > \nu_{\max}`)

    .. math::

        F_\nu \propto
        \begin{cases}
            \nu^{1/3}, & \nu < \nu_m \
            \nu^{-(p-1)/2}, & \nu > \nu_m
        \end{cases}

    The selected regime applies **globally** to the spectrum and does not depend
    on the frequency grid used for evaluation.

    .. rubric:: Parameters

    The parameters entering this SED fall into three conceptual categories.

    .. tab-set::

        .. tab-item:: Free parameters (phenomenological)

            These parameters define the observable structure of the SED and are
            typically inferred directly from broadband data.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Flux normalization
                  - :math:`F_{\nu,\mathrm{norm}}`
                  - Flux density normalization corresponding to the *optically thin* equivalent
                    emission at the dominant electron frequency. See :ref:`sed_normalization` for details
                    on how to determine the appropriate normalization.
                * - Injection frequency
                  - :math:`\nu_m`
                  - Synchrotron frequency of minimum-energy electrons
                * - Cooling frequency
                  - :math:`\nu_c`
                  - Frequency corresponding to the cooling Lorentz factor
                * - Maximum frequency
                  - :math:`\nu_{\max}`
                  - High-energy synchrotron cutoff frequency

        .. tab-item:: Hyper-parameters

            These parameters control the *shape* and smoothness of the spectrum
            but are not usually tightly constrained by broadband observations.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Electron power-law index
                  - :math:`p`
                  - Index of the injected electron energy distribution
                * - Smoothing parameter
                  - :math:`s`
                  - Controls the sharpness of spectral breaks
                * - Minimum Lorentz factor
                  - :math:`\gamma_m`
                  - Minimum electron Lorentz factor (used in normalization)

        .. tab-item:: Derived quantities (internal)

            These quantities are **not user inputs**, but are determined internally
            from the supplied parameters.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Quantity
                  - Symbol
                  - Description
                * - Cooling regime
                  - —
                  - Fast, slow, or effectively non-cooling classification
                * - Regime identifier
                  - —
                  - Discrete index selecting the appropriate SED kernel

    .. rubric:: Normalization and Closure

    The SED is normalized using :math:`F_{\nu,\mathrm{norm}}`, defined as the flux
    density of the dominant optically thin emitting electron population. The
    frequency at which this population emits depends on the cooling regime.

    In the fast-cooling regime, normalization is anchored at the cooling frequency
    :math:`\nu_c`; in the slow- and non-cooling regimes, normalization is anchored
    at the injection frequency :math:`\nu_m`.

    Optional equipartition-based closure relations are provided to map physical
    model parameters onto the phenomenological SED parameters.

    See Also
    --------
    :class:`SynchrotronSED` : Base class for synchrotron SED implementations.
    :class:`MultiSpectrumSynchrotronSED` : Base class for multi-regime synchrotron SEDs.
    :class:`PowerLaw_SynchrotronSED` : Synchrotron SED without cooling or self-absorption.
    :class:`PowerLaw_SSA_SynchrotronSED` : Synchrotron SED with synchrotron self-absorption.
    :class:`PowerLaw_Cooling_SSA_SynchrotronSED` : Synchrotron SED with cooling and self-absorption.
    """

    SPECTRUM_FUNCTIONS = _SynchrotronCoolingSEDFunctions

    # ------------------------------------------------------------ #
    # Regime determination                                        #
    # ------------------------------------------------------------ #
    def _compute_sed_regime(self, log_nu_m: float, log_nu_c: float, log_nu_max: float):
        r"""
        Determine the optically thin synchrotron cooling regime.

        This low-level method classifies the synchrotron spectrum based on the
        ordering of the characteristic frequencies in log-space:

        .. math::

            \nu_m \quad \text{(injection frequency)}, \qquad
            \nu_c \quad \text{(cooling frequency)}, \qquad
            \nu_{\max} \quad \text{(high-frequency cutoff)}.

        The regime determination is **global** and applies to the entire SED;
        it does not depend on the evaluation frequency array ``nu``.

        Parameters
        ----------
        log_nu_m : float
            Natural logarithm of the injection frequency :math:`\nu_m`.
        log_nu_c : float
            Natural logarithm of the cooling frequency :math:`\nu_c`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`.

        Returns
        -------
        regime : enum-like
            A member of :attr:`SPECTRUM_FUNCTIONS` identifying the cooling regime.
        metadata : dict
            Empty dictionary (present for API consistency with multi-regime SEDs
            that require auxiliary derived parameters).

        Notes
        -----
        The regimes correspond to the following orderings:

        - **Fast cooling**: :math:`\nu_c < \nu_m`
        - **Slow cooling**: :math:`\nu_m < \nu_c < \nu_{\max}`
        - **Non-cooling**: :math:`\nu_c > \nu_{\max}`

        This method assumes that all inputs are already in log-space and does
        not perform unit validation.
        """
        if log_nu_c <= log_nu_m:
            regime = self.SPECTRUM_FUNCTIONS.SPECTRUM_1
        elif log_nu_m < log_nu_c < log_nu_max:
            regime = self.SPECTRUM_FUNCTIONS.SPECTRUM_2
        elif log_nu_c > log_nu_max:
            regime = self.SPECTRUM_FUNCTIONS.SPECTRUM_3
        else:
            raise RuntimeError("Could not determine cooling regime.")

        # Now complete, log that we have completed the step, and return.
        if triceratops_logger.isEnabledFor(logging.DEBUG):
            triceratops_logger.debug(
                "PowerLaw_Cooling_SSA_SynchrotronSED: selected regime.",
                extra={
                    "log_nu_m": log_nu_m,
                    "log_nu_c": log_nu_c,
                    "log_nu_max": log_nu_max,
                    "regime": regime.__name__,
                },
            )

        return regime, {}

    def determine_sed_regime(
        self, nu_m: "_UnitBearingScalarLike", nu_c: "_UnitBearingScalarLike", nu_max: "_UnitBearingScalarLike" = np.inf
    ):
        r"""
        Public interface for determining the synchrotron cooling regime.

        This method provides a user-facing wrapper around
        :meth:`_compute_sed_regime`. It handles unit validation and conversion
        to log-space before performing regime classification.

        Parameters
        ----------
        nu_m : ~astropy.units.Quantity or float
            Injection frequency :math:`\nu_m`.
        nu_c : ~astropy.units.Quantity or float
            Cooling frequency :math:`\nu_c`.
        nu_max : ~astropy.units.Quantity or float, optional
            Maximum synchrotron frequency :math:`\nu_{\max}`.
            Defaults to :math:`\infty`.

        Returns
        -------
        regime : enum-like
            Identifier specifying the **global cooling regime** of the synchrotron
            spectrum, determined by the ordering of :math:`\nu_m`, :math:`\nu_c`,
            and :math:`\nu_{\max}`.

            The returned value is a member of
            :attr:`PowerLaw_Cooling_SynchrotronSED.SPECTRUM_FUNCTIONS` and uniquely
            selects the corresponding optically thin spectral kernel.

        Notes
        -----
        - This method does **not** depend on the frequency array used to
          evaluate the SED.
        - The returned regime must be consistent with the spectrum produced
          by :meth:`sed` for the same parameters.
        """
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_c = ensure_in_units(nu_c, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")

        regime, _ = self._compute_sed_regime(
            log_nu_m=np.log(nu_m),
            log_nu_c=np.log(nu_c),
            log_nu_max=np.log(nu_max),
        )

        return regime

    # ------------------------------------------------------------ #
    # Regime-specific kernel                                       #
    # ------------------------------------------------------------ #
    def _log_opt_sed_from_regime(
        self,
        log_nu: float,
        regime: Callable,
        log_nu_m: float,
        log_nu_c: float,
        log_nu_max: float,
        log_F_norm: float,
        p: float,
        s: float,
    ):
        r"""
        Evaluate the log-space synchrotron SED for a fixed cooling regime.

        This method implements the **numerical kernel** for computing the
        synchrotron SED once the cooling regime has been determined. All
        branching on the regime identifier occurs here.

        Parameters
        ----------
        log_nu : array-like
            Natural logarithm of the frequencies at which to evaluate the SED.
        regime : enum-like
            Cooling regime identifier returned by
            :meth:`_compute_sed_regime`.
        log_nu_m : float
            Natural logarithm of the injection frequency :math:`\nu_m`.
        log_nu_c : float
            Natural logarithm of the cooling frequency :math:`\nu_c`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`.
        log_F_norm : float
            Natural logarithm of the flux density normalization corresponding to
            the *optically thin* equivalent emission at the dominant electron frequency.
        p : float
            Electron energy power-law index.
        s : float
            SBPL smoothness parameter.

        Returns
        -------
        log_sed : array-like
            Natural logarithm of the flux density
            :math:`\log F_\nu` evaluated at ``log_nu``.

        Notes
        -----
        - The returned spectrum is **not normalized** until ``log_F_peak``
          is added.
        - This method assumes an optically thin emitting region and does not
          include synchrotron self-absorption.
        - No unit validation is performed.
        """
        if regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_1:
            log_sed = regime(log_nu, log_nu_m, log_nu_c, log_nu_max, p, s)
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_2:
            log_sed = regime(log_nu, log_nu_m, log_nu_c, log_nu_max, p, s)
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_3:
            log_sed = regime(log_nu, log_nu_m, log_nu_max, p, s)
        else:
            raise RuntimeError("Unrecognized cooling regime.")

        return log_sed + log_F_norm

    def _log_opt_sed(self, log_nu, log_nu_m, log_nu_c, log_nu_max, log_F_norm, p, s):
        r"""
        Optimized log-space SED evaluation with regime determination.

        This method orchestrates the full SED evaluation by:

        1. Determining the global spectral regime,
        2. Dispatching to the appropriate regime-specific kernel.

        All inputs are assumed to be in **natural logarithmic CGS units**.

        Parameters
        ----------
        log_nu : "_ArrayLike"
            Natural logarithm of the evaluation frequencies.
        log_nu_m : float
            Natural logarithm of the injection frequency.
        log_nu_c : float
            Natural logarithm of the cooling frequency.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency.
        log_F_norm : float
            Natural logarithm of the flux density normalization corresponding to
            the *optically thin* equivalent emission at the dominant electron frequency.
        p : float
            Electron power-law index.
        s : float
            Smoothing parameter for spectral breaks.


        Returns
        -------
        array-like
            Natural logarithm of the synchrotron SED evaluated at ``log_nu``.
        """
        # Determine the regime
        regime, _ = self._compute_sed_regime(
            log_nu_m,
            log_nu_c,
            log_nu_max,
        )
        # Dispatch to the appropriate regime function
        return self._log_opt_sed_from_regime(
            log_nu,
            regime,
            log_nu_m,
            log_nu_c,
            log_nu_max,
            log_F_norm,
            p,
            s,
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        *,
        nu_m: "_UnitBearingScalarLike",
        nu_c: "_UnitBearingScalarLike",
        F_norm: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf,
        p: float = 2.5,
        s: float = -1.0,
    ):
        r"""
        Evaluate the optically thin synchrotron spectral energy distribution.

        This is the primary user-facing method for computing the synchrotron
        flux density :math:`F_\nu` produced by a power-law electron population
        subject to radiative cooling, **without synchrotron self-absorption**.

        The spectral shape is determined by the ordering of the characteristic
        break frequencies and is evaluated using smoothed broken power laws
        (SBPLs).

        Parameters
        ----------
        nu : ~astropy.units.Quantity or float-like or array-like
            Frequencies at which to evaluate the SED.
        nu_m : ~astropy.units.Quantity or float-like
            Injection (minimum electron) synchrotron frequency
            :math:`\nu_m`.
        nu_c : ~astropy.units.Quantity or float-like
            Cooling break frequency :math:`\nu_c`.
        F_norm : ~astropy.units.Quantity or float-like
            Flux density normalization corresponding to the *optically thin* equivalent emission
            at the dominant electron frequency. See :ref:`sed_normalization` for details
            on how to determine the appropriate normalization.
        nu_max : ~astropy.units.Quantity or float-like, optional
            Maximum synchrotron frequency :math:`\nu_{\max}`.
            Defaults to :math:`\infty`.
        p : float, optional
            Power-law index of the injected electron energy distribution.
            Default is ``2.5``.
        s : float, optional
            SBPL smoothing parameter controlling the sharpness of spectral
            breaks. Must be negative for physical behavior. Default is ``-1.0``.

        Returns
        -------
        flux : astropy.units.Quantity
            Flux density :math:`F_\nu` evaluated at ``nu``.

        Notes
        -----
        - Units are validated and coerced internally.
        - The cooling regime is determined **globally** from the ordering of
          :math:`\nu_m`, :math:`\nu_c`, and :math:`\nu_{\max}` and applies to
          the entire spectrum.
        - The returned SED is continuous and differentiable due to SBPL
          smoothing.
        """
        nu = ensure_in_units(nu, "Hz")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_c = ensure_in_units(nu_c, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")
        F_norm = ensure_in_units(F_norm, "erg s^-1 cm^-2 Hz^-1")

        log_sed = self._log_opt_sed(
            np.log(nu),
            log_nu_m=np.log(nu_m),
            log_nu_c=np.log(nu_c),
            log_nu_max=np.log(nu_max),
            log_F_norm=np.log(F_norm),
            p=p,
            s=s,
        )
        return np.exp(log_sed) * u.erg / (u.s * u.cm**2 * u.Hz)

    # =========================================================== #
    # Closure Relations Implementation                            #
    # =========================================================== #
    # Here we implement the closure relations to go forward and backward
    # between the physics parameters and the phenomenological SED parameters.
    def _opt_from_physics_to_params(
        self,
        log_B: float,
        log_V: float,
        log_D_L: float,
        log_gamma_min: float,
        log_gamma_c: float,
        log_gamma_max: float = np.inf,
        p: float = 2.5,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        pitch_average: bool = False,
    ):
        r"""
        Compute phenomenological synchrotron SED parameters from physical inputs (log-space).

        This low-level method converts physical parameters describing a synchrotron-
        emitting plasma into the **phenomenological parameters** required to evaluate
        an optically thin, cooling synchrotron spectral energy distribution.

        All calculations are performed natively in **logarithmic CGS units** for
        numerical stability and performance. This method is not intended for direct
        user access.

        The normalization follows an **equipartition-based prescription**, in which
        fixed fractions of the post-shock internal energy are assumed to reside in
        relativistic electrons (:math:`\epsilon_E`) and magnetic fields
        (:math:`\epsilon_B`).

        Parameters
        ----------
        log_B : float
            Natural logarithm of the magnetic field strength in Gauss.
        log_V : float
            Natural logarithm of the effective emitting volume in ``cm^3``.
        log_D_L : float
            Natural logarithm of the luminosity distance in ``cm``.
        log_gamma_min : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\gamma_m`.
        log_gamma_c : float
            Natural logarithm of the cooling Lorentz factor
            :math:`\gamma_c`.
        log_gamma_max : float, optional
            Natural logarithm of the maximum electron Lorentz factor.
            Defaults to :math:`+\infty`, corresponding to no high-energy cutoff.
        p : float, optional
            Power-law index of the injected electron energy distribution.
            Default is ``2.5``.
        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic electrons.
            Default is ``0.1``.
        epsilon_B : float, optional
            Fraction of post-shock internal energy carried by magnetic fields.
            Default is ``0.1``.
        alpha : float, optional
            Electron pitch angle in radians. Ignored if ``pitch_average=True``.
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity and
            normalization.

        Returns
        -------
        params : dict
            Dictionary containing logarithmic phenomenological parameters:

            - ``F_peak`` : Natural logarithm of the peak flux density
              :math:`F_{\nu,\mathrm{pk}}`.
            - ``nu_m`` : Natural logarithm of the injection frequency
              :math:`\nu_m`.
            - ``nu_c`` : Natural logarithm of the cooling frequency
              :math:`\nu_c`.
            - ``nu_max`` : Natural logarithm of the maximum synchrotron frequency
              :math:`\nu_{\max}`.
            - ``regime`` : Enum identifying the **global cooling regime** and
              corresponding spectral kernel.

        Notes
        -----
        - The cooling regime is determined **globally** from the ordering of
          :math:`\nu_m`, :math:`\nu_c`, and :math:`\nu_{\max}`.
        - In the fast-cooling regime, normalization is performed at
          :math:`\nu_c`; in other regimes, normalization is performed at
          :math:`\nu_m`.
        - This method assumes a **single-zone, homogeneous emitting region**.
        - No synchrotron self-absorption is included.

        """
        # Handle pitch angle details and the relevant values of
        # the log_chi parameter in each case. This allows us to
        # permit both a fixed pitch angle and pitch-angle averaging.
        sin_alpha = np.sin(alpha)
        log_sin_alpha = np.log(sin_alpha) if not pitch_average else 0.0
        log_chi = _log_chi_cgs_iso if pitch_average else _log_chi_cgs

        # Compute the Doppler factors and redshift corrections for the
        # fluxes and the frequencies.
        beta_bulk = np.sqrt(1 - gamma_bulk**-2)
        delta_bulk = gamma_bulk * (1 + beta_bulk)
        log_delta_bulk = np.log(delta_bulk)
        log_nu_correction = log_delta_bulk - np.log1p(redshift)
        log_flux_correction = 3 * log_delta_bulk - np.log1p(redshift)

        # --- Compute the Frequencies --- #
        # We need to use the electron Lorentz factors to compute
        # the relevant synchrotron frequencies. This should be done for
        # gamma_c, gamma_min, and gamma_max.
        log_nu_m = (
            _opt_compute_log_synch_frequency(
                log_gamma_min,
                log_B,
                sin_alpha=sin_alpha,
                pitch_average=pitch_average,
            )
            + log_nu_correction
        )
        log_nu_max = (
            _opt_compute_log_synch_frequency(
                log_gamma_max,
                log_B,
                sin_alpha=sin_alpha,
                pitch_average=pitch_average,
            )
            + log_nu_correction
        )
        log_nu_c = (
            _opt_compute_log_synch_frequency(
                log_gamma_c,
                log_B,
                sin_alpha=sin_alpha,
                pitch_average=pitch_average,
            )
            + log_nu_correction
        )

        # --- Compute the correct peak flux based on the regime --- #
        if log_nu_c < log_nu_m:
            # This is fast cooling, we need to normalize at nu_c.
            log_electron_norm = _opt_normalize_BPL_from_magnetic_field(
                log_B,
                -2,
                -(p + 1),
                gamma_b=np.exp(log_gamma_min),
                gamma_min=np.exp(log_gamma_c),
                gamma_max=np.exp(log_gamma_max),
                epsilon_E=epsilon_E,
                epsilon_B=epsilon_B,
            )

            log_F_norm = (
                log_chi
                + log_B
                + log_sin_alpha  # Zero if pitch averaged
                + log_electron_norm
                + log_nu_m
                - log_nu_c
                + log_gamma_c
                + log_V
                - 2 * log_D_L
                + log_flux_correction
            )
        elif log_nu_m <= log_nu_c:
            # This is slow cooling, we need to normalize at nu_m.
            log_electron_norm = _opt_normalize_PL_from_magnetic_field(
                log_B,
                p,
                gamma_min=np.exp(log_gamma_min),
                gamma_max=np.exp(log_gamma_max),
                epsilon_E=epsilon_E,
                epsilon_B=epsilon_B,
            )

            log_F_norm = (
                log_chi
                + log_B
                + log_sin_alpha  # Zero if pitch averaged
                + log_electron_norm
                + (1 - p) * log_gamma_min
                + log_V
                - 2 * log_D_L
                + log_flux_correction
            )
        else:
            raise RuntimeError("Unrecognized regime in from_physics_to_params.")

        # Return the relevant parameters.
        regime, _ = self._compute_sed_regime(log_nu_m, log_nu_c, log_nu_max)

        if triceratops_logger.isEnabledFor(logging.DEBUG):
            triceratops_logger.debug(
                "PowerLaw_Cooling_SSA_SynchrotronSED: completed normalization.",
                extra={
                    "log_gamma_min": log_gamma_min,
                    "log_gamma_c": log_gamma_c,
                    "log_gamma_max": log_gamma_max,
                    "log_nu_m": log_nu_m,
                    "log_nu_c": log_nu_c,
                    "log_nu_max": log_nu_max,
                    "regime": regime.__name__,
                    "log_electron_norm": log_electron_norm,
                },
            )

        # Return the output.
        return {
            "F_norm": log_F_norm,
            "nu_m": log_nu_m,
            "nu_c": log_nu_c,
            "nu_max": log_nu_max,
            "regime": regime,
        }

    def from_physics_to_params(
        self,
        B: "_UnitBearingScalarLike",
        V: "_UnitBearingScalarLike",
        D_L: "_UnitBearingScalarLike",
        gamma_min: float,
        gamma_c: float,
        gamma_max: float = np.inf,
        p: float = 2.5,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        pitch_average: bool = False,
    ):
        r"""
        Determine phenomenological synchrotron SED parameters from physical inputs.

        This method computes the phenomenological parameters required to evaluate
        the optically thin synchrotron spectral energy distribution from a set of
        physical inputs describing the emitting plasma.

        The normalization follows an **equipartition-based prescription**, in which
        fixed fractions of the post-shock internal energy are assumed to reside in
        relativistic electrons (:math:`\epsilon_E`) and magnetic fields
        (:math:`\epsilon_B`). The resulting parameters are guaranteed to be
        internally self-consistent with the selected cooling regime.

        Users are **strongly encouraged** to consult :ref:`synch_sed_theory` for a
        detailed discussion of the assumptions and limitations of this approach.

        Parameters
        ----------
        B : ~astropy.units.Quantity or float
            Magnetic field strength. Must be convertible to Gauss.
        V : ~astropy.units.Quantity or float
            Effective emitting volume. Must be convertible to ``cm^3``.
        D_L : ~astropy.units.Quantity or float
            Luminosity distance to the source. Must be convertible to ``cm``.
        gamma_min : float
            Minimum electron Lorentz factor :math:`\gamma_m`.
        gamma_c : float
            Cooling Lorentz factor :math:`\gamma_c`.
        gamma_max : float, optional
            Maximum electron Lorentz factor. Default corresponds to no high-energy
            cutoff.
        p : float, optional
            Power-law index of the injected electron energy distribution.
            Default is ``2.5``.
        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic electrons.
            Default is ``0.1``.
        epsilon_B : float, optional
            Fraction of post-shock internal energy carried by magnetic fields.
            Default is ``0.1``.
        alpha : float, optional
            Electron pitch angle in radians. Ignored if ``pitch_average=True``.
        gamma_bulk : float, optional
            Bulk Lorentz factor of the emitting plasma. Currently ignored, but may
            be used in future implementations to account for relativistic beaming effects.
        redshift : float, optional
            Redshift of the source. Currently ignored, but may be used in future
            implementations to account for cosmological effects on the observed SED.
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity and
            normalization.

        Returns
        -------
        params : dict
            Dictionary containing the phenomenological SED parameters:

            - ``F_peak`` : Peak flux density
              (:math:`F_{\nu,\mathrm{pk}}`) with units
              ``erg cm^-2 s^-1 Hz^-1``.
            - ``nu_m`` : Injection frequency :math:`\nu_m` (Hz).
            - ``nu_c`` : Cooling frequency :math:`\nu_c` (Hz).
            - ``nu_max`` : Maximum synchrotron frequency
              :math:`\nu_{\max}` (Hz).
            - ``regime`` : Enum identifying the **global synchrotron cooling
              regime** and corresponding spectral kernel.

        Notes
        -----
        - This method assumes a **single-zone, homogeneous emitting region**.
        - Equipartition is an **assumption**, not a physical necessity.
        - The cooling regime is determined globally and applies to the entire
          spectrum.
        - Synchrotron self-absorption is **not included** in this model.
        """
        # Coerce things down to unit carrying values.
        B = ensure_in_units(B, "G")
        V = ensure_in_units(V, "cm^3")
        D_L = ensure_in_units(D_L, "cm")

        # Get everything in log-space. The low-level implementation is
        # done natively in log-space for stability.
        log_B = np.log(B)
        log_V = np.log(V)
        log_D_L = np.log(D_L)

        log_gamma_min = np.log(gamma_min)
        log_gamma_c = np.log(gamma_c)
        log_gamma_max = np.log(gamma_max)

        # Dispatch to the optimized low-level implementation.
        params_log = self._opt_from_physics_to_params(
            log_B=log_B,
            log_V=log_V,
            log_D_L=log_D_L,
            log_gamma_min=log_gamma_min,
            log_gamma_c=log_gamma_c,
            log_gamma_max=log_gamma_max,
            p=p,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            alpha=alpha,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            pitch_average=pitch_average,
        )

        # Convert back to linear CGS units and return.
        return {
            "F_norm": np.exp(params_log["F_norm"]) * u.erg / (u.cm**2 * u.s * u.Hz),
            "nu_m": np.exp(params_log["nu_m"]) * u.Hz,
            "nu_c": np.exp(params_log["nu_c"]) * u.Hz,
            "nu_max": np.exp(params_log["nu_max"]) * u.Hz,
            "regime": params_log["regime"],
        }


class PowerLaw_SSA_SynchrotronSED(MultiSpectrumSynchrotronSED):
    r"""
    Synchrotron SED with synchrotron self-absorption and no radiative cooling.

    This class implements the **canonical non-cooling synchrotron spectral energy
    distribution** produced by a power-law electron population, including
    **synchrotron self-absorption (SSA)** but **excluding radiative cooling**.

    It is applicable to homogeneous, single-zone emission regions in which the
    synchrotron cooling frequency lies well above the maximum frequency of
    interest, while synchrotron self-absorption may occur at or near the injection
    frequency. Typical examples include early-time shocks or low-density systems
    where electron cooling is negligible but SSA shapes the low-frequency spectrum.

    The spectrum is constructed using **scale-free smoothed broken power laws
    (SFBPLs)** and evaluated entirely in log-space to ensure numerical stability and
    smooth transitions between spectral segments.

    For theoretical background and derivations, see :ref:`synchrotron_theory` and
    :ref:`synch_sed_theory`.

    .. rubric:: Spectral Structure

    The synchrotron spectrum is classified into one of two **self-absorption
    regimes**, based on the ordering of the characteristic frequencies:

    - Injection frequency :math:`\nu_m`,
    - Self-absorption frequency :math:`\nu_a`,
    - Maximum synchrotron frequency :math:`\nu_{\max}`.

    The supported configurations are:

    - **Optically thin at the injection frequency**
      (:math:`\nu_a < \nu_m < \nu_{\max}`)

    - **Optically thick at the injection frequency**
      (:math:`\nu_m < \nu_a < \nu_{\max}`)

    There is no physically relevant regime with
    :math:`\nu_a > \nu_{\max}` in the absence of radiative cooling.

    The selected regime applies **globally** to the spectrum and does not depend on
    the frequency grid used for evaluation.

    .. rubric:: SED Parameters

    The parameters entering this SED fall into three conceptual categories.

    .. tab-set::

        .. tab-item:: Free parameters (phenomenological)

            These parameters define the observable structure of the SED and are
            typically inferred directly from broadband data.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Normalization flux density
                  - :math:`F_{\nu,\mathrm{norm}}`
                  - Flux density of the **dominant optically thin emitting population**.
                    In the non-cooling SSA case, this corresponds to the optically thin
                    flux at the injection frequency :math:`\nu_m`,
                    i.e. :math:`F_{\nu,\mathrm{norm}} = F_\nu(\nu_m)`.
                * - Injection frequency
                  - :math:`\nu_m`
                  - Synchrotron characteristic frequency of the minimum-energy
                    electrons.
                * - Maximum frequency
                  - :math:`\nu_{\max}`
                  - High-frequency synchrotron cutoff.

        .. tab-item:: Hyper-parameters

            These parameters control the *shape* of the spectrum but are not usually
            directly constrained by broadband observations.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Parameter
                  - Symbol
                  - Description
                * - Electron power-law index
                  - :math:`p`
                  - Index of the injected electron energy distribution,
                    :math:`N(\gamma) \propto \gamma^{-p}`.
                * - Smoothing parameter
                  - :math:`s`
                  - Controls the sharpness of the SSA and injection-frequency
                    transitions.
                * - Emission solid angle
                  - :math:`\Omega`
                  - Effective emitting area divided by distance squared.
                * - Minimum Lorentz factor
                  - :math:`\gamma_m`
                  - Minimum electron Lorentz factor used in normalization.

        .. tab-item:: Derived parameters (internal)

            These quantities are **not user inputs**, but are computed internally
            from the supplied parameters.

            .. list-table::
                :widths: 25 15 60
                :header-rows: 1

                * - Quantity
                  - Symbol
                  - Description
                * - Self-absorption frequency
                  - :math:`\nu_a`
                  - Synchrotron self-absorption break frequency.
                * - SSA regime
                  - —
                  - Discrete identifier selecting the global SSA spectral branch.

    .. rubric:: Normalization and Closure

    The SED is **always normalized using**
    :math:`F_{\nu,\mathrm{norm}}`, defined as the flux density of the dominant
    optically thin electron population.

    In the SSA-only, non-cooling case, this population emits at the injection
    frequency :math:`\nu_m`, so the normalization is anchored at
    :math:`F_\nu(\nu_m)` regardless of whether the spectrum is optically thick or
    thin at that frequency. If the spectrum is optically thick at the peak, the
    normalization is propagated self-consistently to the true spectral maximum.

    Optional closure relations are provided to map physical model parameters
    (e.g. magnetic field strength, emitting volume, electron energy fractions)
    onto the phenomenological SED parameters under assumed microphysical
    conditions such as equipartition.

    See Also
    --------
    :class:`SynchrotronSED`
        Base class for synchrotron SED implementations.
    :class:`MultiSpectrumSynchrotronSED`
        Base class for multi-regime synchrotron SEDs.
    :class:`PowerLaw_SynchrotronSED`
        Canonical optically thin synchrotron SED without cooling or SSA.
    :class:`PowerLaw_Cooling_SynchrotronSED`
        Synchrotron SED with radiative cooling and no SSA.
    :class:`PowerLaw_Cooling_SSA_SynchrotronSED`
        Synchrotron SED including both cooling and self-absorption.

    Examples
    --------
    Evaluate a synchrotron SED with self-absorption and no cooling:

    .. code-block:: python

        import numpy as np
        import astropy.units as u
        from triceratops.radiation.synchrotron import (
            PowerLaw_SSA_SynchrotronSED,
        )

        sed = PowerLaw_SSA_SynchrotronSED()

        nu = np.logspace(8, 14, 500) * u.Hz

        F_nu = sed.sed(
            nu=nu,
            nu_m=1e11 * u.Hz,
            F_norm=1e-26 * u.erg / (u.cm**2 * u.s * u.Hz),
            p=2.5,
            s=-0.05,
        )
    """

    SPECTRUM_FUNCTIONS = _SynchrotronSSASEDFunctions

    # ============================================================ #
    # Regime determination                                        #
    # ============================================================ #
    def _compute_sed_regime(
        self,
        log_nu_m: float,
        log_nu_a: float,
        **_,
    ):
        r"""
        Determine the global SSA spectral regime for a non-cooling synchrotron spectrum.

        This low-level method classifies the synchrotron spectrum into one of two
        **self-absorption regimes**, assuming:

        - No radiative cooling,
        - A single-zone, homogeneous emitting region,
        - A power-law electron energy distribution.

        The classification depends solely on the ordering of the characteristic
        frequencies:

        - Injection frequency :math:`\nu_m`,
        - Synchrotron self-absorption frequency :math:`\nu_a`.

        The supported regimes are:

        - :math:`\nu_a < \nu_m` — optically thin at the injection frequency,
        - :math:`\nu_a > \nu_m` — optically thick at the injection frequency.

        Parameters
        ----------
        log_nu_m : float
            Natural logarithm of the injection frequency :math:`\nu_m`.
        log_nu_a : float
            Natural logarithm of the synchrotron self-absorption frequency
            :math:`\nu_a`.
        **_ :
            Additional unused keyword arguments, accepted for API compatibility with
            multi-regime SED classes.

        Returns
        -------
        regime : enum-like
            Member of :attr:`SPECTRUM_FUNCTIONS` identifying the global SSA spectral
            regime.
        metadata : dict
            Empty dictionary (present for API consistency).

        Notes
        -----
        - This method performs **no unit validation**.
        - Radiative cooling, stratified SSA, and high-energy cutoffs are not considered.
        - The returned regime applies **globally** to the SED.
        """
        if log_nu_a < log_nu_m:
            return self.SPECTRUM_FUNCTIONS.SPECTRUM_1, {}
        else:
            return self.SPECTRUM_FUNCTIONS.SPECTRUM_2, {}

    def determine_sed_regime(
        self,
        nu_m: "_UnitBearingScalarLike",
        F_norm: "_UnitBearingScalarLike",
        omega: float = 4 * np.pi,
        gamma_m: float = 1.0,
        p: float = 3.0,
    ):
        r"""
        Determine the global SSA synchrotron spectral regime from phenomenological inputs.

        This user-facing method determines whether a non-cooling synchrotron spectrum is
        **optically thin or optically thick at the injection frequency**. The
        synchrotron self-absorption frequency :math:`\nu_a` is computed internally from
        the supplied flux normalization and physical parameters.

        The classification is based on the ordering of:

        - the injection frequency :math:`\nu_m`, and
        - the inferred self-absorption frequency :math:`\nu_a`.

        Parameters
        ----------
        nu_m : ~astropy.units.Quantity or float
            Injection (minimum-electron) synchrotron frequency :math:`\nu_m`.
        F_norm : ~astropy.units.Quantity or float
            Flux density normalization anchored at :math:`\nu_m`,
            i.e. :math:`F_\nu(\nu_m)`.
        omega : float, optional
            Effective emission solid angle :math:`\Omega = A / D^2`.
            Default is :math:`4\pi`.
        gamma_m : float, optional
            Minimum electron Lorentz factor :math:`\gamma_m`.
            Default is ``1.0``.
        p : float, optional
            Power-law index of the injected electron energy distribution.

        Returns
        -------
        regime : enum-like
            Member of :attr:`SPECTRUM_FUNCTIONS` identifying the global synchrotron
            self-absorption regime.

        Notes
        -----
        - This method assumes a **non-cooling**, single-zone synchrotron source.
        - The SSA frequency is inferred analytically from the supplied normalization.
        - The returned regime applies **globally** to the SED.
        - Units are validated and coerced internally.
        """
        nu_m = ensure_in_units(nu_m, "Hz")
        F_norm = ensure_in_units(F_norm, "erg s^-1 cm^-2 Hz^-1")

        log_nu_m = np.log(nu_m)
        log_F_norm = np.log(F_norm)
        log_omega = np.log(omega)
        log_gamma_m = np.log(gamma_m)

        log_nu_a = self._compute_ssa_frequency_from_F_norm(
            log_F_norm,
            log_nu_m,
            log_omega,
            log_gamma_m,
            p,
        )

        regime, _ = self._compute_sed_regime(
            log_nu_m=log_nu_m,
            log_nu_a=log_nu_a,
        )
        return regime

    # ============================================================ #
    # SSA frequency computation                                   #
    # ============================================================ #
    def _compute_ssa_frequency_from_F_norm(
        self,
        log_F_norm: float,
        log_nu_m: float,
        log_omega: float,
        log_gamma_m: float,
        p: float,
    ) -> float:
        r"""
        Compute the synchrotron self-absorption frequency from an anchored optically thin normalization.

        This method infers the synchrotron self-absorption frequency :math:`\nu_a` for a
        non-cooling synchrotron spectrum that is **normalized at the injection
        frequency** :math:`\nu_m`, when the ordering of :math:`\nu_a` relative to
        :math:`\nu_m` is not known a priori.

        Two candidate solutions are evaluated:

        1. **Optically thin at the injection frequency** (:math:`\nu_a < \nu_m`)

           .. math::

               \nu_a \propto
               \left(
                   \frac{F_\nu(\nu_m)}
                        {m_e c^2 \, \Omega \, \gamma_m}
               \right)^{6/13}
               \nu_m^{1/13}

        2. **Optically thick at the injection frequency** (:math:`\nu_a > \nu_m`)

           .. math::

               \nu_a \propto
               \left(
                   \frac{F_\nu(\nu_m)}
                        {m_e c^2 \, \Omega \, \gamma_m}
               \right)^{2/(p+4)}
               \nu_m^{p/(p+4)}

        The physically valid solution is selected by enforcing **self-consistent
        frequency ordering**.

        All calculations are performed in **natural logarithmic CGS units**.

        Parameters
        ----------
        log_F_norm : float
            Natural logarithm of the flux normalization anchored at
            :math:`\nu_m`, i.e. :math:`\log F_\nu(\nu_m)`.
        log_nu_m : float
            Natural logarithm of the injection frequency :math:`\nu_m`.
        log_omega : float
            Natural logarithm of the effective emission solid angle
            :math:`\Omega = A / D^2`.
        log_gamma_m : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\gamma_m`.
        p : float
            Power-law index of the injected electron energy distribution.

        Returns
        -------
        log_nu_a : float
            Natural logarithm of the synchrotron self-absorption frequency
            :math:`\nu_a`.

        Raises
        ------
        RuntimeError
            If neither candidate solution satisfies the required frequency ordering.

        Notes
        -----
        - This method applies only to **non-cooling synchrotron spectra**.
        - The returned solution is guaranteed to be **self-consistent** with the
          assumed spectral ordering.
        - No unit validation is performed.
        """
        log_Q = log_F_norm - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m

        # Candidate: optically thin at the peak (nu_a < nu_m)
        log_nu_a_thin = (6 * log_Q / 13) + (log_nu_m / 13)
        log_nu_a_thick = (2 * log_Q / (p + 4)) + (p * log_nu_m / (p + 4))

        if log_nu_a_thin < log_nu_m:
            return log_nu_a_thin
        elif log_nu_a_thick > log_nu_m:
            return log_nu_a_thick
        else:
            raise RuntimeError("Could not determine a self-consistent SSA frequency from the supplied normalization.")

    # ============================================================ #
    # Regime-specific kernel                                      #
    # ============================================================ #
    def _log_opt_sed_from_regime(
        self,
        log_nu,
        regime,
        log_nu_m,
        log_nu_a,
        log_nu_max,
        log_F_norm,
        p,
        s,
    ):
        r"""
        Evaluate the log-space synchrotron SED for a fixed SSA spectral regime.

        This method dispatches to the appropriate **regime-specific numerical
        kernel** for synchrotron emission with self-absorption and applies the
        overall flux normalization.

        All spectral-shape logic is contained in the regime-specific kernel
        functions; this method performs no regime determination.

        Parameters
        ----------
        log_nu : array-like
            Natural logarithm of the frequencies at which to evaluate the SED.
        regime : enum
            SSA spectral regime identifier returned by
            :meth:`_compute_sed_regime`.
        log_nu_m : float
            Natural logarithm of the injection frequency
            :math:`\nu_m`.
        log_nu_a : float
            Natural logarithm of the self-absorption frequency
            :math:`\nu_a`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`.
        log_F_norm : float
            Natural logarithm of the peak flux density
            :math:`F_{\nu,\mathrm{pk}}`.
        p : float
            Electron energy power-law index.
        s : float
            Smoothness parameter for the smoothed broken power-law (SBPL)
            transitions.

        Returns
        -------
        log_sed : array-like
            Natural logarithm of the flux density
            :math:`\log F_\nu` evaluated at ``log_nu``.

        Raises
        ------
        RuntimeError
            If an unrecognized SSA regime identifier is supplied.

        Notes
        -----
        - This method assumes a **non-cooling, SSA-only** synchrotron spectrum.
        - The optically thick branch includes an explicit normalization shift to ensure
          continuity with the anchored normalization at :math:`\nu_m`.
        - No unit validation is performed.
        """
        if regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_1:
            log_sed = _log_powerlaw_sbpl_sed_ssa_1(log_nu, log_nu_m, log_nu_a, log_nu_max, p, s) + log_F_norm
        elif regime == self.SPECTRUM_FUNCTIONS.SPECTRUM_2:
            log_sed = (
                _log_powerlaw_sbpl_sed_ssa_2(log_nu, log_nu_m, log_nu_a, log_nu_max, p, s)
                + log_F_norm
                - ((p - 1) / 2) * (log_nu_a - log_nu_m)
            )
        else:
            raise RuntimeError("Unrecognized SSA regime.")

        return log_sed

    def _log_opt_sed(
        self,
        log_nu: "_ArrayLike",
        log_nu_m: float,
        log_nu_max: float,
        log_F_norm: float,
        log_omega: float,
        log_gamma_m: float,
        p: float = 3.0,
        s: float = -1.0,
    ):
        r"""
        Evaluate the log-space synchrotron SED with self-absorption and no cooling.

        This optimized internal method orchestrates the full SED evaluation by:

        1. Computing the synchrotron self-absorption frequency :math:`\nu_a`,
        2. Determining the global SSA spectral regime,
        3. Dispatching to the appropriate regime-specific SED kernel.

        All calculations are performed in **natural logarithmic CGS units**.

        Parameters
        ----------
        log_nu : array-like
            Natural logarithm of the frequencies at which to evaluate the SED.
        log_nu_m : float
            Natural logarithm of the injection frequency
            :math:`\nu_m`.
        log_nu_max : float
            Natural logarithm of the maximum synchrotron frequency
            :math:`\nu_{\max}`.
        log_F_norm : float
            Natural logarithm of the peak flux density
            :math:`F_{\nu,\mathrm{pk}}`.
        log_omega : float
            Natural logarithm of the effective emission solid angle
            :math:`\Omega = A / D^2`.
        log_gamma_m : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\gamma_m`.
        p : float, optional
            Electron energy power-law index.
            Default is ``3.0``.
        s : float, optional
            SBPL smoothness parameter.
            Must be negative for physical behavior.

        Returns
        -------
        log_sed : array-like
            Natural logarithm of the synchrotron flux density
            :math:`\log F_\nu` evaluated at ``log_nu``.

        Notes
        -----
        - This method assumes a **non-cooling**, single-zone synchrotron source.
        - Synchrotron self-absorption is included using analytic scalings.
        - No unit validation is performed.
        """
        # Compute the self-absorption frequency
        log_nu_a = self._compute_ssa_frequency_from_F_norm(
            log_F_norm,
            log_nu_m,
            log_omega,
            log_gamma_m,
            p,
        )

        # Determine the global SSA regime
        regime, _ = self._compute_sed_regime(
            log_nu_m=log_nu_m,
            log_nu_a=log_nu_a,
        )

        # Dispatch to the regime-specific SED kernel
        return self._log_opt_sed_from_regime(
            log_nu,
            regime,
            log_nu_m,
            log_nu_a,
            log_nu_max,
            log_F_norm,
            p,
            s,
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        *,
        nu_m: "_UnitBearingScalarLike",
        F_norm: "_UnitBearingScalarLike",
        nu_max: "_UnitBearingScalarLike" = np.inf,
        omega: float = 4 * np.pi,
        gamma_m: float = 1.0,
        p: float = 2.5,
        s: float = -1.0,
    ):
        r"""
        Evaluate the synchrotron spectral energy distribution with self-absorption and no radiative cooling.

        This method computes the flux density :math:`F_\nu` for a power-law
        electron population emitting synchrotron radiation in a homogeneous,
        single-zone region, including synchrotron self-absorption but **excluding
        radiative cooling**.

        Parameters
        ----------
        nu : ~astropy.units.Quantity or float-like or array-like
            Frequencies at which to evaluate the SED.
        nu_m : ~astropy.units.Quantity or float-like
            Injection (minimum-electron) synchrotron frequency
            :math:`\nu_m`.
        F_norm : ~astropy.units.Quantity or float-like
            The flux density normalization anchored at :math:`\nu_m`, i.e. :math:`F_\nu(\nu_m)`.
            See :ref:`sed_normalization` for details on how to compute this from physical parameters.
        nu_max : ~astropy.units.Quantity or float-like, optional
            Maximum synchrotron frequency
            :math:`\nu_{\max}`.
            Defaults to :math:`\infty`.
        omega : float, optional
            Effective emission solid angle
            :math:`\Omega = A / D^2`.
            Default is :math:`4\pi`.
        gamma_m : float, optional
            Minimum electron Lorentz factor
            :math:`\gamma_m`.
            Default is ``1.0``.
        p : float, optional
            Electron energy power-law index.
            Default is ``2.5``.
        s : float, optional
            SBPL smoothness parameter.
            Must be negative for physical behavior.

        Returns
        -------
        flux : astropy.units.Quantity
            Flux density :math:`F_\nu` evaluated at ``nu``.

        Notes
        -----
        - Units are validated and coerced internally.
        - The SSA regime is determined automatically and applies globally.
        - This model assumes a **non-cooling** synchrotron source.
        - The returned spectrum is continuous and differentiable due to SBPL
          smoothing.
        """
        nu = ensure_in_units(nu, "Hz")
        nu_m = ensure_in_units(nu_m, "Hz")
        nu_max = ensure_in_units(nu_max, "Hz")
        log_F_norm = ensure_in_units(F_norm, "erg s^-1 cm^-2 Hz^-1")

        log_nu = np.log(nu)
        log_nu_m = np.log(nu_m)
        log_nu_max = np.log(nu_max)
        log_F_norm = np.log(log_F_norm)
        log_omega = np.log(omega)
        log_gamma_m = np.log(gamma_m)

        log_sed = self._log_opt_sed(
            log_nu,
            log_nu_m=log_nu_m,
            log_nu_max=log_nu_max,
            log_F_norm=log_F_norm,
            log_omega=log_omega,
            log_gamma_m=log_gamma_m,
            p=p,
            s=s,
        )

        return np.exp(log_sed) * u.erg / (u.s * u.cm**2 * u.Hz)

    # ============================================================ #
    # Normalization closure relations                              #
    # ============================================================ #
    def _opt_from_physics_to_params(
        self,
        log_B: float,
        log_V: float,
        log_D_L: float,
        log_Omega: float,
        log_gamma_min: float,
        log_gamma_max: float = np.inf,
        p: float = 2.5,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        pitch_average: bool = False,
    ):
        r"""
        Compute phenomenological SSA synchrotron parameters from physical inputs.

        This low-level method converts physical parameters describing a homogeneous,
        single-zone synchrotron source into a **self-consistent set of phenomenological
        SED parameters** for a power-law electron population with synchrotron
        self-absorption and **no radiative cooling**.

        All calculations are performed in **natural logarithmic CGS units** for
        numerical stability. This method is intended for internal use only.

        The electron population is assumed to follow

        .. math::

            \frac{dN}{d\gamma} \propto \gamma^{-p},
            \qquad \gamma \ge \gamma_m,

        and is normalized using an equipartition prescription with energy fractions
        :math:`\epsilon_E` and :math:`\epsilon_B`.

        Parameters
        ----------
        log_B : float
            Natural logarithm of the magnetic field strength (Gauss).
        log_V : float
            Natural logarithm of the effective emitting volume (cm³).
        log_D_L : float
            Natural logarithm of the luminosity distance (cm).
        log_Omega : float
            Natural logarithm of the effective emission solid angle
            :math:`\Omega = A / D^2`.
        log_gamma_min : float
            Natural logarithm of the minimum electron Lorentz factor
            :math:`\gamma_m`.
        log_gamma_max : float, optional
            Natural logarithm of the maximum electron Lorentz factor.
        p : float, optional
            Power-law index of the electron energy distribution.
        epsilon_E : float, optional
            Fraction of post-shock internal energy carried by relativistic electrons.
        epsilon_B : float, optional
            Fraction of post-shock internal energy carried by magnetic fields.
        alpha : float, optional
            Electron pitch angle in radians (ignored if ``pitch_average=True``).
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.

        Returns
        -------
        params : dict
            Dictionary containing logarithmic phenomenological parameters:

            - ``F_peak`` : :math:`\log F_{\nu,\mathrm{pk}}`
            - ``nu_m`` : :math:`\log \nu_m`
            - ``nu_a`` : :math:`\log \nu_a`
            - ``nu_max`` : :math:`\log \nu_{\max}`
            - ``regime`` : Enum identifying the global SSA spectral regime

        Notes
        -----
        - Radiative cooling is **not included**.
        - The SSA frequency is computed self-consistently from the normalization.
        - The returned regime applies **globally** to the spectrum.
        """
        # Handle pitch angle details and the relevant values of
        # the log_chi parameter in each case. This allows us to
        # permit both a fixed pitch angle and pitch-angle averaging.
        sin_alpha = np.sin(alpha)
        log_sin_alpha = np.log(sin_alpha) if not pitch_average else 0.0
        log_chi = _log_chi_cgs_iso if pitch_average else _log_chi_cgs

        # Compute the Doppler factors and redshift corrections for the
        # fluxes and the frequencies.
        beta_bulk = np.sqrt(1 - gamma_bulk**-2)
        delta_bulk = gamma_bulk * (1 + beta_bulk)
        log_delta_bulk = np.log(delta_bulk)
        log_nu_correction = log_delta_bulk - np.log1p(redshift)
        log_flux_correction = 3 * log_delta_bulk - np.log1p(redshift)

        # --- Compute the Frequencies --- #
        # We need to use the electron Lorentz factors to compute
        # the relevant synchrotron frequencies. This should be done for
        # gamma_c, gamma_min, and gamma_max.
        log_nu_m = (
            _opt_compute_log_synch_frequency(
                log_gamma_min,
                log_B,
                sin_alpha=sin_alpha,
                pitch_average=pitch_average,
            )
            + log_nu_correction
        )
        log_nu_max = (
            _opt_compute_log_synch_frequency(
                log_gamma_max,
                log_B,
                sin_alpha=sin_alpha,
                pitch_average=pitch_average,
            )
            + log_nu_correction
        )

        # Compute the normalization flux at nu_m.
        log_electron_norm = _opt_normalize_PL_from_magnetic_field(
            log_B,
            p,
            gamma_min=np.exp(log_gamma_min),
            gamma_max=np.exp(log_gamma_max),
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
        )

        log_F_norm = (
            log_chi
            + log_B
            + log_sin_alpha  # Zero if pitch averaged
            + log_electron_norm
            + (1 - p) * log_gamma_min
            + log_V
            - 2 * log_D_L
            + log_flux_correction
        )

        # Compute nu_ssa
        log_nu_a = self._compute_ssa_frequency_from_F_norm(
            log_F_norm,
            log_nu_m,
            log_Omega,
            log_gamma_min,
            p,
        )
        regime, _ = self._compute_sed_regime(
            log_nu_m,
            log_nu_a,
        )

        # --- LOG AND RETURN --- #
        if triceratops_logger.isEnabledFor(logging.DEBUG):
            triceratops_logger.debug(
                "PowerLaw_Cooling_SSA_SynchrotronSED: completed normalization.",
                extra={
                    "log_gamma_min": log_gamma_min,
                    "log_gamma_max": log_gamma_max,
                    "log_nu_m": log_nu_m,
                    "log_nu_max": log_nu_max,
                    "log_nu_a": log_nu_a,
                    "regime": regime.__name__,
                    "log_electron_norm": log_electron_norm,
                },
            )
        return {
            "F_norm": log_F_norm,
            "nu_m": log_nu_m,
            "nu_a": log_nu_a,
            "nu_max": log_nu_max,
            "regime": regime,
        }

    def from_physics_to_params(
        self,
        B: "_UnitBearingScalarLike",
        V: "_UnitBearingScalarLike",
        D_L: "_UnitBearingScalarLike",
        Omega: float,
        gamma_min: float,
        gamma_max: float = np.inf,
        p: float = 2.5,
        epsilon_E: float = 0.1,
        epsilon_B: float = 0.1,
        alpha: float = 1.0,
        gamma_bulk: float = 1.0,
        redshift: float = 0.0,
        pitch_average: bool = False,
    ):
        r"""
        Determine SSA synchrotron SED parameters from physical inputs.

        This method computes the phenomenological parameters required to evaluate
        a **synchrotron spectral energy distribution with self-absorption and no
        radiative cooling** from a set of physical inputs describing the emitting
        region.

        The normalization follows an equipartition-based prescription, assuming
        fixed fractions of the post-shock internal energy reside in relativistic
        electrons (:math:`\epsilon_E`) and magnetic fields (:math:`\epsilon_B`).

        Parameters
        ----------
        B : ~astropy.units.Quantity or float
            Magnetic field strength. Must be convertible to Gauss.
        V : ~astropy.units.Quantity or float
            Effective emitting volume. Must be convertible to ``cm^3``.
        D_L : ~astropy.units.Quantity or float
            Luminosity distance to the source. Must be convertible to ``cm``.
        Omega : float
            Effective emission solid angle :math:`\Omega = A / D^2`.
        gamma_min : float
            Minimum electron Lorentz factor :math:`\gamma_m`.
        gamma_max : float, optional
            Maximum electron Lorentz factor.
        p : float, optional
            Power-law index of the injected electron energy distribution.
        epsilon_E : float, optional
            Fraction of post-shock internal energy in relativistic electrons.
        epsilon_B : float, optional
            Fraction of post-shock internal energy in magnetic fields.
        gamma_bulk: float, optional
            Bulk Lorentz factor of the emitting region.
        redshift: float, optional
            Cosmological redshift of the source.
        alpha : float, optional
            Electron pitch angle in radians.
        pitch_average : bool, optional
            If ``True``, use pitch-angle averaged synchrotron emissivity.

        Returns
        -------
        params : dict
            Dictionary containing the phenomenological SED parameters:

            - ``F_peak`` : Peak flux density (:math:`F_{\nu,\mathrm{pk}}`)
            - ``nu_m`` : Injection frequency :math:`\nu_m`
            - ``nu_a`` : Self-absorption frequency :math:`\nu_a`
            - ``nu_max`` : Maximum synchrotron frequency :math:`\nu_{\max}`
            - ``regime`` : Enum identifying the SSA spectral regime

        Notes
        -----
        - This method assumes a **single-zone, homogeneous emitting region**.
        - Radiative cooling is **not included**.
        - The returned parameters are guaranteed to be internally self-consistent.

        """
        # Coerce things down to unit carrying values.
        B = ensure_in_units(B, "G")
        V = ensure_in_units(V, "cm^3")
        D_L = ensure_in_units(D_L, "cm")

        # Get everything in log-space. The low-level implementation is
        # done natively in log-space for stability.
        log_B = np.log(B)
        log_V = np.log(V)
        log_D_L = np.log(D_L)
        log_Omega = np.log(Omega)

        log_gamma_min = np.log(gamma_min)
        log_gamma_max = np.log(gamma_max)

        # Dispatch to the optimized low-level implementation.
        params_log = self._opt_from_physics_to_params(
            log_B=log_B,
            log_V=log_V,
            log_D_L=log_D_L,
            log_Omega=log_Omega,
            log_gamma_min=log_gamma_min,
            log_gamma_max=log_gamma_max,
            p=p,
            epsilon_E=epsilon_E,
            epsilon_B=epsilon_B,
            alpha=alpha,
            gamma_bulk=gamma_bulk,
            redshift=redshift,
            pitch_average=pitch_average,
        )

        # Convert back to linear CGS units and return.
        return {
            "F_norm": np.exp(params_log["F_norm"]) * u.erg / (u.cm**2 * u.s * u.Hz),
            "nu_m": np.exp(params_log["nu_m"]) * u.Hz,
            "nu_a": np.exp(params_log["nu_a"]) * u.Hz,
            "nu_max": np.exp(params_log["nu_max"]) * u.Hz,
            "regime": params_log["regime"],
        }


class SSA_SED_PowerLaw(SynchrotronSED):
    r"""
    Synchrotron self-absorbed (SSA) broken power-law spectral energy distribution.

    This class implements a phenomenological synchrotron spectral energy
    distribution characterized by a single smooth spectral break arising from
    synchrotron self-absorption (SSA). Below the break frequency, the spectrum is
    optically thick and follows a power-law slope of :math:`+5/2`; above the
    break, the spectrum is optically thin and follows a power-law slope of
    :math:`-(p-1)/2`, where :math:`p` is the power-law index of the electron
    energy distribution.

    The SED is implemented as a **smoothly broken power law**, ensuring numerical
    stability and differentiability across the SSA turnover. This makes the
    class well-suited for use in parameter inference, optimization, and
    likelihood-based modeling.

    In addition to forward evaluation of the SED, this class implements
    **analytic closure relations** that allow inversion between phenomenological
    SED parameters (the SSA break frequency and peak flux) and underlying
    physical properties of the emitting region (magnetic field strength and
    radius). These closure relations follow the formalism presented in
    :footcite:t:`demarchiRadioAnalysisSN2004C2022`.

    This class is intended for modeling radio synchrotron emission from
    approximately homogeneous emitting regions, such as those encountered in:

    - supernova radio afterglows,
    - compact object-powered transients,
    - synchrotron-emitting shells with a single dominant SSA turnover.

    No assumptions are made about time evolution, shock dynamics, or cooling
    regimes beyond those implicit in the broken power-law description.

    Notes
    -----
    - The break frequency :math:`\nu_{\rm brk}` is defined as the frequency at
      which the optically thick and optically thin asymptotic power laws
      intersect.
    - The closure relations implemented here assume a power-law electron energy
      distribution with finite bounds
      :math:`\gamma_{\rm min} \le \gamma \le \gamma_{\rm max}`.
    - The special case :math:`p = 2` is not supported due to logarithmic
      divergences in the electron energy integral and must be handled
      separately.

    See Also
    --------
    SynchrotronSED
        Abstract base class defining the interface for all synchrotron SED
        implementations.

    References
    ----------
    .. footbibliography::
    """

    # ================================================= #
    # Instantiation                                     #
    # ================================================= #
    def __init__(self):
        r"""Instantiate the SSA SED object."""
        # There are no class-wide constants to pre-compute for this SED.
        super().__init__()

    # ================================================ #
    # SED Function Implementation                      #
    # ================================================ #
    def _log_opt_sed(
        self,
        nu: "_ArrayLike",
        nu_brk: float,
        F_nu_brk: float,
        p: float,
        s: float,
    ) -> "_ArrayLike":
        r"""
        Low-level optimized SSA broken power-law SED.

        This method implements the synchrotron self-absorbed (SSA) broken power-law SED
        in a performance-optimized manner. It assumes that all inputs are provided as
        dimensionless scalars or NumPy arrays in CGS units. No unit validation or safety
        checks are performed.

        Parameters
        ----------
        nu: float or array-like
            Frequency at which to evaluate the SED (in Hz-equivalent CGS).
        nu_brk: float
            Break frequency (Hz-equivalent CGS).
        F_nu_brk: float
            Flux density at the break frequency (CGS units).
        p: float
            Power-law index of the electron energy distribution.
        s: float
            Smoothing parameter for the break.

        Returns
        -------
        float or array-like
            The computed SED value at the specified frequency. The output of this SED
            is in CGS units of erg s^-1 cm^-2 Hz^-1.
        """
        return smoothed_BPL(
            nu,
            F_nu_brk,
            nu_brk,
            -(p - 1) / 2,  # optically thin index
            (5 / 2),  # optically thick index
            s,  # smoothing parameter
        )

    def sed(
        self,
        nu: "_UnitBearingArrayLike",
        *,
        nu_brk: "_UnitBearingArrayLike",
        F_nu_brk: "_UnitBearingArrayLike",
        p: float,
        s: float,
    ):
        r"""
        Compute the synchrotron self-absorbed (SSA) broken power-law SED.

        This is the user-facing interface for the SSA broken power-law spectral
        energy distribution. It validates units, coerces inputs to CGS, and
        dispatches to the optimized low-level backend implementation
        :meth:`_opt_sed`.

        Parameters
        ----------
        nu : float, array-like, or astropy.units.Quantity
            Frequency at which to evaluate the SED. Default units are Hz.
        nu_brk : float or astropy.units.Quantity
            Break frequency of the SED. Default units are Hz.
        F_nu_brk : float or astropy.units.Quantity
            Flux density at the break frequency. Default units are
            erg s^-1 cm^-2 Hz^-1.
        p : float
            Power-law index of the electron energy distribution.
        s : float
            Smoothing parameter controlling the sharpness of the spectral break.

        Returns
        -------
        astropy.units.Quantity
            Flux density evaluated at ``nu`` with units of
            erg s^-1 cm^-2 Hz^-1.
        """
        # --- Unit validation and coercion --- #
        nu = ensure_in_units(nu, u.Hz)
        nu_brk = ensure_in_units(nu_brk, u.Hz)
        F_nu_brk = ensure_in_units(F_nu_brk, u.erg / u.s / u.cm**2 / u.Hz)

        # --- Call optimized backend --- #
        F_nu_cgs = self._log_opt_sed(
            nu=nu,
            nu_brk=nu_brk,
            F_nu_brk=F_nu_brk,
            p=p,
            s=s,
        )

        return F_nu_cgs * u.erg / u.s / u.cm**2 / u.Hz

    # ================================================ #
    # Closure Relations.                               #
    # ================================================ #
    # For this SED, we implement the closure relations to go from
    # the phenomenological parameters (nu_brk, F_nu_brk) to the physical
    # parameters (B, R) and vice versa. This is implemented exactly as
    # done in DeMarchi+22.
    def _opt_from_params_to_physics(
        self,
        nu_brk: Union[float, np.ndarray],
        F_nu_brk: Union[float, np.ndarray],
        distance: Union[float, np.ndarray],
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        r"""
        Compute the magnetic field and radius of the emitting region from a broken power-law synchrotron SED.

        Parameters
        ----------
        nu_brk: float or array-like
            The break frequency :math:`\nu_{\rm brk}` where the synchrotron SED transitions from
            optically thick to optically thin. This should be provided in GHz. May be provided either
            as a scalar float (for a single SED) or as a 1-D array (for multiple SEDs).
        F_nu_brk: float or array-like
            The flux density :math:`F_{\nu_{\rm brk}}` at the break frequency. This should be provided
            in Jansky (Jy). May be provided either as a scalar float (for a single SED) or as a 1-D array (for multiple
            SEDs).
        distance: float or array-like
            The distance to the source in Megaparsecs (Mpc). May be provided either as a scalar float (for a single SED)
            or as a 1-D array (for multiple SEDs).
        p: float or array-like
            The power-law index :math:`p` of the electron energy distribution. By default, this is 3.0. If provided as a
            float, the value is used for all SEDs. If provided as an array, its shape must be compatible with that of
            ``nu_brk``.

            .. warning::

                This function does not handle the case where :math:`p = 2` due to singularities in the underlying
                equations.
                Users must ensure that :math:`p` is not equal to 2 when calling this function.

        f: float or array-like
            The filling factor :math:`f` of the emitting region. Default is ``0.5``. If provided as a float, the value
            is
            used for all SEDs. If provided as an array, its shape must be compatible with that of ``nu_brk``.
        theta: float or array-like
            The pitch angle :math:`\theta` between the magnetic field and the line of sight, in radians.
            Default is ``pi/2`` (i.e., perpendicular). If provided as a float, the value is used for all SEDs.
            If provided as an array, its shape must be compatible with that of ``nu_brk``.
        epsilon_B: float or array-like
            The fraction of post-shock energy in magnetic fields, :math:`\epsilon_B`.
        epsilon_E: float or array-like
            The fraction of post-shock energy in relativistic electrons, :math:`\epsilon_E`.
        gamma_min: float or array-like
            The minimum Lorentz factor :math:`\gamma_{\rm min}` of the electron energy.
        gamma_max: float or array-like
            The maximum Lorentz factor :math:`\gamma_{\rm max}` of the electron energy.

        Returns
        -------
        B: float or array-like
            The computed magnetic field strength :math:`B` in Gauss. The shape matches that of the input parameters.
        R: float or array-like
            The computed radius of the emitting region :math:`R` in cm. The shape matches that of the input parameters.

        Notes
        -----
        See notes in the user-facing wrapper function `compute_BR_from_BPL` for details on the
        underlying physics and assumptions.

        Effective 1/19/26: We have modified this to work exclusively in log space due to issues with floating point
        truncation errors. Catastrophic cancellation caused significant loss of accuracy.
        """
        return _compute_ssa_BR_from_spectrum_dm22(
            nu_brk=nu_brk,
            F_nu_brk=F_nu_brk,
            distance=distance,
            p=p,
            f=f,
            theta=theta,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )

    def from_params_to_physics(
        self,
        nu_brk: Union[float, np.ndarray, u.Quantity],
        F_nu_brk: Union[float, np.ndarray, u.Quantity],
        distance: Union[float, np.ndarray, u.Quantity],
        *,
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Compute the magnetic field strength and emitting radius from a broken power-law SED.

        This function provides a **user-facing, unit-aware interface** for computing
        the physical properties of a synchrotron-emitting region whose radio spectrum
        exhibits a turnover due to synchrotron self-absorption (SSA). Internally, it
        wraps a low-level optimized routine that implements the analytic inversion
        described in :footcite:t:`demarchiRadioAnalysisSN2004C2022` (DM22).

        The function accepts scalar values, NumPy arrays, or Astropy ``Quantity`` objects
        and performs the necessary unit coercion, validation, and shape checking before
        dispatching to the optimized backend.

        Parameters
        ----------
        nu_brk : float, array-like, or astropy.units.Quantity
            The SSA break (turnover) frequency :math:`\nu_{\rm brk}` separating the
            optically thick and optically thin synchrotron regimes. Default units are GHz, but may
            be overridden by providing ``nu_brk`` as a :class:`astropy.units.Quantity` object. May be
            provided as either a scalar (for a single SED) or a 1-D array (for multiple SEDs).

        F_nu_brk : float, array-like, or astropy.units.Quantity
            The flux density :math:`F_{\nu_{\rm brk}}` at the break frequency. Default units
            are Jansky (Jy), but may be overridden by providing ``F_nu_brk`` as a
            :class:`astropy.units.Quantity` object. May be provided as either a scalar
            (for a single SED) or a 1-D array (for multiple SEDs). If provided as an array, shape
            must be compatible with that of ``nu_brk``.

        distance : float, array-like, or astropy.units.Quantity
            The (luminosity) distance to the source. By default, units are Megaparsecs (Mpc),
            but may be overridden by providing ``distance`` as a :class:`astropy.units.Quantity` object.
            May be provided as either a scalar (for a single SED) or a 1-D array (for multiple SEDs). If provided
            as an array, shape must be compatible with that of ``nu_brk``.

        p : float or array-like, optional
            The power-law index :math:`p` of the electron Lorentz factor distribution,
            defined by

            .. math::

                N(\Gamma)\, d\Gamma = K_e \Gamma^{-p}\, d\Gamma.

            Default is ``3.0``.

            .. warning::

                This function **does not support** the case :math:`p = 2`, for which
                the electron energy integral is logarithmically divergent and requires
                a separate analytic treatment. If any value of ``p`` is exactly equal
                to 2, a ``ValueError`` is raised.

        f : float or array-like, optional
            Volume filling factor of the synchrotron-emitting region. Default is ``0.5``.

        theta : float or array-like, optional
            Pitch angle :math:`\theta` between the magnetic field and the electron
            velocity, in radians. Default is ``\pi/2`` (isotropic average).

        epsilon_B : float or array-like, optional
            Fraction of post-shock internal energy in magnetic fields,
            :math:`\epsilon_B`. Default is ``0.1``.

        epsilon_E : float or array-like, optional
            Fraction of post-shock internal energy in relativistic electrons,
            :math:`\epsilon_E`. Default is ``0.1``.

        gamma_min : float or array-like, optional
            Minimum electron Lorentz factor :math:`\gamma_{\rm min}`. Default is ``1``.

        gamma_max : float or array-like, optional
            Maximum electron Lorentz factor :math:`\gamma_{\rm max}`. Default is ``1e6``.

            This parameter only affects the calculation when :math:`p < 2`.

        Returns
        -------
        B : astropy.units.Quantity
            The inferred magnetic field strength :math:`B` in **Gauss**.

        R : astropy.units.Quantity
            The inferred radius of the synchrotron-emitting region :math:`R`
            in **cm**.

        Notes
        -----
        This function follows the formalism laid out in :footcite:t:`demarchiRadioAnalysisSN2004C2022` (DM22) to compute
        the magnetic field strength and radius of the synchrotron-emitting region from the observed break frequency and
        flux density of a broken power-law SED. The calculations assume a power-law distribution of electron energies
        characterized by the index :math:`p`.

        Letting :math:`\nu_{\rm brk}` be the break frequency (in GHz) between the SSA-thick and SSA-thin regimes, and
        :math:`F_{\nu_{\rm brk}}` be the flux density (in Jy) at that frequency, the magnetic field strength :math:`B`
        (in Gauss) and radius :math:`R` (in cm) of the emitting region can be computed
        by requiring that the asymptotic behavior of
        the optically thick and thin synchrotron spectra match at :math:`\nu_{\rm brk}`. The equations used are
        equations
        (16) and (17) from DM22 with minor alterations.

        In treating the electron energy distribution, some additional care is taken based on the value of :math:`p`. In
        particular, we assume a power-law distribution of electron Lorentz factors :math:`\Gamma` such that

        .. math::

            N(\Gamma) d\Gamma = K_e \Gamma^{-p} d\Gamma,\;\; \Gamma_{\rm min} \leq \Gamma \leq \Gamma_{\rm max},

        where :math:`K_e` is the normalization constant, :math:`\Gamma_{\rm min}` is the minimum Lorentz factor,
        and :math:`\Gamma_{\rm max}` is the maximum Lorentz factor. For values of :math:`p > 2`, the total energy is
        dominated by electrons near :math:`\Gamma_{\rm min}`, while for :math:`p < 2`, it is dominated by those near
        :math:`\Gamma_{\rm max}`.

        To account for this, when :math:`p > 2`, we enforce :math:`\Gamma_{\rm max} = \infty` in the energy integral,
        while
        for :math:`p < 2`, we enforce the upper limit on the energy integral to be :math:`\Gamma_{\rm max}`. This leads
        to
        a correction factor :math:`\delta` defined as:

        .. math::

            \delta = \begin{cases} 1, & p > 2 \[6pt]
            \left(\frac{\Gamma_{\rm max}}{\Gamma_{\rm min}}\right)^{2 - p} - 1, & p < 2 \end{cases}

        which modifies the expressions for :math:`B` and :math:`R` accordingly.

        References
        ----------

        .. footbibliography::

        """
        # Validate units of all unit bearing quantities and coerce them to the expected
        # units for the optimized backend.
        nu_brk = ensure_in_units(nu_brk, u.GHz)
        F_nu_brk = ensure_in_units(F_nu_brk, u.Jy)
        distance = ensure_in_units(distance, u.Mpc)

        # Check the validity of p values. We need to ensure that ``p`` behaves as an array at
        # this point, so we cast it explicitly.
        p = np.asarray(p)
        if np.any(p == 2):
            raise ValueError(
                "compute_BR_from_BPL does not support p = 2. "
                "Use p slightly above or below 2, or implement a dedicated "
                "logarithmic normalization for this case."
            )

        # Dispatch to the optimized backend.
        B, R = self._opt_from_params_to_physics(
            nu_brk=nu_brk,
            F_nu_brk=F_nu_brk,
            distance=distance,
            p=p,
            f=f,
            theta=theta,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        return B * u.Gauss, R * u.cm

    def _opt_from_physics_to_params(
        self,
        B: Union[float, np.ndarray],
        R: Union[float, np.ndarray],
        distance: Union[float, np.ndarray],
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ):
        r"""
        Compute the synchrotron self-absorption frequency and peak flux.

        This is a **low-level, performance-optimized routine** that assumes all
        inputs are provided as dimensionless scalars or NumPy arrays in CGS units.
        No unit validation or safety checks are performed.

        The function analytically eliminates the normalization of the electron
        energy distribution using microphysical energy-partition assumptions,
        introducing a correction factor ``delta`` to account for the convergence
        of the electron energy integral when :math:`p < 2`.

        Parameters
        ----------
        B : float or array-like
            Magnetic field strength in Gauss.

        R : float or array-like
            Radius of the emitting region in cm.

        distance : float or array-like
            Distance to the source in cm.

        p, f, theta, epsilon_B, epsilon_E, gamma_min, gamma_max
            See the user-facing function ``compute_BPL_SED_from_BR``.

        Returns
        -------
        nu_brk : float or array-like
            Synchrotron self-absorption break frequency (Hz-equivalent CGS).

        F_nu_brk : float or array-like
            Peak flux density at the break frequency (CGS-equivalent).

        Notes
        -----
        See notes in the user-facing wrapper function `compute_BR_from_BPL` for details on the
        underlying physics and assumptions.
        """
        # Validate input parameters. We do NOT check explicitly for shape correctness of the arrays, instead opting
        # to allow an error to rise naturally if the shapes are incompatible during computation. We do want to ensure
        # that all of the constants are cast properly for array operations and masking.
        p, gamma_min, gamma_max = (
            np.asarray(p, dtype="f8"),
            np.asarray(gamma_min, dtype="f8"),
            np.asarray(gamma_max, dtype="f8"),
        )

        # Obtain the synchrotron coefficients relevant for this scenario. We need c_1,c_5, and c_6.
        c_5 = compute_c5_parameter(p)  # Should match shape of p.
        c_6 = compute_c6_parameter(p)  # Should match shape of p.
        c_1 = c_1_cgs  # Scalar constant.

        # Construct the array for delta. See the documentation notes for details on the procedure here / physical
        # motivation. To do this efficiently, we pre-allocate the array and then fill in values based on the conditions.
        # Because we pre-allocate with ones, we only need to fill in values where p < 2. In THIS IMPLEMENTATION, we
        # ignore
        # cases where p == 2 for efficiency; these should be screened for upstream if needed.
        delta = np.ones_like(p)
        _p_lt_2_mask = p < 2
        delta[_p_lt_2_mask] = (gamma_max[_p_lt_2_mask] / gamma_min[_p_lt_2_mask]) ** (
            2 - p[_p_lt_2_mask]
        ) - 1  # Fill in values where p < 2.

        # Construct the "p_norm" term used in the B and R calculations. This allows us to handle the two branches
        # of the solution (p < 2 and p > 2) in a unified way.
        # Note that we take the absolute value here to avoid issues with negative bases and fractional exponents.
        # The p == 2 case is handled upstream.
        p_norm = np.abs(p - 2.0)

        # Compute the electron energy floor using the gamma_min parameter and the standard formula.
        E_l = electron_rest_energy_cgs * gamma_min

        # Calculate nu_brk. We break this into parts for clarity. See the notes on this function for an explanation of
        # where this formula comes from.
        _nu_brk_coeff = 2 * c_1
        _nu_brk_t1 = R * f * (epsilon_E / (delta * epsilon_B)) * p_norm / (6 * np.pi)
        _nu_brk_t2 = c_6 ** (2 / (p + 4))
        _nu_brk_t3 = np.sin(theta) ** ((p + 2) / (p + 4))
        _nu_brk_E_l_exp = 2 * (p - 2) / (p + 4)
        _nu_brk_B_exp = (p + 6) / (p + 4)

        nu_brk = (
            _nu_brk_coeff
            * (_nu_brk_t1 ** (2 / (p + 4)))
            * (E_l**_nu_brk_E_l_exp)
            * (B**_nu_brk_B_exp)
            * _nu_brk_t2
            * _nu_brk_t3
        )

        # Calculate F_nu_brk by inserting nu_brk into either the optically thick or thin formula.
        # We choose the optically thick formula here (equation 14 of DeMarchi+22) for consistency.
        _F_nu_brk_coeff = (c_5 / c_6) * np.pi * (R / distance) ** 2
        _F_nu_brk_t1 = (B * np.sin(theta)) ** (-1 / 2)
        _F_nu_brk_t2 = (nu_brk / (2 * c_1)) ** (5 / 2)

        F_nu_brk = _F_nu_brk_coeff * _F_nu_brk_t1 * _F_nu_brk_t2

        return nu_brk, F_nu_brk

    def from_physics_to_params(
        self,
        B: Union[float, np.ndarray, u.Quantity],
        R: Union[float, np.ndarray, u.Quantity],
        distance: Union[float, np.ndarray, u.Quantity],
        *,
        p: Union[float, np.ndarray] = 3.0,
        f: Union[float, np.ndarray] = 0.5,
        theta: Union[float, np.ndarray] = np.pi / 2,
        epsilon_B: Union[float, np.ndarray] = 0.1,
        epsilon_E: Union[float, np.ndarray] = 0.1,
        gamma_min: Union[float, np.ndarray] = 1.0,
        gamma_max: Union[float, np.ndarray] = 1e6,
    ):
        r"""
        Compute the synchrotron self-absorption break frequency and peak flux.

        This is a **low-level, performance-optimized routine** that assumes all
        inputs are provided as dimensionless scalars or NumPy arrays in CGS units.
        No unit validation or safety checks are performed.

        The function analytically eliminates the normalization of the electron
        energy distribution using microphysical energy-partition assumptions,
        introducing a correction factor ``delta`` to account for the convergence
        of the electron energy integral when :math:`p < 2`.

        Parameters
        ----------
        B : float or array-like
            Magnetic field strength in Gauss.

        R : float or array-like
            Radius of the emitting region in cm.

        distance : float or array-like
            Distance to the source in Mpc.

        p, f, theta, epsilon_B, epsilon_E, gamma_min, gamma_max
            See the user-facing function ``compute_BPL_SED_from_BR``.

        Returns
        -------
        nu_brk : float or array-like
            Synchrotron self-absorption break frequency (Hz-equivalent CGS).

        F_nu_brk : float or array-like
            Peak flux density at the break frequency (CGS-equivalent).

        Notes
        -----
        In the optically thick regime, the synchrotron SED follows a power law

        .. math::

            F_\nu = \frac{c_5}{c_6} \left(B\sin\theta\right)^{-1/2} \left(\frac{\nu}{2c_1}\right)^{5/2}
            \frac{\pi R^2}{d^2}.

        In the optically thin regime, the SED follows a different power law:

        .. math::

            F_\nu = \frac{4\pi f R^3}{3 d^2} c_5 N_0 \left(B\sin \theta\right)^{(p+1)/2}
            \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

        We define the break frequency :math:`\nu_{\rm brk}` as the frequency where these two power laws intersect and
        the
        corresponding peak flux density :math:`F_{\nu_{\rm brk}}`. By equating the two expressions for :math:`F_\nu` at
        :math:`\nu = \nu_{\rm brk}`, we can solve for :math:`\nu_{\rm brk}` and :math:`F_{\nu_{\rm brk}}` in terms of
        the physical parameters :math:`B`, :math:`R`, and :math:`d`.

        Equation these two equations yields

        .. math::

            \nu_{\rm brk} = 2 c_1 \left[\frac{4}{3} c_6 f N_0\right]^{2/(p+4)} R^{2/(p+4)}
            \left(B\sin\theta\right)^{(p+2)/(p+4)}

        Inserting :math:`\nu_{\rm brk}` back into either expression for :math:`F_\nu` gives the peak flux density
        :math:`F_{\nu_{\rm brk}}`.

        The normalization :math:`N_0` of the electron distribution is eliminated
        analytically by equating the total electron energy density to a fraction
        :math:`\epsilon_E` of the post-shock internal energy density, with magnetic
        energy fraction :math:`\epsilon_B`.

        For :math:`p \neq 2`, this introduces a correction factor

        .. math::

            \delta =
            \begin{cases}
            1, & p > 2 \
            (\gamma_{\max}/\gamma_{\min})^{2-p} - 1, & p < 2
            \end{cases}

        which accounts for the convergence properties of the electron energy integral.
        """
        # Validate units of all unit bearing quantities and coerce them to the expected
        # units for the optimized backend.
        if hasattr(B, "units"):
            B = B.to_value(u.Gauss)
        if hasattr(R, "units"):
            R = R.to_value(u.cm)
        if hasattr(distance, "units"):
            distance = distance.to_value(u.cm)

        # Check the validity of p values. We need to ensure that ``p`` behaves as an array at
        # this point, so we cast it explicitly.
        p = np.asarray(p)
        if np.any(p == 2):
            raise ValueError(
                "compute_BR_from_BPL does not support p = 2. "
                "Use p slightly above or below 2, or implement a dedicated "
                "logarithmic normalization for this case."
            )

        # Dispatch to the optimized backend.
        nu, F_nu = self._opt_from_physics_to_params(
            B=B,
            R=R,
            distance=distance,
            p=p,
            f=f,
            theta=theta,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        return nu * u.Hz, F_nu * u.erg / (u.s * u.cm**2 * u.Hz)
