"""Low-Level 1-Zone synchrotron SED functions."""

import numpy as np

from triceratops._typing import _ArrayLike


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


COOLING_SED_FUNCTION_REGISTRY = {
    "fast_cooling": _log_powerlaw_sbpl_sed_cool_1,
    "slow_cooling": _log_powerlaw_sbpl_sed_cool_2,
    "no_cooling": _log_powerlaw_sbpl_sed,
}
"""dict: Registry mapping cooling regimes to their corresponding SED functions."""


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


SSA_SED_FUNCTION_REGISTRY = {
    "optically_thick": _log_powerlaw_sbpl_sed_ssa_2,
    "optically_thin": _log_powerlaw_sbpl_sed_ssa_1,
}
"""dict: Registry mapping SSA regimes to their corresponding SED functions."""


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
    Synchrotron SED with SSA in the **slow-cooling** regime, assuming the ordering :math:`\nu < \nu_m < \nu_a`.

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


def _log_powerlaw_sbpl_sed_ssa_cool_8(
    log_nu: "_ArrayLike",
    log_nu_m: float,
    log_nu_a: float,
    log_nu_max: float,
    p: float,
    s: float,
):
    r"""
    Synchrotron SED with SSA in the **fast-cooling** regime, assuming the ordering :math:`\nu_c < \nu_m < \nu_a`.

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


SSA_COOLING_SED_FUNCTION_REGISTRY = {
    "Spectrum1": _log_powerlaw_sbpl_sed_ssa_1,
    "Spectrum2": _log_powerlaw_sbpl_sed_ssa_2,
    "Spectrum3": _log_powerlaw_sbpl_sed_ssa_cool_3,
    "Spectrum4": _log_powerlaw_sbpl_sed_ssa_cool_4,
    "Spectrum5": _log_powerlaw_sbpl_sed_ssa_cool_5,
    "Spectrum6": _log_powerlaw_sbpl_sed_ssa_cool_6,
    "Spectrum7": _log_powerlaw_sbpl_sed_ssa_cool_7,
    "Spectrum8": _log_powerlaw_sbpl_sed_ssa_cool_8,
}
"""dict: Registry mapping SSA cooling regimes to their corresponding SED functions."""
