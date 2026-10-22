r"""
Pen-and-paper synchrotron spectral inversions.

This module provides closed-form (and, for one case, semi-analytic) inversions which map a small set of observed
spectral quantities,

.. math::

    \left(t,\; \nu_{\rm brk},\; F_{\rm brk}\right) \longrightarrow \left(B,\; R,\; \dots\right),

to the physical parameters of a one-zone synchrotron source with a self-absorbed (SSA) spectral peak. They are
intended as a quick analysis tool which sits *outside* of the SED generators in
:mod:`trilobite.radiation.synchrotron.SEDs`: there is no SED object to build and no spectral-regime detection, so each
inversion **assumes** the regime it is named after.

.. rubric:: Conventions

- :math:`\nu_{\rm brk}` is the best-fit break frequency of the spectrum. It is identified with the SSA frequency
  :math:`\nu_a`, which is accurate to a fraction of a percent across the relevant parameter space.
- :math:`F_{\rm brk}` is the **asymptotic intercept flux** of the fit: the flux at :math:`\nu_{\rm brk}` of the
  optically thick asymptote extrapolated to the break. The user is responsible for supplying this value. For a
  traditional (sharp) broken power-law fit it is the fitted normalization at the break, so no correction is needed.
  For a smoothed fit, the asymptotic flux must be extracted from the fitted form before calling these functions.
- The source is in the local universe and non-relativistic. Cosmological redshift and Doppler corrections are
  **not** applied, and ``distance`` is simply the distance to the source. These inversions should not be used for
  relativistic outflows such as GRBs.
- Pitch-angle treatment follows the rest of Trilobite. Either a fixed pitch angle ``alpha`` is used
  (default :math:`\pi/2`) or ``pitch_average=True`` selects the isotropic coefficients, including the
  angle-averaged synchrotron loss rate in the cooled inversions.
- Every observable and every microphysical parameter may be a scalar or an array. They are broadcast against one
  another following the usual numpy rules, so e.g. a column of break frequencies and a row of electron indices give a
  grid of solutions. All array-valued entries of the returned dictionaries have the broadcast shape.
- Bare floats are interpreted in CGS units (Hz, :math:`\mathrm{erg\,cm^{-2}\,s^{-1}\,Hz^{-1}}`, cm, s,
  :math:`\mathrm{erg\,s^{-1}}`) unless otherwise stated. The De Marchi inversion keeps the units of its original
  implementation (GHz, Jy, Mpc).

.. rubric:: Inversions

.. list-table::
    :widths: 40 60
    :header-rows: 1

    * - Function
      - Assumed regime
    * - :func:`invert_ssa_Diggins26`
      - Uncooled electrons, :math:`\nu_a > \nu_m`.
    * - :func:`invert_ssa_synchrotron_cooled_Diggins26`
      - Synchrotron-cooled electrons with :math:`\nu_c \ll \nu_a`, cooling time :math:`t`.
    * - :func:`invert_ssa_ic_cooled_Diggins26`
      - Inverse-Compton-cooled electrons with :math:`\nu_c \ll \nu_a`, seed luminosity :math:`L`.
    * - :func:`invert_ssa_combined_cooled_Diggins26`
      - Both loss channels with :math:`\nu_c \ll \nu_a` (numerical).
    * - :func:`invert_ssa_with_cooling_Diggins26`
      - Evaluates the cooled and uncooled branches and selects the self-consistent one.
    * - :func:`invert_ssa_DeMarchi22`
      - Fixed-angle uncooled inversion of :footcite:t:`demarchiRadioAnalysisSN2004C2022`.

The inversions labelled ``Diggins26`` are derived in the appendix of Diggins et al. (2026); the derivations are
summarised in :ref:`synchrotron_spectral_inversion`.

See Also
--------
:ref:`synchrotron_spectral_inversion`
    Derivations and usage notes for the inversions in this module.

References
----------
.. footbibliography::
"""

import warnings
from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import units as u

from trilobite.radiation.constants import (
    c_cgs,
    electron_rest_energy_cgs,
    electron_rest_mass_cgs,
    sigma_T_cgs,
)
from trilobite.radiation.synchrotron.utils import (
    c_1_cgs,
    c_1_gamma_cgs,
    compute_c5_parameter,
    compute_c6_parameter,
)
from trilobite.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from trilobite._typing import _UnitBearingScalarLike

__all__ = [
    "invert_ssa_Diggins26",
    "invert_ssa_synchrotron_cooled_Diggins26",
    "invert_ssa_ic_cooled_Diggins26",
    "invert_ssa_combined_cooled_Diggins26",
    "invert_ssa_with_cooling_Diggins26",
    "invert_ssa_DeMarchi22",
]

_FNU_UNIT = "erg cm^-2 s^-1 Hz^-1"
_A_IC_CGS = 3.0 * np.pi * electron_rest_energy_cgs / sigma_T_cgs
"""float: Inverse-Compton cooling constant ``A_IC`` in ``gamma_c = A_IC R^2 / (L t)``."""

_NU_C_ACCURACY = 0.1
"""float: ``nu_c / nu_a`` below which the cooled asymptotic forms are accurate to < 0.1%."""


# ================================================ #
# PRIVATE HELPERS
# ================================================ #
class _SSAMicrophysics:
    """
    Bundle of microphysics, pitch-angle convention and derived constants (all CGS) for the Diggins26 inversions.

    Every parameter may be array-valued; the derived constants broadcast against one another and against the
    observables. The reduced constants ``A_N``, ``A_nu`` and ``A_F`` are stored for both the uncooled index ``p``
    (suffix ``_u``) and the cooled index ``p + 1`` (suffix ``_c``). The electron normalization ``N_0 = A_N B^2`` is
    always set by the *uncooled* energy budget.
    """

    def __init__(self, p, epsilon_E, epsilon_B, f_A, f_V, alpha, pitch_average, gamma_min, gamma_max):
        self.p, self.gamma_min, self.gamma_max = (np.asarray(x, dtype=float) for x in (p, gamma_min, gamma_max))
        self.f_A, self.f_V = np.asarray(f_A, dtype=float), np.asarray(f_V, dtype=float)

        # Pitch angle. The isotropic synchrotron cooling constant (9 pi m_e c / sigma_T, from <sin^2 alpha> = 2/3)
        # is NOT the sin(alpha) -> 1 limit of the fixed-angle value.
        if pitch_average:
            if alpha is not None:
                warnings.warn(
                    "Pitch averaging is enabled, so the provided `alpha` is ignored. "
                    "(Set `alpha=None` to suppress this warning.)",
                    stacklevel=3,
                )
            self.sin = 1.0
            self.A_s = 9.0 * np.pi * electron_rest_mass_cgs * c_cgs / sigma_T_cgs
        else:
            self.sin = np.sin(np.pi / 2 if alpha is None else np.asarray(alpha, dtype=float))
            self.A_s = 6.0 * np.pi * electron_rest_mass_cgs * c_cgs / (sigma_T_cgs * self.sin**2)

        # First moment of the Lorentz-factor distribution, M = int gamma^(1-p) dgamma, evaluated for array inputs.
        if np.any(self.gamma_max <= self.gamma_min):
            raise ValueError("`gamma_max` must be greater than `gamma_min`.")
        e = 2.0 - self.p
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            M1 = np.where(
                e == 0.0,
                np.log(self.gamma_max / self.gamma_min),
                (self.gamma_max**e - self.gamma_min**e) / e,
            )
        if not np.all(np.isfinite(M1)):
            raise ValueError(
                "The first moment of the electron distribution is not finite. Use p > 2 or a finite gamma_max."
            )
        self.A_N = (np.asarray(epsilon_E, dtype=float) / np.asarray(epsilon_B, dtype=float)) / (
            8.0 * np.pi * electron_rest_energy_cgs * M1
        )

        def reduced(q):
            c5 = compute_c5_parameter(q, pitch_average=pitch_average)
            c6 = compute_c6_parameter(q, pitch_average=pitch_average)
            A_nu = (
                2.0
                * c_1_cgs
                * electron_rest_energy_cgs ** (2.0 * (q - 1.0) / (q + 4.0))
                * (c6 * self.f_V ** (1.0 / 3.0)) ** (2.0 / (q + 4.0))
                * self.sin ** ((q + 2.0) / (q + 4.0))
            )
            A_F = (c5 / c6) * np.pi * self.f_A * self.sin**-0.5 * (A_nu / (2.0 * c_1_cgs)) ** 2.5
            return c5, c6, A_nu, A_F

        self.c5_u, self.c6_u, self.A_nu_u, self.A_F_u = reduced(self.p)
        self.c5_c, self.c6_c, self.A_nu_c, self.A_F_c = reduced(self.p + 1.0)

    def gamma_c(self, B, R, L, t):
        """Cooling Lorentz factor with synchrotron and (optionally) IC losses."""
        return 1.0 / (t * (B**2 / self.A_s + L / (_A_IC_CGS * R**2)))

    def compton_Y(self, B, R, L):
        """Ratio of the IC to the synchrotron loss rate."""
        return self.A_s * L / (_A_IC_CGS * R**2 * B**2)


def _invert_ssa_Diggins26(mp: _SSAMicrophysics, log_nu_brk, log_F_brk, log_D):
    """Uncooled SSA inversion. Natural-log CGS inputs, returns linear ``(B, R)`` in CGS."""
    p = mp.p
    n = 2.0 * p + 13.0
    log_X = log_nu_brk - np.log(mp.A_nu_u)
    log_FD2 = log_F_brk + 2.0 * log_D
    log_AF, log_AN = np.log(mp.A_F_u), np.log(mp.A_N)

    log_B = log_X + (2.0 * log_AF - 4.0 * log_AN - 2.0 * log_FD2) / n
    log_R = -log_X + (-(p + 6.0) * log_AF - log_AN + (p + 6.0) * log_FD2) / n
    return np.exp(log_B), np.exp(log_R)


def _invert_ssa_synchrotron_cooled_Diggins26(mp: _SSAMicrophysics, log_nu_brk, log_F_brk, log_D, log_t):
    """Synchrotron-cooled SSA inversion. Natural-log CGS inputs, returns linear ``(B, R, gamma_c)`` in CGS."""
    p = mp.p
    n = 2.0 * p + 7.0
    log_X = log_nu_brk - np.log(mp.A_nu_c)
    log_Y = log_F_brk + 2.0 * log_D - np.log(mp.A_F_c)
    log_Lam = np.log(mp.A_N * mp.A_s) - log_t

    log_B = ((2.0 * p + 15.0) * log_X - 2.0 * log_Y - 4.0 * log_Lam) / n
    log_R = ((p + 3.0) * log_Y - (2.0 * p + 5.0) * log_X - log_Lam) / n
    B, R = np.exp(log_B), np.exp(log_R)
    return B, R, mp.A_s / (B**2 * np.exp(log_t))


def _invert_ssa_ic_cooled_Diggins26(mp: _SSAMicrophysics, log_nu_brk, log_F_brk, log_D, log_t, log_L):
    """IC-cooled SSA inversion. Natural-log CGS inputs, returns linear ``(B, R, gamma_c)`` in CGS."""
    p = mp.p
    n = 2.0 * p + 17.0
    log_X = log_nu_brk - np.log(mp.A_nu_c)
    log_Y = log_F_brk + 2.0 * log_D - np.log(mp.A_F_c)
    log_Lam = np.log(mp.A_N * _A_IC_CGS) - log_L - log_t

    log_B = ((2.0 * p + 25.0) * log_X - 6.0 * log_Y - 4.0 * log_Lam) / n
    log_R = ((p + 7.0) * log_Y - (2.0 * p + 15.0) * log_X - log_Lam) / n
    B, R = np.exp(log_B), np.exp(log_R)
    return B, R, _A_IC_CGS * R**2 / (np.exp(log_L + log_t))


def _invert_ssa_combined_cooled_Diggins26(
    mp: _SSAMicrophysics, log_nu_brk, log_F_brk, log_D, log_t, log_L, tol=1e-12, max_iter=50
):
    """
    Invert an SSA break cooled by both synchrotron and IC losses.

    Safeguarded Newton iteration on ``g(ln B) = 0`` started from the larger of the single-channel closed forms (each a
    lower bound on ``B``). ``g`` is concave and increasing in ``ln B``, so the iteration converges monotonically.
    Inputs are natural-log CGS; a luminosity of zero (``log_L = -inf``) removes the IC channel. Returns linear
    ``(B, R, gamma_c, n_iter)`` in CGS.
    """
    p = mp.p
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        L = np.exp(log_L)
        t = np.exp(log_t)
        FD2 = np.exp(log_F_brk + 2.0 * log_D)
        nu = np.exp(log_nu_brk)

        # Single-channel closed forms bracket the root from below.
        n7, n17 = 2.0 * p + 7.0, 2.0 * p + 17.0
        log_X = log_nu_brk - np.log(mp.A_nu_c)
        log_Y = log_F_brk + 2.0 * log_D - np.log(mp.A_F_c)
        log_B_s = ((2.0 * p + 15.0) * log_X - 2.0 * log_Y - 4.0 * (np.log(mp.A_N * mp.A_s) - log_t)) / n7
        log_B_ic = ((2.0 * p + 25.0) * log_X - 6.0 * log_Y - 4.0 * (np.log(mp.A_N * _A_IC_CGS) - log_L - log_t)) / n17
        lnB_lo = np.maximum(log_B_s, np.where(L > 0, log_B_ic, -np.inf))

        X = nu / (2.0 * c_1_cgs)
        R_pref = FD2 / (np.pi * mp.f_A * (mp.c5_c / mp.c6_c) * X**2.5)  # R^2 = R_pref * (B sin)^(1/2)
        const = np.log(mp.c6_c * electron_rest_energy_cgs**p * mp.f_V ** (1.0 / 3.0) * mp.A_N) - (
            p + 5.0
        ) / 2.0 * np.log(X)

        def g_and_slope(lnB):
            B = np.exp(lnB)
            R = np.sqrt(R_pref * np.sqrt(B * mp.sin))
            gc = mp.gamma_c(B, R, L, t)
            Y = mp.compton_Y(B, R, L)
            g = const + np.log(R) + 2.0 * lnB + np.log(gc) + (p + 3.0) / 2.0 * np.log(B * mp.sin)
            return g, (2.0 * p + 17.0) / 4.0 - 2.5 / (1.0 + Y)

        lo = np.array(lnB_lo, dtype=float)
        g_lo, _ = g_and_slope(lo)
        hi = lo + np.maximum(-g_lo, 0.0) / ((2.0 * p + 7.0) / 4.0)  # analytic upper bound
        x = lo.copy()
        n_iter = np.zeros_like(x, dtype=int)
        active = np.ones_like(x, dtype=bool)
        for it in range(max_iter):
            g, s = g_and_slope(x)
            lo = np.where(g < 0, x, lo)
            hi = np.where(g > 0, x, hi)
            x_new = x - g / s
            x_new = np.where((x_new < lo) | (x_new > hi), 0.5 * (lo + hi), x_new)  # bisect if Newton leaves bracket
            step = np.abs(x_new - x)
            n_iter = np.where(active, it + 1, n_iter)
            active &= step > tol
            x = x_new
            if not active.any():
                break

        B = np.exp(x)
        R = np.sqrt(R_pref * np.sqrt(B * mp.sin))
        return B, R, mp.gamma_c(B, R, L, t), n_iter


def _cooling_summary(mp: _SSAMicrophysics, B, R, gamma_c, nu_brk, caller: str, L=None) -> dict:
    """Build the result dict (with units) of a cooled inversion; warn if ``nu_c >= nu_a`` for any input."""
    nu_c = c_1_gamma_cgs * B * mp.sin * gamma_c**2
    consistent = nu_c < nu_brk  # the cooled branch requires nu_c < nu_a
    if not np.all(consistent):
        warnings.warn(
            f"{caller}: nu_c >= nu_a for {int(np.sum(~consistent))} of {np.size(consistent)} input(s). The cooled "
            "asymptotic forms assume nu_c << nu_a, so these results are not self-consistent.",
            stacklevel=3,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        # Fractional energy error (gamma_c / gamma_min)^(2-p) / (p-1) of the uncooled normalization (gamma_max -> inf).
        delta_c = np.where(
            np.isinf(mp.gamma_max) & (mp.p > 2.0),
            (gamma_c / mp.gamma_min) ** (2.0 - mp.p) / (mp.p - 1.0),
            np.nan,
        )
    out = {
        "B": B * u.G,
        "R": R * u.cm,
        "gamma_c": gamma_c,
        "nu_c": nu_c * u.Hz,
        "nu_c_over_nu_a": nu_c / nu_brk,
        "delta_c": delta_c,
        "consistent": consistent,
        "accurate": nu_c < _NU_C_ACCURACY * nu_brk,  # cooled asymptotic forms good to < 0.1%
        "slow_cooling": gamma_c > mp.gamma_min,
    }
    if L is not None:
        Y = mp.compton_Y(B, R, L)
        out.update({"Y": Y, "ic_dominated": Y > 1.0})
    return out


def _prepare(nu_brk, F_brk, distance, t=None, L=None):
    """Convert user inputs to a list of broadcast CGS float arrays (``nu, F, D`` then ``t``, ``L`` if given)."""
    nu = ensure_in_units(nu_brk, "Hz")
    F = ensure_in_units(F_brk, _FNU_UNIT)
    D = ensure_in_units(distance, "cm")
    arrays = [nu, F, D]
    if t is not None:
        arrays.append(ensure_in_units(t, "s"))
    if L is not None:
        arrays.append(ensure_in_units(L, "erg s^-1"))
    arrays = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in arrays])
    return arrays


# ================================================ #
# PUBLIC INVERSIONS
# ================================================ #
def invert_ssa_Diggins26(
    nu_brk: "_UnitBearingScalarLike",
    F_brk: "_UnitBearingScalarLike",
    distance: "_UnitBearingScalarLike",
    *,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_A: float = 0.5,
    f_V: float = 0.5,
    alpha: float = None,
    pitch_average: bool = False,
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
) -> dict:
    r"""
    Invert an SSA break for :math:`B` and :math:`R` assuming uncooled electrons.

    Solves the pair of power-law relations for the SSA frequency and the optically thick asymptote of a
    one-zone source with a power-law electron distribution, with :math:`N_0` fixed by equipartition,

    .. math::

        B = \frac{\nu_{\rm brk}}{A_\nu}\,A_F^{\frac{2}{2p+13}}\,A_N^{-\frac{4}{2p+13}}\,
            \left(F_{\rm brk}D^2\right)^{-\frac{2}{2p+13}},

    .. math::

        R = \frac{A_\nu}{\nu_{\rm brk}}\,A_F^{-\frac{p+6}{2p+13}}\,A_N^{-\frac{1}{2p+13}}\,
            \left(F_{\rm brk}D^2\right)^{\frac{p+6}{2p+13}},

    where :math:`A_N`, :math:`A_\nu` and :math:`A_F` are the reduced constants described in
    :ref:`synchrotron_spectral_inversion`.

    Parameters
    ----------
    nu_brk : float, array-like, or ~astropy.units.Quantity
        Best-fit break frequency, identified with :math:`\nu_a`. Bare floats are in Hz.
    F_brk : float, array-like, or ~astropy.units.Quantity
        Asymptotic intercept flux density of the fit at ``nu_brk``. Bare floats are in
        :math:`\mathrm{erg\,cm^{-2}\,s^{-1}\,Hz^{-1}}`.
    distance : float, array-like, or ~astropy.units.Quantity
        Distance to the source. Bare floats are in cm.
    p : float or array-like, optional
        Electron power-law index. Requires :math:`p > 2` when ``gamma_max`` is infinite.
    epsilon_E, epsilon_B : float or array-like, optional
        Fractions of the post-shock energy in relativistic electrons and in the magnetic field.
    f_A, f_V : float or array-like, optional
        Area and volume filling factors.
    alpha : float or array-like, optional
        Pitch angle in radians. Defaults to :math:`\pi/2`. Ignored (with a warning if given) when
        ``pitch_average=True``.
    pitch_average : bool, optional
        If ``True``, use the isotropic pitch-angle coefficients.
    gamma_min, gamma_max : float or array-like, optional
        Bounds of the electron Lorentz-factor distribution.

    Returns
    -------
    dict
        ``"B"`` (:math:`\mathrm{G}`) and ``"R"`` (:math:`\mathrm{cm}`), with the shape obtained by broadcasting all
        of the inputs against one another.

    Notes
    -----
    This inversion makes no statement about cooling. If electrons are cooled at :math:`\nu_a`, use one of the cooled
    inversions or :func:`invert_ssa_with_cooling_Diggins26`. Relativistic effects and redshift are not treated.

    See Also
    --------
    invert_ssa_with_cooling_Diggins26 : Cooled-versus-uncooled consistency check.
    """
    mp = _SSAMicrophysics(p, epsilon_E, epsilon_B, f_A, f_V, alpha, pitch_average, gamma_min, gamma_max)
    nu, F, D = _prepare(nu_brk, F_brk, distance)
    B, R = _invert_ssa_Diggins26(mp, np.log(nu), np.log(F), np.log(D))
    return {"B": B * u.G, "R": R * u.cm}


def invert_ssa_synchrotron_cooled_Diggins26(
    nu_brk: "_UnitBearingScalarLike",
    F_brk: "_UnitBearingScalarLike",
    distance: "_UnitBearingScalarLike",
    t: "_UnitBearingScalarLike",
    *,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_A: float = 0.5,
    f_V: float = 0.5,
    alpha: float = None,
    pitch_average: bool = False,
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
) -> dict:
    r"""
    Invert an SSA break for :math:`B` and :math:`R` assuming synchrotron-cooled electrons.

    The break is assumed to lie in the cooled segment of a slow-cooling spectrum (:math:`\nu_c \ll \nu_a`), with
    :math:`\gamma_c = A_c/(B^2 t)` set by synchrotron losses over the time :math:`t`. The factor :math:`B^2` in the
    equipartition normalization cancels the :math:`B^{-2}` in :math:`\gamma_c`, so the system is again a pair of
    power laws,

    .. math::

        B = \left(\frac{\nu_{\rm brk}}{A_\nu^{(c)}}\right)^{\frac{2p+15}{2p+7}}
            \left(\frac{F_{\rm brk}D^2}{A_F^{(c)}}\right)^{-\frac{2}{2p+7}}\Lambda^{-\frac{4}{2p+7}},
        \qquad
        R = \left(\frac{F_{\rm brk}D^2}{A_F^{(c)}}\right)^{\frac{p+3}{2p+7}}
            \left(\frac{\nu_{\rm brk}}{A_\nu^{(c)}}\right)^{-\frac{2p+5}{2p+7}}\Lambda^{-\frac{1}{2p+7}},

    with :math:`\Lambda = A_N A_c/t`.

    Parameters
    ----------
    nu_brk, F_brk, distance, p, epsilon_E, epsilon_B, f_A, f_V, alpha, pitch_average, gamma_min, gamma_max
        As in :func:`invert_ssa_Diggins26`.
    t : float, array-like, or ~astropy.units.Quantity
        Dynamical (effective cooling) time of the source. Bare floats are in s.

    Returns
    -------
    dict
        - ``"B"``, ``"R"``: magnetic field (G) and radius (cm).
        - ``"gamma_c"``: cooling Lorentz factor.
        - ``"nu_c"``: synchrotron frequency of the cooling electrons (Hz).
        - ``"nu_c_over_nu_a"``: regime diagnostic; the assumed branch requires :math:`\ll 1`.
        - ``"delta_c"``: fractional energy error of the uncooled normalization (NaN unless :math:`p>2` and
          ``gamma_max`` is infinite).
        - ``"consistent"``: ``nu_c < nu_a``.
        - ``"accurate"``: ``nu_c < 0.1 nu_a`` (cooled asymptotic forms good to better than 0.1%).
        - ``"slow_cooling"``: ``gamma_c > gamma_min``.

    Warns
    -----
    UserWarning
        If :math:`\nu_c \ge \nu_a` for any input, since the assumed regime then does not hold.

    Notes
    -----
    The isotropic cooling constant :math:`9\pi m_e c/\sigma_T` (``pitch_average=True``) assumes pitch-angle
    scattering is fast compared with cooling; it is not the :math:`\sin\alpha\to1` limit of the fixed-angle value.
    """
    mp = _SSAMicrophysics(p, epsilon_E, epsilon_B, f_A, f_V, alpha, pitch_average, gamma_min, gamma_max)
    nu, F, D, tt = _prepare(nu_brk, F_brk, distance, t=t)
    B, R, gamma_c = _invert_ssa_synchrotron_cooled_Diggins26(mp, np.log(nu), np.log(F), np.log(D), np.log(tt))
    return _cooling_summary(mp, B, R, gamma_c, nu, "invert_ssa_synchrotron_cooled_Diggins26")


def invert_ssa_ic_cooled_Diggins26(
    nu_brk: "_UnitBearingScalarLike",
    F_brk: "_UnitBearingScalarLike",
    distance: "_UnitBearingScalarLike",
    t: "_UnitBearingScalarLike",
    L: "_UnitBearingScalarLike",
    *,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_A: float = 0.5,
    f_V: float = 0.5,
    alpha: float = None,
    pitch_average: bool = False,
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
) -> dict:
    r"""
    Invert an SSA break for :math:`B` and :math:`R` assuming inverse-Compton-cooled electrons.

    The break is assumed to lie in the cooled segment (:math:`\nu_c \ll \nu_a`), with the cooling Lorentz factor set
    by Thomson-regime IC losses on the radiation field of a central source of luminosity :math:`L`,
    :math:`\gamma_c = A_{\rm IC}R^2/(L t)` with :math:`A_{\rm IC} = 3\pi m_e c^2/\sigma_T`. Since
    :math:`\gamma_c\propto R^2` there is no cancellation against the normalization, but the system remains log-linear,

    .. math::

        B = \left(\frac{\nu_{\rm brk}}{A_\nu^{(c)}}\right)^{\frac{2p+25}{2p+17}}
            \left(\frac{F_{\rm brk}D^2}{A_F^{(c)}}\right)^{-\frac{6}{2p+17}}\Lambda_{\rm IC}^{-\frac{4}{2p+17}},
        \qquad
        R = \left(\frac{F_{\rm brk}D^2}{A_F^{(c)}}\right)^{\frac{p+7}{2p+17}}
            \left(\frac{\nu_{\rm brk}}{A_\nu^{(c)}}\right)^{-\frac{2p+15}{2p+17}}\Lambda_{\rm IC}^{-\frac{1}{2p+17}},

    with :math:`\Lambda_{\rm IC} = A_N A_{\rm IC}/(L t)`.

    Parameters
    ----------
    nu_brk, F_brk, distance, t, p, epsilon_E, epsilon_B, f_A, f_V, alpha, pitch_average, gamma_min, gamma_max
        As in :func:`invert_ssa_synchrotron_cooled_Diggins26`.
    L : float, array-like, or ~astropy.units.Quantity
        Bolometric luminosity of the seed-photon source at the epoch being inverted. Bare floats are in
        :math:`\mathrm{erg\,s^{-1}}`. Must be positive.

    Returns
    -------
    dict
        As :func:`invert_ssa_synchrotron_cooled_Diggins26`, plus

        - ``"Y"``: ratio of the IC to the synchrotron loss rate at the solution.
        - ``"ic_dominated"``: ``Y > 1``.

    Warns
    -----
    UserWarning
        If :math:`\nu_c \ge \nu_a` for any input.
    """
    mp = _SSAMicrophysics(p, epsilon_E, epsilon_B, f_A, f_V, alpha, pitch_average, gamma_min, gamma_max)
    nu, F, D, tt, LL = _prepare(nu_brk, F_brk, distance, t=t, L=L)
    if np.any(LL <= 0):
        raise ValueError("The IC-cooled inversion requires a positive luminosity `L`.")
    B, R, gamma_c = _invert_ssa_ic_cooled_Diggins26(mp, np.log(nu), np.log(F), np.log(D), np.log(tt), np.log(LL))
    return _cooling_summary(mp, B, R, gamma_c, nu, "invert_ssa_ic_cooled_Diggins26", L=LL)


def invert_ssa_combined_cooled_Diggins26(
    nu_brk: "_UnitBearingScalarLike",
    F_brk: "_UnitBearingScalarLike",
    distance: "_UnitBearingScalarLike",
    t: "_UnitBearingScalarLike",
    L: "_UnitBearingScalarLike",
    *,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_A: float = 0.5,
    f_V: float = 0.5,
    alpha: float = None,
    pitch_average: bool = False,
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
) -> dict:
    r"""
    Invert an SSA break assuming electrons cooled by both synchrotron and inverse-Compton losses.

    With both loss channels the rates add,

    .. math::

        \gamma_c = \frac{A_c}{B^2 t}\,\frac{1}{1+Y},
        \qquad
        Y = \frac{A_c L}{A_{\rm IC}R^2B^2},

    and :math:`\gamma_c` is a sum of two power laws, so there is no closed form. The optically thick asymptote fixes
    :math:`R \propto B^{1/4}`, which reduces the problem to a single equation :math:`g(\ln B)=0` in :math:`B`.
    The function :math:`g` is increasing and concave, and each single-channel closed form is a lower bound on
    :math:`B`, so the root is bracketed analytically and found by safeguarded Newton iteration (a few
    iterations). The solver is vectorized over the inputs, e.g. over posterior draws.

    Parameters
    ----------
    nu_brk, F_brk, distance, t, p, epsilon_E, epsilon_B, f_A, f_V, alpha, pitch_average, gamma_min, gamma_max
        As in :func:`invert_ssa_synchrotron_cooled_Diggins26`.
    L : float, array-like, or ~astropy.units.Quantity
        Bolometric luminosity of the seed-photon source at the epoch being inverted. Bare floats are in
        :math:`\mathrm{erg\,s^{-1}}`. ``L = 0`` removes the IC channel.

    Returns
    -------
    dict
        As :func:`invert_ssa_ic_cooled_Diggins26`, plus ``"n_iter"``, the number of Newton iterations used.

    Warns
    -----
    UserWarning
        If :math:`\nu_c \ge \nu_a` for any input.
    """
    mp = _SSAMicrophysics(p, epsilon_E, epsilon_B, f_A, f_V, alpha, pitch_average, gamma_min, gamma_max)
    nu, F, D, tt, LL = _prepare(nu_brk, F_brk, distance, t=t, L=L)
    if np.any(LL < 0):
        raise ValueError("The luminosity `L` must be non-negative.")
    with np.errstate(divide="ignore"):
        log_L = np.log(LL)
    B, R, gamma_c, n_iter = _invert_ssa_combined_cooled_Diggins26(
        mp, np.log(nu), np.log(F), np.log(D), np.log(tt), log_L
    )
    summary = _cooling_summary(mp, B, R, gamma_c, nu, "invert_ssa_combined_cooled_Diggins26", L=LL)
    return {**summary, "n_iter": n_iter}


def invert_ssa_with_cooling_Diggins26(
    nu_brk: "_UnitBearingScalarLike",
    F_brk: "_UnitBearingScalarLike",
    distance: "_UnitBearingScalarLike",
    t: "_UnitBearingScalarLike",
    L: "_UnitBearingScalarLike",
    *,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_A: float = 0.5,
    f_V: float = 0.5,
    alpha: float = None,
    pitch_average: bool = False,
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
) -> dict:
    r"""
    Invert an SSA break, selecting between the cooled and uncooled branches by self-consistency.

    For each input the function computes

    1. the **uncooled** solution (:func:`invert_ssa_Diggins26`) and the cooling break it implies from the *combined*
       synchrotron and IC losses, and
    2. the **cooled** solution (:func:`invert_ssa_combined_cooled_Diggins26`),

    and keeps the branch whose own cooling break is consistent with the assumption it was derived under: the
    uncooled branch needs :math:`\nu_c > \nu_a`, the cooled branch needs :math:`\nu_c < \nu_a` and
    :math:`\gamma_c > \gamma_{\min}`.

    Inputs ``nu_brk``, ``F_brk``, ``distance``, ``t`` and ``L`` are treated as data and broadcast against one
    another; ``L`` may be an array with one value per observation.

    Parameters
    ----------
    nu_brk, F_brk, distance, t, L, p, epsilon_E, epsilon_B, f_A, f_V, alpha, pitch_average, gamma_min, gamma_max
        As in :func:`invert_ssa_combined_cooled_Diggins26`.

    Returns
    -------
    dict
        - ``"B"``, ``"R"``, ``"gamma_c"``, ``"nu_c"``: the selected solution. NaN where ``branch`` is
          ``"ambiguous"`` or ``"fast_cooling"``.
        - ``"Y"``, ``"nu_c_over_nu_a"``: diagnostics of the selected branch.
        - ``"branch"``: array of ``"uncooled"``, ``"cooled"``, ``"fast_cooling"`` or ``"ambiguous"``. A draw is
          ``"fast_cooling"`` when the cooled ordering holds but :math:`\gamma_c \le \gamma_{\min}`, which the
          slow-cooling distribution does not describe, and ``"ambiguous"`` when both or neither branch is consistent
          (typically :math:`\nu_c\approx\nu_a`, where the asymptotic forms fail).
        - ``"accurate"``: the selected branch is in the regime where its asymptotic forms are accurate to better than
          0.1% (:math:`\nu_c < 0.1\nu_a` cooled, :math:`\nu_c > 10\nu_a` uncooled).
        - ``"n_iter"``: Newton iterations used by the cooled solver.
        - ``"cooled"``, ``"uncooled"``: dicts of ``B``, ``R``, ``gamma_c`` and ``nu_c`` for each branch, for
          diagnostics.
    """
    mp = _SSAMicrophysics(p, epsilon_E, epsilon_B, f_A, f_V, alpha, pitch_average, gamma_min, gamma_max)
    nu, F, D, tt, LL = _prepare(nu_brk, F_brk, distance, t=t, L=L)
    if np.any(LL < 0):
        raise ValueError("The luminosity `L` must be non-negative.")
    log_nu, log_F, log_D, log_t = np.log(nu), np.log(F), np.log(D), np.log(tt)
    with np.errstate(divide="ignore"):
        log_L = np.log(LL)

    B_u, R_u = _invert_ssa_Diggins26(mp, log_nu, log_F, log_D)
    B_c, R_c, gc_c, n_iter = _invert_ssa_combined_cooled_Diggins26(mp, log_nu, log_F, log_D, log_t, log_L)

    # The uncooled branch still cools: evaluate the break it implies with both loss channels.
    gc_u = mp.gamma_c(B_u, R_u, LL, tt)
    nuc_u = c_1_gamma_cgs * B_u * mp.sin * gc_u**2
    nuc_c = c_1_gamma_cgs * B_c * mp.sin * gc_c**2

    ok_u = nuc_u > nu
    ok_c = (nuc_c < nu) & (gc_c > mp.gamma_min)
    fast = (nuc_c < nu) & (gc_c <= mp.gamma_min) & ~ok_u
    branch = np.where(
        ok_u & ~ok_c,
        "uncooled",
        np.where(ok_c & ~ok_u, "cooled", np.where(fast, "fast_cooling", "ambiguous")),
    )
    use_c = branch == "cooled"
    flagged = (branch == "ambiguous") | (branch == "fast_cooling")

    def pick(x_c, x_u, mask_flagged=True):
        out = np.where(use_c, x_c, x_u)
        return np.where(flagged, np.nan, out) if mask_flagged else out

    B_sel, R_sel = pick(B_c, B_u, False), pick(R_c, R_u, False)
    nuc_sel = pick(nuc_c, nuc_u, False)
    return {
        "B": pick(B_c, B_u) * u.G,
        "R": pick(R_c, R_u) * u.cm,
        "gamma_c": pick(gc_c, gc_u),
        "nu_c": pick(nuc_c, nuc_u) * u.Hz,
        "Y": np.where(flagged, np.nan, mp.compton_Y(B_sel, R_sel, LL)),
        "nu_c_over_nu_a": np.where(flagged, np.nan, nuc_sel / nu),
        "branch": branch,
        "accurate": np.where(use_c, nuc_c < _NU_C_ACCURACY * nu, nuc_u > 10.0 * nu) & ~flagged,
        "n_iter": n_iter,
        "cooled": {"B": B_c * u.G, "R": R_c * u.cm, "gamma_c": gc_c, "nu_c": nuc_c * u.Hz},
        "uncooled": {"B": B_u * u.G, "R": R_u * u.cm, "gamma_c": gc_u, "nu_c": nuc_u * u.Hz},
    }


# ================================================ #
# DE MARCHI (2022)
# ================================================ #
def _invert_ssa_DeMarchi22(
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
    Invert an SSA turnover using the DeMarchi et al. (2022) analytic closure.

    This low-level helper implements the DeMarchi et al. (2022) inversion
    for a broken power-law synchrotron spectrum with an SSA turnover. It
    returns the magnetic field strength :math:`B` and emitting radius
    :math:`R` implied by the supplied turnover frequency and flux density.

    The implementation works internally in log space for numerical
    stability, but this helper itself accepts linear-valued inputs in the
    units assumed by the underlying analytic formulae.

    Parameters
    ----------
    nu_brk : float or array-like
        SSA turnover frequency :math:`\nu_{\rm brk}` in GHz.
    F_nu_brk : float or array-like
        Flux density at the turnover, :math:`F_{\nu,\rm brk}`, in Jy.
    distance : float or array-like
        Source distance in Mpc.
    p : float or array-like, optional
        Electron power-law index. Values equal to :math:`2` are not
        supported by this implementation.
    f : float or array-like, optional
        Effective filling factor of the emitting region.
    theta : float or array-like, optional
        Pitch angle in radians.
    epsilon_B : float or array-like, optional
        Fraction of internal energy in magnetic fields.
    epsilon_E : float or array-like, optional
        Fraction of internal energy in relativistic electrons.
    gamma_min : float or array-like, optional
        Minimum electron Lorentz factor.
    gamma_max : float or array-like, optional
        Maximum electron Lorentz factor. This matters only for
        :math:`p < 2`.

    Returns
    -------
    tuple of ndarray
        Two-element tuple ``(B, R)`` containing

        - magnetic field strength in G
        - emitting radius in cm

    Notes
    -----
    This is a specialized analytic closure based on the DM22 treatment and
    is distinct from the Diggins26 inversions in this module. It does not pitch-angle average; the pitch angle
    is fixed by ``theta``.

    .. dropdown:: Numerical implementation notes
       :icon: info

       The calculation is carried out in log space to reduce floating-point
       precision loss and catastrophic cancellation in the analytic
       expressions.

    .. dropdown:: Applicability
       :icon: chevron-down

       Use this function when you specifically want the DM22 inversion based
       on an SSA turnover treated as a broken power law. To pitch-angle average,
       use :func:`invert_ssa_Diggins26`.

    See Also
    --------
    invert_ssa_DeMarchi22
        Public, unit-aware wrapper.
    invert_ssa_Diggins26
        Uncooled SSA inversion for a fixed-angle or pitch-averaged source.
    :ref:`synch_sed_theory`
        General synchrotron SED conventions.
    """
    # Cast all parameters to float arrays. Shapes are free to differ; they broadcast against one another and against
    # the observables.
    p, gamma_min, gamma_max = (
        np.asarray(p, dtype="f8"),
        np.asarray(gamma_min, dtype="f8"),
        np.asarray(gamma_max, dtype="f8"),
    )

    # Obtain the synchrotron coefficients relevant for this scenario. We need c_1,c_5, and c_6.
    c_5 = compute_c5_parameter(p)
    c_6 = compute_c6_parameter(p)
    c_1 = c_1_cgs  # Scalar constant.

    # Construct the array for delta. See the documentation notes for details on the procedure here / physical
    # motivation. Delta is 1 for p > 2 and (gamma_max / gamma_min)^(2-p) - 1 for p < 2. In THIS IMPLEMENTATION, we
    # ignore cases where p == 2; these should be screened for upstream if needed.
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.where(p < 2, (gamma_max / gamma_min) ** (2 - p) - 1, 1.0)

    # Construct the "p_norm" term used in the B and R calculations. This allows us to handle the two branches
    # of the solution (p < 2 and p > 2) in a unified way.
    # Note that we take the absolute value here to avoid issues with negative bases and fractional exponents.
    # The p == 2 case is handled upstream.
    p_norm = np.abs(p - 2.0)

    # Compute the electron energy floor using the gamma_min parameter and the standard formula.
    E_l = electron_rest_energy_cgs * gamma_min

    # Normalize unit bearing values and convert them to logarithmic quantities for
    # numerical stability.
    _log_nu_brk = np.log(nu_brk / 5)
    _log_F_nu_brk = np.log(F_nu_brk)
    _log_distance = np.log(distance)
    _log_c1 = np.log(c_1)
    _log_E_l = np.log(E_l)
    _log_delta = np.log(delta)
    _log_sin_theta = np.log(np.sin(theta))
    _log_p_norm = np.log(p_norm)

    # Compute the magnetic field following equation (16) of DeMarchi+22. We break this into
    # components for clarity. The operation should be heavily CPU bound, so this should not have any impact
    # on optimization.
    #
    # Here nu_brk is in GHz, E_l is in erg, distance is in Mpc, F_nu_brk is in Jy, and B will be in Gauss.
    _log_B_coeff = 21.6396 + _log_nu_brk - _log_c1
    _log_B_num = (
        -51.41402455
        + ((4 - 2 * p) * _log_E_l)
        + (2 * _log_delta)
        + (2 * np.log(epsilon_B / epsilon_E))
        + np.log(c_5)
        + ((1 / 2) * (-5 - 2 * p) * _log_sin_theta)
    )
    _log_B_denom = 2 * _log_p_norm + 2 * _log_distance + (_log_F_nu_brk - np.log(0.5)) + (3 * np.log(c_6))
    _log_B = _log_B_coeff + (2 / (13 + 2 * p)) * (_log_B_num - _log_B_denom)

    # Compute the radius following equation (17) of DeMarchi+22. We break this into parts as well on the same
    # basis as above.
    _log_R_coeff = -21.63956 + _log_c1 + (-1 * _log_nu_brk)
    _log_R_t1 = np.log(12 * epsilon_B) + (-6 - p) * np.log(c_5) + (5 + p) * np.log(c_6)
    _log_R_t2 = (6 + p) * 59.818022 + 2 * _log_sin_theta + (-5 - p) * np.log(np.pi) + (12 + 2 * p) * _log_distance
    _log_R_t3 = (2 - p) * _log_E_l + (6 + p) * _log_F_nu_brk
    _log_R_t4 = -1 * (np.log(epsilon_E) + _log_p_norm + np.log(f / 0.5))
    _log_R = _log_R_coeff + (1 / (13 + 2 * p)) * (_log_R_t1 + _log_R_t2 + _log_R_t3 + _log_R_t4)
    return np.exp(_log_B), np.exp(_log_R)


def invert_ssa_DeMarchi22(
    nu_brk: "_UnitBearingScalarLike",
    F_brk: "_UnitBearingScalarLike",
    distance: "_UnitBearingScalarLike",
    *,
    p: Union[float, np.ndarray] = 3.0,
    f: Union[float, np.ndarray] = 0.5,
    theta: Union[float, np.ndarray] = np.pi / 2,
    epsilon_B: Union[float, np.ndarray] = 0.1,
    epsilon_E: Union[float, np.ndarray] = 0.1,
    gamma_min: Union[float, np.ndarray] = 1.0,
    gamma_max: Union[float, np.ndarray] = 1e6,
) -> dict:
    r"""
    Invert an SSA break using the analytic formulae of :footcite:t:`demarchiRadioAnalysisSN2004C2022`.

    This is the public, unit-aware wrapper for the DeMarchi et al. (2022) SSA inversion. It converts an SSA break
    frequency and flux density to the magnetic field strength :math:`B` and emitting radius :math:`R`, using the
    same observable convention as the other inversions in this module: ``nu_brk`` is the best-fit break frequency
    and ``F_brk`` is the asymptotic intercept flux of the fit.

    Parameters
    ----------
    nu_brk : float, array-like, or ~astropy.units.Quantity
        SSA break frequency. Bare floats are in GHz.
    F_brk : float, array-like, or ~astropy.units.Quantity
        Asymptotic intercept flux density at the break. Bare floats are in Jy.
    distance : float, array-like, or ~astropy.units.Quantity
        Distance to the source. Bare floats are in Mpc.
    p : float or array-like, optional
        Electron power-law index. The singular case :math:`p = 2` is not supported.
    f : float or array-like, optional
        Effective filling factor of the emitting region.
    theta : float or array-like, optional
        Pitch angle in radians.
    epsilon_B, epsilon_E : float or array-like, optional
        Fractions of the internal energy in magnetic fields and in relativistic electrons.
    gamma_min, gamma_max : float or array-like, optional
        Bounds of the electron Lorentz-factor distribution. ``gamma_max`` matters only for :math:`p < 2`.

    Returns
    -------
    dict
        ``"B"`` (:math:`\mathrm{G}`) and ``"R"`` (:math:`\mathrm{cm}`).

    Notes
    -----
    .. important::

        This inversion does **not** pitch-angle average. It uses the fixed pitch angle ``theta`` throughout. To
        use the isotropic pitch-angle coefficients, use :func:`invert_ssa_Diggins26`.

    The source is treated as local and non-relativistic, as for the other inversions in this module.

    See Also
    --------
    invert_ssa_Diggins26 : Uncooled inversion with optional pitch-angle averaging.

    References
    ----------
    .. footbibliography::
    """
    nu_brk = ensure_in_units(nu_brk, u.GHz)
    F_brk = ensure_in_units(F_brk, u.Jy)
    distance = ensure_in_units(distance, u.Mpc)

    p = np.asarray(p)
    if np.any(p == 2):
        raise ValueError(
            "The DeMarchi et al. (2022) inversion does not support p = 2. "
            "Use p slightly above or below 2, or implement a dedicated logarithmic normalization for this case."
        )

    B, R = _invert_ssa_DeMarchi22(
        nu_brk=nu_brk,
        F_nu_brk=F_brk,
        distance=distance,
        p=p,
        f=f,
        theta=theta,
        epsilon_B=epsilon_B,
        epsilon_E=epsilon_E,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )
    # R does not depend on every input (e.g. gamma_max), so broadcast both outputs to the full input shape.
    shape = np.broadcast_shapes(
        *(np.shape(x) for x in (nu_brk, F_brk, distance, p, f, theta, epsilon_B, epsilon_E, gamma_min, gamma_max))
    )
    return {"B": np.broadcast_to(B, shape) * u.G, "R": np.broadcast_to(R, shape) * u.cm}
