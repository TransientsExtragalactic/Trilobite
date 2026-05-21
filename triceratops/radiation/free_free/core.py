"""
Core free–free (bremsstrahlung) radiation functions.

This module provides low-level and public-facing routines for computing
**thermal free–free (bremsstrahlung) emissivity and absorption coefficients**
in a fully-ionized plasma.  The formulas follow
:footcite:t:`RybickiLightman` (Eqs. 5.14–5.19) and are valid in the
non-relativistic thermal limit.

The module is organised in two layers:

* **Private backend** (``_log_ff_*``): all inputs are pre-converted to natural
  logarithms and the results are returned as natural logarithms.  Working in
  log-space gives numerical stability over the large dynamic range encountered
  in astrophysical plasmas without any unit overhead.
* **Public API** (``compute_ff_*``): unit-aware wrappers that accept plain
  :class:`float` or :class:`~astropy.units.Quantity` inputs, perform unit
  coercion via :func:`~triceratops.utils.misc_utils.ensure_in_units`, delegate
  to the backend, and return :class:`~astropy.units.Quantity` results.

.. note::

    All CGS coefficients are pre-computed at module load time and stored as
    module-level constants for performance.  These constants are **not** part
    of the public API and should not be imported directly.
"""

from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import units as u

from triceratops.radiation.constants import h_cgs, kB_cgs
from triceratops.utils.misc_utils import ensure_in_units

from .gaunt_factor import (
    compute_ff_gaunt_factor,
    compute_ff_gaunt_factor_comp,
    compute_mean_ff_gaunt_factor,
    compute_mean_ff_gaunt_factor_comp,
)

if TYPE_CHECKING:
    from triceratops._typing import _ArrayLike, _UnitBearingArrayLike

# =============================================== #
# Module-level CGS constants
# =============================================== #
#: Pre-computed free–free emissivity coefficient :math:`C_{ff}` in CGS
#: (erg s⁻¹ cm⁻³ Hz⁻¹ sr⁻¹), from Rybicki & Lightman Eq. 5.14.
_ff_emissivity_coefficient_cgs = 6.8e-38
_log_ff_emissivity_coefficient_cgs = np.log(_ff_emissivity_coefficient_cgs)

#: Pre-computed free–free absorption coefficient :math:`C_\alpha` in CGS
#: (cm⁵ K^{1/2} Hz^3), from Rybicki & Lightman Eq. 5.19.  Used by the
#: *exact* (non-RJ) absorption backend which carries T^{-1/2} and ν^{-3}.
_ff_absorption_coefficient_cgs = 3.7e8
_log_ff_absorption_coefficient_cgs = np.log(_ff_absorption_coefficient_cgs)

#: Rayleigh–Jeans absorption coefficient :math:`C_\alpha^{\rm RJ}` in CGS
#: (cm⁵ K^{3/2} Hz^2).  Derived from the exact coefficient via the RJ
#: substitution :math:`(1 - e^{-h\nu/k_BT}) \approx h\nu/k_BT`:
#:
#:   :math:`C_\alpha^{\rm RJ} = C_\alpha \cdot h / k_B \approx 0.01776`.
#:
#: This is the constant that multiplies T^{-3/2} and ν^{-2} in the RJ form.
_ff_RJ_absorption_coefficient_cgs = _ff_absorption_coefficient_cgs * h_cgs / kB_cgs
_log_ff_RJ_absorption_coefficient_cgs = np.log(_ff_RJ_absorption_coefficient_cgs)

_ff_cooling_time_coefficient_cgs = (4.6e5 * u.yr).to_value(u.s)
_log_ff_cooling_time_coefficient_cgs = np.log(_ff_cooling_time_coefficient_cgs)


# =============================================== #
# Low-Level Backend for Free-Free Core
# =============================================== #
def _log_ff_emissivity(
    log_nu: "Union[float, _ArrayLike]",
    log_n_e: "Union[float, _ArrayLike]",
    log_n_i: "Union[float, _ArrayLike]",
    Z: "Union[float, _ArrayLike]",
    log_T: "Union[float, _ArrayLike]",
    g_ff: "Union[float, _ArrayLike]",
) -> "Union[float, np.ndarray]":
    r"""
    Natural logarithm of the thermal free–free emissivity (exact form).

    Evaluates the logarithm of the Rybicki & Lightman bremsstrahlung
    emissivity formula:

    .. math::

        j_\nu =
        C_{\rm ff}\,
        Z^2\, n_e\, n_i\,
        T^{-1/2}\,
        e^{-h\nu / k_B T}\,
        g_{\rm ff}

    where :math:`C_{ff} = 6.8 \times 10^{-38}` CGS and :math:`g_{ff}` is the
    velocity-averaged free–free Gaunt factor.  All inputs are expected in
    **natural-log CGS** form to exploit the log-linear structure of the
    expression and avoid numerical overflow across wide dynamic ranges.

    Parameters
    ----------
    log_nu : float or ~numpy.ndarray
        Natural logarithm of the photon frequency :math:`\nu` [Hz].
    log_n_e : float or ~numpy.ndarray
        Natural logarithm of the electron number density :math:`n_e`
        [cm\ :sup:`-3`].
    log_n_i : float or ~numpy.ndarray
        Natural logarithm of the ion number density :math:`n_i`
        [cm\ :sup:`-3`].
    Z : float or ~numpy.ndarray
        Mean ionic charge :math:`Z` (dimensionless, must be positive).
    log_T : float or ~numpy.ndarray
        Natural logarithm of the electron temperature :math:`T` [K].
    g_ff : float or ~numpy.ndarray
        Free–free Gaunt factor (dimensionless, must be positive).

    Returns
    -------
    log_jnu : float or :class:`~numpy.ndarray`
        Natural logarithm of the emissivity
        :math:`j_\nu` [erg s\ :sup:`-1` cm\ :sup:`-3` Hz\ :sup:`-1` sr\ :sup:`-1`].

    See Also
    --------
    _log_ff_absorption : Corresponding absorption coefficient backend.
    _log_ff_RJ_emissivity : Rayleigh–Jeans limit (:math:`h\nu \ll k_B T`).
    _log_ff_Wien_emissivity : Wien limit (:math:`h\nu \gg k_B T`).
    """
    # Recover the linear frequency and temperature for the exponential cutoff.
    # All other terms remain in log-space.
    nu = np.exp(log_nu)
    T = np.exp(log_T)

    # Dimensionless exponent −hν / k_B T
    exp_term = -h_cgs * nu / (kB_cgs * T)

    return (
        _log_ff_emissivity_coefficient_cgs
        + 2 * np.log(Z)
        + log_n_e
        + log_n_i
        - 0.5 * log_T
        + exp_term  # log(exp(−hν/kT)) = −hν/kT
        + np.log(g_ff)
    )


def _log_ff_absorption(
    log_nu: "Union[float, _ArrayLike]",
    log_n_e: "Union[float, _ArrayLike]",
    log_n_i: "Union[float, _ArrayLike]",
    Z: "Union[float, _ArrayLike]",
    log_T: "Union[float, _ArrayLike]",
    g_ff: "Union[float, _ArrayLike]",
) -> "Union[float, np.ndarray]":
    r"""
    Natural logarithm of the thermal free–free absorption coefficient (exact form).

    Evaluates the logarithm of the bremsstrahlung absorption coefficient:

    .. math::

        \alpha_\nu =
        C_\alpha\,
        Z^2\, n_e\, n_i\,
        T^{-1/2}\,
        \nu^{-3}\,
        \bigl(1 - e^{-h\nu / k_B T}\bigr)\,
        g_{ff}

    where :math:`C_\alpha = 3.7 \times 10^{8}` CGS.  The factor
    :math:`(1 - e^{-h\nu/k_B T})` enforces detailed balance between
    emission and absorption (Kirchhoff's law).  It is computed via
    :func:`numpy.log1p` to avoid catastrophic cancellation near
    :math:`x = h\nu / k_B T \to 0`.

    Parameters
    ----------
    log_nu : float or ~numpy.ndarray
        Natural logarithm of the photon frequency :math:`\nu` [Hz].
    log_n_e : float or ~numpy.ndarray
        Natural logarithm of the electron number density :math:`n_e`
        [cm\ :sup:`-3`].
    log_n_i : float or ~numpy.ndarray
        Natural logarithm of the ion number density :math:`n_i`
        [cm\ :sup:`-3`].
    Z : float or ~numpy.ndarray
        Mean ionic charge :math:`Z` (dimensionless, must be positive).
    log_T : float or ~numpy.ndarray
        Natural logarithm of the electron temperature :math:`T` [K].
    g_ff : float or ~numpy.ndarray
        Free–free Gaunt factor (dimensionless, must be positive).

    Returns
    -------
    log_alpha : float or :class:`~numpy.ndarray`
        Natural logarithm of the absorption coefficient
        :math:`\alpha_\nu` [cm\ :sup:`-1`].

    Notes
    -----
    The factor :math:`\log(1 - e^{-h\nu/k_B T})` is evaluated as
    ``numpy.log1p(-numpy.exp(-h*nu/(k_B*T)))`` rather than the naive form
    to preserve precision when :math:`h\nu \ll k_B T`.

    See Also
    --------
    _log_ff_emissivity : Corresponding emissivity backend.
    _log_ff_RJ_absorption : Rayleigh–Jeans limit (:math:`h\nu \ll k_B T`).
    _log_ff_Wien_absorption : Wien limit (:math:`h\nu \gg k_B T`).
    """
    nu = np.exp(log_nu)
    T = np.exp(log_T)

    # Exponential argument −hν / k_B T (always ≤ 0)
    exp_term = -h_cgs * nu / (kB_cgs * T)

    return (
        _log_ff_absorption_coefficient_cgs
        + 2 * np.log(Z)
        + log_n_e
        + log_n_i
        - 0.5 * log_T
        - 3 * log_nu
        + np.log1p(-np.exp(exp_term))  # log(1 − exp(−hν/kT)), numerically stable
        + np.log(g_ff)
    )


def _log_ff_RJ_emissivity(
    log_nu: "Union[float, _ArrayLike]",
    log_n_e: "Union[float, _ArrayLike]",
    log_n_i: "Union[float, _ArrayLike]",
    Z: "Union[float, _ArrayLike]",
    log_T: "Union[float, _ArrayLike]",
    g_ff: "Union[float, _ArrayLike]",
) -> "Union[float, np.ndarray]":
    r"""
    Natural logarithm of the Rayleigh–Jeans free–free emissivity.

    In the **Rayleigh–Jeans limit** :math:`h\nu \ll k_B T` the exponential
    suppression factor satisfies :math:`e^{-h\nu/k_B T} \approx 1`, so the
    emissivity simplifies to

    .. math::

        j_\nu^{\rm RJ} =
        C_{\rm ff}\,
        Z^2\, n_e\, n_i\,
        T^{-1/2}\,
        g_{\rm ff}

    i.e.\ it is **independent of frequency** and has a shallower temperature
    dependence than the full expression.  This approximation is appropriate for
    radio and microwave frequencies in hot (:math:`T \gtrsim 10^4` K) plasmas.

    Parameters
    ----------
    log_nu : float or ~numpy.ndarray
        Natural logarithm of the photon frequency :math:`\nu` [Hz].
        Accepted for API consistency but not used in the computation.
    log_n_e : float or ~numpy.ndarray
        Natural logarithm of the electron number density :math:`n_e`
        [cm\ :sup:`-3`].
    log_n_i : float or ~numpy.ndarray
        Natural logarithm of the ion number density :math:`n_i`
        [cm\ :sup:`-3`].
    Z : float or ~numpy.ndarray
        Mean ionic charge :math:`Z` (dimensionless, must be positive).
    log_T : float or ~numpy.ndarray
        Natural logarithm of the electron temperature :math:`T` [K].
    g_ff : float or ~numpy.ndarray
        Free–free Gaunt factor (dimensionless, must be positive).

    Returns
    -------
    log_jnu : float or :class:`~numpy.ndarray`
        Natural logarithm of the emissivity
        :math:`j_\nu^{\rm RJ}` [erg s\ :sup:`-1` cm\ :sup:`-3` Hz\ :sup:`-1` sr\ :sup:`-1`].

    Notes
    -----
    The ``log_nu`` argument is accepted for interface consistency but is not
    used in the computation.  This makes it safe to swap between
    :func:`_log_ff_emissivity` and :func:`_log_ff_RJ_emissivity` without
    changing the call signature.

    See Also
    --------
    _log_ff_emissivity : Exact form of the emissivity.
    _log_ff_RJ_absorption : Matching RJ absorption coefficient.
    """
    # log_nu is not used in the computation (RJ emissivity is frequency-independent),
    # but we broadcast against it so the output shape matches the input shape of nu.
    return (
        _log_ff_emissivity_coefficient_cgs
        + 2 * np.log(Z)
        + log_n_e
        + log_n_i
        - 0.5 * log_T
        + np.log(g_ff)
        + np.zeros_like(np.asarray(log_nu, dtype=float))
    )


def _log_ff_RJ_absorption(
    log_nu: "Union[float, _ArrayLike]",
    log_n_e: "Union[float, _ArrayLike]",
    log_n_i: "Union[float, _ArrayLike]",
    Z: "Union[float, _ArrayLike]",
    log_T: "Union[float, _ArrayLike]",
    g_ff: "Union[float, _ArrayLike]",
) -> "Union[float, np.ndarray]":
    r"""
    Natural logarithm of the Rayleigh–Jeans free–free absorption coefficient.

    In the **Rayleigh–Jeans limit** :math:`h\nu \ll k_B T`, the stimulated-
    emission factor satisfies

    .. math::

        1 - e^{-h\nu / k_B T} \approx \frac{h\nu}{k_B T}

    and the absorption coefficient reduces to

    .. math::

        \alpha_\nu^{\rm RJ} \propto
        Z^2\, n_e\, n_i\,
        T^{-3/2}\,
        \nu^{-2}\,
        g_{ff}

    This is the classical **radio opacity** scaling commonly quoted for
    HII regions and radio-emitting plasma shells.

    Parameters
    ----------
    log_nu : float or ~numpy.ndarray
        Natural logarithm of the photon frequency :math:`\nu` [Hz].
    log_n_e : float or ~numpy.ndarray
        Natural logarithm of the electron number density :math:`n_e`
        [cm\ :sup:`-3`].
    log_n_i : float or ~numpy.ndarray
        Natural logarithm of the ion number density :math:`n_i`
        [cm\ :sup:`-3`].
    Z : float or ~numpy.ndarray
        Mean ionic charge :math:`Z` (dimensionless, must be positive).
    log_T : float or ~numpy.ndarray
        Natural logarithm of the electron temperature :math:`T` [K].
    g_ff : float or ~numpy.ndarray
        Free–free Gaunt factor (dimensionless, must be positive).

    Returns
    -------
    log_alpha : float or :class:`~numpy.ndarray`
        Natural logarithm of the absorption coefficient
        :math:`\alpha_\nu^{\rm RJ}` [cm\ :sup:`-1`].

    Notes
    -----
    Compared with :func:`_log_ff_absorption`, the temperature exponent changes
    from :math:`-1/2` to :math:`-3/2` and the frequency exponent from
    :math:`-3` to :math:`-2`, reflecting the substitution
    :math:`(1 - e^{-h\nu/k_B T}) \to h\nu / k_B T`.  The :math:`h/k_B`
    prefactor is absorbed into :data:`_ff_absorption_coefficient_cgs`.

    See Also
    --------
    _log_ff_absorption : Exact form of the absorption coefficient.
    _log_ff_RJ_emissivity : Matching RJ emissivity.
    """
    return (
        _log_ff_RJ_absorption_coefficient_cgs
        + 2 * np.log(Z)
        + log_n_e
        + log_n_i
        - 1.5 * log_T
        - 2 * log_nu
        + np.log(g_ff)
    )


def _log_ff_emissivity_comp(
    log_nu: "Union[float, _ArrayLike]",
    log_n_e: "Union[float, _ArrayLike]",
    log_n_i: "Union[float, _ArrayLike]",
    log_T: "Union[float, _ArrayLike]",
    Zs: "_ArrayLike",
    Xs: "_ArrayLike",
    gffs: "_ArrayLike",
) -> "Union[float, np.ndarray]":
    r"""
    Natural logarithm of the exact free–free emissivity for a multi-species plasma.

    Computes the composition-weighted effective Gaunt factor

    .. math::

        g_{\rm ff,eff} = \sum_i Z_i^2\, x_i\, g_{{\rm ff},i}

    and evaluates the log-emissivity as if :math:`Z = 1` with
    :math:`g_{\rm ff} = g_{\rm ff,eff}`, reproducing the full multi-species sum

    .. math::

        j_\nu = C_{\rm ff}\, n_e\, n_i\, T^{-1/2}\, e^{-h\nu/k_B T}\,
                \sum_i Z_i^2\, x_i\, g_{{\rm ff},i}.

    Parameters
    ----------
    log_nu : float or ~numpy.ndarray
        Natural logarithm of the photon frequency :math:`\nu` [Hz].
    log_n_e : float or ~numpy.ndarray
        Natural logarithm of the electron number density [cm\ :sup:`-3`].
    log_n_i : float or ~numpy.ndarray
        Natural logarithm of the total ion number density [cm\ :sup:`-3`].
    log_T : float or ~numpy.ndarray
        Natural logarithm of the electron temperature [K].
    Zs : ~numpy.ndarray
        Ionic charge numbers of each species (:math:`Z_i \geq 1`).
    Xs : ~numpy.ndarray
        Number fractions of each species (:math:`\sum_i x_i = 1`).
    gffs : ~numpy.ndarray
        Pre-computed free–free Gaunt factor for each species (dimensionless).

    Returns
    -------
    log_jnu : float or ~numpy.ndarray
        Natural logarithm of the emissivity
        :math:`j_\nu` [erg s\ :sup:`-1` cm\ :sup:`-3` Hz\ :sup:`-1` sr\ :sup:`-1`].

    Raises
    ------
    ValueError
        If ``Zs``, ``Xs``, and ``gffs`` do not all have the same 1-D shape.

    See Also
    --------
    _log_ff_emissivity : Single-species exact emissivity.
    _log_ff_absorption_comp : Composition-weighted exact absorption coefficient.
    _log_ff_RJ_emissivity_comp : Composition-weighted Rayleigh–Jeans emissivity.
    """
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)
    gffs = np.asarray(gffs, dtype=float)

    if not (Zs.shape == Xs.shape == gffs.shape):
        raise ValueError(f"Zs, Xs, and gffs must have the same shape; got {Zs.shape}, {Xs.shape}, {gffs.shape}.")
    if Zs.ndim != 1:
        raise ValueError(f"Zs, Xs, and gffs must be 1-D arrays; got shape {Zs.shape}.")

    g_ff_eff = np.sum(Zs**2 * Xs * gffs)
    return _log_ff_emissivity(log_nu, log_n_e, log_n_i, 1.0, log_T, g_ff_eff)


def _log_ff_absorption_comp(
    log_nu: "Union[float, _ArrayLike]",
    log_n_e: "Union[float, _ArrayLike]",
    log_n_i: "Union[float, _ArrayLike]",
    log_T: "Union[float, _ArrayLike]",
    Zs: "_ArrayLike",
    Xs: "_ArrayLike",
    gffs: "_ArrayLike",
) -> "Union[float, np.ndarray]":
    r"""
    Natural logarithm of the exact free–free absorption coefficient for a multi-species plasma.

    Computes the composition-weighted effective Gaunt factor

    .. math::

        g_{\rm ff,eff} = \sum_i Z_i^2\, x_i\, g_{{\rm ff},i}

    and evaluates the log-absorption coefficient as if :math:`Z = 1` with
    :math:`g_{\rm ff} = g_{\rm ff,eff}`, reproducing

    .. math::

        \alpha_\nu = C_\alpha\, n_e\, n_i\, T^{-1/2}\, \nu^{-3}\,
                     (1 - e^{-h\nu/k_B T})\,
                     \sum_i Z_i^2\, x_i\, g_{{\rm ff},i}.

    Parameters
    ----------
    log_nu : float or ~numpy.ndarray
        Natural logarithm of the photon frequency :math:`\nu` [Hz].
    log_n_e : float or ~numpy.ndarray
        Natural logarithm of the electron number density [cm\ :sup:`-3`].
    log_n_i : float or ~numpy.ndarray
        Natural logarithm of the total ion number density [cm\ :sup:`-3`].
    log_T : float or ~numpy.ndarray
        Natural logarithm of the electron temperature [K].
    Zs : ~numpy.ndarray
        Ionic charge numbers of each species (:math:`Z_i \geq 1`).
    Xs : ~numpy.ndarray
        Number fractions of each species (:math:`\sum_i x_i = 1`).
    gffs : ~numpy.ndarray
        Pre-computed free–free Gaunt factor for each species (dimensionless).

    Returns
    -------
    log_alpha : float or ~numpy.ndarray
        Natural logarithm of the absorption coefficient
        :math:`\alpha_\nu` [cm\ :sup:`-1`].

    Raises
    ------
    ValueError
        If ``Zs``, ``Xs``, and ``gffs`` do not all have the same 1-D shape.

    See Also
    --------
    _log_ff_absorption : Single-species exact absorption coefficient.
    _log_ff_emissivity_comp : Composition-weighted exact emissivity.
    _log_ff_RJ_absorption_comp : Composition-weighted Rayleigh–Jeans absorption.
    """
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)
    gffs = np.asarray(gffs, dtype=float)

    if not (Zs.shape == Xs.shape == gffs.shape):
        raise ValueError(f"Zs, Xs, and gffs must have the same shape; got {Zs.shape}, {Xs.shape}, {gffs.shape}.")
    if Zs.ndim != 1:
        raise ValueError(f"Zs, Xs, and gffs must be 1-D arrays; got shape {Zs.shape}.")

    g_ff_eff = np.sum(Zs**2 * Xs * gffs)
    return _log_ff_absorption(log_nu, log_n_e, log_n_i, 1.0, log_T, g_ff_eff)


def _log_ff_RJ_emissivity_comp(
    log_nu: "Union[float, _ArrayLike]",
    log_n_e: "Union[float, _ArrayLike]",
    log_n_i: "Union[float, _ArrayLike]",
    log_T: "Union[float, _ArrayLike]",
    Zs: "_ArrayLike",
    Xs: "_ArrayLike",
    gffs: "_ArrayLike",
) -> "Union[float, np.ndarray]":
    r"""
    Natural logarithm of the Rayleigh–Jeans free–free emissivity for a multi-species plasma.

    Rayleigh–Jeans limit (:math:`h\nu \ll k_B T`) of
    :func:`_log_ff_emissivity_comp`.  Computes

    .. math::

        g_{\rm ff,eff} = \sum_i Z_i^2\, x_i\, g_{{\rm ff},i}

    and evaluates :func:`_log_ff_RJ_emissivity` with :math:`Z = 1` and
    :math:`g_{\rm ff} = g_{\rm ff,eff}`.

    Parameters
    ----------
    log_nu : float or ~numpy.ndarray
        Natural logarithm of the photon frequency :math:`\nu` [Hz].
    log_n_e : float or ~numpy.ndarray
        Natural logarithm of the electron number density [cm\ :sup:`-3`].
    log_n_i : float or ~numpy.ndarray
        Natural logarithm of the total ion number density [cm\ :sup:`-3`].
    log_T : float or ~numpy.ndarray
        Natural logarithm of the electron temperature [K].
    Zs : ~numpy.ndarray
        Ionic charge numbers of each species (:math:`Z_i \geq 1`).
    Xs : ~numpy.ndarray
        Number fractions of each species (:math:`\sum_i x_i = 1`).
    gffs : ~numpy.ndarray
        Pre-computed free–free Gaunt factor for each species (dimensionless).

    Returns
    -------
    log_jnu : float or ~numpy.ndarray
        Natural logarithm of the RJ emissivity
        :math:`j_\nu^{\rm RJ}` [erg s\ :sup:`-1` cm\ :sup:`-3` Hz\ :sup:`-1` sr\ :sup:`-1`].

    Raises
    ------
    ValueError
        If ``Zs``, ``Xs``, and ``gffs`` do not all have the same 1-D shape.

    See Also
    --------
    _log_ff_RJ_emissivity : Single-species RJ emissivity.
    _log_ff_emissivity_comp : Composition-weighted exact emissivity.
    _log_ff_RJ_absorption_comp : Composition-weighted RJ absorption coefficient.
    """
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)
    gffs = np.asarray(gffs, dtype=float)

    if not (Zs.shape == Xs.shape == gffs.shape):
        raise ValueError(f"Zs, Xs, and gffs must have the same shape; got {Zs.shape}, {Xs.shape}, {gffs.shape}.")
    if Zs.ndim != 1:
        raise ValueError(f"Zs, Xs, and gffs must be 1-D arrays; got shape {Zs.shape}.")

    g_ff_eff = np.sum(Zs**2 * Xs * gffs)
    return _log_ff_RJ_emissivity(log_nu, log_n_e, log_n_i, 1.0, log_T, g_ff_eff)


def _log_ff_RJ_absorption_comp(
    log_nu: "Union[float, _ArrayLike]",
    log_n_e: "Union[float, _ArrayLike]",
    log_n_i: "Union[float, _ArrayLike]",
    log_T: "Union[float, _ArrayLike]",
    Zs: "_ArrayLike",
    Xs: "_ArrayLike",
    gffs: "_ArrayLike",
) -> "Union[float, np.ndarray]":
    r"""
    Natural logarithm of the Rayleigh–Jeans free–free absorption coefficient for a multi-species plasma.

    Rayleigh–Jeans limit (:math:`h\nu \ll k_B T`) of
    :func:`_log_ff_absorption_comp`.  Computes

    .. math::

        g_{\rm ff,eff} = \sum_i Z_i^2\, x_i\, g_{{\rm ff},i}

    and evaluates :func:`_log_ff_RJ_absorption` with :math:`Z = 1` and
    :math:`g_{\rm ff} = g_{\rm ff,eff}`.

    Parameters
    ----------
    log_nu : float or ~numpy.ndarray
        Natural logarithm of the photon frequency :math:`\nu` [Hz].
    log_n_e : float or ~numpy.ndarray
        Natural logarithm of the electron number density [cm\ :sup:`-3`].
    log_n_i : float or ~numpy.ndarray
        Natural logarithm of the total ion number density [cm\ :sup:`-3`].
    log_T : float or ~numpy.ndarray
        Natural logarithm of the electron temperature [K].
    Zs : ~numpy.ndarray
        Ionic charge numbers of each species (:math:`Z_i \geq 1`).
    Xs : ~numpy.ndarray
        Number fractions of each species (:math:`\sum_i x_i = 1`).
    gffs : ~numpy.ndarray
        Pre-computed free–free Gaunt factor for each species (dimensionless).

    Returns
    -------
    log_alpha : float or ~numpy.ndarray
        Natural logarithm of the RJ absorption coefficient
        :math:`\alpha_\nu^{\rm RJ}` [cm\ :sup:`-1`].

    Raises
    ------
    ValueError
        If ``Zs``, ``Xs``, and ``gffs`` do not all have the same 1-D shape.

    See Also
    --------
    _log_ff_RJ_absorption : Single-species RJ absorption coefficient.
    _log_ff_absorption_comp : Composition-weighted exact absorption coefficient.
    _log_ff_RJ_emissivity_comp : Composition-weighted RJ emissivity.
    """
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)
    gffs = np.asarray(gffs, dtype=float)

    if not (Zs.shape == Xs.shape == gffs.shape):
        raise ValueError(f"Zs, Xs, and gffs must have the same shape; got {Zs.shape}, {Xs.shape}, {gffs.shape}.")
    if Zs.ndim != 1:
        raise ValueError(f"Zs, Xs, and gffs must be 1-D arrays; got shape {Zs.shape}.")

    g_ff_eff = np.sum(Zs**2 * Xs * gffs)
    return _log_ff_RJ_absorption(log_nu, log_n_e, log_n_i, 1.0, log_T, g_ff_eff)


def _log_ff_Wien_emissivity(
    log_nu: "Union[float, _ArrayLike]",
    log_n_e: "Union[float, _ArrayLike]",
    log_n_i: "Union[float, _ArrayLike]",
    Z: "Union[float, _ArrayLike]",
    log_T: "Union[float, _ArrayLike]",
    g_ff: "Union[float, _ArrayLike]",
) -> "Union[float, np.ndarray]":
    r"""
    Natural logarithm of the Wien-limit free–free emissivity.

    In the **Wien limit** :math:`h\nu \gg k_B T` the emissivity is dominated by
    the exponential Boltzmann factor:

    .. math::

        j_\nu^{\rm Wien} =
        C_{ff}\,
        Z^2\, n_e\, n_i\,
        T^{-1/2}\,
        e^{-h\nu / k_B T}\,
        g_{ff}

    .. note::

        This expression is mathematically **identical** to the exact form
        computed by :func:`_log_ff_emissivity`.  This backend exists as a
        separate dispatch target so that higher-level code can select the
        Wien approximation explicitly without sacrificing readability.

    Parameters
    ----------
    log_nu : float or ~numpy.ndarray
        Natural logarithm of the photon frequency :math:`\nu` [Hz].
    log_n_e : float or ~numpy.ndarray
        Natural logarithm of the electron number density :math:`n_e`
        [cm\ :sup:`-3`].
    log_n_i : float or ~numpy.ndarray
        Natural logarithm of the ion number density :math:`n_i`
        [cm\ :sup:`-3`].
    Z : float or ~numpy.ndarray
        Mean ionic charge :math:`Z` (dimensionless, must be positive).
    log_T : float or ~numpy.ndarray
        Natural logarithm of the electron temperature :math:`T` [K].
    g_ff : float or ~numpy.ndarray
        Free–free Gaunt factor (dimensionless, must be positive).

    Returns
    -------
    log_jnu : float or :class:`~numpy.ndarray`
        Natural logarithm of the emissivity
        :math:`j_\nu^{\rm Wien}` [erg s\ :sup:`-1` cm\ :sup:`-3` Hz\ :sup:`-1` sr\ :sup:`-1`].

    See Also
    --------
    _log_ff_emissivity : Exact (and numerically equivalent) form.
    _log_ff_Wien_absorption : Matching Wien absorption coefficient.
    _log_ff_RJ_emissivity : Rayleigh–Jeans limit (:math:`h\nu \ll k_B T`).
    """
    nu = np.exp(log_nu)
    T = np.exp(log_T)

    exp_term = -h_cgs * nu / (kB_cgs * T)

    return (
        _log_ff_emissivity_coefficient_cgs + 2 * np.log(Z) + log_n_e + log_n_i - 0.5 * log_T + exp_term + np.log(g_ff)
    )


def _log_ff_Wien_absorption(
    log_nu: "Union[float, _ArrayLike]",
    log_n_e: "Union[float, _ArrayLike]",
    log_n_i: "Union[float, _ArrayLike]",
    Z: "Union[float, _ArrayLike]",
    log_T: "Union[float, _ArrayLike]",
    g_ff: "Union[float, _ArrayLike]",
) -> "Union[float, np.ndarray]":
    r"""
    Natural logarithm of the Wien-limit free–free absorption coefficient.

    In the **Wien limit** :math:`h\nu \gg k_B T`, the stimulated-emission
    correction satisfies

    .. math::

        1 - e^{-h\nu / k_B T} \approx 1

    and the absorption coefficient reduces to

    .. math::

        \alpha_\nu^{\rm Wien} =
        C_\alpha\,
        Z^2\, n_e\, n_i\,
        T^{-1/2}\,
        \nu^{-3}\,
        g_{ff}

    This is appropriate for X-ray and gamma-ray frequencies where stimulated
    emission is negligible.

    Parameters
    ----------
    log_nu : float or ~numpy.ndarray
        Natural logarithm of the photon frequency :math:`\nu` [Hz].
    log_n_e : float or ~numpy.ndarray
        Natural logarithm of the electron number density :math:`n_e`
        [cm\ :sup:`-3`].
    log_n_i : float or ~numpy.ndarray
        Natural logarithm of the ion number density :math:`n_i`
        [cm\ :sup:`-3`].
    Z : float or ~numpy.ndarray
        Mean ionic charge :math:`Z` (dimensionless, must be positive).
    log_T : float or ~numpy.ndarray
        Natural logarithm of the electron temperature :math:`T` [K].
    g_ff : float or ~numpy.ndarray
        Free–free Gaunt factor (dimensionless, must be positive).

    Returns
    -------
    log_alpha : float or :class:`~numpy.ndarray`
        Natural logarithm of the absorption coefficient
        :math:`\alpha_\nu^{\rm Wien}` [cm\ :sup:`-1`].

    Notes
    -----
    Compared with :func:`_log_ff_absorption`, this form omits the
    :math:`\log(1 - e^{-h\nu/k_B T})` term (set to zero, i.e.\ the factor is
    treated as unity), giving a shallower frequency dependence than the full
    expression in the optically thin limit.

    See Also
    --------
    _log_ff_absorption : Exact form valid at all frequencies.
    _log_ff_Wien_emissivity : Matching Wien emissivity.
    _log_ff_RJ_absorption : Rayleigh–Jeans limit (:math:`h\nu \ll k_B T`).
    """
    return (
        _log_ff_absorption_coefficient_cgs + 2 * np.log(Z) + log_n_e + log_n_i - 0.5 * log_T - 3 * log_nu + np.log(g_ff)
    )


# ---------------------------------------------- #
# Miscellaneous Functions
# ---------------------------------------------- #
def _log_ff_cooling_time(
    log_n_i: "Union[float, _ArrayLike]",
    log_T: "Union[float, _ArrayLike]",
    Z: "Union[float, _ArrayLike]",
    g_ff: "Union[float, _ArrayLike]",
) -> "Union[float, np.ndarray]":
    r"""
    Natural logarithm of the thermal free–free cooling time.

    Evaluates

    .. math::

        t_{\rm cool} = C_{\rm cool}\,
        \left(\frac{T}{10^4\,\mathrm{K}}\right)^{1/2}
        \frac{1}{n_i\, Z^2\, \bar{g}_{\rm ff}}

    where :math:`C_{\rm cool} = 4.6 \times 10^5\,\mathrm{yr}`.

    Parameters
    ----------
    log_n_i : float or array_like
        Natural logarithm of the ion number density :math:`n_i` [cm\ :sup:`-3`].
    log_T : float or array_like
        Natural logarithm of the electron temperature :math:`T` [K].
    Z : float or array_like
        Mean ionic charge :math:`Z` (dimensionless, must be positive).
    g_ff : float or array_like
        Frequency-averaged free–free Gaunt factor (dimensionless, must be positive).

    Returns
    -------
    log_t : float or ~numpy.ndarray
        Natural logarithm of the cooling time :math:`t_{\rm cool}` [s].

    See Also
    --------
    _log_ff_cooling_time_comp : Composition-weighted variant.
    """
    result = _log_ff_cooling_time_coefficient_cgs + 0.5 * (log_T - np.log(1e4)) - log_n_i - 2 * np.log(Z) - np.log(g_ff)
    return result if np.isfinite(result) else -np.inf


def _log_ff_cooling_time_comp(
    log_n_i: "Union[float, _ArrayLike]",
    log_T: "Union[float, _ArrayLike]",
    Zs: "_ArrayLike",
    Xs: "_ArrayLike",
    gffs: "_ArrayLike",
) -> "Union[float, np.ndarray]":
    r"""
    Natural logarithm of the thermal free–free cooling time for a multi-species plasma.

    Computes the composition-weighted effective Gaunt factor
    :math:`g_{\rm ff,eff} = \sum_i Z_i^2\, x_i\, \bar{g}_{{\rm ff},i}` and
    delegates to :func:`_log_ff_cooling_time` with :math:`Z = 1`.

    Parameters
    ----------
    log_n_i : float or array_like
        Natural logarithm of the total ion number density :math:`n_i` [cm\ :sup:`-3`].
    log_T : float or array_like
        Natural logarithm of the electron temperature :math:`T` [K].
    Zs : (N,) array_like
        Ionic charge numbers of each species.
    Xs : (N,) array_like
        Number fractions of each species (:math:`\sum_i x_i = 1`).
    gffs : (N,) array_like
        Frequency-averaged Gaunt factor for each species (dimensionless).

    Returns
    -------
    log_t : float or ~numpy.ndarray
        Natural logarithm of the cooling time :math:`t_{\rm cool}` [s].

    Raises
    ------
    ValueError
        If ``Zs``, ``Xs``, and ``gffs`` do not all have the same 1-D shape.

    See Also
    --------
    _log_ff_cooling_time : Single-species variant.
    """
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)
    gffs = np.asarray(gffs, dtype=float)

    if not (Zs.shape == Xs.shape == gffs.shape):
        raise ValueError(f"Zs, Xs, and gffs must have the same shape; got {Zs.shape}, {Xs.shape}, {gffs.shape}.")
    if Zs.ndim != 1:
        raise ValueError(f"Zs, Xs, and gffs must be 1-D arrays; got shape {Zs.shape}.")

    g_ff_eff = np.sum(Zs**2 * Xs * gffs)
    return _log_ff_cooling_time(log_n_i, log_T, 1.0, g_ff_eff)


# =============================================== #
# Public Functions for Free-Free
# =============================================== #
def compute_ff_emissivity(
    nu: "Union[float, _UnitBearingArrayLike]",
    n_e: "Union[float, _UnitBearingArrayLike]",
    n_i: "Union[float, _UnitBearingArrayLike]",
    Z: "Union[float, np.ndarray]",
    T: "Union[float, _UnitBearingArrayLike]",
    g_ff: "Union[float, np.ndarray, None]" = None,
    approx: str = "lu",
) -> u.Quantity:
    r"""
    Thermal free–free (bremsstrahlung) spectral emissivity.

    Computes the monochromatic bremsstrahlung emissivity of a fully-ionized
    thermal plasma using the Rybicki & Lightman formula:

    .. math::

        j_\nu =
        C_{ff}\,
        Z^2\, n_e\, n_i\,
        T^{-1/2}\,
        e^{-h\nu / k_B T}\,
        g_{ff}

    where :math:`C_{ff} = 6.8 \times 10^{-38}` CGS (integrated over the
    solid angle :math:`4\pi` for isotropic emission) and :math:`g_{ff}` is the
    velocity-averaged free–free Gaunt factor.

    Parameters
    ----------
    nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Photon frequency :math:`\nu`.  Bare floats are assumed to be in **Hz**.
    n_e : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron number density :math:`n_e`.
        Bare floats are assumed to be in **cm**\ :sup:`-3`.
    n_i : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Ion number density :math:`n_i`.
        Bare floats are assumed to be in **cm**\ :sup:`-3`.
    Z : float or ~numpy.ndarray
        Mean ionic charge (dimensionless).  For hydrogen plasma, ``Z = 1``.
    T : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron temperature :math:`T`.  Bare floats are assumed to be in **K**.
    g_ff : float, ~numpy.ndarray, or None, optional
        Free–free Gaunt factor (dimensionless).  If ``None`` (default), the
        Gaunt factor is computed automatically via ``approx``.  Pass a scalar
        or array to override.
    approx : str, optional
        Approximation method forwarded to
        :func:`~.gaunt_factor.compute_ff_gaunt_factor` when ``g_ff`` is
        ``None``.  Default is ``'lu'``.

    Returns
    -------
    j_nu : ~astropy.units.Quantity
        Monochromatic emissivity
        :math:`j_\nu` [erg s\ :sup:`-1` cm\ :sup:`-3` Hz\ :sup:`-1` sr\ :sup:`-1`].

    Notes
    -----
    The Gaunt factor :math:`g_{ff}` varies weakly with frequency and
    temperature; :footcite:t:`RybickiLightman` tabulate values and discuss
    limiting forms.  For many practical applications a constant value
    :math:`g_{ff} \approx 1`–\ :math:`5` is adequate.

    The computation is performed in log-space via :func:`_log_ff_emissivity` to
    ensure numerical stability at extreme frequencies or densities.

    Examples
    --------
    Emissivity of a hot coronal plasma at 1 GHz:

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.core import (
            compute_ff_emissivity,
        )

        j = compute_ff_emissivity(
            nu=1e9 * u.Hz,
            n_e=1e3 * u.cm**-3,
            n_i=1e3 * u.cm**-3,
            Z=1,
            T=1e6 * u.K,
        )
        print(j.to(u.erg / u.s / u.cm**3 / u.Hz / u.sr))

    Vectorised over a frequency array:

    .. code-block:: python

        nu_arr = np.geomspace(1e8, 1e15, 200) * u.Hz
        j_arr = compute_ff_emissivity(
            nu=nu_arr,
            n_e=1e4 * u.cm**-3,
            n_i=1e4 * u.cm**-3,
            Z=1,
            T=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_absorption : Free–free absorption coefficient for the same plasma.
    compute_ff_RJ_emissivity : Fast Rayleigh–Jeans approximation (:math:`h\nu \ll k_B T`).
    """
    nu = ensure_in_units(nu, u.Hz)
    n_e = ensure_in_units(n_e, u.cm**-3)
    n_i = ensure_in_units(n_i, u.cm**-3)
    T = ensure_in_units(T, u.K)

    if g_ff is None:
        g_ff = compute_ff_gaunt_factor(nu, T, Z, approx=approx)

    log_j = _log_ff_emissivity(
        np.log(nu),
        np.log(n_e),
        np.log(n_i),
        Z,
        np.log(T),
        g_ff,
    )

    return np.exp(log_j) * u.erg / (u.s * u.cm**3 * u.Hz * u.sr)


def compute_ff_absorption(
    nu: "Union[float, _UnitBearingArrayLike]",
    n_e: "Union[float, _UnitBearingArrayLike]",
    n_i: "Union[float, _UnitBearingArrayLike]",
    Z: "Union[float, np.ndarray]",
    T: "Union[float, _UnitBearingArrayLike]",
    g_ff: "Union[float, np.ndarray, None]" = None,
    approx: str = "lu",
) -> u.Quantity:
    r"""
    Thermal free–free (bremsstrahlung) absorption coefficient.

    Computes the monochromatic free–free absorption coefficient:

    .. math::

        \alpha_\nu =
        C_\alpha\,
        Z^2\, n_e\, n_i\,
        T^{-1/2}\,
        \nu^{-3}\,
        \bigl(1 - e^{-h\nu / k_B T}\bigr)\,
        g_{ff}

    where :math:`C_\alpha = 3.7 \times 10^8` CGS.  The factor
    :math:`(1 - e^{-h\nu/k_B T})` encodes detailed balance and approaches
    :math:`h\nu / k_B T` in the Rayleigh–Jeans limit and unity in the Wien
    limit.

    The **free–free optical depth** along a line of sight of depth :math:`L` is

    .. math::

        \tau_{\rm ff} = \int_0^L \alpha_\nu(l)\, dl \approx \alpha_\nu\, L

    so sources with :math:`\tau_{\rm ff} \gg 1` appear as optically thick
    emitters.

    Parameters
    ----------
    nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Photon frequency :math:`\nu`.  Bare floats are assumed to be in **Hz**.
    n_e : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron number density :math:`n_e`.
        Bare floats are assumed to be in **cm**\ :sup:`-3`.
    n_i : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Ion number density :math:`n_i`.
        Bare floats are assumed to be in **cm**\ :sup:`-3`.
    Z : float or ~numpy.ndarray
        Mean ionic charge (dimensionless).
    T : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron temperature :math:`T`.  Bare floats are assumed to be in **K**.
    g_ff : float, ~numpy.ndarray, or None, optional
        Free–free Gaunt factor (dimensionless).  If ``None`` (default),
        computed automatically via ``approx``.
    approx : str, optional
        Approximation method forwarded to
        :func:`~.gaunt_factor.compute_ff_gaunt_factor` when ``g_ff`` is
        ``None``.  Default is ``'lu'``.

    Returns
    -------
    alpha_nu : ~astropy.units.Quantity
        Absorption coefficient :math:`\alpha_\nu` [cm\ :sup:`-1`].

    Notes
    -----
    The stimulated-emission factor :math:`(1 - e^{-h\nu/k_B T})` is evaluated
    via :func:`numpy.log1p` in the log-space backend to avoid catastrophic
    cancellation at low frequencies where :math:`h\nu / k_B T \ll 1`.

    Use :func:`compute_ff_RJ_absorption` when the Rayleigh–Jeans condition
    :math:`h\nu \ll k_B T` is firmly satisfied, as it is both faster and
    avoids the numerical subtlety entirely.

    Examples
    --------
    Absorption coefficient of a dense, cool nebula at 5 GHz:

    .. code-block:: python

        from astropy import units as u
        from triceratops.radiation.free_free.core import (
            compute_ff_absorption,
        )

        alpha = compute_ff_absorption(
            nu=5e9 * u.Hz,
            n_e=1e4 * u.cm**-3,
            n_i=1e4 * u.cm**-3,
            Z=1,
            T=1e4 * u.K,
        )
        print(alpha.to(u.cm**-1))

    Estimating the turnover frequency where :math:`\alpha_\nu L = 1`:

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.core import (
            compute_ff_absorption,
        )

        nu_arr = np.geomspace(1e8, 1e12, 300) * u.Hz
        alpha_arr = compute_ff_absorption(
            nu=nu_arr,
            n_e=1e5 * u.cm**-3,
            n_i=1e5 * u.cm**-3,
            Z=1,
            T=1e4 * u.K,
        )
        L = 1e17 * u.cm  # depth of the shell
        tau = (alpha_arr * L).decompose()

    See Also
    --------
    compute_ff_emissivity : Corresponding spectral emissivity.
    compute_ff_RJ_absorption : Rayleigh–Jeans approximation (:math:`h\nu \ll k_B T`).
    """
    nu = ensure_in_units(nu, u.Hz)
    n_e = ensure_in_units(n_e, u.cm**-3)
    n_i = ensure_in_units(n_i, u.cm**-3)
    T = ensure_in_units(T, u.K)

    if g_ff is None:
        g_ff = compute_ff_gaunt_factor(nu, T, Z, approx=approx)

    log_alpha = _log_ff_absorption(
        np.log(nu),
        np.log(n_e),
        np.log(n_i),
        Z,
        np.log(T),
        g_ff,
    )

    return np.exp(log_alpha) / u.cm


def compute_ff_RJ_emissivity(
    nu: "Union[float, _UnitBearingArrayLike]",
    n_e: "Union[float, _UnitBearingArrayLike]",
    n_i: "Union[float, _UnitBearingArrayLike]",
    Z: "Union[float, np.ndarray]",
    T: "Union[float, _UnitBearingArrayLike]",
    g_ff: "Union[float, np.ndarray, None]" = None,
    approx: str = "draine",
) -> u.Quantity:
    r"""
    Rayleigh–Jeans limit free–free spectral emissivity.

    Valid when :math:`h\nu \ll k_B T` (i.e.\ radio through sub-mm frequencies
    for plasmas with :math:`T \gtrsim 10^4` K), this approximation drops the
    exponential suppression factor:

    .. math::

        j_\nu^{\rm RJ} =
        C_{ff}\,
        Z^2\, n_e\, n_i\,
        T^{-1/2}\,
        g_{ff}

    The emissivity is then **frequency-independent** (flat spectrum) and scales
    as :math:`T^{-1/2}`.  This is the appropriate form for fast evaluations of
    radio bremsstrahlung in hydrodynamical simulations or radiative transfer
    codes.

    Parameters
    ----------
    nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Photon frequency :math:`\nu`.  Accepted for interface consistency with
        :func:`compute_ff_emissivity` but **not used** in the computation.
        Bare floats are assumed to be in **Hz**.
    n_e : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron number density :math:`n_e`.
        Bare floats are assumed to be in **cm**\ :sup:`-3`.
    n_i : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Ion number density :math:`n_i`.
        Bare floats are assumed to be in **cm**\ :sup:`-3`.
    Z : float or ~numpy.ndarray
        Mean ionic charge (dimensionless).
    T : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron temperature :math:`T`.  Bare floats are assumed to be in **K**.
    g_ff : float, ~numpy.ndarray, or None, optional
        Free–free Gaunt factor (dimensionless).  If ``None`` (default),
        computed automatically using the frequency-averaged approximation
        selected by ``approx``.
    approx : str, optional
        Approximation method forwarded to
        :func:`~.gaunt_factor.compute_mean_ff_gaunt_factor` when ``g_ff`` is
        ``None``.  Default is ``'draine'``.

    Returns
    -------
    j_nu : ~astropy.units.Quantity
        Emissivity
        :math:`j_\nu^{\rm RJ}` [erg s\ :sup:`-1` cm\ :sup:`-3` Hz\ :sup:`-1` sr\ :sup:`-1`].

    Notes
    -----
    The Rayleigh–Jeans condition is satisfied when

    .. math::

        \frac{h\nu}{k_B T} \ll 1
        \quad\Longleftrightarrow\quad
        \nu \ll \frac{k_B T}{h}
        \approx 2.08 \times 10^{10}\,\frac{T}{1\,\mathrm{K}}\;\mathrm{Hz}

    For :math:`T = 10^4` K this limits the approximation to
    :math:`\nu \lesssim 2 \times 10^{14}` Hz.

    Examples
    --------
    Compare the exact and RJ emissivities to verify the approximation:

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.core import (
            compute_ff_emissivity,
            compute_ff_RJ_emissivity,
        )

        nu = (
            1e9 * u.Hz
        )  # 1 GHz — safely in the RJ regime for T = 1e4 K
        n_e = n_i = 1e3 * u.cm**-3
        T = 1e4 * u.K

        j_exact = compute_ff_emissivity(
            nu=nu, n_e=n_e, n_i=n_i, Z=1, T=T
        )
        j_rj = compute_ff_RJ_emissivity(
            nu=nu, n_e=n_e, n_i=n_i, Z=1, T=T
        )

        print(f"Exact : {j_exact:.3e}")
        print(f"RJ    : {j_rj:.3e}")
        print(
            f"Ratio : {(j_rj / j_exact).decompose():.6f}"
        )  # should be ≈ 1

    See Also
    --------
    compute_ff_emissivity : Exact emissivity valid at all frequencies.
    compute_ff_RJ_absorption : Matching RJ absorption coefficient.
    """
    nu = ensure_in_units(nu, u.Hz)
    n_e = ensure_in_units(n_e, u.cm**-3)
    n_i = ensure_in_units(n_i, u.cm**-3)
    T = ensure_in_units(T, u.K)

    if g_ff is None:
        g_ff = compute_mean_ff_gaunt_factor(T, Z, approx=approx)

    log_j = _log_ff_RJ_emissivity(
        np.log(nu),
        np.log(n_e),
        np.log(n_i),
        Z,
        np.log(T),
        g_ff,
    )

    return np.exp(log_j) * u.erg / (u.s * u.cm**3 * u.Hz * u.sr)


def compute_ff_RJ_absorption(
    nu: "Union[float, _UnitBearingArrayLike]",
    n_e: "Union[float, _UnitBearingArrayLike]",
    n_i: "Union[float, _UnitBearingArrayLike]",
    Z: "Union[float, np.ndarray]",
    T: "Union[float, _UnitBearingArrayLike]",
    g_ff: "Union[float, np.ndarray, None]" = None,
    approx: str = "draine",
) -> u.Quantity:
    r"""
    Rayleigh–Jeans limit free–free absorption coefficient.

    Valid when :math:`h\nu \ll k_B T`, the stimulated-emission correction
    simplifies to :math:`(1 - e^{-h\nu/k_B T}) \approx h\nu / k_B T` and
    the absorption coefficient becomes

    .. math::

        \alpha_\nu^{\rm RJ} \propto
        Z^2\, n_e\, n_i\,
        T^{-3/2}\,
        \nu^{-2}\,
        g_{ff}

    This is the standard **radio bremsstrahlung opacity** scaling: steep
    frequency dependence (:math:`\nu^{-2}`) and a stronger inverse temperature
    dependence (:math:`T^{-3/2}`) than the exact form.  HII regions, radio
    supernovae shells, and cool stellar coronae are often modelled with this
    approximation.

    Parameters
    ----------
    nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Photon frequency :math:`\nu`.  Bare floats are assumed to be in **Hz**.
    n_e : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron number density :math:`n_e`.
        Bare floats are assumed to be in **cm**\ :sup:`-3`.
    n_i : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Ion number density :math:`n_i`.
        Bare floats are assumed to be in **cm**\ :sup:`-3`.
    Z : float or ~numpy.ndarray
        Mean ionic charge (dimensionless).
    T : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron temperature :math:`T`.  Bare floats are assumed to be in **K**.
    g_ff : float, ~numpy.ndarray, or None, optional
        Free–free Gaunt factor (dimensionless).  If ``None`` (default),
        computed automatically using the frequency-averaged approximation
        selected by ``approx``.
    approx : str, optional
        Approximation method forwarded to
        :func:`~.gaunt_factor.compute_mean_ff_gaunt_factor` when ``g_ff`` is
        ``None``.  Default is ``'draine'``.

    Returns
    -------
    alpha_nu : ~astropy.units.Quantity
        Absorption coefficient
        :math:`\alpha_\nu^{\rm RJ}` [cm\ :sup:`-1`].

    Notes
    -----
    The free–free **turnover frequency** :math:`\nu_{\rm ff}` at which
    :math:`\tau_{\rm ff} = \alpha_\nu^{\rm RJ} L = 1` scales as

    .. math::

        \nu_{\rm ff} \propto \bigl(n_e^2\, L\, T^{-3/2}\bigr)^{1/2}

    (the emission measure :math:`{\rm EM} = n_e^2 L` sets the optical depth).

    Examples
    --------
    Absorption in a supernova radio shell at 1 GHz:

    .. code-block:: python

        from astropy import units as u
        from triceratops.radiation.free_free.core import (
            compute_ff_RJ_absorption,
        )

        alpha = compute_ff_RJ_absorption(
            nu=1e9 * u.Hz,
            n_e=1e5 * u.cm**-3,
            n_i=1e5 * u.cm**-3,
            Z=1,
            T=1e4 * u.K,
        )
        print(alpha.to(u.cm**-1))

    Frequency dependence of the optical depth for a fixed shell:

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.core import (
            compute_ff_RJ_absorption,
        )

        nu_arr = np.geomspace(1e8, 1e11, 200) * u.Hz
        alpha_arr = compute_ff_RJ_absorption(
            nu=nu_arr,
            n_e=1e4 * u.cm**-3,
            n_i=1e4 * u.cm**-3,
            Z=1,
            T=1e4 * u.K,
        )
        L_shell = 1e16 * u.cm
        tau_arr = (
            alpha_arr * L_shell
        ).decompose()  # optical depth vs frequency

    See Also
    --------
    compute_ff_absorption : Exact absorption valid at all frequencies.
    compute_ff_RJ_emissivity : Matching RJ emissivity.
    """
    nu = ensure_in_units(nu, u.Hz)
    n_e = ensure_in_units(n_e, u.cm**-3)
    n_i = ensure_in_units(n_i, u.cm**-3)
    T = ensure_in_units(T, u.K)

    if g_ff is None:
        g_ff = compute_mean_ff_gaunt_factor(T, Z, approx=approx)

    log_alpha = _log_ff_RJ_absorption(
        np.log(nu),
        np.log(n_e),
        np.log(n_i),
        Z,
        np.log(T),
        g_ff,
    )

    return np.exp(log_alpha) / u.cm


# =============================================== #
# Composition-weighted public wrappers            #
# =============================================== #
def compute_ff_emissivity_comp(
    nu: "Union[float, _UnitBearingArrayLike]",
    n_e: "Union[float, _UnitBearingArrayLike]",
    n_i: "Union[float, _UnitBearingArrayLike]",
    T: "Union[float, _UnitBearingArrayLike]",
    Zs: "_ArrayLike",
    Xs: "_ArrayLike",
    g_ffs: "Union[_ArrayLike, None]" = None,
    approx: str = "lu",
) -> u.Quantity:
    r"""Thermal free–free emissivity for a multi-species plasma.

    Computes

    .. math::

        j_\nu = C_{\rm ff}\, n_e\, n_i\, T^{-1/2}\, e^{-h\nu/k_B T}\,
                \sum_i Z_i^2\, x_i\, g_{{\rm ff},i}

    Parameters
    ----------
    nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Photon frequency :math:`\nu`.  Bare floats assumed to be in **Hz**.
    n_e : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron number density.  Bare floats assumed to be in **cm**\ :sup:`-3`.
    n_i : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Total ion number density.  Bare floats assumed to be in **cm**\ :sup:`-3`.
    T : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron temperature.  Bare floats assumed to be in **K**.
    Zs : ~numpy.ndarray
        Ionic charge numbers of each species.
    Xs : ~numpy.ndarray
        Number fractions of each species (:math:`\sum_i x_i = 1`).
    g_ffs : ~numpy.ndarray or None, optional
        Pre-computed per-species Gaunt factors.  If ``None`` (default),
        computed via ``approx``.
    approx : str, optional
        Forwarded to :func:`~.gaunt_factor.compute_ff_gaunt_factor_comp`
        when ``g_ffs`` is ``None``.  Default is ``'lu'``.

    Returns
    -------
    j_nu : ~astropy.units.Quantity
        Emissivity [erg s\ :sup:`-1` cm\ :sup:`-3` Hz\ :sup:`-1` sr\ :sup:`-1`].

    See Also
    --------
    compute_ff_emissivity : Single-species variant.
    compute_ff_absorption_comp : Composition-weighted absorption coefficient.
    compute_ff_RJ_emissivity_comp : Rayleigh–Jeans variant.
    """
    nu = ensure_in_units(nu, u.Hz)
    n_e = ensure_in_units(n_e, u.cm**-3)
    n_i = ensure_in_units(n_i, u.cm**-3)
    T = ensure_in_units(T, u.K)
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)

    if g_ffs is None:
        g_ff_eff = compute_ff_gaunt_factor_comp(nu, T, Zs, Xs, approx=approx)
    else:
        g_ffs = np.asarray(g_ffs, dtype=float)
        g_ff_eff = np.sum(Zs**2 * Xs * g_ffs)

    log_j = _log_ff_emissivity(np.log(nu), np.log(n_e), np.log(n_i), 1.0, np.log(T), g_ff_eff)
    return np.exp(log_j) * u.erg / (u.s * u.cm**3 * u.Hz * u.sr)


def compute_ff_absorption_comp(
    nu: "Union[float, _UnitBearingArrayLike]",
    n_e: "Union[float, _UnitBearingArrayLike]",
    n_i: "Union[float, _UnitBearingArrayLike]",
    T: "Union[float, _UnitBearingArrayLike]",
    Zs: "_ArrayLike",
    Xs: "_ArrayLike",
    g_ffs: "Union[_ArrayLike, None]" = None,
    approx: str = "lu",
) -> u.Quantity:
    r"""Thermal free–free absorption coefficient for a multi-species plasma.

    Computes

    .. math::

        \alpha_\nu = C_\alpha\, n_e\, n_i\, T^{-1/2}\, \nu^{-3}\,
                     (1 - e^{-h\nu/k_B T})\,
                     \sum_i Z_i^2\, x_i\, g_{{\rm ff},i}

    Parameters
    ----------
    nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Photon frequency :math:`\nu`.  Bare floats assumed to be in **Hz**.
    n_e : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron number density.  Bare floats assumed to be in **cm**\ :sup:`-3`.
    n_i : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Total ion number density.  Bare floats assumed to be in **cm**\ :sup:`-3`.
    T : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron temperature.  Bare floats assumed to be in **K**.
    Zs : ~numpy.ndarray
        Ionic charge numbers of each species.
    Xs : ~numpy.ndarray
        Number fractions of each species (:math:`\sum_i x_i = 1`).
    g_ffs : ~numpy.ndarray or None, optional
        Pre-computed per-species Gaunt factors.  If ``None`` (default),
        computed via ``approx``.
    approx : str, optional
        Forwarded to :func:`~.gaunt_factor.compute_ff_gaunt_factor_comp`
        when ``g_ffs`` is ``None``.  Default is ``'lu'``.

    Returns
    -------
    alpha_nu : ~astropy.units.Quantity
        Absorption coefficient [cm\ :sup:`-1`].

    See Also
    --------
    compute_ff_absorption : Single-species variant.
    compute_ff_emissivity_comp : Composition-weighted emissivity.
    compute_ff_RJ_absorption_comp : Rayleigh–Jeans variant.
    """
    nu = ensure_in_units(nu, u.Hz)
    n_e = ensure_in_units(n_e, u.cm**-3)
    n_i = ensure_in_units(n_i, u.cm**-3)
    T = ensure_in_units(T, u.K)
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)

    if g_ffs is None:
        g_ff_eff = compute_ff_gaunt_factor_comp(nu, T, Zs, Xs, approx=approx)
    else:
        g_ffs = np.asarray(g_ffs, dtype=float)
        g_ff_eff = np.sum(Zs**2 * Xs * g_ffs)

    log_alpha = _log_ff_absorption(np.log(nu), np.log(n_e), np.log(n_i), 1.0, np.log(T), g_ff_eff)
    return np.exp(log_alpha) / u.cm


def compute_ff_RJ_emissivity_comp(
    nu: "Union[float, _UnitBearingArrayLike]",
    n_e: "Union[float, _UnitBearingArrayLike]",
    n_i: "Union[float, _UnitBearingArrayLike]",
    T: "Union[float, _UnitBearingArrayLike]",
    Zs: "_ArrayLike",
    Xs: "_ArrayLike",
    g_ffs: "Union[_ArrayLike, None]" = None,
    approx: str = "draine",
) -> u.Quantity:
    r"""Rayleigh–Jeans free–free emissivity for a multi-species plasma.

    Rayleigh–Jeans limit (:math:`h\nu \ll k_B T`) of
    :func:`compute_ff_emissivity_comp`.  Uses the frequency-averaged Gaunt
    factor :math:`\bar{g}_{\rm ff}` by default.

    Parameters
    ----------
    nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Photon frequency :math:`\nu`.  Bare floats assumed to be in **Hz**.
    n_e : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron number density.  Bare floats assumed to be in **cm**\ :sup:`-3`.
    n_i : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Total ion number density.  Bare floats assumed to be in **cm**\ :sup:`-3`.
    T : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron temperature.  Bare floats assumed to be in **K**.
    Zs : ~numpy.ndarray
        Ionic charge numbers of each species.
    Xs : ~numpy.ndarray
        Number fractions of each species (:math:`\sum_i x_i = 1`).
    g_ffs : ~numpy.ndarray or None, optional
        Pre-computed per-species Gaunt factors.  If ``None`` (default),
        computed via the frequency-averaged ``approx``.
    approx : str, optional
        Forwarded to :func:`~.gaunt_factor.compute_mean_ff_gaunt_factor_comp`
        when ``g_ffs`` is ``None``.  Default is ``'draine'``.

    Returns
    -------
    j_nu : ~astropy.units.Quantity
        RJ emissivity [erg s\ :sup:`-1` cm\ :sup:`-3` Hz\ :sup:`-1` sr\ :sup:`-1`].

    See Also
    --------
    compute_ff_RJ_emissivity : Single-species RJ emissivity.
    compute_ff_emissivity_comp : Exact composition-weighted emissivity.
    compute_ff_RJ_absorption_comp : Composition-weighted RJ absorption.
    """
    nu = ensure_in_units(nu, u.Hz)
    n_e = ensure_in_units(n_e, u.cm**-3)
    n_i = ensure_in_units(n_i, u.cm**-3)
    T = ensure_in_units(T, u.K)
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)

    if g_ffs is None:
        g_ff_eff = compute_mean_ff_gaunt_factor_comp(T, Zs, Xs, approx=approx)
    else:
        g_ffs = np.asarray(g_ffs, dtype=float)
        g_ff_eff = np.sum(Zs**2 * Xs * g_ffs)

    log_j = _log_ff_RJ_emissivity(np.log(nu), np.log(n_e), np.log(n_i), 1.0, np.log(T), g_ff_eff)
    return np.exp(log_j) * u.erg / (u.s * u.cm**3 * u.Hz * u.sr)


def compute_ff_RJ_absorption_comp(
    nu: "Union[float, _UnitBearingArrayLike]",
    n_e: "Union[float, _UnitBearingArrayLike]",
    n_i: "Union[float, _UnitBearingArrayLike]",
    T: "Union[float, _UnitBearingArrayLike]",
    Zs: "_ArrayLike",
    Xs: "_ArrayLike",
    g_ffs: "Union[_ArrayLike, None]" = None,
    approx: str = "draine",
) -> u.Quantity:
    r"""Rayleigh–Jeans free–free absorption coefficient for a multi-species plasma.

    Rayleigh–Jeans limit (:math:`h\nu \ll k_B T`) of
    :func:`compute_ff_absorption_comp`.  Uses the frequency-averaged Gaunt
    factor :math:`\bar{g}_{\rm ff}` by default.

    Parameters
    ----------
    nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Photon frequency :math:`\nu`.  Bare floats assumed to be in **Hz**.
    n_e : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron number density.  Bare floats assumed to be in **cm**\ :sup:`-3`.
    n_i : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Total ion number density.  Bare floats assumed to be in **cm**\ :sup:`-3`.
    T : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron temperature.  Bare floats assumed to be in **K**.
    Zs : ~numpy.ndarray
        Ionic charge numbers of each species.
    Xs : ~numpy.ndarray
        Number fractions of each species (:math:`\sum_i x_i = 1`).
    g_ffs : ~numpy.ndarray or None, optional
        Pre-computed per-species Gaunt factors.  If ``None`` (default),
        computed via the frequency-averaged ``approx``.
    approx : str, optional
        Forwarded to :func:`~.gaunt_factor.compute_mean_ff_gaunt_factor_comp`
        when ``g_ffs`` is ``None``.  Default is ``'draine'``.

    Returns
    -------
    alpha_nu : ~astropy.units.Quantity
        RJ absorption coefficient [cm\ :sup:`-1`].

    See Also
    --------
    compute_ff_RJ_absorption : Single-species RJ absorption.
    compute_ff_absorption_comp : Exact composition-weighted absorption.
    compute_ff_RJ_emissivity_comp : Composition-weighted RJ emissivity.
    """
    nu = ensure_in_units(nu, u.Hz)
    n_e = ensure_in_units(n_e, u.cm**-3)
    n_i = ensure_in_units(n_i, u.cm**-3)
    T = ensure_in_units(T, u.K)
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)

    if g_ffs is None:
        g_ff_eff = compute_mean_ff_gaunt_factor_comp(T, Zs, Xs, approx=approx)
    else:
        g_ffs = np.asarray(g_ffs, dtype=float)
        g_ff_eff = np.sum(Zs**2 * Xs * g_ffs)

    log_alpha = _log_ff_RJ_absorption(np.log(nu), np.log(n_e), np.log(n_i), 1.0, np.log(T), g_ff_eff)
    return np.exp(log_alpha) / u.cm
