r"""
Analytic Closure Relations for Synchrotron SED Inversion.

This module provides **analytic, literature-based closure relations**
for mapping phenomenological synchrotron spectral parameters
(e.g., break frequencies and flux normalizations) to approximate
physical quantities such as magnetic field strength and emitting radius.

These relations are **not fundamental** to the SED framework. They are
implemented strictly as convenience utilities for:

- quick-look parameter estimation,
- reproduction of commonly used literature scalings,
- initializing forward-model inference,
- sanity checks against published analytic results.

Philosophical Context
---------------------

The synchrotron SED classes in :mod:`radiation.synchrotron.SEDs` are
intentionally implemented in a **scale-free** manner. In general, a
given set of spectral break frequencies and flux normalizations does
*not* uniquely determine physical parameters without additional
assumptions.

To obtain physical quantities from spectral parameters, one must adopt
a **closure relation**, such as:

- equipartition between electron and magnetic energy,
- a prescribed filling factor,
- a specific electron energy distribution,
- or a particular geometric configuration.

The functions in this module encode such assumptions explicitly.
They should therefore be understood as:

    • effective relations
    • assumption-dependent mappings
    • valid only within specific spectral regimes

They are not universally applicable and should not be used when a
self-consistent dynamical model is available.

Intended Use Cases
------------------

These utilities are most appropriate when:

- The spectrum exhibits a clear synchrotron self-absorption (SSA)
  turnover.
- A broken power-law description is adequate.
- One wishes to reproduce standard analytic estimates from the
  literature (e.g., DeMarchi+22).
- Full forward modeling is unnecessary or computationally undesirable.

When performing rigorous inference, users are encouraged instead to:

1. Define a physical model (e.g., shock dynamics, energy injection).
2. Map physical parameters → SED parameters.
3. Perform forward modeling through the SED evaluation pipeline.

Scope and Limitations
---------------------

- These relations assume specific spectral orderings and asymptotic
  behavior.
- They typically require :math:`p \neq 2`.
- They rely on equipartition-like assumptions and fixed geometry.
- They may break down outside the regime in which the original
  analytic derivation was performed.

Because these formulas often involve large exponents and subtractive
cancellation, they are implemented internally in logarithmic space
for numerical stability.
"""

from typing import Union

import numpy as np
from astropy import units as u

from triceratops.utils.misc_utils import ensure_in_units

from ..constants import electron_rest_energy_cgs
from .utils import c_1_cgs, compute_c5_parameter, compute_c6_parameter


def _compute_ssa_BR_from_spectrum_dm22(
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


def compute_ssa_BR_from_spectrum_dm22(
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
    B, R = _compute_ssa_BR_from_spectrum_dm22(
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
