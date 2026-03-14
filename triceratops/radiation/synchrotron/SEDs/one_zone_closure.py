r"""
Public single-zone synchrotron inversion closures.

This module provides public and private inversion helpers that map
phenomenological synchrotron SED observables, such as
:math:`\nu_{\rm pk}` and :math:`F_{\nu,\rm pk}`, to inferred physical
source parameters such as the emitting radius :math:`R`, magnetic field
strength :math:`B`, and, in the implicit-cooling closures, the cooling
Lorentz factor :math:`\gamma_c`.

The implemented closures cover several increasingly specialized cases:

- the standard optically thin single-zone inversion,
- explicitly cooled optically thin inversions,
- synchrotron self-absorbed (SSA) inversions,
- combined cooling + SSA inversions,
- implicit synchrotron-cooling closures in which :math:`\gamma_c` is
  derived from the synchrotron cooling time rather than supplied as an
  independent input,
- the :footcite:t:`demarchiRadioAnalysisSN2004C2022` SSA inversion.

The low-level private helpers operate directly on logarithmic,
CGS-consistent quantities and perform no unit validation. The public
functions wrap those helpers with unit handling, cosmological distance
resolution, and minimal observer-to-comoving-frame transformations.

See Also
--------
:ref:`synch_sed_theory`
    Overview of the synchrotron SED formalism and spectral conventions.
:ref:`single_zone_sed_inversion`
    Documentation for the standard one-zone inversion framework.
:ref:`synchrotron_cooling_closure`
    Documentation for the implicit synchrotron-cooling closure.

References
----------
.. footbibliography::
"""

import warnings
from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import units as u

from triceratops.radiation.constants import electron_rest_energy_cgs
from triceratops.radiation.synchrotron.utils import (
    c_1_cgs,
    compute_c5_parameter,
    compute_c6_parameter,
)
from triceratops.utils.cosmology import get_cosmology, resolve_cosmological_distances
from triceratops.utils.misc_utils import ensure_in_units
from triceratops.utils.sr_utils import compute_doppler_factor

from ._one_zone_closure import (
    COOLING_INV_FUNCTION_REGISTRY,
    SSA_COOLING_INV_FUNCTION_REGISTRY,
    SSA_INV_FUNCTION_REGISTRY,
    _inv_log_powerlaw_sbpl_sed,
    _inv_log_powerlaw_sbpl_sed_implicit_cool_2,
    _inv_log_powerlaw_sbpl_sed_ssa_implicit_cool_4,
    _inv_log_powerlaw_sbpl_sed_ssa_implicit_cool_7,
)

# ====================================== #
# Static Type Checking
# ====================================== #
if TYPE_CHECKING:
    from astropy import cosmology as cosmo

    from triceratops._typing import _ArrayLike, _UnitBearingScalarLike

__all__ = [
    "invert_powerlaw_sed",
    "invert_powerlaw_cooling_sed",
    "invert_powerlaw_cooling_ssa_sed",
    "invert_powerlaw_ssa_sed",
    "invert_powerlaw_implicit_cooling_sed",
    "invert_powerlaw_ssa_sed_demarchi",
    "invert_powerlaw_implicit_cooling_ssa_sed",
    "_invert_powerlaw_sed",
    "_invert_powerlaw_cooling_sed",
    "_invert_powerlaw_cooling_ssa_sed",
    "_invert_powerlaw_ssa_sed",
    "_invert_powerlaw_implicit_cooling_sed",
    "_invert_powerlaw_ssa_sed_demarchi",
    "_invert_powerlaw_implicit_cooling_ssa_sed",
]


# ================================================ #
# PRIVATE DISPATCH FUNCTIONS
# ================================================ #
# These are the private dispatch functions which make up the low-level API without the
# component functions defined in the private module. These do not do any unit handling
# or cosmology. They are simply dispatchers for the various inversion functions based on the
# specified regime.
def _invert_powerlaw_sed(
    log_F_peak: "_ArrayLike",
    log_nu_peak: "_ArrayLike",
    log_D_L: "_ArrayLike",
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 0.5,
    gamma_bulk: float = 1.0,
    redshift: float = 0.0,
    alpha: float = None,
    pitch_average: bool = False,
):
    r"""
    Infer physical source parameters from an optically thin synchrotron peak.

    This low-level helper implements the default inverse closure for the
    single-zone, optically thin, non-SSA synchrotron SED. It maps the
    observed peak frequency and peak flux density to the emitting-region
    radius :math:`R` and magnetic field strength :math:`B` under the
    standard Triceratops one-zone closure assumptions.

    All inputs must already be expressed in logarithmic CGS units. No unit
    validation, cosmological distance resolution, or shape checking beyond
    what NumPy naturally provides is performed here. Observer-frame
    quantities are first transformed to the comoving frame using the same
    minimal on-axis Doppler and redshift corrections used throughout the
    single-zone inversion framework.

    This routine is intended as the internal implementation underlying the
    public function :func:`invert_powerlaw_sed`.

    Parameters
    ----------
    log_F_peak : array-like
        Natural logarithm of the observed peak flux density in
        ``erg cm^-2 s^-1 Hz^-1``.
    log_nu_peak : array-like
        Natural logarithm of the observed peak frequency in Hz.
    log_D_L : array-like
        Natural logarithm of the luminosity distance in cm.
    gamma_min : float, optional
        Minimum electron Lorentz factor, :math:`\gamma_{\min}`.
    gamma_max : float, optional
        Maximum electron Lorentz factor, :math:`\gamma_{\max}`.
    p : float, optional
        Power-law index of the electron distribution,
        :math:`N(\gamma) \propto \gamma^{-p}`.
    epsilon_E : float, optional
        Fraction of internal energy carried by relativistic electrons.
    epsilon_B : float, optional
        Fraction of internal energy stored in magnetic fields.
    f_V : float, optional
        Effective emitting-volume filling factor.
    gamma_bulk : float, optional
        Bulk Lorentz factor of the emitting region. A simple on-axis
        Doppler correction is applied before inversion.
    redshift : float, optional
        Source redshift. Used in the observer-to-comoving transformation.
    alpha : float, optional
        Fixed electron pitch angle in radians. Required when
        ``pitch_average=False``.
    pitch_average : bool, optional
        If ``True``, use the pitch-angle-averaged synchrotron coefficients.
        If ``False``, use a fixed pitch angle given by ``alpha``.

    Returns
    -------
    dict
        Dictionary of inferred physical parameters in linear CGS units with
        the keys

        - ``"R"`` : source radius in cm
        - ``"B"`` : magnetic field strength in G

    Notes
    -----
    This closure assumes that the fitted peak is the standard optically thin
    synchrotron peak associated with the injection Lorentz factor
    :math:`\gamma_{\min}`. It does **not** model synchrotron
    self-absorption, an explicit cooling break, multi-zone structure, or
    detailed relativistic transfer effects.

    .. dropdown:: Assumptions and interpretation
       :icon: info

       The inversion assumes

       - a homogeneous single emission zone,
       - a power-law electron distribution,
       - optically thin emission at the fitted peak,
       - the standard Triceratops microphysical closure,
       - no explicit SSA turnover shaping the peak,
       - no explicit cooling break shaping the peak,
       - effectively on-axis relativistic boosting when ``gamma_bulk > 1``.

       The returned values are therefore closure-dependent inferred
       parameters, not direct observables.

    .. dropdown:: Available regime
       :icon: chevron-down

       This function implements only the standard optically thin one-zone
       closure. For cooled, SSA, or implicit-cooling cases, use the
       corresponding specialized inversion helpers instead.

    See Also
    --------
    invert_powerlaw_sed
        Public, unit-aware wrapper for this inversion.
    _invert_powerlaw_cooling_sed
        Explicit-cooling optically thin inversion.
    _invert_powerlaw_ssa_sed
        SSA inversion without implicit cooling.
    _invert_powerlaw_cooling_ssa_sed
        Combined cooling + SSA inversion.
    _invert_powerlaw_implicit_cooling_sed
        Optically thin inversion with implicit synchrotron cooling closure.
    :ref:`single_zone_sed_inversion`
        Standard one-zone inversion documentation.
    :ref:`synch_sed_theory`
        Synchrotron SED conventions and theory.
    """
    # ------------------------------------------------------------------ #
    # Pitch-angle treatment
    # ------------------------------------------------------------------ #
    if pitch_average:
        if alpha is not None:
            warnings.warn(
                "Pitch averaging is enabled, so the provided `alpha` is ignored. "
                "(Set `alpha=None` to suppress this warning.)",
                stacklevel=2,
            )
        sin_alpha = None
    else:
        if alpha is None:
            raise ValueError("Pitch averaging is disabled, so a fixed pitch angle must be provided via `alpha`.")
        sin_alpha = np.sin(alpha)

    # ------------------------------------------------------------------ #
    # Transform observed peak quantities into the comoving frame.
    #
    # This applies the same minimal relativistic treatment used elsewhere
    # in the default closure: an on-axis Doppler correction plus the usual
    # cosmological redshift correction.
    #
    # IMPORTANT: this is intentionally simple and is not a substitute for a
    # full relativistic radiative-transfer treatment.
    # ------------------------------------------------------------------ #
    doppler_factor = compute_doppler_factor(gamma_bulk, 0.0)

    log_nu_peak_comoving = log_nu_peak - np.log(doppler_factor) + np.log1p(redshift)
    log_F_peak_comoving = log_F_peak - 3.0 * np.log(doppler_factor) + np.log1p(redshift)

    # ------------------------------------------------------------------ #
    # Invert the default closure in the comoving frame.
    # ------------------------------------------------------------------ #
    log_R, log_B = _inv_log_powerlaw_sbpl_sed(
        log_nu_peak_comoving,
        log_F_peak_comoving,
        log_D_L,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        p=p,
        epsilon_e=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        sin_alpha=sin_alpha,
    )

    return {
        "R": np.exp(log_R),
        "B": np.exp(log_B),
    }


def _invert_powerlaw_cooling_sed(
    regime: str,
    log_F_peak: "_ArrayLike",
    log_nu_peak: "_ArrayLike",
    log_D_L: "_ArrayLike",
    gamma_min: float = 1.0,
    gamma_c: float = np.inf,
    gamma_max: float = np.inf,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 0.5,
    gamma_bulk: float = 1.0,
    redshift: float = 0.0,
    alpha: float = None,
    pitch_average: bool = False,
):
    r"""
    Infer physical parameters for an explicitly cooled optically thin SED.

    This low-level helper implements the inverse closure for the optically
    thin single-zone synchrotron SED when the cooling regime is specified
    explicitly and the cooling Lorentz factor :math:`\gamma_c` is supplied
    as an input where required. It maps the observed peak frequency and peak
    flux density to the emitting radius :math:`R` and magnetic field
    strength :math:`B`.

    All inputs must already be expressed in logarithmic CGS units. The
    routine performs no unit validation and is intended for internal use by
    :func:`invert_powerlaw_cooling_sed`.

    Parameters
    ----------
    regime : str
        Name of the optically thin cooling regime to invert.
    log_F_peak : array-like
        Natural logarithm of the observed peak flux density in
        ``erg cm^-2 s^-1 Hz^-1``.
    log_nu_peak : array-like
        Natural logarithm of the observed peak frequency in Hz.
    log_D_L : array-like
        Natural logarithm of the luminosity distance in cm.
    gamma_min : float, optional
        Minimum electron Lorentz factor, :math:`\gamma_{\min}`.
    gamma_c : float, optional
        Cooling Lorentz factor, :math:`\gamma_c`. This must be finite in
        regimes where the cooling break explicitly enters the inversion.
    gamma_max : float, optional
        Maximum electron Lorentz factor, :math:`\gamma_{\max}`.
    p : float, optional
        Electron energy power-law index.
    epsilon_E : float, optional
        Fraction of internal energy in relativistic electrons.
    epsilon_B : float, optional
        Fraction of internal energy in magnetic fields.
    f_V : float, optional
        Effective emitting-volume filling factor.
    gamma_bulk : float, optional
        Bulk Lorentz factor used for the simple observer-to-comoving Doppler
        correction.
    redshift : float, optional
        Source redshift used in the observer-to-comoving transformation.
    alpha : float, optional
        Fixed pitch angle in radians when ``pitch_average=False``.
    pitch_average : bool, optional
        If ``True``, use pitch-angle-averaged synchrotron coefficients.

    Returns
    -------
    dict
        Dictionary containing

        - ``"R"`` : inferred source radius in cm
        - ``"B"`` : inferred magnetic field strength in G

    Notes
    -----
    This function handles the explicitly cooled, optically thin closures in
    which the spectral regime is known in advance. The chosen regime
    determines which analytic inversion is dispatched.

    .. dropdown:: Available regimes
       :icon: chevron-down

       The following regimes are currently supported through
       ``COOLING_INV_FUNCTION_REGISTRY``:

       - ``"fast_cooling"``:
         :math:`\nu_c < \nu_m < \nu_{\max}`
       - ``"slow_cooling"``:
         :math:`\nu_m < \nu_c < \nu_{\max}`
       - ``"no_cooling"``:
         :math:`\nu_m < \nu_{\max} < \nu_c`

       The user is responsible for supplying a regime consistent with the
       modeled spectrum.

    .. dropdown:: Relation to other closures
       :icon: info

       This routine assumes :math:`\gamma_c` is supplied explicitly when it
       matters. If instead :math:`\gamma_c` should be determined from the
       synchrotron cooling time, use the implicit-cooling closure described
       in :ref:`synchrotron_cooling_closure`.

    See Also
    --------
    invert_powerlaw_cooling_sed
        Public, unit-aware wrapper.
    _invert_powerlaw_sed
        Standard optically thin inversion without explicit cooling.
    _invert_powerlaw_cooling_ssa_sed
        Explicit-cooling inversion including SSA.
    _invert_powerlaw_implicit_cooling_sed
        Implicit synchrotron-cooling inversion.
    :ref:`single_zone_sed_inversion`
        Standard inversion framework.
    :ref:`synch_sed_theory`
        SED regime definitions and conventions.
    :ref:`synchrotron_cooling_closure`
        Implicit-cooling closure formalism.
    """
    # ------------------------------------------------------------------ #
    # Pitch-angle treatment
    # ------------------------------------------------------------------ #
    if pitch_average:
        if alpha is not None:
            warnings.warn(
                "Pitch averaging is enabled, so the provided `alpha` is ignored. "
                "(Set `alpha=None` to suppress this warning.)",
                stacklevel=2,
            )
        sin_alpha = None
    else:
        if alpha is None:
            raise ValueError("Pitch averaging is disabled, so a fixed pitch angle must be provided via `alpha`.")
        sin_alpha = np.sin(alpha)

    # ------------------------------------------------------------------ #
    # Handle the regime-specification
    # ------------------------------------------------------------------ #
    if regime not in COOLING_INV_FUNCTION_REGISTRY:
        raise ValueError(
            f"Unrecognized SED regime: {regime}. Available regimes: {list(COOLING_INV_FUNCTION_REGISTRY.keys())}"
        )

    inversion_function = COOLING_INV_FUNCTION_REGISTRY[regime]
    # ------------------------------------------------------------------ #
    # Transform observed peak quantities into the comoving frame.
    #
    # This applies the same minimal relativistic treatment used elsewhere
    # in the default closure: an on-axis Doppler correction plus the usual
    # cosmological redshift correction.
    #
    # IMPORTANT: this is intentionally simple and is not a substitute for a
    # full relativistic radiative-transfer treatment.
    # ------------------------------------------------------------------ #
    doppler_factor = compute_doppler_factor(gamma_bulk, 0.0)

    log_nu_peak_comoving = log_nu_peak - np.log(doppler_factor) + np.log1p(redshift)
    log_F_peak_comoving = log_F_peak - 3.0 * np.log(doppler_factor) + np.log1p(redshift)

    # ------------------------------------------------------------------ #
    # Invert the default closure in the comoving frame.
    # ------------------------------------------------------------------ #
    if regime == "fast_cooling":  # nu_c < nu_m < nu_max
        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            gamma_c=gamma_c,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            sin_alpha=sin_alpha,
        )
    elif regime == "slow_cooling":  # nu_m < nu_c < nu_max
        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            gamma_c=gamma_c,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            sin_alpha=sin_alpha,
        )
    elif regime == "no_cooling":  # nu_m < nu_max < nu_c
        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            sin_alpha=sin_alpha,
        )
    else:
        raise ValueError(
            f"Unrecognized SED regime: {regime}. Available regimes: {list(COOLING_INV_FUNCTION_REGISTRY.keys())}"
        )

    return {
        "R": np.exp(log_R),
        "B": np.exp(log_B),
    }


def _invert_powerlaw_cooling_ssa_sed(
    regime: str,
    log_F_peak: "_ArrayLike",
    log_nu_peak: "_ArrayLike",
    log_D_L: "_ArrayLike",
    log_D_A: "_ArrayLike" = None,
    gamma_min: float = 1.0,
    gamma_c: float = np.inf,
    gamma_max: float = np.inf,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 0.5,
    f_A: float = 0.5,
    gamma_bulk: float = 1.0,
    redshift: float = 0.0,
    alpha: float = None,
    pitch_average: bool = False,
):
    r"""
    Infer physical parameters for an explicitly cooled SSA synchrotron SED.

    This low-level helper implements the inverse closure for the
    single-zone synchrotron SED when synchrotron self-absorption (SSA) is
    present and the cooling regime is specified explicitly. Depending on the
    selected spectral regime, the inversion maps the observed peak
    quantities to the emitting radius :math:`R` and magnetic field strength
    :math:`B`, with the cooling Lorentz factor :math:`\gamma_c` supplied
    explicitly where required.

    All inputs must already be expressed in logarithmic CGS units. No unit
    validation is performed. This routine is the internal backend for
    :func:`invert_powerlaw_cooling_ssa_sed`.

    Parameters
    ----------
    regime : str
        Name of the SSA + cooling spectral regime to invert.
    log_F_peak : array-like
        Natural logarithm of the observed peak flux density in
        ``erg cm^-2 s^-1 Hz^-1``.
    log_nu_peak : array-like
        Natural logarithm of the observed peak frequency in Hz.
    log_D_L : array-like
        Natural logarithm of the luminosity distance in cm.
    log_D_A : array-like, optional
        Natural logarithm of the angular-diameter distance in cm. This is
        required for regimes in which the SSA peak depends explicitly on the
        projected angular area.
    gamma_min : float, optional
        Minimum electron Lorentz factor.
    gamma_c : float, optional
        Cooling Lorentz factor. Must be finite in regimes that explicitly
        depend on the cooling break.
    gamma_max : float, optional
        Maximum electron Lorentz factor.
    p : float, optional
        Electron power-law index.
    epsilon_E : float, optional
        Fraction of internal energy in relativistic electrons.
    epsilon_B : float, optional
        Fraction of internal energy in magnetic fields.
    f_V : float, optional
        Effective volume filling factor.
    f_A : float, optional
        Effective projected-area filling factor used in SSA regimes.
    gamma_bulk : float, optional
        Bulk Lorentz factor for the observer-to-comoving correction.
    redshift : float, optional
        Source redshift.
    alpha : float, optional
        Fixed pitch angle in radians when ``pitch_average=False``.
    pitch_average : bool, optional
        If ``True``, use pitch-angle-averaged synchrotron coefficients.

    Returns
    -------
    dict
        Dictionary containing

        - ``"R"`` : inferred radius in cm
        - ``"B"`` : inferred magnetic field strength in G

    Notes
    -----
    This helper handles the full explicitly cooled SSA inversion family.
    Different regimes require different auxiliary quantities; in particular,
    some regimes require ``log_D_A`` because the SSA peak depends on the
    projected source size.

    .. dropdown:: Available regimes
       :icon: chevron-down

       The following regimes are currently supported through
       ``SSA_COOLING_INV_FUNCTION_REGISTRY``:

       - ``"Spectrum1"``:
         :math:`\nu_a < \nu_m < \nu_{\max} < \nu_c`
       - ``"Spectrum2"``:
         :math:`\nu_m < \nu_a < \nu_{\max} < \nu_c`
       - ``"Spectrum3"``:
         :math:`\nu_a < \nu_m < \nu_c < \nu_{\max}`
       - ``"Spectrum4"``:
         :math:`\nu_m < \nu_a < \nu_c < \nu_{\max}`
       - ``"Spectrum5"``:
         :math:`\nu_a < \nu_c < \nu_m < \nu_{\max}`
       - ``"Spectrum6"``:
         :math:`\nu_c < \nu_a < \nu_m < \nu_{\max}`
       - ``"Spectrum7"``:
         :math:`\nu_c, \nu_m < \nu_a < \nu_{\max}`
       - ``"Spectrum8"``:
         :math:`\nu_c < \nu_a < \nu_m < \nu_{\max}`

       The supplied regime must be consistent with the intended SED
       interpretation.

    .. dropdown:: Distance requirements
       :icon: info

       Regimes whose closure depends on the SSA blackbody peak condition
       require an angular-diameter distance because the inversion depends on
       the apparent angular size of the source.

    See Also
    --------
    invert_powerlaw_cooling_ssa_sed
        Public, unit-aware wrapper.
    _invert_powerlaw_ssa_sed
        SSA inversion without explicit cooling.
    _invert_powerlaw_cooling_sed
        Explicit-cooling inversion without SSA.
    _invert_powerlaw_implicit_cooling_ssa_sed
        SSA inversion with implicit synchrotron cooling.
    :ref:`single_zone_sed_inversion`
        Standard inversion framework.
    :ref:`synch_sed_theory`
        SED regime definitions and conventions.
    :ref:`synchrotron_cooling_closure`
        Implicit synchrotron-cooling closure formalism.
    """
    # ------------------------------------------------------------------ #
    # Pitch-angle treatment
    # ------------------------------------------------------------------ #
    if pitch_average:
        if alpha is not None:
            warnings.warn(
                "Pitch averaging is enabled, so the provided `alpha` is ignored. "
                "(Set `alpha=None` to suppress this warning.)",
                stacklevel=2,
            )
        sin_alpha = None
    else:
        if alpha is None:
            raise ValueError("Pitch averaging is disabled, so a fixed pitch angle must be provided via `alpha`.")
        sin_alpha = np.sin(alpha)

    # ------------------------------------------------------------------ #
    # Handle the regime-specification
    # ------------------------------------------------------------------ #
    if regime not in SSA_COOLING_INV_FUNCTION_REGISTRY:
        raise ValueError(
            f"Unrecognized SED regime: {regime}. Available regimes: {list(SSA_COOLING_INV_FUNCTION_REGISTRY.keys())}"
        )

    inversion_function = SSA_COOLING_INV_FUNCTION_REGISTRY[regime]
    # ------------------------------------------------------------------ #
    # Transform observed peak quantities into the comoving frame.
    #
    # This applies the same minimal relativistic treatment used elsewhere
    # in the default closure: an on-axis Doppler correction plus the usual
    # cosmological redshift correction.
    #
    # IMPORTANT: this is intentionally simple and is not a substitute for a
    # full relativistic radiative-transfer treatment.
    # ------------------------------------------------------------------ #
    doppler_factor = compute_doppler_factor(gamma_bulk, 0.0)

    log_nu_peak_comoving = log_nu_peak - np.log(doppler_factor) + np.log1p(redshift)
    log_F_peak_comoving = log_F_peak - 3.0 * np.log(doppler_factor) + np.log1p(redshift)

    # ------------------------------------------------------------------ #
    # Invert the default closure in the comoving frame.
    # ------------------------------------------------------------------ #
    if regime == "Spectrum1":  # nu_a < nu_m < nu_max < nu_c.
        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            sin_alpha=sin_alpha,
        )
    elif regime == "Spectrum2":  # nu_m < nu_a < nu_max < nu_c.
        if log_D_A is None:
            raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum2 regime.")

        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            log_D_A,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            sin_alpha=sin_alpha,
        )
    elif regime == "Spectrum3":  # nu_a < nu_m < nu_c < nu_max.
        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            gamma_min=gamma_min,
            gamma_c=gamma_c,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            sin_alpha=sin_alpha,
        )
    elif regime == "Spectrum4":  # nu_m < nu_a < nu_c < nu_max.
        if log_D_A is None:
            raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum4 regime.")
        if not np.all(np.isfinite(gamma_c)):
            raise ValueError("Cooling Lorentz factor `gamma_c` must be finite for Spectrum4 regime.")

        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            log_D_A,
            gamma_min=gamma_min,
            gamma_c=gamma_c,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            sin_alpha=sin_alpha,
        )
    elif regime == "Spectrum5":  # nu_a < nu_c < nu_m < nu_max.
        if not np.all(np.isfinite(gamma_c)):
            raise ValueError("Cooling Lorentz factor `gamma_c` must be finite for Spectrum5 regime.")

        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            gamma_min=gamma_min,
            gamma_c=gamma_c,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            sin_alpha=sin_alpha,
        )
    elif regime == "Spectrum6":  # nu_c < nu_a < nu_m < nu_max.
        if log_D_A is None:
            raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum6 regime.")
        if not np.all(np.isfinite(gamma_c)):
            raise ValueError("Cooling Lorentz factor `gamma_c` must be finite for Spectrum6 regime.")

        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            log_D_A,
            gamma_min=gamma_min,
            gamma_c=gamma_c,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            sin_alpha=sin_alpha,
        )
    elif regime == "Spectrum7":  # nu_c, nu_m < nu_a < nu_max.
        if log_D_A is None:
            raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum7 regime.")
        if not np.all(np.isfinite(gamma_c)):
            raise ValueError("Cooling Lorentz factor `gamma_c` must be finite for Spectrum7 regime.")

        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            log_D_A,
            gamma_min=gamma_min,
            gamma_c=gamma_c,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            sin_alpha=sin_alpha,
        )
    elif regime == "Spectrum8":  # nu_c < nu_a < nu_m < nu_max.
        if log_D_A is None:
            raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum8 regime.")
        if not np.all(np.isfinite(gamma_c)):
            raise ValueError("Cooling Lorentz factor `gamma_c` must be finite for Spectrum8 regime.")

        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            log_D_A,
            gamma_min=gamma_min,
            gamma_c=gamma_c,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            sin_alpha=sin_alpha,
        )
    else:
        raise ValueError(
            f"Unrecognized SED regime: {regime}. Available regimes: {list(SSA_COOLING_INV_FUNCTION_REGISTRY.keys())}"
        )

    return {
        "R": np.exp(log_R),
        "B": np.exp(log_B),
    }


def _invert_powerlaw_ssa_sed(
    regime: str,
    log_F_peak: "_ArrayLike",
    log_nu_peak: "_ArrayLike",
    log_D_L: "_ArrayLike",
    log_D_A: "_ArrayLike" = None,
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 0.5,
    f_A: float = 0.5,
    gamma_bulk: float = 1.0,
    redshift: float = 0.0,
    alpha: float = None,
    pitch_average: bool = False,
):
    r"""
    Infer physical parameters for an SSA synchrotron SED without cooling.

    This low-level helper implements the inverse closure for the
    single-zone synchrotron SED when synchrotron self-absorption is present
    but the inversion does not require an explicit cooling closure. It maps
    the observed peak flux density and peak frequency to the inferred source
    radius :math:`R` and magnetic field strength :math:`B`.

    All inputs must already be expressed in logarithmic CGS units. This
    routine performs no unit handling and serves as the internal backend for
    :func:`invert_powerlaw_ssa_sed`.

    Parameters
    ----------
    regime : str
        Name of the SSA regime to invert.
    log_F_peak : array-like
        Natural logarithm of the observed peak flux density in
        ``erg cm^-2 s^-1 Hz^-1``.
    log_nu_peak : array-like
        Natural logarithm of the observed peak frequency in Hz.
    log_D_L : array-like
        Natural logarithm of the luminosity distance in cm.
    log_D_A : array-like, optional
        Natural logarithm of the angular-diameter distance in cm. Required
        for optically thick SSA inversions.
    gamma_min : float, optional
        Minimum electron Lorentz factor.
    gamma_max : float, optional
        Maximum electron Lorentz factor.
    p : float, optional
        Electron power-law index.
    epsilon_E : float, optional
        Fraction of internal energy in relativistic electrons.
    epsilon_B : float, optional
        Fraction of internal energy in magnetic fields.
    f_V : float, optional
        Effective volume filling factor.
    f_A : float, optional
        Effective projected-area filling factor for SSA closures.
    gamma_bulk : float, optional
        Bulk Lorentz factor for the observer-to-comoving correction.
    redshift : float, optional
        Source redshift.
    alpha : float, optional
        Fixed pitch angle in radians when ``pitch_average=False``.
    pitch_average : bool, optional
        If ``True``, use pitch-angle-averaged synchrotron coefficients.

    Returns
    -------
    dict
        Dictionary containing

        - ``"R"`` : inferred source radius in cm
        - ``"B"`` : inferred magnetic field strength in G

    Notes
    -----
    This routine is for the non-implicit-cooling SSA closures. It is
    appropriate when the turnover is attributed to SSA but the cooling break
    does not enter the inversion explicitly.

    .. dropdown:: Available regimes
       :icon: chevron-down

       The following regimes are currently supported through
       ``SSA_INV_FUNCTION_REGISTRY``:

       - ``"optically_thick"``
       - ``"optically_thin"``

       These correspond to the two SSA inversion branches implemented in the
       private closure backend.

    .. dropdown:: When to use other closures
       :icon: info

       Use :func:`_invert_powerlaw_cooling_ssa_sed` when an explicit cooling
       Lorentz factor must be supplied, and
       :func:`_invert_powerlaw_implicit_cooling_ssa_sed` when
       :math:`\gamma_c` should instead be obtained from the synchrotron
       cooling closure.

    See Also
    --------
    invert_powerlaw_ssa_sed
        Public, unit-aware wrapper.
    _invert_powerlaw_sed
        Standard optically thin inversion.
    _invert_powerlaw_cooling_ssa_sed
        Explicit-cooling SSA inversion.
    _invert_powerlaw_implicit_cooling_ssa_sed
        Implicit-cooling SSA inversion.
    :ref:`single_zone_sed_inversion`
        Standard inversion framework.
    :ref:`synch_sed_theory`
        SED conventions and SSA theory.
    """
    # ------------------------------------------------------------------ #
    # Pitch-angle treatment
    # ------------------------------------------------------------------ #
    if pitch_average:
        if alpha is not None:
            warnings.warn(
                "Pitch averaging is enabled, so the provided `alpha` is ignored. "
                "(Set `alpha=None` to suppress this warning.)",
                stacklevel=2,
            )
        sin_alpha = None
    else:
        if alpha is None:
            raise ValueError("Pitch averaging is disabled, so a fixed pitch angle must be provided via `alpha`.")
        sin_alpha = np.sin(alpha)

    # ------------------------------------------------------------------ #
    # Handle the regime-specification
    # ------------------------------------------------------------------ #
    if regime not in SSA_INV_FUNCTION_REGISTRY:
        raise ValueError(
            f"Unrecognized SED regime: {regime}. Available regimes: {list(SSA_INV_FUNCTION_REGISTRY.keys())}"
        )

    inversion_function = SSA_INV_FUNCTION_REGISTRY[regime]
    # ------------------------------------------------------------------ #
    # Transform observed peak quantities into the comoving frame.
    #
    # This applies the same minimal relativistic treatment used elsewhere
    # in the default closure: an on-axis Doppler correction plus the usual
    # cosmological redshift correction.
    #
    # IMPORTANT: this is intentionally simple and is not a substitute for a
    # full relativistic radiative-transfer treatment.
    # ------------------------------------------------------------------ #
    doppler_factor = compute_doppler_factor(gamma_bulk, 0.0)

    log_nu_peak_comoving = log_nu_peak - np.log(doppler_factor) + np.log1p(redshift)
    log_F_peak_comoving = log_F_peak - 3.0 * np.log(doppler_factor) + np.log1p(redshift)

    # ------------------------------------------------------------------ #
    # Invert the default closure in the comoving frame.
    # ------------------------------------------------------------------ #
    if regime == "optically_thick":  # nu_a < nu_m < nu_max < nu_c.
        if log_D_A is None:
            raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum2 regime.")

        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            log_D_A,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            sin_alpha=sin_alpha,
        )
    elif regime == "optically_thin":  # nu_m < nu_a < nu_max < nu_c.
        log_R, log_B = inversion_function(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_D_L,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            sin_alpha=sin_alpha,
        )
    else:
        raise ValueError(
            f"Unrecognized SED regime: {regime}. Available regimes: {list(SSA_INV_FUNCTION_REGISTRY.keys())}"
        )

    return {
        "R": np.exp(log_R),
        "B": np.exp(log_B),
    }


def _invert_powerlaw_implicit_cooling_sed(
    regime: str,
    log_F_peak: "_ArrayLike",
    log_nu_peak: "_ArrayLike",
    log_t: "_ArrayLike",
    log_D_L: "_ArrayLike",
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 0.5,
    gamma_bulk: float = 1.0,
    redshift: float = 0.0,
    alpha: float = None,
    pitch_average: bool = False,
):
    r"""
    Infer physical parameters using the implicit synchrotron-cooling closure.

    This low-level helper implements the optically thin inversion in the
    special case where the cooling Lorentz factor is not supplied directly
    but is instead computed from the synchrotron cooling time according to
    the implicit cooling closure described in
    :ref:`synchrotron_cooling_closure`.

    The inversion maps the observed peak frequency and peak flux density,
    together with a source age :math:`t`, to the inferred source radius
    :math:`R`, magnetic field strength :math:`B`, and cooling Lorentz factor
    :math:`\gamma_c`.

    All inputs must already be given in logarithmic CGS units. This routine
    performs no unit handling and is intended for internal use by
    :func:`invert_powerlaw_implicit_cooling_sed`.

    Parameters
    ----------
    regime : str
        Name of the implicit-cooling regime to invert.
    log_F_peak : array-like
        Natural logarithm of the observed peak flux density in
        ``erg cm^-2 s^-1 Hz^-1``.
    log_nu_peak : array-like
        Natural logarithm of the observed peak frequency in Hz.
    log_t : array-like
        Natural logarithm of the source age in s.
    log_D_L : array-like
        Natural logarithm of the luminosity distance in cm.
    gamma_min : float, optional
        Minimum electron Lorentz factor.
    gamma_max : float, optional
        Maximum electron Lorentz factor.
    p : float, optional
        Electron power-law index.
    epsilon_E : float, optional
        Fraction of internal energy in relativistic electrons.
    epsilon_B : float, optional
        Fraction of internal energy in magnetic fields.
    f_V : float, optional
        Effective volume filling factor.
    gamma_bulk : float, optional
        Bulk Lorentz factor for the observer-to-comoving correction.
    redshift : float, optional
        Source redshift.
    alpha : float, optional
        Fixed pitch angle in radians when ``pitch_average=False``.
    pitch_average : bool, optional
        If ``True``, use pitch-angle-averaged synchrotron coefficients.

    Returns
    -------
    dict
        Dictionary containing

        - ``"R"`` : inferred source radius in cm
        - ``"B"`` : inferred magnetic field strength in G
        - ``"gamma_c"`` : inferred cooling Lorentz factor

    Notes
    -----
    This closure assumes synchrotron cooling and eliminates
    :math:`\gamma_c` as an independent parameter by using the source age and
    the inferred magnetic field.

    .. dropdown:: Available regime
       :icon: chevron-down

       Only the following regime is currently supported:

       - ``"slow_cooling"``:
         :math:`\nu_m < \nu_c < \nu_{\max}`

       This restriction is intentional and reflects the assumptions
       documented in :ref:`synchrotron_cooling_closure`.

    .. dropdown:: Important limitations
       :icon: alert

       This closure is valid only for the slow-cooling single-power-law
       treatment implemented in the internal backend. It should not be used
       for fast-cooling or no-cooling configurations, and it becomes
       unreliable when the cooling and injection breaks are not well
       separated.

    See Also
    --------
    invert_powerlaw_implicit_cooling_sed
        Public, unit-aware wrapper.
    _invert_powerlaw_sed
        Standard optically thin inversion.
    _invert_powerlaw_cooling_sed
        Explicit-cooling inversion with user-supplied ``gamma_c``.
    _invert_powerlaw_implicit_cooling_ssa_sed
        Implicit-cooling inversion including SSA.
    :ref:`synchrotron_cooling_closure`
        Formal description of this closure and its caveats.
    :ref:`single_zone_sed_inversion`
        Standard inversion framework.
    :ref:`synch_sed_theory`
        Synchrotron SED conventions.
    """
    # ------------------------------------------------------------------ #
    # Pitch-angle treatment
    # ------------------------------------------------------------------ #
    if pitch_average:
        if alpha is not None:
            warnings.warn(
                "Pitch averaging is enabled, so the provided `alpha` is ignored. "
                "(Set `alpha=None` to suppress this warning.)",
                stacklevel=2,
            )
        sin_alpha = None
    else:
        if alpha is None:
            raise ValueError("Pitch averaging is disabled, so a fixed pitch angle must be provided via `alpha`.")
        sin_alpha = np.sin(alpha)

    # ------------------------------------------------------------------ #
    # Transform observed peak quantities into the comoving frame.
    #
    # This applies the same minimal relativistic treatment used elsewhere
    # in the default closure: an on-axis Doppler correction plus the usual
    # cosmological redshift correction.
    #
    # IMPORTANT: this is intentionally simple and is not a substitute for a
    # full relativistic radiative-transfer treatment.
    # ------------------------------------------------------------------ #
    doppler_factor = compute_doppler_factor(gamma_bulk, 0.0)

    log_nu_peak_comoving = log_nu_peak - np.log(doppler_factor) + np.log1p(redshift)
    log_F_peak_comoving = log_F_peak - 3.0 * np.log(doppler_factor) + np.log1p(redshift)

    # ------------------------------------------------------------------ #
    # Invert the default closure in the comoving frame.
    # ------------------------------------------------------------------ #
    if regime == "slow_cooling":  # nu_m < nu_c < nu_max
        log_R, log_B, log_gamma_c = _inv_log_powerlaw_sbpl_sed_implicit_cool_2(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_t,
            log_D_L,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            sin_alpha=sin_alpha,
        )
    else:
        raise ValueError(
            "Implicit synchrotron cooling cannot be performed outside of the slow-cooling regime. "
            f"Received regime: {regime}.\n"
            "See the documentation for a description of this inversion and its limitations."
        )

    return {
        "R": np.exp(log_R),
        "B": np.exp(log_B),
        "gamma_c": np.exp(log_gamma_c),
    }


def _invert_powerlaw_implicit_cooling_ssa_sed(
    regime: str,
    log_F_peak: "_ArrayLike",
    log_nu_peak: "_ArrayLike",
    log_t: "_ArrayLike",
    log_D_L: "_ArrayLike",
    log_D_A: "_ArrayLike" = None,
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 0.5,
    f_A: float = 0.5,
    gamma_bulk: float = 1.0,
    redshift: float = 0.0,
    alpha: float = None,
    pitch_average: bool = False,
):
    r"""
    Infer physical parameters for SSA spectra with implicit synchrotron cooling.

    This low-level helper implements the SSA inversion family in which the
    cooling Lorentz factor :math:`\gamma_c` is derived from the synchrotron
    cooling closure rather than supplied explicitly. It maps the observed
    peak flux density, peak frequency, and source age to the inferred source
    radius :math:`R`, magnetic field strength :math:`B`, and cooling
    Lorentz factor :math:`\gamma_c`.

    All inputs must already be expressed in logarithmic CGS units. This
    routine is the internal backend for
    :func:`invert_powerlaw_implicit_cooling_ssa_sed`.

    Parameters
    ----------
    regime : str
        Name of the implicit-cooling SSA regime to invert.
    log_F_peak : array-like
        Natural logarithm of the observed peak flux density in
        ``erg cm^-2 s^-1 Hz^-1``.
    log_nu_peak : array-like
        Natural logarithm of the observed peak frequency in Hz.
    log_t : array-like
        Natural logarithm of the source age in s.
    log_D_L : array-like
        Natural logarithm of the luminosity distance in cm.
    log_D_A : array-like, optional
        Natural logarithm of the angular-diameter distance in cm. Required
        for the SSA regimes whose closure depends on projected source area.
    gamma_min : float, optional
        Minimum electron Lorentz factor.
    gamma_max : float, optional
        Maximum electron Lorentz factor.
    p : float, optional
        Electron power-law index.
    epsilon_E : float, optional
        Fraction of internal energy in relativistic electrons.
    epsilon_B : float, optional
        Fraction of internal energy in magnetic fields.
    f_V : float, optional
        Effective volume filling factor.
    f_A : float, optional
        Effective projected-area filling factor.
    gamma_bulk : float, optional
        Bulk Lorentz factor for the observer-to-comoving correction.
    redshift : float, optional
        Source redshift.
    alpha : float, optional
        Fixed pitch angle in radians when ``pitch_average=False``.
    pitch_average : bool, optional
        If ``True``, use pitch-angle-averaged synchrotron coefficients.

    Returns
    -------
    dict
        Dictionary containing

        - ``"R"`` : inferred source radius in cm
        - ``"B"`` : inferred magnetic field strength in G
        - ``"gamma_c"`` : inferred cooling Lorentz factor

    Notes
    -----
    This helper combines the SSA closure with the implicit synchrotron
    cooling closure documented in :ref:`synchrotron_cooling_closure`.

    .. dropdown:: Available regimes
       :icon: chevron-down

       The currently implemented implicit-cooling SSA regimes are

       - ``"Spectrum3"``:
         :math:`\nu_a < \nu_m < \nu_c < \nu_{\max}`
       - ``"Spectrum4"``:
         :math:`\nu_m < \nu_a < \nu_c < \nu_{\max}`
       - ``"Spectrum7"``:
         :math:`\nu_c, \nu_m < \nu_a < \nu_{\max}`

       ``"Spectrum3"`` reduces to the same inversion used by the
       optically thin implicit-cooling case because the SSA turnover does
       not enter the inversion explicitly there.

    .. dropdown:: Relation to other closures
       :icon: info

       Use this routine only when

       - SSA is part of the intended spectral interpretation, and
       - :math:`\gamma_c` should be solved from the synchrotron cooling
         timescale rather than supplied directly.

       For explicit-cooling SSA inversions, use
       :func:`_invert_powerlaw_cooling_ssa_sed`.

    See Also
    --------
    invert_powerlaw_implicit_cooling_ssa_sed
        Public, unit-aware wrapper.
    _invert_powerlaw_implicit_cooling_sed
        Implicit-cooling inversion without SSA.
    _invert_powerlaw_cooling_ssa_sed
        Explicit-cooling SSA inversion.
    :ref:`synchrotron_cooling_closure`
        Formal description of the implicit cooling closure.
    :ref:`single_zone_sed_inversion`
        Standard inversion framework.
    :ref:`synch_sed_theory`
        SED regime definitions and conventions.
    """
    # ------------------------------------------------------------------ #
    # Pitch-angle treatment
    # ------------------------------------------------------------------ #
    if pitch_average:
        if alpha is not None:
            warnings.warn(
                "Pitch averaging is enabled, so the provided `alpha` is ignored. "
                "(Set `alpha=None` to suppress this warning.)",
                stacklevel=2,
            )
        sin_alpha = None
    else:
        if alpha is None:
            raise ValueError("Pitch averaging is disabled, so a fixed pitch angle must be provided via `alpha`.")
        sin_alpha = np.sin(alpha)

    # ------------------------------------------------------------------ #
    # Handle the regime-specification
    # ------------------------------------------------------------------ #
    if regime not in ["Spectrum4", "Spectrum7", "Spectrum3"]:
        raise ValueError(f"Unrecognized SED regime: {regime}. Available regimes: ['Spectrum4','Spectrum7']")
    # ------------------------------------------------------------------ #
    # Transform observed peak quantities into the comoving frame.
    #
    # This applies the same minimal relativistic treatment used elsewhere
    # in the default closure: an on-axis Doppler correction plus the usual
    # cosmological redshift correction.
    #
    # IMPORTANT: this is intentionally simple and is not a substitute for a
    # full relativistic radiative-transfer treatment.
    # ------------------------------------------------------------------ #
    doppler_factor = compute_doppler_factor(gamma_bulk, 0.0)

    log_nu_peak_comoving = log_nu_peak - np.log(doppler_factor) + np.log1p(redshift)
    log_F_peak_comoving = log_F_peak - 3.0 * np.log(doppler_factor) + np.log1p(redshift)

    # ------------------------------------------------------------------ #
    # Invert the default closure in the comoving frame.
    # ------------------------------------------------------------------ #
    if regime == "Spectrum3":  # nu_a < nu_m < nu_c < nu_max.
        # This case has no nu_A influence for inversion, so we can use the standard
        # inversion from the non-ssa scenario.
        log_R, log_B, log_gamma_c = _inv_log_powerlaw_sbpl_sed_implicit_cool_2(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_t,
            log_D_L,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            sin_alpha=sin_alpha,
        )
    elif regime == "Spectrum4":  # nu_m < nu_a < nu_c < nu_max.
        if log_D_A is None:
            raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum4 regime.")
        log_R, log_B, log_gamma_c = _inv_log_powerlaw_sbpl_sed_ssa_implicit_cool_4(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_t,
            log_D_L,
            log_D_A,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            sin_alpha=sin_alpha,
        )
    elif regime == "Spectrum7":  # nu_c, nu_m < nu_a < nu_max.
        if log_D_A is None:
            raise ValueError("Angular diameter distance `D_A` must be provided for Spectrum7 regime.")

        log_R, log_B, log_gamma_c = _inv_log_powerlaw_sbpl_sed_ssa_implicit_cool_7(
            log_nu_peak_comoving,
            log_F_peak_comoving,
            log_t,
            log_D_L,
            log_D_A,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            p=p,
            epsilon_e=epsilon_E,
            epsilon_B=epsilon_B,
            f_V=f_V,
            f_A=f_A,
            sin_alpha=sin_alpha,
        )
    else:
        raise ValueError(f"Unrecognized SED regime: {regime}. Available regimes: ['Spectrum4','Spectrum7']")

    return {
        "R": np.exp(log_R),
        "B": np.exp(log_B),
        "gamma_c": np.exp(log_gamma_c),
    }


def _invert_powerlaw_ssa_sed_demarchi(
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
    is distinct from the standard Triceratops single-zone inversion family.

    .. dropdown:: Numerical implementation notes
       :icon: info

       The calculation is carried out in log space to reduce floating-point
       precision loss and catastrophic cancellation in the analytic
       expressions.

    .. dropdown:: Applicability
       :icon: chevron-down

       Use this function when you specifically want the DM22 inversion based
       on an SSA turnover treated as a broken power law. For the standard
       Triceratops inversion framework, use the other helpers in this
       module.

    See Also
    --------
    invert_powerlaw_ssa_sed_demarchi
        Public, unit-aware wrapper.
    _invert_powerlaw_ssa_sed
        Standard Triceratops SSA inversion.
    :ref:`synch_sed_theory`
        General synchrotron SED conventions.
    :ref:`single_zone_sed_inversion`
        Standard one-zone inversion framework.
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


# ================================================ #
# PUBLIC METHODS
# ================================================ #
# These are the unit handling and cosmology enabled public methods which wrap the low-level dispatch
# functions defined above. These are the methods which are intended to be used by external code and users,
# and they perform unit validation and conversion.
def invert_powerlaw_sed(
    F_peak: "_UnitBearingScalarLike",
    nu_peak: "_UnitBearingScalarLike",
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 0.5,
    gamma_bulk: float = 1.0,
    redshift: float = None,
    luminosity_distance: "_UnitBearingScalarLike" = None,
    angular_diameter_distance: "_UnitBearingScalarLike" = None,
    proper_distance: "_UnitBearingScalarLike" = None,
    cosmology: "cosmo.Cosmology" = None,
    alpha: float = None,
    pitch_average: bool = False,
):
    r"""
    Infer physical source parameters from an optically thin synchrotron peak.

    This is the public, unit-aware wrapper for the default single-zone
    optically thin inversion. It converts an observed peak flux density and
    peak frequency into the inferred emitting radius :math:`R` and magnetic
    field strength :math:`B` under the standard Triceratops one-zone
    closure assumptions.

    The supplied observables are interpreted using the standard
    single-component, optically thin synchrotron closure described in
    :ref:`single_zone_sed_inversion`. Cosmological distances are resolved
    using :func:`resolve_cosmological_distances`, all quantities are
    converted to CGS units, and the inversion is then delegated to
    :func:`_invert_powerlaw_sed`.

    Parameters
    ----------
    F_peak : float, array-like, or ~astropy.units.Quantity
        Observed peak flux density. Must be convertible to
        ``erg cm^-2 s^-1 Hz^-1``.
    nu_peak : float, array-like, or ~astropy.units.Quantity
        Observed peak frequency. Must be convertible to Hz.
    gamma_min : float, optional
        Minimum electron Lorentz factor.
    gamma_max : float, optional
        Maximum electron Lorentz factor.
    p : float, optional
        Electron power-law index.
    epsilon_E : float, optional
        Fraction of internal energy in relativistic electrons.
    epsilon_B : float, optional
        Fraction of internal energy in magnetic fields.
    f_V : float, optional
        Effective emitting-volume filling factor.
    gamma_bulk : float, optional
        Bulk Lorentz factor used for the observer-to-comoving correction.
    redshift : float, optional
        Source redshift. If omitted, it may be inferred from the supplied
        cosmological distance information.
    luminosity_distance : float, array-like, or ~astropy.units.Quantity, optional
        Luminosity distance to the source.
    angular_diameter_distance : float, array-like, or ~astropy.units.Quantity, optional
        Angular-diameter distance to the source.
    proper_distance : float, array-like, or ~astropy.units.Quantity, optional
        Proper distance to the source.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Cosmology used to resolve missing distance information.
    alpha : float, optional
        Fixed pitch angle in radians when ``pitch_average=False``.
    pitch_average : bool, optional
        If ``True``, use pitch-angle-averaged synchrotron coefficients.

    Returns
    -------
    dict
        Dictionary containing

        - ``"R"`` : inferred source radius as a Quantity in cm
        - ``"B"`` : inferred magnetic field strength as a Quantity in G

    Notes
    -----
    This is the standard public inversion for the simplest one-zone closure.

    .. dropdown:: Available regime
       :icon: chevron-down

       This function implements only the standard optically thin closure in
       which the observed peak is identified with the non-SSA synchrotron
       peak of the one-zone model.

    .. dropdown:: When to use another inversion
       :icon: info

       Use a different closure when the fitted peak is shaped by cooling,
       synchrotron self-absorption, or both. The corresponding public
       wrappers in this module expose those alternatives.

    See Also
    --------
    _invert_powerlaw_sed
        Low-level log-space backend.
    invert_powerlaw_cooling_sed
        Optically thin inversion with explicit cooling regime.
    invert_powerlaw_ssa_sed
        SSA inversion without implicit cooling.
    invert_powerlaw_cooling_ssa_sed
        Combined explicit-cooling + SSA inversion.
    invert_powerlaw_implicit_cooling_sed
        Optically thin inversion with implicit synchrotron cooling closure.
    :ref:`single_zone_sed_inversion`
        Standard inversion documentation.
    :ref:`synch_sed_theory`
        General synchrotron theory and conventions.
    """
    # ------------------------------------------------------------------ #
    # Resolve cosmological information.
    # ------------------------------------------------------------------ #
    cosmology = get_cosmology(cosmology=cosmology)
    distances = resolve_cosmological_distances(
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        angular_diameter_distance=angular_diameter_distance,
        proper_distance=proper_distance,
        cosmology=cosmology,
    )
    D_L = distances["luminosity_distance"]
    redshift = distances["redshift"]

    # ------------------------------------------------------------------ #
    # Unit enforcement and conversion to internal log-space representation.
    # ------------------------------------------------------------------ #
    log_F_peak = np.log(ensure_in_units(F_peak, "erg cm^-2 s^-1 Hz^-1"))
    log_nu_peak = np.log(ensure_in_units(nu_peak, "Hz"))
    log_D_L = np.log(ensure_in_units(D_L, "cm"))

    # ------------------------------------------------------------------ #
    # Dispatch to the low-level inverse closure.
    # ------------------------------------------------------------------ #
    results = _invert_powerlaw_sed(
        log_F_peak=log_F_peak,
        log_nu_peak=log_nu_peak,
        log_D_L=log_D_L,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        gamma_bulk=gamma_bulk,
        redshift=redshift,
        alpha=alpha,
        pitch_average=pitch_average,
    )

    return {
        "R": results["R"] * u.cm,
        "B": results["B"] * u.G,
    }


def invert_powerlaw_cooling_ssa_sed(
    regime: str,
    F_peak: "_UnitBearingScalarLike",
    nu_peak: "_UnitBearingScalarLike",
    gamma_min: float = 1.0,
    gamma_c: float = np.inf,
    gamma_max: float = np.inf,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 0.5,
    f_A: float = 0.5,
    gamma_bulk: float = 1.0,
    redshift: float = None,
    luminosity_distance: "_UnitBearingScalarLike" = None,
    angular_diameter_distance: "_UnitBearingScalarLike" = None,
    proper_distance: "_UnitBearingScalarLike" = None,
    cosmology: "cosmo.Cosmology" = None,
    alpha: float = None,
    pitch_average: bool = False,
):
    r"""
    Infer physical parameters for a cooled SSA synchrotron spectrum.

    This is the public, unit-aware wrapper for the combined
    cooling + synchrotron-self-absorption inversion. It converts an
    observed peak frequency and peak flux density to the inferred source
    radius :math:`R` and magnetic field strength :math:`B`, assuming the
    spectrum belongs to one of the explicitly named cooling + SSA regimes.

    Distance information is resolved cosmologically, inputs are converted to
    CGS units, and the inversion is delegated to
    :func:`_invert_powerlaw_cooling_ssa_sed`.

    Parameters
    ----------
    regime : str
        Name of the cooling + SSA regime to invert.
    F_peak : float, array-like, or ~astropy.units.Quantity
        Observed peak flux density. Must be convertible to
        ``erg cm^-2 s^-1 Hz^-1``.
    nu_peak : float, array-like, or ~astropy.units.Quantity
        Observed peak frequency. Must be convertible to Hz.
    gamma_min : float, optional
        Minimum electron Lorentz factor.
    gamma_c : float, optional
        Cooling Lorentz factor used by regimes that explicitly depend on the
        cooling break.
    gamma_max : float, optional
        Maximum electron Lorentz factor.
    p : float, optional
        Electron power-law index.
    epsilon_E : float, optional
        Fraction of internal energy in relativistic electrons.
    epsilon_B : float, optional
        Fraction of internal energy in magnetic fields.
    f_V : float, optional
        Effective volume filling factor.
    f_A : float, optional
        Effective projected-area filling factor used in SSA closures.
    gamma_bulk : float, optional
        Bulk Lorentz factor for the observer-to-comoving correction.
    redshift : float, optional
        Source redshift.
    luminosity_distance : float, array-like, or ~astropy.units.Quantity, optional
        Luminosity distance to the source.
    angular_diameter_distance : float, array-like, or ~astropy.units.Quantity, optional
        Angular-diameter distance to the source.
    proper_distance : float, array-like, or ~astropy.units.Quantity, optional
        Proper distance to the source.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Cosmology used to resolve missing distances.
    alpha : float, optional
        Fixed pitch angle in radians when ``pitch_average=False``.
    pitch_average : bool, optional
        If ``True``, use pitch-angle-averaged synchrotron coefficients.

    Returns
    -------
    dict
        Dictionary containing

        - ``"R"`` : inferred source radius as a Quantity in cm
        - ``"B"`` : inferred magnetic field strength as a Quantity in G

    Notes
    -----
    The supplied ``regime`` selects the analytic inversion branch.

    .. dropdown:: Available regimes
       :icon: chevron-down

       The currently supported cooling + SSA regimes are

       - ``"Spectrum1"``:
         :math:`\nu_a < \nu_m < \nu_{\max} < \nu_c`
       - ``"Spectrum2"``:
         :math:`\nu_m < \nu_a < \nu_{\max} < \nu_c`
       - ``"Spectrum3"``:
         :math:`\nu_a < \nu_m < \nu_c < \nu_{\max}`
       - ``"Spectrum4"``:
         :math:`\nu_m < \nu_a < \nu_c < \nu_{\max}`
       - ``"Spectrum5"``:
         :math:`\nu_a < \nu_c < \nu_m < \nu_{\max}`
       - ``"Spectrum6"``:
         :math:`\nu_c < \nu_a < \nu_m < \nu_{\max}`
       - ``"Spectrum7"``:
         :math:`\nu_c, \nu_m < \nu_a < \nu_{\max}`
       - ``"Spectrum8"``:
         :math:`\nu_c < \nu_a < \nu_m < \nu_{\max}`

    .. dropdown:: Choosing the correct closure
       :icon: info

       This function assumes that :math:`\gamma_c` is provided explicitly
       when the chosen regime requires it. If instead
       :math:`\gamma_c` should be inferred from the synchrotron cooling time,
       use :func:`invert_powerlaw_implicit_cooling_ssa_sed`.

    See Also
    --------
    _invert_powerlaw_cooling_ssa_sed
        Low-level log-space backend.
    invert_powerlaw_ssa_sed
        SSA inversion without explicit cooling.
    invert_powerlaw_cooling_sed
        Optically thin inversion with explicit cooling.
    invert_powerlaw_implicit_cooling_ssa_sed
        SSA inversion with implicit synchrotron cooling closure.
    :ref:`single_zone_sed_inversion`
        Standard inversion framework.
    :ref:`synch_sed_theory`
        Spectral regime definitions and conventions.
    :ref:`synchrotron_cooling_closure`
        Implicit-cooling formalism.
    """
    # ------------------------------------------------------------------ #
    # Resolve cosmological information.
    # ------------------------------------------------------------------ #
    cosmology = get_cosmology(cosmology=cosmology)
    distances = resolve_cosmological_distances(
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        angular_diameter_distance=angular_diameter_distance,
        proper_distance=proper_distance,
        cosmology=cosmology,
    )
    D_L = distances["luminosity_distance"]
    D_A = distances["angular_diameter_distance"]
    redshift = distances["redshift"]

    # ------------------------------------------------------------------ #
    # Unit enforcement and conversion to internal log-space representation.
    # ------------------------------------------------------------------ #
    log_F_peak = np.log(ensure_in_units(F_peak, "erg cm^-2 s^-1 Hz^-1"))
    log_nu_peak = np.log(ensure_in_units(nu_peak, "Hz"))
    log_D_L = np.log(ensure_in_units(D_L, "cm"))
    log_D_A = np.log(ensure_in_units(D_A, "cm"))

    # ------------------------------------------------------------------ #
    # Dispatch to the low-level inverse closure.
    # ------------------------------------------------------------------ #
    results = _invert_powerlaw_cooling_ssa_sed(
        regime=regime,
        log_F_peak=log_F_peak,
        log_nu_peak=log_nu_peak,
        log_D_L=log_D_L,
        log_D_A=log_D_A,
        gamma_min=gamma_min,
        gamma_c=gamma_c,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        f_A=f_A,
        gamma_bulk=gamma_bulk,
        redshift=redshift,
        alpha=alpha,
        pitch_average=pitch_average,
    )

    return {
        "R": results["R"] * u.cm,
        "B": results["B"] * u.G,
    }


def invert_powerlaw_cooling_sed(
    regime: str,
    F_peak: "_UnitBearingScalarLike",
    nu_peak: "_UnitBearingScalarLike",
    gamma_min: float = 1.0,
    gamma_c: float = np.inf,
    gamma_max: float = np.inf,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 0.5,
    gamma_bulk: float = 1.0,
    redshift: float = None,
    luminosity_distance: "_UnitBearingScalarLike" = None,
    angular_diameter_distance: "_UnitBearingScalarLike" = None,
    proper_distance: "_UnitBearingScalarLike" = None,
    cosmology: "cosmo.Cosmology" = None,
    alpha: float = None,
    pitch_average: bool = False,
):
    r"""
    Infer physical parameters for an explicitly cooled optically thin spectrum.

    This is the public, unit-aware wrapper for the optically thin cooling
    inversion in which the spectral regime is specified explicitly and the
    cooling Lorentz factor :math:`\gamma_c` is supplied where needed. It
    maps the observed peak frequency and peak flux density to the inferred
    source radius :math:`R` and magnetic field strength :math:`B`.

    Parameters
    ----------
    regime : str
        Name of the optically thin cooling regime to invert.
    F_peak : float, array-like, or ~astropy.units.Quantity
        Observed peak flux density. Must be convertible to
        ``erg cm^-2 s^-1 Hz^-1``.
    nu_peak : float, array-like, or ~astropy.units.Quantity
        Observed peak frequency. Must be convertible to Hz.
    gamma_min : float, optional
        Minimum electron Lorentz factor.
    gamma_c : float, optional
        Cooling Lorentz factor required by regimes that explicitly depend on
        the cooling break.
    gamma_max : float, optional
        Maximum electron Lorentz factor.
    p : float, optional
        Electron power-law index.
    epsilon_E : float, optional
        Fraction of internal energy in relativistic electrons.
    epsilon_B : float, optional
        Fraction of internal energy in magnetic fields.
    f_V : float, optional
        Effective emitting-volume filling factor.
    gamma_bulk : float, optional
        Bulk Lorentz factor for the observer-to-comoving correction.
    redshift : float, optional
        Source redshift.
    luminosity_distance : float, array-like, or ~astropy.units.Quantity, optional
        Luminosity distance to the source.
    angular_diameter_distance : float, array-like, or ~astropy.units.Quantity, optional
        Angular-diameter distance to the source.
    proper_distance : float, array-like, or ~astropy.units.Quantity, optional
        Proper distance to the source.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Cosmology used to resolve missing distances.
    alpha : float, optional
        Fixed pitch angle in radians when ``pitch_average=False``.
    pitch_average : bool, optional
        If ``True``, use pitch-angle-averaged synchrotron coefficients.

    Returns
    -------
    dict
        Dictionary containing

        - ``"R"`` : inferred source radius as a Quantity in cm
        - ``"B"`` : inferred magnetic field strength as a Quantity in G

    Notes
    -----
    This wrapper exposes the explicitly cooled optically thin inversion
    family.

    .. dropdown:: Available regimes
       :icon: chevron-down

       The currently supported optically thin cooling regimes are

       - ``"fast_cooling"``
       - ``"slow_cooling"``
       - ``"no_cooling"``

    .. dropdown:: Choosing between explicit and implicit cooling
       :icon: info

       Use this function when :math:`\gamma_c` is known or should be treated
       as an explicit model parameter. If :math:`\gamma_c` should instead be
       derived from the synchrotron cooling time and the source age, use
       :func:`invert_powerlaw_implicit_cooling_sed`.

    See Also
    --------
    _invert_powerlaw_cooling_sed
        Low-level log-space backend.
    invert_powerlaw_sed
        Standard optically thin inversion.
    invert_powerlaw_implicit_cooling_sed
        Implicit synchrotron-cooling inversion.
    invert_powerlaw_cooling_ssa_sed
        Combined explicit-cooling + SSA inversion.
    :ref:`single_zone_sed_inversion`
        Standard inversion framework.
    :ref:`synch_sed_theory`
        Regime definitions and synchrotron conventions.
    :ref:`synchrotron_cooling_closure`
        Implicit-cooling closure formalism.
    """
    # ------------------------------------------------------------------ #
    # Resolve cosmological information.
    # ------------------------------------------------------------------ #
    cosmology = get_cosmology(cosmology=cosmology)
    distances = resolve_cosmological_distances(
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        angular_diameter_distance=angular_diameter_distance,
        proper_distance=proper_distance,
        cosmology=cosmology,
    )
    D_L = distances["luminosity_distance"]
    redshift = distances["redshift"]

    # ------------------------------------------------------------------ #
    # Unit enforcement and conversion to internal log-space representation.
    # ------------------------------------------------------------------ #
    log_F_peak = np.log(ensure_in_units(F_peak, "erg cm^-2 s^-1 Hz^-1"))
    log_nu_peak = np.log(ensure_in_units(nu_peak, "Hz"))
    log_D_L = np.log(ensure_in_units(D_L, "cm"))

    # ------------------------------------------------------------------ #
    # Dispatch to the low-level inverse closure.
    # ------------------------------------------------------------------ #
    results = _invert_powerlaw_cooling_sed(
        regime=regime,
        log_F_peak=log_F_peak,
        log_nu_peak=log_nu_peak,
        log_D_L=log_D_L,
        gamma_min=gamma_min,
        gamma_c=gamma_c,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        gamma_bulk=gamma_bulk,
        redshift=redshift,
        alpha=alpha,
        pitch_average=pitch_average,
    )

    return {
        "R": results["R"] * u.cm,
        "B": results["B"] * u.G,
    }


def invert_powerlaw_ssa_sed(
    regime: str,
    F_peak: "_UnitBearingScalarLike",
    nu_peak: "_UnitBearingScalarLike",
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 0.5,
    f_A: float = 0.5,
    gamma_bulk: float = 1.0,
    redshift: float = None,
    luminosity_distance: "_UnitBearingScalarLike" = None,
    angular_diameter_distance: "_UnitBearingScalarLike" = None,
    proper_distance: "_UnitBearingScalarLike" = None,
    cosmology: "cosmo.Cosmology" = None,
    alpha: float = None,
    pitch_average: bool = False,
):
    r"""
    Infer physical parameters for an SSA synchrotron spectrum without cooling closure.

    This is the public, unit-aware wrapper for the SSA inversion family in
    which the spectral turnover is attributed to synchrotron self-absorption
    but no implicit cooling closure is applied. It maps the observed peak
    frequency and peak flux density to the inferred source radius
    :math:`R` and magnetic field strength :math:`B`.

    Parameters
    ----------
    regime : str
        Name of the SSA regime to invert.
    F_peak : float, array-like, or ~astropy.units.Quantity
        Observed peak flux density. Must be convertible to
        ``erg cm^-2 s^-1 Hz^-1``.
    nu_peak : float, array-like, or ~astropy.units.Quantity
        Observed peak frequency. Must be convertible to Hz.
    gamma_min : float, optional
        Minimum electron Lorentz factor.
    gamma_max : float, optional
        Maximum electron Lorentz factor.
    p : float, optional
        Electron power-law index.
    epsilon_E : float, optional
        Fraction of internal energy in relativistic electrons.
    epsilon_B : float, optional
        Fraction of internal energy in magnetic fields.
    f_V : float, optional
        Effective volume filling factor.
    f_A : float, optional
        Effective projected-area filling factor for SSA closures.
    gamma_bulk : float, optional
        Bulk Lorentz factor for the observer-to-comoving correction.
    redshift : float, optional
        Source redshift.
    luminosity_distance : float, array-like, or ~astropy.units.Quantity, optional
        Luminosity distance to the source.
    angular_diameter_distance : float, array-like, or ~astropy.units.Quantity, optional
        Angular-diameter distance to the source.
    proper_distance : float, array-like, or ~astropy.units.Quantity, optional
        Proper distance to the source.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Cosmology used to resolve missing distances.
    alpha : float, optional
        Fixed pitch angle in radians when ``pitch_average=False``.
    pitch_average : bool, optional
        If ``True``, use pitch-angle-averaged synchrotron coefficients.

    Returns
    -------
    dict
        Dictionary containing

        - ``"R"`` : inferred source radius as a Quantity in cm
        - ``"B"`` : inferred magnetic field strength as a Quantity in G

    Notes
    -----
    This wrapper exposes the non-implicit-cooling SSA inversion family.

    .. dropdown:: Available regimes
       :icon: chevron-down

       The currently implemented SSA regimes are

       - ``"optically_thick"``
       - ``"optically_thin"``

    .. dropdown:: Choosing between SSA closures
       :icon: info

       Use this function when the SSA turnover matters but the cooling break
       does not need to be solved via the synchrotron cooling closure. For
       the implicit-cooling SSA case, use
       :func:`invert_powerlaw_implicit_cooling_ssa_sed`.

    See Also
    --------
    _invert_powerlaw_ssa_sed
        Low-level log-space backend.
    invert_powerlaw_sed
        Standard optically thin inversion.
    invert_powerlaw_cooling_ssa_sed
        Explicit-cooling SSA inversion.
    invert_powerlaw_implicit_cooling_ssa_sed
        SSA inversion with implicit synchrotron cooling closure.
    :ref:`single_zone_sed_inversion`
        Standard inversion framework.
    :ref:`synch_sed_theory`
        Synchrotron and SSA theory.
    """
    # ------------------------------------------------------------------ #
    # Resolve cosmological information.
    # ------------------------------------------------------------------ #
    cosmology = get_cosmology(cosmology=cosmology)
    distances = resolve_cosmological_distances(
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        angular_diameter_distance=angular_diameter_distance,
        proper_distance=proper_distance,
        cosmology=cosmology,
    )
    D_L = distances["luminosity_distance"]
    D_A = distances["angular_diameter_distance"]
    redshift = distances["redshift"]

    # ------------------------------------------------------------------ #
    # Unit enforcement and conversion to internal log-space representation.
    # ------------------------------------------------------------------ #
    log_F_peak = np.log(ensure_in_units(F_peak, "erg cm^-2 s^-1 Hz^-1"))
    log_nu_peak = np.log(ensure_in_units(nu_peak, "Hz"))
    log_D_L = np.log(ensure_in_units(D_L, "cm"))
    log_D_A = np.log(ensure_in_units(D_A, "cm"))

    # ------------------------------------------------------------------ #
    # Dispatch to the low-level inverse closure.
    # ------------------------------------------------------------------ #
    results = _invert_powerlaw_ssa_sed(
        regime=regime,
        log_F_peak=log_F_peak,
        log_nu_peak=log_nu_peak,
        log_D_L=log_D_L,
        log_D_A=log_D_A,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        f_A=f_A,
        gamma_bulk=gamma_bulk,
        redshift=redshift,
        alpha=alpha,
        pitch_average=pitch_average,
    )

    return {
        "R": results["R"] * u.cm,
        "B": results["B"] * u.G,
    }


def invert_powerlaw_implicit_cooling_sed(
    regime: str,
    F_peak: "_UnitBearingScalarLike",
    nu_peak: "_UnitBearingScalarLike",
    t: "_UnitBearingScalarLike",
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 0.5,
    gamma_bulk: float = 1.0,
    redshift: float = None,
    luminosity_distance: "_UnitBearingScalarLike" = None,
    angular_diameter_distance: "_UnitBearingScalarLike" = None,
    proper_distance: "_UnitBearingScalarLike" = None,
    cosmology: "cosmo.Cosmology" = None,
    alpha: float = None,
    pitch_average: bool = False,
):
    r"""
    Infer physical parameters using the implicit synchrotron-cooling closure.

    This is the public, unit-aware wrapper for the optically thin inversion
    in which the cooling Lorentz factor :math:`\gamma_c` is computed from
    the synchrotron cooling time rather than supplied explicitly. The
    function maps the observed peak frequency, peak flux density, and source
    age to the inferred source radius :math:`R`, magnetic field strength
    :math:`B`, and cooling Lorentz factor :math:`\gamma_c`.

    Parameters
    ----------
    regime : str
        Name of the implicit-cooling regime to invert.
    F_peak : float, array-like, or ~astropy.units.Quantity
        Observed peak flux density. Must be convertible to
        ``erg cm^-2 s^-1 Hz^-1``.
    nu_peak : float, array-like, or ~astropy.units.Quantity
        Observed peak frequency. Must be convertible to Hz.
    t : float, array-like, or ~astropy.units.Quantity
        Source age. Must be convertible to s.
    gamma_min : float, optional
        Minimum electron Lorentz factor.
    gamma_max : float, optional
        Maximum electron Lorentz factor.
    p : float, optional
        Electron power-law index.
    epsilon_E : float, optional
        Fraction of internal energy in relativistic electrons.
    epsilon_B : float, optional
        Fraction of internal energy in magnetic fields.
    f_V : float, optional
        Effective volume filling factor.
    gamma_bulk : float, optional
        Bulk Lorentz factor for the observer-to-comoving correction.
    redshift : float, optional
        Source redshift.
    luminosity_distance : float, array-like, or ~astropy.units.Quantity, optional
        Luminosity distance to the source.
    angular_diameter_distance : float, array-like, or ~astropy.units.Quantity, optional
        Angular-diameter distance to the source.
    proper_distance : float, array-like, or ~astropy.units.Quantity, optional
        Proper distance to the source.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Cosmology used to resolve missing distances.
    alpha : float, optional
        Fixed pitch angle in radians when ``pitch_average=False``.
    pitch_average : bool, optional
        If ``True``, use pitch-angle-averaged synchrotron coefficients.

    Returns
    -------
    dict
        Dictionary containing

        - ``"R"`` : inferred source radius as a Quantity in cm
        - ``"B"`` : inferred magnetic field strength as a Quantity in G
        - ``"gamma_c"`` : inferred cooling Lorentz factor

    Notes
    -----
    This function implements the public implicit-cooling closure documented
    in :ref:`synchrotron_cooling_closure`.

    .. dropdown:: Available regime
       :icon: chevron-down

       Only the following regime is currently supported:

       - ``"slow_cooling"``

       This corresponds to the slow-cooling optically thin inversion in
       which the synchrotron cooling closure is mathematically consistent
       with the assumptions of the implemented analytic solution.

    .. dropdown:: Important caveats
       :icon: alert

       This closure assumes synchrotron cooling and a single-power-law
       electron treatment. It is not the appropriate choice for fast-cooling
       or effectively no-cooling cases.

    See Also
    --------
    _invert_powerlaw_implicit_cooling_sed
        Low-level log-space backend.
    invert_powerlaw_cooling_sed
        Explicit-cooling inversion with user-supplied ``gamma_c``.
    invert_powerlaw_implicit_cooling_ssa_sed
        Implicit-cooling inversion including SSA.
    :ref:`synchrotron_cooling_closure`
        Formal description of this closure.
    :ref:`single_zone_sed_inversion`
        Standard inversion framework.
    :ref:`synch_sed_theory`
        Synchrotron theory and conventions.
    """
    # ------------------------------------------------------------------ #
    # Resolve cosmological information.
    # ------------------------------------------------------------------ #
    cosmology = get_cosmology(cosmology=cosmology)
    distances = resolve_cosmological_distances(
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        angular_diameter_distance=angular_diameter_distance,
        proper_distance=proper_distance,
        cosmology=cosmology,
    )
    D_L = distances["luminosity_distance"]
    redshift = distances["redshift"]

    # ------------------------------------------------------------------ #
    # Unit enforcement and conversion to internal log-space representation.
    # ------------------------------------------------------------------ #
    log_F_peak = np.log(ensure_in_units(F_peak, "erg cm^-2 s^-1 Hz^-1"))
    log_nu_peak = np.log(ensure_in_units(nu_peak, "Hz"))
    log_t = np.log(ensure_in_units(t, "s"))
    log_D_L = np.log(ensure_in_units(D_L, "cm"))

    # ------------------------------------------------------------------ #
    # Dispatch to the low-level inverse closure.
    # ------------------------------------------------------------------ #
    results = _invert_powerlaw_implicit_cooling_sed(
        regime=regime,
        log_F_peak=log_F_peak,
        log_nu_peak=log_nu_peak,
        log_t=log_t,
        log_D_L=log_D_L,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        gamma_bulk=gamma_bulk,
        redshift=redshift,
        alpha=alpha,
        pitch_average=pitch_average,
    )

    return {
        "R": results["R"] * u.cm,
        "B": results["B"] * u.G,
        "gamma_c": results["gamma_c"],
    }


def invert_powerlaw_implicit_cooling_ssa_sed(
    regime: str,
    F_peak: "_UnitBearingScalarLike",
    nu_peak: "_UnitBearingScalarLike",
    t: "_UnitBearingScalarLike",
    gamma_min: float = 1.0,
    gamma_max: float = np.inf,
    p: float = 3.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 0.5,
    f_A: float = 0.5,
    gamma_bulk: float = 1.0,
    redshift: float = None,
    luminosity_distance: "_UnitBearingScalarLike" = None,
    angular_diameter_distance: "_UnitBearingScalarLike" = None,
    proper_distance: "_UnitBearingScalarLike" = None,
    cosmology: "cosmo.Cosmology" = None,
    alpha: float = None,
    pitch_average: bool = False,
):
    r"""
    Infer physical parameters for SSA spectra with implicit synchrotron cooling.

    This is the public, unit-aware wrapper for the SSA inversion family in
    which the cooling Lorentz factor :math:`\gamma_c` is obtained from the
    synchrotron cooling closure rather than supplied explicitly. The
    function maps the observed peak frequency, peak flux density, and source
    age to the inferred source radius :math:`R`, magnetic field strength
    :math:`B`, and cooling Lorentz factor :math:`\gamma_c`.

    Parameters
    ----------
    regime : str
        Name of the implicit-cooling SSA regime to invert.
    F_peak : float, array-like, or ~astropy.units.Quantity
        Observed peak flux density. Must be convertible to
        ``erg cm^-2 s^-1 Hz^-1``.
    nu_peak : float, array-like, or ~astropy.units.Quantity
        Observed peak frequency. Must be convertible to Hz.
    t : float, array-like, or ~astropy.units.Quantity
        Source age. Must be convertible to s.
    gamma_min : float, optional
        Minimum electron Lorentz factor.
    gamma_max : float, optional
        Maximum electron Lorentz factor.
    p : float, optional
        Electron power-law index.
    epsilon_E : float, optional
        Fraction of internal energy in relativistic electrons.
    epsilon_B : float, optional
        Fraction of internal energy in magnetic fields.
    f_V : float, optional
        Effective volume filling factor.
    f_A : float, optional
        Effective projected-area filling factor for SSA closures.
    gamma_bulk : float, optional
        Bulk Lorentz factor for the observer-to-comoving correction.
    redshift : float, optional
        Source redshift.
    luminosity_distance : float, array-like, or ~astropy.units.Quantity, optional
        Luminosity distance to the source.
    angular_diameter_distance : float, array-like, or ~astropy.units.Quantity, optional
        Angular-diameter distance to the source.
    proper_distance : float, array-like, or ~astropy.units.Quantity, optional
        Proper distance to the source.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Cosmology used to resolve missing distances.
    alpha : float, optional
        Fixed pitch angle in radians when ``pitch_average=False``.
    pitch_average : bool, optional
        If ``True``, use pitch-angle-averaged synchrotron coefficients.

    Returns
    -------
    dict
        Dictionary containing

        - ``"R"`` : inferred source radius as a Quantity in cm
        - ``"B"`` : inferred magnetic field strength as a Quantity in G
        - ``"gamma_c"`` : inferred cooling Lorentz factor

    Notes
    -----
    This function exposes the SSA + implicit-cooling closures documented in
    :ref:`synchrotron_cooling_closure`.

    .. dropdown:: Available regimes
       :icon: chevron-down

       The currently implemented regimes are

       - ``"Spectrum3"``
       - ``"Spectrum4"``
       - ``"Spectrum7"``

    .. dropdown:: When to use this function
       :icon: info

       Use this wrapper when both of the following are true:

       - the SED interpretation requires SSA, and
       - :math:`\gamma_c` should be derived from the synchrotron cooling
         timescale rather than supplied directly.

    See Also
    --------
    _invert_powerlaw_implicit_cooling_ssa_sed
        Low-level log-space backend.
    invert_powerlaw_implicit_cooling_sed
        Implicit-cooling inversion without SSA.
    invert_powerlaw_cooling_ssa_sed
        Explicit-cooling SSA inversion.
    :ref:`synchrotron_cooling_closure`
        Formal description of the implicit cooling closure.
    :ref:`single_zone_sed_inversion`
        Standard inversion framework.
    :ref:`synch_sed_theory`
        Synchrotron and SSA regime conventions.
    """
    # ------------------------------------------------------------------ #
    # Resolve cosmological information.
    # ------------------------------------------------------------------ #
    cosmology = get_cosmology(cosmology=cosmology)
    distances = resolve_cosmological_distances(
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        angular_diameter_distance=angular_diameter_distance,
        proper_distance=proper_distance,
        cosmology=cosmology,
    )
    D_L = distances["luminosity_distance"]
    D_A = distances["angular_diameter_distance"]
    redshift = distances["redshift"]

    # ------------------------------------------------------------------ #
    # Unit enforcement and conversion to internal log-space representation.
    # ------------------------------------------------------------------ #
    log_F_peak = np.log(ensure_in_units(F_peak, "erg cm^-2 s^-1 Hz^-1"))
    log_nu_peak = np.log(ensure_in_units(nu_peak, "Hz"))
    log_D_L = np.log(ensure_in_units(D_L, "cm"))
    log_t = np.log(ensure_in_units(t, "s"))
    log_D_A = np.log(ensure_in_units(D_A, "cm"))

    # ------------------------------------------------------------------ #
    # Dispatch to the low-level inverse closure.
    # ------------------------------------------------------------------ #
    results = _invert_powerlaw_implicit_cooling_ssa_sed(
        regime=regime,
        log_F_peak=log_F_peak,
        log_nu_peak=log_nu_peak,
        log_t=log_t,
        log_D_L=log_D_L,
        log_D_A=log_D_A,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        f_V=f_V,
        f_A=f_A,
        gamma_bulk=gamma_bulk,
        redshift=redshift,
        alpha=alpha,
        pitch_average=pitch_average,
    )

    return {
        "R": results["R"] * u.cm,
        "B": results["B"] * u.G,
        "gamma_c": results["gamma_c"],
    }


def invert_powerlaw_ssa_sed_demarchi(
    nu_brk: Union[float, np.ndarray, u.Quantity],
    F_nu_brk: Union[float, np.ndarray, u.Quantity],
    *,
    p: Union[float, np.ndarray] = 3.0,
    f: Union[float, np.ndarray] = 0.5,
    theta: Union[float, np.ndarray] = np.pi / 2,
    epsilon_B: Union[float, np.ndarray] = 0.1,
    epsilon_E: Union[float, np.ndarray] = 0.1,
    gamma_min: Union[float, np.ndarray] = 1.0,
    gamma_max: Union[float, np.ndarray] = 1e6,
    redshift: float = None,
    luminosity_distance: "_UnitBearingScalarLike" = None,
    angular_diameter_distance: "_UnitBearingScalarLike" = None,
    proper_distance: "_UnitBearingScalarLike" = None,
    cosmology: "cosmo.Cosmology" = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Invert an SSA turnover using the DeMarchi et al. (2022) analytic formulae.

    This is the public, unit-aware wrapper for the :footcite:t:`demarchiRadioAnalysisSN2004C2022`
    SSA inversion. It converts an observed turnover frequency and flux
    density to the inferred magnetic field strength :math:`B` and emitting
    radius :math:`R` using the DM22 broken-power-law closure.

    Parameters
    ----------
    nu_brk : float, array-like, or ~astropy.units.Quantity
        SSA turnover frequency :math:`\nu_{\rm brk}`. If unitless, it is
        interpreted in GHz.
    F_nu_brk : float, array-like, or ~astropy.units.Quantity
        Flux density at the turnover. If unitless, it is interpreted in Jy.
    p : float or array-like, optional
        Electron power-law index.
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
        Maximum electron Lorentz factor.
    redshift : float, optional
        Source redshift used to resolve cosmological distances.
    luminosity_distance : float, array-like, or ~astropy.units.Quantity, optional
        Luminosity distance to the source.
    angular_diameter_distance : float, array-like, or ~astropy.units.Quantity, optional
        Angular-diameter distance to the source.
    proper_distance : float, array-like, or ~astropy.units.Quantity, optional
        Proper distance to the source.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Cosmology used to resolve missing distances.

    Returns
    -------
    tuple
        Two-element tuple ``(B, R)`` where

        - ``B`` is returned as a Quantity in G
        - ``R`` is returned as a Quantity in cm

    Notes
    -----
    This closure is separate from the main Triceratops single-zone inversion
    family and specifically follows the DeMarchi et al. analytic treatment.

    .. dropdown:: Applicability
       :icon: chevron-down

       Use this function when you want the DM22 SSA inversion based on an
       observed turnover modeled as a broken power law.

    .. dropdown:: Important caveat
       :icon: alert

       The implementation does not support :math:`p = 2` and should not be
       used in that singular case without a separate treatment.

    See Also
    --------
    _invert_powerlaw_ssa_sed_demarchi
        Low-level backend implementing the DM22 closure.
    invert_powerlaw_ssa_sed
        Standard Triceratops SSA inversion.
    :ref:`synch_sed_theory`
        General synchrotron conventions.
    :ref:`single_zone_sed_inversion`
        Standard one-zone inversion framework.

    References
    ----------
    .. footbibliography::
    """
    # ------------------------------------------------------------------ #
    # Resolve cosmological information.
    # ------------------------------------------------------------------ #
    cosmology = get_cosmology(cosmology=cosmology)
    distances = resolve_cosmological_distances(
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        angular_diameter_distance=angular_diameter_distance,
        proper_distance=proper_distance,
        cosmology=cosmology,
    )
    D_L = distances["luminosity_distance"]

    # Validate units of all unit bearing quantities and coerce them to the expected
    # units for the optimized backend.
    nu_brk = ensure_in_units(nu_brk, u.GHz)
    F_nu_brk = ensure_in_units(F_nu_brk, u.Jy)
    distance = ensure_in_units(D_L, u.Mpc)

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
    B, R = _invert_powerlaw_ssa_sed_demarchi(
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
