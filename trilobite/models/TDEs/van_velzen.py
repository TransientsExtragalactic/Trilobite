"""
The gaussian rise exponential decay light curve model from Van Velzen+ 2021.

Implements a purely phenomenological optical/UV tidal disruption event (TDE)
light curve model: the bolometric luminosity rises as a Gaussian and decays
exponentially about a peak time, while the blackbody temperature is held
fixed in time. The photospheric radius at each epoch follows from the
instantaneous luminosity and the (constant) temperature via the
Stefan-Boltzmann relation, and the observed flux density follows from the
standard spherical-blackbody flux formula.
"""

from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from trilobite.models.core.optical import OpticalModel
from trilobite.models.core.parameters import ModelParameter
from trilobite.radiation.blackbody import _log_photospheric_radius, _log_specific_flux_fnu_cgs

if TYPE_CHECKING:
    from trilobite._typing import _ModelParametersInputRaw
    from trilobite.utils.phot_utils import FilterBundle

__all__ = ["VanVelzenModel"]


# ============================================================
# Luminosity Light Curve
# ============================================================
def _log_luminosity_gaussian_rise_exp_decay_cgs(
    t: np.ndarray,
    t_peak: float,
    sigma_rise: float,
    tau_decay: float,
    log_L_peak: float,
) -> np.ndarray:
    r"""
    Natural log of a Gaussian-rise / exponential-decay luminosity light curve.

    .. math::

        L(t) =
        \begin{cases}
            L_{\rm peak}\,
            \exp\!\left[-\dfrac{(t-t_{\rm peak})^2}{2\sigma_{\rm rise}^2}\right],
            & t \le t_{\rm peak} \\[6pt]
            L_{\rm peak}\,
            \exp\!\left[-\dfrac{t-t_{\rm peak}}{\tau_{\rm decay}}\right],
            & t > t_{\rm peak}
        \end{cases}

    Parameters
    ----------
    t : ndarray
        Time in seconds.
    t_peak : float
        Time of peak luminosity, in seconds.
    sigma_rise : float
        Gaussian rise width, in seconds.
    tau_decay : float
        Exponential decay e-folding time, in seconds.
    log_L_peak : float
        Natural logarithm of the peak luminosity.

    Returns
    -------
    ndarray
        Natural logarithm of the luminosity at each input time.

    Notes
    -----
    Continuous at :math:`t = t_{\rm peak}` by construction: both branches
    evaluate to :math:`L_{\rm peak}` there.
    """
    t = np.asarray(t, dtype=float)
    dt = t - t_peak

    return np.where(
        dt <= 0,
        log_L_peak - 0.5 * (dt / sigma_rise) ** 2,
        log_L_peak - dt / tau_decay,
    )


# ============================================================
# Parameter Declarations
# ============================================================
_VV_PARAMETERS = (
    ModelParameter(
        "L_peak",
        1.0e44,
        base_units=u.Unit("erg s-1"),
        description="Peak bolometric luminosity.",
        latex=r"$L_{\rm peak}$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "t_peak",
        0.0,
        base_units=u.s,
        description="Time of peak bolometric luminosity.",
        latex=r"$t_{\rm peak}$",
    ),
    ModelParameter(
        "sigma_rise",
        8.64e5,  # 10 days
        base_units=u.s,
        description="Gaussian rise width of the luminosity light curve.",
        latex=r"$\sigma_{\rm rise}$",
        bounds=(1.0, None),
    ),
    ModelParameter(
        "tau_decay",
        4.32e6,  # 50 days
        base_units=u.s,
        description="Exponential decay e-folding time of the luminosity light curve.",
        latex=r"$\tau_{\rm decay}$",
        bounds=(1.0, None),
    ),
    ModelParameter(
        "T",
        2.0e4,
        base_units=u.K,
        description="Blackbody temperature (constant in time).",
        latex=r"$T$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "D",
        3.086e26,
        base_units=u.cm,
        description="Luminosity distance.",
        latex=r"$D$",
        bounds=(0.0, None),
    ),
)


# ============================================================
# Model Definition
# ============================================================
class VanVelzenModel(OpticalModel):
    r"""
    Gaussian-rise / exponential-decay blackbody TDE light curve model.

    This model describes a tidal disruption event's optical/UV emission as a
    constant-temperature blackbody whose bolometric luminosity rises as a
    Gaussian and decays exponentially about a peak time:

    .. math::

        L(t) =
        \begin{cases}
            L_{\rm peak}\,
            \exp\!\left[-\dfrac{(t-t_{\rm peak})^2}{2\sigma_{\rm rise}^2}\right],
            & t \le t_{\rm peak} \\[6pt]
            L_{\rm peak}\,
            \exp\!\left[-\dfrac{t-t_{\rm peak}}{\tau_{\rm decay}}\right],
            & t > t_{\rm peak}
        \end{cases}

    At each epoch, the photospheric radius follows from the instantaneous
    luminosity and the (time-independent) temperature :math:`T` via the
    Stefan-Boltzmann relation, :math:`L(t) = 4\pi R(t)^2 \sigma_{\rm SB} T^4`,
    and the observed flux density is the standard spherical-blackbody flux

    .. math::

        f_\nu(\nu,t) = \pi B_\nu(\nu,T) \left(\frac{R(t)}{D}\right)^2.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``L_peak``
             - :math:`L_{\rm peak}`
             - Peak bolometric luminosity.
           * - ``t_peak``
             - :math:`t_{\rm peak}`
             - Time of peak bolometric luminosity.
           * - ``sigma_rise``
             - :math:`\sigma_{\rm rise}`
             - Gaussian rise width.
           * - ``tau_decay``
             - :math:`\tau_{\rm decay}`
             - Exponential decay e-folding time.
           * - ``T``
             - :math:`T`
             - Blackbody temperature (constant in time).
           * - ``D``
             - :math:`D`
             - Luminosity distance.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``band_index``
             - :math:`b`
             - Integer index into the attached FilterBundle (or band name).
           * - ``time``
             - :math:`t`
             - Observation time.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux``
             - :math:`f_\nu(\nu,t)`
             - Filter-convolved flux density.

    Notes
    -----
    - The luminosity light curve is continuous at :math:`t = t_{\rm peak}`.
    - The temperature is assumed constant across the full light curve; this
      is the simplest version of the model and does not capture any
      color evolution.
    - No cosmological corrections (redshift, K-correction) are applied;
      ``D`` is treated as a plain luminosity distance.

    Parameters
    ----------
    bundle : FilterBundle
        Filter set defining the observable bands.

    References
    ----------
    van Velzen et al. 2021, ApJ, 908, 4.
    """

    PARAMETERS = _VV_PARAMETERS
    DESCRIPTION = (
        "Gaussian-rise, exponential-decay bolometric luminosity light curve, "
        "radiated as a constant-temperature blackbody; the photospheric radius "
        "at each epoch follows from the instantaneous luminosity via the "
        "Stefan-Boltzmann relation."
    )
    REFERENCE = "van Velzen et al. 2021, ApJ, 908, 4."

    def __init__(self, bundle: "FilterBundle") -> None:
        super().__init__(bundle)

    def _compute_sed(
        self,
        nu_grid: np.ndarray,
        t_unique: np.ndarray,
        parameters: "_ModelParametersInputRaw",
    ) -> np.ndarray:
        log_nu = np.log(nu_grid)  # (N_nu,)
        log_T = np.log(parameters["T"])
        log_D = np.log(parameters["D"])
        log_L_peak = np.log(parameters["L_peak"])

        log_L = _log_luminosity_gaussian_rise_exp_decay_cgs(
            t_unique,
            parameters["t_peak"],
            parameters["sigma_rise"],
            parameters["tau_decay"],
            log_L_peak,
        )  # (N_t,)

        log_R = _log_photospheric_radius(log_L, log_T)  # (N_t,)

        log_fnu = _log_specific_flux_fnu_cgs(log_nu[None, :], log_T, log_R[:, None], log_D)  # (N_t, N_nu)

        return np.exp(log_fnu)
