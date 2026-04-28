"""
Blackbody optical photometry models.

Provides the simplest possible concrete implementations of
:class:`~triceratops.models.core.optical.OpticalModel` and
:class:`~triceratops.models.core.optical.OpticalEpochModel`: a static
(time-independent) spherical blackbody at a fixed temperature, radius,
and luminosity distance.

These models are primarily intended to validate the optical model stack
end-to-end (model → ``to_inference_data`` → ``GaussianLikelihood`` →
``InferenceProblem``) and to serve as a reference implementation for more
complex SED models.
"""

from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from triceratops.models.core.optical import OpticalEpochModel, OpticalModel
from triceratops.models.core.parameters import ModelParameter
from triceratops.radiation.blackbody import _log_specific_flux_fnu_cgs

if TYPE_CHECKING:
    from triceratops._typing import _ModelParametersInputRaw
    from triceratops.utils.phot_utils import FilterBundle

__all__ = ["BlackbodyOpticalModel", "BlackbodyOpticalEpochModel"]

_BB_PARAMETERS = (
    ModelParameter(
        "T",
        1.0e4,
        base_units=u.K,
        description="Blackbody temperature.",
        latex=r"$T$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "R",
        1.0e14,
        base_units=u.cm,
        description="Photospheric radius.",
        latex=r"$R$",
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


class BlackbodyOpticalModel(OpticalModel):
    r"""
    Static spherical blackbody: time-evolving multi-band optical photometry.

    Temperature, radius, and distance are time-independent; the SED shape is
    identical at every epoch.  Intended for end-to-end validation and as a
    starting point for more physical models (e.g., one that feeds T(t), R(t)
    from a shock engine into :meth:`_compute_sed`).

    Parameters
    ----------
    bundle : FilterBundle
        Filter set defining the observable bands.
    """

    PARAMETERS = _BB_PARAMETERS
    DESCRIPTION = "Static spherical blackbody observed through an optical filter set."
    REFERENCE = "Planck 1901."

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
        log_R = np.log(parameters["R"])
        log_D = np.log(parameters["D"])

        log_fnu = _log_specific_flux_fnu_cgs(log_nu, log_T, log_R, log_D)  # (N_nu,)

        # Broadcast to (N_t, N_nu): same SED at every time.
        return np.tile(np.exp(log_fnu), (len(t_unique), 1))


class BlackbodyOpticalEpochModel(OpticalEpochModel):
    r"""
    Static spherical blackbody: single-epoch multi-band optical photometry.

    Parameters
    ----------
    bundle : FilterBundle
        Filter set defining the observable bands.
    """

    PARAMETERS = _BB_PARAMETERS
    DESCRIPTION = "Static spherical blackbody SED observed through an optical filter set."
    REFERENCE = "Planck 1901."

    def __init__(self, bundle: "FilterBundle") -> None:
        super().__init__(bundle)

    def _compute_sed(
        self,
        nu_grid: np.ndarray,
        parameters: "_ModelParametersInputRaw",
    ) -> np.ndarray:
        log_nu = np.log(nu_grid)  # (N_nu,)
        log_T = np.log(parameters["T"])
        log_R = np.log(parameters["R"])
        log_D = np.log(parameters["D"])

        return np.exp(_log_specific_flux_fnu_cgs(log_nu, log_T, log_R, log_D))  # (N_nu,)
