r"""
Coupled optical photometry models.

Each model in this module pairs a spectral SED component (blackbody or
power law) with a temporal normalization (FRED, Generalized-FRED, or
Gaussian) to produce a multi-band optical light curve.

The forward model is:

.. math::

    F_\mathrm{band}(t) =
    \phi(t) \times
    \bigl[\mathbf{W}\, F_\nu^\mathrm{SED}(\boldsymbol{\nu})\bigr]_{\mathrm{band\_idx}}

where :math:`\mathbf{W}` is the filter weight matrix, :math:`\phi(t)` is
the dimensionless temporal normalization (peak = 1), and the SED amplitude
parameter controls the absolute flux scale.

All models accept a :class:`~triceratops.utils.phot_utils.FilterBundle` as
the first constructor argument.  The bundle is used for SED convolution and
for resolving ``band_name → band_idx`` when the model is passed to
:meth:`~triceratops.data.optical_photometry.OpticalPhotometryContainer.to_inference_data`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from ..core import ModelParameter
from ..core.optical import Blackbody, CoupledOpticalModel, PowerLawSED

if TYPE_CHECKING:
    from triceratops.utils.phot_utils import FilterBundle

__all__ = [
    "FREDBlackbodyModel",
    "FREDPowerLawModel",
    "GeneralizedFREDBlackbodyModel",
    "GaussianBlackbodyModel",
]

# ---------------------------------------------------------------------------
# Shared temporal parameter sets (time in seconds)
# ---------------------------------------------------------------------------
_FRED_TEMPORAL_PARAMETERS = (
    ModelParameter(
        "t_0",
        86_400.0,
        base_units=u.s,
        description="Flare onset time.",
        latex=r"$t_0$",
    ),
    ModelParameter(
        "tau_r",
        86_400.0,
        base_units=u.s,
        bounds=(1.0, None),
        description="Rise timescale.",
        latex=r"$\tau_r$",
    ),
    ModelParameter(
        "tau_d",
        864_000.0,
        base_units=u.s,
        bounds=(1.0, None),
        description="Decay timescale.",
        latex=r"$\tau_d$",
    ),
)

_GFRED_TEMPORAL_PARAMETERS = (
    ModelParameter(
        "t_0",
        86_400.0,
        base_units=u.s,
        description="Flare onset time.",
        latex=r"$t_0$",
    ),
    ModelParameter(
        "tau_r",
        86_400.0,
        base_units=u.s,
        bounds=(1.0, None),
        description="Rise timescale.",
        latex=r"$\tau_r$",
    ),
    ModelParameter(
        "tau_d",
        864_000.0,
        base_units=u.s,
        bounds=(1.0, None),
        description="Decay timescale.",
        latex=r"$\tau_d$",
    ),
    ModelParameter(
        "nu_r",
        1.0,
        base_units=u.dimensionless_unscaled,
        bounds=(0.0, None),
        description="Rise shape exponent (ν_r = 1 recovers FRED).",
        latex=r"$\nu_r$",
    ),
    ModelParameter(
        "nu_d",
        1.0,
        base_units=u.dimensionless_unscaled,
        bounds=(0.0, None),
        description="Decay shape exponent (ν_d = 1 recovers FRED).",
        latex=r"$\nu_d$",
    ),
)

_GAUSSIAN_TEMPORAL_PARAMETERS = (
    ModelParameter(
        "t_peak",
        86_400.0,
        base_units=u.s,
        description="Peak time of the Gaussian pulse.",
        latex=r"$t_\mathrm{peak}$",
    ),
    ModelParameter(
        "sigma_t",
        86_400.0,
        base_units=u.s,
        bounds=(1.0, None),
        description="Width (standard deviation) of the Gaussian pulse.",
        latex=r"$\sigma_t$",
    ),
)


# ---------------------------------------------------------------------------
# Helper: FRED temporal shape
# ---------------------------------------------------------------------------
def _fred_shape(time_s: np.ndarray, t_0: float, tau_r: float, tau_d: float) -> np.ndarray:
    dt = time_s - t_0
    return np.where(
        dt >= 0.0,
        (1.0 - np.exp(-dt / tau_r)) * np.exp(-dt / tau_d),
        0.0,
    )


def _gfred_shape(
    time_s: np.ndarray,
    t_0: float,
    tau_r: float,
    tau_d: float,
    nu_r: float,
    nu_d: float,
) -> np.ndarray:
    dt = time_s - t_0
    return np.where(
        dt >= 0.0,
        (1.0 - np.exp(-((dt / tau_r) ** nu_r))) * np.exp(-((dt / tau_d) ** nu_d)),
        0.0,
    )


def _gaussian_shape(time_s: np.ndarray, t_peak: float, sigma_t: float) -> np.ndarray:
    return np.exp(-0.5 * ((time_s - t_peak) / sigma_t) ** 2)


# ---------------------------------------------------------------------------
# FREDBlackbodyModel
# ---------------------------------------------------------------------------
class FREDBlackbodyModel(CoupledOpticalModel):
    r"""
    FRED temporal envelope with a Planck (blackbody) SED.

    The model evaluates:

    .. math::

        F_\mathrm{band}(t) =
        \phi_\mathrm{FRED}(t;\, t_0, \tau_r, \tau_d) \times
        \bigl[\mathbf{W}\, F_\nu^\mathrm{BB}(\boldsymbol{\nu};\, T_\mathrm{eff}, A)\bigr]_{\mathrm{band\_idx}}

    where :math:`\phi_\mathrm{FRED}` is the dimensionless FRED shape

    .. math::

        \phi_\mathrm{FRED}(t) =
        \begin{cases}
            0 & t < t_0 \\
            \left(1 - e^{-(t-t_0)/\tau_r}\right) e^{-(t-t_0)/\tau_d} & t \ge t_0
        \end{cases}

    and :math:`A` is the band-averaged V-pivot flux at peak.

    Parameters
    ----------
    bundle : FilterBundle
        Filter bundle providing the common frequency grid and weight matrix.
    """

    PARAMETERS = (
        *_FRED_TEMPORAL_PARAMETERS,
        *Blackbody.PARAMETERS,
    )

    DESCRIPTION = "FRED temporal profile coupled with a blackbody SED."

    def __init__(self, bundle: FilterBundle):
        self._sed = Blackbody()
        super().__init__(bundle)

    def _evaluate_sed(self, frequency_hz: np.ndarray, parameters: dict) -> np.ndarray:
        return self._sed.evaluate(frequency_hz, parameters)

    def _evaluate_temporal(self, time_s: np.ndarray, parameters: dict) -> np.ndarray:
        return _fred_shape(time_s, parameters["t_0"], parameters["tau_r"], parameters["tau_d"])


# ---------------------------------------------------------------------------
# FREDPowerLawModel
# ---------------------------------------------------------------------------
class FREDPowerLawModel(CoupledOpticalModel):
    r"""
    FRED temporal envelope with a power-law SED.

    .. math::

        F_\mathrm{band}(t) =
        \phi_\mathrm{FRED}(t;\, t_0, \tau_r, \tau_d) \times
        \bigl[\mathbf{W}\, F_\nu^\mathrm{PL}(\boldsymbol{\nu};\, \alpha, A)\bigr]_{\mathrm{band\_idx}}

    Parameters
    ----------
    bundle : FilterBundle
        Filter bundle providing the common frequency grid and weight matrix.
    """

    PARAMETERS = (
        *_FRED_TEMPORAL_PARAMETERS,
        *PowerLawSED.PARAMETERS,
    )

    DESCRIPTION = "FRED temporal profile coupled with a power-law SED."

    def __init__(self, bundle: FilterBundle):
        self._sed = PowerLawSED()
        super().__init__(bundle)

    def _evaluate_sed(self, frequency_hz: np.ndarray, parameters: dict) -> np.ndarray:
        return self._sed.evaluate(frequency_hz, parameters)

    def _evaluate_temporal(self, time_s: np.ndarray, parameters: dict) -> np.ndarray:
        return _fred_shape(time_s, parameters["t_0"], parameters["tau_r"], parameters["tau_d"])


# ---------------------------------------------------------------------------
# GeneralizedFREDBlackbodyModel
# ---------------------------------------------------------------------------
class GeneralizedFREDBlackbodyModel(CoupledOpticalModel):
    r"""
    Generalized-FRED temporal envelope with a blackbody SED.

    The generalized FRED shape is

    .. math::

        \phi(t) =
        \begin{cases}
            0 & t < t_0 \\
            \left(1 - e^{-\left(\frac{t-t_0}{\tau_r}\right)^{\nu_r}}\right)
            e^{-\left(\frac{t-t_0}{\tau_d}\right)^{\nu_d}} & t \ge t_0
        \end{cases}

    Setting :math:`\nu_r = \nu_d = 1` recovers the standard FRED shape.

    Parameters
    ----------
    bundle : FilterBundle
        Filter bundle providing the common frequency grid and weight matrix.
    """

    PARAMETERS = (
        *_GFRED_TEMPORAL_PARAMETERS,
        *Blackbody.PARAMETERS,
    )

    DESCRIPTION = "Generalized-FRED temporal profile coupled with a blackbody SED."

    def __init__(self, bundle: FilterBundle):
        self._sed = Blackbody()
        super().__init__(bundle)

    def _evaluate_sed(self, frequency_hz: np.ndarray, parameters: dict) -> np.ndarray:
        return self._sed.evaluate(frequency_hz, parameters)

    def _evaluate_temporal(self, time_s: np.ndarray, parameters: dict) -> np.ndarray:
        return _gfred_shape(
            time_s,
            parameters["t_0"],
            parameters["tau_r"],
            parameters["tau_d"],
            parameters["nu_r"],
            parameters["nu_d"],
        )


# ---------------------------------------------------------------------------
# GaussianBlackbodyModel
# ---------------------------------------------------------------------------
class GaussianBlackbodyModel(CoupledOpticalModel):
    r"""
    Gaussian temporal envelope with a blackbody SED.

    .. math::

        \phi(t) = \exp\!\left[-\frac{1}{2}\left(\frac{t - t_\mathrm{peak}}{\sigma_t}\right)^2\right]

    Parameters
    ----------
    bundle : FilterBundle
        Filter bundle providing the common frequency grid and weight matrix.
    """

    PARAMETERS = (
        *_GAUSSIAN_TEMPORAL_PARAMETERS,
        *Blackbody.PARAMETERS,
    )

    DESCRIPTION = "Gaussian temporal pulse coupled with a blackbody SED."

    def __init__(self, bundle: FilterBundle):
        self._sed = Blackbody()
        super().__init__(bundle)

    def _evaluate_sed(self, frequency_hz: np.ndarray, parameters: dict) -> np.ndarray:
        return self._sed.evaluate(frequency_hz, parameters)

    def _evaluate_temporal(self, time_s: np.ndarray, parameters: dict) -> np.ndarray:
        return _gaussian_shape(time_s, parameters["t_peak"], parameters["sigma_t"])
