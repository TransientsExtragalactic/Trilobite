r"""
Concrete one-zone accretion disk models.

Three model classes are provided, each encoding a distinct thermal physics
closure.  All three support runtime-configurable opacity and an optional
power-law fallback mass supply:

See Also
--------
:mod:`triceratops.dynamics.accretion.one_zone.base` :
    Abstract base class and result container.
:ref:`one_zone_disk`, :ref:`one_zone_disk_theory`
"""

from typing import Any

import numpy as np

from triceratops.dynamics.accretion.one_zone._typing import _RunParams
from triceratops.dynamics.accretion.one_zone.base import OneZoneAccretionDiskBase
from triceratops.physics_utils import IdealGasEOS, RadiativeIdealGas
from triceratops.radiation.opacity.utils import get_opacity

from ._closure_specs import (
    ADV_FB_CYTHON_FIELD_MAP,
    ADV_FB_RESULT_FIELDS,
    BASE_INITIAL_CONDITIONS,
    FALLBACK_CYTHON_FIELD_MAP,
    FALLBACK_RESULT_FIELDS,
    FALLBACK_RUNTIME_PARAMETERS,
)


# ================================================================== #
# Concrete Models                                                    #
# ================================================================== #
class GasPressureDisk(OneZoneAccretionDiskBase):
    r"""
    One-zone disk model with gas pressure and configurable opacity.

    The simplest physically self-consistent closure within the one-zone
    framework.  The midplane temperature is solved **analytically** when
    electron-scattering opacity is active, making this the fastest closure.

    For alternative opacity models, the temperature is determined by root finding
    under the condition that the disk reach thermal stability at each timestep. Thus,

    .. math::

        Q^{+} = Q^{-}.

    Examples
    --------
    .. code-block:: python

        from astropy import constants as const
        from astropy import units as u
        from triceratops.dynamics.accretion.one_zone import (
            GasPressureDisk,
        )

        disk = GasPressureDisk(mu=0.62)
        result = disk.solve(
            initial_conditions={
                "M_D_0": 0.1 * const.M_sun,
                "J_D_0": 1.5e49 * u.g * u.cm**2 / u.s,
            },
            runtime_parameters={
                "M_BH": 3.0 * const.M_sun,
                "R_in": 3.0e6 * u.cm,
                "alpha": 0.1,
            },
            t_span=(1.0e6 * u.s, 5.0e9 * u.s),
        )

    See Also
    --------
    FullPressureDisk :
        Sibling with combined gas + radiation pressure.
    """

    # ================================================== #
    # Model-specific parameters and metadata             #
    # ================================================== #
    # Each of these is used to declare the parameters, outputs, initial conditions, etc.
    # of the model.  See the base class for details on how these are used.
    #
    # In this class, to permit specifying if fallback should be included
    # or not, we use the FALLBACK parameters and simply ignore the relevant
    # parameters when fallback is not turned on. This saves us a bit
    # of namespace complexity.
    CONTEXT_PARAMETERS: dict = {
        "mu": {"description": "Mean molecular weight", "base_units": None, "default": 0.6},
        "opacity": {"description": "Opacity model name", "base_units": None, "default": "electron_scattering"},
        "fallback": {"description": "Enable power-law fallback supply", "base_units": None, "default": False},
    }
    RUNTIME_PARAMETERS: dict = FALLBACK_RUNTIME_PARAMETERS
    INITIAL_CONDITIONS: dict = BASE_INITIAL_CONDITIONS
    RESULT_FIELDS: dict = FALLBACK_RESULT_FIELDS
    CYTHON_FIELD_MAP: dict = FALLBACK_CYTHON_FIELD_MAP

    # ================================================== #
    # Initialization and Cython closure construction     #
    # ================================================== #
    def __init__(self, mu: float = 0.6, opacity: str = "electron_scattering", fallback: bool = False):
        """
        Instantiate the class.

        Parameters
        ----------
        mu : float, optional
            Mean molecular weight of the disk gas (dimensionless).
            Default ``0.6``.
        opacity : str or GreyOpacityLaw, optional
            Opacity model.  Accepted strings: ``"electron_scattering"`` (default),
            ``"kramers_ff"``, ``"kramers_bf"``, ``"kramers"``,
            ``"kramers_ff_es"``, ``"kramers_bf_es"``, ``"kramers_es"``.
            A :class:`~triceratops.radiation.opacity.base.GreyOpacityLaw` instance
            may also be passed directly.
        fallback : bool, optional
            If ``True``, enable a power-law debris-stream mass supply.  The
            runtime parameters ``M_fb_0``, ``t_fb``, and ``beta_fb`` must then
            be provided to :meth:`solve`.  Default ``False``.
        """
        # Callback to the superclass to get things setup. This ensures
        # the context parameters are generated.
        super().__init__(mu=mu, opacity=opacity, fallback=fallback)
        self.fallback = fallback

        # Instantiate an equation of state. THIS DOESN'T TIE TO C. We provide this
        # as a helper for the python-side convenience functions, not because it does
        # anything at the C-level.
        self.eos = IdealGasEOS(mu=self._context_parameters["mu"])

        # Generate the opacity.
        try:
            self.opacity = get_opacity(opacity)
        except Exception as e:
            raise ValueError(f"Error initializing opacity: {e}") from e

    # ================================================== #
    # C-LEVEL CLOSURE CONSTRUCTION                       #
    # ================================================== #
    def _build_cython_closure(self) -> Any:
        from triceratops.dynamics.accretion.one_zone.models._igP import FullPressureClosure

        _closure = FullPressureClosure(
            gas_pressure_only=True,
            with_fallback=self.fallback,
            mu=self._context_parameters["mu"],
        )
        _closure.opacity = self.opacity
        return _closure

    def _compute_derived_result_fields(self, result_array: np.ndarray, run_params: _RunParams) -> dict:
        n_steps = result_array.shape[1]
        if not self._context_parameters["fallback"]:
            return {"mdot_fb": np.zeros(n_steps)}
        t = result_array[1, :]
        M_fb_0 = np.exp(run_params["log_M_fb_0"])
        t_fb = np.exp(run_params["log_t_fb"])
        beta = run_params["beta_fb"]
        return {"mdot_fb": M_fb_0 * (t / t_fb) ** (-beta)}


class FullPressureDisk(OneZoneAccretionDiskBase):
    r"""
    One-zone disk model with combined gas and radiation pressure.

    The midplane temperature is solved **iteratively** at each timestep
    via bracket expansion + Brent's method.

    Examples
    --------
    .. code-block:: python

        from astropy import constants as const
        from astropy import units as u
        from triceratops.dynamics.accretion.one_zone import (
            FullPressureDisk,
        )

        disk = FullPressureDisk(mu=0.62)
        result = disk.solve(
            initial_conditions={
                "M_D_0": 0.5 * const.M_sun,
                "J_D_0": 1.5e49 * u.g * u.cm**2 / u.s,
            },
            runtime_parameters={
                "M_BH": 3.0 * const.M_sun,
                "R_in": 3.0e6 * u.cm,
                "alpha": 0.1,
            },
            t_span=(1.0e6 * u.s, 5.0e9 * u.s),
        )

    See Also
    --------
    GasPressureDisk :
        Simpler closure with analytic temperature solve.
    AdvectiveDisk :
        Extended closure with advective cooling term.
    """

    # ================================================== #
    # Model-specific parameters and metadata             #
    # ================================================== #
    # Each of these is used to declare the parameters, outputs, initial conditions, etc.
    # of the model.  See the base class for details on how these are used.
    #
    # In this class, to permit specifying if fallback should be included
    # or not, we use the FALLBACK parameters and simply ignore the relevant
    # parameters when fallback is not turned on. This saves us a bit
    # of namespace complexity.
    CONTEXT_PARAMETERS: dict = {
        "mu": {"description": "Mean molecular weight", "base_units": None, "default": 0.6},
        "opacity": {"description": "Opacity model name", "base_units": None, "default": "electron_scattering"},
        "fallback": {"description": "Enable power-law fallback supply", "base_units": None, "default": False},
    }
    RUNTIME_PARAMETERS: dict = FALLBACK_RUNTIME_PARAMETERS
    INITIAL_CONDITIONS: dict = BASE_INITIAL_CONDITIONS
    RESULT_FIELDS: dict = FALLBACK_RESULT_FIELDS
    CYTHON_FIELD_MAP: dict = FALLBACK_CYTHON_FIELD_MAP

    # ================================================== #
    # Initialization and Cython closure construction     #
    # ================================================== #
    def __init__(self, mu: float = 0.6, opacity: str = "electron_scattering", fallback: bool = False):
        """
        Instantiate the class.

        Parameters
        ----------
        mu : float, optional
            Mean molecular weight of the disk gas (dimensionless).
            Default ``0.6``.
        opacity : str or GreyOpacityLaw, optional
            Opacity model.  Accepted strings: ``"electron_scattering"`` (default),
            ``"kramers_ff"``, ``"kramers_bf"``, ``"kramers"``,
            ``"kramers_ff_es"``, ``"kramers_bf_es"``, ``"kramers_es"``.
            A :class:`~triceratops.radiation.opacity.base.GreyOpacityLaw` instance
            may also be passed directly.
        fallback : bool, optional
            If ``True``, enable a power-law debris-stream mass supply.  The
            runtime parameters ``M_fb_0``, ``t_fb``, and ``beta_fb`` must then
            be provided to :meth:`solve`.  Default ``False``.
        """
        # Callback to the superclass to get things setup. This ensures
        # the context parameters are generated.
        super().__init__(mu=mu, opacity=opacity, fallback=fallback)
        self.fallback = fallback

        # Instantiate an equation of state. THIS DOESN'T TIE TO C. We provide this
        # as a helper for the python-side convenience functions, not because it does
        # anything at the C-level.
        self.eos = RadiativeIdealGas(mu=self._context_parameters["mu"])

        # Generate the opacity.
        try:
            self.opacity = get_opacity(opacity)
        except Exception as e:
            raise ValueError(f"Error initializing opacity: {e}") from e

    # ================================================== #
    # C-LEVEL CLOSURE CONSTRUCTION                       #
    # ================================================== #
    def _build_cython_closure(self) -> Any:
        from triceratops.dynamics.accretion.one_zone.models._igP import FullPressureClosure

        _closure = FullPressureClosure(
            gas_pressure_only=False,
            with_fallback=self.fallback,
            mu=self._context_parameters["mu"],
        )
        _closure.opacity = self.opacity
        return _closure

    def _compute_derived_result_fields(self, result_array: np.ndarray, run_params: _RunParams) -> dict:
        n_steps = result_array.shape[1]
        if not self._context_parameters["fallback"]:
            return {"mdot_fb": np.zeros(n_steps)}
        t = result_array[1, :]
        M_fb_0 = np.exp(run_params["log_M_fb_0"])
        t_fb = np.exp(run_params["log_t_fb"])
        beta = run_params["beta_fb"]
        return {"mdot_fb": M_fb_0 * (t / t_fb) ** (-beta)}


# ================================================================== #
# Advective Disks                                                    #
# ================================================================== #
# We now add in advection as an additional source of cooling. This is commonly
# used to emulate the inefficient cooling present in many super-Eddington
# disks.


class AdvectiveDisk(OneZoneAccretionDiskBase):
    r"""
    One-zone disk with gas + radiation pressure and advective cooling.

    Extends :class:`FullPressureDisk` by including an advective energy
    transport term in the energy balance:

    .. math::

        q_{\rm visc} = q_{\rm rad} + q_{\rm adv}.

    The dimensionless entropy gradient parameter :math:`\xi` controls the
    strength of advection.  Setting :math:`\xi \to 0` recovers the
    non-advective :class:`FullPressureDisk` limit.

    Parameters
    ----------
    mu : float, optional
        Mean molecular weight of the disk gas (dimensionless).
        Default ``0.6``.
    xi : float, optional
        Entropy gradient parameter (dimensionless, > 0).  Default ``0.5``.
    opacity : str or GreyOpacityLaw, optional
        Opacity model.  Accepted strings: ``"electron_scattering"`` (default),
        ``"kramers_ff"``, ``"kramers_bf"``, ``"kramers"``,
        ``"kramers_ff_es"``, ``"kramers_bf_es"``, ``"kramers_es"``.
        A :class:`~triceratops.radiation.opacity.base.GreyOpacityLaw` instance
        may also be passed directly.
    fallback : bool, optional
        If ``True``, enable a power-law debris-stream mass supply.
        Default ``False``.

    Examples
    --------
    .. code-block:: python

        from astropy import constants as const
        from astropy import units as u
        from triceratops.dynamics.accretion.one_zone import (
            AdvectiveDisk,
        )

        disk = AdvectiveDisk(mu=0.62, xi=0.5)
        result = disk.solve(
            initial_conditions={
                "M_D_0": 0.1 * const.M_sun,
                "J_D_0": 1.5e49 * u.g * u.cm**2 / u.s,
            },
            runtime_parameters={
                "M_BH": 3.0 * const.M_sun,
                "R_in": 3.0e6 * u.cm,
                "alpha": 0.1,
            },
            t_span=(1.0e6 * u.s, 5.0e9 * u.s),
        )
        print(
            result.data["Q_adv"] / result.data["Q_visc"]
        )

    See Also
    --------
    FullPressureDisk :
        Non-advective sibling (xi → 0 limit).
    """

    # ================================================== #
    # Model-specific parameters and metadata             #
    # ================================================== #
    # Each of these is used to declare the parameters, outputs, initial conditions, etc.
    # of the model.  See the base class for details on how these are used.
    #
    # In this class, to permit specifying if fallback should be included
    # or not, we use the FALLBACK parameters and simply ignore the relevant
    # parameters when fallback is not turned on. This saves us a bit
    # of namespace complexity.
    CONTEXT_PARAMETERS: dict = {
        "mu": {"description": "Mean molecular weight", "base_units": None, "default": 0.6},
        "xi": {"description": "Advection parameter", "base_units": None, "default": 0.5},
        "opacity": {"description": "Opacity model name", "base_units": None, "default": "electron_scattering"},
        "fallback": {"description": "Enable power-law fallback supply", "base_units": None, "default": False},
    }
    RUNTIME_PARAMETERS: dict = FALLBACK_RUNTIME_PARAMETERS
    INITIAL_CONDITIONS: dict = BASE_INITIAL_CONDITIONS
    RESULT_FIELDS: dict = ADV_FB_RESULT_FIELDS
    CYTHON_FIELD_MAP: dict = ADV_FB_CYTHON_FIELD_MAP

    def __init__(
        self,
        mu: float = 0.6,
        xi: float = 1.5,
        opacity: str = "electron_scattering",
        fallback: bool = False,
    ):
        # Callback to the superclass to get things setup. This ensures
        # the context parameters are generated.
        super().__init__(mu=mu, xi=xi, opacity=opacity, fallback=fallback)
        self.fallback = fallback

        # Instantiate an equation of state. THIS DOESN'T TIE TO C. We provide this
        # as a helper for the python-side convenience functions, not because it does
        # anything at the C-level.
        self.eos = RadiativeIdealGas(mu=self._context_parameters["mu"])

        # Generate the opacity.
        try:
            self.opacity = get_opacity(opacity)
        except Exception as e:
            raise ValueError(f"Error initializing opacity: {e}") from e

    # ================================================== #
    # C-LEVEL CLOSURE CONSTRUCTION                       #
    # ================================================== #
    def _build_cython_closure(self) -> Any:
        from .models._igP_adv import AdvectiveClosure

        _closure = AdvectiveClosure(
            with_fallback=self.fallback,
            mu=self._context_parameters["mu"],
            xi=self._context_parameters["xi"],
        )
        _closure.opacity = self.opacity
        return _closure

    def _compute_derived_result_fields(self, result_array: np.ndarray, run_params: _RunParams) -> dict:
        n_steps = result_array.shape[1]
        if not self._context_parameters["fallback"]:
            return {"mdot_fb": np.zeros(n_steps)}
        t = result_array[1, :]
        M_fb_0 = np.exp(run_params["log_M_fb_0"])
        t_fb = np.exp(run_params["log_t_fb"])
        beta = run_params["beta_fb"]
        return {"mdot_fb": M_fb_0 * (t / t_fb) ** (-beta)}


# ================================================================== #
# Backward-compatible aliases                                        #
# ================================================================== #
# These names are preserved so that existing code continues to work.
# Note: the *_fb aliases refer to the same class as their non-fb
# counterparts — pass ``fallback=True`` to enable the fallback supply.

gP_esDisk = GasPressureDisk
igP_esDisk = FullPressureDisk
igP_es_advDisk = AdvectiveDisk

# Fallback aliases: identical classes — callers must pass fallback=True.
gP_es_fbDisk = GasPressureDisk
igP_es_fbDisk = FullPressureDisk
igP_es_adv_fbDisk = AdvectiveDisk
