r"""
Concrete one-zone accretion disk models.

Three model classes are provided, each encoding a distinct thermal physics
closure.  All three support runtime-configurable opacity and an optional
power-law fallback mass supply:

===================  ================================================
Class                Physics
===================  ================================================
:class:`GasPressureDisk`   Gas pressure only; analytic T solve (fast).
:class:`FullPressureDisk`  Gas + radiation pressure; iterative T solve.
:class:`AdvectiveDisk`     Gas + radiation pressure + advective cooling.
===================  ================================================

**Opacity**

The ``opacity`` context parameter selects the opacity model at construction
time.  Currently supported values:

* ``"electron_scattering"`` (default) — constant ES opacity, κ = 0.34 cm² g⁻¹.

Stubs (not yet implemented):

* ``"kramers_ff"``  — free-free Kramers opacity.
* ``"kramers_bf"``  — bound-free Kramers opacity.

**Fallback supply**

Setting ``fallback=True`` installs a power-law debris-stream source term.
The three extra runtime parameters ``M_fb_0``, ``t_fb``, and ``beta_fb``
must then be supplied to :meth:`~.base.OneZoneAccretionDiskBase.solve`.

**Backward-compatible aliases**

The original six-class API is preserved through module-level aliases::

    gP_esDisk = GasPressureDisk  # fallback=False
    igP_esDisk = FullPressureDisk  # fallback=False
    igP_es_advDisk = AdvectiveDisk  # fallback=False
    gP_es_fbDisk = (
        GasPressureDisk  # must pass fallback=True
    )
    igP_es_fbDisk = (
        FullPressureDisk  # must pass fallback=True
    )
    igP_es_adv_fbDisk = (
        AdvectiveDisk  # must pass fallback=True
    )

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
from triceratops.physics_utils import IdealGasEOS

# ================================================================== #
# Shared parameter declarations                                      #
# ================================================================== #

_DEFAULT_BETA_FB: float = 5.0 / 3.0
_DEFAULT_XI: float = 0.5

_BASE_RUNTIME_PARAMETERS: dict = {
    "M_BH": {
        "description": "Black hole mass",
        "base_units": "g",
        "default": None,
        "log_transform": True,
    },
    "R_in": {
        "description": "Inner truncation radius",
        "base_units": "cm",
        "default": None,
        "log_transform": True,
    },
    "alpha": {
        "description": "Shakura-Sunyaev alpha",
        "base_units": None,
        "default": None,
        "log_transform": False,
    },
    # Fallback parameters — always declared; only required when fallback=True.
    # Default values are harmless dummies: they are packed into the parameter
    # array regardless of fallback mode, but the Cython source function is
    # NULL when fallback=False, so these values are never read.
    "M_fb_0": {
        "description": "Fallback mass supply rate at t_fb",
        "base_units": "g/s",
        "default": 1.0,  # dummy; user must override when fallback=True
        "log_transform": True,
    },
    "t_fb": {
        "description": "Fallback reference time",
        "base_units": "s",
        "default": 1.0,  # dummy; user must override when fallback=True
        "log_transform": True,
    },
    "beta_fb": {
        "description": "Fallback power-law index",
        "base_units": None,
        "default": _DEFAULT_BETA_FB,
        "log_transform": False,
    },
}

_BASE_INITIAL_CONDITIONS: dict = {
    "M_D_0": {
        "description": "Initial disk mass",
        "base_units": "g",
        "default": None,
        "log_transform": True,
    },
    "J_D_0": {
        "description": "Initial disk angular momentum",
        "base_units": "g*cm**2/s",
        "default": None,
        "log_transform": True,
    },
}

_BASE_RESULT_FIELDS: dict = {
    "R_D": {"description": "Disk outer radius", "units": "cm"},
    "Sigma": {"description": "Surface density", "units": "g/cm**2"},
    "Omega": {"description": "Angular velocity", "units": "rad/s"},
    "T_eff": {"description": "Effective temperature", "units": "K"},
    "T_c": {"description": "Central temperature", "units": "K"},
    "tau": {"description": "Optical depth", "units": None},
    "c_s": {"description": "Sound speed", "units": "cm/s"},
    "nu": {"description": "Kinematic viscosity", "units": "cm**2/s"},
    "t_visc": {"description": "Viscous timescale", "units": "s"},
    "Q_visc": {"description": "Viscous dissipation rate", "units": "erg cm-2 s-1"},
    "mdot": {"description": "Accretion rate", "units": "g/s"},
    "H": {"description": "Scale height", "units": "cm"},
    "H_over_R": {"description": "Aspect ratio", "units": None},
    "rho": {"description": "Midplane density", "units": "g/cm**3"},
    # Fallback field — always declared; only populated in data when fallback=True.
    "mdot_fb": {"description": "Fallback mass supply rate", "units": "g/s"},
}

# Row layout (N_RESULT_FIELDS = 20):
#   0=step_index  1=t  2=M  3=J  4=R  5=Sigma  6=Omega  7=T_eff  8=T_c
#   9=tau  10=cs  11=nu  12=q_visc  13=dM_dt  14=dJ_dt  15=dt
#   16=t_visc  17=H  18=H/R  19=rho
_BASE_CYTHON_FIELD_MAP: dict = {
    "R_D": 4,
    "Sigma": 5,
    "Omega": 6,
    "T_eff": 7,
    "T_c": 8,
    "tau": 9,
    "c_s": 10,
    "nu": 11,
    "Q_visc": 12,
    "mdot": 13,
    "t_visc": 16,
    "H": 17,
    "H_over_R": 18,
    "rho": 19,
    "mdot_fb": None,  # computed Python-side
}

_ADV_RESULT_FIELDS: dict = {
    **{k: v for k, v in _BASE_RESULT_FIELDS.items() if k not in ("mdot_fb",)},
    "Q_adv": {"description": "Advective cooling rate", "units": "erg cm-2 s-1"},
    "mdot_fb": {"description": "Fallback mass supply rate", "units": "g/s"},
}

# Row layout (ADV_N_RESULT_FIELDS = 21):
#   0=step_index  1=t  2=M  3=J  4=R  5=Sigma  6=Omega  7=T_eff  8=T_c
#   9=tau  10=cs  11=nu  12=q_visc  13=q_adv  14=dM_dt  15=dJ_dt
#   16=dt  17=t_visc  18=H  19=H/R  20=rho
_ADV_CYTHON_FIELD_MAP: dict = {
    "R_D": 4,
    "Sigma": 5,
    "Omega": 6,
    "T_eff": 7,
    "T_c": 8,
    "tau": 9,
    "c_s": 10,
    "nu": 11,
    "Q_visc": 12,
    "Q_adv": 13,
    "mdot": 14,
    "t_visc": 17,
    "H": 18,
    "H_over_R": 19,
    "rho": 20,
    "mdot_fb": None,  # computed Python-side
}


# ================================================================== #
# Helper: build a validated Cython closure with opacity installed    #
# ================================================================== #


def _build_closure_with_opacity(closure_cls, opacity: str, with_fallback: bool):
    """Instantiate *closure_cls*, install the requested opacity, return it."""
    from triceratops.dynamics.accretion.one_zone.physics._opacity import get_kappa_ptr

    closure = closure_cls(with_fallback=with_fallback)
    fn_ptr, data_ptr = get_kappa_ptr(opacity)
    closure.set_opacity(fn_ptr, data_ptr)
    return closure


# ================================================================== #
# Concrete Models                                                    #
# ================================================================== #


class GasPressureDisk(OneZoneAccretionDiskBase):
    r"""
    One-zone disk model with gas pressure and configurable opacity.

    The simplest physically self-consistent closure within the one-zone
    framework.  The midplane temperature is solved **analytically** when
    electron-scattering opacity is active, making this the fastest closure.

    Parameters
    ----------
    mu : float, optional
        Mean molecular weight of the disk gas (dimensionless).
        Default ``0.6``.
    opacity : str, optional
        Opacity model name.  Default ``"electron_scattering"``.
        See :func:`~.physics._opacity.get_kappa_ptr` for available names.
    fallback : bool, optional
        If ``True``, enable a power-law debris-stream mass supply.  The
        runtime parameters ``M_fb_0``, ``t_fb``, and ``beta_fb`` must then
        be provided to :meth:`solve`.  Default ``False``.

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

    _A: float = 1.62
    _B: float = 1.33
    _F: float = 1.6

    CONTEXT_PARAMETERS: dict = {
        "mu": {"description": "Mean molecular weight", "base_units": None, "default": 0.6},
        "opacity": {"description": "Opacity model name", "base_units": None, "default": "electron_scattering"},
        "fallback": {"description": "Enable power-law fallback supply", "base_units": None, "default": False},
    }
    RUNTIME_PARAMETERS: dict = _BASE_RUNTIME_PARAMETERS
    INITIAL_CONDITIONS: dict = _BASE_INITIAL_CONDITIONS
    RESULT_FIELDS: dict = _BASE_RESULT_FIELDS
    CYTHON_FIELD_MAP: dict = _BASE_CYTHON_FIELD_MAP

    def __init__(self, mu: float = 0.6, opacity: str = "electron_scattering", fallback: bool = False):
        super().__init__(mu=mu, opacity=opacity, fallback=fallback)
        self.eos = IdealGasEOS(mu=self._context_parameters["mu"])

    def _build_cython_closure(self) -> Any:
        from triceratops.dynamics.accretion.one_zone.models._gP import gPClosure

        return _build_closure_with_opacity(
            gPClosure,
            self._context_parameters["opacity"],
            self._context_parameters["fallback"],
        )

    def _pack_cython_parameters(self, run_params: _RunParams) -> np.ndarray:
        base = self._pack_base_cython_parameters(run_params)
        extra = np.array(
            [
                np.exp(run_params["log_M_fb_0"]),
                np.exp(run_params["log_t_fb"]),
                run_params["beta_fb"],
            ],
            dtype=np.float64,
        )
        return np.concatenate([base, extra])

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

    Parameters
    ----------
    mu : float, optional
        Mean molecular weight of the disk gas (dimensionless).
        Default ``0.6``.
    opacity : str, optional
        Opacity model name.  Default ``"electron_scattering"``.
    fallback : bool, optional
        If ``True``, enable a power-law debris-stream mass supply.
        Default ``False``.

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

    _A: float = 1.62
    _B: float = 1.33
    _F: float = 1.6

    CONTEXT_PARAMETERS: dict = {
        "mu": {"description": "Mean molecular weight", "base_units": None, "default": 0.6},
        "opacity": {"description": "Opacity model name", "base_units": None, "default": "electron_scattering"},
        "fallback": {"description": "Enable power-law fallback supply", "base_units": None, "default": False},
    }
    RUNTIME_PARAMETERS: dict = _BASE_RUNTIME_PARAMETERS
    INITIAL_CONDITIONS: dict = _BASE_INITIAL_CONDITIONS
    RESULT_FIELDS: dict = _BASE_RESULT_FIELDS
    CYTHON_FIELD_MAP: dict = _BASE_CYTHON_FIELD_MAP

    def __init__(self, mu: float = 0.6, opacity: str = "electron_scattering", fallback: bool = False):
        super().__init__(mu=mu, opacity=opacity, fallback=fallback)
        self.eos = IdealGasEOS(mu=self._context_parameters["mu"])

    def _build_cython_closure(self) -> Any:
        from triceratops.dynamics.accretion.one_zone.models._igP import igPClosure

        return _build_closure_with_opacity(
            igPClosure,
            self._context_parameters["opacity"],
            self._context_parameters["fallback"],
        )

    def _pack_cython_parameters(self, run_params: _RunParams) -> np.ndarray:
        base = self._pack_base_cython_parameters(run_params)
        extra = np.array(
            [
                np.exp(run_params["log_M_fb_0"]),
                np.exp(run_params["log_t_fb"]),
                run_params["beta_fb"],
            ],
            dtype=np.float64,
        )
        return np.concatenate([base, extra])

    def _compute_derived_result_fields(self, result_array: np.ndarray, run_params: _RunParams) -> dict:
        n_steps = result_array.shape[1]
        if not self._context_parameters["fallback"]:
            return {"mdot_fb": np.zeros(n_steps)}
        t = result_array[1, :]
        M_fb_0 = np.exp(run_params["log_M_fb_0"])
        t_fb = np.exp(run_params["log_t_fb"])
        beta = run_params["beta_fb"]
        return {"mdot_fb": M_fb_0 * (t / t_fb) ** (-beta)}


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
    opacity : str, optional
        Opacity model name.  Default ``"electron_scattering"``.
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

    _A: float = 1.62
    _B: float = 1.33
    _F: float = 1.6

    CONTEXT_PARAMETERS: dict = {
        "mu": {"description": "Mean molecular weight", "base_units": None, "default": 0.6},
        "xi": {"description": "Entropy gradient parameter", "base_units": None, "default": _DEFAULT_XI},
        "opacity": {"description": "Opacity model name", "base_units": None, "default": "electron_scattering"},
        "fallback": {"description": "Enable power-law fallback supply", "base_units": None, "default": False},
    }
    RUNTIME_PARAMETERS: dict = {
        **_BASE_RUNTIME_PARAMETERS,
    }
    INITIAL_CONDITIONS: dict = _BASE_INITIAL_CONDITIONS
    RESULT_FIELDS: dict = _ADV_RESULT_FIELDS
    CYTHON_FIELD_MAP: dict = _ADV_CYTHON_FIELD_MAP

    def __init__(
        self,
        mu: float = 0.6,
        xi: float = _DEFAULT_XI,
        opacity: str = "electron_scattering",
        fallback: bool = False,
    ):
        super().__init__(mu=mu, xi=xi, opacity=opacity, fallback=fallback)
        self.eos = IdealGasEOS(mu=self._context_parameters["mu"])

    def _build_cython_closure(self) -> Any:
        from triceratops.dynamics.accretion.one_zone.models._igP_adv import igPAdvClosure

        return _build_closure_with_opacity(
            igPAdvClosure,
            self._context_parameters["opacity"],
            self._context_parameters["fallback"],
        )

    def _pack_cython_parameters(self, run_params: _RunParams) -> np.ndarray:
        base = self._pack_base_cython_parameters(run_params)
        xi = self._context_parameters["xi"]
        extra = np.array(
            [
                xi,
                np.exp(run_params["log_M_fb_0"]),
                np.exp(run_params["log_t_fb"]),
                run_params["beta_fb"],
            ],
            dtype=np.float64,
        )
        return np.concatenate([base, extra])

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
