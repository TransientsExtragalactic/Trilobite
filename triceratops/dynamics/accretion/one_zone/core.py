r"""
Concrete, analytically closed one-zone accretion disk models.

This module provides a selection of analytically closed one-zone disk models which
can be used as drop-in closures for the :ref:`one_zone_disk` pipeline.

See Also
--------
:mod:`triceratops.dynamics.accretion.one_zone.base` :
    Abstract base class and result container.
:mod:`triceratops.dynamics.accretion.utils` :
    Low-level disk utility functions used by the closure pipeline.
:ref:`one_zone_disk`, :ref:`one_zone_disk_theory`
"""

from typing import Any

import numpy as np

from triceratops.dynamics.accretion.one_zone._typing import _RunParams, _SpecDict
from triceratops.dynamics.accretion.one_zone.base import OneZoneAccretionDiskBase
from triceratops.physics_utils import IdealGasEOS

# ================================================================== #
# Ideal Gas Disks                                                    #
# ================================================================== #
# These models provide one-zone disks with a few different combinations of pressure and
# opacity specifications.


class GasPressureElectronScatteringDisk(OneZoneAccretionDiskBase):
    r"""
    One-zone disk model dominated by gas pressure with electron-scattering opacity.

    This is the simplest physically self-consistent closure within the one-zone
    framework and serves as the canonical reference implementation.  It follows
    the formulation of :footcite:t:`metzgerTimeDependentModelsAccretion2008`
    and assumes:

    - **Opacity**: pure electron scattering,
      :math:`\kappa_{\rm es} = 0.34\,\text{cm}^2\,\text{g}^{-1}`.
    - **Pressure**: ideal gas,
      :math:`P = \rho k_B T_c / (\mu m_p)`.
    - **Viscosity**: Shakura-Sunyaev :math:`\alpha`-prescription,
      :math:`\nu = \alpha c_s^2 / \Omega`.
    - **Cooling**: optically thick blackbody radiation,
      :math:`T_c = (3\tau/4)^{1/4} T_{\rm eff}`.

    **One-zone geometry**

    The disk is reduced to a single spatial zone by introducing two
    dimensionless radial-averaging constants :math:`A` and :math:`B` that
    relate the area-averaged surface density and the angular-momentum-weighted
    radius to the total disk mass :math:`M_D` and angular momentum :math:`J_D`:

    .. math::

        \Sigma &= \frac{M_D}{\pi A\, R_D^2}, \\[4pt]
        J_D    &= \frac{B}{A}\, M_D \sqrt{G M_{\rm BH} R_D}.

    The ratio :math:`\xi \equiv B/A` allows the disk outer radius
    :math:`R_D` to be recovered from the state vector alone:

    .. math::

        R_D = \frac{J_D^2}{\xi^2\, M_D^2\, G M_{\rm BH}}.

    **Thermodynamic closure**

    .. note:: The thermodynamic closure is solved analytically at each timestep by the Cython integrator.

    **Angular-momentum evolution**

    Accreted matter leaves the disk through the inner edge carrying the
    Keplerian specific angular momentum
    :math:`\ell_{\rm in} = \sqrt{G M_{\rm BH} R_{\rm in}}`.
    The angular-momentum sink is therefore:

    .. math::

        \dot{J}_{\rm visc}
        = -f_{\rm disk}\,\frac{M_D}{t_{\rm visc}}
          \,\sqrt{G M_{\rm BH} R_{\rm in}}.

    Because :math:`\ell_{\rm in} \ll \ell_D = J_D / M_D` for
    :math:`R_{\rm in} \ll R_D`, angular momentum drains more slowly than mass
    and the disk outer radius :math:`R_D` grows over time as material
    accretes inward — the standard viscous spreading behaviour.

    Parameters
    ----------
    mu : float, optional
        Mean molecular weight of the disk gas (dimensionless).
        Default ``0.6``, appropriate for a solar-composition fully-ionised
        plasma (:math:`\approx 0.62` for solar abundances).

    Notes
    -----
    The disk geometry constants :math:`A = 1.62`, :math:`B = 1.33`, and
    :math:`F_0 = 1.6` are fixed at the Cython layer as compile-time constants
    (from :footcite:t:`metzgerTimeDependentModelsAccretion2008`) and are
    accessible as class attributes :attr:`_A`, :attr:`_B`, :attr:`_F`.

    Examples
    --------
    Construct a disk model, solve for 5 viscous timescales, and inspect the
    central temperature evolution:

    .. code-block:: python

        import numpy as np
        from astropy import constants as const
        from astropy import units as u
        from triceratops.dynamics.accretion.one_zone import (
            GasPressureElectronScatteringDisk,
        )

        disk = GasPressureElectronScatteringDisk(mu=0.62)

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
            max_steps=50_000,
        )

        data = result.data
        print(data["T_c"].to("K"))
        print(data["R_D"].to("cm"))

    See Also
    --------
    OneZoneAccretionDiskBase :
        Abstract base class describing the full closure pipeline and the
        declarative parameter system.
    :func:`~triceratops.dynamics.accretion.utils.alpha_viscosity` :
        Public interface to the Shakura-Sunyaev viscosity formula used here.
    :ref:`one_zone_disk`, :ref:`one_zone_disk_theory`

    References
    ----------
    .. footbibliography::
    """

    # ==================================================== #
    # Parameter Declarations                               #
    # ==================================================== #
    # Geometry constants are fixed Cython compile-time values; only mu is user-tunable.
    _A: float = 1.62  # Metzger+08 surface-density averaging constant
    _B: float = 1.33  # Metzger+08 angular-momentum averaging constant
    _F: float = 1.6  # Metzger+08 geometric mass-loss factor

    CONTEXT_PARAMETERS: dict[str, dict] = {
        "mu": {"description": "Mean molecular weight", "base_units": None, "default": 0.6},
    }
    RUNTIME_PARAMETERS: dict[str, dict] = {
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
    }

    INITIAL_CONDITIONS: dict[str, dict] = {
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

    RESULT_FIELDS: dict[str, dict] = {
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
    }

    # Maps RESULT_FIELDS keys to Cython result-array row indices (None = derived in Python).
    # Row layout (GAS_PRESSURE_ES_N_RESULT_FIELDS = 20):
    #   0=step_index  1=t        2=M        3=J        4=R        5=Sigma
    #   6=Omega       7=T_eff    8=T_c      9=tau      10=cs      11=nu
    #   12=q_visc     13=dM_dt   14=dJ_dt   15=dt      16=t_visc
    #   17=H          18=H/R     19=rho
    CYTHON_FIELD_MAP: dict = {
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
    }

    # ==================================================== #
    # Initialization                                       #
    # ==================================================== #

    def __init__(self, mu: float = 0.6):
        super().__init__(mu=mu)
        self.eos = IdealGasEOS(mu=self._context_parameters["mu"])

    # ==================================================== #
    # Serialization                                        #
    # ==================================================== #

    def to_spec_dict(self) -> _SpecDict:
        r"""
        Return a JSON-serialisable spec dict for this model instance.

        Returns
        -------
        dict
            JSON-serialisable specification dict with keys ``"target"`` and
            ``"mu"``.

        Examples
        --------
        .. code-block:: python

            spec = disk.to_spec_dict()
            import json

            print(json.dumps(spec, indent=2))

        See Also
        --------
        from_spec_dict :
            Class method that reconstructs the model from this dict.
        """
        return {
            "target": "triceratops.dynamics.accretion.one_zone.core:GasPressureElectronScatteringDisk",
            "mu": self._context_parameters["mu"],
        }

    @classmethod
    def from_spec_dict(cls, data: _SpecDict) -> "GasPressureElectronScatteringDisk":
        r"""
        Construct a :class:`GasPressureElectronScatteringDisk` from a spec dict.

        Parameters
        ----------
        data : dict
            Spec dict produced by :meth:`to_spec_dict`.  The ``"target"``
            key is silently ignored.

        Returns
        -------
        GasPressureElectronScatteringDisk

        Examples
        --------
        .. code-block:: python

            spec = disk.to_spec_dict()
            loaded = GasPressureElectronScatteringDisk.from_spec_dict(
                spec
            )
            assert np.isclose(loaded.eos.mu, disk.eos.mu)

        See Also
        --------
        to_spec_dict :
            Instance method that produces the dict this class method reads.
        """
        kwargs = {k: v for k, v in data.items() if k != "target"}
        return cls(**kwargs)

    # ==================================================== #
    # Cython Integration Interface                        #
    # ==================================================== #

    def _build_cython_closure(self) -> Any:
        """Return a GasPressureElectronScatteringClosure for the Cython integrator."""
        from triceratops.dynamics.accretion.one_zone._ideal_gas_closure import (
            GasPressureElectronScatteringClosure,
        )

        return GasPressureElectronScatteringClosure()

    def _pack_cython_parameters(self, run_params: _RunParams) -> np.ndarray:
        """Return ``[MBH (g), R_in (cm), alpha, mu]`` as a float64 array."""
        return self._pack_base_cython_parameters(run_params)


# ================================================================== #
# Gas + Radiation Pressure Disks                                     #
# ================================================================== #
class FullPressureElectronScatteringDisk(OneZoneAccretionDiskBase):
    r"""
    One-zone disk model with combined gas and radiation pressure, and electron-scattering opacity.

    Extends :class:`GasPressureElectronScatteringDisk` by including the
    radiation-pressure contribution to the equation of state.  The midplane
    temperature can no longer be found analytically; instead, the Cython layer
    solves the energy balance

    .. math::

        q^+(T_c) = q^-(T_c)

    iteratively at every timestep using bracket expansion followed by Brent's
    method (see :mod:`triceratops.math_utils._bracket_root_finder`).

    The model assumes:

    - **Opacity**: pure electron scattering,
      :math:`\kappa_{\rm es} = 0.34\,\text{cm}^2\,\text{g}^{-1}`.
    - **Pressure**: combined gas and radiation,
      :math:`c_s^2 = k_B T_c / (\mu m_p) + (a T_c^4 / 3) / \rho`, solved
      self-consistently for :math:`c_s` at each temperature trial.
    - **Viscosity**: Shakura-Sunyaev :math:`\alpha`-prescription,
      :math:`\nu = \alpha c_s^2 / \Omega`.
    - **Cooling**: optically thick blackbody radiation.

    **Regime of applicability**

    In the gas-pressure-dominated regime the results converge to those of
    :class:`GasPressureElectronScatteringDisk`.  Differences become significant
    when the ratio of radiation pressure to gas pressure,

    .. math::

        \beta^{-1} \equiv \frac{a T_c^4 / 3}{n k_B T_c}
                  \sim \frac{a T_c^3 \mu m_p}{3 \rho k_B},

    is no longer negligible, which occurs at high temperatures and low
    densities (high :math:`\alpha`, high :math:`\dot{M}`, or large disk radius).

    Parameters
    ----------
    mu : float, optional
        Mean molecular weight of the disk gas (dimensionless).
        Default ``0.6``.

    Notes
    -----
    The disk geometry constants :math:`A = 1.62`, :math:`B = 1.33`, and
    :math:`F_0 = 1.6` are fixed Cython compile-time constants.

    Examples
    --------
    .. code-block:: python

        from astropy import constants as const
        from astropy import units as u
        from triceratops.dynamics.accretion.one_zone import (
            FullPressureElectronScatteringDisk,
        )

        disk = FullPressureElectronScatteringDisk(mu=0.62)

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
            max_steps=50_000,
        )

        data = result.data
        print(data["T_c"].to("K"))

    See Also
    --------
    GasPressureElectronScatteringDisk :
        Simpler closure with analytic temperature solve (gas pressure only).
    OneZoneAccretionDiskBase :
        Abstract base class describing the full closure pipeline.

    References
    ----------
    .. footbibliography::
    """

    # ==================================================== #
    # Parameter Declarations                               #
    # ==================================================== #

    _A: float = 1.62
    _B: float = 1.33
    _F: float = 1.6

    CONTEXT_PARAMETERS: dict[str, dict] = {
        "mu": {"description": "Mean molecular weight", "base_units": None, "default": 0.6},
    }
    RUNTIME_PARAMETERS: dict[str, dict] = {
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
    }

    INITIAL_CONDITIONS: dict[str, dict] = {
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

    RESULT_FIELDS: dict[str, dict] = {
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
    }

    # Row layout identical to GasPressureElectronScatteringClosure (same writer).
    CYTHON_FIELD_MAP: dict = {
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
    }

    # ==================================================== #
    # Initialization                                       #
    # ==================================================== #

    def __init__(self, mu: float = 0.6):
        super().__init__(mu=mu)
        self.eos = IdealGasEOS(mu=self._context_parameters["mu"])

    # ==================================================== #
    # Serialization                                        #
    # ==================================================== #

    def to_spec_dict(self) -> _SpecDict:
        """Return a JSON-serialisable spec dict for this model instance.

        Returns
        -------
        dict
            Spec dict with keys ``"target"`` and ``"mu"``.

        See Also
        --------
        from_spec_dict :
            Reconstructs the model from this dict.
        """
        return {
            "target": "triceratops.dynamics.accretion.one_zone.core:FullPressureElectronScatteringDisk",
            "mu": self._context_parameters["mu"],
        }

    @classmethod
    def from_spec_dict(cls, data: _SpecDict) -> "FullPressureElectronScatteringDisk":
        """Construct a :class:`FullPressureElectronScatteringDisk` from a spec dict.

        Parameters
        ----------
        data : dict
            Spec dict produced by :meth:`to_spec_dict`.  The ``"target"``
            key is silently ignored.

        Returns
        -------
        FullPressureElectronScatteringDisk

        See Also
        --------
        to_spec_dict :
            Instance method that produces the dict this class method reads.
        """
        kwargs = {k: v for k, v in data.items() if k != "target"}
        return cls(**kwargs)

    # ==================================================== #
    # Cython Integration Interface                        #
    # ==================================================== #

    def _build_cython_closure(self) -> Any:
        """Return a FullPressureElectronScatteringClosure for the Cython integrator."""
        from triceratops.dynamics.accretion.one_zone._ideal_gas_closure import (
            FullPressureElectronScatteringClosure,
        )

        return FullPressureElectronScatteringClosure()

    def _pack_cython_parameters(self, run_params: _RunParams) -> np.ndarray:
        """Return ``[MBH (g), R_in (cm), alpha, mu]`` as a float64 array."""
        return self._pack_base_cython_parameters(run_params)


# ================================================================== #
# Fallback Supply Disks                                              #
# ================================================================== #
# These models extend the ideal-gas disks by adding a continuous power-law
# mass supply (e.g. TDE debris stream or collapsar fallback).

_DEFAULT_BETA_FB: float = 5.0 / 3.0

_FALLBACK_RUNTIME_PARAMETERS: dict[str, dict] = {
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
    "M_fb_0": {
        "description": "Fallback mass supply rate at t_fb",
        "base_units": "g/s",
        "default": None,
        "log_transform": True,
    },
    "t_fb": {
        "description": "Fallback reference time",
        "base_units": "s",
        "default": None,
        "log_transform": True,
    },
    "beta_fb": {
        "description": "Fallback power-law index",
        "base_units": None,
        "default": _DEFAULT_BETA_FB,
        "log_transform": False,
    },
}

_FALLBACK_RESULT_FIELDS: dict[str, dict] = {
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
    "mdot_fb": {"description": "Fallback mass supply rate", "units": "g/s"},
}

_FALLBACK_CYTHON_FIELD_MAP: dict = {
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
    "mdot_fb": None,  # computed Python-side from time array
}


class GasPressureFallbackDisk(OneZoneAccretionDiskBase):
    r"""
    Gas-pressure / electron-scattering disk with a continuous fallback supply.

    Extends :class:`GasPressureElectronScatteringDisk` by adding a power-law
    mass supply from an external debris stream (TDE fallback, collapsar
    fallback, etc.):

    .. math::

        \dot{M}_{\rm fb}(t) = M_{\rm fb,0}\,\left(\frac{t}{t_{\rm fb}}\right)^{-\beta_{\rm fb}}.

    The fallback material is assumed to circularise at the current disk outer
    radius :math:`R_D`, depositing specific angular momentum
    :math:`\ell_{\rm circ} = \sqrt{G M_{\rm BH} R_D}`.

    All thermodynamic and viscous quantities follow the gas-pressure /
    electron-scattering closure (analytic temperature solve).

    Parameters
    ----------
    mu : float, optional
        Mean molecular weight (dimensionless).  Default ``0.6``.

    Notes
    -----
    The fallback supply is implemented as a ``source_func`` that runs *after*
    the base viscous derivative at each timestep.  The extra parameters
    ``[M_fb_0, t_fb, beta_fb]`` are appended to the standard
    ``[MBH, R_in, alpha, mu]`` parameter array and accessed inside the
    Cython hot loop via ``params.extra``.

    Examples
    --------
    .. code-block:: python

        from astropy import constants as const
        from astropy import units as u
        from triceratops.dynamics.accretion.one_zone import (
            GasPressureFallbackDisk,
        )

        disk = GasPressureFallbackDisk()
        result = disk.solve(
            initial_conditions={
                "M_D_0": 0.05 * const.M_sun,
                "J_D_0": 1e49 * u.g * u.cm**2 / u.s,
            },
            runtime_parameters={
                "M_BH": 3 * const.M_sun,
                "R_in": 3e6 * u.cm,
                "alpha": 0.1,
                "M_fb_0": 1e28 * u.g / u.s,
                "t_fb": 1e4 * u.s,
            },
            t_span=(1e4 * u.s, 1e8 * u.s),
        )
        print(result.data["mdot_fb"][0])

    See Also
    --------
    GasPressureElectronScatteringDisk :
        Base disk without fallback supply.
    FullPressureFallbackDisk :
        Same model with combined gas + radiation pressure.
    """

    _A: float = 1.62
    _B: float = 1.33
    _F: float = 1.6

    CONTEXT_PARAMETERS: dict[str, dict] = {
        "mu": {"description": "Mean molecular weight", "base_units": None, "default": 0.6},
    }
    RUNTIME_PARAMETERS: dict[str, dict] = _FALLBACK_RUNTIME_PARAMETERS
    INITIAL_CONDITIONS: dict[str, dict] = {
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
    RESULT_FIELDS: dict[str, dict] = _FALLBACK_RESULT_FIELDS
    CYTHON_FIELD_MAP: dict = _FALLBACK_CYTHON_FIELD_MAP

    def __init__(self, mu: float = 0.6):
        super().__init__(mu=mu)
        self.eos = IdealGasEOS(mu=self._context_parameters["mu"])

    def to_spec_dict(self) -> _SpecDict:
        """Return a JSON-serialisable spec dict for this model instance.

        Returns
        -------
        dict
            Spec dict with keys ``"target"`` and ``"mu"``.
        """
        return {
            "target": "triceratops.dynamics.accretion.one_zone.core:GasPressureFallbackDisk",
            "mu": self._context_parameters["mu"],
        }

    @classmethod
    def from_spec_dict(cls, data: _SpecDict) -> "GasPressureFallbackDisk":
        """Construct a :class:`GasPressureFallbackDisk` from a spec dict.

        Parameters
        ----------
        data : dict
            Spec dict produced by :meth:`to_spec_dict`.

        Returns
        -------
        GasPressureFallbackDisk
        """
        kwargs = {k: v for k, v in data.items() if k != "target"}
        return cls(**kwargs)

    def _build_cython_closure(self) -> Any:
        """Return a GasPressureFallbackClosure for the Cython integrator."""
        from triceratops.dynamics.accretion.one_zone._source_terms import (
            GasPressureFallbackClosure,
        )

        return GasPressureFallbackClosure()

    def _pack_cython_parameters(self, run_params: _RunParams) -> np.ndarray:
        """Return ``[MBH, R_in, alpha, mu, M_fb_0, t_fb, beta_fb]`` as float64."""
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
        """Compute ``mdot_fb`` from the time array and fallback parameters.

        Parameters
        ----------
        result_array : np.ndarray
            Raw Cython result array; row 1 is the time axis.
        run_params : dict
            Processed runtime parameters.

        Returns
        -------
        dict
            ``{"mdot_fb": np.ndarray}`` in CGS (g s⁻¹).
        """
        t = result_array[1, :]
        M_fb_0 = np.exp(run_params["log_M_fb_0"])
        t_fb = np.exp(run_params["log_t_fb"])
        beta = run_params["beta_fb"]
        return {"mdot_fb": M_fb_0 * (t / t_fb) ** (-beta)}


class FullPressureFallbackDisk(OneZoneAccretionDiskBase):
    r"""
    Full-pressure (gas + radiation) / electron-scattering disk with fallback supply.

    Extends :class:`FullPressureElectronScatteringDisk` by adding the same
    power-law mass supply as :class:`GasPressureFallbackDisk`.  The midplane
    temperature is solved iteratively at each timestep (bracket expansion +
    Brent's method) rather than analytically.

    Parameters
    ----------
    mu : float, optional
        Mean molecular weight (dimensionless).  Default ``0.6``.

    Notes
    -----
    Identical to :class:`GasPressureFallbackDisk` in all respects except that
    the Cython closure uses :class:`~._source_terms.FullPressureFallbackClosure`
    instead of :class:`~._source_terms.GasPressureFallbackClosure`.

    Examples
    --------
    .. code-block:: python

        from astropy import constants as const
        from astropy import units as u
        from triceratops.dynamics.accretion.one_zone import (
            FullPressureFallbackDisk,
        )

        disk = FullPressureFallbackDisk()
        result = disk.solve(
            initial_conditions={
                "M_D_0": 0.5 * const.M_sun,
                "J_D_0": 1e49 * u.g * u.cm**2 / u.s,
            },
            runtime_parameters={
                "M_BH": 3 * const.M_sun,
                "R_in": 3e6 * u.cm,
                "alpha": 0.1,
                "M_fb_0": 1e28 * u.g / u.s,
                "t_fb": 1e4 * u.s,
            },
            t_span=(1e4 * u.s, 1e8 * u.s),
        )
        print(result.data["T_c"][-1])

    See Also
    --------
    FullPressureElectronScatteringDisk :
        Base disk without fallback supply.
    GasPressureFallbackDisk :
        Same model with gas pressure only.
    """

    _A: float = 1.62
    _B: float = 1.33
    _F: float = 1.6

    CONTEXT_PARAMETERS: dict[str, dict] = {
        "mu": {"description": "Mean molecular weight", "base_units": None, "default": 0.6},
    }
    RUNTIME_PARAMETERS: dict[str, dict] = _FALLBACK_RUNTIME_PARAMETERS
    INITIAL_CONDITIONS: dict[str, dict] = {
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
    RESULT_FIELDS: dict[str, dict] = _FALLBACK_RESULT_FIELDS
    CYTHON_FIELD_MAP: dict = _FALLBACK_CYTHON_FIELD_MAP

    def __init__(self, mu: float = 0.6):
        super().__init__(mu=mu)
        self.eos = IdealGasEOS(mu=self._context_parameters["mu"])

    def to_spec_dict(self) -> _SpecDict:
        """Return a JSON-serialisable spec dict for this model instance.

        Returns
        -------
        dict
            Spec dict with keys ``"target"`` and ``"mu"``.
        """
        return {
            "target": "triceratops.dynamics.accretion.one_zone.core:FullPressureFallbackDisk",
            "mu": self._context_parameters["mu"],
        }

    @classmethod
    def from_spec_dict(cls, data: _SpecDict) -> "FullPressureFallbackDisk":
        """Construct a :class:`FullPressureFallbackDisk` from a spec dict.

        Parameters
        ----------
        data : dict
            Spec dict produced by :meth:`to_spec_dict`.

        Returns
        -------
        FullPressureFallbackDisk
        """
        kwargs = {k: v for k, v in data.items() if k != "target"}
        return cls(**kwargs)

    def _build_cython_closure(self) -> Any:
        """Return a FullPressureFallbackClosure for the Cython integrator."""
        from triceratops.dynamics.accretion.one_zone._source_terms import (
            FullPressureFallbackClosure,
        )

        return FullPressureFallbackClosure()

    def _pack_cython_parameters(self, run_params: _RunParams) -> np.ndarray:
        """Return ``[MBH, R_in, alpha, mu, M_fb_0, t_fb, beta_fb]`` as float64."""
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
        """Compute ``mdot_fb`` from the time array and fallback parameters.

        Parameters
        ----------
        result_array : np.ndarray
            Raw Cython result array; row 1 is the time axis.
        run_params : dict
            Processed runtime parameters.

        Returns
        -------
        dict
            ``{"mdot_fb": np.ndarray}`` in CGS (g s⁻¹).
        """
        t = result_array[1, :]
        M_fb_0 = np.exp(run_params["log_M_fb_0"])
        t_fb = np.exp(run_params["log_t_fb"])
        beta = run_params["beta_fb"]
        return {"mdot_fb": M_fb_0 * (t / t_fb) ** (-beta)}
