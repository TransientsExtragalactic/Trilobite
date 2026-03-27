r"""
One-zone (vertically-integrated) accretion disk models.

This module implements simplified, time-dependent accretion disk models
in which the disk is represented by a *single characteristic annulus*
("one-zone") rather than a fully resolved radial structure.

Instead of evolving a full surface-density profile :math:`\Sigma(R, t)`,
the disk is described by two global quantities:

- Total disk mass :math:`M_D`
- Total disk angular momentum :math:`J_D`

From these, a characteristic radius :math:`R_D` and surface density
:math:`\Sigma` are inferred under the assumption of a narrow ring or
self-similar spreading solution.

The time evolution follows Metzger et al. (2008), with coupled ODEs:

.. math::

    \frac{dM_D}{dt}
    = -\dot{M}_{\rm acc}
      + \dot{M}_{\rm fb}
      - \dot{M}_{\rm out},

.. math::

    \frac{dJ_D}{dt}
    = \dot{J}_{\rm fb}
      - \dot{J}_{\rm out},

where:

- :math:`\dot{M}_{\rm acc} \sim f M_D / t_{\rm visc}` is the viscous drain
- :math:`\dot{M}_{\rm fb}` is an optional fallback source (e.g. TDE debris)
- :math:`\dot{M}_{\rm out}` represents mass loss via winds (model-dependent)

The viscous timescale is set by an alpha-disk prescription:

.. math::

    t_{\rm visc} \sim \frac{R_D^2}{\nu}, \quad
    \nu = \alpha c_s^2 / \Omega.

The thermodynamics of the disk (temperature, pressure, opacity) are
determined through a *closure relation*, which depends on the chosen model.

Physical assumptions
--------------------

- The disk is geometrically thin or moderately thick, but vertically integrated.
- Radial structure is not resolved; instead, a characteristic radius is used.
- Angular momentum transport is modeled via an alpha viscosity.
- The disk is Keplerian (:math:`\Omega \approx \Omega_K`).
- Thermodynamics are determined locally via an equation of state and opacity law.
- Energy balance may include radiative cooling and (optionally) advective transport.

This framework is especially useful for:

- Tidal disruption events (TDEs)
- Transient accretion episodes
- Situations where global disk evolution is more important than detailed structure

Model classes
-------------

:class:`~.core.GasPressureDisk`
    Gas-pressure-dominated disk (:math:`P = P_{\rm gas}`) with analytic or
    iterative temperature closure.

:class:`~.core.FullPressureDisk`
    Includes both gas and radiation pressure (:math:`P = P_{\rm gas} + P_{\rm rad}`),
    requiring iterative temperature solves.

:class:`~.core.AdvectiveDisk`
    Full-pressure disk including advective cooling. Introduces an entropy-gradient
    parameter :math:`\xi` to model radial energy transport.

All models accept:

- ``opacity``: either a string identifier or a
  :class:`~triceratops.radiation.opacity.base.GreyOpacityLaw` instance
- ``fallback``: whether to include a mass fallback source term

Submodules
----------

closure
    C-level interface defining the closure contract, including structs and
    function signatures used by the integrator.

integrator
    High-performance explicit time integrator that evolves the disk state
    using closure-provided physics.

models
    Concrete implementations of closure relations (gas pressure, full pressure,
    advective).

physics
    Low-level physical building blocks (equations of state, viscosity,
    source terms).

base
    Python-facing interface defining model classes and result containers.

utils
    Tools for equilibrium solutions, S-curves, and diagnostic calculations.

Notes
-----
This implementation is designed for high performance: the core evolution
loop is written in Cython and operates without Python overhead, while
retaining a flexible object-oriented interface for specifying physical
closures such as opacity laws and equations of state.
"""

from triceratops.physics_utils.eos import radiative_ideal_gas_disk_sound_speed
from triceratops.physics_utils.general_relativity import compute_ISCO, compute_schwarzschild_radius

from .base import OneZoneAccretionDiskBase, OneZoneAccretionResult
from .core import (
    AdvectiveDisk,
    FullPressureDisk,
    # New canonical names
    GasPressureDisk,
    gP_es_fbDisk,
    # Backward-compatible aliases (old six-class API)
    gP_esDisk,
    igP_es_adv_fbDisk,
    igP_es_advDisk,
    igP_es_fbDisk,
    igP_esDisk,
)
from .utils import (
    compute_advective_equilibrium_temperature,
    compute_advective_s_curve,
    compute_equilibrium_temperature,
    compute_standard_s_curve,
    igP_es_adv_trial_curves,
    igP_es_trial_curves,
)

__all__ = [
    # Base
    "OneZoneAccretionDiskBase",
    "OneZoneAccretionResult",
    # Canonical model classes
    "GasPressureDisk",
    "FullPressureDisk",
    "AdvectiveDisk",
    # Backward-compat aliases
    "gP_esDisk",
    "igP_esDisk",
    "igP_es_advDisk",
    "gP_es_fbDisk",
    "igP_es_fbDisk",
    "igP_es_adv_fbDisk",
    # Utilities
    "radiative_ideal_gas_disk_sound_speed",
    "igP_es_trial_curves",
    "igP_es_adv_trial_curves",
    "compute_equilibrium_temperature",
    "compute_advective_equilibrium_temperature",
    "compute_standard_s_curve",
    "compute_advective_s_curve",
    # GR utilities
    "compute_ISCO",
    "compute_schwarzschild_radius",
]
