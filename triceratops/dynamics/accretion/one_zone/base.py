r"""
One-zone accretion disk evolution models.

This module implements time-dependent **one-zone** (vertically-integrated,
radially-averaged) accretion disk models following the framework of
:footcite:t:`metzgerTimeDependentModelsAccretion2008`.  The disk is described
by two global state variables

— disk mass :math:`M_D` and
- angular momentum :math:`J_D`

which are evolved as a coupled ODE system:

.. math::

    \frac{d \ln M_D}{dt} &= \frac{\dot{M}_{\rm src}}{M_D}
                            - \frac{f_{\rm disk}}{t_{\rm visc}}, \\[6pt]
    \frac{d \ln J_D}{dt} &= \frac{\dot{J}_{\rm src}}{J_D}.

All structural, thermodynamic, and viscous quantities — disk radius, surface
density, central temperature, kinematic viscosity, etc. — are recovered at
each solver step via a **three-step closure pipeline**:

1. **Disk closure** (:meth:`~OneZoneAccretionDiskBase._compute_disk_closure`):
   recover the one-zone disk radius :math:`R_D`, surface density :math:`\Sigma`,
   and angular velocity :math:`\Omega` from the current state vector.

2. **Thermodynamic closure**
   (:meth:`~OneZoneAccretionDiskBase._compute_thermodynamic_closure`): compute
   the central temperature :math:`T_c`, optical depth :math:`\tau`, effective
   temperature :math:`T_{\rm eff}`, and sound speed :math:`c_s`.  This step
   uses the *previous* viscous timescale as a seed (warm-start) to break a
   circular dependency in the viscous heating term.

3. **Viscous closure**
   (:meth:`~OneZoneAccretionDiskBase._compute_viscous_closure`): update the
   kinematic viscosity :math:`\nu` and viscous timescale
   :math:`t_{\rm visc} = R_D^2 / \nu` self-consistently.

The result of each solve is a :class:`OneZoneAccretionResult`, a thin
dataclass that stores the raw integration arrays and delegates all derived-
quantity reconstruction to the model that produced it.

See Also
--------
:mod:`triceratops.dynamics.accretion.one_zone.core` :
    Concrete disk model (gas pressure + electron-scattering opacity).
:mod:`triceratops.dynamics.accretion.utils` :
    Low-level disk utility functions consumed by closures.
:ref:`one_zone_disk`, :ref:`one_zone_disk_theory`
"""

import importlib
import json
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from astropy import units as u
from astropy.table import QTable
from scipy.integrate import solve_ivp

from triceratops.utils.misc_utils import ensure_in_units

# ------------------------------------------------------------- #
# Package versioning and Type checking                          #
# ------------------------------------------------------------- #
try:
    __pkg_version__ = version("triceratops")
except PackageNotFoundError:
    __pkg_version__ = "unknown"


# ================================================================== #
# Base Class                                                         #
# ================================================================== #
class _OneZoneMeta(ABCMeta):
    """Metaclass for all one-zone accretion disk models.

    One-zone disk models are highly **declarative**: subclasses specify their
    physics entirely through four class-level dicts
    (:attr:`~OneZoneAccretionDiskBase.RUNTIME_PARAMETERS`,
    :attr:`~OneZoneAccretionDiskBase.CONTEXT_PARAMETERS`,
    :attr:`~OneZoneAccretionDiskBase.INITIAL_CONDITIONS`,
    :attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS`).  Because those dicts
    drive runtime behaviour — unit conversion, log-transform, ODE state-vector
    packing, and result reconstruction — a malformed entry causes a confusing
    error deep inside the solver rather than at definition time.

    This metaclass moves that failure **as early as possible**: to the moment
    the class body is evaluated.  :meth:`__new__` calls
    :meth:`_validate_parameter_declarations` on the freshly built class,
    checking:

    - All required keys are present in each spec dict.
    - No unrecognised (possibly misspelled) extra keys are present.
    - Each field value carries the correct Python type.

    Validation applies to every class that uses this metaclass — both
    concrete implementations and intermediate abstract subclasses — so errors
    in partial declarations are caught before further subclassing obscures
    their origin.
    """

    # Required (and exhaustive) field sets for each declaration dict.
    # Stored as class attributes so they travel with the metaclass and are
    # trivially accessible inside _validate_param_spec without relying on
    # module-level globals.
    _RUNTIME_PARAM_FIELDS: frozenset = frozenset({"description", "base_units", "default", "log_transform"})
    _CONTEXT_PARAM_FIELDS: frozenset = frozenset({"description", "base_units", "default"})
    _IC_FIELDS: frozenset = frozenset({"description", "base_units", "default", "log_transform"})
    _RESULT_FIELDS: frozenset = frozenset({"description", "units"})

    def __new__(mcls, name: str, bases: tuple, namespace: dict, **kwargs):
        """Construct *cls* and immediately validate its parameter declarations.

        Called by Python at *class-definition time* (not at instantiation).
        After delegating to :class:`~abc.ABCMeta` for the standard
        class-construction machinery, :meth:`_validate_parameter_declarations`
        is invoked on the freshly built class.

        Parameters
        ----------
        name : str
            Name of the class being constructed.
        bases : tuple
            Base classes.
        namespace : dict
            Class namespace (body).
        **kwargs
            Forwarded to :class:`~abc.ABCMeta`.

        Returns
        -------
        type
            The newly constructed class.

        Raises
        ------
        TypeError
            If any declaration dict on the new class contains a malformed
            parameter spec.
        """
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        mcls._validate_parameter_declarations(cls)
        return cls

    @classmethod
    def _validate_param_spec(
        mcls, cls_name: str, dict_name: str, param_name: str, spec: object, required: frozenset
    ) -> None:
        """Validate one parameter-spec entry against its required-field schema.

        Enforces three layers of correctness in order:

        1. *Type check* — *spec* itself must be a :class:`dict`.
        2. *Completeness* — every key in *required* must be present.
        3. *No extraneous keys* — keys outside *required* are rejected so
           that typos (e.g. ``"log_trnasform"``) are caught immediately
           rather than being silently ignored at runtime.
        4. *Value types* — each field carries the expected Python type:

           - ``"description"`` : non-empty :class:`str`
           - ``"base_units"`` / ``"units"`` : :class:`str` or ``None``
           - ``"log_transform"`` : :class:`bool` (when present in *required*)
           - ``"default"``       : any value (``None`` signals *required at call-time*)

        Parameters
        ----------
        cls_name : str
            Name of the class being validated (used in error messages).
        dict_name : str
            Name of the declaration dict, e.g. ``"RUNTIME_PARAMETERS"``.
        param_name : str
            Name of the individual parameter entry.
        spec : object
            The spec value.  Must be a :class:`dict`.
        required : frozenset
            The exhaustive set of required (and allowed) keys.

        Raises
        ------
        TypeError
            On the first violation found, with a message that names the
            offending class, dict, parameter, and field.
        """
        location = f"{cls_name}.{dict_name}['{param_name}']"

        # 1. spec must be a dict
        if not isinstance(spec, dict):
            raise TypeError(
                f"{location} must be a dict, got {type(spec).__name__!r}. "
                f"Each entry in {dict_name} must map a parameter name to a "
                f"schema dict with keys: {sorted(required)}."
            )

        actual_keys = frozenset(spec.keys())

        # 2. missing required keys
        missing = required - actual_keys
        if missing:
            raise TypeError(
                f"{location} is missing required field(s): {sorted(missing)}. "
                f"Every {dict_name} entry must declare all of: {sorted(required)}."
            )

        # 3. unrecognised extra keys (catches typos)
        extra = actual_keys - required
        if extra:
            raise TypeError(
                f"{location} contains unrecognised field(s): {sorted(extra)}. "
                f"Allowed fields are: {sorted(required)}. "
                f"Check for typos — an unknown key is silently ignored at runtime."
            )

        # 4. field value types
        description = spec["description"]
        if not isinstance(description, str) or not description.strip():
            raise TypeError(f"{location}['description'] must be a non-empty str, got {description!r}.")

        units_key = "units" if "units" in required else "base_units"
        units_value = spec[units_key]
        if units_value is not None and not isinstance(units_value, str):
            raise TypeError(
                f"{location}['{units_key}'] must be a str (astropy unit expression) "
                f"or None for dimensionless quantities, "
                f"got {type(units_value).__name__!r}: {units_value!r}."
            )

        if "log_transform" in required:
            log_transform = spec["log_transform"]
            if not isinstance(log_transform, bool):
                raise TypeError(
                    f"{location}['log_transform'] must be True or False, "
                    f"got {type(log_transform).__name__!r}: {log_transform!r}."
                )

    @classmethod
    def _validate_parameter_declarations(mcls, cls) -> None:
        """Run all declaration-dict checks for *cls*.

        Iterates each of the four declaration dicts and delegates per-entry
        validation to :meth:`_validate_param_spec`.  Called unconditionally
        for every class — abstract or concrete — whose metaclass is
        :class:`_OneZoneMeta`.

        Parameters
        ----------
        cls : type
            The class to validate.

        Raises
        ------
        TypeError
            Propagated from :meth:`_validate_param_spec` on the first
            offending entry found.
        """
        for param_name, spec in cls.RUNTIME_PARAMETERS.items():
            mcls._validate_param_spec(cls.__name__, "RUNTIME_PARAMETERS", param_name, spec, mcls._RUNTIME_PARAM_FIELDS)
        for param_name, spec in cls.CONTEXT_PARAMETERS.items():
            mcls._validate_param_spec(cls.__name__, "CONTEXT_PARAMETERS", param_name, spec, mcls._CONTEXT_PARAM_FIELDS)
        for param_name, spec in cls.INITIAL_CONDITIONS.items():
            mcls._validate_param_spec(cls.__name__, "INITIAL_CONDITIONS", param_name, spec, mcls._IC_FIELDS)
        for param_name, spec in cls.RESULT_FIELDS.items():
            mcls._validate_param_spec(cls.__name__, "RESULT_FIELDS", param_name, spec, mcls._RESULT_FIELDS)


class OneZoneAccretionDiskBase(ABC, metaclass=_OneZoneMeta):
    r"""
    Abstract base class for one-zone accretion disk models.

    A one-zone disk model collapses the two-dimensional disk structure into two
    global, time-evolving variables: disk mass :math:`M_D` and angular momentum
    :math:`J_D`.  All other physical quantities are recovered at each timestep
    via a **declarative closure pipeline** whose steps are defined by subclass
    implementations.

    Notes
    -----
    **The ODE system**

    The model evolves the natural logarithm of each state variable, which keeps
    the ODE in a numerically comfortable range and naturally enforces positivity:

    .. math::

        \frac{d \ln M_D}{dt} &= \frac{\dot{M}_{\rm src}}{M_D}
                                - \frac{f_{\rm disk}}{t_{\rm visc}}, \\[6pt]
        \frac{d \ln J_D}{dt} &= \frac{\dot{J}_{\rm src}}{J_D},

    where :math:`\dot{M}_{\rm src}` and :math:`\dot{J}_{\rm src}` are external
    source terms (zero by default for an isolated disk) and the universal
    viscous mass drain :math:`f_{\rm disk} M_D / t_{\rm visc}` is applied
    in :meth:`_compute_RHS`.

    **Declarative parameter system**

    Subclasses declare their parameters through four class-level dictionaries:

    - :attr:`CONTEXT_PARAMETERS` — constructor arguments (e.g. equation of
      state parameters, one-zone correction factors).
    - :attr:`RUNTIME_PARAMETERS` — per-solve physics inputs (e.g. black hole
      mass, inner truncation radius, :math:`\alpha`, initial viscous
      timescale).
    - :attr:`INITIAL_CONDITIONS` — ODE initial state (e.g. :math:`M_{D,0}`,
      :math:`J_{D,0}`).
    - :attr:`RESULT_FIELDS` — derived quantities to reconstruct after the
      solve (e.g. temperatures, densities, viscosity).

    Each entry specifies the expected units, whether the value is log-
    transformed before storage, and an optional default.
    :meth:`process_runtime_parameters` and
    :meth:`process_initial_conditions` validate, unit-convert, and pack
    these into the flat float dicts and state vectors consumed by the ODE
    solver.  An :meth:`__init_subclass__` hook automatically validates the
    dict schema whenever a new subclass is defined, catching missing or
    misspelled fields at class-definition time.

    **The closure pipeline**

    At each solver step, :meth:`_IVP_kernel` runs three abstract closure
    methods in sequence, each of which modifies a shared ``params`` dict
    in-place:

    1. :meth:`_compute_disk_closure` — geometric and kinematic quantities
       (:math:`R_D`, :math:`\Sigma`, :math:`\Omega`).
    2. :meth:`_compute_thermodynamic_closure` — thermal quantities
       (:math:`T_c`, :math:`T_{\rm eff}`, :math:`c_s`, :math:`\tau`).
    3. :meth:`_compute_viscous_closure` — :math:`\nu` and
       :math:`t_{\rm visc}`.

    The thermodynamic closure requires :math:`t_{\rm visc}` to compute
    viscous heating, creating an apparent circular dependency.  This is
    resolved by a **warm-start cache** (:attr:`_runtime_cache`): the
    timescale computed at the *previous* step seeds the current step.  The
    cache is initialised with ``t_visc_0`` (a required runtime parameter)
    before the first solver call and cleared after the solve completes.

    **Implementing a new model**

    Subclass :class:`OneZoneAccretionDiskBase`, then:

    1. Populate the four declaration dicts.
    2. Implement :meth:`_compute_disk_closure`,
       :meth:`_compute_thermodynamic_closure`, and
       :meth:`_compute_viscous_closure`.
    3. Implement :meth:`_extract_result_fields` to map closure parameters
       onto the keys declared in :attr:`RESULT_FIELDS`.
    4. Implement :meth:`to_spec_dict` and :meth:`from_spec_dict` to support
       HDF5 persistence.

    Optionally, override :meth:`_compute_mass_sources` and / or
    :meth:`_compute_angular_momentum_sources` to inject external mass or
    angular-momentum source terms (e.g. tidal fallback, GW inspiral torques).

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
                "t_visc_0": 1.0e8 * u.s,
            },
            t_span=(1.0e6, 5.0e9),
            t_eval=np.geomspace(1.0e6, 5.0e9, 200),
        )

        print(result.success, result.M_D_solar[-1])
        data = result.reconstruct()
        result.to_hdf5("disk_run.h5", overwrite=True)

    See Also
    --------
    GasPressureElectronScatteringDisk :
        The canonical concrete implementation of this base class.
    OneZoneAccretionResult :
        The result container returned by :meth:`solve`.
    :mod:`triceratops.dynamics.accretion.utils` :
        Low-level disk utility functions used in closure implementations.
    :ref:`one_zone_disk`, :ref:`one_zone_disk_theory`
    """

    # ==================================================== #
    # PARAMETER DECLARATIONS                               #
    # ==================================================== #
    CONTEXT_PARAMETERS: dict[str, dict] = {}
    r"""dict : Constructor (context) parameters.

    Each entry maps a parameter name to a schema dict:

    .. code-block:: python

        {
            "description": str,      # human-readable label
            "base_units":  str|None, # astropy unit string; None = dimensionless
            "default":     value,    # None means required
        }

    Values are stored verbatim in :attr:`_context_parameters` and are
    available for the lifetime of the model instance.

    Notes
    -----
    For developers, the context parameters should be used for hyperparameters which are
    not relevant in inference implementations. Thus, we should not place parameters in
    the context parameters if they would require re-instantiation of the disk engine everytime
    they are changed.
    """
    RUNTIME_PARAMETERS: dict[str, dict] = {}
    r"""dict : Per-solve runtime parameters.

    Each entry maps a parameter name to a schema dict:

    .. code-block:: python

        {
            "description":   str,
            "base_units":    str|None,
            "default":       value,    # None means required
            "log_transform": bool,     # True → stored as log(CGS value)
        }

    :meth:`process_runtime_parameters` iterates this dict to validate,
    convert, and pack user-supplied values into the flat float dict
    threaded through the closure pipeline.  Log-transformed parameters
    are stored under ``"log_{key}"``; others under ``"{key}"``.

    Notes
    -----
    These parameters should be effectively the core parameters for each run.
    """
    INITIAL_CONDITIONS: dict[str, dict] = {}
    r"""dict : ODE initial-condition declarations.

    Same schema as :attr:`RUNTIME_PARAMETERS`.
    :meth:`process_initial_conditions` iterates this dict in declaration
    order and packs values into the state vector ``y0`` passed to the ODE
    solver.

    The initial conditions may ONLY include the ODE state variables (e.g. :math:`M_{D,0}`,
    :math:`J_{D,0}`) and not any auxiliary parameters (e.g. :math:`R_{D,0}`, :math:`t_{\rm visc,0}`) which
    are computed self-consistently from the state variables and context parameters.
    """
    RESULT_FIELDS: dict[str, dict] = {}
    """dict : Derived output fields populated during reconstruction.

    Each entry maps a field name to:

    .. code-block:: python

        {
            "description": str,
            "units":       str|None,  # astropy unit string; None = dimensionless
        }

    :meth:`reconstruct` pre-allocates one float array per field, calls
    :meth:`_extract_result_fields` at each timestep to fill them, then
    attaches units from ``"units"`` via ``astropy.units.Unit``.
    """

    # ==================================================== #
    # Initialization                                       #
    # ==================================================== #
    def __init__(self, **context_parameters):
        """
        Initialise the one-zone disk model.

        In subclasses, the parameters should be declared explicitly and
        then passed into the base class constructor via ``super().__init__(**context_parameters)``.

        Stores all keyword arguments in :attr:`_context_parameters` and
        initialises an empty :attr:`_runtime_cache`.  Subclass
        ``__init__`` methods should call ``super().__init__(**kwargs)``
        after constructing any objects (e.g. equation-of-state engines)
        that depend on the context parameters.

        Parameters
        ----------
        **context_parameters
            Context parameters matching the keys declared in
            :attr:`CONTEXT_PARAMETERS`.
        """
        self._context_parameters: dict[str, Any] = context_parameters
        self._runtime_cache: dict[str, Any] = {}

    # ==================================================== #
    # Dunder Methods                                       #
    # ==================================================== #
    def __repr__(self) -> str:
        """Return a concise, unambiguous string representation.

        The output mirrors a valid constructor call: class name followed by
        the context parameters as keyword arguments.  This makes it easy to
        identify and recreate a model instance from a REPL or log output.

        Returns
        -------
        str
            ``ClassName(param1=val1, param2=val2, ...)``

        Examples
        --------
        >>> disk = GasPressureElectronScatteringDisk(mu=0.62)
        >>> repr(disk)
        'GasPressureElectronScatteringDisk(mu=0.62)'
        """
        ctx = ", ".join(f"{k}={v!r}" for k, v in self._context_parameters.items())
        return f"{type(self).__name__}({ctx})"

    def __str__(self) -> str:
        r"""Return a human-readable summary of all declared parameters.

        Produces a multi-line table that lists:

        - **Context parameters** — stored values, units, and descriptions.
        - **Runtime parameters** — declared names, units, and whether each
          is required or has a default.
        - **Initial conditions** — same layout as runtime parameters.
        - **Result fields** — output field names and units.

        This is the primary "quick look" representation at the REPL.

        Returns
        -------
        str
            Multi-line plain-text parameter summary.

        Examples
        --------
        >>> print(disk)
        GasPressureElectronScatteringDisk
          Context parameters (1):
            mu = 0.62  [dimensionless]  -- mean molecular weight
          Runtime parameters (4):
            M_BH      [g]       required  -- black hole mass
            ...
        """
        lines = [type(self).__name__]

        if self.CONTEXT_PARAMETERS:
            lines.append(f"  Context parameters ({len(self.CONTEXT_PARAMETERS)}):")
            for name, spec in self.CONTEXT_PARAMETERS.items():
                val = self._context_parameters.get(name, spec["default"])
                units = f"[{spec['base_units']}]" if spec["base_units"] else "[dimensionless]"
                lines.append(f"    {name} = {val!r}  {units}  -- {spec['description']}")

        if self.RUNTIME_PARAMETERS:
            lines.append(f"  Runtime parameters ({len(self.RUNTIME_PARAMETERS)}):")
            for name, spec in self.RUNTIME_PARAMETERS.items():
                units = f"[{spec['base_units']}]" if spec["base_units"] else "[dimensionless]"
                default = f"default={spec['default']!r}" if spec["default"] is not None else "required"
                lines.append(f"    {name:<16}{units:<20}{default}  -- {spec['description']}")

        if self.INITIAL_CONDITIONS:
            lines.append(f"  Initial conditions ({len(self.INITIAL_CONDITIONS)}):")
            for name, spec in self.INITIAL_CONDITIONS.items():
                units = f"[{spec['base_units']}]" if spec["base_units"] else "[dimensionless]"
                default = f"default={spec['default']!r}" if spec["default"] is not None else "required"
                lines.append(f"    {name:<16}{units:<20}{default}  -- {spec['description']}")

        if self.RESULT_FIELDS:
            lines.append(f"  Result fields ({len(self.RESULT_FIELDS)}):")
            for name, spec in self.RESULT_FIELDS.items():
                units = f"[{spec['units']}]" if spec["units"] else "[dimensionless]"
                lines.append(f"    {name:<16}{units}  -- {spec['description']}")

        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        """Return ``True`` if *other* is the same model type with identical context parameters.

        Two disk models are considered equal if and only if:

        1. They are the **same concrete subclass** (same type, not just a
           common ancestor), since different subclasses encode entirely
           different physics.
        2. They carry **identical context parameters** (same keys and values).

        This allows result containers to check whether the model they hold
        still matches the calling model, and makes model de-duplication in
        parameter-sweep loops straightforward.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` if *other* is the same model; ``NotImplemented`` if it
            is not a :class:`OneZoneAccretionDiskBase` instance, delegating
            the comparison to the other operand.
        """
        if not isinstance(other, OneZoneAccretionDiskBase):
            return NotImplemented
        return type(self) is type(other) and self._context_parameters == other._context_parameters

    # Instances are mutable (_runtime_cache changes during solves) so they
    # must not be hashable once __eq__ is defined.
    __hash__ = None

    # ==================================================== #
    # Parameter Processing                                 #
    # ==================================================== #
    def process_runtime_parameters(self, runtime_parameters: dict) -> dict:
        r"""
        Validate and convert runtime parameters to a log-CGS float dict.

        Iterates over :attr:`RUNTIME_PARAMETERS` in declaration order.
        For each parameter:

        - The value is taken from *runtime_parameters* if present, or the
          declared ``"default"`` is used if not ``None``.
        - Missing required parameters (``"default": None``) raise
          :exc:`ValueError` with the parameter name in the message.
        - Unit-bearing values are converted to ``"base_units"`` via
          :func:`~triceratops.utils.misc_utils.ensure_in_units`.
        - If ``"log_transform": True``, the result is stored as
          ``"log_{key}": np.log(cgs_value)``; otherwise as
          ``"{key}": float(cgs_value)``.

        Parameters
        ----------
        runtime_parameters : dict
            User-supplied runtime parameters.  Values may be plain floats
            or :class:`~astropy.units.Quantity` objects; unit conversion
            is applied automatically.

        Returns
        -------
        dict
            Flat dict of plain floats (or log-floats) ready for use in
            the closure pipeline.  No unit objects are present in the
            returned dict.

        Raises
        ------
        ValueError
            If a required parameter (one with ``"default": None``) is
            absent from *runtime_parameters*.

        Examples
        --------
        .. code-block:: python

            from astropy import constants as const
            from astropy import units as u

            run_params = disk.process_runtime_parameters(
                {
                    "M_BH": 3.0 * const.M_sun,
                    "R_in": 3.0e6 * u.cm,
                    "alpha": 0.1,
                    "t_visc_0": 1.0e8 * u.s,
                }
            )
            # M_BH is log-transformed:
            print(
                run_params["log_M_BH"]
            )  # np.log(M_BH in grams)
            # alpha is not log-transformed:
            print(run_params["alpha"])  # 0.1
        """
        run_params = {}
        for key, spec in self.RUNTIME_PARAMETERS.items():
            if key in runtime_parameters:
                val = runtime_parameters[key]
            elif spec["default"] is not None:
                val = spec["default"]
            else:
                raise ValueError(
                    f"Required runtime parameter '{key}' ({spec['description']}) was not provided and has no default."
                )

            if spec["base_units"] is not None:
                cgs_value = float(ensure_in_units(val, spec["base_units"]))
            else:
                cgs_value = float(val)

            if spec["log_transform"]:
                run_params[f"log_{key}"] = np.log(cgs_value)
            else:
                run_params[key] = cgs_value

        return run_params

    def process_initial_conditions(self, initial_conditions: dict) -> np.ndarray:
        r"""
        Validate and pack initial conditions into a log-CGS state vector.

        Iterates over :attr:`INITIAL_CONDITIONS` in declaration order,
        applying the same validation and unit-conversion logic as
        :meth:`process_runtime_parameters`.  The resulting values are
        assembled into a 1-D NumPy array in the order the keys appear in
        :attr:`INITIAL_CONDITIONS`.

        Parameters
        ----------
        initial_conditions : dict
            User-supplied initial conditions.  Values may be plain floats
            or :class:`~astropy.units.Quantity` objects.

        Returns
        -------
        np.ndarray
            State vector ``y0`` of shape ``(n_ic,)``, where
            ``n_ic = len(INITIAL_CONDITIONS)``.  Each element is the
            log-CGS or plain-CGS representation of the corresponding
            initial condition, ready for :func:`scipy.integrate.solve_ivp`.

        Raises
        ------
        ValueError
            If a required initial condition is absent from
            *initial_conditions*.

        Examples
        --------
        .. code-block:: python

            from astropy import constants as const
            from astropy import units as u

            y0 = disk.process_initial_conditions(
                {
                    "M_D_0": 0.1 * const.M_sun,
                    "J_D_0": 1.5e49 * u.g * u.cm**2 / u.s,
                }
            )
            # y0[0] = log(M_D_0 in g),  y0[1] = log(J_D_0 in g cm^2 / s)
        """
        y0 = []
        for key, spec in self.INITIAL_CONDITIONS.items():
            if key in initial_conditions:
                val = initial_conditions[key]
            elif spec["default"] is not None:
                val = spec["default"]
            else:
                raise ValueError(
                    f"Required initial condition '{key}' ({spec['description']}) was not provided and has no default."
                )

            if spec["base_units"] is not None:
                cgs_value = float(ensure_in_units(val, spec["base_units"]))
            else:
                cgs_value = float(val)

            y0.append(np.log(cgs_value) if spec["log_transform"] else cgs_value)

        return np.array(y0)

    # ==================================================== #
    # Abstract Closures                                    #
    # ==================================================== #

    @abstractmethod
    def _compute_disk_closure(self, t: float, state: np.ndarray, runtime_params: dict) -> dict:
        r"""
        Compute geometric and kinematic disk quantities from the ODE state.

        This is the **first** of the three closure steps.  It translates the
        current log-CGS state vector into one-zone-averaged structural
        parameters.  The returned dict is passed to
        :meth:`_compute_thermodynamic_closure` and
        :meth:`_compute_viscous_closure` in the same kernel step.

        Implementations should at minimum populate:

        - ``"log_r"`` : :math:`\ln R_D` — one-zone disk outer radius [cm].
        - ``"log_sigma"`` : :math:`\ln \Sigma` — one-zone surface density
          [:math:`\text{g\,cm}^{-2}`].
        - ``"log_Omega"`` : :math:`\ln \Omega` — orbital angular velocity
          [:math:`\text{rad\,s}^{-1}`].

        Parameters
        ----------
        t : float
            Current integration time [s].
        state : np.ndarray, shape (n_state,)
            Log-CGS state vector.  For the standard two-variable
            formulation this is ``[log(M_D), log(J_D)]``.
        runtime_params : dict
            Processed per-solve parameters (as returned by
            :meth:`process_runtime_parameters`).

        Returns
        -------
        dict
            Closure parameter dict keyed by subclass-defined names.  Must
            contain at minimum ``"log_r"``, ``"log_sigma"``, and
            ``"log_Omega"``.
        """
        ...

    @abstractmethod
    def _compute_thermodynamic_closure(self, t, state, cycle_params: dict, runtime_params: dict) -> None:
        r"""
        Compute thermal quantities and update *cycle_params* in-place.

        This is the **second** closure step.  Implementations receive the
        dict populated by :meth:`_compute_disk_closure` and should add
        thermal quantities such as:

        - ``"log_T_eff"`` : :math:`\ln T_{\rm eff}` — effective surface
          temperature [K].
        - ``"log_T_c"`` : :math:`\ln T_c` — central midplane temperature [K].
        - ``"log_tau"`` : :math:`\ln \tau` — optical depth (dimensionless).
        - ``"log_cs"`` : :math:`\ln c_s` — isothermal sound speed
          [:math:`\text{cm\,s}^{-1}`].
        - ``"f_disk"`` : geometric mass-loss factor (dimensionless).

        This step typically requires the viscous timescale from the
        *previous* solver step, which is provided via :attr:`_runtime_cache`.
        If the cache does not contain ``"t_visc"``, implementations must
        raise :exc:`ValueError` with the string ``"t_visc"`` in the message
        so that callers and tests can detect the missing seed condition.

        Parameters
        ----------
        t : float
            Current integration time [s].
        state : np.ndarray, shape (n_state,)
            Log-CGS state vector.
        cycle_params : dict
            Modified **in-place**.  Contains the output of
            :meth:`_compute_disk_closure` on entry; thermodynamic
            quantities are added on exit.
        runtime_params : dict
            Processed per-solve parameters.
        """
        ...

    @abstractmethod
    def _compute_viscous_closure(self, t, state, cycle_params: dict, runtime_params: dict) -> None:
        r"""
        Compute viscous quantities and update *cycle_params* in-place.

        This is the **third** closure step.  Implementations receive the
        dict after both :meth:`_compute_disk_closure` and
        :meth:`_compute_thermodynamic_closure` have run, and should add:

        - ``"log_nu"`` : :math:`\ln \nu` — kinematic viscosity
          [:math:`\text{cm}^2\,\text{s}^{-1}`].
        - ``"t_visc"`` : :math:`t_{\rm visc} = R_D^2 / \nu` — viscous
          timescale [s].  This value must be a plain positive float (not
          log-transformed), as it is stored directly in :attr:`_runtime_cache`
          and used in the ODE right-hand side.

        The computed ``"t_visc"`` is stored in :attr:`_runtime_cache` by
        :meth:`_IVP_kernel` immediately after this method returns, warm-
        starting the thermodynamic closure at the next solver step.

        Parameters
        ----------
        t : float
            Current integration time [s].
        state : np.ndarray, shape (n_state,)
            Log-CGS state vector.
        cycle_params : dict
            Modified **in-place**.  Contains the output of the first two
            closure steps on entry; viscous quantities are added on exit.
        runtime_params : dict
            Processed per-solve parameters.
        """
        ...

    @abstractmethod
    def _extract_result_fields(self, params: dict, state_i: np.ndarray) -> dict:
        r"""
        Extract raw (unit-free) values for every key in :attr:`RESULT_FIELDS`.

        Called once per timestep during :meth:`reconstruct`.
        Implementations have full access to the ``params`` dict (populated
        by all three closure steps) and the current state slice, and may
        compute derived quantities not present in the forward-pass closures
        (e.g. scale height :math:`H`, midplane density
        :math:`\rho_{\rm mid}`, instantaneous accretion rate
        :math:`\dot{M}`).

        The returned dict **must** contain exactly the keys declared in
        :attr:`RESULT_FIELDS` — no more, no fewer.  Values must be plain
        floats or 0-D arrays (no unit objects); units are attached
        generically by :meth:`reconstruct` using the ``"units"`` field
        in :attr:`RESULT_FIELDS`.

        Parameters
        ----------
        params : dict
            Fully populated closure parameter dict for this timestep (output
            of all three closure steps applied by :meth:`reconstruct_state`).
        state_i : np.ndarray, shape (n_state,)
            Log-CGS state vector at this timestep.

        Returns
        -------
        dict
            Raw float values keyed **exactly** by :attr:`RESULT_FIELDS`.
        """
        ...

    # ==================================================== #
    # Abstract Serialization Hooks                         #
    # ==================================================== #

    @abstractmethod
    def to_spec_dict(self) -> dict:
        r"""
        Return a JSON-serialisable dict that fully describes this model instance.

        The dict is embedded as a JSON attribute in HDF5 files written by
        :meth:`~OneZoneAccretionResult.to_hdf5` and is used by
        :meth:`~OneZoneAccretionResult.from_hdf5` to reconstruct the model
        without any user-supplied configuration.

        The dict **must** contain a ``"target"`` key of the form
        ``"module.path:ClassName"`` so that :func:`_build_model_from_spec`
        can import the correct subclass.  All other entries should correspond
        to constructor parameters (they are forwarded to
        :meth:`from_spec_dict` after stripping ``"target"``).

        Returns
        -------
        dict
            JSON-serialisable specification dict containing at least
            ``"target"``.

        See Also
        --------
        from_spec_dict :
            The class method that reconstructs the model from this dict.
        """
        ...

    @classmethod
    @abstractmethod
    def from_spec_dict(cls, data: dict) -> "OneZoneAccretionDiskBase":
        r"""
        Construct a model instance from a spec dict.

        Called by :meth:`~OneZoneAccretionResult.from_hdf5` via
        :func:`_build_model_from_spec`.  Implementations should extract
        the relevant constructor arguments from *data* (ignoring the
        ``"target"`` key, which is stripped by the calling infrastructure)
        and return a fully initialised model instance.

        Parameters
        ----------
        data : dict
            Spec dict produced by :meth:`to_spec_dict`.  The ``"target"``
            key has already been removed by the calling infrastructure
            before this method is invoked.

        Returns
        -------
        OneZoneAccretionDiskBase
            Reconstructed model instance of the implementing subclass.

        See Also
        --------
        to_spec_dict :
            The instance method that produces the dict this method reads.
        """
        ...

    # ==================================================== #
    # Default Source Terms (Overridable)                   #
    # ==================================================== #
    def _compute_mass_sources(self, t, state, cycle_params: dict, runtime_params: dict) -> float:
        r"""External mass source rate :math:`\dot{M}_{\rm src}` [:math:`\text{g s}^{-1}`].

        Returns the net external mass injection or removal rate.  Note that
        the universal viscous drain :math:`-f_{\rm disk} M_D / t_{\rm visc}`
        is **not** included here; it is applied directly in
        :meth:`_compute_RHS`.  This method should return only physically
        distinct external contributions such as:

        - Tidal debris fallback onto a TDE disk
          (:math:`\dot{M} \propto t^{-5/3}` at late times).
        - Stellar wind accretion.
        - Mass transfer from a companion star.

        Override in subclasses to add external mass injection or removal.
        The default returns zero (isolated disk with no external mass source).

        Parameters
        ----------
        t : float
            Current integration time [s].
        state : np.ndarray, shape (n_state,)
            Log-CGS state vector.
        cycle_params : dict
            Fully populated closure parameter dict for this step.
        runtime_params : dict
            Processed per-solve parameters.

        Returns
        -------
        float
            :math:`\dot{M}_{\rm src}` [:math:`\text{g s}^{-1}`].
            Positive values add mass; negative values remove it.
        """
        return 0.0

    def _compute_angular_momentum_sources(self, t, state, cycle_params: dict, runtime_params: dict) -> float:
        r"""External angular-momentum source rate :math:`\dot{J}_{\rm src}` [:math:`\text{g cm}^2 \text{s}^{-2}`].

        Returns the net external angular-momentum injection rate.  Common
        non-zero terms include:

        - **Viscous drain at the inner edge** — accreted matter carries away
          the Keplerian specific angular momentum
          :math:`\ell_{\rm in} = \sqrt{G M_{\rm BH} R_{\rm in}}`, so
          :math:`\dot{J}_{\rm src} = -\dot{M} \ell_{\rm in}`.  This is the
          term implemented by
          :class:`~triceratops.dynamics.accretion.one_zone.core.GasPressureElectronScatteringDisk`.
        - **Gravitational-wave inspiral torques** for compact-object disks.
        - **External magnetic or tidal torques**.

        Override in subclasses to add any of these.  The default returns
        zero (no angular-momentum exchange with external systems).

        Parameters
        ----------
        t : float
            Current integration time [s].
        state : np.ndarray, shape (n_state,)
            Log-CGS state vector.
        cycle_params : dict
            Fully populated closure parameter dict for this step.
        runtime_params : dict
            Processed per-solve parameters.

        Returns
        -------
        float
            :math:`\dot{J}_{\rm src}` [:math:`\text{g cm}^2 \text{s}^{-2}`].
            Positive values add angular momentum; negative values remove it.
        """
        return 0.0

    # ==================================================== #
    # RHS Construction                                     #
    # ==================================================== #
    def _compute_RHS(self, t, state, mass_source_term, angular_momentum_source_term, params):
        r"""
        Construct the log-space ODE right-hand side vector.

        Combines the external source terms with the universal viscous mass
        drain to produce the log-space time derivatives:

        .. math::

            \frac{d \ln M_D}{dt}
            &= \frac{\dot{M}_{\rm src}}{M_D}
               - \frac{f_{\rm disk}}{t_{\rm visc}}, \\[6pt]
            \frac{d \ln J_D}{dt}
            &= \frac{\dot{J}_{\rm src}}{J_D}.

        The viscous mass drain :math:`f_{\rm disk} M_D / t_{\rm visc}` is
        applied universally here and must **not** be included in
        :meth:`_compute_mass_sources`; it is a structural property of the
        one-zone framework that applies to all models.  The angular-momentum
        drain (typically the Keplerian specific angular momentum carried
        away by accreted matter at the inner edge) belongs in the
        overridable :meth:`_compute_angular_momentum_sources`.

        Parameters
        ----------
        t : float
            Current integration time [s].
        state : np.ndarray, shape (n_state,)
            Log-CGS state vector ``[log(M_D), log(J_D)]``.
        mass_source_term : float
            External mass source rate
            :math:`\dot{M}_{\rm src}` [:math:`\text{g s}^{-1}`] from
            :meth:`_compute_mass_sources`.
        angular_momentum_source_term : float
            External angular-momentum source rate
            :math:`\dot{J}_{\rm src}` [:math:`\text{g cm}^2 \text{s}^{-2}`]
            from :meth:`_compute_angular_momentum_sources`.
        params : dict
            Fully populated closure parameter dict.  Must contain
            ``"f_disk"`` and ``"t_visc"``.

        Returns
        -------
        np.ndarray, shape (n_state,)
            Log-space time derivatives
            :math:`[d\ln M_D/dt,\; d\ln J_D/dt]` in
            :math:`\text{s}^{-1}`.
        """
        log_m, log_j = state
        m = np.exp(log_m)
        j = np.exp(log_j)

        dm_dt = mass_source_term - params["f_disk"] * m / params["t_visc"]
        dj_dt = angular_momentum_source_term

        return np.array([dm_dt / m, dj_dt / j])

    # ==================================================== #
    # IVP Kernel                                           #
    # ==================================================== #
    def _IVP_kernel(self, t, state, run_params: dict) -> np.ndarray:
        r"""
        Core ODE evaluation function called by the integrator at each step.

        Executes the three-step closure pipeline, computes source terms,
        stores the completed ``params`` dict in :attr:`_runtime_cache` for
        warm-starting the thermodynamic closure at the next step, and
        returns the log-space derivative vector.

        The sequence of operations at each step is:

        1. :meth:`_compute_disk_closure` → ``params``
        2. :meth:`_compute_thermodynamic_closure` → updates ``params``
        3. :meth:`_compute_viscous_closure` → updates ``params``
        4. :meth:`_compute_mass_sources` → scalar ``mass_src``
        5. :meth:`_compute_angular_momentum_sources` → scalar ``am_src``
        6. ``self._runtime_cache = params``  (warm-start for next step)
        7. :meth:`_compute_RHS` → derivative vector

        Parameters
        ----------
        t : float
            Current integration time [s].
        state : np.ndarray, shape (n_state,)
            Log-CGS state vector at the current step.
        run_params : dict
            Per-solve parameters threaded through all closures.

        Returns
        -------
        np.ndarray, shape (n_state,)
            Log-space time derivatives :math:`[\text{s}^{-1}]`.
        """
        params = self._compute_disk_closure(t, state, run_params)
        self._compute_thermodynamic_closure(t, state, params, run_params)
        self._compute_viscous_closure(t, state, params, run_params)

        mass_src = self._compute_mass_sources(t, state, params, run_params)
        am_src = self._compute_angular_momentum_sources(t, state, params, run_params)

        self._runtime_cache = params  # warm-start next step

        return self._compute_RHS(t, state, mass_src, am_src, params)

    # ==================================================== #
    # Public Solver Interface                              #
    # ==================================================== #
    def solve(
        self,
        initial_conditions: dict,
        runtime_parameters: dict,
        t_span: tuple,
        **solver_kwargs,
    ) -> "OneZoneAccretionResult":
        r"""
        Solve the one-zone disk evolution equations.

        Processes the user-supplied parameters and initial conditions into
        the internal log-CGS representation, seeds :attr:`_runtime_cache`
        with the initial viscous timescale, integrates the ODE system with
        :func:`scipy.integrate.solve_ivp`, and returns the solution wrapped
        in a :class:`OneZoneAccretionResult`.

        .. important::

            ``t_visc_0`` (the initial viscous timescale) is a **required**
            runtime parameter and must be physically consistent with the
            initial state.  An inconsistent seed can cause the integration
            to start in an unphysical regime.  Use a short fixed-point
            iteration on :meth:`reconstruct_state` to find a self-consistent
            value before calling :meth:`solve` (see the test suite for a
            worked example).

        Parameters
        ----------
        initial_conditions : dict
            Initial state of the disk.  Keys must match those declared in
            :attr:`INITIAL_CONDITIONS` (e.g. ``"M_D_0"``, ``"J_D_0"``).
            Values may be plain floats or :class:`~astropy.units.Quantity`
            objects; unit conversion is applied automatically.
        runtime_parameters : dict
            Per-solve physics parameters.  Keys must match those declared
            in :attr:`RUNTIME_PARAMETERS` (e.g. ``"M_BH"``, ``"R_in"``,
            ``"alpha"``, ``"t_visc_0"``).
        t_span : tuple of float
            Integration interval ``(t_start, t_end)`` in seconds.
        **solver_kwargs
            Additional keyword arguments forwarded to
            :func:`scipy.integrate.solve_ivp`.  Sensible defaults are
            applied for ``method`` (``"RK45"``), ``rtol`` (``1e-6``), and
            ``atol`` (``1e-9``) if not supplied.

        Returns
        -------
        OneZoneAccretionResult
            Result dataclass containing the integration arrays, the
            processed runtime parameters, solver metadata, and a reference
            to this model instance.

        Notes
        -----
        :attr:`_runtime_cache` is cleared (set to ``{}``) after the solve
        returns, whether or not the integration succeeded.  This prevents
        stale values from a failed solve from contaminating a subsequent
        call.

        Examples
        --------
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
                    "t_visc_0": 1.0e8 * u.s,
                },
                t_span=(1.0e6, 5.0e9),
                t_eval=np.geomspace(1.0e6, 5.0e9, 200),
            )

            print(result.success, result.message)
            print(result.M_D_solar[-1])

        See Also
        --------
        OneZoneAccretionResult :
            The object returned by this method.
        reconstruct_state :
            Single-timestep reconstruction used for bootstrapping a
            self-consistent ``t_visc_0``.
        """
        run_params = self.process_runtime_parameters(runtime_parameters)
        y0 = self.process_initial_conditions(initial_conditions)

        self._runtime_cache = {"t_visc": run_params["t_visc_0"]}

        method = solver_kwargs.pop("method", "RK45")
        rtol = solver_kwargs.pop("rtol", 1e-6)
        atol = solver_kwargs.pop("atol", 1e-9)

        def rhs(t, y):
            return self._IVP_kernel(t, y, run_params)

        ode_result = solve_ivp(
            rhs,
            t_span,
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
            **solver_kwargs,
        )

        self._runtime_cache = {}

        return OneZoneAccretionResult(
            model=self,
            t=ode_result.t,
            state=ode_result.y,
            run_params=run_params,
            success=ode_result.success,
            message=ode_result.message,
        )

    # ==================================================== #
    # Reconstruction                                       #
    # ==================================================== #
    def reconstruct_state(self, t, state, runtime_params: dict) -> dict:
        r"""
        Reconstruct all closure parameters at a single timestep.

        Runs the three closure steps in sequence and returns the fully
        populated ``params`` dict.  The caller is responsible for seeding
        :attr:`_runtime_cache` with ``{"t_visc": <seed_value>}`` before
        the first call.  For a sequence of timesteps, the cache should be
        updated with the returned ``params`` after each call to warm-start
        the thermodynamic closure.

        This method is used internally by :meth:`reconstruct` and is also
        useful for bootstrapping a self-consistent ``t_visc_0`` before
        calling :meth:`solve`:

        .. code-block:: python

            y0 = disk.process_initial_conditions(ic)
            t_visc = 1.0e8  # initial guess [s]
            for _ in range(30):
                rp = disk.process_runtime_parameters(
                    {..., "t_visc_0": t_visc * u.s}
                )
                disk._runtime_cache = {"t_visc": t_visc}
                params = disk.reconstruct_state(0.0, y0, rp)
                disk._runtime_cache = {}
                t_visc_new = params["t_visc"]
                if abs(t_visc_new - t_visc) / t_visc < 1e-8:
                    break
                t_visc = t_visc_new

        Parameters
        ----------
        t : float
            Time [s] at which to evaluate the closures.
        state : np.ndarray, shape (n_state,)
            Log-CGS state vector at time *t*.
        runtime_params : dict
            Processed per-solve parameters (as returned by
            :meth:`process_runtime_parameters`).

        Returns
        -------
        dict
            Fully populated closure parameter dict for timestep *t*,
            containing the outputs of all three closure steps.

        See Also
        --------
        reconstruct :
            Drives this method over an array of timesteps and assembles
            unit-bearing output arrays.
        """
        params = self._compute_disk_closure(t, state, runtime_params)
        self._compute_thermodynamic_closure(t, state, params, runtime_params)
        self._compute_viscous_closure(t, state, params, runtime_params)
        return params

    def reconstruct(self, t_arr: np.ndarray, state_arr: np.ndarray, run_params: dict) -> dict:
        r"""
        Reconstruct all derived quantities over a stored time grid.

        Drives the loop over timesteps: at each step it calls
        :meth:`reconstruct_state` (seeded with the ``t_visc`` from the
        previous step) and then :meth:`_extract_result_fields` to extract
        the raw float values.  Output arrays are pre-allocated to avoid
        repeated memory allocations, then unit-bearing
        :class:`~astropy.units.Quantity` objects are attached using the
        ``"units"`` strings declared in :attr:`RESULT_FIELDS`.

        Parameters
        ----------
        t_arr : np.ndarray, shape (N,)
            Integration times [s] at which the state is stored.
        state_arr : np.ndarray, shape (n_state, N)
            Log-CGS state array as returned by the ODE solver.
        run_params : dict
            Processed runtime parameters (as returned by
            :meth:`process_runtime_parameters`).  Must contain
            ``"t_visc_0"`` to seed the warm-start cache.

        Returns
        -------
        dict
            Dictionary of unit-bearing arrays keyed by every entry in
            :attr:`RESULT_FIELDS` plus the state-variable keys ``'t'``,
            ``'M_D'``, and ``'J_D'``.  All arrays have shape ``(N,)``.

        Notes
        -----
        :attr:`_runtime_cache` is cleared after reconstruction completes.
        This prevents stale values from leaking into subsequent operations
        on the same model instance.

        See Also
        --------
        reconstruct_state :
            Single-timestep version used internally by this method.
        OneZoneAccretionResult.reconstruct :
            The thin wrapper on this method that is the user-facing entry
            point.
        """
        n_t = len(t_arr)

        # Pre-allocate one contiguous float array per declared result field
        raw: dict[str, np.ndarray] = {key: np.empty(n_t) for key in self.RESULT_FIELDS}

        self._runtime_cache = {"t_visc": run_params["t_visc_0"]}

        for i in range(n_t):
            params = self.reconstruct_state(t_arr[i], state_arr[:, i], run_params)
            self._runtime_cache = params
            extracted = self._extract_result_fields(params, state_arr[:, i])
            for key in self.RESULT_FIELDS:
                raw[key][i] = extracted[key]

        self._runtime_cache = {}

        # Attach units from RESULT_FIELDS declarations
        result: dict[str, Any] = {
            "t": t_arr * u.s,
            "M_D": np.exp(state_arr[0]) * u.g,
            "J_D": np.exp(state_arr[1]) * u.g * u.cm**2 / u.s,
        }
        for key, spec in self.RESULT_FIELDS.items():
            units_str = spec["units"]
            result[key] = raw[key] * u.Unit(units_str) if units_str is not None else raw[key]

        return result


@dataclass
class OneZoneAccretionResult:
    r"""
    Minimal result container for a one-zone accretion disk ODE solve.

    :class:`OneZoneAccretionResult` stores only the raw integration output
    (time array, log-CGS state array, processed runtime parameters, and solver
    metadata) and delegates all physics to the :attr:`model` that produced it.
    This design keeps the result object lightweight while allowing the model
    to determine exactly which derived quantities are reconstructed and how.

    The state array encodes the ODE state vector in log-space CGS.  For the
    standard two-variable formulation:

    .. math::

        \text{state}[0, i] &= \ln M_D(t_i) \quad [\text{in g}], \\
        \text{state}[1, i] &= \ln J_D(t_i) \quad
                              [\text{in g\,cm}^2\,\text{s}^{-1}].

    **Accessing state variables**

    The disk mass and angular momentum are exposed as unit-bearing array
    properties:

    .. code-block:: python

        result = disk.solve(...)

        print(result.M_D_solar)   # disk mass in solar masses, shape (N,)
        print(result.J_D)         # angular momentum in g cm^2 / s, shape (N,)

    **Reconstructing derived quantities**

    Call :meth:`reconstruct` to recover all quantities declared in
    :attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS` (temperatures, surface
    density, viscosity, etc.) over the stored time grid:

    .. code-block:: python

        data = result.reconstruct()
        print(data["T_c"])   # central temperature vs time
        print(data["R_D"])   # disk outer radius vs time

    **Tabular output**

    :meth:`to_table` wraps :meth:`reconstruct` and returns an Astropy
    :class:`~astropy.table.QTable` suitable for writing to FITS, ASCII, or
    ECSV:

    .. code-block:: python

        table = result.to_table()
        table.write("disk_output.ecsv", overwrite=True)

    **HDF5 persistence**

    Results can be saved to and loaded from HDF5 files.  The model
    specification is embedded as a JSON attribute, so the correct model
    subclass is reconstructed automatically on load — no user configuration
    is required:

    .. code-block:: python

        result.to_hdf5("disk_run.h5", overwrite=True)
        loaded = OneZoneAccretionResult.from_hdf5("disk_run.h5")
        assert isinstance(loaded.model, GasPressureElectronScatteringDisk)

    Parameters
    ----------
    model : OneZoneAccretionDiskBase
        The disk model instance that produced this result.  All
        reconstruction logic (closure pipeline, unit assignment, etc.)
        is delegated back to this object.
    t : np.ndarray
        Integration times, shape ``(N,)``, in seconds.
    state : np.ndarray
        ODE state array, shape ``(n_state, N)``.  Values are natural
        logarithms of CGS quantities.
    run_params : dict
        Processed runtime parameters as returned by
        :meth:`~OneZoneAccretionDiskBase.process_runtime_parameters`.
        All values are plain floats; log-transformed parameters are
        stored under ``"log_{key}"``.
    success : bool
        Whether the ODE integration completed successfully.
    message : str
        Solver termination message from
        :func:`scipy.integrate.solve_ivp`.

    See Also
    --------
    OneZoneAccretionDiskBase :
        Base class that defines the closure pipeline and reconstruction
        logic consumed by this result.
    OneZoneAccretionDiskBase.solve :
        The method that constructs and returns a
        :class:`OneZoneAccretionResult`.
    """

    model: "OneZoneAccretionDiskBase"
    t: np.ndarray
    state: np.ndarray
    run_params: dict
    success: bool
    message: str

    FORMAT: ClassVar[str] = "OneZoneAccretionResult"
    VERSION: ClassVar[str] = __pkg_version__

    def __post_init__(self):
        n_ic = len(self.model.INITIAL_CONDITIONS)
        if self.state.shape[0] != n_ic:
            raise ValueError(
                f"state.shape[0] = {self.state.shape[0]} does not match len(model.INITIAL_CONDITIONS) = {n_ic}."
            )

    # ---------------------------------------------- #
    # State-variable Properties                      #
    # ---------------------------------------------- #

    @property
    def M_D(self) -> u.Quantity:
        r"""
        Disk mass :math:`M_D` as a unit-bearing array.

        Recovered by exponentiating the first row of :attr:`state`:

        .. math::

            M_D(t_i) = \exp\!\bigl(\text{state}[0,\,i]\bigr)
            \quad [\text{g}]

        Returns
        -------
        `~astropy.units.Quantity`
            Disk mass in grams, shape ``(N,)``.
        """
        return np.exp(self.state[0]) * u.g

    @property
    def M_D_solar(self) -> u.Quantity:
        r"""
        Disk mass :math:`M_D` in solar masses.

        Returns
        -------
        `~astropy.units.Quantity`
            Disk mass in :math:`M_\odot`, shape ``(N,)``.
        """
        return self.M_D.to(u.Msun)

    @property
    def J_D(self) -> u.Quantity:
        r"""
        Disk angular momentum :math:`J_D` as a unit-bearing array.

        Recovered by exponentiating the second row of :attr:`state`:

        .. math::

            J_D(t_i) = \exp\!\bigl(\text{state}[1,\,i]\bigr)
            \quad [\text{g\,cm}^2\,\text{s}^{-1}]

        Returns
        -------
        `~astropy.units.Quantity`
            Angular momentum in :math:`\text{g\,cm}^2\,\text{s}^{-1}`,
            shape ``(N,)``.
        """
        return np.exp(self.state[1]) * u.g * u.cm**2 / u.s

    @property
    def t_quantity(self) -> u.Quantity:
        """
        Integration times as a unit-bearing :class:`~astropy.units.Quantity`.

        Returns
        -------
        `~astropy.units.Quantity`
            Time in seconds, shape ``(N,)``.
        """
        return self.t * u.s

    # ---------------------------------------------- #
    # Reconstruction (delegated to model)            #
    # ---------------------------------------------- #

    def reconstruct(self) -> dict:
        r"""
        Reconstruct all derived quantities over the stored time grid.

        Delegates entirely to
        :meth:`OneZoneAccretionDiskBase.reconstruct`, which drives the
        three-step closure pipeline at each stored timestep and attaches
        units from :attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS`.

        Returns
        -------
        dict
            Dictionary of unit-bearing arrays keyed by every entry in
            :attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS` plus the
            state-variable keys ``'t'``, ``'M_D'``, and ``'J_D'``.
            All arrays have shape ``(N,)`` where ``N`` is the number of
            stored timesteps.

        Notes
        -----
        Each call re-runs the full closure pipeline across all stored
        timesteps.  For large grids or expensive closures, cache the
        result in a local variable rather than calling :meth:`reconstruct`
        repeatedly.

        See Also
        --------
        to_table :
            Wraps this method and returns an :class:`~astropy.table.QTable`.
        OneZoneAccretionDiskBase.reconstruct :
            The base-class method that performs the actual computation.
        """
        return self.model.reconstruct(self.t, self.state, self.run_params)

    def to_table(self) -> QTable:
        r"""
        Reconstruct all derived quantities and return as an Astropy table.

        Convenience wrapper around :meth:`reconstruct` that packages the
        result dict into a :class:`~astropy.table.QTable` with one row
        per stored timestep and one column per quantity.

        Returns
        -------
        `~astropy.table.QTable`
            Table with columns for every entry in
            :attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS` plus
            ``'t'``, ``'M_D'``, and ``'J_D'``.

        Examples
        --------
        .. code-block:: python

            table = result.to_table()
            print(table.colnames)
            table.write("disk_output.ecsv", overwrite=True)

        See Also
        --------
        reconstruct :
            Underlying method that computes the column data.
        """
        return QTable(self.reconstruct())

    # ---------------------------------------------- #
    # HDF5 Persistence                               #
    # ---------------------------------------------- #

    def to_hdf5(self, filename, overwrite: bool = False) -> None:
        r"""
        Write the result to an HDF5 file.

        The model specification is serialised as a JSON string stored in an
        HDF5 attribute, so the model can be reconstructed automatically by
        :meth:`from_hdf5` without any user-supplied configuration.

        **File layout**::

            result.h5
            ├── [attr] format      : "OneZoneAccretionResult"
            ├── [attr] version     : str  (package version)
            ├── [attr] model_spec  : JSON str  ← model.to_spec_dict()
            ├── [attr] run_params  : JSON str  ← run_params dict (floats)
            ├── [attr] success     : bool
            ├── [attr] message     : str
            ├── t     : float64 dataset, shape (N,)          [seconds]
            └── state : float64 dataset, shape (n_state, N)  [log-CGS]

        Parameters
        ----------
        filename : str or Path
            Output path for the HDF5 file.  The parent directory must
            already exist.
        overwrite : bool, optional
            If ``False`` (the default) and the file already exists, raise
            :exc:`FileExistsError` without modifying the file.  Set to
            ``True`` to silently replace an existing file.

        Raises
        ------
        FileExistsError
            If *filename* already exists and ``overwrite=False``.

        Examples
        --------
        .. code-block:: python

            result.to_hdf5("disk_run.h5")

            # Overwrite a previous run:
            result.to_hdf5("disk_run.h5", overwrite=True)

        See Also
        --------
        from_hdf5 :
            Class method to load a result from a file written by this
            method.
        """
        import h5py

        path = Path(filename)
        if path.exists() and not overwrite:
            raise FileExistsError(f"'{path}' already exists. Pass overwrite=True to overwrite.")

        with h5py.File(path, "w") as f:
            f.attrs["format"] = self.FORMAT
            f.attrs["version"] = self.VERSION
            f.attrs["model_spec"] = json.dumps(self.model.to_spec_dict(), separators=(",", ":"))
            f.attrs["run_params"] = json.dumps(self.run_params, separators=(",", ":"))
            f.attrs["success"] = self.success
            f.attrs["message"] = self.message
            f.create_dataset("t", data=self.t)
            f.create_dataset("state", data=self.state)

    @classmethod
    def from_hdf5(cls, filename) -> "OneZoneAccretionResult":
        r"""
        Load a result from an HDF5 file written by :meth:`to_hdf5`.

        The model subclass is identified from the embedded ``"model_spec"``
        JSON attribute and reconstructed via :func:`_build_model_from_spec`,
        which dynamically imports the correct class and calls its
        :meth:`~OneZoneAccretionDiskBase.from_spec_dict` class method.  No
        user configuration is required.

        Parameters
        ----------
        filename : str or Path
            Path to the HDF5 file to read.

        Returns
        -------
        OneZoneAccretionResult
            Fully reconstructed result with a live model instance.

        Raises
        ------
        ValueError
            If the file's ``"format"`` attribute does not equal
            :attr:`FORMAT` (``"OneZoneAccretionResult"``).

        Examples
        --------
        .. code-block:: python

            from triceratops.dynamics.accretion.one_zone import (
                OneZoneAccretionResult,
            )

            loaded = OneZoneAccretionResult.from_hdf5(
                "disk_run.h5"
            )
            print(loaded.M_D_solar)
            data = loaded.reconstruct()

        See Also
        --------
        to_hdf5 :
            Instance method that writes the file this class method reads.
        """
        import h5py

        with h5py.File(Path(filename), "r") as f:
            fmt = f.attrs["format"]
            if fmt != cls.FORMAT:
                raise ValueError(f"format mismatch: file has '{fmt}', expected '{cls.FORMAT}'.")
            spec_dict = json.loads(f.attrs["model_spec"])
            run_params = json.loads(f.attrs["run_params"])
            success = bool(f.attrs["success"])
            message = str(f.attrs["message"])
            t = f["t"][:]
            state = f["state"][:]

        model = _build_model_from_spec(spec_dict)
        return cls(
            model=model,
            t=t,
            state=state,
            run_params=run_params,
            success=success,
            message=message,
        )


def _build_model_from_spec(spec_dict: dict) -> "OneZoneAccretionDiskBase":
    r"""
    Reconstruct a disk model instance from a spec dict.

    Reads ``spec_dict["target"]`` (``"module:ClassName"``), imports the class
    dynamically, and calls ``cls.from_spec_dict(spec_dict)``.

    Parameters
    ----------
    spec_dict : dict
        Spec dict produced by
        :meth:`OneZoneAccretionDiskBase.to_spec_dict`.  Must contain
        a ``"target"`` key of the form ``"module.path:ClassName"``.

    Returns
    -------
    OneZoneAccretionDiskBase
        Reconstructed model instance of the appropriate subclass.

    Raises
    ------
    ValueError
        If the ``"target"`` string does not contain a colon separator.
    """
    target = spec_dict["target"]
    if ":" not in target:
        raise ValueError(f"Invalid target format '{target}'. Expected 'module:ClassName'.")
    module_path, class_name = target.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls.from_spec_dict(spec_dict)
