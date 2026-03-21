r"""
One-zone accretion disk evolution models.

This module implements time-dependent **one-zone** (vertically-integrated,
radially-averaged) accretion disk models following the framework of
:footcite:t:`metzgerTimeDependentModelsAccretion2008`.  The disk is described
by two global state variables

— disk mass :math:`M_D` and
- angular momentum :math:`J_D`

which are evolved by a compiled Cython explicit-Euler integrator.

All structural, thermodynamic, and viscous quantities (disk radius, surface
density, temperature, kinematic viscosity, etc.) are computed inside the
Cython hot loop at every timestep and returned in a compact result array
(:attr:`~OneZoneAccretionResult.result_array`).

The result of each solve is a :class:`OneZoneAccretionResult`, a thin
dataclass that wraps the raw Cython result array and exposes fields via the
:attr:`~OneZoneAccretionResult.data` property.

See Also
--------
:mod:`triceratops.dynamics.accretion.one_zone.core` :
    Concrete disk model (gas pressure + electron-scattering opacity).
:ref:`one_zone_disk`, :ref:`one_zone_disk_theory`
"""

import importlib
import json
import warnings
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
from astropy import units as u
from astropy.table import QTable

from triceratops.utils.misc_utils import ensure_in_units

from ._typing import (
    _DataDict,
    _FieldValue,
    _FilePath,
    _ParamDict,
    _RunParams,
    _SpecDict,
)

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
    _CYTHON_INDEX_TYPES: tuple = (int, type(None))

    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict, **kwargs):
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
        mcls, cls_name: str, dict_name: str, param_name: str, spec: Any, required: frozenset
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
        spec : Any
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

        # Validate that CYTHON_FIELD_MAP covers every RESULT_FIELDS key.
        cython_map = getattr(cls, "CYTHON_FIELD_MAP", {})
        for field_name in cls.RESULT_FIELDS:
            if field_name not in cython_map:
                raise TypeError(
                    f"{cls.__name__}.CYTHON_FIELD_MAP is missing key '{field_name}' which is declared in RESULT_FIELDS."
                )
            idx = cython_map[field_name]
            if not isinstance(idx, mcls._CYTHON_INDEX_TYPES):
                raise TypeError(
                    f"{cls.__name__}.CYTHON_FIELD_MAP['{field_name}'] must be int or None "
                    f"(None = derived in Python), got {type(idx).__name__!r}."
                )


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

    The model evolves disk mass :math:`M_D` and angular momentum :math:`J_D`
    via a compiled Cython explicit-Euler integrator.  All structural,
    thermodynamic, and viscous quantities are computed inside the Cython hot
    loop at every timestep.

    **Declarative parameter system**

    Subclasses declare their parameters through four class-level dictionaries:

    - :attr:`CONTEXT_PARAMETERS` — constructor arguments (e.g. equation of
      state parameters, one-zone correction factors).
    - :attr:`RUNTIME_PARAMETERS` — per-solve physics inputs (e.g. black hole
      mass, inner truncation radius, :math:`\alpha`).
    - :attr:`INITIAL_CONDITIONS` — ODE initial state (:math:`M_{D,0}`,
      :math:`J_{D,0}`).
    - :attr:`RESULT_FIELDS` — derived quantities exposed via the result.

    Each entry specifies the expected units, whether the value is log-
    transformed before storage, and an optional default.

    **Implementing a new model**

    Subclass :class:`OneZoneAccretionDiskBase`, then:

    1. Populate the four declaration dicts and :attr:`CYTHON_FIELD_MAP`.
    2. Implement :meth:`_build_cython_closure` to return the compiled closure.
    3. Implement :meth:`_pack_cython_parameters` to pack the parameter array.
    4. Implement :meth:`to_spec_dict` and :meth:`from_spec_dict` for HDF5.

    .. code-block:: python

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

        print(result.success, result.n_steps)
        data = result.data
        result.to_hdf5("disk_run.h5", overwrite=True)

    See Also
    --------
    GasPressureElectronScatteringDisk :
        The canonical concrete implementation of this base class.
    OneZoneAccretionResult :
        The result container returned by :meth:`solve`.
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
    """dict : Derived output fields exposed via :attr:`OneZoneAccretionResult.data`.

    Each entry maps a field name to:

    .. code-block:: python

        {
            "description": str,
            "units":       str|None,  # astropy unit string; None = dimensionless
        }

    Fields are read from the Cython result array by row index (see
    :attr:`CYTHON_FIELD_MAP`), or computed as derived quantities in Python.
    """

    CYTHON_FIELD_MAP: ClassVar[dict] = {}
    """dict : Mapping from :attr:`RESULT_FIELDS` keys to Cython result row indices.

    Each entry maps a field name to either an :class:`int` row index in the
    Cython result array (``result_array[idx, :]``), or ``None`` if the field
    is computed as a derived quantity in Python inside
    :attr:`OneZoneAccretionResult.data`.

    Every key in :attr:`RESULT_FIELDS` must appear here; missing keys are
    rejected by :class:`_OneZoneMeta` at class-definition time.
    """

    # ==================================================== #
    # Initialization                                       #
    # ==================================================== #
    def __init__(self, **context_parameters):
        """
        Initialise the one-zone disk model.

        In subclasses, the parameters should be declared explicitly and
        then passed into the base class constructor via ``super().__init__(**context_parameters)``.

        Stores all keyword arguments in :attr:`_context_parameters`.
        Subclass ``__init__`` methods should call
        ``super().__init__(**kwargs)`` after constructing any objects
        (e.g. equation-of-state engines) that depend on the context
        parameters.

        Parameters
        ----------
        **context_parameters
            Context parameters matching the keys declared in
            :attr:`CONTEXT_PARAMETERS`.
        """
        self._context_parameters: dict[str, Any] = context_parameters

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

        .. code-block:: python

            $ disk = GasPressureElectronScatteringDisk(mu=0.62)
            $ repr(disk)
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

        .. code-block:: python

            $ print(disk)
            GasPressureElectronScatteringDisk
              Context parameters (1):
                mu = 0.62  [dimensionless]  -- mean molecular weight
              Runtime parameters (3):
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

    # __eq__ is defined, so __hash__ must be explicitly set to None.
    __hash__ = None

    # ==================================================== #
    # Parameter Processing                                 #
    # ==================================================== #
    def process_runtime_parameters(self, runtime_parameters: _ParamDict) -> _RunParams:
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

    def process_initial_conditions(
        self,
        initial_conditions: _ParamDict,
    ) -> np.ndarray:
        r"""
        Validate and pack initial conditions into a CGS state vector.

        Iterates over :attr:`INITIAL_CONDITIONS` in declaration order,
        applying the same validation and unit-conversion logic as
        :meth:`process_runtime_parameters`.  The resulting values are
        assembled into a 1-D NumPy array in the order the keys appear in
        :attr:`INITIAL_CONDITIONS`.

        This converts the dictionary of initial conditions into the form
        expected by the Cython-level ODE solver.

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
            initial condition, ready for the Cython integrator
            (:func:`run_one_zone_model`).

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
    # Abstract Cython Interface                            #
    # ==================================================== #

    @abstractmethod
    def _build_cython_closure(self):
        """Return a fully-initialised Cython ``OneZoneClosure`` for this model.

        Implementations should import the appropriate concrete closure class
        from the Cython layer and return an instance:

        .. code-block:: python

            def _build_cython_closure(self):
                from ._ideal_gas_closure import (
                    GasPressureElectronScatteringClosure,
                )

                return GasPressureElectronScatteringClosure()

        Returns
        -------
        ~triceratops.dynamics.accretion.one_zone._integrator.OneZoneClosure
            A Cython extension type with all three function pointers installed
            (``is_ready()`` returns ``True``).
        """
        ...

    def _pack_base_cython_parameters(self, run_params: _RunParams) -> np.ndarray:
        """Pack the standard 4-element base parameter array ``[MBH, R_in, alpha, mu]``.

        Parameters
        ----------
        run_params : dict[str, float]
            Processed per-solve parameters as returned by
            :meth:`process_runtime_parameters`.

        Returns
        -------
        np.ndarray, shape (4,)
            ``[MBH (g), R_in (cm), alpha, mu]`` as a C-contiguous float64 array.
        """
        return np.array(
            [
                np.exp(run_params["log_M_BH"]),
                np.exp(run_params["log_R_in"]),
                run_params["alpha"],
                self._context_parameters["mu"],
            ],
            dtype=np.float64,
        )

    @abstractmethod
    def _pack_cython_parameters(self, run_params: _RunParams) -> np.ndarray:
        """Return the parameter array consumed by ``run_one_zone_model``.

        The Cython integrator expects a contiguous float64 array of shape
        ``(>=4,)`` with elements ``[MBH (g), R_in (cm), alpha, mu, ...]``
        in linear CGS.  Elements beyond index 3 are closure-specific extra
        parameters accessible via ``params.extra``.

        Parameters
        ----------
        run_params : dict[str, float]
            Processed per-solve parameters as returned by
            :meth:`process_runtime_parameters`.  Log-transformed parameters
            are stored under ``"log_{key}"``; others under ``"{key}"``.

        Returns
        -------
        np.ndarray, shape (>=4,)
            ``[MBH (g), R_in (cm), alpha, mu, ...]`` as a C-contiguous
            float64 array.
        """
        ...

    def _compute_derived_result_fields(self, result_array: np.ndarray, run_params: _RunParams) -> dict:
        """Compute Python-side derived result fields.

        Called by :attr:`OneZoneAccretionResult.data` for every
        :attr:`~OneZoneAccretionDiskBase.CYTHON_FIELD_MAP` entry whose value
        is ``None``.  Override in subclasses that declare such entries.

        Parameters
        ----------
        result_array : np.ndarray, shape (n_result_fields, n_steps)
            Raw Cython result array from :func:`~._integrator.run_one_zone_model`.
        run_params : dict[str, float]
            Processed per-solve parameters as returned by
            :meth:`process_runtime_parameters`.

        Returns
        -------
        dict[str, np.ndarray]
            Maps each ``None``-mapped field name to its 1-D NumPy array.
            The base implementation returns an empty dict.
        """
        return {}

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
    def from_spec_dict(cls, data: _SpecDict) -> "OneZoneAccretionDiskBase":
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
    # Public Solver Interface                              #
    # ==================================================== #
    def solve(
        self,
        initial_conditions: dict,
        runtime_parameters: dict,
        t_span: tuple[Any, Any],  # (t_start, t_end); plain float [s] or Quantity
        max_steps: int = 100_000,
        fail_fast: bool = True,
    ) -> "OneZoneAccretionResult":
        r"""
        Solve the one-zone disk evolution equations via the Cython integrator.

        Processes the user-supplied parameters and initial conditions into
        linear CGS, passes them to the compiled Cython integrator, and
        returns the solution as a :class:`OneZoneAccretionResult`.

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
            ``"alpha"``).
        t_span : tuple
            Integration interval ``(t_start, t_end)``; values may be plain
            floats (seconds) or :class:`~astropy.units.Quantity` objects.
        max_steps : int, optional
            Maximum number of integration steps.  Default is 100 000.
        fail_fast : bool, optional
            If ``True`` (default), raise :exc:`RuntimeError` when the
            integrator signals failure.  If ``False``, issue a warning and
            return the (possibly incomplete) result.

        Returns
        -------
        OneZoneAccretionResult
            Result containing the raw Cython result array, the processed
            runtime parameters, and solver metadata.

        Examples
        --------
        .. code-block:: python

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

            print(result.success, result.n_steps)
            data = result.data
            print(data["T_c"][[0, -1]])

        See Also
        --------
        OneZoneAccretionResult :
            The object returned by this method.
        """
        from ._integrator import run_one_zone_model

        # --- Process time span --- #
        t_start = float(ensure_in_units(t_span[0], "s"))
        t_end = float(ensure_in_units(t_span[1], "s"))

        # --- Process parameters and initial conditions --- #
        run_params = self.process_runtime_parameters(runtime_parameters)
        y0 = self.process_initial_conditions(initial_conditions)

        # y0 is log-CGS; the Cython integrator expects linear CGS.
        initial_state = np.ascontiguousarray(np.exp(y0), dtype=np.float64)
        params_array = np.ascontiguousarray(self._pack_cython_parameters(run_params), dtype=np.float64)
        closure = self._build_cython_closure()

        # --- Run the Cython integrator --- #
        success = True
        message = "Cython integrator completed successfully."
        raw = None
        try:
            raw = run_one_zone_model(initial_state, params_array, t_start, t_end, max_steps, closure)
        except (RuntimeError, ValueError) as exc:
            success = False
            message = str(exc)
            warnings.warn(f"Integration failed: {message}", RuntimeWarning, stacklevel=2)
            if fail_fast:
                raise

        if raw is None:
            raw = np.zeros((closure.n_result_fields, 1), dtype=np.float64)

        return OneZoneAccretionResult(
            model=self,
            result_array=raw,
            run_params=run_params,
            success=success,
            message=message,
        )

    def solve_parameter_grid(
        self,
        initial_conditions_grid: dict[str, list],
        runtime_parameters_grid: dict[str, list],
        t_span: tuple[Any, Any],
        max_steps: int = 100_000,
        fail_fast: bool = False,
    ) -> "dict[tuple, OneZoneAccretionResult]":
        r"""Solve the disk ODE for every combination of grid parameter values.

        Takes a grid of possible values for each initial condition and runtime
        parameter, constructs all combinations with
        :func:`itertools.product`, and calls :meth:`solve` for each.  Progress
        is tracked with a :mod:`tqdm` progress bar that respects the project
        configuration.

        Parameters
        ----------
        initial_conditions_grid : dict[str, list]
            Maps each initial-condition key to the list of values to sweep.
            Keys must match :attr:`INITIAL_CONDITIONS`.  Parameters not
            listed here take their declared default (or must be present in
            every ``initial_conditions`` call — i.e. they must have a
            default).
        runtime_parameters_grid : dict[str, list]
            Maps each runtime-parameter key to the list of values to sweep.
            Keys must match :attr:`RUNTIME_PARAMETERS`.
        t_span : tuple
            Integration interval ``(t_start, t_end)`` forwarded to every
            :meth:`solve` call.
        max_steps : int, optional
            Maximum integration steps per solve.  Default 100 000.
        fail_fast : bool, optional
            If ``True``, re-raise the first integration error and abort the
            sweep.  If ``False`` (default for grid runs), store the failed
            result and continue.

        Returns
        -------
        dict[tuple, OneZoneAccretionResult]
            Maps each parameter combination, encoded as a tuple of
            ``(key, value)`` pairs sorted alphabetically by key, to the
            corresponding :class:`OneZoneAccretionResult`.

        Examples
        --------
        .. code-block:: python

            from astropy import constants as const
            from astropy import units as u

            results = disk.solve_parameter_grid(
                initial_conditions_grid={
                    "M_D_0": [
                        0.05 * const.M_sun,
                        0.1 * const.M_sun,
                    ]
                },
                runtime_parameters_grid={
                    "alpha": [0.05, 0.1],
                    "M_BH": [3.0 * const.M_sun],
                },
                t_span=(1e6 * u.s, 5e9 * u.s),
            )
            for combo, result in results.items():
                print(
                    dict(combo),
                    result.success,
                    result.n_steps,
                )
        """
        import itertools

        from tqdm.auto import tqdm

        from triceratops.utils.config import triceratops_config

        # Build ordered lists of (key, [values]) so itertools.product gives
        # consistent column ordering and the tuple key is reproducible.
        ic_items = sorted(initial_conditions_grid.items())
        rp_items = sorted(runtime_parameters_grid.items())

        ic_keys = [k for k, _ in ic_items]
        ic_vals = [v for _, v in ic_items]
        rp_keys = [k for k, _ in rp_items]
        rp_vals = [v for _, v in rp_items]

        all_combos = list(itertools.product(*(ic_vals + rp_vals)))
        n_ic = len(ic_keys)

        results: dict = {}
        for combo in tqdm(
            all_combos,
            total=len(all_combos),
            desc=f"Parameter grid ({type(self).__name__})",
            bar_format=triceratops_config["system.appearance.progress_bar_format"],
            disable=triceratops_config["system.appearance.disable_progress_bars"],
        ):
            ic_combo = dict(zip(ic_keys, combo[:n_ic], strict=False))
            rp_combo = dict(zip(rp_keys, combo[n_ic:], strict=False))

            # Canonical key: sorted (name, value) pairs spanning both dicts.
            combo_key = tuple(sorted({**ic_combo, **rp_combo}.items()))

            result = self.solve(
                initial_conditions=ic_combo,
                runtime_parameters=rp_combo,
                t_span=t_span,
                max_steps=max_steps,
                fail_fast=fail_fast,
            )
            results[combo_key] = result

        return results

    def solve_initial_state(
        self,
        initial_conditions: dict,
        runtime_parameters: dict,
        t_span: tuple[Any, Any],
        fail_fast: bool = True,
    ) -> "OneZoneAccretionResult":
        r"""Evaluate the closure at the initial state without integrating forward.

        Runs the Cython integrator for exactly one step, so the returned
        result contains a single column (column 0) that holds the initial
        conditions with their full thermodynamic closure computed.  The state
        is *not* evolved; this call is therefore much cheaper than a full
        :meth:`solve` and is useful for:

        - Checking that initial conditions produce physically reasonable
          derived quantities before committing to a long run.
        - Inference pipelines that need to evaluate the disk state at a
          single point in time without stepping forward.

        Parameters
        ----------
        initial_conditions : dict
            Initial state of the disk (same keys as :meth:`solve`).
        runtime_parameters : dict
            Per-solve physics parameters (same keys as :meth:`solve`).
        t_span : tuple
            Integration interval ``(t_start, t_end)``; only ``t_start``
            is used (``t_end`` must be strictly greater, but the integrator
            will write column 0 and return after one step regardless).
        fail_fast : bool, optional
            Forwarded to :meth:`solve`.

        Returns
        -------
        OneZoneAccretionResult
            Result with ``n_steps == 1``; ``result_array`` has shape
            ``(n_result_fields, 1)``.

        Examples
        --------
        .. code-block:: python

            result = disk.solve_initial_state(
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
                result.data["T_c"]
            )  # midplane temperature at t_start
        """
        return self.solve(
            initial_conditions=initial_conditions,
            runtime_parameters=runtime_parameters,
            t_span=t_span,
            max_steps=1,
            fail_fast=fail_fast,
        )

    def test_initial_state(
        self,
        initial_conditions: dict,
        runtime_parameters: dict,
        t_span: tuple[Any, Any],
    ) -> dict:
        r"""Return a diagnostic report for the closure at the initial state.

        Calls :meth:`solve_initial_state` and inspects the resulting derived
        quantities for physical consistency.  No exceptions are raised; each
        check is reported as a ``(passed, value, note)`` tuple so the caller
        can decide how to react.

        Checks performed
        ----------------
        - **finite** — every result field is finite and non-NaN.
        - **T_c_positive** — midplane temperature is positive.
        - **tau_thick** — optical depth :math:`\tau \geq 1` (thick-disk
          assumption valid).
        - **H_over_R** — :math:`H/R < 1` (geometrically thin-disk assumption
          valid).
        - **R_D_gt_R_in** — disk outer radius exceeds the inner truncation
          radius (disk not fully drained).

        Parameters
        ----------
        initial_conditions : dict
            Forwarded to :meth:`solve_initial_state`.
        runtime_parameters : dict
            Forwarded to :meth:`solve_initial_state`.
        t_span : tuple
            Forwarded to :meth:`solve_initial_state`.

        Returns
        -------
        dict
            Keyed by check name.  Each value is a dict::

                {
                    "passed": bool,
                    "value":  the computed quantity (Quantity or float),
                    "note":   str description,
                }

            An additional ``"result"`` key holds the raw
            :class:`OneZoneAccretionResult`.

        Examples
        --------
        .. code-block:: python

            report = disk.test_initial_state(
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
            for name, chk in report.items():
                if name == "result":
                    continue
                status = "OK" if chk["passed"] else "FAIL"
                print(
                    f"  [{status}] {name}: {chk['value']}  — {chk['note']}"
                )
        """
        result = self.solve_initial_state(
            initial_conditions=initial_conditions,
            runtime_parameters=runtime_parameters,
            t_span=t_span,
            fail_fast=True,
        )

        data = result.data
        report: dict = {"result": result}

        # --- finite ----------------------------------------------------- #
        all_finite = all(bool(np.all(np.isfinite(v.value if isinstance(v, u.Quantity) else v))) for v in data.values())
        report["finite"] = {
            "passed": all_finite,
            "value": all_finite,
            "note": "All result fields are finite and non-NaN.",
        }

        # --- T_c_positive ------------------------------------------------- #
        if "T_c" in data:
            T_c_val = data["T_c"][0]
            report["T_c_positive"] = {
                "passed": bool(T_c_val.value > 0),
                "value": T_c_val,
                "note": "Midplane temperature T_c > 0.",
            }

        # --- tau_thick ---------------------------------------------------- #
        if "tau" in data:
            tau_val = data["tau"][0]
            report["tau_thick"] = {
                "passed": bool(tau_val.value >= 1.0),
                "value": tau_val,
                "note": "Optical depth tau >= 1 (thick-disk assumption).",
            }

        # --- H/R ---------------------------------------------------------- #
        if "H" in data and "R_D" in data:
            H_over_R = (data["H"][0] / data["R_D"][0]).decompose()
            report["H_over_R"] = {
                "passed": bool(H_over_R.value < 1.0),
                "value": H_over_R,
                "note": "H/R < 1 (geometrically thin-disk assumption).",
            }

        # --- R_D > R_in --------------------------------------------------- #
        if "R_D" in data and "R_in" in runtime_parameters:
            R_D_val = data["R_D"][0]
            R_in_val = ensure_in_units(runtime_parameters["R_in"], "cm") * u.cm
            report["R_D_gt_R_in"] = {
                "passed": bool(R_D_val > R_in_val),
                "value": R_D_val,
                "note": f"Disk outer radius R_D > R_in ({R_in_val:.3e}).",
            }

        return report

    # ==================================================== #
    # Helpers and Utilities                                #
    # ==================================================== #
    def generate_initial_conditions(
        self,
        M_BH: Any,
        M_D_0: Optional[Any] = None,
        J_D_0: Optional[Any] = None,
        R_D_0: Optional[Any] = None,
    ) -> dict:
        r"""Compute a complete set of initial conditions from any two of the three disk state variables.

        Uses the Metzger+08 kinematic constraint

        .. math::

            J_D = \xi\, M_D \sqrt{G M_{\rm BH} R_D},
            \qquad \xi \equiv B / A,

        to infer the missing variable.  Exactly two of ``M_D_0``,
        ``J_D_0``, ``R_D_0`` must be supplied; the third is computed and
        included in the returned dict.

        Parameters
        ----------
        M_BH : float or Quantity
            Black hole mass.  Converted to grams if a Quantity is supplied.
        M_D_0 : float or Quantity, optional
            Initial disk mass.  Converted to grams if a Quantity is supplied.
        J_D_0 : float or Quantity, optional
            Initial disk angular momentum.  Converted to g cm² s⁻¹ if a
            Quantity is supplied.
        R_D_0 : float or Quantity, optional
            Initial disk outer radius.  Converted to cm if a Quantity is
            supplied.

        Returns
        -------
        dict
            ``{"M_D_0": Quantity [g], "J_D_0": Quantity [g cm² s⁻¹]}`` — the
            two keys consumed by :meth:`solve`.

        Raises
        ------
        AttributeError
            If the subclass does not define ``_A`` and ``_B`` (the Metzger+08
            geometry constants).
        ValueError
            If the number of supplied variables is not exactly two.

        Examples
        --------
        .. code-block:: python

            ic = disk.generate_initial_conditions(
                M_BH=3.0 * const.M_sun,
                M_D_0=0.1 * const.M_sun,
                R_D_0=3.0e13 * u.cm,
            )
            result = disk.solve(
                initial_conditions=ic,
                runtime_parameters={
                    "M_BH": 3.0 * const.M_sun,
                    "R_in": 3.0e6 * u.cm,
                    "alpha": 0.1,
                },
                t_span=(1.0e6 * u.s, 5.0e9 * u.s),
            )
        """
        from astropy import constants as const

        # Define the constants.
        A, B, _ = 1.62, 1.33, 1.6
        xi = B / A

        supplied = {k: v for k, v in [("M_D_0", M_D_0), ("J_D_0", J_D_0), ("R_D_0", R_D_0)] if v is not None}
        if len(supplied) != 2:
            raise ValueError(f"Exactly 2 of M_D_0, J_D_0, R_D_0 must be supplied; got {list(supplied.keys())}.")

        if "M_D_0" not in supplied:
            # Solve for M_D_0 given J_D_0 and R_D_0.
            M_D_0 = J_D_0 / (xi * np.sqrt(const.G * M_BH * R_D_0))
            return {"M_D_0": M_D_0 * u.g, "J_D_0": J_D_0}

        elif "J_D_0" not in supplied:
            # Solve for J_D_0 given M_D_0 and R_D_0.
            J_D_0 = xi * M_D_0 * np.sqrt(const.G * M_BH * R_D_0)
            return {"M_D_0": M_D_0, "J_D_0": J_D_0}

        else:
            # M_D_0 and J_D_0 both supplied — normalise units and return.
            return {"M_D_0": M_D_0, "J_D_0": J_D_0}


# ==================================================== #
# Result Container                                     #
# ==================================================== #
@dataclass
class OneZoneAccretionResult:
    r"""
    Result container for a one-zone accretion disk ODE solve.

    Wraps the raw Cython result array (shape ``(n_result_fields, N)``, linear
    CGS) and exposes derived quantities via the :attr:`data` property.

    **Accessing state variables**

    .. code-block:: python

        result = disk.solve(...)
        print(
            result.M_D_solar
        )  # disk mass in solar masses, shape (N,)
        print(
            result.J_D
        )  # angular momentum in g cm^2 / s, shape (N,)

    **Accessing derived quantities**

    .. code-block:: python

        data = result.data
        print(data["T_c"])  # central temperature vs time
        print(data["R_D"])  # disk outer radius vs time

    **Subscript access**

    .. code-block:: python

        T_c = result["T_c"]

    **HDF5 persistence**

    .. code-block:: python

        result.to_hdf5("disk_run.h5", overwrite=True)
        loaded = OneZoneAccretionResult.from_hdf5(
            "disk_run.h5"
        )

    Parameters
    ----------
    model : OneZoneAccretionDiskBase
        The disk model instance that produced this result.
    result_array : np.ndarray
        Raw Cython result array, shape ``(n_result_fields, N)``, linear CGS.
    run_params : dict
        Processed runtime parameters as returned by
        :meth:`~OneZoneAccretionDiskBase.process_runtime_parameters`.
    success : bool
        Whether the integration completed successfully.
    message : str
        Integrator status message.

    See Also
    --------
    OneZoneAccretionDiskBase.solve :
        The method that constructs and returns a :class:`OneZoneAccretionResult`.
    """

    model: "OneZoneAccretionDiskBase"
    result_array: np.ndarray
    run_params: _RunParams
    success: bool
    message: str

    FORMAT: ClassVar[str] = "OneZoneAccretionResult"
    VERSION: ClassVar[str] = __pkg_version__

    # User-added fields (not derived from result_array).
    _extra_fields: Optional[dict] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.result_array = np.asarray(self.result_array, dtype=float)
        if self.result_array.ndim != 2:
            raise ValueError(f"result_array must be 2-D, got shape {self.result_array.shape}.")

    # ---------------------------------------------- #
    # State-variable Properties                      #
    # ---------------------------------------------- #
    @property
    def t(self) -> np.ndarray:
        """Integration times in seconds, shape ``(N,)``."""
        return self.result_array[1, :]

    @property
    def M_D(self) -> u.Quantity:
        """Disk mass :math:`M_D` in grams, shape ``(N,)``."""
        return self.result_array[2, :] * u.g

    @property
    def M_D_solar(self) -> u.Quantity:
        """Disk mass :math:`M_D` in solar masses, shape ``(N,)``."""
        return self.M_D.to(u.Msun)

    @property
    def J_D(self) -> u.Quantity:
        r"""Disk angular momentum :math:`J_D` in g cm² s⁻¹, shape ``(N,)``."""
        return self.result_array[3, :] * u.g * u.cm**2 / u.s

    @property
    def t_quantity(self) -> u.Quantity:
        """Integration times as a unit-bearing Quantity in seconds."""
        return self.t * u.s

    @property
    def n_steps(self) -> int:
        """Number of stored time steps ``N``."""
        return self.result_array.shape[1]

    @property
    def field_list(self) -> list[str]:
        """Field names declared in :attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS`."""
        return list(self.model.RESULT_FIELDS.keys())

    @property
    def derived_field_list(self) -> list[str]:
        """Declared field names plus user-added field names, sorted."""
        available: set = set(self.field_list) | {"t", "M_D", "J_D"}
        if self._extra_fields:
            available |= set(self._extra_fields.keys())
        return sorted(available)

    @property
    def data(self) -> _DataDict:
        """Return a dict of unit-bearing arrays for every key in ``model.RESULT_FIELDS``.

        Fields are read directly from :attr:`result_array` by the row index
        given in :attr:`~OneZoneAccretionDiskBase.CYTHON_FIELD_MAP`.  Fields
        mapped to ``None`` are omitted.  The ``mdot`` field has its sign
        flipped (``dM_dt`` is stored negative).  User-added fields from
        :meth:`add_field` are merged last.

        The dict always includes ``"t"``, ``"M_D"``, and ``"J_D"`` as well.

        Returns
        -------
        dict
            Keys: ``"t"``, ``"M_D"``, ``"J_D"``, every entry in
            :attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS`, and any
            user-added fields.
        """
        out: dict = {
            "t": self.t * u.s,
            "M_D": self.M_D,
            "J_D": self.J_D,
        }
        for name, spec in self.model.RESULT_FIELDS.items():
            idx = self.model.CYTHON_FIELD_MAP.get(name)
            units_str = spec["units"]
            unit = u.Unit(units_str) if units_str else u.dimensionless_unscaled
            if idx is not None:
                arr = self.result_array[idx, :].copy()
                if name == "mdot":
                    arr = -arr  # dM_dt is stored negative in the Cython integrator
                out[name] = arr * unit

        # Python-side derived fields (CYTHON_FIELD_MAP value is None)
        derived = self.model._compute_derived_result_fields(self.result_array, self.run_params)
        for name, spec in self.model.RESULT_FIELDS.items():
            if self.model.CYTHON_FIELD_MAP.get(name) is None and name in derived:
                units_str = spec["units"]
                unit = u.Unit(units_str) if units_str else u.dimensionless_unscaled
                out[name] = derived[name] * unit

        # User-added fields
        if self._extra_fields:
            out.update(self._extra_fields)
        return out

    @property
    def initial_state(self) -> _DataDict:
        """All data fields evaluated at the first time step.

        Returns
        -------
        dict
            Same keys as :attr:`data` but each value is the scalar
            (zero-dimensional Quantity or float) at step 0.
        """
        return {k: v[0] for k, v in self.data.items()}

    @property
    def final_state(self) -> _DataDict:
        """All data fields evaluated at the last time step.

        Returns
        -------
        dict
            Same keys as :attr:`data` but each value is the scalar
            (zero-dimensional Quantity or float) at step -1.
        """
        return {k: v[-1] for k, v in self.data.items()}

    # ---------------------------------------------- #
    # Dunder Methods                                 #
    # ---------------------------------------------- #

    def __len__(self) -> int:
        """Return the number of stored time steps.

        Returns
        -------
        int
            Equivalent to :attr:`n_steps`.

        """
        return self.n_steps

    def __repr__(self) -> str:
        """Return a concise, unambiguous string representation.

        Returns
        -------
        str
            ``OneZoneAccretionResult(model=..., n_steps=N, success=True/False)``
        """
        return f"OneZoneAccretionResult(model={self.model!r}, n_steps={self.n_steps}, success={self.success!r})"

    def __str__(self) -> str:
        r"""Return a human-readable summary of the result.

        Returns
        -------
        str
            Multi-line plain-text summary including model identity, time range,
            initial and final disk masses, solver status, and cache state.

        Examples
        --------

        .. code-block:: python

            print(result)

            OneZoneAccretionResult
              Model    : GasPressureElectronScatteringDisk(mu=0.62)
              N steps  : 20
              t range  : [1.000e+06, 5.000e+08] s
              M_D(0)   : 0.100 solMass
              M_D(-1)  : 0.073 solMass
              Status   : OK — The solver successfully reached the end of the integration interval.
              Cache    : cold
        """
        lines = [
            "OneZoneAccretionResult",
            f"  Model    : {self.model!r}",
            f"  N steps  : {self.n_steps}",
            f"  t range  : [{self.t[0]:.3e}, {self.t[-1]:.3e}] s",
            f"  M_D(0)   : {self.M_D[0].to(u.Msun):.3f}",
            f"  M_D(-1)  : {self.M_D[-1].to(u.Msun):.3f}",
            f"  Status   : {'OK' if self.success else 'FAILED'} \u2014 {self.message}",
        ]
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        """Return ``True`` if *other* is an equivalent result."""
        if not isinstance(other, OneZoneAccretionResult):
            return NotImplemented
        return (
            self.model == other.model
            and np.array_equal(self.result_array, other.result_array)
            and self.run_params == other.run_params
            and self.success == other.success
            and self.message == other.message
        )

    def __contains__(self, key: str) -> bool:
        """Return ``True`` if *key* is present in :attr:`data`."""
        return key in self.data

    def __getitem__(self, key: str) -> _FieldValue:
        """Return a single field by name from :attr:`data`.

        Parameters
        ----------
        key : str
            Name of a field (see :attr:`data`).

        Returns
        -------
        `~astropy.units.Quantity` or `numpy.ndarray`

        Raises
        ------
        KeyError
            If *key* is not present in :attr:`data`.
        """
        return self.data[key]

    def keys(self) -> list[str]:
        """Return all field names present in :attr:`data`, sorted."""
        return sorted(self.data.keys())

    # ---------------------------------------------- #
    # HDF5 Persistence                               #
    # ---------------------------------------------- #
    def to_hdf5(self, filename: _FilePath, overwrite: bool = False) -> None:
        r"""
        Write the result to an HDF5 file.

        The model specification is serialised as a JSON string stored in an
        HDF5 attribute, so the model can be reconstructed automatically by
        :meth:`from_hdf5` without any user-supplied configuration.

        **File layout**::

            result.h5
            ├── [attr] format      : "OneZoneAccretionResult"
            ├── [attr] version     : str  (triceratops package version)
            ├── [attr] model_spec  : JSON str  ← model.to_spec_dict()
            ├── [attr] run_params  : JSON str  ← processed float dict
            ├── [attr] success     : bool
            ├── [attr] message     : str
            ├── result_array : float64, shape (n_result_fields, N)  [linear CGS]
            └── extra_fields/  : group; present only if add_field() was called
                └── <name>     : float64, shape (N,)  [attr: units="..."]

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
            f.create_dataset("result_array", data=self.result_array)

            if self._extra_fields:
                grp = f.create_group("extra_fields")
                for key, val in self._extra_fields.items():
                    arr = val.value if isinstance(val, u.Quantity) else np.asarray(val)
                    ds = grp.create_dataset(key, data=arr)
                    ds.attrs["units"] = str(val.unit) if isinstance(val, u.Quantity) else ""

    @classmethod
    def from_hdf5(cls, filename: _FilePath) -> "OneZoneAccretionResult":
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
            Fully reconstructed result with a live model instance.  Any
            ``extra_fields/`` group written by :meth:`add_field` is also
            restored.

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
            data = loaded.data

        See Also
        --------
        to_hdf5 :
            Instance method that writes the file this class method reads.
        """
        import h5py

        extra_fields = None

        with h5py.File(Path(filename), "r") as f:
            fmt = f.attrs["format"]
            if fmt != cls.FORMAT:
                raise ValueError(f"format mismatch: file has '{fmt}', expected '{cls.FORMAT}'.")
            spec_dict = json.loads(f.attrs["model_spec"])
            run_params = json.loads(f.attrs["run_params"])
            success = bool(f.attrs["success"])
            message = str(f.attrs["message"])
            result_array = f["result_array"][:]

            if "extra_fields" in f:
                extra_fields = {}
                grp = f["extra_fields"]
                for key in grp:
                    arr = grp[key][:]
                    units_str = grp[key].attrs.get("units", "")
                    extra_fields[key] = arr * u.Unit(units_str) if units_str else arr

        model = _build_model_from_spec(spec_dict)
        result = cls(
            model=model,
            result_array=result_array,
            run_params=run_params,
            success=success,
            message=message,
        )
        result._extra_fields = extra_fields
        return result

    # ---------------------------------------------- #
    # Plotting                                       #
    # ---------------------------------------------- #

    def plot_state_variables(self, fig=None, axes=None) -> tuple:
        r"""Plot disk mass and angular momentum versus time.

        Produces a two-panel log-log plot: disk mass :math:`M_D` in solar
        masses (left) and angular momentum :math:`J_D` in CGS (right), both
        as functions of time.

        Parameters
        ----------
        fig : `~matplotlib.figure.Figure`, optional
            Existing figure to draw into.  If ``None``, a new figure is created.
        axes : array-like of `~matplotlib.axes.Axes`, optional
            A sequence of two axes ``[ax_M_D, ax_J_D]``.  If ``None``, axes
            are created from *fig*.

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
        axes : `numpy.ndarray` of `~matplotlib.axes.Axes`, shape (2,)

        Examples
        --------
        .. code-block:: python

            fig, axes = result.plot_state_variables()
            fig.savefig("state_variables.png", dpi=150)
        """
        import matplotlib.pyplot as plt

        from triceratops.utils.plot_utils import set_plot_style

        set_plot_style()

        if fig is None or axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].loglog(self.t, self.M_D.to(u.Msun).value)
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel(r"$M_D\;[M_\odot]$")
        axes[0].set_title("Disk mass")

        axes[1].loglog(self.t, self.J_D.value)
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel(r"$J_D\;[\mathrm{g\,cm^2\,s^{-1}}]$")
        axes[1].set_title("Disk angular momentum")

        fig.suptitle(repr(self.model))
        fig.tight_layout()

        return fig, np.asarray(axes)

    def plot_field(self, field_name: str, fig=None, ax=None, **plot_kwargs) -> tuple:
        r"""Plot a single derived field versus time.

        Parameters
        ----------
        field_name : str
            Name of a field declared in
            :attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS` or one of the
            built-in state keys ``'t'``, ``'M_D'``, ``'J_D'``.
        fig : `~matplotlib.figure.Figure`, optional
            Existing figure to draw into.  If ``None``, a new figure is created.
        ax : `~matplotlib.axes.Axes`, optional
            Existing axes to draw into.  If ``None``, axes are created from *fig*.
        **plot_kwargs
            Additional keyword arguments forwarded to
            :func:`matplotlib.axes.Axes.loglog`.

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
        ax : `~matplotlib.axes.Axes`

        Raises
        ------
        KeyError
            If *field* is not present in the data dict.

        Examples
        --------
        .. code-block:: python

            fig, ax = result.plot_field("T_c")
            ax.set_title("Central temperature")
        """
        from triceratops.utils.plot_utils import resolve_fig_axes, set_plot_style

        set_plot_style()
        fig, ax = resolve_fig_axes(fig, ax)

        val = self[field_name]
        y = val.value if isinstance(val, u.Quantity) else np.asarray(val)
        units_str = str(val.unit) if isinstance(val, u.Quantity) else ""

        ax.loglog(self.t, np.abs(y), **plot_kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{field_name}" + (f" [{units_str}]" if units_str else ""))

        if field_name in self.model.RESULT_FIELDS:
            ax.set_title(self.model.RESULT_FIELDS[field_name]["description"])

        return fig, ax

    def plot_all_fields(self, ncols: int = 3, fig=None, axes=None) -> tuple:
        r"""Plot every :attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS` entry in a grid.

        Produces one log-log subplot per declared result field, arranged in a
        grid with *ncols* columns.  Unused grid cells are hidden.

        Parameters
        ----------
        ncols : int, optional
            Number of columns in the subplot grid.  Default is 3.
        fig : `~matplotlib.figure.Figure`, optional
            Existing figure to draw into.  If ``None``, a new figure is created.
        axes : array-like of `~matplotlib.axes.Axes`, optional
            Pre-allocated axes matching the grid shape.  If ``None``, axes are
            created from *fig*.

        Returns
        -------
        fig : `~matplotlib.figure.Figure`
        axes : `numpy.ndarray` of `~matplotlib.axes.Axes`

        Examples
        --------
        .. code-block:: python

            fig, axes = result.plot_all_fields(ncols=4)
            fig.savefig(
                "all_fields.png", dpi=150, bbox_inches="tight"
            )
        """
        import matplotlib.pyplot as plt

        from triceratops.utils.plot_utils import set_plot_style

        set_plot_style()

        fields = list(self.model.RESULT_FIELDS.keys())
        n_fields = len(fields)
        nrows = (n_fields + ncols - 1) // ncols

        if fig is None or axes is None:
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))

        axes_flat = np.asarray(axes).ravel()
        data = self.data

        for ax, field_name in zip(axes_flat, fields, strict=False):
            val = data[field_name]
            y = val.value if isinstance(val, u.Quantity) else np.asarray(val)
            units_str = str(val.unit) if isinstance(val, u.Quantity) else ""
            ax.loglog(self.t, np.abs(y))
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(f"[{units_str}]" if units_str else "\u2014")
            ax.set_title(field_name)

        for ax in axes_flat[n_fields:]:
            ax.set_visible(False)

        fig.suptitle(f"{type(self.model).__name__} \u2014 all fields", fontsize=13)
        fig.tight_layout()

        return fig, np.asarray(axes)

    # ---------------------------------------------- #
    # Field Interaction Methods                      #
    # ---------------------------------------------- #

    def get_field(self, name: str) -> _FieldValue:
        """Return a single field by name from :attr:`data` with an informative error.

        Similar to ``result[name]`` but raises a :exc:`KeyError` that lists
        all available field names when *name* is not found, making interactive
        exploration more convenient.

        Parameters
        ----------
        name : str
            Name of a field in the data dict (see :meth:`keys`).

        Returns
        -------
        `~astropy.units.Quantity` or `numpy.ndarray`
            The array for *name* from :attr:`data`, with units attached where
            declared.

        Raises
        ------
        KeyError
            If *name* is not present in the data dict.  The error
            message lists all available keys.

        Examples
        --------

        .. code-block:: python

            $ result.get_field("T_c")
            <Quantity [...] K>
            $ result.get_field("typo")
            KeyError: "Field 'typo' not found. Available fields: ['J_D', 'M_D', ...]"
        """
        d = self.data
        if name not in d:
            raise KeyError(f"Field '{name}' not found. Available fields: {sorted(d.keys())}")
        return d[name]

    def has_field(self, name: str) -> bool:
        """Return ``True`` if *name* is present in :attr:`data`."""
        return name in self.data

    def add_field(self, name: str, value: _FieldValue, overwrite: bool = False) -> None:
        r"""Add a user-computed field that is included in :attr:`data` and saved to HDF5.

        Parameters
        ----------
        name : str
            Field name.  Must not clash with protected fields
            (``RESULT_FIELDS`` keys or ``'t'``, ``'M_D'``, ``'J_D'``)
            unless *overwrite* is ``True``.
        value : array-like or `~astropy.units.Quantity`
            Field data, shape ``(N,)`` expected.
        overwrite : bool, optional
            Replace an existing user-added field silently.

        Raises
        ------
        ValueError
            If *name* already exists in ``_extra_fields`` and
            ``overwrite=False``.
        """
        if self._extra_fields is None:
            self._extra_fields = {}
        if name in self.data and not overwrite:
            raise ValueError(f"Field '{name}' already exists. Pass overwrite=True to replace it.")
        self._extra_fields[name] = value

    def remove_field(self, name: str) -> None:
        """Remove a user-added field.

        Parameters
        ----------
        name : str
            Name of the field to remove.  Must be a user-added field (not a
            protected field from ``RESULT_FIELDS`` or the built-in state keys).

        Raises
        ------
        ValueError
            If *name* is a protected field.
        KeyError
            If *name* is not present in user-added fields.
        """
        _protected = set(self.model.RESULT_FIELDS) | {"t", "M_D", "J_D"}
        if name in _protected:
            raise ValueError(f"'{name}' is a protected field and cannot be removed.")
        if self._extra_fields is None or name not in self._extra_fields:
            raise KeyError(
                f"Field '{name}' is not present in user-added fields. Available: {sorted(self._extra_fields or {})}"
            )
        del self._extra_fields[name]

    # ==================================================== #
    # Field Interpolation                                  #
    # ==================================================== #

    def get_field_interpolator(self, field_name: str, **kwargs):
        r"""Return a cubic spline interpolator for a result field as a function of time.

        Builds an :class:`~scipy.interpolate.InterpolatedUnivariateSpline`
        (default ``k=3``) on the stored time grid.  The returned callable
        accepts time in seconds and returns raw float values (units must be
        looked up separately from :attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS`
        or the known built-in fields).

        Parameters
        ----------
        field_name : str
            Name of a field in :attr:`data` (``"t"``, ``"M_D"``, ``"J_D"``,
            or any :attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS` key).
        **kwargs
            Forwarded to :class:`~scipy.interpolate.InterpolatedUnivariateSpline`
            (e.g. ``k=1`` for linear, ``ext=3`` to clamp extrapolation).

        Returns
        -------
        `~scipy.interpolate.InterpolatedUnivariateSpline`
            Callable ``f(t_s) -> float`` where ``t_s`` is time in seconds.

        Raises
        ------
        KeyError
            If *field_name* is not present in :attr:`data`.

        Examples
        --------
        .. code-block:: python

            spl = result.get_field_interpolator("T_c")
            T_c_at_1e8 = spl(
                1e8
            )  # raw float in the field's CGS units
        """
        from scipy.interpolate import InterpolatedUnivariateSpline

        arr = self.data[field_name]
        y = arr.value if isinstance(arr, u.Quantity) else np.asarray(arr, dtype=float)
        kwargs.setdefault("k", 3)
        return InterpolatedUnivariateSpline(self.t, y, **kwargs)

    def interpolate_field(
        self,
        field_name: str,
        t_new: Any,
        interpolator=None,
        **kwargs,
    ) -> _FieldValue:
        r"""Evaluate a result field at arbitrary time(s) via spline interpolation.

        Parameters
        ----------
        field_name : str
            Name of a field in :attr:`data`.
        t_new : float or Quantity or array-like
            Time(s) at which to evaluate the field.  Quantities are converted
            to seconds automatically.
        interpolator : callable, optional
            A pre-built spline from :meth:`get_field_interpolator`.  If
            ``None``, one is built on-the-fly.
        **kwargs
            Forwarded to :meth:`get_field_interpolator` when *interpolator*
            is ``None``.

        Returns
        -------
        `~astropy.units.Quantity` or `numpy.ndarray`
            Interpolated value(s) with units attached when the field declares
            them in :attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS`.

        Examples
        --------
        .. code-block:: python

            T_c = result.interpolate_field("T_c", 5e8 * u.s)
        """
        if interpolator is None:
            interpolator = self.get_field_interpolator(field_name, **kwargs)

        if isinstance(t_new, u.Quantity):
            t_s = np.asarray(ensure_in_units(t_new, "s"), dtype=float)
        else:
            t_s = np.asarray(t_new, dtype=float)

        values = interpolator(t_s)

        # Reattach units when declared in RESULT_FIELDS.
        spec = self.model.RESULT_FIELDS.get(field_name, {})
        units_str = spec.get("units", "")
        if units_str:
            return values * u.Unit(units_str)

        # Built-in state fields with known units.
        _builtin_units = {"M_D": u.g, "J_D": u.g * u.cm**2 / u.s, "t": u.s}
        if field_name in _builtin_units:
            return values * _builtin_units[field_name]

        return values

    def compute_thin_disk_effective_temperature(
        self,
        radius: Any,
        time: Any,
        interpolator=None,
    ) -> u.Quantity:
        r"""Compute the local effective temperature at *radius* and *time*.

        Uses the standard thin-disk temperature profile:

        .. math::

            T_{\rm eff}(R) = T_{\rm eff}(R_D)\,\left(\frac{R_D}{R}\right)^{3/4}

        where :math:`T_{\rm eff}(R_D)` is the one-zone effective temperature
        (the outer-edge temperature stored in the result) evaluated at the
        requested time via spline interpolation.

        Parameters
        ----------
        radius : float or Quantity
            Radial location at which to evaluate :math:`T_{\rm eff}`.
            Quantities are converted to centimetres.
        time : float or Quantity
            Time at which to evaluate :math:`T_{\rm eff}(R_D)`.
            Quantities are converted to seconds.
        interpolator : callable, optional
            Pre-built spline for the ``"T_eff"`` field from
            :meth:`get_field_interpolator`.  Built on-the-fly if ``None``.

        Returns
        -------
        `~astropy.units.Quantity`
            Effective temperature at *radius* in Kelvin.

        Examples
        --------
        .. code-block:: python

            T = result.compute_thin_disk_effective_temperature(
                radius=1e10 * u.cm, time=1e7 * u.s
            )
        """
        T_eff_RD = self.interpolate_field("T_eff", time, interpolator=interpolator)
        R_D = self.interpolate_field("R_D", time)

        T_eff_RD_K = T_eff_RD.value if isinstance(T_eff_RD, u.Quantity) else float(T_eff_RD)
        R_D_cm = R_D.value if isinstance(R_D, u.Quantity) else float(R_D)
        radius_cm = float(ensure_in_units(radius, "cm"))

        return T_eff_RD_K * (R_D_cm / radius_cm) ** 0.75 * u.K

    def compute_luminosity(self, time: Any, interpolator=None) -> u.Quantity:
        r"""Compute the disk bolometric luminosity at *time*.

        Uses the thin-disk viscous-dissipation estimate:

        .. math::

            L = \frac{G\,M_{\rm BH}\,\dot{M}}{R_{\rm in}}

        where :math:`\dot{M}` is interpolated from the result at *time* and
        :math:`R_{\rm in}` is read from the stored runtime parameters.

        Parameters
        ----------
        time : float or Quantity
            Time at which to evaluate the luminosity.
        interpolator : callable, optional
            Pre-built spline for the ``"mdot"`` field.  Built on-the-fly if
            ``None``.

        Returns
        -------
        `~astropy.units.Quantity`
            Luminosity in erg s⁻¹.

        Examples
        --------
        .. code-block:: python

            L = result.compute_luminosity(1e8 * u.s)
            print(L.to(u.Lsun))
        """
        from astropy.constants import G

        M_BH = np.exp(self.run_params["log_M_BH"])  # g
        R_in = np.exp(self.run_params["log_R_in"])  # cm

        mdot = self.interpolate_field("mdot", time, interpolator=interpolator)
        mdot_cgs = mdot.value if isinstance(mdot, u.Quantity) else float(mdot)

        return G * M_BH * mdot_cgs / (2 * R_in) * u.Unit("erg/s")

    # ==================================================== #
    # Data Management
    # ==================================================== #
    def to_dataframe(self):
        """Convert the result data to a Pandas DataFrame.

        Returns
        -------
        `pandas.DataFrame`
            DataFrame with one column per field and one row per time step.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe() but is not installed. Install it with: pip install pandas"
            ) from exc

        return pd.DataFrame(self.data)

    def to_table(self) -> QTable:
        r"""Return :attr:`data` as an Astropy :class:`~astropy.table.QTable`.

        Returns
        -------
        `~astropy.table.QTable`
            Table with one column per field and one row per time step.
        """
        return QTable(self.data)


def _build_model_from_spec(spec_dict: _SpecDict) -> "OneZoneAccretionDiskBase":
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
