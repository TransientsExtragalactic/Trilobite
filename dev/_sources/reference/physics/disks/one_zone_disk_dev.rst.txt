.. _one_zone_disk_dev:

=================================================
Developer Guide: Adding One-Zone Disk Closures
=================================================

This guide walks through every step required to add a new thermodynamic closure to
Triceratops's one-zone accretion disk framework.  A "closure" is the combination of an
equation of state, an opacity prescription, and heating/cooling terms that closes the
energy-balance equation and determines the midplane temperature :math:`T_c` at each
integration step.

For user documentation on running the existing models see :ref:`one_zone_disk`.
For the underlying theory see :ref:`one_zone_disk_theory`.

.. contents::
    :local:
    :depth: 2

----

Architecture Overview
----------------------

The one-zone disk framework is built in two layers:

.. code-block:: none

    Python layer (base.py, core.py)
    ├── OneZoneAccretionDiskBase   (abstract, validated by _OneZoneMeta)
    │   ├── GasPressureDisk       (gP_esDisk)
    │   ├── FullPressureDisk      (igP_esDisk)
    │   └── AdvectiveDisk         (igP_es_advDisk)
    │
    Cython layer (models/*.pyx)
    ├── gPClosure                 ← compiled C function pointers
    ├── igPClosure
    └── igPAdvClosure

The Python class declares *what* parameters the model has, *how* to pack them for Cython,
and *how* to serialize the model.  The Cython closure implements *the actual physics* — the
energy-balance root-find, the ODE derivative, and the output writer — inside a GIL-free
explicit-Euler loop.

**Adding a new closure requires three coordinated steps:**

1. Write the Cython extension type (the hot inner loop).
2. Declare the Python model class (parameter schema + metadata).
3. Register a test class that inherits from ``BaseTestOneZoneDisk``.

----

The Metaclass: ``_OneZoneMeta``
---------------------------------

All Python model classes are validated at **class-definition time** by
:class:`~triceratops.dynamics.accretion.one_zone.base._OneZoneMeta`.  You do not
interact with the metaclass directly, but you must satisfy its constraints: the four
class-level declaration dicts must be present and well-formed, or a descriptive
:exc:`TypeError` is raised immediately when the class body is evaluated.

The four required declaration dicts are:

.. list-table::
    :header-rows: 1

    * - Attribute
      - Purpose
      - Where used
    * - ``CONTEXT_PARAMETERS``
      - Constructor arguments fixed for the lifetime of an instance.
      - ``__init__``, ``to_spec_dict``, ``from_spec_dict``, ``__repr__``
    * - ``RUNTIME_PARAMETERS``
      - Per-solve physics inputs passed to :meth:`~.base.OneZoneAccretionDiskBase.solve`.
      - ``process_runtime_parameters``, Cython parameter packing
    * - ``INITIAL_CONDITIONS``
      - ODE initial state consumed by :meth:`~.base.OneZoneAccretionDiskBase.solve`.
      - ``process_initial_conditions``
    * - ``RESULT_FIELDS``
      - All output quantities exposed via ``result.data``.
      - ``OneZoneAccretionResult``, HDF5 serialization, plotting utilities

Each dict entry has a well-defined schema that the metaclass enforces.  See
:ref:`parameter-spec-schema` below for the field definitions.

.. _parameter-spec-schema:

Parameter and Result-Field Specification Schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every entry in ``CONTEXT_PARAMETERS``, ``RUNTIME_PARAMETERS``, and
``INITIAL_CONDITIONS`` must be a dict with exactly these keys:

.. list-table::
    :header-rows: 1
    :widths: 20 15 65

    * - Key
      - Type
      - Description
    * - ``description``
      - ``str`` (non-empty)
      - Human-readable description used in ``__str__`` and documentation.
    * - ``base_units``
      - ``str`` or ``None``
      - The CGS unit string to which the parameter is converted (e.g. ``"g"``,
        ``"cm"``, ``"g*cm**2/s"``).  Use ``None`` for dimensionless parameters.
    * - ``default``
      - any or ``None``
      - Default value in base units.  Use ``None`` for required parameters (no default).
    * - ``log_transform``
      - ``bool``
      - If ``True``, the parameter is stored internally as ``ln(value)`` under the
        key ``"log_{name}"``.  Use for parameters that span many decades.

Every entry in ``RESULT_FIELDS`` must be a dict with exactly these keys:

.. list-table::
    :header-rows: 1
    :widths: 20 15 65

    * - Key
      - Type
      - Description
    * - ``description``
      - ``str`` (non-empty)
      - Human-readable description.
    * - ``units``
      - ``str`` or ``None``
      - Astropy unit string for the output Quantity (e.g. ``"K"``, ``"g"``,
        ``"erg/cm**2/s"``).  Use ``None`` for dimensionless fields.

----

Step 1 — Write the Cython Closure
-----------------------------------

The Cython closure is a subclass of
:class:`~triceratops.dynamics.accretion.one_zone.closure.OneZoneClosure` defined in a
``.pyx`` file under ``triceratops/dynamics/accretion/one_zone/models/``.

The subclass must supply three C function pointers in its ``__cinit__``:

.. list-table::
    :header-rows: 1

    * - Pointer
      - Signature
      - Purpose
    * - ``_closure_fn``
      - ``(state, params, result_ptr) → int``
      - Solves the energy-balance equation for :math:`T_c`, fills a
        ``ClosureResult`` struct, computes :math:`\nu`.
    * - ``_derivative_fn``
      - ``(closure_result, params, dt_ptr) → int``
      - Evaluates :math:`\dot{M}_D`, :math:`\dot{J}_D`, sets the adaptive timestep.
    * - ``_writer_fn``
      - ``(result_array, col, closure_result, state) → int``
      - Writes all output fields to the result array at column index ``col``.

The ``ClosureResult`` struct carries the standard set of thermodynamic and structural
outputs that all closures must compute:

.. code-block:: c

    cdef struct ClosureResult:
        double dM_dt       # g s^{-1}  (< 0: drain, > 0: source)
        double dJ_dt       # g cm^2 s^{-2}
        double t_visc      # s  — warm-start seed for next step's timestep estimate
        double log_T_eff   # ln(K)
        double log_T_c     # ln(K)
        double log_tau     # ln(dimensionless)
        double log_cs      # ln(cm s^{-1})
        double log_nu      # ln(cm^2 s^{-1})
        double log_q_visc  # ln(erg cm^{-2} s^{-1})

If your closure produces **additional output fields** (e.g. advective cooling rate
``Q_adv``), store them in extra ``double`` attributes on the extension type and have the
writer function copy them into the additional rows of the result array.  Set
``n_result_fields`` in ``__cinit__`` to the total number of rows.

Minimal Cython Template
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cython

    # triceratops/dynamics/accretion/one_zone/models/_my_closure.pyx

    from libc.math cimport exp, log, sqrt
    from triceratops.dynamics.accretion.one_zone.closure cimport (
        OneZoneClosure, ClosureResult,
        closure_fn_t, derivative_fn_t, writer_fn_t,
    )

    # ------------------------------------------------------------------ #
    # C-level function implementations                                    #
    # ------------------------------------------------------------------ #

    cdef int _my_closure_fn(
        double[::1] state, double[::1] params,
        ClosureResult* result
    ) noexcept nogil:
        """Solve the energy-balance equation for T_c."""
        # Unpack the state vector: state[0] = M_D, state[1] = J_D
        cdef double M_D = state[0]
        cdef double J_D = state[1]

        # Unpack the parameter vector (order must match _pack_cython_parameters)
        cdef double M_BH  = exp(params[0])   # log_M_BH → M_BH
        cdef double R_in  = exp(params[1])   # log_R_in  → R_in
        cdef double alpha = params[2]
        cdef double mu    = params[3]
        # ... add any additional parameters ...

        # Compute geometry: R_D, Sigma, Omega, H
        # (use the constants _A, _B, _F compiled into the base layer)
        # ...

        # Solve energy balance for T_c:
        # Q+ = Q-   →  9/8 * alpha * c_s^2 * Sigma * Omega = 16*sigma*T_c^4 / (3*kappa*Sigma)
        # For ES opacity with gas EOS: analytic solution exists.
        # For more complex EOS: use the bracket/Brent root-finder.
        # ...

        # Fill the ClosureResult struct
        result.dM_dt = ...
        result.dJ_dt = ...
        result.t_visc = ...
        result.log_T_c = log(T_c)
        # ...
        return 0    # 0 = SUCCESS; non-zero triggers an error in the integrator

    cdef int _my_derivative_fn(
        ClosureResult* cr, double[::1] params, double* dt
    ) noexcept nogil:
        """Compute the adaptive timestep."""
        dt[0] = ...   # e.g. 0.1 * min(M_D / |dM_dt|, J_D / |dJ_dt|)
        return 0

    cdef int _my_writer_fn(
        double[:,::1] result, int col,
        ClosureResult* cr, double[::1] state
    ) noexcept nogil:
        """Write output fields to the result array."""
        # Row indices must match CYTHON_FIELD_MAP in the Python class
        result[0,  col] = state[0]         # M_D
        result[1,  col] = state[1]         # J_D
        result[2,  col] = exp(cr.log_T_c)  # T_c
        # ... write all rows declared in RESULT_FIELDS ...
        return 0

    # ------------------------------------------------------------------ #
    # Extension type                                                      #
    # ------------------------------------------------------------------ #

    cdef class MyClosureType(OneZoneClosure):

        def __cinit__(self, int n_result_fields):
            self.n_result_fields = n_result_fields
            self._closure_fn    = _my_closure_fn
            self._derivative_fn = _my_derivative_fn
            self._writer_fn     = _my_writer_fn

.. important::

    The C function signatures are declared in
    :mod:`triceratops.dynamics.accretion.one_zone.closure` (``.pxd``); import them
    with ``cimport``.  Never call :class:`OneZoneClosure` methods from inside the GIL-free
    functions — only use ``cdef`` functions and C standard-library calls.

----

Step 2 — Declare the Python Model Class
-----------------------------------------

Subclass :class:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase`
and declare the four required class-level dicts.  The full template below includes all
required elements:

.. code-block:: python

    # triceratops/dynamics/accretion/one_zone/core.py  (append or modify)

    from triceratops.dynamics.accretion.one_zone.base import OneZoneAccretionDiskBase
    from triceratops.dynamics.accretion.one_zone._closure_specs import (
        BASE_RUNTIME_PARAMETERS,
        BASE_INITIAL_CONDITIONS,
        BASE_RESULT_FIELDS,
        CYTHON_FIELD_MAP,
    )

    class MyDisk(OneZoneAccretionDiskBase):
        """One-zone disk with a custom thermodynamic closure.

        Parameters
        ----------
        mu : float, optional
            Mean molecular weight of the plasma (dimensionless).  Default 0.6.
        my_param : float, optional
            A custom context parameter (dimensionless).  Default 1.0.
        """

        # Metzger+08 geometry constants (compiled into the Cython layer)
        _A: float = 1.62
        _B: float = 1.33
        _F: float = 1.6

        # ---- Declaration dicts -------------------------------------------

        CONTEXT_PARAMETERS: dict = {
            "mu": {
                "description": "Mean molecular weight of the plasma",
                "base_units": None,      # dimensionless
                "default": 0.6,
                "log_transform": False,
            },
            "my_param": {
                "description": "Custom dimensionless closure parameter",
                "base_units": None,
                "default": 1.0,
                "log_transform": False,
            },
        }

        # Inherit the standard set of runtime parameters.
        RUNTIME_PARAMETERS: dict = BASE_RUNTIME_PARAMETERS

        INITIAL_CONDITIONS: dict = BASE_INITIAL_CONDITIONS

        # Add any extra output fields beyond the base set.
        RESULT_FIELDS: dict = {
            **BASE_RESULT_FIELDS,
            "my_extra_field": {
                "description": "Custom output quantity",
                "units": "erg/cm**2/s",
            },
        }

        CYTHON_FIELD_MAP: dict = {
            **CYTHON_FIELD_MAP,
            "my_extra_field": 17,   # assign the next available row index
        }

        # ---- Required methods --------------------------------------------

        def __init__(self, mu: float = 0.6, my_param: float = 1.0):
            super().__init__(mu=mu, my_param=my_param)

        def _build_cython_closure(self):
            from .models._my_closure import MyClosureType
            n_fields = len(self.RESULT_FIELDS)
            return MyClosureType(n_result_fields=n_fields)

        def _pack_cython_parameters(self, run_params: dict) -> "np.ndarray":
            import numpy as np
            return np.array([
                run_params["log_M_BH"],          # packed as log value
                run_params["log_R_in"],
                run_params["alpha"],
                self._context_parameters["mu"],
                self._context_parameters["my_param"],
            ], dtype=np.float64)

        def to_spec_dict(self) -> dict:
            return {
                "target": f"{type(self).__module__}:{type(self).__name__}",
                "mu":       self._context_parameters["mu"],
                "my_param": self._context_parameters["my_param"],
            }

        @classmethod
        def from_spec_dict(cls, data: dict) -> "MyDisk":
            return cls(**{k: v for k, v in data.items() if k != "target"})

.. note::

    ``BASE_RUNTIME_PARAMETERS``, ``BASE_INITIAL_CONDITIONS``, ``BASE_RESULT_FIELDS``,
    and ``CYTHON_FIELD_MAP`` are centrally defined in
    :mod:`triceratops.dynamics.accretion.one_zone._closure_specs` and contain the
    standard set of parameters and fields shared by all one-zone models.  Reuse them
    unless your closure fundamentally changes the ODE variables or adds extra runtime
    parameters (e.g. fallback rate ``M_fb_0`` and ``t_fb`` for fallback disks).

Adding Fallback Mass Supply
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fallback disks require two additional runtime parameters (``M_fb_0``, ``t_fb``) that enter the
mass and angular-momentum evolution equations.  Use
``FALLBACK_RUNTIME_PARAMETERS`` from ``_closure_specs`` in addition to the base set:

.. code-block:: python

    from triceratops.dynamics.accretion.one_zone._closure_specs import (
        BASE_RUNTIME_PARAMETERS,
        FALLBACK_RUNTIME_PARAMETERS,
    )

    class MyFallbackDisk(MyDisk):
        RUNTIME_PARAMETERS = {
            **BASE_RUNTIME_PARAMETERS,
            **FALLBACK_RUNTIME_PARAMETERS,
        }

----

Step 3 — Register Aliases and Exports
---------------------------------------

If you want backward-compatible aliases (following the existing naming convention):

.. code-block:: python

    # bottom of core.py
    myDisk    = MyDisk     # alias without fallback
    myFbDisk  = MyDisk     # pass fallback=True at construction

Export from the package ``__init__.py``:

.. code-block:: python

    # triceratops/dynamics/accretion/one_zone/__init__.py
    from .core import MyDisk, myDisk

----

Step 4 — Build the Cython Extension
--------------------------------------

Add the new ``.pyx`` file to the build system in ``setup.py`` (or
``pyproject.toml`` if using scikit-build-core):

.. code-block:: python

    Extension(
        "triceratops.dynamics.accretion.one_zone.models._my_closure",
        sources=["triceratops/dynamics/accretion/one_zone/models/_my_closure.pyx"],
        include_dirs=[np.get_include()],
    ),

Rebuild the package:

.. code-block:: bash

    pip install -e ".[dev]"

----

Step 5 — Add Tests
--------------------

The test module ``tests/test_dynamics/test_one_zone_disk_base.py`` provides
:class:`BaseTestOneZoneDisk`, a shared pytest base class that automatically runs the
full suite of over 60 physical-consistency, round-trip, and result-field checks.
Inheriting from it is the easiest way to ensure your new model is correct.

.. code-block:: python

    # tests/test_dynamics/test_my_closure.py
    import numpy as np
    import pytest
    from astropy import constants as const
    from astropy import units as u

    from triceratops.dynamics.accretion.one_zone import MyDisk
    from test_one_zone_disk_base import BaseTestOneZoneDisk


    class TestMyDisk(BaseTestOneZoneDisk):
        """Full common test suite for MyDisk.

        Physical setup
        --------------
        - 3 M☉ BH, 0.1 M☉ disk, α = 0.1
        - R_D_0 / R_in = 1e7 — comfortably in the one-zone regime
        """

        MODEL = MyDisk
        DISK_KWARGS = {"mu": 0.62, "my_param": 1.0}

        M_BH  = 3.0 * const.M_sun
        M_D_0 = 0.1 * const.M_sun
        R_IN  = 3.0e6 * u.cm
        R_D_0 = 3.0e13 * u.cm
        ALPHA = 0.1

        # ——— Model-specific assertions ———

        def test_my_param_stored(self):
            """my_param context parameter is stored correctly."""
            assert np.isclose(self._disk()._context_parameters["my_param"], 1.0)

        def test_closure_is_correct_type(self):
            """_build_cython_closure() returns a MyClosureType."""
            from triceratops.dynamics.accretion.one_zone.models._my_closure import (
                MyClosureType,
            )
            assert isinstance(self._disk()._build_cython_closure(), MyClosureType)

        def test_all_fields_finite(self):
            """Every field in result.data is finite and non-NaN."""
            data = self._solve_result().data
            for name, arr in data.items():
                assert np.all(np.isfinite(arr.value)), (
                    f"Field '{name}' contains non-finite values"
                )

        def test_from_spec_dict_preserves_my_param(self):
            """Spec-dict round-trip preserves my_param."""
            disk = self._disk()
            loaded = MyDisk.from_spec_dict(disk.to_spec_dict())
            assert np.isclose(
                loaded._context_parameters["my_param"],
                disk._context_parameters["my_param"],
            )

Run the tests with:

.. code-block:: bash

    pytest tests/test_dynamics/test_my_closure.py -v

or run the entire dynamics suite:

.. code-block:: bash

    pytest tests/test_dynamics/ -v

----

Understanding ``BaseTestOneZoneDisk``
--------------------------------------

:class:`BaseTestOneZoneDisk` runs more than 60 inherited test methods organized into
the following categories.  All of these run automatically for any subclass — you do not
need to implement them.

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Category
      - What is tested
    * - Metaclass validation
      - Malformed declarations raise descriptive ``TypeError`` at class-definition time.
    * - Parameter processing
      - ``process_runtime_parameters`` returns correct keys and finite values;
        ``process_initial_conditions`` returns an ndarray of the right shape.
    * - Cython integration
      - ``_build_cython_closure`` returns a ready object; parameter binding succeeds.
    * - ODE solve
      - ``solve()`` completes; returns a ``OneZoneAccretionResult``; disk mass
        decreases (viscous drain); initial mass matches declaration.
    * - Result properties
      - All ``RESULT_FIELDS`` present and finite; unit checks; shape checks.
    * - HDF5 round-trip
      - ``to_hdf5`` / ``from_hdf5`` round-trips preserve time, arrays, parameters,
        and model type.
    * - Serialization
      - ``to_spec_dict`` / ``from_spec_dict`` round-trips; JSON-serializable spec.
    * - Dunder methods
      - ``__repr__``, ``__str__``, ``__eq__``, ``__hash__``.
    * - Result field access
      - ``result[key]``, ``get_field``, ``has_field``, ``add_field``, ``remove_field``.
    * - Plotting utilities
      - ``plot_state_variables``, ``plot_field``, ``plot_all_fields`` run without error.

Override any inherited test in your subclass if the default assertion is too tight
for your closure's physics (e.g. if ``M_D`` grows at early times due to strong
fallback supply, override ``test_M_D_decreases``).

----

Checklist
----------

.. list-table::
    :header-rows: 1

    * - Step
      - Action
      - Required
    * - 1
      - Write ``.pyx`` Cython extension with ``_closure_fn``, ``_derivative_fn``,
        ``_writer_fn``, and extension type
      - ✓ Always
    * - 2
      - Declare ``CONTEXT_PARAMETERS``, ``RUNTIME_PARAMETERS``,
        ``INITIAL_CONDITIONS``, ``RESULT_FIELDS``, ``CYTHON_FIELD_MAP``
      - ✓ Always
    * - 3
      - Implement ``_build_cython_closure``, ``_pack_cython_parameters``,
        ``to_spec_dict``, ``from_spec_dict``
      - ✓ Always
    * - 4
      - Export from ``core.py`` and ``one_zone/__init__.py``
      - ✓ Always
    * - 5
      - Add ``.pyx`` to build system; run ``pip install -e ".[dev]"``
      - ✓ Always
    * - 6
      - Inherit ``BaseTestOneZoneDisk``; add closure-specific tests
      - ✓ Always
    * - 7
      - Add backward-compatible aliases if needed
      - Optional
    * - 8
      - Document in the user guide and theory pages
      - Recommended

.. footbibliography::
