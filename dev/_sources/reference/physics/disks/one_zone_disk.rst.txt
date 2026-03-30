.. _one_zone_disk:

=========================================
One-Zone Accretion Disk Models
=========================================

The one-zone disk module provides **time-dependent accretion disk evolution** in a
compact, inference-friendly package.  The disk is reduced to a two-component ODE
system (mass :math:`M_D` and angular momentum :math:`J_D`), integrated by a compiled
Cython explicit-Euler kernel that evaluates all thermodynamic and structural
quantities at every step.

This document covers everything a user needs to run simulations with existing models
and everything a developer needs to add new thermodynamic closures.  For the
underlying physics see :ref:`one_zone_disk_theory`.

.. contents::
    :local:
    :depth: 2

----

Quick Start
-----------

The minimal workflow is four steps: instantiate a disk model, call
:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.solve`,
and read the result.

.. code-block:: python

    from astropy import constants as const
    from astropy import units as u
    from triceratops.dynamics.accretion.one_zone import gP_esDisk

    # 1. Create the disk model.
    disk = gP_esDisk(mu=0.62)

    # 2. Solve the evolution.
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

    # 3. Access the results.
    data = result.data
    print(data["T_c"][[0, -1]])     # central temperature at start and end
    print(data["R_D"][[0, -1]].to(u.cm))   # outer radius in cm

.. hint::

    All parameter values and time-span endpoints accept either plain floats
    (in the base CGS unit for that parameter) or
    :class:`~astropy.units.Quantity` objects — unit conversion is applied
    automatically by the solver.

----

Available Models
----------------

.. list-table::
    :header-rows: 1
    :widths: 30 25 25 20

    * - Class
      - Pressure
      - Opacity
      - Temperature solve
    * - :class:`~triceratops.dynamics.accretion.one_zone.core.gP_esDisk`
      - Gas only
      - Electron scattering
      - Analytic (:math:`T_c \propto Q_0^{1/3}`)
    * - :class:`~triceratops.dynamics.accretion.one_zone.core.igP_esDisk`
      - Gas + Radiation
      - Electron scattering
      - Implicit (Brent's method)
    * - :class:`~triceratops.dynamics.accretion.one_zone.core.gP_es_fbDisk`
      - Gas only + fallback supply
      - Electron scattering
      - Analytic (:math:`T_c \propto Q_0^{1/3}`)
    * - :class:`~triceratops.dynamics.accretion.one_zone.core.igP_es_fbDisk`
      - Gas + Radiation + fallback supply
      - Electron scattering
      - Implicit (Brent's method)
    * - :class:`~triceratops.dynamics.accretion.one_zone.core.igP_es_advDisk`
      - Gas + Radiation + advection
      - Electron scattering
      - Implicit (Brent's method)
    * - :class:`~triceratops.dynamics.accretion.one_zone.core.igP_es_adv_fbDisk`
      - Gas + Radiation + advection + fallback
      - Electron scattering
      - Implicit (Brent's method)

All models are importable from the package top-level:

.. code-block:: python

    from triceratops.dynamics.accretion.one_zone import (
        gP_esDisk,
        igP_esDisk,
        gP_es_fbDisk,
        igP_es_fbDisk,
        igP_es_advDisk,
        igP_es_adv_fbDisk,
    )

----

Creating a Disk Model
---------------------

Each disk class is instantiated with **context parameters** — physical constants
that are fixed for the lifetime of a model instance and do not change between
solves.

.. tab-set::

    .. tab-item:: gP_esDisk

        .. list-table::
            :header-rows: 1

            * - Parameter
              - Description
              - Default
            * - ``mu``
              - Mean molecular weight :math:`\mu` (dimensionless).  Appropriate
                values are :math:`\approx 0.6` for a fully-ionised solar-composition
                plasma or :math:`\approx 0.5` for a pure hydrogen plasma.
              - ``0.6``

        .. code-block:: python

            disk = gP_esDisk(mu=0.62)

    .. tab-item:: igP_esDisk

        .. list-table::
            :header-rows: 1

            * - Parameter
              - Description
              - Default
            * - ``mu``
              - Mean molecular weight :math:`\mu` (dimensionless).
              - ``0.6``

        .. code-block:: python

            disk = igP_esDisk(mu=0.62)

    .. tab-item:: igP_es_advDisk

        .. list-table::
            :header-rows: 1

            * - Parameter
              - Description
              - Default
            * - ``mu``
              - Mean molecular weight :math:`\mu` (dimensionless).
              - ``0.6``
            * - ``xi``
              - Entropy gradient parameter (dimensionless, :math:`>0`).  Controls
                the advective fraction :math:`q_{\rm adv}/q_{\rm visc} \propto
                \xi\,\alpha\,(H/R)^2`.  Setting :math:`\xi \to 0` recovers
                :class:`~triceratops.dynamics.accretion.one_zone.core.igP_esDisk`.
              - ``0.5``

        .. code-block:: python

            disk = igP_es_advDisk(mu=0.62, xi=0.5)

    .. tab-item:: igP_es_adv_fbDisk

        .. list-table::
            :header-rows: 1

            * - Parameter
              - Description
              - Default
            * - ``mu``
              - Mean molecular weight :math:`\mu` (dimensionless).
              - ``0.6``
            * - ``xi``
              - Entropy gradient parameter (dimensionless, :math:`>0`).
              - ``0.5``

        Requires ``M_fb_0`` and ``t_fb`` as runtime parameters in addition to the
        standard ``M_BH``, ``R_in``, ``alpha``.

        .. code-block:: python

            disk = igP_es_adv_fbDisk(mu=0.62, xi=0.5)

The Metzger+08 geometry constants :math:`A = 1.62`, :math:`B = 1.33`, and
:math:`F_0 = 1.6` are compiled into the Cython layer and are accessible as
read-only class attributes ``disk._A``, ``disk._B``, and ``disk._F``.

----

Setting Up Initial Conditions
------------------------------

The solver requires the initial **disk mass** :math:`M_{D,0}` and **angular
momentum** :math:`J_{D,0}`.  The simplest approach is to set them manually:

.. code-block:: python

    initial_conditions = {
        "M_D_0": 0.1 * const.M_sun,
        "J_D_0": 1.5e49 * u.g * u.cm**2 / u.s,
    }

In many practical situations it is more natural to specify the disk outer radius
:math:`R_{D,0}` rather than the angular momentum.  The helper method
:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.generate_initial_conditions`
uses the Metzger+08 kinematic constraint

.. math::

    J_D = \xi\, M_D\, \sqrt{G M_{\rm BH} R_D}, \qquad \xi \equiv B/A,

to infer the missing variable from any two of :math:`M_{D,0}`, :math:`J_{D,0}`,
and :math:`R_{D,0}`.

.. code-block:: python

    ic = disk.generate_initial_conditions(
        M_BH=3.0 * const.M_sun,
        M_D_0=0.1 * const.M_sun,
        R_D_0=3.0e13 * u.cm,     # supply R_D_0 instead of J_D_0
    )
    # ic = {"M_D_0": ..., "J_D_0": ...}

    result = disk.solve(
        initial_conditions=ic,
        runtime_parameters={"M_BH": 3.0 * const.M_sun, "R_in": 3.0e6 * u.cm, "alpha": 0.1},
        t_span=(1.0e6 * u.s, 5.0e9 * u.s),
    )

Exactly two of the three disk state variables must be provided.

----

Running the Solver
------------------

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.solve`
is the primary entry point.

.. code-block:: python

    result = disk.solve(
        initial_conditions=ic,
        runtime_parameters=run_params,
        t_span=(1.0e6 * u.s, 5.0e9 * u.s),
        max_steps=100_000,
        fail_fast=True,
    )

**Runtime parameters** are the per-solve physics inputs:

.. list-table::
    :header-rows: 1

    * - Parameter
      - Description
      - Base unit
    * - ``M_BH``
      - Black hole mass.
      - g
    * - ``R_in``
      - Inner disk truncation radius (e.g. ISCO).
      - cm
    * - ``alpha``
      - Shakura--Sunyaev viscosity parameter (dimensionless).
      - —

**Solver options**:

.. list-table::
    :header-rows: 1

    * - Option
      - Type
      - Default
      - Description
    * - ``max_steps``
      - int
      - ``100_000``
      - Hard limit on integration steps.  The solver terminates early if
        ``t_end`` is reached first.
    * - ``fail_fast``
      - bool
      - ``True``
      - If ``True``, re-raise the integrator ``RuntimeError`` on failure.
        If ``False``, issue a warning and return the (possibly empty) result.

Integrator Error Codes
^^^^^^^^^^^^^^^^^^^^^^

If the Cython integrator encounters an unrecoverable failure it raises a
:exc:`RuntimeError` with one of the following messages:

.. list-table::
    :header-rows: 1

    * - Code
      - Name
      - Meaning
    * - ``-1``
      - ``CLOSURE / FUNC_ERROR``
      - The root-finding residual returned NaN or Inf (unphysical disk state).
    * - ``-2``
      - ``CLOSURE / EXPAND_FAIL``
      - Bracket expansion could not find a sign change — no thermal equilibrium
        exists for this set of disk parameters.
    * - ``-3``
      - ``CLOSURE / NO_BRACKET``
      - Both bracket endpoints had the same sign after expansion.
    * - ``-4``
      - ``CLOSURE / MAX_ITER``
      - Brent's method did not converge within the iteration limit.
    * - ``-10``
      - ``DERIVATIVE_FAIL``
      - The ODE derivative function returned a non-zero status.
    * - ``-20``
      - ``WRITER_FAIL``
      - The result-writer function returned a non-zero status.

.. hint::

    ``EXPAND_FAIL`` typically indicates extreme parameters (very high surface
    density, very high :math:`\alpha`) where the energy balance has no physical
    equilibrium.  Reducing :math:`\alpha` or the initial disk mass usually resolves
    the issue.

----

Accessing Results
-----------------

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.solve`
returns a
:class:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult`
dataclass.

The ``data`` Property
^^^^^^^^^^^^^^^^^^^^^

The primary interface to all output fields is the
:attr:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.data`
property, which returns a dictionary of unit-bearing
:class:`~astropy.units.Quantity` arrays indexed by field name:

.. code-block:: python

    data = result.data

    print(data["T_c"])          # central temperature, shape (N,)  [K]
    print(data["R_D"].to(u.au)) # outer disk radius in AU
    print(data["mdot"])         # accretion rate, shape (N,)  [g/s]

Subscript notation is also supported:

.. code-block:: python

    T_c = result["T_c"]   # equivalent to result.data["T_c"]

Field Reference
^^^^^^^^^^^^^^^

The following fields are available in ``result.data`` for
:class:`~triceratops.dynamics.accretion.one_zone.core.gP_esDisk`,
:class:`~triceratops.dynamics.accretion.one_zone.core.igP_esDisk`,
and all fallback variants.  :class:`~triceratops.dynamics.accretion.one_zone.core.igP_es_advDisk`
and :class:`~triceratops.dynamics.accretion.one_zone.core.igP_es_adv_fbDisk` additionally
expose ``Q_adv`` (advective cooling rate, erg cm⁻² s⁻¹):

.. list-table::
    :header-rows: 1

    * - Field name
      - Description
      - Units
    * - ``t``
      - Integration time
      - s
    * - ``M_D``
      - Disk mass
      - g
    * - ``J_D``
      - Disk angular momentum
      - g cm² s⁻¹
    * - ``R_D``
      - Disk outer radius
      - cm
    * - ``Sigma``
      - Area-averaged surface density
      - g cm⁻²
    * - ``Omega``
      - Orbital angular velocity at :math:`R_D`
      - rad s⁻¹
    * - ``T_eff``
      - Effective (surface) temperature
      - K
    * - ``T_c``
      - Midplane temperature
      - K
    * - ``tau``
      - Vertical optical depth
      - dimensionless
    * - ``c_s``
      - Isothermal sound speed
      - cm s⁻¹
    * - ``nu``
      - Kinematic viscosity
      - cm² s⁻¹
    * - ``t_visc``
      - Viscous timescale at :math:`R_D`
      - s
    * - ``Q_visc``
      - Viscous dissipation rate per unit area
      - erg cm⁻² s⁻¹
    * - ``mdot``
      - Mass accretion rate (positive = inflow)
      - g s⁻¹
    * - ``H``
      - Disk scale height
      - cm
    * - ``H_over_R``
      - Disk aspect ratio :math:`H/R_D`
      - dimensionless
    * - ``rho``
      - Midplane mass density
      - g cm⁻³

State-Variable Shortcuts
^^^^^^^^^^^^^^^^^^^^^^^^

Frequently needed state variables are also available as direct properties:

.. code-block:: python

    result.t          # integration times [s], shape (N,)
    result.t_quantity # integration times as a Quantity [s]
    result.M_D        # disk mass [g]
    result.M_D_solar  # disk mass [M_sun]
    result.J_D        # angular momentum [g cm^2 / s]
    result.n_steps    # number of stored timesteps N

Snapshots: Initial and Final State
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:attr:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.initial_state`
and
:attr:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.final_state`
return dictionaries containing every field evaluated at the first and last
timestep respectively:

.. code-block:: python

    t0 = result.initial_state
    tf = result.final_state

    print(f"Initial T_c  = {t0['T_c']:.3e}")
    print(f"Final M_D    = {tf['M_D'].to(u.Msun):.4f}")

----

Diagnostic Utilities
--------------------

Before committing to a long parameter-grid sweep it is useful to sanity-check the
initial conditions.

Single-Step Solve
^^^^^^^^^^^^^^^^^

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.solve_initial_state`
runs exactly one integration step and returns the corresponding
:class:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult`:

.. code-block:: python

    result0 = disk.solve_initial_state(
        initial_conditions=ic,
        runtime_parameters=run_params,
    )
    print(result0["T_c"])

Physical Self-Consistency Checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.test_initial_state`
runs a set of physical consistency checks on the initial disk state and returns a
report dictionary:

.. code-block:: python

    report = disk.test_initial_state(
        initial_conditions=ic,
        runtime_parameters=run_params,
    )
    for name, result in report.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {name}: {result['note']}")

The following checks are performed:

.. list-table::
    :header-rows: 1

    * - Check name
      - Condition
      - Diagnostic hint
    * - ``finite``
      - All output fields are finite
      - Indicates unphysical disk state (extreme parameters)
    * - ``T_c_positive``
      - :math:`T_c > 0`
      - Should always pass if ``finite`` passes
    * - ``tau_thick``
      - :math:`\tau > 1` (optically thick)
      - Required for the diffusion approximation to hold
    * - ``H_over_R``
      - :math:`H/R \ll 1` (geometrically thin)
      - Thin-disk assumption; values :math:`\gtrsim 0.3` indicate a fat disk
    * - ``R_D_gt_R_in``
      - :math:`R_D > R_{\rm in}`
      - The disk must extend outside the inner truncation radius

----

Time Interpolation
------------------

The explicit-Euler integrator outputs values at **adaptively spaced** timesteps
governed by the local viscous timescale.  To evaluate a field at an arbitrary time
use the interpolation utilities.

Getting a Spline Interpolator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.get_field_interpolator`
returns a :class:`~scipy.interpolate.InterpolatedUnivariateSpline` over the
integration time axis:

.. code-block:: python

    interp = result.get_field_interpolator("T_c")       # cubic spline (k=3)
    interp_lin = result.get_field_interpolator("mdot", k=1)  # linear spline

Evaluating at New Times
^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.interpolate_field`
handles unit conversion and reattaches the correct :class:`~astropy.units.Quantity`
units to the output:

.. code-block:: python

    t_eval = np.geomspace(1e6, 5e9, 500) * u.s
    T_c_interp = result.interpolate_field("T_c", t_eval)
    mdot_interp = result.interpolate_field("mdot", t_eval)

A pre-computed interpolator can be reused by passing it as the ``interpolator``
keyword, avoiding redundant spline construction in loops:

.. code-block:: python

    interp = result.get_field_interpolator("T_c")
    for t in t_grid:
        T = result.interpolate_field("T_c", t, interpolator=interp)

----

Parameter Grid Sweeps
---------------------

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.solve_parameter_grid`
evaluates the disk model over a full Cartesian product of parameter values:

.. code-block:: python

    results = disk.solve_parameter_grid(
        initial_conditions_grid={
            "M_D_0": [0.05 * const.M_sun, 0.1 * const.M_sun, 0.2 * const.M_sun],
        },
        runtime_parameters_grid={
            "M_BH": [3.0 * const.M_sun, 10.0 * const.M_sun],
            "R_in": [3.0e6 * u.cm],
            "alpha": [0.05, 0.1, 0.3],
        },
        t_span=(1.0e6 * u.s, 5.0e9 * u.s),
        max_steps=50_000,
        fail_fast=False,    # continue on failure so one bad run doesn't abort the grid
    )

The return value is a dictionary keyed by tuples of ``(parameter_name, value)``
pairs (sorted alphabetically) to :class:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult`
objects.  A :mod:`tqdm` progress bar is shown automatically when the project
configuration enables progress bars.

.. note::

    Each combination is run sequentially.  For large grids consider wrapping
    :meth:`solve` calls in a :class:`concurrent.futures.ProcessPoolExecutor`
    for parallelism.

----

Physical Observables
--------------------

The result object provides convenience methods for deriving common astrophysical
observables.

Accretion Luminosity
^^^^^^^^^^^^^^^^^^^^

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.compute_luminosity`
evaluates the gravitational accretion luminosity :math:`L = G M_{\rm BH} \dot{M} / R_{\rm in}`
at a specified time:

.. code-block:: python

    L = result.compute_luminosity(1.0e8 * u.s)
    print(L.to(u.Lsun))

An existing :math:`\dot{M}` spline interpolator can be reused with the
``interpolator`` keyword.

Spectral (Specific) Luminosity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.compute_specific_luminosity`
computes the disk spectral luminosity by integrating a multi-colour blackbody
spectrum over the disk face:

.. math::

    L_\nu = 4\pi^2 \int_{R_{\rm in}}^{R_D} B_\nu\!\left(T(R)\right)\, R\, dR,
    \qquad
    T(R) = T_{\rm eff}(R_D) \left(\frac{R_D}{R}\right)^{3/4}.

.. code-block:: python

    import astropy.units as u

    L_nu = result.compute_specific_luminosity(
        time=1.0e8 * u.s,
        frequency=3.0e14 * u.Hz,   # optical V-band
        n_radii=500,
    )
    print(L_nu.to(u.erg / (u.s * u.Hz)))

Disk Effective Temperature Profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.compute_thin_disk_effective_temperature`
evaluates the multi-colour temperature profile at a given radius and time:

.. math::

    T_{\rm eff}(R, t) = T_{\rm eff}(R_D, t)\,\left(\frac{R_D}{R}\right)^{3/4}.

.. code-block:: python

    radii = np.geomspace(3e6, 3e13, 200) * u.cm
    T_profile = result.compute_thin_disk_effective_temperature(
        radius=radii,
        time=1.0e8 * u.s,
    )

----

Persistence
-----------

HDF5
^^^^

Results can be saved to and loaded from HDF5 files:

.. code-block:: python

    # Save
    result.to_hdf5("disk_run.h5", overwrite=True)

    # Load
    from triceratops.dynamics.accretion.one_zone import OneZoneAccretionResult
    loaded = OneZoneAccretionResult.from_hdf5("disk_run.h5")

Astropy QTable
^^^^^^^^^^^^^^

.. code-block:: python

    tbl = result.to_table()
    tbl.write("disk_run.ecsv", overwrite=True)

Pandas DataFrame
^^^^^^^^^^^^^^^^^

.. code-block:: python

    df = result.to_dataframe()   # requires pandas
    df.to_csv("disk_run.csv", index=False)

.. note::

    :meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.to_dataframe`
    requires ``pandas``, which is not a mandatory dependency of Triceratops.  If
    not installed, a descriptive :exc:`ImportError` is raised directing you to
    ``pip install pandas``.

Serialisation Round-Trip
^^^^^^^^^^^^^^^^^^^^^^^^^

Disk model instances (not results) can be serialised to a plain dict and
reconstructed:

.. code-block:: python

    spec = disk.to_spec_dict()
    import json
    print(json.dumps(spec, indent=2))
    # {"target": "...core:gP_esDisk", "mu": 0.62}

    disk2 = gP_esDisk.from_spec_dict(spec)

----

API Reference
-------------

Disk Models
^^^^^^^^^^^

.. currentmodule:: triceratops.dynamics.accretion.one_zone

.. autosummary::
    :toctree: generated/

    gP_esDisk
    igP_esDisk
    gP_es_fbDisk
    igP_es_fbDisk
    igP_es_advDisk
    igP_es_adv_fbDisk
    OneZoneAccretionDiskBase

Result Container
^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    OneZoneAccretionResult

----

Developer Guide
---------------

For step-by-step instructions on adding a new thermodynamic closure — including the
Cython extension type, the Python model class declaration, parameter packing, and the
test conventions — see the dedicated :ref:`one_zone_disk_dev`.

.. footbibliography::
