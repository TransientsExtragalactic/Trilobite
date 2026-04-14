.. _one_zone_disk:

=========================================
One-Zone Accretion Disk Models
=========================================

The one-zone disk module provides **time-dependent accretion disk evolution** in a
compact, inference-friendly package.  The disk is reduced to a two-component ODE
system (mass :math:`M_D` and angular momentum :math:`J_D`), integrated by a compiled
Cython explicit-Euler kernel that evaluates all thermodynamic and structural
quantities at every step.

.. note::

    This document covers everything a user needs to run simulations with existing models
    and everything a developer needs to add new thermodynamic closures.  For the
    underlying physics see :ref:`one_zone_disk_theory`.

.. contents::
    :local:
    :depth: 2

----

Quick Start
-----------

To get started with the one-zone accretion disk module, it is only necessary to follow a couple of simple
steps:

- Choose a model class from the list below and instantiate it with the desired context parameters.
- Call the :meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.solve` method
  with the initial conditions, runtime parameters, and solver options.
- Access the results through the returned :class:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult`
  object.

Here is the minimal workflow for a compact accretion disk around a stellar-mass black hole:

.. plot::
    :include-source:

    import matplotlib.pyplot as plt
    from astropy import constants as const
    from astropy import units as u
    from triceratops.dynamics.accretion.one_zone import GasPressureDisk
    from triceratops.utils.plot_utils import set_plot_style

    # 1. Setup Environment
    set_plot_style()

    # 2. Configure Disk Parameters
    M_BH = 1e6 * const.M_sun
    M_D_0 = 3 * const.M_sun
    R_D_0 = 3.0e15 * u.cm

    disk = GasPressureDisk(mu=0.62)
    ic = disk.generate_initial_conditions(M_BH=M_BH, M_D_0=M_D_0, R_D_0=R_D_0)

    # 3. Solve the System
    result = disk.solve(
    initial_conditions=ic,
    runtime_parameters={
        "M_BH": M_BH,
        "R_in": 3.0e6 * u.cm,
        "alpha": 0.1
    },
    t_span=(0, 1e7 * u.yr),
    max_steps=100_000,
    )

    # 4. Extract Data
    data = result.data
    t_yr = data["t"].to(u.yr).value
    m_disk = data["M_D"].to(u.M_sun).value

    # 5. Visualization
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogx(t_yr, m_disk, label=r"$M_D$ (Disk Mass)", lw=2)

    ax.set_xlabel("Time [yr]")
    ax.set_ylabel(r"Disk Mass [$M_\odot$]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

.. hint::

    All parameter values and time-span endpoints accept either plain floats
    (in the base CGS unit for that parameter) or
    :class:`~astropy.units.Quantity` objects. Unit conversion is applied
    automatically by the solver.

There are many **different disk models** to choose from, each covering a different
physical regime.  The list of available models is as follows:

.. list-table::
    :header-rows: 1
    :widths: 30 25 25 20

    * - Class
      - Pressure
      - Opacity
      - Temperature solve
    * - :class:`~triceratops.dynamics.accretion.one_zone.core.GasPressureDisk`
      - Gas only
      - Electron scattering
      - Analytic (:math:`T_c \propto Q_0^{1/3}`)
    * - :class:`~triceratops.dynamics.accretion.one_zone.core.FullPressureDisk`
      - Gas + Radiation
      - Electron scattering
      - Implicit (Brent's method)
    * - :class:`~triceratops.dynamics.accretion.one_zone.core.GasPressureDisk` (``fallback=True``)
      - Gas only + fallback supply
      - Electron scattering
      - Analytic (:math:`T_c \propto Q_0^{1/3}`)
    * - :class:`~triceratops.dynamics.accretion.one_zone.core.FullPressureDisk` (``fallback=True``)
      - Gas + Radiation + fallback supply
      - Electron scattering
      - Implicit (Brent's method)
    * - :class:`~triceratops.dynamics.accretion.one_zone.core.AdvectiveDisk`
      - Gas + Radiation + advection
      - Electron scattering
      - Implicit (Brent's method)
    * - :class:`~triceratops.dynamics.accretion.one_zone.core.AdvectiveDisk` (``fallback=True``)
      - Gas + Radiation + advection + fallback
      - Electron scattering
      - Implicit (Brent's method)

All models are importable from the package top-level:

.. code-block:: python

    from triceratops.dynamics.accretion.one_zone import (
        GasPressureDisk,
        FullPressureDisk,
        AdvectiveDisk,
    )

.. note::

    The legacy aliases ``gP_esDisk``, ``igP_esDisk``, ``igP_es_advDisk``,
    ``gP_es_fbDisk``, ``igP_es_fbDisk``, and ``igP_es_adv_fbDisk`` remain
    importable for backward compatibility but refer to the same classes.

In the sections below, we'll describe the various methods and available features for each of the models.

----

Creating a Disk Model
---------------------

Each disk class is instantiated with **context parameters**: physical constants
that are fixed for the lifetime of a model instance and do not change between
solves.

.. hint::

    Think of the **context parameters** as the physics configuration. This is where you'll find flags
    for enabling / disabling various different physical effects, setting up an equation of state, or
    defining the opacity law.

The various models have several different sets of available context parameters:

.. tab-set::

    .. tab-item:: GasPressureDisk

        *API Doc*: :class:`~triceratops.dynamics.accretion.one_zone.core.GasPressureDisk`

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

            disk = GasPressureDisk(mu=0.62)

    .. tab-item:: FullPressureDisk

        *API Doc*: :class:`~triceratops.dynamics.accretion.one_zone.core.FullPressureDisk`

        .. list-table::
            :header-rows: 1

            * - Parameter
              - Description
              - Default
            * - ``mu``
              - Mean molecular weight :math:`\mu` (dimensionless).
              - ``0.6``

        .. code-block:: python

            disk = FullPressureDisk(mu=0.62)

    .. tab-item:: AdvectiveDisk

        *API Doc*: :class:`~triceratops.dynamics.accretion.one_zone.core.AdvectiveDisk`

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
                :class:`~triceratops.dynamics.accretion.one_zone.core.FullPressureDisk`. See
                :footcite:t:`wataraiNewAnalyticalFormulae2006`.
              - ``0.5``

        .. code-block:: python

            disk = AdvectiveDisk(mu=0.62, xi=0.5)

    .. tab-item:: AdvectiveDisk (fallback)

        *API Doc*: :class:`~triceratops.dynamics.accretion.one_zone.core.AdvectiveDisk`

        .. list-table::
            :header-rows: 1

            * - Parameter
              - Description
              - Default
            * - ``mu``
              - Mean molecular weight :math:`\mu` (dimensionless).
              - ``0.6``
            * - ``xi``
              - Entropy gradient parameter (dimensionless, :math:`>0`). See
                :footcite:t:`wataraiNewAnalyticalFormulae2006`.
              - ``0.5``

        Requires ``M_fb_0`` and ``t_fb`` as runtime parameters in addition to the
        standard ``M_BH``, ``R_in``, ``alpha``.

        .. code-block:: python

            disk = AdvectiveDisk(mu=0.62, xi=0.5, fallback=True)

----

Running a Simulation
--------------------

Once you've initialised a disk object, the next step is to set up the initial
conditions and runtime parameters, then call the solver.

Initial Conditions
^^^^^^^^^^^^^^^^^^

Every disk model requires, at minimum, the initial disk mass :math:`M_{D,0}` and
angular momentum :math:`J_{D,0}`.  The simplest approach is to supply them directly:

.. code-block:: python

    initial_conditions = {
        "M_D_0": 0.1 * const.M_sun,
        "J_D_0": 1.5e49 * u.g * u.cm**2 / u.s,
    }

In many cases the angular momentum is not the natural parameter to specify —
it is more intuitive to think in terms of an initial disk outer radius
:math:`R_{D,0}`.  The helper method
:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.generate_initial_conditions`
accepts any two of the three state variables and solves for the third.

.. dropdown:: Example — specifying initial radius instead of angular momentum

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

Running the Solver
^^^^^^^^^^^^^^^^^^

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.solve`
is the primary entry point.  It takes the initial conditions, per-solve physics
inputs (**runtime parameters**), and a time span, then hands control to the
compiled Cython integrator:

.. code-block:: python

    result = disk.solve(
        initial_conditions=ic,
        runtime_parameters=run_params,
        t_span=(1.0e6 * u.s, 5.0e9 * u.s),
        max_steps=100_000,
        fail_fast=True,
    )

Runtime Parameters
*******************

Runtime parameters vary between models, but most models require at least:

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

Solver Options
****************

Like most other numerical integrators, the one-zone solver accepts some options that control the integration
behaviour and error handling.  The most commonly used options are:

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

.. warning::

    If the Cython integrator encounters an unrecoverable failure it raises a
    :exc:`RuntimeError`.  The error code (embedded in the message) identifies
    the cause:

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

    ``EXPAND_FAIL`` typically means the energy balance has no physical equilibrium
    for the chosen parameters.  Reducing :math:`\alpha` or the initial disk mass
    usually resolves the issue.

----

Accessing Results
-----------------

Once you have performed a simulation, the result returned by
:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.solve`
is an
:class:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult`
object.  This section covers everything you can do with it: reading raw field
data, time interpolation, and computing derived physical observables.

The ``data`` Dictionary
^^^^^^^^^^^^^^^^^^^^^^^

The primary interface is the
:attr:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.data`
property, which returns a dictionary of unit-bearing
:class:`~astropy.units.Quantity` arrays — one value per integration step:

.. code-block:: python

    data = result.data

    print(data["T_c"])           # midplane temperature, shape (N,)  [K]
    print(data["R_D"].to(u.au))  # outer disk radius in AU
    print(data["mdot"])          # accretion rate, shape (N,)  [g/s]

Subscript notation on the result object itself is also supported as a shortcut:

.. code-block:: python

    T_c = result["T_c"]    # equivalent to result.data["T_c"]
    R_D = result["R_D"]

The following fields are available for all models.
:class:`~triceratops.dynamics.accretion.one_zone.core.AdvectiveDisk` additionally
exposes ``Q_adv`` (advective cooling rate, erg cm⁻² s⁻¹):

.. hint::

    Each model class also lists its available fields in the API documentation.

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

Convenience Properties and Snapshots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Frequently needed state variables are also exposed as direct properties of the
result object:

.. code-block:: python

    result.t          # integration times [s], shape (N,)
    result.t_quantity # integration times as a Quantity [s]
    result.M_D        # disk mass [g]
    result.M_D_solar  # disk mass [M_sun]
    result.J_D        # angular momentum [g cm^2 / s]
    result.n_steps    # number of stored timesteps N

For a quick sanity check, the
:attr:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.initial_state`
and
:attr:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.final_state`
properties return dictionaries of every field evaluated at the first and last
timestep:

.. code-block:: python

    t0 = result.initial_state
    tf = result.final_state

    print(f"Initial T_c  = {t0['T_c']:.3e}")
    print(f"Final M_D    = {tf['M_D'].to(u.Msun):.4f}")

Time Interpolation
^^^^^^^^^^^^^^^^^^

The explicit-Euler integrator outputs values at **adaptively spaced** timesteps
governed by the local viscous timescale — the step size shrinks when the disk is
evolving rapidly and grows when it is in a quasi-steady state.  To evaluate any
field at a uniform or user-specified time grid, use the interpolation utilities.

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.get_field_interpolator`
builds a :class:`~scipy.interpolate.InterpolatedUnivariateSpline` over the
integration time axis:

.. code-block:: python

    interp_Tc   = result.get_field_interpolator("T_c")       # cubic spline (k=3)
    interp_mdot = result.get_field_interpolator("mdot", k=1)  # linear spline

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.interpolate_field`
evaluates the spline at new times and re-attaches the correct astropy units:

.. code-block:: python

    t_eval      = np.geomspace(1e6, 5e9, 500) * u.s
    T_c_interp  = result.interpolate_field("T_c", t_eval)
    mdot_interp = result.interpolate_field("mdot", t_eval)

When evaluating many fields at the same set of times, pass a pre-built
interpolator to avoid redundant spline construction:

.. code-block:: python

    interp = result.get_field_interpolator("T_c")
    for t in t_grid:
        T = result.interpolate_field("T_c", t, interpolator=interp)

Derived Physical Observables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the standard output fields, several additional quantities may be derived as needed directly
from the results object. These are useful for computing observable features or
coupling to other physical processes without having to re-run the disk evolution.

Accretion Luminosity
*********************

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.compute_luminosity`
evaluates the gravitational accretion luminosity
:math:`L = G M_{\rm BH} \dot{M} / R_{\rm in}` at a specified time, using spline
interpolation on the stored :math:`\dot{M}` trace:

.. code-block:: python

    L = result.compute_luminosity(1.0e8 * u.s)
    print(L.to(u.Lsun))

    # Reuse an existing interpolator for speed in loops:
    interp = result.get_field_interpolator("mdot")
    L_arr = [result.compute_luminosity(t, interpolator=interp) for t in t_grid]

Spectral (Specific) Luminosity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.compute_specific_luminosity`
approximates the disk as a multi-colour blackbody and integrates the Planck
function over the disk face using a power-law temperature profile anchored to the
one-zone effective temperature:

.. math::

    L_\nu = 4\pi^2 \int_{R_{\rm in}}^{R_D} B_\nu\!\left(T(R)\right)\, R\, dR,
    \qquad
    T(R) = T_{\rm eff}(R_D) \left(\frac{R_D}{R}\right)^{3/4}.

.. code-block:: python

    L_nu = result.compute_specific_luminosity(
        time=1.0e8 * u.s,
        frequency=3.0e14 * u.Hz,   # optical V-band
        n_radii=500,
    )
    print(L_nu.to(u.erg / (u.s * u.Hz)))

.. seealso::

    :class:`~triceratops.dynamics.accretion.AlphaDisk` provides a full radial
    structure and multi-colour SED using the Shakura-Sunyaev scalings, which is
    more accurate for spectral fitting than the single-zone approximation above.
    See :ref:`thin_disk` for details.

Disk Effective Temperature Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionResult.compute_thin_disk_effective_temperature`
evaluates the radial temperature profile at a given time, using the same
power-law approximation:

.. math::

    T_{\rm eff}(R, t) = T_{\rm eff}(R_D, t)\,\left(\frac{R_D}{R}\right)^{3/4}.

.. code-block:: python

    radii = np.geomspace(3e6, 3e13, 200) * u.cm
    T_profile = result.compute_thin_disk_effective_temperature(
        radius=radii,
        time=1.0e8 * u.s,
    )



----

Diagnostic Utilities
--------------------

Before committing to a long run or a large parameter sweep, it is worth
confirming that the initial conditions correspond to a physically self-consistent
disk state.

Single-Step Solve
^^^^^^^^^^^^^^^^^

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.solve_initial_state`
runs exactly one integration step and returns the full result object.  This is
a fast way to inspect the initial thermodynamic state before launching the full
integration:

.. code-block:: python

    result0 = disk.solve_initial_state(
        initial_conditions=ic,
        runtime_parameters=run_params,
    )
    print(result0["T_c"])     # midplane temperature at t=0
    print(result0["H_over_R"])  # check the thin-disk assumption

Physical Self-Consistency Checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.test_initial_state`
runs a battery of physical consistency checks on the initial disk state and
returns a report dictionary.  Each entry records whether the check passed and
a diagnostic note:

.. code-block:: python

    report = disk.test_initial_state(
        initial_conditions=ic,
        runtime_parameters=run_params,
    )
    for name, result in report.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {name}: {result['note']}")

The checks performed are:

.. list-table::
    :header-rows: 1

    * - Check name
      - Condition
      - What a failure means
    * - ``finite``
      - All output fields are finite
      - Unphysical disk state — try less extreme parameters
    * - ``T_c_positive``
      - :math:`T_c > 0`
      - Should always pass if ``finite`` passes
    * - ``tau_thick``
      - :math:`\tau > 1` (optically thick)
      - Diffusion approximation breaks down at low :math:`\tau`
    * - ``H_over_R``
      - :math:`H/R \ll 1` (geometrically thin)
      - Values :math:`\gtrsim 0.3` indicate a geometrically thick, sub-Keplerian disk
    * - ``R_D_gt_R_in``
      - :math:`R_D > R_{\rm in}`
      - The disk must extend beyond the inner truncation radius

----

Parameter Grid Sweeps
---------------------

:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.solve_parameter_grid`
runs the disk model over the full Cartesian product of a set of parameter values,
returning a dictionary of results keyed by parameter tuples:

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

    # Access a specific run: keys are sorted (name, value) tuples
    key = (("M_BH", 3.0 * const.M_sun), ("M_D_0", 0.1 * const.M_sun),
           ("R_in", 3.0e6 * u.cm), ("alpha", 0.1))
    r = results[key]

A :mod:`tqdm` progress bar is shown automatically when progress bars are enabled
in the project configuration.

.. note::

    Each combination is run sequentially in the current process.  For large grids,
    wrap individual :meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase.solve` calls in a
    :class:`concurrent.futures.ProcessPoolExecutor` for parallelism. see the
    :mod:`triceratops.parallel` module for convenience wrappers.

----

Persistence
-----------

Results can be saved and reloaded in several formats.  HDF5 is recommended for
large outputs: it preserves units metadata and loads much faster than plain text.

.. tab-set::

    .. tab-item:: HDF5

        .. code-block:: python

            # Save
            result.to_hdf5("disk_run.h5", overwrite=True)

            # Load
            from triceratops.dynamics.accretion.one_zone import OneZoneAccretionResult
            loaded = OneZoneAccretionResult.from_hdf5("disk_run.h5")

    .. tab-item:: Astropy QTable

        .. code-block:: python

            tbl = result.to_table()
            tbl.write("disk_run.ecsv", overwrite=True)

    .. tab-item:: Pandas DataFrame

        .. code-block:: python

            df = result.to_dataframe()   # requires pandas
            df.to_csv("disk_run.csv", index=False)

        .. note::

            ``to_dataframe()`` requires ``pandas``, which is not a mandatory
            dependency of Triceratops.  If not installed, a descriptive
            :exc:`ImportError` is raised.

Disk model instances (not results) can also be serialised to a plain dict and
reconstructed, which is useful for logging run configurations or building
reproducible pipelines:

.. code-block:: python

    spec  = disk.to_spec_dict()
    import json
    print(json.dumps(spec, indent=2))
    # {"target": "...core:GasPressureDisk", "mu": 0.62}

    disk2 = GasPressureDisk.from_spec_dict(spec)

----

API Reference
-------------

Disk Models
^^^^^^^^^^^

.. currentmodule:: triceratops.dynamics.accretion.one_zone

.. autosummary::
    :toctree: generated/

    GasPressureDisk
    FullPressureDisk
    AdvectiveDisk
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

References
-----------
.. footbibliography::
