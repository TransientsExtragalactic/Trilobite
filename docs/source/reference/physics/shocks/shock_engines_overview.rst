.. _shock_engines:
===========================================
Shock Engines
===========================================

The evolution of an astrophysical shock is the central dynamical ingredient in modeling
radio and X-ray transients. Whether one is describing the ejecta--CSM interaction of a
supernova :footcite:p:`chevalierSelfsimilarSolutionsInteraction1982`, the deceleration of
a gamma-ray burst afterglow, or the late-time blast wave of a tidal disruption event, the
shock radius :math:`R(t)` and velocity :math:`v(t)` set the stage for everything that
follows: they determine the post-shock conditions that feed into the synchrotron and
thermal emission models, and ultimately shape the light curves and SEDs that observers
measure.

Triceratops provides a unified **shock engine** interface that wraps the physics of shock
evolution and returns time-varying shock properties in a consistent format, regardless of
whether the underlying model is a closed-form self-similar solution, a numerical ODE
integrator, or a bespoke prescription.  The engines are intentionally **stateless** —
physical parameters are passed at call time rather than stored on the object — so that they
can be freely used in parameter sweeps and Bayesian inference pipelines without hidden state
or unexpected side effects.

.. hint::

    For users unfamiliar with the underlying shock physics, the theory guides provide
    self-contained derivations of each solution:

    - :ref:`chevalier_theory` — the Chevalier self-similar ejecta--CSM interaction
    - :ref:`sedov_taylor_theory` — the Sedov--Taylor point-explosion blast wave
    - :ref:`numeric_shocks_theory` — the thin-shell numerical ODE formulation

    :footcite:t:`ChevalierFranssonHandbook` is an excellent review of supernovae in
    circumstellar media; :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`
    is the primary reference for the self-similar solutions implemented here.

.. contents::
    :local:
    :depth: 2

----

Module Overview
---------------

In a *typical shock dynamics problem*, the user has a physical scenario — for example, a
supernova explosion into a stellar wind — and wants to predict how the shock evolves with
time.  The workflow generally proceeds in two steps:

1. **Parameterize the environment**: Use the normalization utilities (e.g.,
   :func:`~triceratops.dynamics.shocks.chevalier.normalize_supernova_ejecta` or
   :func:`~triceratops.dynamics.shocks.chevalier.compute_wind_csm_parameters`) to convert
   physical observables such as ejecta mass, explosion energy, or progenitor mass-loss rate
   into the normalization constants required by the shock model.

2. **Evolve the shock**: Call the engine's
   :meth:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine.compute_shock_properties`
   method — or simply call the engine as a function, since all engines are callable — with a
   time array to obtain the shock radius, velocity, and any additional post-shock
   thermodynamic quantities supported by that particular engine.

The result is a dictionary of :class:`~astropy.units.Quantity` arrays that can be fed
directly into the synchrotron or thermal emission models in :mod:`~triceratops.radiation`.

Module Structure
^^^^^^^^^^^^^^^^

The shock engine infrastructure spans two layers of :mod:`~triceratops.dynamics.shocks`:

.. list-table::
    :header-rows: 1
    :widths: 42 58

    * - Class
      - Responsibility
    * - :class:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine`
      - Abstract base class defining the two-method contract that all engines must satisfy.
        Users encounter this class only when building custom engines; see
        `Building a Custom Shock Engine`_ below.
    * - :class:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarShockEngine`
      - Self-similar ejecta--CSM interaction for arbitrary power-law density profiles
        :footcite:p:`chevalierSelfsimilarSolutionsInteraction1982`.
        See :ref:`chevalier_theory`.
    * - :class:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarWindShockEngine`
      - Specialization of the Chevalier engine for a steady stellar wind CSM
        (:math:`s = 2`). Accepts the mass-loss rate :math:`\dot{M}` and wind velocity
        :math:`v_w` directly, rather than a pre-computed normalization constant.
    * - :class:`~triceratops.dynamics.shocks.sedov_taylor.SedovTaylorShockEngine`
      - Point-explosion blast wave in a uniform ambient medium
        :footcite:p:`sedov1946propagation` :footcite:p:`taylor1950formation`.
        Returns post-shock density, pressure, and temperature in addition to kinematics.
        See :ref:`sedov_taylor_theory`.
    * - :class:`~triceratops.dynamics.shocks.numerical.NumericalThinShellShockEngine`
      - ODE-based thin-shell engine for *arbitrary* ejecta and CSM density profiles.
        Suitable when the self-similar approximation breaks down.
        See :ref:`numeric_shocks_theory`.

----

Quickstart
----------

The example below demonstrates the most common use case: evolving a Chevalier wind-CSM
shock and plotting the resulting radius and velocity.  The
:class:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarWindShockEngine` is the
simplest entry point — it requires only four physical quantities (explosion energy, ejecta
mass, wind mass-loss rate, and wind velocity) and handles all normalization internally.

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from triceratops.dynamics.shocks import ChevalierSelfSimilarWindShockEngine

    # Instantiate the engine.  No physical parameters are stored at construction time.
    engine = ChevalierSelfSimilarWindShockEngine()

    # ── Physical parameters ────────────────────────────────────────────────────
    E_ej   = 1e51 * u.erg                  # typical core-collapse SN explosion energy
    M_ej   = 10.0 * u.Msun                 # ejecta mass
    M_dot  = 1e-5 * u.Msun / u.yr          # progenitor wind mass-loss rate
    v_wind = 1000.0 * u.km / u.s           # progenitor wind velocity
    n      = 10.0                           # outer ejecta density power-law index

    # ── Time grid: 10 days to ~10 years ───────────────────────────────────────
    t = np.logspace(1, np.log10(3650), 300) * u.day

    # ── Evaluate the shock engine ──────────────────────────────────────────────
    # The engine is callable; engine(t, ...) is equivalent to
    # engine.compute_shock_properties(t, ...).
    props = engine(t, E_ej=E_ej, M_ej=M_ej, M_dot=M_dot, v_wind=v_wind, n=n)

    # ── Plot radius and velocity ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].loglog(
        t.to(u.day).value,
        props['radius'].to(u.cm).value / 1e16,
        color='steelblue', lw=2,
    )
    axes[0].set_xlabel('Time [days]')
    axes[0].set_ylabel(r'Shock radius [$10^{16}$ cm]')
    axes[0].set_title('Shock Radius')
    axes[0].grid(True, which='both', ls='--', alpha=0.4)

    axes[1].loglog(
        t.to(u.day).value,
        props['velocity'].to(u.km / u.s).value,
        color='darkorange', lw=2,
    )
    axes[1].set_xlabel('Time [days]')
    axes[1].set_ylabel(r'Shock velocity [km s$^{-1}$]')
    axes[1].set_title('Shock Velocity')
    axes[1].grid(True, which='both', ls='--', alpha=0.4)

    plt.suptitle(
        r'Chevalier wind-CSM shock: '
        r'$E_{\rm ej} = 10^{51}\ {\rm erg},\; M_{\rm ej} = 10\,M_\odot$',
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

----

Types of Shock Engines
-----------------------

Triceratops provides two broad categories of shock engine, each suited to a different level
of physical complexity and computational cost.

Self-Similar Shock Engines
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

    :ref:`self_similar_shocks_overview` — detailed API reference.
    :ref:`chevalier_theory` and :ref:`sedov_taylor_theory` — theoretical derivations.

Self-similar shock solutions arise when the governing equations admit a scale-free form.
The shock structure then evolves such that a single power-law in time describes both the
radius and velocity, converting what is nominally a partial differential equation into an
ordinary one and, in the most favorable cases, yielding a fully analytic result.  These
solutions are computationally cheap and carry intuitive physical scaling relations, making
them the preferred first choice for parameter inference.

Three self-similar engines are currently implemented:

.. list-table::
    :header-rows: 1
    :widths: 42 16 16 26

    * - Class
      - Geometry
      - Medium
      - Use when...
    * - :class:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarShockEngine`
      - Spherical
      - Power-law (:math:`\rho \propto r^{-s}`)
      - SN ejecta--CSM interaction with a general power-law CSM
    * - :class:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarWindShockEngine`
      - Spherical
      - Stellar wind (:math:`s = 2`)
      - SN ejecta in a steady wind; :math:`\dot{M}` and :math:`v_w` as direct inputs
    * - :class:`~triceratops.dynamics.shocks.sedov_taylor.SedovTaylorShockEngine`
      - Spherical
      - Uniform (:math:`\rho = \rho_0`)
      - Point explosion in a uniform medium; full post-shock thermodynamics returned

**Chevalier engines.**
Both Chevalier engines model the supernova ejecta as a broken power-law density profile
parameterized by the outer index :math:`n` and the inner index :math:`\delta`.  The helper
function :func:`~triceratops.dynamics.shocks.chevalier.normalize_supernova_ejecta` converts
the total kinetic energy :math:`E_{\rm ej}` and ejecta mass :math:`M_{\rm ej}` into the
normalization constant :math:`K_{\rm ej}` and transition velocity :math:`v_t` that enter
the self-similar solution. For wind-like CSM, the function
:func:`~triceratops.dynamics.shocks.chevalier.compute_wind_csm_parameters` converts the
mass-loss rate and wind velocity directly into the CSM normalization constant
:math:`K_{\rm CSM}`.

Under these assumptions, the shock radius evolves as

.. math::

    R(t) = \left(\frac{\zeta\, K_{\rm CSM}}{K_{\rm ej}}\right)^{\!\frac{1}{s-n}}
             t^{\frac{3-n}{s-n}},

where :math:`\zeta` is an order-unity factor computed by
:meth:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarShockEngine.compute_scale_parameter`
that depends only on :math:`n` and :math:`s`.  The shock velocity follows directly from
differentiation, :math:`v(t) = \dot{R}(t)`.

.. dropdown:: Example: Chevalier Wind Shock — Varying Ejecta Mass

    This example shows how the shock radius evolves for three different ejecta masses at
    fixed explosion energy, illustrating the well-known result that heavier ejecta decelerates
    the shock more slowly at early times but sweeps up more CSM mass at later times.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u
        from triceratops.dynamics.shocks import ChevalierSelfSimilarWindShockEngine

        engine = ChevalierSelfSimilarWindShockEngine()

        E_ej   = 1e51 * u.erg
        M_dot  = 1e-5 * u.Msun / u.yr
        v_wind = 1000.0 * u.km / u.s
        n      = 10.0

        t = np.logspace(1, np.log10(3650), 300) * u.day

        fig, ax = plt.subplots(figsize=(7, 4))

        for M_ej, color, label in [
            (3.0  * u.Msun, 'steelblue',  r'$M_{\rm ej} = 3\,M_\odot$'),
            (10.0 * u.Msun, 'darkorange', r'$M_{\rm ej} = 10\,M_\odot$'),
            (30.0 * u.Msun, 'firebrick',  r'$M_{\rm ej} = 30\,M_\odot$'),
        ]:
            props = engine(t, E_ej=E_ej, M_ej=M_ej, M_dot=M_dot, v_wind=v_wind, n=n)
            ax.loglog(
                t.to(u.day).value,
                props['radius'].to(u.cm).value / 1e16,
                color=color, lw=2, label=label,
            )

        ax.set_xlabel('Time [days]')
        ax.set_ylabel(r'Shock radius [$10^{16}$ cm]')
        ax.set_title(r'Chevalier wind shock, $E_{\rm ej} = 10^{51}$ erg')
        ax.legend()
        ax.grid(True, which='both', ls='--', alpha=0.4)
        plt.tight_layout()

**Sedov--Taylor engine.**
The :class:`~triceratops.dynamics.shocks.sedov_taylor.SedovTaylorShockEngine` is
appropriate for a *point explosion* in a uniform ambient medium — the classic scenario
first solved by :footcite:t:`sedov1946propagation` and :footcite:t:`taylor1950formation`.
In this limit,

.. math::

    R_s(t) = \beta(\gamma)\left(\frac{E\,t^2}{\rho_0}\right)^{\!1/5}, \qquad
    v_s(t) = \frac{2}{5}\,\frac{R_s(t)}{t},

where the normalization coefficient :math:`\beta(\gamma)` is computed once at
instantiation by :func:`~triceratops.dynamics.shocks.sedov_taylor.sedov_taylor_beta`
via numerical integration of the self-similar Sedov profiles.  Unlike the Chevalier
engines, the Sedov--Taylor engine also returns the immediate post-shock density, pressure,
temperature, and thermal energy density derived from the strong-shock Rankine--Hugoniot
conditions.

.. dropdown:: Example: Sedov--Taylor Blast — Kinematics and Post-Shock Temperature

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u
        from triceratops.dynamics.shocks import SedovTaylorShockEngine

        engine = SedovTaylorShockEngine(gamma=5.0 / 3.0)

        E     = 1e51 * u.erg
        rho_0 = 1.67e-24 * u.g / u.cm**3    # approximately one proton per cm^3
        t     = np.logspace(9, 13, 300) * u.s

        props = engine(t, E=E, rho_0=rho_0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].loglog(
            t.to(u.yr).value,
            props['radius'].to(u.pc).value,
            color='steelblue', lw=2,
        )
        axes[0].set_xlabel('Time [yr]')
        axes[0].set_ylabel('Shock radius [pc]')
        axes[0].set_title('Sedov--Taylor blast radius')
        axes[0].grid(True, which='both', ls='--', alpha=0.4)

        axes[1].loglog(
            t.to(u.yr).value,
            props['post_shock_temperature'].to(u.K).value,
            color='firebrick', lw=2,
        )
        axes[1].set_xlabel('Time [yr]')
        axes[1].set_ylabel('Post-shock temperature [K]')
        axes[1].set_title('Sedov--Taylor post-shock temperature')
        axes[1].grid(True, which='both', ls='--', alpha=0.4)

        plt.tight_layout()

Numerical Shock Engines
^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

    :ref:`numerical_shocks_overview` — API reference.
    :ref:`numeric_shocks_theory` — derivation of the thin-shell equations of motion.

When the self-similar approximation fails — for instance, when the density profiles cannot
be described by simple power laws, or when the shock transitions between dynamical regimes
during its evolution — a numerical shock engine is necessary.  The
:class:`~triceratops.dynamics.shocks.numerical.NumericalThinShellShockEngine` integrates
the thin-shell equations of motion using a standard ODE solver and can therefore handle
*any* ejecta or CSM density profile that can be expressed as a Python callable.

The model represents the shocked ejecta and swept-up CSM as a single thin shell.  Given
the shell mass :math:`M_{\rm sh}`, shell velocity :math:`v_{\rm sh}`, and shock radius
:math:`R_{\rm sh}`, momentum conservation requires

.. math::

    \frac{dR_{\rm sh}}{dt} = v_{\rm sh}, \qquad
    \frac{dM_{\rm sh}}{dt} = 4\pi R_{\rm sh}^2
        \Bigl[\rho_{\rm CSM}(R)\,v_{\rm sh}
              + t^{-3} G(v_{\rm ej})\,\Delta\Bigr],

where :math:`G(v)` is the time-independent ejecta kernel and :math:`\Delta = v_{\rm ej} - v_{\rm sh}`
is the relative velocity between the ejecta at the shock radius and the shell.  The
deceleration equation closes the system; see :ref:`numeric_shocks_theory` for the full
derivation.

Working with the numerical engine requires three steps:

1. **Construct density callables**: Build Python functions for the ejecta kernel
   :math:`G(v)` and the CSM density :math:`\rho_{\rm CSM}(r)` in CGS units.  The helper
   functions :func:`~triceratops.dynamics.shocks.chevalier.get_broken_power_law_ejecta_kernel_func`
   and :func:`~triceratops.dynamics.shocks.chevalier.get_wind_csm_density_func` generate the
   standard Chevalier-style profiles as callables, but any function with the correct signature
   works.
2. **Instantiate the engine**: Create a
   :class:`~triceratops.dynamics.shocks.numerical.NumericalThinShellShockEngine` with the
   two density callables.
3. **Evolve**: Call
   :meth:`~triceratops.dynamics.shocks.numerical.NumericalThinShellShockEngine.compute_shock_properties`
   with a time array to obtain the shock properties at each requested time.

.. note::

    Because the numerical engine integrates an ODE from initial conditions, the solution
    accuracy depends on the quality of the initial state provided at the first time step.
    For typical supernova scenarios, initializing from the corresponding Chevalier
    self-similar solution is recommended.  See :ref:`numerical_shocks_overview` for
    details.

----

Building a Custom Shock Engine
--------------------------------

Every shock engine in Triceratops inherits from the
:class:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine` abstract base class.
The contract is intentionally minimal — two methods — and is designed so that a custom
engine can be dropped into any model or inference pipeline that expects a
:class:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine`, without requiring
any changes to the surrounding infrastructure.

The ShockEngine Class
^^^^^^^^^^^^^^^^^^^^^^^

The abstract base class defines a two-method contract:

.. list-table::
    :header-rows: 1
    :widths: 42 58

    * - Method
      - Role
    * - :meth:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine.compute_shock_properties`
      - High-level, unit-aware entry point. Accepts :class:`~astropy.units.Quantity` inputs
        (or bare floats interpreted as CGS), validates arguments, and returns a dictionary
        of :class:`~astropy.units.Quantity` arrays. This is the method users call.
    * - :meth:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine._compute_shock_properties_cgs`
      - Low-level, pure-CGS computation. Accepts and returns bare floats or arrays with no
        unit overhead. Called internally by ``compute_shock_properties`` and available
        directly for performance-critical code paths such as tight inference loops.

All engines are also directly callable: ``engine(time, **parameters)`` is an alias for
``engine.compute_shock_properties(time, **parameters)``.

.. important::

    The :class:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine` is designed to
    be **stateless**.  The ``__init__`` method should only set up instance-level
    configuration — numerical tolerances, pre-cached coefficients, or similar — and must
    *never* store physical model parameters.  All physical parameters must be passed as
    arguments to ``compute_shock_properties`` at call time.

    This design ensures that a single engine instance can be reused across many sets of
    parameters in a sweep or MCMC chain without unintended cross-contamination between
    evaluations.

The example below shows the minimum viable implementation of a custom shock engine: a
simple free-expansion model in which the shock radius grows as :math:`R(t) = v_0 t`.

.. dropdown:: Example: Minimal Custom Shock Engine

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.dynamics.shocks.core.shock_engine import ShockEngine
        from triceratops.utils.misc_utils import ensure_in_units


        class FreeExpansionShockEngine(ShockEngine):
            """Shock engine for constant-velocity free expansion: R(t) = v0 * t."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def compute_shock_properties(self, time, v0=1e9 * u.cm / u.s, **kwargs):
                t_s   = ensure_in_units(time, u.s)
                v0_cs = ensure_in_units(v0, u.cm / u.s)
                props = self._compute_shock_properties_cgs(t_s, v0=v0_cs)
                return {
                    'radius':   props['radius']   * u.cm,
                    'velocity': props['velocity'] * (u.cm / u.s),
                }

            def _compute_shock_properties_cgs(self, time, v0=1e9, **kwargs):
                time = np.asarray(time, dtype=float)
                return {
                    'radius':   v0 * time,
                    'velocity': np.full_like(time, v0),
                }


        engine = FreeExpansionShockEngine()
        t = np.logspace(6, 8, 5) * u.s
        props = engine(t, v0=3e9 * u.cm / u.s)
        print(props['radius'].to(u.AU))

----

Further Reading
---------------

The table below maps common modeling tasks to the relevant documentation:

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - If you want to...
      - See...
    * - Understand the Chevalier self-similar solution
      - :ref:`chevalier_theory`
    * - Understand the Sedov--Taylor blast wave
      - :ref:`sedov_taylor_theory`
    * - Use the Chevalier or Sedov--Taylor engines (API)
      - :ref:`self_similar_shocks_overview`
    * - Understand the thin-shell ODE formulation
      - :ref:`numeric_shocks_theory`
    * - Use the numerical engine with custom density profiles
      - :ref:`numerical_shocks_overview`
    * - Compute Rankine--Hugoniot jump conditions directly
      - :ref:`rankine_hugoniot_overview`
    * - Model synchrotron emission driven by a shock
      - :ref:`synchrotron_overview`
    * - Get a broad introduction to shock physics in Triceratops
      - :ref:`shock_overview`

.. footbibliography::
