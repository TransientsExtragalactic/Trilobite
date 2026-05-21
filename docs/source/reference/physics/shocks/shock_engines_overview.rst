.. _shock_engines:
===========================================
Shock Engines
===========================================

The evolution of an astrophysical shock is the **central dynamical ingredient** in modeling
radio and X-ray transients. Whether one is describing the ejecta--CSM interaction of a
supernova\ :footcite:p:`chevalierSelfsimilarSolutionsInteraction1982`, the deceleration of
a gamma-ray burst afterglow, or the late-time blast wave of a tidal disruption event, the
shock radius :math:`R(t)` and velocity :math:`v(t)` set the stage for everything that
follows: they determine the post-shock conditions that feed into the synchrotron and
thermal emission models, and ultimately shape the light curves and SEDs that observers
measure.

Trilobite provides a unified **shock engine** interface that wraps the physics of shock
evolution and returns time-varying shock properties in a consistent format, regardless of
whether the underlying model is a closed-form self-similar solution, a numerical ODE
integrator, or a bespoke prescription.  The engines are intentionally **stateless**,
physical parameters are passed at call time rather than stored on the object, so that they
can be freely used in parameter sweeps and Bayesian inference pipelines without hidden state
or unexpected side effects.

These configurations can also be used to train neural surrogates of shock dynamics, which are
orders of magnitude faster to evaluate than numerical engines and can be used in inference
workflows that would otherwise be prohibitively expensive.

.. hint::

    For users unfamiliar with the underlying shock physics, the theory guides provide
    self-contained derivations of each solution:

    - :ref:`chevalier_theory` — the Chevalier self-similar ejecta--CSM interaction
    - :ref:`sedov_taylor_theory` — the Sedov--Taylor point-explosion blast wave
    - :ref:`blandford_mckee_theory` — the Blandford--McKee ultra-relativistic blast wave
    - :ref:`numeric_shocks_theory` — the thin-shell ODE formulations, including the
      :ref:`pressure-driven <pressure_driven_thin_shell_model>` and
      :ref:`mechanical internal-energy <mechanical_internal_energy_model>` models

    :footcite:t:`ChevalierFranssonHandbook` is an excellent review of supernovae in
    circumstellar media; :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`
    is the primary reference for the Chevalier self-similar solutions.
    The Blandford--McKee solution is derived in :footcite:t:`1976PhFl...19.1130B`.
    The mechanical shock engine follows :footcite:t:`beloborodovMechanicalModelRelativistic2006a`.

.. contents::
    :local:
    :depth: 2

----

Module Overview
---------------

In a *typical shock dynamics problem*, the user has a physical scenario, for example, a
supernova explosion into a stellar wind, and wants to predict how the shock evolves with
time.  The workflow generally proceeds in two steps:

1. **Build Initial Conditions**: For numerical solvers, use the factory functions in
   :mod:`~trilobite.dynamics.shocks.utils` to construct callable density and velocity
   profiles for the ejecta and circumstellar medium. For self-similar models,
   :mod:`~trilobite.dynamics.shocks.utils` provides various helpers to convert physical
   parameters like explosion energy and mass-loss rate into the normalization constants that
   the self-similar solutions require.
   See `Profiles and Model Setup`_
   below for a full reference table.

3. **Evolve the shock**: Call the engine's
   :meth:`~trilobite.dynamics.shocks.core.shock_engine.ShockEngine.compute_shock_properties`
   method, or simply call the engine as a function, since all engines are callable, with a
   time array to obtain the shock radius, velocity, and any additional post-shock
   thermodynamic quantities supported by that particular engine.

Each engine class returns a specific set of shock properties in a custom state class (a named tuple).

Module Structure
^^^^^^^^^^^^^^^^

The shock engine infrastructure spans various layers of the larger :mod:`~trilobite.dynamics.shocks` subpackage.
The tables below summarize the key classes and functions in each layer, along with their responsibilities.
For detailed API references and worked examples, see the linked sections.

.. rubric:: Infrastructure

At the core of the shock-engine architecture is the abstract base class, contained in
the :mod:`~trilobite.dynamics.shocks.core.shock_engine` module, which defines the two-method
contract that all engines must satisfy.

.. list-table::
    :header-rows: 1
    :widths: 42 58

    * - Class
      - Responsibility
    * - :class:`~trilobite.dynamics.shocks.core.shock_engine.ShockEngine`
      - Abstract base class defining the two-method contract that all engines must satisfy.
        Users encounter this class only when building custom engines; see
        `Building a Custom Shock Engine`_ below.

.. rubric:: Self-Similar Engines

The simplest (and most commonly used) set of shock engines are those corresponding to so-called
`self-similar <https://en.wikipedia.org/wiki/Self-similar_solution>`__ solutions, which are computationally cheap
analytic prescriptions valid when density profiles take special forms.

.. list-table::
    :header-rows: 1
    :widths: 42 58

    * - Class
      - Responsibility
    * - :class:`~trilobite.dynamics.shocks.chevalier.ChevalierSelfSimilarShockEngine`
      - Self-similar ejecta--CSM interaction for arbitrary power-law density profiles
        :footcite:p:`chevalierSelfsimilarSolutionsInteraction1982`.
        Single-surface (contact discontinuity only) with analytic thin-shell normalization.
        See :ref:`chevalier_theory`.
    * - :class:`~trilobite.dynamics.shocks.chevalier.ChevalierSelfSimilarWindShockEngine`
      - Specialization of the single-surface Chevalier engine for a steady stellar wind CSM
        (:math:`s = 2`). Accepts the mass-loss rate :math:`\dot{M}` and wind velocity
        :math:`v_w` directly, rather than a pre-computed normalization constant.
    * - :class:`~trilobite.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarEngine`
      - Two-surface Chevalier engine that resolves the **forward and reverse shocks
        separately** using a pre-tabulated :math:`(n,s)` grid of self-similar constants
        (:math:`A`, :math:`R_{\rm fs}/R_c`, :math:`R_{\rm rs}/R_c`).  More accurate
        post-shock thermodynamics than the single-surface engine.
        See :ref:`chevalier_two_shock_engine`.
    * - :class:`~trilobite.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarWindEngine`
      - Steady-wind (:math:`s = 2`) specialization of
        :class:`~trilobite.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarEngine`.
        Accepts :math:`\dot{M}` and :math:`v_w` directly.
    * - :class:`~trilobite.dynamics.shocks.sedov_taylor.SedovTaylorShockEngine`
      - Point-explosion blast wave in a uniform ambient medium
        :footcite:p:`sedov1946propagation` :footcite:p:`taylor1950formation`.
        Returns post-shock density, pressure, and temperature in addition to kinematics.
        See :ref:`sedov_taylor_theory`.
    * - :class:`~trilobite.dynamics.shocks.blandford_mckee.BlandfordMcKeeShockEngine`
      - Ultra-relativistic blast wave in a power-law external medium
        :footcite:p:`1976PhFl...19.1130B`.  Fully analytic normalization
        :math:`C_E(k) = 8\pi/(17-4k)` for arbitrary :math:`k < 3`.
        Returns both proper (comoving) and lab-frame post-shock quantities.
        See :ref:`blandford_mckee_engine`.
    * - :class:`~trilobite.dynamics.shocks.blandford_mckee.BlandfordMcKeeWindShockEngine`
      - Stellar-wind (:math:`k=2`) specialization of
        :class:`~trilobite.dynamics.shocks.blandford_mckee.BlandfordMcKeeShockEngine`.
        Accepts :math:`\dot{M}` and :math:`v_w` directly.

.. rubric:: Numerical Engines

In cases when self-similar solvers are insufficient, one must turn to more complex engines which rely
on numerical methods. These engines integrate ODEs and can handle arbitrary ejecta and CSM density profiles, but are
more computationally expensive than their self-similar counterparts. Trilobite currently provides two
fully implemented ODE-based engines, with four relativistic and momentum-conserving
extensions reserved as planned engines for future releases.

.. list-table::
    :header-rows: 1
    :widths: 42 58

    * - Class
      - Responsibility
    * - :class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`
      - ODE-based thin-shell engine driven by the instantaneous Rankine--Hugoniot pressure
        difference across the interaction region.  Supports arbitrary ejecta and CSM
        callables.  See :ref:`pressure_driven_thin_shell_model`.
    * - :class:`~trilobite.dynamics.shocks.numerical.MechanicalShockEngine`
      - Non-relativistic mechanical engine that self-consistently evolves separate internal
        energies for the shocked ejecta and CSM layers, following
        :footcite:t:`beloborodovMechanicalModelRelativistic2006a`.
        See :ref:`mechanical_internal_energy_model`.

.. rubric:: Utilities

To help users with setting up various density and velocity profiles for the ejecta and CSM, the
:mod:`~trilobite.dynamics.shocks.utils` module provides factory functions that return unit-free
CGS callables for efficient evaluation inside ODE right-hand sides. These utilities are essential
for preparing the initial conditions for numerical shock engines,
and they also include normalization helpers for self-similar solutions.

.. list-table::
    :header-rows: 1
    :widths: 42 58

    * - Module
      - Responsibility
    * - :mod:`~trilobite.dynamics.shocks.utils`
      - Factory functions for velocity profiles, ejecta kernels, ejecta density profiles,
        and CSM density profiles.  All returned callables are unit-free CGS for efficient
        evaluation inside ODE right-hand sides.  See `Profiles and Model Setup`_ below.

----

Quickstart
----------

The example below demonstrates the most common use case: evolving a Chevalier wind-CSM
shock and plotting the resulting radius and velocity.  The
:class:`~trilobite.dynamics.shocks.chevalier.ChevalierSelfSimilarWindShockEngine` is the
simplest entry point — it requires only four physical quantities (explosion energy, ejecta
mass, wind mass-loss rate, and wind velocity) and handles all normalization internally.

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as u
    from trilobite.dynamics.shocks import ChevalierSelfSimilarWindShockEngine

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
        props.radius.to(u.cm).value / 1e16,
        color='steelblue', lw=2,
    )
    axes[0].set_xlabel('Time [days]')
    axes[0].set_ylabel(r'Shock radius [$10^{16}$ cm]')
    axes[0].set_title('Shock Radius')
    axes[0].grid(True, which='both', ls='--', alpha=0.4)

    axes[1].loglog(
        t.to(u.day).value,
        props.velocity.to(u.km / u.s).value,
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

Trilobite provides two broad categories of shock engine: **self-similar** solutions,
which are computationally cheap analytic prescriptions valid when density profiles take
simple power-law form, and **numerical** engines, which integrate ODEs and can handle
arbitrary ejecta and CSM density profiles.

Self-Similar Shock Engines
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

    :ref:`self_similar_shocks_overview` — detailed API reference and worked examples.

    :ref:`chevalier_theory` and :ref:`sedov_taylor_theory` — theoretical derivations.

Self-similar shock solutions arise when the governing equations admit a scale-free form.
The shock then evolves such that a single power law in time describes both the radius and
velocity, converting what is nominally a partial differential equation into an ordinary one
and, in the most favorable cases, yielding a fully analytic result.  These solutions are
computationally cheap and carry intuitive physical scaling relations, making them the
preferred first choice for parameter inference.

Five self-similar engines are currently implemented:

.. list-table::
    :header-rows: 1
    :widths: 38 14 16 32

    * - Class
      - Geometry
      - Medium
      - Use when...
    * - :class:`~trilobite.dynamics.shocks.chevalier.ChevalierSelfSimilarShockEngine`
        — :ref:`details <chevalier_engine>`
      - Spherical
      - Power-law (:math:`\rho \propto r^{-s}`)
      - Fast single-surface estimate; inference pipelines where speed matters most
    * - :class:`~trilobite.dynamics.shocks.chevalier.ChevalierSelfSimilarWindShockEngine`
        — :ref:`details <chevalier_engine>`
      - Spherical
      - Stellar wind (:math:`s = 2`)
      - Same as above; :math:`\dot{M}` and :math:`v_w` as direct inputs
    * - :class:`~trilobite.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarEngine`
        — :ref:`details <chevalier_two_shock_engine>`
      - Spherical
      - Power-law (:math:`\rho \propto r^{-s}`)
      - Separate forward/reverse shock thermodynamics; accurate normalization from ODE table
    * - :class:`~trilobite.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarWindEngine`
        — :ref:`details <chevalier_two_shock_engine>`
      - Spherical
      - Stellar wind (:math:`s = 2`)
      - Same as above; :math:`\dot{M}` and :math:`v_w` as direct inputs
    * - :class:`~trilobite.dynamics.shocks.sedov_taylor.SedovTaylorShockEngine`
        — :ref:`details <sedov_taylor_engine>`
      - Spherical
      - Uniform (:math:`\rho = \rho_0`)
      - Point explosion in a uniform medium; full post-shock thermodynamics returned
    * - :class:`~trilobite.dynamics.shocks.blandford_mckee.BlandfordMcKeeShockEngine`
        — :ref:`details <blandford_mckee_engine>`
      - Spherical
      - Power-law (:math:`\rho \propto r^{-k},\; k < 3`)
      - Ultra-relativistic (:math:`\Gamma \gg 1`) blast wave; both proper and
        lab-frame post-shock quantities; returns :math:`\Gamma`, :math:`\beta`,
        and :math:`\gamma_{2,\mathrm{lab}}`
    * - :class:`~trilobite.dynamics.shocks.blandford_mckee.BlandfordMcKeeWindShockEngine`
        — :ref:`details <blandford_mckee_engine>`
      - Spherical
      - Stellar wind (:math:`k = 2`)
      - Same as above; :math:`\dot{M}` and :math:`v_w` as direct inputs


Numerical Shock Engines
^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

    :ref:`numerical_shocks_overview` — detailed API reference and worked examples.

    :ref:`numeric_shocks_theory` — derivation of the thin-shell equations of motion.

When the self-similar approximation fails — for instance when density profiles cannot be
described by simple power laws, or when the shock transitions between dynamical regimes
during its evolution — a numerical shock engine is necessary.  Trilobite provides two
fully implemented ODE-based engines.  Both share the same callable interface and accept
upstream density and velocity profiles built with the factory functions in
:mod:`~trilobite.dynamics.shocks.utils` (see `Profiles and Model Setup`_ below).
Four relativistic and momentum-conserving extensions are reserved as planned engines
and will be added in future releases.

.. list-table::
    :header-rows: 1
    :widths: 44 14 42

    * - Class
      - Status
      - Description
    * - :class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`
        — :ref:`details <pressure_driven_thin_shell_engine>`
      - Available
      - Collapses the shocked interaction region to a single thin shell driven by the
        net Rankine--Hugoniot pressure difference.  The simplest numerical engine;
        suitable when a full thermal energy budget is not required.
    * - :class:`~trilobite.dynamics.shocks.numerical.MechanicalShockEngine`
        — :ref:`details <mechanical_shock_engine>`
      - Available
      - Self-consistently evolves separate internal energies for the shocked ejecta and
        CSM layers :footcite:p:`beloborodovMechanicalModelRelativistic2006a`.  Retains
        the minimal two-shock structure while remaining inexpensive enough for parameter
        studies and inference workflows.
    * - :class:`~trilobite.dynamics.shocks.numerical.RelPressureDrivenThinShellShockEngine`
        — :ref:`details <rel_pressure_driven_thin_shell_engine>`
      - **[planned]**
      - Relativistic extension of the pressure-driven thin-shell engine.
        **Not yet implemented — planned for a future release.**
    * - :class:`~trilobite.dynamics.shocks.numerical.RelMechanicalShockEngine`
        — :ref:`details <rel_mechanical_shock_engine>`
      - **[planned]**
      - Relativistic extension of the mechanical engine following
        :footcite:t:`beloborodovMechanicalModelRelativistic2006a`.
        **Not yet implemented — planned for a future release.**
    * - :class:`~trilobite.dynamics.shocks.numerical.MomentumConservingShockEngine`
        — :ref:`details <momentum_conserving_shock_engine>`
      - **[planned]**
      - Non-relativistic momentum-conserving shock engine.
        **Not yet implemented — planned for a future release.**
    * - :class:`~trilobite.dynamics.shocks.numerical.RelMomentumConservingShockEngine`
        — :ref:`details <rel_momentum_conserving_shock_engine>`
      - **[planned]**
      - Relativistic momentum-conserving shock engine.
        **Not yet implemented — planned for a future release.**

----

Profiles and Model Setup
-------------------------

The :mod:`~trilobite.dynamics.shocks.utils` module provides factory functions for the
upstream density and velocity profiles that numerical shock engines require as boundary
conditions.  All returned callables are unit-free CGS so they can be evaluated efficiently
inside ODE right-hand sides without unit-conversion overhead.  Unit handling is performed
once at factory-call time.

The module is organized into three groups: **velocity profiles** for the bulk upstream
flow, **ejecta profiles** for freely expanding supernova ejecta, and **CSM profiles** for
the ambient circumstellar medium.  A fourth helper assembles these into the four-callable
tuple expected by numerical shock engines.

Velocity Profiles
^^^^^^^^^^^^^^^^^

Simple two-argument callables ``u(r, t)`` representing the bulk velocity field of the
upstream gas.  All three are accepted wherever a velocity-profile callable is expected.

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - Function
      - Description
    * - :func:`~trilobite.dynamics.shocks.utils.stationary_velocity_profile`
      - Returns zero velocity for all inputs.  Suitable for a stationary CSM.
    * - :func:`~trilobite.dynamics.shocks.utils.homologous_velocity_profile`
      - Returns :math:`u(r,t) = r/t`.  Standard model for freely expanding ejecta.
    * - :func:`~trilobite.dynamics.shocks.utils.get_constant_velocity_profile`
      - Factory returning :math:`u(r,t) = u_0` for a fixed bulk-flow speed.

Ejecta Profiles
^^^^^^^^^^^^^^^

The ejecta density is modeled under the homologous approximation

.. math::

    \rho_{\rm ej}(r,t) = t^{-3}\,G(r/t),

where :math:`G(v)` is a time-independent velocity-space kernel.  Two profile families are
supported: **broken power-law** (BPL, after
:footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`) and **exponential**, each
available with or without an outer velocity truncation.  Each family is exposed at three
levels of abstraction:

- *Normalization helpers* — compute the scale parameters (:math:`v_t, K` or
  :math:`v_e, K`) from :math:`E_{\rm ej}` and :math:`M_{\rm ej}` and return
  :class:`~astropy.units.Quantity` scalars.
- *Kernel factories* — return a CGS callable :math:`G(v)` for use in
  :func:`~trilobite.dynamics.shocks.utils.make_homologous_stationary_sources` or a
  custom ODE right-hand side.
- *Profile factories* — return a CGS callable :math:`\rho_{\rm ej}(r,t)` that wraps the
  kernel with the :math:`t^{-3}` prefactor.

.. list-table::
    :header-rows: 1
    :widths: 50 22 28

    * - Function
      - Family
      - Output
    * - :func:`~trilobite.dynamics.shocks.utils.normalize_bpl_ejecta`
      - BPL
      - :math:`v_t`,\ :math:`K` (:class:`~astropy.units.Quantity`)
    * - :func:`~trilobite.dynamics.shocks.utils.normalize_truncated_bpl_ejecta`
      - BPL, :math:`v_{\max}` truncated
      - :math:`v_t`,\ :math:`K` (:class:`~astropy.units.Quantity`)
    * - :func:`~trilobite.dynamics.shocks.utils.normalize_exponential_ejecta`
      - Exponential
      - :math:`v_e`,\ :math:`K` (:class:`~astropy.units.Quantity`)
    * - :func:`~trilobite.dynamics.shocks.utils.normalize_truncated_exponential_ejecta`
      - Exponential, truncated
      - :math:`v_e`,\ :math:`K` (:class:`~astropy.units.Quantity`)
    * - :func:`~trilobite.dynamics.shocks.utils.get_bpl_ejecta_kernel`
      - BPL
      - :math:`G(v)` callable (CGS)
    * - :func:`~trilobite.dynamics.shocks.utils.get_truncated_bpl_ejecta_kernel`
      - BPL, :math:`v_{\max}` truncated
      - :math:`G(v)` callable (CGS)
    * - :func:`~trilobite.dynamics.shocks.utils.get_exponential_ejecta_kernel`
      - Exponential
      - :math:`G(v)` callable (CGS)
    * - :func:`~trilobite.dynamics.shocks.utils.get_truncated_exponential_ejecta_kernel`
      - Exponential, truncated
      - :math:`G(v)` callable (CGS)
    * - :func:`~trilobite.dynamics.shocks.utils.get_bpl_ejecta_profile`
      - BPL
      - :math:`\rho_{\rm ej}(r,t)` callable (CGS)
    * - :func:`~trilobite.dynamics.shocks.utils.get_truncated_bpl_ejecta_profile`
      - BPL, :math:`v_{\max}` truncated
      - :math:`\rho_{\rm ej}(r,t)` callable (CGS)
    * - :func:`~trilobite.dynamics.shocks.utils.get_exponential_ejecta_profile`
      - Exponential
      - :math:`\rho_{\rm ej}(r,t)` callable (CGS)
    * - :func:`~trilobite.dynamics.shocks.utils.get_truncated_exponential_ejecta_profile`
      - Exponential, truncated
      - :math:`\rho_{\rm ej}(r,t)` callable (CGS)

CSM Profiles
^^^^^^^^^^^^

Factory functions for the circumstellar medium geometries most commonly encountered in
transient modeling.  Each factory accepts physical parameters as
:class:`~astropy.units.Quantity` objects, converts units once at call time, and returns a
unit-free CGS callable ``rho_csm(r, t=None)``.  The optional ``t`` argument is accepted
for API compatibility with shock-engine source functions and is ignored for stationary
profiles.

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - Function
      - Profile
    * - :func:`~trilobite.dynamics.shocks.utils.compute_wind_csm_parameters`
      - Computes the wind normalization
        :math:`A = \dot{M}/(4\pi v_w)` in :math:`\mathrm{g\,cm^{-1}}`.
        Returns a :class:`~astropy.units.Quantity` scalar, not a callable.
    * - :func:`~trilobite.dynamics.shocks.utils.get_uniform_csm_density_func`
      - :math:`\rho(r) = \rho_0` — uniform medium.
    * - :func:`~trilobite.dynamics.shocks.utils.get_wind_csm_density_func`
      - :math:`\rho(r) = A r^{-2}` — steady spherical wind.
    * - :func:`~trilobite.dynamics.shocks.utils.get_truncated_wind_csm_density_func`
      - Wind profile that transitions sharply to a constant density floor outside
        :math:`r_{\max}`.
    * - :func:`~trilobite.dynamics.shocks.utils.get_smooth_truncated_wind_csm_density_func`
      - Wind profile with a smooth :math:`\tanh` cutoff near :math:`r_{\max}`.
    * - :func:`~trilobite.dynamics.shocks.utils.get_wind_with_floor_csm_density_func`
      - :math:`\rho(r) = A r^{-2} + \rho_{\rm floor}` — wind plus a constant ambient
        density.
    * - :func:`~trilobite.dynamics.shocks.utils.get_power_law_csm_density_func`
      - :math:`\rho(r) = \rho_{\rm ref}(r/r_{\rm ref})^{-s}` — general power law.
    * - :func:`~trilobite.dynamics.shocks.utils.get_broken_power_law_csm_density_func`
      - Continuous broken power law with independently specified inner and outer slopes.
    * - :func:`~trilobite.dynamics.shocks.utils.get_shell_csm_density_func`
      - Top-hat shell: constant density inside :math:`[r_{\rm in},r_{\rm out}]`, floor
        density outside.
    * - :func:`~trilobite.dynamics.shocks.utils.get_gaussian_shell_csm_density_func`
      - Gaussian-shell overdensity on a background:
        :math:`\rho(r) = \rho_{\rm bg} + \rho_{\rm shell}
        \exp\!\left[-\tfrac{1}{2}\left((r-R_s)/\sigma\right)^2\right]`.

Source Assembly
^^^^^^^^^^^^^^^

Once ejecta and CSM profile callables are in hand, the convenience factory
:func:`~trilobite.dynamics.shocks.utils.make_homologous_stationary_sources` assembles
the four two-argument source callables ``(rho_1, u_1, rho_4, u_4)`` expected by the
numerical shock engines for the standard scenario of homologous ejecta running into a
stationary CSM:

.. math::

    \rho_1(r,t) = t^{-3}G_{\rm ej}(r/t),
    \qquad
    u_1(r,t) = \frac{r}{t},
    \qquad
    \rho_4(r,t) = \rho_{\rm CSM}(r),
    \qquad
    u_4(r,t) = 0.

The factory accepts a kernel callable :math:`G_{\rm ej}(v)` — such as one returned by
:func:`~trilobite.dynamics.shocks.utils.get_bpl_ejecta_kernel` — and a CSM density
callable :math:`\rho_{\rm CSM}(r)` — such as one returned by
:func:`~trilobite.dynamics.shocks.utils.get_wind_csm_density_func` — and returns the
four-callable tuple ready to pass to
:meth:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine.compute_shock_properties`
or
:meth:`~trilobite.dynamics.shocks.numerical.MechanicalShockEngine.compute_shock_properties`.
For worked examples see :ref:`numerical_shocks_overview`.

----

Building a Custom Shock Engine
--------------------------------

Every shock engine in Trilobite inherits from the
:class:`~trilobite.dynamics.shocks.core.shock_engine.ShockEngine` abstract base class.
The contract is intentionally minimal — two methods — and is designed so that a custom
engine can be dropped into any model or inference pipeline that expects a
:class:`~trilobite.dynamics.shocks.core.shock_engine.ShockEngine`, without requiring
any changes to the surrounding infrastructure.

The ShockEngine Class
^^^^^^^^^^^^^^^^^^^^^^^

The abstract base class defines a two-method contract:

.. list-table::
    :header-rows: 1
    :widths: 42 58

    * - Method
      - Role
    * - :meth:`~trilobite.dynamics.shocks.core.shock_engine.ShockEngine.compute_shock_properties`
      - High-level, unit-aware entry point. Accepts :class:`~astropy.units.Quantity` inputs
        (or bare floats interpreted as CGS), validates arguments, and returns a dictionary
        of :class:`~astropy.units.Quantity` arrays. This is the method users call.
    * - :meth:`~trilobite.dynamics.shocks.core.shock_engine.ShockEngine._compute_shock_properties_cgs`
      - Low-level, pure-CGS computation. Accepts and returns bare floats or arrays with no
        unit overhead. Called internally by ``compute_shock_properties`` and available
        directly for performance-critical code paths such as tight inference loops.

All engines are also directly callable: ``engine(time, **parameters)`` is an alias for
``engine.compute_shock_properties(time, **parameters)``.

.. important::

    Physical model parameters must **never** be stored at construction time; all must be
    passed as arguments to ``compute_shock_properties`` at call time.  The ``__init__``
    method may set up instance-level configuration that does **not** depend on physical
    parameters — for example, numerical tolerances, pre-cached algorithmic coefficients,
    or lookup tables derived from solver settings (such as the :math:`(n,s)` grid used by
    :class:`~trilobite.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarEngine`).

    This design ensures that a single engine instance can be reused across many sets of
    parameters in a sweep or MCMC chain without unintended cross-contamination between
    evaluations.

The example below shows the minimum viable implementation of a custom shock engine: a
simple free-expansion model in which the shock radius grows as :math:`R(t) = v_0 t`.

.. dropdown:: Example: Minimal Custom Shock Engine

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from trilobite.dynamics.shocks.core.shock_engine import ShockEngine
        from trilobite.utils.misc_utils import ensure_in_units


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
    * - Understand the Blandford--McKee ultra-relativistic blast wave
      - :ref:`blandford_mckee_theory`
    * - Use the self-similar engines (API and examples)
      - :ref:`self_similar_shocks_overview`
    * - Understand the thin-shell ODE formulations
      - :ref:`numeric_shocks_theory`
    * - Use the numerical engines with custom density profiles
      - :ref:`numerical_shocks_overview`
    * - Build ejecta or CSM profile callables
      - `Profiles and Model Setup`_ (this page)
    * - Compute Rankine--Hugoniot jump conditions directly
      - :ref:`rankine_hugoniot_overview`
    * - Model synchrotron emission driven by a shock
      - :ref:`synchrotron_overview`
    * - Get a broad introduction to shock physics in Trilobite
      - :ref:`shock_overview`

.. footbibliography::
