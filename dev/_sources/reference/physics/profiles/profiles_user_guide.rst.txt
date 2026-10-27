.. _profiles_user_guide:

===================
Dynamical Profiles
===================

.. hint::

    Profiles are used as source functions inside shock engines, ODE integrators, and
    parameter-grid sweeps throughout :mod:`trilobite.dynamics`. See
    :ref:`shock_engines` for how they plug in to shock calculations.

The :mod:`trilobite.dynamics.profiles` module provides **parameterized analytic field
classes** for the density and velocity distributions used in dynamical calculations.
Three families are provided: CSM (circumstellar-medium) density profiles, homologous
ejecta density profiles, and upstream velocity profiles.

The central design principle is **statelessness**: no physical parameters are ever stored
on a profile class or its instances. Instead, parameters are passed at evaluation time.
This makes profiles safe for MCMC inference and parameter sweeps, where the same profile
form is evaluated thousands of times with different parameter values, and it keeps the
classes trivially serializable and thread-safe.

All profiles inherit from the abstract base class
:class:`~trilobite.dynamics.profiles.core._DynamicalProfile`, which defines the shared
evaluation interface described below.

.. contents::
    :local:
    :depth: 2

----

The Shared Evaluation Interface
---------------------------------

Every profile class exposes the same four entry points, regardless of which physical
quantity it represents.

``eval`` — safe, unit-aware evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~trilobite.dynamics.profiles.core._DynamicalProfile.eval` is the user-facing
path. It accepts :class:`~astropy.units.Quantity` inputs (or bare floats, which are
interpreted as CGS), converts everything to CGS internally, runs the optimized numerical
kernel, and returns the result as a :class:`~astropy.units.Quantity` in the profile's
native output units. This is the right method to call in analysis scripts, notebooks, and
high-level model code.

.. code-block:: python

    import astropy.units as u
    from trilobite.dynamics.profiles import WindCSMProfile

    r = 1e16 * u.cm
    rho = WindCSMProfile.eval(r, A_star=1.0)  # returns Quantity in g cm⁻³

``feval`` — fast, unit-free evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~trilobite.dynamics.profiles.core._DynamicalProfile.feval` calls the numerical
kernel directly, with no validation, unit conversion, or output-unit attachment. Inputs
must already be plain floats or arrays in CGS. This is the path for solver hot loops and
Cython extension modules where unit-handling overhead would be significant.

.. code-block:: python

    rho_cgs = WindCSMProfile.feval(1e16, A=5e11)  # bare CGS in, bare CGS out

``as_callable`` — unit-aware frozen closure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~trilobite.dynamics.profiles.core._DynamicalProfile.as_callable` validates and
processes a set of parameters **once**, then returns a lightweight unit-aware closure
``f(r, t)`` — or ``f(r)`` for stationary profiles. Use this when you want to pass a
profile to a higher-level function that expects a callable.

.. code-block:: python

    rho_func = WindCSMProfile.as_callable(A_star=1.0)
    rho = rho_func(1e16 * u.cm)   # Quantity in g cm⁻³

``as_optimized_callable`` — unit-free frozen closure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~trilobite.dynamics.profiles.core._DynamicalProfile.as_optimized_callable`
does the same parameter processing once, but returns a unit-free closure that calls
``feval`` directly. This is the appropriate callable to pass into ODE right-hand sides
or other numerical kernels that already work in CGS.

.. code-block:: python

    rho_func = WindCSMProfile.as_optimized_callable(A_star=1.0)
    rho_cgs  = rho_func(1e16)   # plain float, no units

Separating processing from evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For cases where you want to validate and convert parameters once and then call ``feval``
manually in a loop, the explicit separation step is available via
:meth:`~trilobite.dynamics.profiles.core._DynamicalProfile.process_parameters` and
:meth:`~trilobite.dynamics.profiles.core._DynamicalProfile.process_arguments`:

.. code-block:: python

    processed = WindCSMProfile.process_parameters(A_star=1.0)
    r_cgs, = WindCSMProfile.process_arguments(1e16 * u.cm)   # note: unpacked tuple

    for r in radii_cgs:
        rho = WindCSMProfile.feval(r, **processed)

.. note::

    ``as_optimized_callable`` and the ``process_*`` / ``feval`` split are
    functionally equivalent — the former is simply more ergonomic when the
    callable is passed directly to a solver.

----

CSM Density Profiles
---------------------

CSM profiles represent the upstream circumstellar-medium density field

.. math::

    \rho_{\rm CSM} = \rho_{\rm CSM}(r, t;\,\boldsymbol{\theta}),

evaluated at radius :math:`r` and time :math:`t`. All currently implemented profiles
are **stationary** (i.e., :math:`\rho` does not depend on :math:`t`), but they accept
the time argument for interface compatibility with time-dependent density fields in
shock engines. Stationary profiles simply ignore ``t`` when it is provided.

.. list-table:: Available CSM Density Profiles
    :header-rows: 1
    :widths: 28 20 52

    * - Class
      - Key Parameters
      - Description
    * - :class:`~trilobite.dynamics.profiles.csm.WindCSMProfile`
      - ``A``, ``A_star``, or ``mass_loss_rate`` + ``wind_velocity``
      - Steady wind: :math:`\rho = A\,r^{-2}`. Three equivalent parameterisation routes.
    * - :class:`~trilobite.dynamics.profiles.csm.UniformCSMProfile`
      - ``rho_0``
      - Uniform ISM density: :math:`\rho = \rho_0`.
    * - :class:`~trilobite.dynamics.profiles.csm.PowerLawCSMProfile`
      - ``rho_ref``, ``r_ref``, ``s``
      - Single power law: :math:`\rho = \rho_{\rm ref}(r/r_{\rm ref})^{-s}`.
    * - :class:`~trilobite.dynamics.profiles.csm.ShellCSMProfile`
      - ``rho_0``, ``r_shell``, ``dr_shell``
      - Top-hat shell of width ``dr_shell`` centred at ``r_shell``.
    * - :class:`~trilobite.dynamics.profiles.csm.GaussianShellCSMProfile`
      - ``rho_0``, ``r_shell``, ``sigma_r``
      - Gaussian shell: :math:`\rho \propto \exp[-(r - r_{\rm shell})^2 / 2\sigma_r^2]`.
    * - :class:`~trilobite.dynamics.profiles.csm.BrokenPowerLawCSMProfile`
      - ``rho_ref``, ``r_ref``, ``s_inner``, ``s_outer``
      - Broken power law with inner and outer slopes.
    * - :class:`~trilobite.dynamics.profiles.csm.TruncatedWindCSMProfile`
      - ``A`` (or ``A_star``), ``r_max``
      - Wind profile sharply truncated at ``r_max``.
    * - :class:`~trilobite.dynamics.profiles.csm.WindWithFloorCSMProfile`
      - ``A`` (or ``A_star``), ``rho_floor``
      - Wind profile with a uniform density floor: :math:`\rho = \max(A r^{-2},\,\rho_{\rm floor})`.
    * - :class:`~trilobite.dynamics.profiles.csm.SmoothTruncatedWindCSMProfile`
      - ``A`` (or ``A_star``), ``r_max``, ``q``
      - Wind profile with a smooth power-law transition to zero beyond ``r_max``.

The wind profile is the most commonly used CSM environment and therefore the most
feature-rich class. :class:`~trilobite.dynamics.profiles.csm.WindCSMProfile` accepts three
equivalent parameterisation routes, illustrated below.

.. code-block:: python

    import astropy.units as u
    from trilobite.dynamics.profiles import WindCSMProfile

    r = 1e16 * u.cm

    # Route 1 — supply A directly
    rho = WindCSMProfile.eval(r, A=5e11 * u.g / u.cm)

    # Route 2 — dimensionless wind-density parameter A★
    rho = WindCSMProfile.eval(r, A_star=1.0)

    # Route 3 — physical mass-loss rate and wind velocity
    rho = WindCSMProfile.eval(
        r,
        mass_loss_rate=1e-5 * u.Msun / u.yr,
        wind_velocity=1000 * u.km / u.s,
    )

    # Compute A★ from physical parameters without evaluating the profile
    A_star = WindCSMProfile.normalize_A_star(
        mass_loss_rate=1e-5 * u.Msun / u.yr,
        wind_velocity=1000 * u.km / u.s,
    )  # → 1.0 (by definition of A★)

.. note::

    All CSM profile classes inherit from
    :class:`~trilobite.dynamics.profiles.csm.StationaryCSMDensityProfile`, which makes the
    time argument fully optional. You can call ``eval(r, **params)`` without providing
    ``t`` at all, or pass a dummy ``t`` for interface uniformity — both are valid.

----

Ejecta Density Profiles
------------------------

Ejecta profiles describe homologously expanding supernova ejecta. All profiles in this
family assume the factored form

.. math::

    \rho_{\rm ej}(r, t) = t^{-3}\,G(r/t),

where :math:`G(v)` is the time-independent **velocity-space kernel**. The :math:`t^{-3}`
prefactor is handled automatically by
:class:`~trilobite.dynamics.profiles.ejecta.EjectaDensityProfile`; subclasses only need to
implement :math:`G(v)`.

.. list-table:: Available Ejecta Density Profiles
    :header-rows: 1
    :widths: 38 30 32

    * - Class
      - Key Parameters
      - Description
    * - :class:`~trilobite.dynamics.profiles.ejecta.BrokenPowerLawEjectaProfile`
      - ``K``, ``v_t``, ``n``, ``delta``, ``v_max`` (opt.)
      - Chevalier-style broken power law with flat inner core and steep outer
        envelope.
    * - :class:`~trilobite.dynamics.profiles.ejecta.ExponentialEjectaProfile`
      - ``K``, ``v_e``, ``v_min`` (opt.), ``v_max`` (opt.)
      - Exponential kernel :math:`G(v) = K\exp(-v/v_e)`, optionally truncated.

Normalization
^^^^^^^^^^^^^

Rather than specifying ``K`` and ``v_t`` (or ``v_e``) by hand, both classes
expose a ``normalize`` classmethod that computes the parameters from the
physically motivated total ejecta mass :math:`M_{\rm ej}` and kinetic energy
:math:`E_{\rm ej}`:

.. code-block:: python

    import astropy.units as u
    from trilobite.dynamics.profiles import (
        BrokenPowerLawEjectaProfile,
        ExponentialEjectaProfile,
    )

    # Broken power law — typical Chevalier (1982) parametrization
    K, v_t = BrokenPowerLawEjectaProfile.normalize(
        E_ej=1e51 * u.erg,
        M_ej=1.0 * u.Msun,
        n=10,      # outer power-law index (must be > 5)
        delta=0,   # inner power-law index (must be < 3)
    )
    rho = BrokenPowerLawEjectaProfile.eval(
        1e16 * u.cm, 10 * u.day,
        K=K, v_t=v_t, n=10, delta=0,
    )

    # Exponential kernel
    K_exp, v_e = ExponentialEjectaProfile.normalize(
        E_ej=1e51 * u.erg,
        M_ej=1.0 * u.Msun,
    )
    rho_exp = ExponentialEjectaProfile.eval(
        1e16 * u.cm, 10 * u.day,
        K=K_exp, v_e=v_e,
    )

Both ``normalize`` methods optionally accept velocity cutoffs (``v_max`` for the
broken power law; ``v_min`` and ``v_max`` for the exponential) to describe ejecta with
a finite velocity range.

The kernel interface
^^^^^^^^^^^^^^^^^^^^^

The velocity-space kernel :math:`G(v)` is directly accessible through two methods,
which is useful when the velocity-space distribution is needed independently of the
full space-time density (for example, when computing mass- or energy-weighted moments):

.. code-block:: python

    import numpy as np
    import astropy.units as u
    from trilobite.dynamics.profiles import BrokenPowerLawEjectaProfile

    K, v_t = BrokenPowerLawEjectaProfile.normalize(1e51 * u.erg, 1.0 * u.Msun, n=10, delta=0)

    v = np.geomspace(1e7, 1e10, 200) * u.cm / u.s

    # unit-aware
    G = BrokenPowerLawEjectaProfile.kernel(v, K=K, v_t=v_t, n=10, delta=0)

    # unit-free (hot-loop path)
    G_raw = BrokenPowerLawEjectaProfile.fkernel(
        v.to(u.cm / u.s).value,
        **BrokenPowerLawEjectaProfile.process_parameters(K=K, v_t=v_t, n=10, delta=0),
    )

----

Velocity Profiles
-----------------

Velocity profiles describe the bulk upstream velocity field

.. math::

    u = u(r, t;\,\boldsymbol{\theta})

in :math:`\mathrm{cm\,s^{-1}}`. They are used primarily by numerical shock engines to
set the upstream flow velocity at each integration step.

.. list-table:: Available Velocity Profiles
    :header-rows: 1
    :widths: 38 22 40

    * - Class
      - Key Parameters
      - Description
    * - :class:`~trilobite.dynamics.profiles.velocity.StaticVelocityProfile`
      - (none)
      - Zero velocity field: :math:`u(r, t) = 0`. Stationary medium.
    * - :class:`~trilobite.dynamics.profiles.velocity.HomologousVelocityProfile`
      - (none)
      - Homologous expansion: :math:`u(r, t) = r/t`. Freely expanding medium.
    * - :class:`~trilobite.dynamics.profiles.velocity.ConstantVelocityProfile`
      - ``v_0``
      - Spatially uniform constant velocity: :math:`u(r, t) = v_0`.

.. code-block:: python

    import astropy.units as u
    from trilobite.dynamics.profiles import (
        StaticVelocityProfile,
        HomologousVelocityProfile,
        ConstantVelocityProfile,
    )

    r, t = 1e16 * u.cm, 10 * u.day

    u_static     = StaticVelocityProfile.eval(r, t)             # 0 cm/s
    u_homologous = HomologousVelocityProfile.eval(r, t)         # r/t
    u_const      = ConstantVelocityProfile.eval(r, t, v_0=1e9 * u.cm / u.s)

----

API Reference
--------------

Base Classes
^^^^^^^^^^^^^

.. currentmodule:: trilobite.dynamics.profiles.core

.. autosummary::
    :toctree: ../../../_as_gen

    _DynamicalProfile

.. currentmodule:: trilobite.dynamics.profiles.csm

.. autosummary::
    :toctree: ../../../_as_gen

    CSMDensityProfile
    StationaryCSMDensityProfile

.. currentmodule:: trilobite.dynamics.profiles.ejecta

.. autosummary::
    :toctree: ../../../_as_gen

    EjectaDensityProfile

.. currentmodule:: trilobite.dynamics.profiles.velocity

.. autosummary::
    :toctree: ../../../_as_gen

    VelocityProfile

CSM Density Profiles
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: trilobite.dynamics.profiles.csm

.. autosummary::
    :toctree: ../../../_as_gen

    WindCSMProfile
    UniformCSMProfile
    PowerLawCSMProfile
    ShellCSMProfile
    GaussianShellCSMProfile
    BrokenPowerLawCSMProfile
    TruncatedWindCSMProfile
    WindWithFloorCSMProfile
    SmoothTruncatedWindCSMProfile

Ejecta Density Profiles
^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: trilobite.dynamics.profiles.ejecta

.. autosummary::
    :toctree: ../../../_as_gen

    BrokenPowerLawEjectaProfile
    ExponentialEjectaProfile

Velocity Profiles
^^^^^^^^^^^^^^^^^

.. currentmodule:: trilobite.dynamics.profiles.velocity

.. autosummary::
    :toctree: ../../../_as_gen

    StaticVelocityProfile
    HomologousVelocityProfile
    ConstantVelocityProfile
