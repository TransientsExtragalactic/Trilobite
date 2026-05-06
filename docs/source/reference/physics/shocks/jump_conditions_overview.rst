.. _rankine_hugoniot_overview:

===========================================
The Rankine-Hugoniot Jump Conditions
===========================================

.. seealso::

    :ref:`rankine_hugoniot_theory` for the theoretical background and derivation of the
    Rankine-Hugoniot conditions, including the general conservation-law form, the Mach-number
    parameterization, and the strong-shock and cold-upstream limits.

The `Rankine-Hugoniot <https://en.wikipedia.org/wiki/Rankine%E2%80%93Hugoniot_conditions>`__
jump conditions relate the physical properties of a fluid on either side of a shock front.
They follow from mass, momentum, and energy conservation applied across the shock discontinuity
and are foundational to modeling astrophysical shock waves in supernovae, stellar winds, and
jet interactions.

Triceratops implements the jump conditions as a hierarchy of class-method solvers, each
targeting a well-defined physical regime. All classes share a unified interface: a single
:meth:`~triceratops.dynamics.shocks.core.rankine_hugoniot.JumpConditions.solve` class method
that accepts physical inputs and returns a named tuple of post- (or pre-) shock quantities
with attached :class:`~astropy.units.Quantity` units.

.. note::

    For detailed theoretical references, see :footcite:t:`landau1987fluid`,
    :footcite:t:`thorne2017modern`, and :footcite:t:`clarke2007principles`.

.. contents::
   :local:
   :depth: 2

----

.. _rh_quick_start:

Quick Start
-----------

The following example uses :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongColdShockConditions`
— the most common regime in astrophysical modeling — to compute all post-shock quantities from a shock
propagating into a cold ambient medium:

.. code-block:: python

    import astropy.units as u
    from triceratops.dynamics.shocks.core import StrongColdShockConditions

    result = StrongColdShockConditions.solve(
        shock_velocity  = 1e9 * u.cm / u.s,   # 10,000 km/s
        flow_density    = 1e-24 * u.g / u.cm**3,
    )

    print(result.compression_ratio)              # dimensionless
    print(result.post_shock_temperature)         # in K
    print(result.post_shock_thermal_energy_density)  # in erg/cm^3

The returned object is a named tuple; all thermodynamic fields carry
:class:`~astropy.units.Quantity` units. The ``upstream=True`` flag inverts the
problem — given downstream conditions, it recovers the upstream state:

.. code-block:: python

    upstream = StrongColdShockConditions.solve(
        shock_velocity = 1e9 * u.cm / u.s,
        flow_density   = 4e-24 * u.g / u.cm**3,   # downstream density
        upstream=True,
    )

    print(upstream.pre_shock_density)

----

.. _rh_choosing:

Choosing a Solver
-----------------

Four concrete classes are available, organized along two axes: **shock strength**
(strong-shock limit vs. finite Mach number) and **upstream thermodynamics**
(cold vs. hot upstream medium). Use the table below to select the appropriate class:

.. list-table::
   :widths: 40 15 15 30
   :header-rows: 1

   * - Class
     - Regime
     - Upstream
     - Use when...
   * - :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongColdShockConditions`
     - :math:`\mathcal{M}_1 \gg 1`
     - Cold (:math:`P_1 \approx 0`)
     - Supernova blast waves, fast ejecta into tenuous CSM; no upstream thermal state needed
   * - :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongShockConditions`
     - :math:`\mathcal{M}_1 \gg 1`
     - Hot (finite :math:`P_1`)
     - Strong shocks where upstream pressure contributes; requires upstream :math:`P_1` or :math:`T_1`
   * - :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.WeakColdShockConditions`
     - Finite :math:`\mathcal{M}_1`
     - Cold (:math:`P_1 \approx 0`)
     - Moderate Mach-number shocks into cold medium; Mach number must be supplied explicitly
   * - :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.WeakShockConditions`
     - Finite :math:`\mathcal{M}_1`
     - Hot (finite :math:`P_1`)
     - Full Rankine-Hugoniot solution at arbitrary Mach number and upstream thermal state

.. note::

    "Weak shock" here is a naming convention, not a restriction to :math:`\mathcal{M}_1 \lesssim 1`.
    :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.WeakShockConditions` evaluates the
    full Mach-number-dependent relations and is valid for any supersonic shock. It reduces to
    :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongShockConditions` as
    :math:`\mathcal{M}_1 \to \infty`.

----

.. _rh_strong_shocks:

Strong Shock Conditions
-----------------------

In the strong-shock limit (:math:`\mathcal{M}_1 \gg 1`), the compression ratio saturates to
a constant that depends only on the adiabatic index:

.. math::

    R \equiv \frac{\rho_2}{\rho_1} = \frac{\gamma + 1}{\gamma - 1}.

For a monatomic ideal gas (:math:`\gamma = 5/3`), this gives :math:`R = 4`. All density, velocity,
and kinematic quantities can then be expressed without explicit knowledge of the Mach number. The
classes in this section exploit that simplification.

.. _rh_strong_cold:

StrongColdShockConditions
^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongColdShockConditions` applies
when the upstream medium is cold (:math:`P_1 \approx 0`, :math:`T_1 \approx 0`) — the common
assumption for astrophysical blast waves where the ambient medium is far cooler than the post-shock
gas. In this limit, the downstream thermodynamics depends on the shock velocity alone, and the
post-shock temperature becomes density-independent:

.. math::

    T_2 = \frac{\mu m_p}{k_B} \frac{R - 1}{R^2} u_1^2.

Because no upstream thermal state is needed, the :meth:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongColdShockConditions.solve`
signature is minimal:

.. code-block:: python

    from triceratops.dynamics.shocks.core import StrongColdShockConditions
    import astropy.units as u

    result = StrongColdShockConditions.solve(
        shock_velocity = 5e8 * u.cm / u.s,
        flow_density   = 1e-24 * u.g / u.cm**3,
        gamma          = 5/3,
        mu             = 0.61,
    )

The result is a named tuple with the following fields:

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Field
     - Units
     - Description
   * - ``compression_ratio``
     - dimensionless
     - :math:`R = (\gamma+1)/(\gamma-1)`
   * - ``post_shock_density``
     - :math:`\mathrm{g\,cm^{-3}}`
     - :math:`\rho_2 = R\,\rho_1`
   * - ``post_shock_number_density``
     - :math:`\mathrm{cm^{-3}}`
     - :math:`n_2 = \rho_2 / (\mu m_p)`
   * - ``post_shock_velocity``
     - :math:`\mathrm{cm\,s^{-1}}`
     - Lab-frame downstream bulk velocity
   * - ``post_shock_pressure``
     - :math:`\mathrm{dyn\,cm^{-2}}`
     - :math:`P_2 = \rho_1 u_1^2 (1 - 1/R)`
   * - ``post_shock_temperature``
     - :math:`\mathrm{K}`
     - :math:`T_2 = (\mu m_p / k_B)(R-1)/R^2 \, u_1^2`
   * - ``post_shock_thermal_energy_density``
     - :math:`\mathrm{erg\,cm^{-3}}`
     - :math:`e_2 = P_2 / (\gamma - 1)`

Passing ``upstream=True`` instead returns the inverse problem: given a downstream density, it
recovers the upstream density, number density, and velocity (thermodynamic inversion is not defined
in the cold limit):

.. code-block:: python

    upstream = StrongColdShockConditions.solve(
        shock_velocity = 5e8 * u.cm / u.s,
        flow_density   = 4e-24 * u.g / u.cm**3,   # post-shock density
        upstream=True,
    )
    print(upstream.pre_shock_density)

.. _rh_strong_hot:

StrongShockConditions
^^^^^^^^^^^^^^^^^^^^^^

:class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongShockConditions` handles the case
where the upstream medium has a non-negligible thermal state. The full momentum jump condition is

.. math::

    P_2 = P_1 + \rho_1 u_1^2 \left(1 - \frac{1}{R}\right),

so either the upstream pressure :math:`P_1` or the upstream temperature :math:`T_1` must be
supplied (they are mutually exclusive):

.. code-block:: python

    from triceratops.dynamics.shocks.core import StrongShockConditions
    import astropy.units as u

    result = StrongShockConditions.solve(
        shock_velocity    = 5e8 * u.cm / u.s,
        flow_density      = 1e-24 * u.g / u.cm**3,
        flow_temperature  = 1e4 * u.K,             # upstream temperature
    )

    # Equivalently, supply pressure:
    result = StrongShockConditions.solve(
        shock_velocity = 5e8 * u.cm / u.s,
        flow_density   = 1e-24 * u.g / u.cm**3,
        flow_pressure  = 1e-12 * u.dyn / u.cm**2,
    )

The output fields are identical to :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongColdShockConditions`
except that the energy density field is named ``post_shock_energy_density`` (not
``post_shock_thermal_energy_density``). When ``upstream=True``, the full pre-shock
thermodynamic state (pressure, temperature, energy density) is also returned.

.. important::

    Exactly **one** of ``flow_pressure`` and ``flow_temperature`` must be supplied. Providing
    neither — or both — raises a :exc:`ValueError`.

Individual quantities can also be computed without going through :meth:`~triceratops.dynamics.shocks.core.rankine_hugoniot.JumpConditions.solve`
using dedicated class methods:

.. code-block:: python

    # Compression ratio only (no upstream conditions needed)
    R = StrongShockConditions.compute_compression_ratio(gamma=5/3)

    # Post-shock temperature only
    T2 = StrongShockConditions.compute_post_shock_temperature(
        shock_velocity       = 5e8 * u.cm / u.s,
        upstream_density     = 1e-24 * u.g / u.cm**3,
        upstream_temperature = 1e4 * u.K,
    )

The full list of individual methods mirrors the named tuple fields above, with
``compute_post_shock_*`` and ``compute_pre_shock_*`` variants for each quantity.

----

.. _rh_finite_mach:

Finite Mach Number Conditions
------------------------------

When the upstream Mach number :math:`\mathcal{M}_1` is not large enough to justify the strong-shock
approximation, the compression ratio retains its Mach-number dependence:

.. math::

    R(\mathcal{M}_1)
    = \frac{(\gamma + 1)\,\mathcal{M}_1^2}{(\gamma - 1)\,\mathcal{M}_1^2 + 2}.

The classes in this section evaluate the full Rankine-Hugoniot relations without approximation.

.. _rh_weak_cold:

WeakColdShockConditions
^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.WeakColdShockConditions` combines the
finite Mach number treatment with the cold-upstream assumption. Because the cold limit suppresses the
upstream sound speed, the Mach number cannot be inferred from the thermodynamic state and must be
supplied explicitly as ``mach_number``:

.. code-block:: python

    from triceratops.dynamics.shocks.core import WeakColdShockConditions
    import astropy.units as u

    result = WeakColdShockConditions.solve(
        shock_velocity   = 2e8 * u.cm / u.s,
        flow_density     = 1e-24 * u.g / u.cm**3,
        flow_mach_number = 5.0,
    )

    print(result.compression_ratio)      # depends on M_1, not just gamma
    print(result.post_shock_temperature) # still density-independent in cold limit

The post-shock temperature formula matches the strong-cold case, but with
:math:`R = R(\mathcal{M}_1)` rather than the saturated strong-shock value:

.. math::

    T_2 = \frac{\mu m_p}{k_B}\,\frac{R(\mathcal{M}_1) - 1}{R(\mathcal{M}_1)^2}\,u_1^2.

As with :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongColdShockConditions`,
thermodynamic inversion (``upstream=True``) is restricted to kinematic quantities — density,
number density, and velocity — since pre-shock pressure and temperature are undefined in the
cold limit.

.. _rh_weak_hot:

WeakShockConditions
^^^^^^^^^^^^^^^^^^^^

:class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.WeakShockConditions` is the most general
solver. It evaluates the complete Rankine-Hugoniot relations at finite Mach number with a non-negligible
upstream thermal state. The Mach number is computed self-consistently from the shock-frame velocity and
the upstream thermodynamic state:

.. math::

    \mathcal{M}_1 = \frac{v_{\rm sh} - v_1}{c_s},

where the sound speed :math:`c_s` is resolved from whichever of temperature, pressure, or density
is provided:

.. code-block:: python

    from triceratops.dynamics.shocks.core import WeakShockConditions
    import astropy.units as u

    # Sound speed resolved from upstream temperature
    result = WeakShockConditions.solve(
        shock_velocity   = 2e8 * u.cm / u.s,
        flow_density     = 1e-24 * u.g / u.cm**3,
        flow_temperature = 1e6 * u.K,
    )

    # Sound speed resolved from upstream pressure + density
    result = WeakShockConditions.solve(
        shock_velocity = 2e8 * u.cm / u.s,
        flow_density   = 1e-24 * u.g / u.cm**3,
        flow_pressure  = 1e-12 * u.dyn / u.cm**2,
    )

The output fields are identical to :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongShockConditions`.
The downstream Mach number is also available via an individual class method:

.. code-block:: python

    M2 = WeakShockConditions.compute_post_shock_mach_number(
        upstream_mach_number = 5.0,
        gamma = 5/3,
    )

.. note::

    The downstream flow is always subsonic (:math:`\mathcal{M}_2 < 1`) for
    :math:`\mathcal{M}_1 > 1`, which is the condition for a physically valid shock.

----


.. _relativistic_jump_conditions_overview:

Relativistic Jump Conditions
-----------------------------

Classical Rankine--Hugoniot relations assume that the shock and fluid velocities are
small compared with the speed of light. For GRB afterglows, relativistic jets, TDE
outflows, and other high-velocity transients, this approximation breaks down: mass,
momentum, and energy fluxes must be conserved using relativistic velocity addition,
Lorentz factors, and proper-frame thermodynamic variables.

Triceratops provides a parallel hierarchy of relativistic jump-condition solvers for
this regime. These classes compute the downstream shock-frame velocity, compression
ratio, proper post-shock density, pressure, temperature, and energy density from a
lab-frame shock velocity and upstream state.

.. important::

   Relativistic solvers use two different frames:

   * ``shock_velocity`` and ``upstream_velocity`` are **lab-frame velocities**.
   * Density, pressure, temperature, and energy density are **proper comoving-frame
     thermodynamic quantities**.
   * Internally, Triceratops transforms the upstream flow into the shock frame using
     relativistic velocity addition before applying the jump conditions.

Choosing a Relativistic Solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The relativistic solvers are organized by two physical choices:

1. whether the shock is treated in the **ultra-relativistic limit**
   (:math:`\Gamma_1 \gg 1`) or at arbitrary Lorentz factor, and
2. whether the upstream medium is **cold** (:math:`P_1 \simeq 0`) or has a finite
   thermal state.

.. list-table::
   :widths: 38 22 40
   :header-rows: 1

   * - Class
     - Solver type
     - Use when...
   * - :class:`~triceratops.dynamics.shocks.core.relativistic_jump_conditions.UltraRelativisticColdShockConditions`
     - Analytic, cold upstream
     - You want the canonical GRB-style limit: :math:`\Gamma_1 \gg 1`,
       :math:`P_1 = 0`, and :math:`\hat{\gamma}=4/3`, giving
       :math:`\beta_2 = 1/3`.
   * - :class:`~triceratops.dynamics.shocks.core.relativistic_jump_conditions.UltraRelativisticShockConditions`
     - Analytic, warm upstream
     - The shock is ultra-relativistic, but the upstream enthalpy is not negligible.
       The analytic downstream velocity is unchanged, but the reconstructed
       pressure, energy density, and temperature depend on the upstream enthalpy.
   * - :class:`~triceratops.dynamics.shocks.core.relativistic_jump_conditions.RelativisticColdShockConditions`
     - Numerical, cold upstream
     - The upstream medium is cold, but the shock may be only mildly or moderately
       relativistic. This is the safer cold-upstream solver when
       :math:`\Gamma_1 \gg 1` is not guaranteed.
   * - :class:`~triceratops.dynamics.shocks.core.relativistic_jump_conditions.RelativisticShockConditions`
     - Numerical, warm upstream
     - The most general solver: arbitrary relativistic shock speed with a finite
       upstream pressure or temperature.

.. note::

   The ultra-relativistic classes avoid numerical root-finding by using the analytic
   result :math:`\beta_2 = \hat{\gamma} - 1`. The general relativistic classes solve
   the implicit jump condition with a bracketed root finder and then reconstruct the
   full downstream state.

Quick Start: Cold Ultra-Relativistic Shock
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a cold upstream medium in the ultra-relativistic limit, use
:class:`~triceratops.dynamics.shocks.core.relativistic_jump_conditions.UltraRelativisticColdShockConditions`.
This is the simplest relativistic solver and is often the right starting point for
GRB-like blast waves:

.. code-block:: python

   import astropy.units as u
   import astropy.constants as const

   from triceratops.dynamics.shocks.core import UltraRelativisticColdShockConditions

   result = UltraRelativisticColdShockConditions.solve(
       shock_velocity   = 0.999 * const.c,
       upstream_density = 1e-24 * u.g / u.cm**3,
       gamma_hat        = 4 / 3,
       mu               = 0.61,
   )

   print(result.post_shock_beta)          # ≈ 1/3 for gamma_hat = 4/3
   print(result.compression_ratio)
   print(result.post_shock_pressure)
   print(result.post_shock_temperature)

The returned object is a named tuple. Dimensionless quantities such as
``compression_ratio``, ``post_shock_beta``, and ``post_shock_lorentz_factor`` are
returned as floats; thermodynamic quantities are returned with
:class:`~astropy.units.Quantity` units.

General Relativistic Shocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :class:`~triceratops.dynamics.shocks.core.relativistic_jump_conditions.RelativisticShockConditions`
when the upstream medium has a finite pressure or temperature and the shock is not
assumed to be asymptotically ultra-relativistic.

You may provide either ``upstream_pressure`` or ``upstream_temperature``. If a
temperature is supplied, Triceratops computes the upstream pressure using the ideal-gas
relation.

.. code-block:: python

   import astropy.units as u
   import astropy.constants as const

   from triceratops.dynamics.shocks.core import RelativisticShockConditions

   result = RelativisticShockConditions.solve(
       shock_velocity       = 0.8 * const.c,
       upstream_density     = 1e-24 * u.g / u.cm**3,
       upstream_temperature = 1e7 * u.K,
       gamma_hat            = 4 / 3,
       mu                   = 0.61,
   )

   print(result.post_shock_beta)
   print(result.post_shock_density)
   print(result.post_shock_energy_density)

Equivalently, pass a pressure directly:

.. code-block:: python

   result = RelativisticShockConditions.solve(
       shock_velocity    = 0.8 * const.c,
       upstream_density  = 1e-24 * u.g / u.cm**3,
       upstream_pressure = 1e-5 * u.dyn / u.cm**2,
   )

Cold Upstream at Arbitrary Lorentz Factor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :class:`~triceratops.dynamics.shocks.core.relativistic_jump_conditions.RelativisticColdShockConditions`
when the upstream medium is cold but the shock is not necessarily in the
ultra-relativistic limit:

.. code-block:: python

   import astropy.units as u
   import astropy.constants as const

   from triceratops.dynamics.shocks.core import RelativisticColdShockConditions

   result = RelativisticColdShockConditions.solve(
       shock_velocity   = 0.5 * const.c,
       upstream_density = 1e-24 * u.g / u.cm**3,
   )

This class fixes :math:`P_1 = 0`, so no upstream pressure or temperature is required.

Output Fields
^^^^^^^^^^^^^

All relativistic solvers return the same set of post-shock fields:

.. list-table::
   :widths: 34 22 44
   :header-rows: 1

   * - Field
     - Units
     - Meaning
   * - ``compression_ratio``
     - dimensionless
     - Proper density ratio :math:`\rho_2/\rho_1`.
   * - ``post_shock_beta``
     - dimensionless
     - Downstream velocity in the shock frame, :math:`\beta_2 = v_2/c`.
   * - ``post_shock_lorentz_factor``
     - dimensionless
     - Downstream Lorentz factor :math:`\Gamma_2`.
   * - ``post_shock_density``
     - :math:`\mathrm{g\,cm^{-3}}`
     - Proper downstream rest-mass density :math:`\rho_2`.
   * - ``post_shock_number_density``
     - :math:`\mathrm{cm^{-3}}`
     - Proper downstream number density, :math:`n_2 = \rho_2 / (\mu m_p)`.
   * - ``post_shock_pressure``
     - :math:`\mathrm{dyn\,cm^{-2}}`
     - Proper downstream pressure.
   * - ``post_shock_temperature``
     - :math:`\mathrm{K}`
     - Temperature obtained by inverting the Maxwell--Jüttner internal-energy
       relation.
   * - ``post_shock_energy_density``
     - :math:`\mathrm{erg\,cm^{-3}}`
     - Total proper downstream energy density, including rest-mass and internal
       energy.

Implementation Notes
^^^^^^^^^^^^^^^^^^^^

The relativistic solvers share a common reconstruction step. Once the downstream
shock-frame velocity :math:`\beta_2` is known, Triceratops reconstructs the downstream
state from mass flux conservation and relativistic energy-momentum conservation.

For the numerical solvers, :math:`\beta_2` is found by solving the implicit jump
condition on the physical interval :math:`0 < \beta_2 < \beta_1`. For the
ultra-relativistic solvers, the analytic expression

.. math::

   \beta_2 = \hat{\gamma} - 1

is used instead.

.. warning::

   The relativistic jump-condition classes currently provide only the aggregate
   :meth:`solve` interface. They do not implement ``upstream=True`` inverse solves
   or individual ``compute_post_shock_*`` helper methods.

----

.. _rh_two_level_api:

Two-Level API
--------------

Like all physics submodules in Triceratops, the Rankine-Hugoniot module exposes a private
**CGS backend** and a public **unit-aware interface**. Most users interact only with the public
interface, but the private layer is available for performance-critical applications such as
inference hot loops where unit overhead must be minimized.

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Layer
     - Methods
     - Convention
   * - Public
     - ``solve()``, ``compute_post_shock_*()``, ``compute_pre_shock_*()``
     - Accept bare floats (interpreted as CGS) or :class:`~astropy.units.Quantity`; return :class:`~astropy.units.Quantity`
   * - Private
     - ``_solve()``, ``_post_shock_*()``, ``_pre_shock_*()``
     - Accept and return plain Python floats or :class:`~numpy.ndarray` in CGS; no unit handling

.. code-block:: python

    # High-level (public) — recommended for interactive use
    result = StrongColdShockConditions.solve(
        1e9 * u.cm/u.s, 1e-24 * u.g/u.cm**3
    )

    # Low-level (private) — for inference loops
    d = StrongColdShockConditions._solve(
        1e9, 1e-24      # plain CGS floats
    )
    rho2 = d['post_shock_density']   # float in g/cm^3

----

.. _rh_class_hierarchy:

Class Hierarchy
---------------

The solvers are organized in a two-level hierarchy. The abstract base class
:class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.JumpConditions` defines the interface;
concrete classes implement the physical regime:

.. code-block:: text

   JumpConditions  (ABC)
   │   Common interface: solve(), _solve(), OUTPUT_FIELDS, UPSTREAM_OUTPUT_FIELDS
   │
   ├── StrongShockConditions
   │       Strong-shock limit, hot upstream.
   │       Compression ratio: R = (gamma+1)/(gamma-1)
   │
   │   └── StrongColdShockConditions
   │           Strong-shock limit, cold upstream (P_1 = 0).
   │           Simpler signature; temperature is density-independent.
   │
   └── WeakShockConditions
           Finite Mach number, hot upstream.
           Compression ratio: R = (gamma+1)M^2 / [(gamma-1)M^2 + 2]
           Self-consistent Mach number resolution.

       └── WeakColdShockConditions
               Finite Mach number, cold upstream (P_1 = 0).
               Mach number must be supplied explicitly.

Each class auto-generates a named-tuple type (``StrongColdShockConditionsResult``, etc.)
from its :attr:`~triceratops.dynamics.shocks.core.rankine_hugoniot.JumpConditions.OUTPUT_FIELDS`
at class creation time. This means new fields can be added by subclasses without any changes to
the base class.

----

.. _rh_api_reference:

API Reference
-------------

.. currentmodule:: triceratops.dynamics.shocks.core.rankine_hugoniot

.. rubric:: Classical Jump Condition Classes

.. autosummary::
   :nosignatures:

   JumpConditions
   StrongShockConditions
   StrongColdShockConditions
   WeakShockConditions
   WeakColdShockConditions

.. currentmodule:: triceratops.dynamics.shocks.core.relativistic_jump_conditions

.. rubric:: Relativistic Jump Condition Classes

.. autosummary::
   :nosignatures:

   RelativisticJumpConditions
   UltraRelativisticShockConditions
   UltraRelativisticColdShockConditions
   RelativisticShockConditions
   RelativisticColdShockConditions

----

References
----------

.. footbibliography::
