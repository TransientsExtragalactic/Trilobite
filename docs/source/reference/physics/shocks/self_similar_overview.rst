.. _self_similar_shocks_overview:

============================================
Self-Similar Shock Engines
============================================

Self-similar shock engines provide fast, analytic prescriptions for shock evolution when
the ejecta and CSM density profiles take simple power-law forms.  For an overview of all
available shock engines and guidance on choosing between self-similar and numerical
approaches, see :ref:`shock_engines`.

.. contents::
    :local:
    :depth: 1

----

Non-Relativistic Self-Similar Shocks
-------------------------------------

The first major class of self-similar shock engines in Triceratops are non-relativistic, based on the
classical Rankine-Hugoniot jump conditions.  Three engines cover the most common scenarios; the
table below summarises their applicability.

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Engine
     - Best For
   * - :class:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarShockEngine`
     - **Ejecta dominated** supernova evolution; single-surface (contact discontinuity only)
       with analytic thin-shell normalization. Fast, stateless.
   * - :class:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarWindShockEngine`
     - Same as above for a **steady-wind CSM** (:math:`s=2`); accepts :math:`\dot{M}` and :math:`v_w` directly.
   * - :class:`~triceratops.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarEngine`
     - **Two-surface** (forward + reverse shock) Chevalier evolution with power-law CSM.
       Uses a tabulated :math:`(n,s)` grid for accurate normalization and separate
       post-shock conditions at each surface. Recommended when thermodynamics at each
       shock matters.
   * - :class:`~triceratops.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarWindEngine`
     - Same as above for a **steady-wind CSM** (:math:`s=2`).
   * - :class:`~triceratops.dynamics.shocks.sedov_taylor.SedovTaylorShockEngine`
     - **Sedov-Taylor Phase** of SNe, when the shock is driven by thermal pressure in the ejecta. Requires
       a uniform ambient medium.


.. _chevalier_engine:

Chevalier Self-Similar Shock Engines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both Chevalier engines implement the self-similar ejecta--CSM interaction of
:footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`. This should be used during the
**early phase** of supernova evolution when the dynamics are **ejecta dominated**.

The ejecta follow a broken power-law in velocity space,

.. math::

    \rho_{\rm ej}(r,t) = K_{\rm ej}\,t^{-3}
    \begin{cases}
        v^{-\delta}, & v < v_t, \\
        v_t^{n-\delta}\,v^{-n}, & v \ge v_t,
    \end{cases}

where :math:`v_t` and :math:`K_{\rm ej}` are fixed by the total ejecta energy :math:`E_{\rm ej}`
and mass :math:`M_{\rm ej}`.  The CSM follows :math:`\rho_{\rm CSM}(r) = K_{\rm CSM}\,r^{-s}`.
Under these assumptions the shock interface evolves self-similarly,

.. math::

    R(t) = \left(\frac{\zeta\,K_{\rm CSM}}{K_{\rm ej}}\right)^{1/(s-n)} t^{(3-n)/(s-n)},

where :math:`\zeta` is a dimensionless constant of order unity that depends on :math:`n` and
:math:`s`.  For the full derivation see :ref:`chevalier_theory`.

.. hint::

    The :class:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarShockEngine` handles
    arbitrary :math:`s`.  The
    :class:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarWindShockEngine` is a
    convenience specialization for :math:`s = 2`: it accepts :math:`\dot{M}` and :math:`v_w`
    directly and derives :math:`K_{\rm CSM} = \dot{M}/(4\pi v_w)` internally.

Problem Setup
~~~~~~~~~~~~~~~

The ejecta profile is parameterized by its total kinetic energy :math:`E_{\rm ej}`, total mass
:math:`M_{\rm ej}`, outer velocity index :math:`n`, and inner index :math:`\delta`.
Convergence of the self-similar solution requires :math:`n > 5` and :math:`\delta < 3`; in
practice :math:`\delta = 0` or :math:`1` is standard for Type II supernovae.

The CSM is parameterized by its power-law index :math:`s` and the normalization constant
:math:`K_{\rm CSM}`.  The static helper
:meth:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarShockEngine.normalize_csm_density`
converts a reference density :math:`\rho_0` at a reference radius :math:`r_0` into
:math:`K_{\rm CSM} = \rho_0\,r_0^s`.  For the wind engine the normalization is derived
internally from :math:`\dot{M}` and :math:`v_w`, so no explicit :math:`K_{\rm CSM}` is needed.

Both engines are stateless: the same instance can be called repeatedly with different parameters
without re-instantiation.

.. dropdown:: Example — general engine setup

    .. code-block:: python

        from astropy import units as u
        from triceratops.dynamics.shocks import ChevalierSelfSimilarShockEngine

        engine = ChevalierSelfSimilarShockEngine()

        # Ejecta parameters
        E_ej  = 1e51 * u.erg
        M_ej  = 5.0  * u.Msun
        n     = 10.0   # outer ejecta index (must be > 5)
        delta = 1.0    # inner ejecta index (must be < 3)
        s     = 2.0    # CSM power-law index

        # CSM normalization from a reference density at a reference radius
        K_csm = ChevalierSelfSimilarShockEngine.normalize_csm_density(
            rho_0=1e-20 * u.g / u.cm**3,
            r_0=1e16 * u.cm,
            s=s,
        )

.. dropdown:: Example — wind engine setup

    .. code-block:: python

        from astropy import units as u
        from triceratops.dynamics.shocks import ChevalierSelfSimilarWindShockEngine

        engine = ChevalierSelfSimilarWindShockEngine()

        # Ejecta parameters
        E_ej   = 1e51 * u.erg
        M_ej   = 5.0  * u.Msun
        n      = 10.0
        delta  = 1.0

        # Wind parameters — K_csm is derived internally from these
        M_dot  = 1e-5 * u.Msun / u.yr
        v_wind = 100.0 * u.km / u.s


Solving The Shock Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call
:meth:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarShockEngine.compute_shock_properties`
with a time array and the physical parameters.  The method returns a
:class:`~triceratops.dynamics.shocks.chevalier.ChevalierShockState` named tuple with six
:class:`~astropy.units.Quantity` fields:

- ``radius`` — the shock contact-discontinuity radius :math:`R(t)`.
- ``velocity`` — the contact-discontinuity velocity :math:`v_{\rm cd}(t) = \dot{R}(t)`.
- ``post_shock_density`` — immediate post-shock density :math:`\rho_s`.
- ``post_shock_pressure`` — immediate post-shock pressure :math:`p_s`.
- ``post_shock_temperature`` — immediate post-shock temperature :math:`T_s`.
- ``thermal_energy_density`` — post-shock thermal energy density :math:`e_{\rm th} = p_s/(\gamma-1)`.

.. note::

    Post-shock thermodynamics are computed via
    :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongColdShockConditions`
    using :math:`v_{\rm cd}` as the shock velocity, following the standard convention in the
    supernova radio-afterglow literature.  The true forward shock outruns the contact
    discontinuity by a factor :math:`R_{\rm fs}/R_c > 1` that depends on :math:`n` and
    :math:`s`; the :ref:`chevalier_two_shock_engine` below resolves both surfaces separately.

.. dropdown:: Example — radius and velocity evolution (general engine)

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u

        from triceratops.dynamics.shocks import ChevalierSelfSimilarShockEngine
        from triceratops.utils.plot_utils import set_plot_style

        engine = ChevalierSelfSimilarShockEngine()
        time   = np.geomspace(1, 1000, 500) * u.day

        K_csm = ChevalierSelfSimilarShockEngine.normalize_csm_density(
            rho_0=1e-20 * u.g / u.cm**3,
            r_0=1e16 * u.cm,
            s=2.0,
        )

        state = engine.compute_shock_properties(
            time=time,
            E_ej=1e51 * u.erg,
            M_ej=5.0 * u.Msun,
            K_csm=K_csm,
            n=10.0,
            s=2.0,
            delta=1.0,
        )

        set_plot_style()
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        axes[0].loglog(time.to_value(u.day), state.radius.to_value(u.cm))
        axes[0].set_ylabel("Radius (cm)")

        axes[1].loglog(time.to_value(u.day), state.velocity.to_value(u.km / u.s))
        axes[1].set_ylabel(r"Velocity (km s$^{-1}$)")
        axes[1].set_xlabel("Time (days)")

        plt.tight_layout()
        plt.show()

.. dropdown:: Example — radius and velocity evolution (wind engine)

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u

        from triceratops.dynamics.shocks import ChevalierSelfSimilarWindShockEngine
        from triceratops.utils.plot_utils import set_plot_style

        engine = ChevalierSelfSimilarWindShockEngine()
        time   = np.geomspace(1, 1000, 500) * u.day

        state = engine.compute_shock_properties(
            time=time,
            E_ej=1e51 * u.erg,
            M_ej=5.0 * u.Msun,
            M_dot=1e-5 * u.Msun / u.yr,
            v_wind=100.0 * u.km / u.s,
            n=10.0,
            delta=1.0,
        )

        set_plot_style()
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        axes[0].loglog(time.to_value(u.day), state.radius.to_value(u.cm))
        axes[0].set_ylabel("Radius (cm)")

        axes[1].loglog(time.to_value(u.day), state.velocity.to_value(u.km / u.s))
        axes[1].set_ylabel(r"Velocity (km s$^{-1}$)")
        axes[1].set_xlabel("Time (days)")

        plt.tight_layout()
        plt.show()

----

.. _chevalier_two_shock_engine:

Chevalier Two-Shock Engines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two-shock engines resolve the **forward and reverse shock surfaces separately**,
rather than treating the shocked interaction region as a single thin shell. This
gives accurate post-shock thermodynamics at each surface and uses the tabulated
normalization constant :math:`A` from the full ODE solution instead of the thin-shell
approximation :math:`\zeta`.

At instantiation both engines call
:func:`~triceratops.dynamics.shocks.chevalier.compute_self_similar_critical_grid`
to precompute a table of the three dimensionless self-similar constants

.. math::

   A, \qquad \frac{R_{\rm fs}}{R_c}, \qquad \frac{R_{\rm rs}}{R_c},

over a grid of :math:`(n,\,s)` values. At runtime these constants are bilinearly
interpolated for the requested :math:`(n, s)`, so repeated evaluations have
negligible cost after the initial build. For the theoretical background on the ODE
system, boundary conditions, and pressure matching that fix :math:`A` and the radius
ratios, see :ref:`chevalier_theory`.

.. hint::

   :class:`~triceratops.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarEngine`
   handles arbitrary :math:`s`. The
   :class:`~triceratops.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarWindEngine`
   is the :math:`s=2` specialization that accepts :math:`\dot{M}` and :math:`v_w` directly.

.. important::

   Unlike the single-surface engines, the two-shock engines are **stateful**: the
   :math:`(n,s)` table is built at instantiation. Building the default 9 × 6 grid
   takes a few seconds on a modern machine. Once constructed, the same instance can
   be evaluated at any :math:`(n, s)` within the tabulated range with negligible
   overhead.  The ``n_grid`` and ``s_grid`` constructor arguments allow the grid to
   be customised; requesting :math:`(n, s)` outside the grid raises a
   :class:`ValueError`.

Problem Setup
~~~~~~~~~~~~~~~

The physical inputs are identical to those of
:class:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarShockEngine`.

.. dropdown:: Example — two-shock general engine

   .. code-block:: python

      from astropy import units as u
      from triceratops.dynamics.shocks import ChevalierTwoShockSelfSimilarEngine

      # Table is built once here
      engine = ChevalierTwoShockSelfSimilarEngine()

      E_ej  = 1e51 * u.erg
      M_ej  = 5.0  * u.Msun
      n     = 10.0   # outer ejecta index (must be > 5)
      delta = 1.0    # inner ejecta index (must be < 3)
      s     = 2.0    # CSM power-law index

      K_csm = (5e-16 * u.g / u.cm**3) * (1e15 * u.cm)**2  # K_CSM for s=2

.. dropdown:: Example — two-shock wind engine

   .. code-block:: python

      from astropy import units as u
      from triceratops.dynamics.shocks import ChevalierTwoShockSelfSimilarWindEngine

      engine = ChevalierTwoShockSelfSimilarWindEngine()

      E_ej   = 1e51 * u.erg
      M_ej   = 5.0  * u.Msun
      n      = 10.0
      delta  = 1.0

      # K_csm is derived internally from these
      M_dot  = 1e-5 * u.Msun / u.yr
      v_wind = 100.0 * u.km / u.s

Solving The Shock Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call
:meth:`~triceratops.dynamics.shocks.chevalier.ChevalierTwoShockSelfSimilarEngine.compute_shock_properties`
with a time array and the physical parameters. The method returns a
:class:`~triceratops.dynamics.shocks.chevalier.ChevalierTwoShockState` named tuple with
fourteen :class:`~astropy.units.Quantity` fields — six kinematic and eight thermodynamic:

- ``radius_cd``, ``radius_fs``, ``radius_rs`` — positions of the contact discontinuity,
  forward shock, and reverse shock.
- ``velocity_cd``, ``velocity_fs``, ``velocity_rs`` — lab-frame velocities
  :math:`v = m\,R/t` for each surface, where :math:`m = (n-3)/(n-s)`.
- ``density_fs``, ``density_rs`` — immediate post-shock densities.
- ``pressure_fs``, ``pressure_rs`` — immediate post-shock pressures.
- ``temperature_fs``, ``temperature_rs`` — immediate post-shock temperatures.
- ``thermal_energy_density_fs``, ``thermal_energy_density_rs`` — post-shock thermal energy
  densities :math:`e_{\rm th} = p/(\gamma-1)`.

.. note::

   At the **forward shock**, the Rankine--Hugoniot conditions are applied using the
   lab-frame shock speed :math:`v_{\rm fs}` against stationary CSM at density
   :math:`K_{\rm CSM}\,R_{\rm fs}^{-s}`.

   At the **reverse shock**, they are applied in the shock frame using the ejecta
   relative velocity :math:`v_{\rm rel} = (1-m)\,R_{\rm rs}/t` against upstream ejecta at
   density :math:`K_{\rm ej}\,t^{n-3}\,R_{\rm rs}^{-n}`.  Because the ejecta are denser and
   slower than the surrounding CSM at comparable radii, the reverse shock is typically
   cooler than the forward shock by one to two orders of magnitude.

.. dropdown:: Example — two-shock kinematics and temperatures

   .. plot::
      :include-source:

      import numpy as np
      import matplotlib.pyplot as plt
      from astropy import units as u

      from triceratops.dynamics.shocks import ChevalierTwoShockSelfSimilarEngine
      from triceratops.utils.plot_utils import set_plot_style

      engine = ChevalierTwoShockSelfSimilarEngine()
      time   = np.geomspace(1, 1000, 300) * u.day

      K_csm = (5e-16 * u.g / u.cm**3) * (1e15 * u.cm)**2

      state = engine.compute_shock_properties(
          time=time,
          E_ej=1e51 * u.erg,
          M_ej=5.0 * u.Msun,
          K_csm=K_csm,
          n=10.0,
          s=2.0,
          delta=1.0,
      )

      set_plot_style()
      fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
      t_day = time.to_value(u.day)

      axes[0].loglog(t_day, state.radius_fs.to_value(u.cm), label=r"$R_{\rm fs}$")
      axes[0].loglog(t_day, state.radius_cd.to_value(u.cm), ls="--", label=r"$R_{\rm cd}$")
      axes[0].loglog(t_day, state.radius_rs.to_value(u.cm), label=r"$R_{\rm rs}$")
      axes[0].set_ylabel("Radius (cm)")
      axes[0].legend()

      axes[1].loglog(t_day, state.velocity_fs.to_value(u.km / u.s), label=r"$v_{\rm fs}$")
      axes[1].loglog(t_day, state.velocity_cd.to_value(u.km / u.s), ls="--", label=r"$v_{\rm cd}$")
      axes[1].loglog(t_day, state.velocity_rs.to_value(u.km / u.s), label=r"$v_{\rm rs}$")
      axes[1].set_ylabel(r"Velocity (km s$^{-1}$)")
      axes[1].legend()

      axes[2].loglog(t_day, state.temperature_fs.to_value(u.K), label=r"$T_{\rm fs}$")
      axes[2].loglog(t_day, state.temperature_rs.to_value(u.K), label=r"$T_{\rm rs}$")
      axes[2].set_ylabel("Post-shock temperature (K)")
      axes[2].set_xlabel("Time (days)")
      axes[2].legend()

      plt.tight_layout()
      plt.show()

----

.. _sedov_taylor_engine:

Sedov--Taylor Shock Engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~triceratops.dynamics.shocks.sedov_taylor.SedovTaylorShockEngine` implements the
classic point-explosion blast wave in a uniform ambient medium
:footcite:p:`sedov1946propagation` :footcite:p:`taylor1950formation`.  The shock radius and
velocity evolve as

.. math::

    R_s(t) = \beta(\gamma)\left(\frac{E\,t^2}{\rho_0}\right)^{1/5}, \qquad
    v_s(t) = \frac{2}{5}\frac{R_s(t)}{t},

where :math:`\beta(\gamma)` is computed once at instantiation by numerically integrating the
self-similar Sedov profiles.  In addition to the shock kinematics, the engine returns
immediate post-shock thermodynamic quantities from the strong-shock Rankine--Hugoniot
conditions,

.. math::

    \rho_s = \frac{\gamma+1}{\gamma-1}\,\rho_0, \qquad
    p_s    = \frac{2}{\gamma+1}\,\rho_0\,v_s^2, \qquad
    T_s    = \frac{p_s\,\mu\,m_p}{\rho_s\,k_B}, \qquad
    e_{\rm th} = \frac{p_s}{\gamma - 1}.

For the theoretical background see :ref:`sedov_taylor_theory`.

Problem Setup
~~~~~~~~~~~~~~~

Unlike the Chevalier engines, the gas parameters :math:`\gamma` (adiabatic index) and
:math:`\mu` (mean molecular weight in units of the proton mass) are fixed at instantiation
because :math:`\beta(\gamma)` is evaluated by numerical quadrature at that point.  The
explosion energy :math:`E` and ambient density :math:`\rho_0` are supplied at call time.

Common choices are :math:`\gamma = 5/3` (monatomic ideal gas, the default) and
:math:`\gamma = 4/3` (radiation-dominated).  The default :math:`\mu = 0.5` is appropriate for
a fully ionized hydrogen plasma.

.. dropdown:: Example — instantiation

    .. code-block:: python

        from triceratops.dynamics.shocks import SedovTaylorShockEngine

        # Default: monatomic ideal gas, fully ionized hydrogen
        engine = SedovTaylorShockEngine()

        # Radiation-dominated gas
        engine_rad = SedovTaylorShockEngine(gamma=4.0 / 3.0, mu=0.5)


Solving The Shock Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call
:meth:`~triceratops.dynamics.shocks.sedov_taylor.SedovTaylorShockEngine.compute_shock_properties`
with a time array, an explosion energy, and an ambient density.  The method returns a
:class:`~triceratops.dynamics.shocks.sedov_taylor.SedovTaylorShockState` named tuple with six
:class:`~astropy.units.Quantity` fields:

- ``radius`` — shock radius :math:`R_s(t)`.
- ``velocity`` — shock velocity :math:`v_s(t)`.
- ``post_shock_density`` — immediate post-shock density :math:`\rho_s`.
- ``post_shock_pressure`` — immediate post-shock pressure :math:`p_s`.
- ``post_shock_temperature`` — immediate post-shock temperature :math:`T_s`.
- ``thermal_energy_density`` — post-shock thermal energy density :math:`e_{\rm th}`.

.. dropdown:: Example — kinematics and post-shock temperature

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u

        from triceratops.dynamics.shocks import SedovTaylorShockEngine
        from triceratops.utils.plot_utils import set_plot_style

        engine = SedovTaylorShockEngine()
        time   = np.geomspace(100, 1e6, 500) * u.yr

        state = engine.compute_shock_properties(
            time=time,
            E=1e51 * u.erg,
            rho_0=1.67e-24 * u.g / u.cm**3,   # approximately 1 H atom per cm^3
        )

        set_plot_style()
        fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

        t_yr = time.to_value(u.yr)

        axes[0].loglog(t_yr, state.radius.to_value(u.pc))
        axes[0].set_ylabel("Radius (pc)")

        axes[1].loglog(t_yr, state.velocity.to_value(u.km / u.s))
        axes[1].set_ylabel(r"Velocity (km s$^{-1}$)")

        axes[2].loglog(t_yr, state.post_shock_temperature.to_value(u.K))
        axes[2].set_ylabel("Post-shock temperature (K)")
        axes[2].set_xlabel("Time (yr)")

        plt.tight_layout()
        plt.show()

Weaver Wind Shock Engine
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    This engine is not yet implemented.



----

Relativistic Self-Similar Shocks
-------------------------------------

In addition to the non-relativistic shock engines available in Triceratops, we also provide a number of fully
relativistic self-similar shock engines based on the various known solutions to ultra-relativistic flows. These
are very useful for things like GRB modeling, jets, etc.

.. _blandford_mckee_engine:

Blandford--McKee Shock Engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Blandford--McKee (BM) engine implements the classic ultra-relativistic
self-similar blast-wave solution of :footcite:t:`1976PhFl...19.1130B` for a
relativistic explosion of isotropic-equivalent energy :math:`E` expanding into an
external medium with a power-law density profile

.. math::

    \rho_{\rm ext}(r) = K r^{-k}.

This is the natural starting point for GRB afterglow modeling and any other scenario
in which the blast wave is ultra-relativistic (:math:`\Gamma \gg 1`).

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Engine
     - Best For
   * - :class:`~triceratops.dynamics.shocks.blandford_mckee.BlandfordMcKeeShockEngine`
     - Ultra-relativistic blast wave in a **general power-law** external medium
       (:math:`k < 3`). All analytic; no ODE integration required.
   * - :class:`~triceratops.dynamics.shocks.blandford_mckee.BlandfordMcKeeWindShockEngine`
     - Same as above for a **stellar-wind** external medium (:math:`k = 2`).
       Accepts :math:`\dot{M}` and :math:`v_w` directly rather than :math:`K`.

The core kinematic relation follows from integrating the BM self-similar energy
density over the shocked shell.  The closed-form profiles

.. math::

    g(\chi) = \chi^{-1},
    \qquad
    f(\chi) = \chi^{(4k-17)/[3(4-k)]},
    \qquad
    h(\chi) = \chi^{(2k-7)/(4-k)},

where :math:`\chi = 1 + 2(m+1)\Gamma^2(1-r/R)` and :math:`m = 3-k`, yield the
analytic normalization coefficient

.. math::

    C_E(k) = \frac{8\pi}{17 - 4k},

so that energy conservation gives the shock Lorentz factor as a function of radius:

.. math::

    \Gamma^2(R) = \frac{(17-4k)\,E}{8\pi K c^2 R^{3-k}}.

In the ultra-relativistic limit :math:`R \approx ct`, so

.. math::

    \Gamma(t) \propto t^{-(3-k)/2}.

For the full derivation see :ref:`blandford_mckee_theory`.

.. hint::

    :class:`~triceratops.dynamics.shocks.blandford_mckee.BlandfordMcKeeShockEngine`
    handles any :math:`k < 3`.  The
    :class:`~triceratops.dynamics.shocks.blandford_mckee.BlandfordMcKeeWindShockEngine`
    fixes :math:`k = 2` and accepts :math:`\dot{M}` and :math:`v_w` directly,
    computing :math:`K = \dot{M}/(4\pi v_w)` internally.

Problem Setup
~~~~~~~~~~~~~~~

The two physical inputs are the isotropic-equivalent explosion energy :math:`E` and
the external medium normalization :math:`K`.  The static helper
:meth:`~triceratops.dynamics.shocks.blandford_mckee.BlandfordMcKeeShockEngine.normalize_csm_density`
converts a reference density :math:`\rho_0` at radius :math:`r_0` into
:math:`K = \rho_0\,r_0^k`.  For the wind engine, :math:`K` is computed internally from
:math:`\dot{M}` and :math:`v_w`.

Two constructor parameters control numerical behavior rather than physics:

- ``lorentz_warn_threshold`` (default 2.0) — a :class:`RuntimeWarning` is emitted
  whenever any :math:`\Gamma(t)` drops below this value, indicating that the
  ultra-relativistic approximation is breaking down.
- ``gamma_hat`` (default :math:`4/3`) — the adiabatic index used for post-shock
  thermodynamics.

A :class:`ValueError` is raised if :math:`k \geq 3`.

.. dropdown:: Example — general engine setup

    .. code-block:: python

        from astropy import units as u
        from triceratops.dynamics.shocks import BlandfordMcKeeShockEngine

        engine = BlandfordMcKeeShockEngine()

        E = 1e52 * u.erg       # isotropic-equivalent explosion energy
        k = 0.0                # uniform ISM

        # CSM normalization from a reference density
        K = BlandfordMcKeeShockEngine.normalize_csm_density(
            rho_0=1.67e-24 * u.g / u.cm**3,   # 1 proton per cm^3
            r_0=1.0 * u.cm,
            k=k,
        )

.. dropdown:: Example — wind engine setup

    .. code-block:: python

        from astropy import units as u
        from triceratops.dynamics.shocks import BlandfordMcKeeWindShockEngine

        engine = BlandfordMcKeeWindShockEngine()

        E      = 1e52 * u.erg
        M_dot  = 1e-5 * u.Msun / u.yr
        v_wind = 1000.0 * u.km / u.s   # K derived internally as M_dot/(4 pi v_wind)


Solving The Shock Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call
:meth:`~triceratops.dynamics.shocks.blandford_mckee.BlandfordMcKeeShockEngine.compute_shock_properties`
with a time array and the physical parameters.  The method returns a
:class:`~triceratops.dynamics.shocks.blandford_mckee.BlandfordMcKeeShockState` named tuple
with eleven fields.  Dimensionless fields are returned as bare arrays; all fields with
physical dimensions carry :class:`~astropy.units.Quantity` units.

**Shock kinematics:**

- ``radius`` — shock radius :math:`R(t) \approx ct`.
- ``lorentz_factor`` — shock Lorentz factor :math:`\Gamma(t)`.
- ``beta`` — shock speed in units of :math:`c`, :math:`\beta = v/c`.
- ``velocity`` — shock speed :math:`v = \beta c`.
- ``fluid_lorentz_factor`` — lab-frame Lorentz factor of the immediately post-shock
  fluid, :math:`\gamma_2 = \Gamma/\sqrt{2}` (from the BM profile :math:`g(\chi=1)=1`).

**Post-shock proper (comoving-frame) quantities:**

- ``post_shock_pressure`` — proper pressure :math:`p_2 = \tfrac{2}{3}\rho_{\rm ext}c^2\Gamma^2`.
- ``post_shock_temperature`` — proper temperature :math:`T_2` from the
  Maxwell--Jüttner internal-energy inversion.
- ``post_shock_comoving_density`` — proper rest-mass density :math:`\rho_2`.

**Lab-frame quantities:**

- ``post_shock_lab_density`` — lab-frame rest-mass density :math:`\gamma_2\rho_2`.
- ``thermal_energy_density_comoving`` — proper internal (thermal) energy density
  :math:`U_{\rm int} = p_2/(\hat\gamma-1)`.
- ``thermal_energy_density_lab`` — lab-frame energy density
  :math:`T^{00} = (e_2+p_2)\gamma_2^2-p_2`.

.. note::

    All post-shock quantities are computed analytically from :math:`\Gamma` using the
    BM profile values at the shock surface (:math:`\chi=1`), bypassing the relativistic
    Rankine--Hugoniot root-finder.  This makes every field exact to floating-point
    precision for any :math:`\Gamma`.

.. dropdown:: Example — Lorentz factor and radius evolution

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u

        from triceratops.dynamics.shocks import (
            BlandfordMcKeeShockEngine,
            BlandfordMcKeeWindShockEngine,
        )
        from triceratops.utils.plot_utils import set_plot_style

        set_plot_style()
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        time = np.geomspace(0.1, 300, 400) * u.day
        E    = 1e52 * u.erg

        # --- ISM (k = 0) ---
        ism = BlandfordMcKeeShockEngine()
        K_ism = BlandfordMcKeeShockEngine.normalize_csm_density(
            rho_0=1.67e-24 * u.g / u.cm**3,
            r_0=1.0 * u.cm,
            k=0.0,
        )
        s_ism = ism.compute_shock_properties(time=time, E=E, K_csm=K_ism, k=0.0)

        # --- Wind (k = 2) ---
        wind = BlandfordMcKeeWindShockEngine()
        s_wind = wind.compute_shock_properties(
            time=time, E=E,
            M_dot=1e-5 * u.Msun / u.yr,
            v_wind=1000.0 * u.km / u.s,
        )

        t_day = time.to_value(u.day)

        axes[0].loglog(t_day, s_ism.lorentz_factor,  label=r"ISM ($k=0$)")
        axes[0].loglog(t_day, s_wind.lorentz_factor, label=r"Wind ($k=2$)", ls="--")
        axes[0].axhline(2, color="gray", lw=0.8, ls=":", label=r"$\Gamma = 2$")
        axes[0].set_ylabel(r"Shock Lorentz factor $\Gamma$")
        axes[0].legend()

        axes[1].loglog(t_day, s_ism.radius.to_value(u.cm),  label=r"ISM ($k=0$)")
        axes[1].loglog(t_day, s_wind.radius.to_value(u.cm), label=r"Wind ($k=2$)", ls="--")
        axes[1].set_ylabel("Radius (cm)")
        axes[1].set_xlabel("Time (days)")

        plt.tight_layout()
        plt.show()

----

References
----------

.. footbibliography::
