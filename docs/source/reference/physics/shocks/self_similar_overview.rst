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
   :widths: 35 60

   * - Engine
     - Best For
   * - :class:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarShockEngine`
     - **Ejecta dominated** supernova evolution with power-law CSM and ejecta.
   * - :class:`~triceratops.dynamics.shocks.chevalier.ChevalierSelfSimilarWindShockEngine`
     - **Ejecta dominated** supernova evolution with wind CSM and power-law ejecta.
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
    discontinuity by a factor of order unity that depends on :math:`n` and :math:`s`; the
    full two-surface solution will be provided in a future update.

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


< write table here >

.. note::

    This engine is not yet implemented.

----

References
----------

.. footbibliography::
