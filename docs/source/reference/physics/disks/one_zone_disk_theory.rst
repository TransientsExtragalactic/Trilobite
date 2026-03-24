.. _one_zone_disk_theory:
=============================================
One-Zone Accretion Disk Models: Theory
=============================================

A central challenge in accretion disk modelling is that the viscous timescale,

.. math::
    :label: eq:viscous_timescale

    t_{\rm visc} = \alpha^{-1} \left(\frac{R}{H}\right)^2 \Omega^{-1},

scales as :math:`t_{\rm visc} \propto R^{3/2}`, so different parts of the disk
evolve on wildly different timescales.  Rather than resolving this hierarchy with
a full radial grid and adaptive timestepping, we follow
:footcite:t:`metzgerTimeDependentModelsAccretion2008` and represent the entire disk
with a **single spatial zone** located at the characteristic radius where the viscous
timescale is comparable to the elapsed time.  This *one-zone* approach reduces the
disk evolution to a coupled two-component ODE system that can be integrated very
efficiently while still capturing global viscous spreading, thermodynamic structure,
and accretion onto the central object.

This document describes the theoretical framework underlying the one-zone disk models
implemented in Triceratops.  For usage instructions see :ref:`one_zone_disk`.

.. contents::
    :local:
    :depth: 2

----

The Core Model
--------------

.. note::

    We follow the development of the one-zone formalism in
    :footcite:t:`metzgerTimeDependentModelsAccretion2008`.

The one-zone approximation partitions the disk into three characteristic regions
based on the ratio :math:`t_{\rm visc}/t`:

- **Inner disk** (:math:`t_{\rm visc} \ll t`) — in quasi-steady state; mass is
  rapidly accreted and this region is dynamically subdominant.
- **Outer disk** (:math:`t_{\rm visc} \gg t`) — has not yet had time to evolve away
  from the initial conditions.
- **Middle disk** (:math:`t_{\rm visc} \sim t`) — currently spreading viscously and
  dominant in mass.

The one-zone model tracks the middle disk, treating the disk mass :math:`M_D` and
angular momentum :math:`J_D` as its only two degrees of freedom.

One-Zone Geometry
^^^^^^^^^^^^^^^^^

Following the Green's function analysis of a viscously spreading ring
:footcite:p:`metzgerTimeDependentModelsAccretion2008`, the area-averaged surface
density and characteristic disk radius are related to the state variables by

.. math::

    M_D = A\, \pi\, \Sigma_D\, R_D^2,
    \qquad
    J_D = \frac{B}{A}\, M_D\, \sqrt{G M_{\rm BH} R_D},

where :math:`A = 1.62` and :math:`B = 1.33` are dimensionless correction constants
calibrated to match the exact spreading-ring solution.  Defining the ratio

.. math::

    \xi \equiv \frac{B}{A} \approx 0.821,

the outer disk radius and surface density follow directly from the state vector:

.. math::
    :label: eq:disk_radius

    R_D = \frac{J_D^2}{\xi^2\, M_D^2\, G M_{\rm BH}},
    \qquad
    \Sigma_D = \frac{M_D}{\pi\, A\, R_D^2}.

These algebraic relations are re-evaluated at every integration timestep and require
no additional free parameters beyond :math:`M_{\rm BH}`.

Evolution Equations
^^^^^^^^^^^^^^^^^^^

The disk mass and angular momentum evolve as

.. math::
    :label: eq:disk_evolution

    \begin{aligned}
    \dot{M}_D &= -\dot{M}_{\rm acc} + \dot{M}_{\rm inflow} - \dot{M}_{\rm outflow}, \\
    \dot{J}_D &= -\dot{J}_{\rm acc} + \dot{J}_{\rm inflow} - \dot{J}_{\rm outflow}.
    \end{aligned}

In the base implementation, only the accretion sink terms are active:

.. math::

    \dot{M}_{\rm acc}
    = f_D\,\frac{M_D}{t_{\rm visc}}
    = f_D\,\nu\,\frac{M_D}{R_D^2},

where :math:`\nu` is the kinematic viscosity and

.. math::
    :label: eq:fd

    f_D = \frac{F_0}{1 - \sqrt{R_{\rm in}/R_D}},
    \qquad F_0 = 1.6,

enforces a zero-torque inner boundary condition (:math:`F_0 = 1.6` is the
Metzger+08 calibration value).  Matter draining through the inner edge at
:math:`R_{\rm in}` carries the local Keplerian specific angular momentum
:math:`\ell_{\rm in} = \sqrt{G M_{\rm BH} R_{\rm in}}`, so

.. math::

    \dot{J}_{\rm acc} = \dot{M}_{\rm acc}\,\sqrt{G M_{\rm BH} R_{\rm in}}.

Because :math:`\ell_{\rm in} \ll J_D / M_D` when :math:`R_{\rm in} \ll R_D`,
angular momentum drains more slowly than mass and the disk outer radius grows over
time — the standard viscous spreading behaviour.

Thermodynamics
--------------

Viscous Heating and Radiative Cooling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The thermal structure of the disk is set by balancing viscous heating
against radiative cooling:

.. math::
    :label: eq:energy_balance

    Q^+(T_c) = Q^-(T_c).

The viscous heating rate per unit area is

.. math::
    :label: eq:q_plus

    Q^+ = \frac{9}{8}\,\nu\,\Sigma\,\Omega^2
        = \frac{9}{8}\,\alpha\,\frac{c_s^2}{\Omega}\,\Sigma\,\Omega^2
        = \frac{9}{8}\,\alpha\,c_s^2\,\Sigma\,\Omega,

and the radiative cooling rate per unit area (for an optically thick disk in the
diffusion limit) is

.. math::
    :label: eq:q_minus

    Q^- = \sigma_{\rm SB}\,T_{\rm eff}^4.

The midplane temperature :math:`T_c` and effective temperature :math:`T_{\rm eff}`
are related through radiative diffusion:

.. math::
    :label: eq:tc_teff

    T_c^4 = \frac{3}{4}\,\tau\,T_{\rm eff}^4,

where the optical depth :math:`\tau = \kappa\,\Sigma` depends on the opacity
prescription.

Energy Balance Residual
^^^^^^^^^^^^^^^^^^^^^^^

Combining :eq:`eq:q_plus`, :eq:`eq:q_minus`, and :eq:`eq:tc_teff` and substituting
:math:`\nu = \alpha c_s^2/\Omega`, the energy balance :eq:`eq:energy_balance`
reduces to a single equation in :math:`T_c`:

.. math::
    :label: eq:root_equation

    \mathcal{F}(T_c)
    \equiv
    \ln Q_0 + 2\ln c_s(T_c) - 4\ln T_c = 0,

where the quantity

.. math::
    :label: eq:Q0

    Q_0 \equiv \frac{27}{32}\,\frac{\alpha\,\kappa\,\Sigma^2\,\Omega}{\sigma_{\rm SB}}

collects all terms that do not depend on :math:`T_c`.  The sound speed
:math:`c_s(T_c)` is determined by the equation of state and depends on the
thermodynamic closure (see below).

.. note::

    All quantities in :eq:`eq:root_equation` are evaluated in log-space
    internally to avoid floating-point underflow for the extreme temperatures
    (:math:`10\,{\rm K} \lesssim T_c \lesssim 10^{14}\,{\rm K}`) that arise
    across the full range of one-zone disk models.

Solution Strategy
^^^^^^^^^^^^^^^^^

The form of :eq:`eq:root_equation` depends on whether :math:`c_s(T_c)` can
be expressed analytically.

.. tab-set::

    .. tab-item:: Gas-Pressure Dominated (Analytic)

        For a pure ideal-gas equation of state,

        .. math::

            c_s^2 = \frac{k_B T_c}{\mu m_p},

        the residual :eq:`eq:root_equation` becomes

        .. math::

            \ln Q_0 + \ln\frac{k_B T_c}{\mu m_p} - 4\ln T_c = 0.

        Rearranging:

        .. math::

            \ln Q_0 + \ln\frac{k_B}{\mu m_p} = 3\ln T_c,

        giving a **closed-form solution**:

        .. math::
            :label: eq:Tc_gas_only

            \boxed{
            T_c = \left(\frac{Q_0\, k_B}{\mu\, m_p}\right)^{1/3}.
            }

        No root-finding is needed; the Cython closure evaluates this formula
        directly at every timestep.

    .. tab-item:: Gas + Radiation Pressure (Implicit)

        When radiation pressure is included, the equation of state is

        .. math::

            P = \frac{\rho k_B T_c}{\mu m_p} + \frac{a T_c^4}{3},

        and :math:`c_s^2 = P/\rho`.  The midplane density is found from vertical
        hydrostatic equilibrium,

        .. math::

            \rho = \frac{\Sigma}{2 H} = \frac{\Sigma\,\Omega}{2\,c_s},

        which gives a quadratic relation between :math:`c_s` and :math:`T_c` for
        fixed :math:`\Sigma` and :math:`\Omega`:

        .. math::
            :label: eq:cs_quadratic

            c_s^2 - \frac{2\,a\,T_c^4}{3\,\Sigma\,\Omega}\,c_s
                  - \frac{k_B T_c}{\mu m_p} = 0.

        The physically meaningful (positive) root is

        .. math::

            c_s(T_c)
            =
            \frac{D + \sqrt{D^2 + 4k_B T_c/(\mu m_p)}}{2},
            \qquad
            D \equiv \frac{2\,a\,T_c^4}{3\,\Sigma\,\Omega}.

        Substituting this into :eq:`eq:root_equation` yields a **transcendental
        equation** in :math:`T_c` that must be solved numerically.  Triceratops
        uses a two-step procedure:

        1. **Bracket expansion** — starting from an initial guess
           :math:`T_c^{\rm guess}` derived from the gas-only formula, the bracket
           :math:`[T_{\rm lo}, T_{\rm hi}]` is expanded geometrically until
           :math:`\mathcal{F}` changes sign.
        2. **Brent's method** — once a bracket is established, Brent's method
           converges to the root with a mix of bisection and inverse quadratic
           interpolation, guaranteeing superlinear convergence.

        Both steps are implemented in Cython and run entirely without the GIL.

        .. important::

            For physical disk parameters there is always exactly one root:
            :math:`\mathcal{F}` is a monotone function of :math:`T_c` because
            :math:`c_s` grows more slowly than :math:`T_c` in both the gas-dominated
            (:math:`c_s \propto T_c^{1/2}`) and radiation-dominated
            (:math:`c_s \propto T_c`) limits.  If no root is found, the integrator
            returns an ``EXPAND_FAIL`` or ``NO_BRACKET`` error (see
            :ref:`one_zone_disk` for the full error code reference).

Viscosity
^^^^^^^^^

Given :math:`T_c` and therefore :math:`c_s`, the kinematic viscosity follows from
the Shakura--Sunyaev :math:`\alpha`-prescription:

.. math::

    \nu = \alpha\,c_s\,H = \frac{\alpha\,c_s^2}{\Omega},

and the disk scale height is

.. math::

    H = \frac{c_s}{\Omega}.

The viscous timescale that sets the integration timestep is

.. math::

    t_{\rm visc} = \frac{R_D^2}{\nu}.

Numerical Architecture
----------------------

.. figure:: ../../../images/theory/disks/DiskDiagram1.drawio

The hot integration loop in Triceratops is a compiled Cython explicit-Euler kernel
(:func:`~triceratops.dynamics.accretion.one_zone._integrator.run_one_zone_model`)
that advances the state vector :math:`(M_D, J_D)` by one step at a time.  For each
step the kernel executes the following operations in strict order:

1. **Geometry** — compute :math:`R_D`, :math:`\Sigma`, :math:`\Omega` from the
   current state using :eq:`eq:disk_radius`.
2. **Closure** — call the thermodynamic closure (a C function pointer injected at
   construction time) to solve :eq:`eq:root_equation` for :math:`T_c` and to
   derive all thermodynamic and viscous quantities:
   :math:`T_{\rm eff}`, :math:`\tau`, :math:`c_s`, :math:`\nu`, :math:`t_{\rm visc}`.
3. **Derivative** — evaluate :math:`(\dot{M}_D, \dot{J}_D)` using the viscosity
   from step 2.
4. **Write** — record all quantities to the result array at the current column.
5. **Euler update** — :math:`M_D \leftarrow M_D + \dot{M}_D \Delta t`,
   :math:`J_D \leftarrow J_D + \dot{J}_D \Delta t`, with the adaptive timestep
   :math:`\Delta t = f_D \cdot t_{\rm visc}`.

The entire loop runs without the GIL, making it safe to call from multi-threaded
contexts.

Closure extensibility is achieved through the :class:`~triceratops.dynamics.accretion.one_zone._integrator.OneZoneClosure`
extension type.  Each concrete model subclass implements
:meth:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase._build_cython_closure`
to return a closure object with three C function pointers installed:
a *closure function* (solves the energy balance), a *derivative function* (evaluates
the ODE right-hand side), and a *writer function* (serialises the full state to the
result array).  The base class and integrator kernel are completely independent of
the specific physics choice.

Implemented Closures
--------------------

Gas Pressure, Electron-Scattering Opacity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest physically self-consistent closure within the one-zone framework
follows :footcite:t:`metzgerTimeDependentModelsAccretion2008` and assumes:

- **Opacity**: pure electron scattering,
  :math:`\kappa = \kappa_{\rm es} = 0.34\,{\rm cm^2\,g^{-1}}`.
- **Pressure**: ideal gas,
  :math:`P = \rho k_B T_c / (\mu m_p)`.

The energy balance :eq:`eq:root_equation` then has the closed-form solution
:eq:`eq:Tc_gas_only`.  The full solution sequence per timestep is:

1. Evaluate :math:`Q_0 = (27/32)\,\alpha\,\kappa_{\rm es}\,\Sigma^2\,\Omega / \sigma_{\rm SB}`.
2. Solve analytically:
   :math:`T_c = (Q_0 k_B / \mu m_p)^{1/3}`.
3. Compute the effective temperature from :eq:`eq:tc_teff`:
   :math:`T_{\rm eff} = (T_c^4 / [(3/4)\kappa_{\rm es}\Sigma])^{1/4}`.
4. Compute the sound speed: :math:`c_s = \sqrt{k_B T_c / (\mu m_p)}`.
5. Compute the viscosity: :math:`\nu = \alpha c_s^2 / \Omega`.

This closure is implemented in
:class:`~triceratops.dynamics.accretion.one_zone.core.gP_esDisk`.

Gas + Radiation Pressure, Electron-Scattering Opacity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This closure extends the gas-only case by including radiation pressure
in the equation of state:

.. math::

    P = P_{\rm gas} + P_{\rm rad}
      = \frac{\rho k_B T_c}{\mu m_p} + \frac{a T_c^4}{3}.

The sound speed :math:`c_s(T_c)` is found from the quadratic :eq:`eq:cs_quadratic`,
and the energy balance :eq:`eq:root_equation` is solved for :math:`T_c` using
bracket expansion followed by Brent's method.  The gas-only analytic formula
provides the initial bracket centre.

In the gas-pressure-dominated limit :math:`(a T_c^4 / 3 \ll \rho k_B T_c / \mu m_p)`,
the two closures agree.  Differences become significant when

.. math::

    \frac{P_{\rm rad}}{P_{\rm gas}} \sim \frac{a T_c^3 \mu m_p}{3 \rho k_B} \gtrsim 1,

which occurs at high temperatures or low densities (high :math:`\alpha`, high
accretion rate, or large disk radius).

This closure is implemented in
:class:`~triceratops.dynamics.accretion.one_zone.core.igP_esDisk`.

----

.. footbibliography::
