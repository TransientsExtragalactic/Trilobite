.. _numeric_shocks_theory:

==========================================
Theory: Numerical Shock Engines
==========================================

In many transient applications, the shock evolution is not well approximated by
a self-similar solution. Self-similar models are extremely useful when the
ejecta and circumstellar medium (CSM) have simple power-law structures, but real
systems may have arbitrary ejecta profiles, structured winds, finite shells,
density jumps, or other environmental features.

The numerical shock engines evolve
reduced, low-dimensional models for the shock dynamics by integrating a small set
of ordinary differential equations. They do not resolve the full hydrodynamic
structure of the shocked gas; instead, they replace the shocked interaction region
by a compact set of global variables whose evolution is governed by conservation
laws and closure assumptions.

This page develops the theory behind three non-relativistic shock models:

#. the :ref:`conservative snowplow model <conservative_snowplow_model>`
   (:class:`~triceratops.dynamics.shocks.numerical.MomentumConservingShockEngine`);
#. the :ref:`pressure-driven thin-shell model <pressure_driven_thin_shell_model>`
   (:class:`~triceratops.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`);
#. the :ref:`mechanical internal-energy model <mechanical_internal_energy_model>`
   (:class:`~triceratops.dynamics.shocks.numerical.MechanicalShockEngine`).

Each model makes different assumptions about the relevant physics and is
applicable in different regimes.

.. contents::
    :local:
    :depth: 3


Overview
--------

While each of the closures described in this document differ in their physics, assumptions,
and implementations, they share a common derivation starting point: conservation
laws integrated over a moving spherical control volume. Below we establish the
geometry and derive the mass, momentum, and energy equations that serve as the
common foundation for all three models.

Geometry
^^^^^^^^^

In establishing these models, we treat the typical geometry of a shock driven by
an expanding spherical outflow into an ambient medium.
We use the standard four-region notation for a supernova--CSM interaction:

.. math::

    \begin{aligned}
        1 &: \text{unshocked ejecta},\\
        2 &: \text{shocked ejecta},\\
        3 &: \text{shocked CSM},\\
        4 &: \text{unshocked CSM}.
    \end{aligned}

The reverse shock, contact discontinuity, and forward shock are located at

.. math::

    R_{\rm rs}(t)
    <
    R_{\rm cd}(t)
    <
    R_{\rm fs}(t),

with speeds

.. math::

    D_{\rm rs} = \dot R_{\rm rs},
    \qquad
    v_{\rm cd} = \dot R_{\rm cd},
    \qquad
    D_{\rm fs} = \dot R_{\rm fs}.

The upstream ejecta and CSM are prescribed by

.. math::

    \rho_1(r,t),\qquad u_1(r,t),
    \qquad
    \rho_4(r,t),\qquad u_4(r,t).

Unless stated otherwise, the upstream media are assumed to be cold,

.. math::

    P_1 \simeq 0,
    \qquad
    P_4 \simeq 0.

.. hint::

    For homologously expanding ejecta,

    .. math::

        u_1(r,t) = \frac{r}{t},

    and the ejecta density can be written as

    .. math::

        \rho_1(r,t)
        =
        t^{-3}G\!\left(\frac{r}{t}\right).

    Equivalently, if the ejecta mass distribution in velocity space is

    .. math::

        \frac{dM_{\rm ej}}{dv}=f(v),

    then

    .. math::

        G(v)
        =
        \frac{f(v)}{4\pi v^2}.


Equations of Motion
^^^^^^^^^^^^^^^^^^^^

The Conservation Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuity equation in spherical symmetry is

.. math::

    \frac{\partial\rho}{\partial t}
    +
    \frac{1}{r^2}
    \frac{\partial}{\partial r}
    \left(r^2\rho u\right)
    =
    0.

Define the mass in the moving control volume :math:`a(t)<r<b(t)` as

.. math::

    M_{[a,b]}
    =
    4\pi
    \int_{a(t)}^{b(t)}
    \rho r^2\,dr.

Multiplying the continuity equation by :math:`4\pi r^2` and integrating over
:math:`[a,b]` using the Leibniz rule gives the moving-control-volume mass equation,

.. math::

    \boxed{
    \frac{dM_{[a,b]}}{dt}
    =
    4\pi a^2
    \left[
        \rho(u-\dot a)
    \right]_{r=a}
    -
    4\pi b^2
    \left[
        \rho(u-\dot b)
    \right]_{r=b}.
    }

Each term represents the net mass flux through a moving boundary: mass entering
from the inner face at :math:`r=a` with relative inflow velocity :math:`u-\dot a`,
and leaving through the outer face at :math:`r=b` with relative outflow
velocity :math:`u-\dot b`.

.. dropdown:: Derivation: moving-boundary mass equation

    Multiplying the continuity equation by :math:`4\pi r^2`,

    .. math::

        \frac{\partial}{\partial t}\left(4\pi r^2\rho\right)
        +
        \frac{\partial}{\partial r}\left(4\pi r^2\rho u\right)
        =
        0.

    Integrating over :math:`a(t)<r<b(t)`,

    .. math::

        \int_a^b
        \frac{\partial}{\partial t}
        \left(4\pi r^2\rho\right)\,dr
        +
        \left[
            4\pi r^2\rho u
        \right]_a^b
        =
        0.

    By the Leibniz rule,

    .. math::

        \frac{d}{dt}
        \int_a^b 4\pi r^2\rho\,dr
        =
        \int_a^b
        \frac{\partial}{\partial t}
        \left(4\pi r^2\rho\right)\,dr
        +
        4\pi b^2\rho|_b\,\dot b
        -
        4\pi a^2\rho|_a\,\dot a.

    Substituting and rearranging gives

    .. math::

        \frac{dM_{[a,b]}}{dt}
        =
        4\pi a^2
        \left[
            \rho(u-\dot a)
        \right]_{r=a}
        -
        4\pi b^2
        \left[
            \rho(u-\dot b)
        \right]_{r=b}.


The Momentum Equation
~~~~~~~~~~~~~~~~~~~~~~

The non-relativistic radial momentum equation in spherical symmetry is

.. math::

    \frac{\partial(\rho u)}{\partial t}
    +
    \frac{1}{r^2}
    \frac{\partial}{\partial r}
    \left[
        r^2(\rho u^2+P)
    \right]
    =
    \frac{2P}{r}.

Define the momentum in the control volume :math:`a(t)<r<b(t)` as

.. math::

    \Pi_{[a,b]}
    =
    4\pi
    \int_{a(t)}^{b(t)}
    \rho u r^2\,dr.

The moving-control-volume momentum equation is

.. math::

    \boxed{
    \begin{aligned}
    \frac{d\Pi_{[a,b]}}{dt}
    &=
    4\pi a^2
    \left[
        \rho u(u-\dot a)+P
    \right]_{r=a}
    -
    4\pi b^2
    \left[
        \rho u(u-\dot b)+P
    \right]_{r=b}
    \\[4pt]
    &\qquad
    +
    8\pi
    \int_a^b
    P r\,dr.
    \end{aligned}
    }

The first two terms are boundary momentum fluxes plus boundary pressure forces;
the last term is the spherical-geometry pressure contribution that
vanishes in the plane-parallel limit.

.. dropdown:: Derivation: moving-boundary momentum equation

    Multiplying the momentum equation by :math:`4\pi r^2`,

    .. math::

        \frac{\partial}{\partial t}
        \left(4\pi r^2\rho u\right)
        +
        \frac{\partial}{\partial r}
        \left[
            4\pi r^2(\rho u^2+P)
        \right]
        =
        8\pi rP.

    Integrating over :math:`a(t)<r<b(t)`,

    .. math::

        \int_a^b
        \frac{\partial}{\partial t}
        \left(4\pi r^2\rho u\right)\,dr
        +
        \left[
            4\pi r^2(\rho u^2+P)
        \right]_a^b
        =
        8\pi\int_a^b Pr\,dr.

    By the Leibniz rule,

    .. math::

        \frac{d}{dt}
        \int_a^b 4\pi r^2\rho u\,dr
        =
        \int_a^b
        \frac{\partial}{\partial t}
        \left(4\pi r^2\rho u\right)\,dr
        +
        4\pi b^2\rho u|_b\,\dot b
        -
        4\pi a^2\rho u|_a\,\dot a.

    Rearranging gives

    .. math::

        \frac{d\Pi_{[a,b]}}{dt}
        =
        4\pi a^2
        \left[
            \rho u(u-\dot a)+P
        \right]_a
        -
        4\pi b^2
        \left[
            \rho u(u-\dot b)+P
        \right]_b
        +
        8\pi
        \int_a^b Pr\,dr.


The Energy Equation
~~~~~~~~~~~~~~~~~~~~

Define the total energy density as

.. math::

    E = \frac{1}{2}\rho u^2 + \rho\epsilon,

where :math:`\epsilon` is the specific internal energy. In inviscid spherical flow
with a volumetric source term :math:`\dot{q}` (e.g., radiative cooling),

.. math::

    \frac{\partial E}{\partial t}
    +
    \frac{1}{r^2}
    \frac{\partial}{\partial r}
    \left[r^2(E+P)u\right]
    =
    \dot{q}.

Define the total energy in the control volume :math:`a(t)<r<b(t)` as

.. math::

    \mathcal{E}_{[a,b]}
    =
    4\pi\int_{a(t)}^{b(t)}
    E r^2\,dr.

The moving-control-volume energy equation is

.. math::

    \boxed{
    \frac{d\mathcal{E}_{[a,b]}}{dt}
    =
    4\pi a^2
    \left[
        (E+P)(u-\dot a)
    \right]_{r=a}
    -
    4\pi b^2
    \left[
        (E+P)(u-\dot b)
    \right]_{r=b}
    +
    \mathcal{Q}_{[a,b]},
    }

where :math:`\mathcal{Q}_{[a,b]} = 4\pi\int_a^b \dot{q}\,r^2\,dr` is the net
volumetric source integrated over the control volume. The energy flux through each
boundary involves the enthalpy :math:`E + P` rather than the internal energy alone,
reflecting the :math:`P\,dV` work done as mass crosses each face.

.. note::

    The three numerical models treat the energy equation differently:

    - The :ref:`conservative snowplow model <conservative_snowplow_model>` discards
      the energy equation entirely; shock-generated thermal energy is assumed to be
      radiated away immediately and no internal energy is retained.
    - The :ref:`pressure-driven thin-shell model <pressure_driven_thin_shell_model>`
      replaces the energy equation with algebraic Rankine--Hugoniot expressions for
      the post-shock pressures; no internal energy is evolved.
    - The :ref:`mechanical model <mechanical_internal_energy_model>` evolves separate
      internal energies :math:`U_2` and :math:`U_3` for the shocked ejecta and
      shocked CSM as independent ODEs.

.. dropdown:: Derivation: moving-boundary energy equation

    Multiplying the energy equation by :math:`4\pi r^2`,

    .. math::

        \frac{\partial}{\partial t}\left(4\pi r^2 E\right)
        +
        \frac{\partial}{\partial r}\left[4\pi r^2(E+P)u\right]
        =
        4\pi r^2\dot{q}.

    Integrating over :math:`a(t)<r<b(t)`,

    .. math::

        \int_a^b
        \frac{\partial}{\partial t}\left(4\pi r^2 E\right)\,dr
        +
        \left[4\pi r^2(E+P)u\right]_a^b
        =
        \mathcal{Q}_{[a,b]}.

    By the Leibniz rule,

    .. math::

        \frac{d}{dt}
        \int_a^b 4\pi r^2 E\,dr
        =
        \int_a^b
        \frac{\partial}{\partial t}\left(4\pi r^2 E\right)\,dr
        +
        4\pi b^2 E|_b\,\dot b
        -
        4\pi a^2 E|_a\,\dot a.

    Rearranging gives

    .. math::

        \frac{d\mathcal{E}_{[a,b]}}{dt}
        =
        4\pi a^2
        \left[
            (E+P)(u-\dot a)
        \right]_{r=a}
        -
        4\pi b^2
        \left[
            (E+P)(u-\dot b)
        \right]_{r=b}
        +
        \mathcal{Q}_{[a,b]}.


Closure Assumptions
^^^^^^^^^^^^^^^^^^^^^

All three models describe the same physical interaction: freely expanding ejecta
collide with an external medium, generating a reverse shock, a contact
discontinuity, and a forward shock. The distinction between the models lies not in
the upstream physics, but in the closure used for the shocked region. The table
below summarizes the key properties of each closure.

.. list-table::
    :header-rows: 1
    :widths: 22 26 26 26

    * - Property
      - :class:`~triceratops.dynamics.shocks.numerical.MomentumConservingShockEngine`
      - :class:`~triceratops.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`
      - :class:`~triceratops.dynamics.shocks.numerical.MechanicalShockEngine`
    * - Geometry
      - Razor-thin single shell; :math:`R_{\rm rs}\approx R_{\rm cd}\approx R_{\rm fs}`
      - Thin single shell; :math:`R_{\rm rs}\approx R_{\rm cd}\approx R_{\rm fs}`
      - Two finite-width layers separated by :math:`R_{\rm cd}`
    * - State variables
      - :math:`R_{\rm sh},\;M_{\rm sh},\;\Pi_{\rm sh}`
      - :math:`R_{\rm sh},\;v_{\rm sh},\;M_{\rm sh}`
      - :math:`R_{\rm cd},\;v_{\rm cd},\;M_2,\;M_3,\;U_2,\;U_3,\;\Delta_2,\;\Delta_3`
    * - Shell acceleration
      - Advective momentum fluxes from upstream
      - Pressure difference :math:`P_2-P_3` (Rankine--Hugoniot)
      - Pressure difference :math:`P_2-P_3` (layer-averaged)
    * - Pressure model
      - None; thermal energy is discarded
      - Instantaneous post-shock Rankine--Hugoniot
      - Layer-averaged from evolved internal energies :math:`U_i`
    * - Internal energy
      - Discarded (fully radiative limit)
      - Not evolved; algebraic
      - Evolved as independent ODEs
    * - Radiative losses
      - Implicit (all shock energy radiated immediately)
      - Optional source term
      - Explicit :math:`\dot{U}_{{\rm rad},i}` per layer
    * - Best suited for
      - Radiative snowplow; late-time remnants
      - Intermediate regime; semi-analytic pressure balance
      - Adiabatic or partially radiative shocked gas


----

.. _conservative_snowplow_model:

Momentum-Conserving Snowplow
-----------------------------

The momentum-conserving snowplow
(:class:`~triceratops.dynamics.shocks.numerical.MomentumConservingShockEngine`)
is the simplest of the three numerical closures. It collapses the entire shocked
interaction region into a single razor-thin shell of mass :math:`M_{\rm sh}` and
evolves that shell purely through conservation of mass and momentum. No internal
energy is retained: shock-generated thermal energy is assumed to be radiated away
immediately, so the shell remains cold and its only dynamical degree of freedom is
its bulk momentum.

This closure is the natural description of a **radiative snowplow** — the phase of
shock evolution in which cooling is so efficient that the post-shock gas cannot
remain hot and instead accumulates in a thin, dense, cold slab.

Assumptions
^^^^^^^^^^^^

- The shocked region (:math:`R_{\rm rs} < r < R_{\rm fs}`) collapses to a
  razor-thin shell at a single radius :math:`R_{\rm sh}`, so that
  :math:`R_{\rm rs}\simeq R_{\rm cd}\simeq R_{\rm fs}\equiv R_{\rm sh}`.
- The shell velocity equals the contact-discontinuity velocity:
  :math:`D_{\rm rs}\simeq v_{\rm cd}\simeq D_{\rm fs}\equiv v_{\rm sh}`.
- Shock-generated thermal energy is radiated instantly; no pressure reservoir is
  retained.
- Mass and momentum are conserved; the energy equation is discarded.
- The upstream media are cold: :math:`P_1 \simeq P_4 \simeq 0`.

Model Equations
^^^^^^^^^^^^^^^^

Taking :math:`a=R_{\rm rs}` and :math:`b=R_{\rm fs}` in the momentum equation
and collapsing to the razor-thin limit, the geometric pressure integral is
smaller than the boundary terms by a factor of order the fractional shell
thickness and is neglected. Applying the Rankine--Hugoniot momentum condition at
each shock front replaces the downstream flux-plus-pressure terms by upstream
momentum fluxes, so that for cold upstreams,

.. math::

    \boxed{
    \frac{d\Pi_{\rm sh}}{dt}
    =
    4\pi R_{\rm sh}^2
    \rho_1(R_{\rm sh},t)\,u_1(R_{\rm sh},t)\bigl[u_1(R_{\rm sh},t)-v_{\rm sh}\bigr]
    -
    4\pi R_{\rm sh}^2
    \rho_4(R_{\rm sh},t)\,u_4(R_{\rm sh},t)\bigl[v_{\rm sh}-u_4(R_{\rm sh},t)\bigr].
    }

The mass-loading rates at each shock are

.. math::

    \dot M_1
    =
    4\pi R_{\rm sh}^2
    \rho_1(R_{\rm sh},t)
    \left[
        u_1(R_{\rm sh},t)-v_{\rm sh}
    \right],
    \qquad
    \dot M_4
    =
    4\pi R_{\rm sh}^2
    \rho_4(R_{\rm sh},t)
    \left[
        v_{\rm sh}-u_4(R_{\rm sh},t)
    \right].

The complete conservative snowplow system is

.. math::

    \boxed{
    \begin{aligned}
        \frac{dR_{\rm sh}}{dt}
        &=
        v_{\rm sh},
        \\[4pt]
        \frac{dM_{\rm sh}}{dt}
        &=
        \dot M_1+\dot M_4,
        \\[4pt]
        \frac{d\Pi_{\rm sh}}{dt}
        &=
        u_{1,\rm sh}\dot M_1
        +
        u_{4,\rm sh}\dot M_4,
        \\[4pt]
        v_{\rm sh}
        &=
        \frac{\Pi_{\rm sh}}{M_{\rm sh}},
    \end{aligned}
    }

where
:math:`u_{1,\rm sh}=u_1(R_{\rm sh},t)` and :math:`u_{4,\rm sh}=u_4(R_{\rm sh},t)`.

.. dropdown:: Derivation: acceleration form

    Since :math:`\Pi_{\rm sh}=M_{\rm sh}v_{\rm sh}`, the product rule gives

    .. math::

        \frac{d\Pi_{\rm sh}}{dt}
        =
        M_{\rm sh}\frac{dv_{\rm sh}}{dt}
        +
        v_{\rm sh}\frac{dM_{\rm sh}}{dt}.

    Substituting the mass-loading rates,

    .. math::

        M_{\rm sh}\frac{dv_{\rm sh}}{dt}
        =
        (u_{1,\rm sh}-v_{\rm sh})\dot M_1
        +
        (u_{4,\rm sh}-v_{\rm sh})\dot M_4.

    Expanding,

    .. math::

        \boxed{
        M_{\rm sh}\frac{dv_{\rm sh}}{dt}
        =
        4\pi R_{\rm sh}^2
        \left[
            \rho_{1,\rm sh}
            (u_{1,\rm sh}-v_{\rm sh})^2
            -
            \rho_{4,\rm sh}
            (v_{\rm sh}-u_{4,\rm sh})^2
        \right].
        }

    This form makes the physics explicit: faster ejecta overtaking the shell from
    behind accelerates it, while swept-up CSM ahead of the shell decelerates it.

For **homologous ejecta** (:math:`u_1 = r/t`) and **stationary CSM**
(:math:`u_4 = 0`), the system reduces to

.. math::

    \boxed{
    \begin{aligned}
        \frac{dR_{\rm sh}}{dt}
        &=
        v_{\rm sh},
        \\[4pt]
        \frac{dM_{\rm sh}}{dt}
        &=
        4\pi R_{\rm sh}^2
        \left[
            \rho_1(R_{\rm sh},t)
            \left(
                \frac{R_{\rm sh}}{t}-v_{\rm sh}
            \right)
            +
            \rho_4(R_{\rm sh},t)\,v_{\rm sh}
        \right],
        \\[4pt]
        \frac{d\Pi_{\rm sh}}{dt}
        &=
        4\pi R_{\rm sh}^2
        \rho_1(R_{\rm sh},t)
        \frac{R_{\rm sh}}{t}
        \left(
            \frac{R_{\rm sh}}{t}-v_{\rm sh}
        \right),
        \\[4pt]
        v_{\rm sh}
        &=
        \frac{\Pi_{\rm sh}}{M_{\rm sh}}.
    \end{aligned}
    }


----

.. _pressure_driven_thin_shell_model:

Pressure-Driven Thin-Shell Model
---------------------------------

The pressure-driven thin-shell model
(:class:`~triceratops.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`)
also collapses the shocked interaction region into a single thin shell of mass
:math:`M_{\rm sh}=M_2+M_3`, radius :math:`R_{\rm sh}`, and velocity
:math:`v_{\rm sh}`. However, rather than driving the shell with advective momentum
fluxes, it estimates the immediate post-shock pressures from the Rankine--Hugoniot
jump conditions and uses their net difference to accelerate the shell.

This closure is intermediate in complexity: it incorporates pressure physics
without requiring the evolution of separate internal-energy variables for each
shocked layer. It is particularly well suited to scenarios where one wants a
pressure-balance approximation without the additional overhead of the full
mechanical model.

Assumptions
^^^^^^^^^^^^

- The shocked region collapses to a thin shell at :math:`R_{\rm sh}`, so that
  :math:`R_{\rm rs}\simeq R_{\rm cd}\simeq R_{\rm fs}\equiv R_{\rm sh}`.
- The shell is accelerated by the pressure difference
  :math:`P_2 - P_3` across the contact discontinuity.
- The pressures :math:`P_2` and :math:`P_3` are evaluated algebraically from the
  instantaneous Rankine--Hugoniot jump conditions; no internal energy is evolved.
- The upstream media are cold: :math:`P_1 \simeq P_4 \simeq 0`.

.. important::

    The pressures :math:`P_2` and :math:`P_3` in this model are **instantaneous**
    algebraic functions of the upstream state and the shell velocity — they are not
    evolved as thermodynamic quantities.

Model Equations
^^^^^^^^^^^^^^^^

The shell acceleration is closed by the net post-shock pressure force,

.. math::

    \boxed{
    M_{\rm sh}
    \frac{dv_{\rm sh}}{dt}
    =
    4\pi R_{\rm sh}^2
    \left(P_2-P_3\right).
    }

For a cold, strong, non-relativistic shock with compression ratio
:math:`\chi = (\hat\gamma+1)/(\hat\gamma-1)`, the downstream pressure is

.. math::

    P_d
    =
    \rho_u v_{\rm rel}^2
    \left(
        1-\frac{1}{\chi}
    \right).

Defining the ejecta--shell velocity lag :math:`\Delta \equiv u_1(R_{\rm sh},t) - v_{\rm sh}`,
the post-shock pressures are

.. math::

    \boxed{
    P_2
    =
    \rho_1(R_{\rm sh},t)
    \,\Delta^2
    \left(
        1-\frac{1}{\chi}
    \right),
    \qquad
    P_3
    =
    \rho_4(R_{\rm sh},t)
    \left[
        v_{\rm sh}-u_4(R_{\rm sh},t)
    \right]^2
    \left(
        1-\frac{1}{\chi}
    \right).
    }

The shell mass evolves through mass loading at both shock fronts:

.. math::

    \frac{dM_2}{dt}
    =
    4\pi R_{\rm sh}^2\,
    \rho_1(R_{\rm sh},t)\,\Delta,
    \qquad
    \frac{dM_3}{dt}
    =
    4\pi R_{\rm sh}^2\,
    \rho_4(R_{\rm sh},t)
    \left[
        v_{\rm sh}-u_4(R_{\rm sh},t)
    \right].

The complete pressure-driven thin-shell system is

.. math::

    \boxed{
    \begin{aligned}
        \frac{dR_{\rm sh}}{dt}
        &=
        v_{\rm sh},
        \\[6pt]
        \frac{dv_{\rm sh}}{dt}
        &=
        \frac{4\pi R_{\rm sh}^2}{M_{\rm sh}}
        \left(
            1-\frac{1}{\chi}
        \right)
        \left[
            \rho_1(R_{\rm sh},t)\,\Delta^2
            -
            \rho_4(R_{\rm sh},t)
            \left(v_{\rm sh}-u_4(R_{\rm sh},t)\right)^2
        \right],
        \\[6pt]
        \frac{dM_{\rm sh}}{dt}
        &=
        4\pi R_{\rm sh}^2
        \left[
            \rho_1(R_{\rm sh},t)\,\Delta
            +
            \rho_4(R_{\rm sh},t)
            \left(v_{\rm sh}-u_4(R_{\rm sh},t)\right)
        \right],
    \end{aligned}
    }

where :math:`\Delta \equiv u_1(R_{\rm sh},t) - v_{\rm sh}`.

.. dropdown:: Derivation: post-shock pressure from Rankine--Hugoniot

    In the shock rest frame, momentum conservation across a non-relativistic shock is

    .. math::

        P_u+\rho_u v_u^2
        =
        P_d+\rho_d v_d^2.

    For a cold upstream (:math:`P_u\simeq 0`) with compression ratio
    :math:`\chi=\rho_d/\rho_u`, mass conservation gives :math:`v_d = v_u/\chi`, so

    .. math::

        \rho_u v_u^2
        =
        P_d+\rho_u\frac{v_u^2}{\chi}.

    Hence

    .. math::

        P_d
        =
        \rho_u v_u^2
        \left(1-\frac{1}{\chi}\right).

    For the reverse shock, :math:`v_u=\Delta\equiv u_1(R_{\rm sh},t)-v_{\rm sh}`;
    for the forward shock, :math:`v_u=v_{\rm sh}-u_4(R_{\rm sh},t)`.

For **homologous ejecta** (:math:`u_1 = r/t`) and **stationary CSM**
(:math:`u_4 = 0`), the velocity lag simplifies to
:math:`\Delta = R_{\rm sh}/t - v_{\rm sh}` and the system reduces to

.. math::

    \boxed{
    \begin{aligned}
        \frac{dR_{\rm sh}}{dt}
        &=
        v_{\rm sh},
        \\[6pt]
        \frac{dv_{\rm sh}}{dt}
        &=
        \frac{4\pi R_{\rm sh}^2}{M_{\rm sh}}
        \left(
            1-\frac{1}{\chi}
        \right)
        \left[
            \rho_1(R_{\rm sh},t)\,\Delta^2
            -
            \rho_4(R_{\rm sh},t)\,v_{\rm sh}^2
        \right],
        \\[6pt]
        \frac{dM_{\rm sh}}{dt}
        &=
        4\pi R_{\rm sh}^2
        \left[
            \rho_1(R_{\rm sh},t)\,\Delta
            +
            \rho_4(R_{\rm sh},t)\,v_{\rm sh}
        \right],
    \end{aligned}
    }

where :math:`\Delta = R_{\rm sh}/t - v_{\rm sh}`.


----

.. _mechanical_internal_energy_model:

Mechanical Internal-Energy Model
----------------------------------

The mechanical model
(:class:`~triceratops.dynamics.shocks.numerical.MechanicalShockEngine`)
is the most physically complete of the three numerical closures. Rather than
collapsing the shocked region into a single shell, it assigns separate masses,
internal energies, and effective volumes to the shocked ejecta (Region 2) and
shocked CSM (Region 3), and evolves all of them as coupled ODEs.

This closure is appropriate when the shocked gas stores thermal energy that
continues to provide pressure support over dynamical timescales — the regime
relevant to adiabatic or partially radiative shocks early in the evolution of a
remnant.

Assumptions
^^^^^^^^^^^^

- The contact discontinuity at :math:`R_{\rm cd}` separates two finite-width
  shocked layers:

  .. math::

      R_{\rm rs} = R_{\rm cd} - \Delta_2,
      \qquad
      R_{\rm fs} = R_{\rm cd} + \Delta_3.

- The shock speeds are :math:`D_{\rm rs} = v_{\rm cd} - \dot\Delta_2` and
  :math:`D_{\rm fs} = v_{\rm cd} + \dot\Delta_3`.
- The volumes of the shocked layers are approximated by thin-slab expressions,

  .. math::

      V_2 \simeq 4\pi R_{\rm cd}^2\Delta_2,
      \qquad
      V_3 \simeq 4\pi R_{\rm cd}^2\Delta_3.

- The layer-averaged pressures are thermodynamic functions of the evolved internal
  energies:

  .. math::

      P_i = (\hat\gamma_i-1)\frac{U_i}{V_i}.

- The contact discontinuity is accelerated by the net pressure force with effective
  mass :math:`M_{\rm eff}=M_2+M_3`.
- The upstream media are cold: :math:`P_1\simeq P_4\simeq 0`.

.. important::

    In this model, :math:`P_2` and :math:`P_3` are **not** instantaneous
    Rankine--Hugoniot pressures. They are layer-averaged pressures obtained from
    the internal-energy budget of each shocked region, and they evolve continuously
    as mass is swept up and work is done by expansion.

Model Equations
^^^^^^^^^^^^^^^^

The full state vector is

.. math::

    {\bf y}
    =
    \left(
        R_{\rm cd},\;
        v_{\rm cd},\;
        M_2,\;
        M_3,\;
        U_2,\;
        U_3,\;
        \Delta_2,\;
        \Delta_3
    \right).

The mass-loading equations are

.. math::

    \boxed{
    \frac{dM_2}{dt}
    =
    4\pi R_{\rm rs}^2
    \rho_1(R_{\rm rs},t)
    \left[
        u_1(R_{\rm rs},t)-D_{\rm rs}
    \right],
    \qquad
    \frac{dM_3}{dt}
    =
    4\pi R_{\rm fs}^2
    \rho_4(R_{\rm fs},t)
    \left[
        D_{\rm fs}-u_4(R_{\rm fs},t)
    \right].
    }

The contact-discontinuity equation of motion is

.. math::

    \boxed{
    (M_2+M_3)
    \frac{dv_{\rm cd}}{dt}
    =
    4\pi R_{\rm cd}^2(P_2-P_3).
    }

The internal energies evolve according to

.. math::

    \boxed{
    \frac{dU_i}{dt}
    =
    \dot U_{{\rm sh},i}
    +
    \dot U_{{\rm ad},i}
    +
    \dot U_{{\rm rad},i},
    \qquad
    i\in\{2,3\},
    }

with the shock-heating term

.. math::

    \boxed{
    \dot U_{{\rm sh},i}
    =
    \frac{1}{\hat\gamma_i-1}
    \left(
        \frac{\chi_i-1}{\chi_i^2}
    \right)
    v_{{\rm rel},i}^2
    \frac{dM_i}{dt},
    }

where
:math:`v_{{\rm rel},2}=u_1(R_{\rm rs},t)-D_{\rm rs}` and
:math:`v_{{\rm rel},3}=D_{\rm fs}-u_4(R_{\rm fs},t)`,
the adiabatic expansion term

.. math::

    \boxed{
    \dot U_{{\rm ad},i}
    =
    -(\hat\gamma_i-1)U_i
    \left(
        2\frac{v_{\rm cd}}{R_{\rm cd}}
        +
        \frac{\dot\Delta_i}{\Delta_i}
    \right),
    }

and :math:`\dot U_{{\rm rad},i}` an external radiative loss to be specified by the
user. The layer widths are closed by the layer sound speed,

.. math::

    \boxed{
    \dot\Delta_i\simeq c_{s,i},
    \qquad
    c_{s,i}^2
    \simeq
    \hat\gamma_i\frac{P_i V_i}{M_i}.
    }

Collecting all equations, the minimal mechanical model is

.. math::

    \boxed{
    \begin{aligned}
        \frac{dR_{\rm cd}}{dt}
        &=
        v_{\rm cd},
        \\[6pt]
        \frac{dM_2}{dt}
        &=
        4\pi R_{\rm rs}^2
        \rho_1(R_{\rm rs},t)
        \left[
            u_1(R_{\rm rs},t)-D_{\rm rs}
        \right],
        \\[6pt]
        \frac{dM_3}{dt}
        &=
        4\pi R_{\rm fs}^2
        \rho_4(R_{\rm fs},t)
        \left[
            D_{\rm fs}-u_4(R_{\rm fs},t)
        \right],
        \\[6pt]
        (M_2+M_3)
        \frac{dv_{\rm cd}}{dt}
        &=
        4\pi R_{\rm cd}^2(P_2-P_3),
        \\[6pt]
        \frac{dU_i}{dt}
        &=
        \dot U_{{\rm sh},i}
        +
        \dot U_{{\rm ad},i}
        +
        \dot U_{{\rm rad},i},
        \\[6pt]
        \dot\Delta_i
        &\simeq
        c_{s,i}.
    \end{aligned}
    }

.. dropdown:: Derivation: shock-heating source term

    For a cold, strong, non-relativistic shock, the compression ratio is

    .. math::

        \chi_i
        =
        \frac{\hat\gamma_i+1}{\hat\gamma_i-1}.

    The downstream pressure generated by upstream relative velocity :math:`v_{\rm rel}` is

    .. math::

        P_d
        =
        \rho_u v_{\rm rel}^2
        \left(1-\frac{1}{\chi_i}\right).

    Since :math:`\rho_d=\chi_i\rho_u`, the downstream specific internal energy is

    .. math::

        \varepsilon_{{\rm int},d}
        =
        \frac{1}{\hat\gamma_i-1}
        \frac{P_d}{\rho_d}
        =
        \frac{1}{\hat\gamma_i-1}
        \left(
            \frac{\chi_i-1}{\chi_i^2}
        \right)
        v_{\rm rel}^2.

    Multiplying by the mass-loading rate gives the shock-heating source
    :math:`\dot U_{{\rm sh},i} = \varepsilon_{{\rm int},i}\,dM_i/dt`.

.. dropdown:: Derivation: adiabatic energy loss

    The adiabatic term follows from :math:`P\,dV` work:

    .. math::

        \dot U_{{\rm ad},i}
        =
        -P_i\frac{dV_i}{dt}
        =
        -(\hat\gamma_i-1)U_i
        \frac{d\ln V_i}{dt}.

    With :math:`V_i\simeq4\pi R_{\rm cd}^2\Delta_i`,

    .. math::

        \frac{d\ln V_i}{dt}
        =
        2\frac{v_{\rm cd}}{R_{\rm cd}}
        +
        \frac{\dot\Delta_i}{\Delta_i},

    giving the boxed expression for :math:`\dot U_{{\rm ad},i}`.


----

.. _model_comparison_summary:

Summary of Modeling Choices
----------------------------

The three models differ in what is retained after the shocked region is reduced to
a small number of degrees of freedom.

The **momentum-conserving snowplow** retains only mass and momentum. It is the
most direct thin-shell closure and is appropriate when shock-generated thermal
energy is not dynamically important — for example, in the fully radiative phase of
a supernova remnant where cooling is efficient.

The **pressure-driven thin-shell model** retains mass and velocity, and computes
the shell acceleration from instantaneous post-shock pressures without evolving
separate internal energies. It is useful as a semi-analytic pressure-balance
approximation for intermediate regimes.

The **mechanical model** retains mass, velocity, internal energy, and effective
layer-width information for each shocked zone. It is the appropriate closure when
shocked gas stores thermal energy that continues to provide pressure support over
dynamical timescales.

.. warning::

    The pressure-driven and mechanical models both contain a term of the form
    :math:`4\pi R^2(P_2-P_3)`, but the meaning of the pressures differs.
    In the pressure-driven model, :math:`P_2` and :math:`P_3` are instantaneous
    Rankine--Hugoniot pressures evaluated algebraically from the upstream state.
    In the mechanical model, they are layer-averaged pressures computed from the
    evolved internal energies :math:`U_2` and :math:`U_3`.
