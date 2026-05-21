.. _rankine_hugoniot_theory:

===========================================
Methods: (Non-Relativistic) Jump Conditions
===========================================

Consider a **shock front** dividing two regions of fluid. We refer to each side as the
**upstream** and **downstream** side of the shock front. For simplicity, assume the
front occurs at :math:`x = 0` in some reference frame and that, on either side,
we have :math:`\rho_{1,2}`, :math:`p_{1,2}`, and :math:`u_{1,2}`.

Certain conservation laws must still hold—namely those expressed by the
**Euler equations**. These allow us to characterize the behavior across the
shock front in terms of conservation principles.

.. note::

    A **critical** realization is that the Rankine–Hugoniot relations derived below
    are valid **only in the rest frame of the shock**. Thus, we are always implicitly
    performing a Galilean transformation into that frame when applying them.

    This can make the physical interpretation subtle. For example, in a bow shock
    propagating into the intracluster medium (ICM), the ICM gas is the **upstream**
    side because it moves *toward* the shock in the shock’s reference frame.

The General Form of the Rankine–Hugoniot Conditions
---------------------------------------------------
To begin, we present a short derivation of the Rankine–Hugoniot conditions for those unfamiliar with
then. In each case, these conditions are based on a conservation law integrated across an infinitesimal control
volume enclosing the shock front.

The Continuity Condition
^^^^^^^^^^^^^^^^^^^^^^^^

In Eulerian form, the continuity equation is

.. math::

    \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho u) = 0.

Integrating across an infinitesimal region of width :math:`\delta x` around the
shock front:

.. math::

    \frac{\partial}{\partial t} \int_{-\delta x}^{\delta x} \rho \, dx
    + (\rho u)_{\delta x} - (\rho u)_{-\delta x} = 0.

Taking the limit :math:`\delta x \to 0`, the integral term vanishes, yielding

.. math::

    \rho_1 u_1 = \rho_2 u_2.

This expresses conservation of mass flux across the shock.

The Momentum Condition
^^^^^^^^^^^^^^^^^^^^^^^^

In one-dimensional inviscid flow with external force :math:`{\bf f}_{\rm ext}`,
the momentum equation in conservative form is

.. math::

    \frac{\partial (\rho u)}{\partial t}
    + \frac{\partial}{\partial x} \left( \rho u^2 + p \right)
    = \rho {\bf f}_{\rm ext}.

Integrating across a thin control volume enclosing the shock:

.. math::

    \int_{-\delta x}^{+\delta x}
    \frac{\partial}{\partial x} \left( \rho u^2 + p \right) dx
    =
    \int_{-\delta x}^{+\delta x} \rho {\bf f}_{\rm ext} \, dx.

If :math:`{\bf f}_{\rm ext}` is bounded, the right-hand side vanishes as
:math:`\delta x \to 0`, giving

.. math::

    \left[ \rho u^2 + p \right]_1^2 = 0.

This yields the **momentum Rankine–Hugoniot condition**:

.. math::

    \rho_1 u_1^2 + p_1 = \rho_2 u_2^2 + p_2.

The Energy Condition
^^^^^^^^^^^^^^^^^^^^^

To derive the energy condition, we assume:

1. **Adiabatic flow** (no heating or cooling)
2. **Inviscid flow** (no viscous dissipation)

The energy equation becomes

.. math::

    \frac{\partial E}{\partial t}
    + \nabla \cdot \left[(E+p)\mathbf{u}\right] = 0,

where

.. math::

    E = \frac{1}{2}\rho u^2 + \rho \epsilon.

In terms of enthalpy :math:`h = \epsilon + p/\rho`, this becomes

.. math::

    \frac{\partial E}{\partial t}
    + \nabla \cdot \left[\left(\frac{1}{2}\rho u^2 + \rho h\right)\mathbf{u}\right] = 0.

Integrating across the shock:

.. math::

    \left[ u(E+p) \right]_1^2 = 0,

or equivalently,

.. math::

    \left[ u\left(\frac{1}{2}\rho u^2 + \rho h\right) \right]_1^2 = 0.

Using mass conservation (:math:`\rho u = \text{const}`), this simplifies to

.. math::

    \left[ \mathcal{E} + \frac{p}{\rho} \right]_1^2 = 0,

or

.. math::

    \left[ \frac{1}{2}u^2 + h \right]_1^2 = 0,

where :math:`\mathcal{E} = \epsilon + \frac{1}{2}u^2`.

A useful parameterization of the energy condition is to invoke a **polytropic equation of state** from which we can
show that

.. math::

    h = \int \frac{dp}{\rho} = \frac{\Gamma}{\Gamma - 1} \frac{p}{\rho} = \frac{c_s^2}{\Gamma - 1},

where :math:`c_s` is the sound speed. Thus, the energy condition can be expressed as

.. math::

    \left[ \frac{1}{2}u^2 + \frac{c_s^2}{\Gamma - 1} \right]_1^2 = 0.

The Mach-Number Form of the Jump Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is often useful to express the Rankine–Hugoniot conditions in terms of the
**Mach number**,

.. math::

    M \equiv \frac{u}{c_s},

which combines the flow velocity and sound speed into a single dimensionless
quantity. In astrophysical contexts, shocks are typically characterized by the
**upstream Mach number** :math:`M_1`.

Starting from the conservation relations and eliminating intermediate variables,
one can derive a closed expression for the **compression ratio**:

.. math::

    \frac{\rho_2}{\rho_1}
    = \frac{(\Gamma + 1) M_1^2}{(\Gamma - 1) M_1^2 + 2}.

This shows that the density jump depends only on the upstream Mach number.
In the strong-shock limit (:math:`M_1 \to \infty`), this reduces to

.. math::

    \frac{\rho_2}{\rho_1} \to \frac{\Gamma + 1}{\Gamma - 1},

which equals :math:`4` for a monatomic ideal gas (:math:`\Gamma = 5/3`).

The corresponding **pressure jump** can be written as

.. math::

    \frac{P_2}{P_1}
    = 1 + \frac{2\Gamma}{\Gamma + 1} (M_1^2 - 1).

These expressions provide a compact and practical parameterization of shock
strength in terms of a single variable.

For completeness, one may also eliminate the Mach number entirely and express
the density ratio directly in terms of the pressure ratio:

.. math::

    \frac{\rho_2}{\rho_1}
    = \frac{(\Gamma - 1) P_1 + (\Gamma + 1) P_2}
           {(\Gamma + 1) P_1 + (\Gamma - 1) P_2}.

This form is particularly useful in observational settings where pressure jumps
are measured directly.

The Temperature Jump Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The temperature jump across a shock follows from the equation of state and the
previously derived pressure and density ratios. In general,

.. math::

    \frac{T_2}{T_1} = \frac{P_2}{P_1} \cdot \frac{\rho_1}{\rho_2}.

Substituting the Mach-number expressions for the pressure and density ratios,

.. math::

    \frac{P_2}{P_1}
    = 1 + \frac{2\Gamma}{\Gamma + 1} (M_1^2 - 1),

.. math::

    \frac{\rho_2}{\rho_1}
    = \frac{(\Gamma + 1) M_1^2}{(\Gamma - 1) M_1^2 + 2},

we obtain

.. math::

    \frac{T_2}{T_1}
    =
    \left[
        1 + \frac{2\Gamma}{\Gamma + 1} (M_1^2 - 1)
    \right]
    \left[
        \frac{(\Gamma - 1) M_1^2 + 2}{(\Gamma + 1) M_1^2}
    \right].

This simplifies to the compact form

.. math::

    \frac{T_2}{T_1}
    =
    \frac{\left[2\Gamma M_1^2 - (\Gamma - 1)\right]
          \left[(\Gamma - 1) M_1^2 + 2\right]}
         {(\Gamma + 1)^2 M_1^2}.

In the strong-shock limit (:math:`M_1 \to \infty`), this reduces to

.. math::

    \frac{T_2}{T_1}
    \to
    \frac{2\Gamma(\Gamma - 1)}{(\Gamma + 1)^2} M_1^2.

For a monatomic ideal gas (:math:`\Gamma = 5/3`), this becomes

.. math::

    \frac{T_2}{T_1} \to \frac{5}{16} M_1^2.

----

Weak vs. Strong Shocks
----------------------

The expressions derived above make clear that the **Mach number** serves as the
natural parameter controlling the strength of a shock. It is therefore useful to
distinguish between the regimes of **weak** and **strong** shocks, which represent
two qualitatively different limits of the same underlying Rankine–Hugoniot relations.

A **weak shock** is characterized by an upstream Mach number only slightly greater
than unity,

.. math::

    M_1 \gtrsim 1.

In this regime, the jump conditions reduce to small perturbations about the
upstream state. Expanding the compression ratio for :math:`M_1^2 = 1 + \epsilon`
with :math:`\epsilon \ll 1`, one finds

.. math::

    \frac{\rho_2}{\rho_1}
    \approx 1 + \frac{2}{\Gamma + 1} (M_1^2 - 1).

Thus, weak shocks produce only modest changes in density, pressure, and
temperature. Physically, they behave more like nonlinear sound waves than
discontinuous, highly dissipative structures. In many astrophysical environments,
such shocks are difficult to detect observationally because of their small
contrast.

By contrast, a **strong shock** corresponds to the limit

.. math::

    M_1 \gg 1,

in which the upstream kinetic energy dominates over the upstream thermal energy.
In this regime, the jump conditions asymptote to simple, Mach-independent forms.
For example, the compression ratio becomes

.. math::

    \frac{\rho_2}{\rho_1}
    \to \frac{\Gamma + 1}{\Gamma - 1},

which is a constant determined solely by the equation of state. This saturation
is a key feature of strong shocks: once the flow is sufficiently supersonic,
further increases in Mach number no longer increase the density jump, but instead
manifest primarily as increases in downstream temperature and pressure.

The distinction between weak and strong shocks is therefore not merely quantitative,
but qualitative:

- **Weak shocks** are perturbative and nearly reversible.
- **Strong shocks** are highly dissipative and efficiently convert bulk kinetic
  energy into thermal energy.

These regimes are continuously connected through the full Mach-number-dependent
Rankine–Hugoniot relations, but separating them provides useful physical intuition
and motivates the simplified limits often used in modeling.

Cold vs. Hot Upstream Media
---------------------------

A second, equally important distinction concerns the **thermodynamic state of the
upstream medium**. In many applications, it is useful to distinguish between
**cold** and **hot** shocks, depending on whether the upstream thermal pressure
is dynamically important.

A **cold shock** refers to the limit in which the upstream pressure and temperature
are negligible,

.. math::

    P_1 \approx 0, \qquad T_1 \approx 0.

In this case, the upstream sound speed is small, and the Mach number is effectively
determined by the shock velocity alone. The momentum condition simplifies to

.. math::

    P_2 \approx \rho_1 u_1^2 \left(1 - \frac{1}{R}\right),

so that the downstream pressure is set entirely by the incoming kinetic energy
flux. A particularly important consequence of this limit is that the **post-shock
temperature becomes independent of the upstream density**:

.. math::

    T_2 \propto u_1^2.

This density-independence is a defining feature of cold shocks and is frequently
used in astrophysical modeling (e.g., supernova remnants, blast waves), where the
ambient medium is sufficiently tenuous that its initial thermal energy can be
neglected.

In contrast, a **hot shock** occurs when the upstream thermal pressure is
non-negligible. In this case, the full Rankine–Hugoniot relations must be retained,
and the upstream state contributes explicitly to the downstream quantities:

.. math::

    P_2 = P_1 + \rho_1 u_1^2 \left(1 - \frac{1}{R}\right).

Here, the upstream pressure :math:`P_1` provides a finite offset, and the Mach
number must be computed self-consistently from both the velocity and the upstream
thermodynamic state. Physically, this means that part of the downstream pressure
is inherited from the upstream medium rather than generated purely by shock
dissipation.

The difference between cold and hot shocks can therefore be summarized as follows:

- **Cold shocks** are kinetically dominated (:math:`P_1 \to 0`) and lead to
  simple, density-independent temperature scalings.
- **Hot shocks** retain memory of the upstream thermodynamic state and require
  full evaluation of the jump conditions.
