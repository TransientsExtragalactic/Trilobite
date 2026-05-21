.. _sedov_taylor_theory:

===========================================
Theory: Sedov-Taylor Self-Similar Solution
===========================================

The Sedov-Taylor blast-wave solution
(:class:`~trilobite.dynamics.shocks.sedov_taylor.SedovTaylorShockEngine`)
describes the adiabatic expansion of a strong spherical shock driven by a
point explosion into a uniform ambient medium. Originally derived independently
by :footcite:t:`sedov1946propagation` and :footcite:t:`taylor1950formation` in
the context of nuclear blasts, the solution has since become a foundational
framework for the **adiabatic phase** of supernova remnant evolution—the epoch
during which the swept-up mass exceeds the initial ejecta mass but radiative
losses remain unimportant.

This page derives the Sedov-Taylor scalings from first principles and documents
in full the normalization procedure implemented in
:func:`~trilobite.dynamics.shocks.sedov_taylor.sedov_taylor_beta`, including
the explicit parametric similarity profiles and the energy integral that fixes
the normalization coefficient :math:`\beta(\gamma)`.

.. contents::
   :local:
   :depth: 2

Overview
--------

We consider the idealized problem of an instantaneous explosion in a cold,
uniform, initially static medium of density :math:`\rho_0`. The ambient
pressure is taken to be negligible,

.. math::

    p_0 \simeq 0,

and the deposited energy :math:`E_0` is assumed to be conserved during the
phase of interest. In supernova remnants, this **Sedov--Taylor phase** typically
begins after the freely expanding ejecta have swept up a mass comparable to
their own and can last for hundreds to thousands of years before cooling becomes
important.

The assumptions of the Sedov--Taylor problem are therefore:

#. the explosion is effectively instantaneous;
#. the ambient medium is uniform, cold, and initially at rest;
#. the shock is strong, so the upstream Mach number is large;
#. the evolution is adiabatic and energy conserving;
#. gravity, radiative cooling, and continued energy injection are negligible.

We denote upstream, unshocked quantities by subscript :math:`0` and downstream,
post-shock quantities by subscript :math:`1`. The shock radius is :math:`R(t)`,
and its velocity in the lab frame is

.. math::

    \dot{R} \equiv \frac{dR}{dt}.

Thin-Shell Estimate
-------------------

Before deriving the similarity solution, it is useful to obtain the basic
scaling behavior using an approximate thin-shell argument. Since the shock is
strong, the Rankine--Hugoniot jump conditions give the density contrast

.. math::

    \frac{\rho_1}{\rho_0}
    =
    \frac{\gamma + 1}{\gamma - 1},

where :math:`\gamma` is the adiabatic index.

If the swept-up mass is concentrated into a thin shell of radius :math:`R` and
thickness :math:`D`, then mass conservation gives

.. math::

    4 \pi R^2 D \rho_1
    =
    \frac{4\pi}{3} R^3 \rho_0.

Therefore,

.. math::

    D
    =
    \frac{R}{3}\frac{\rho_0}{\rho_1}
    =
    \frac{R}{3}
    \frac{\gamma - 1}{\gamma + 1}.

Thus, for ordinary gases with :math:`\gamma` of order unity, the shell is
geometrically thin compared to its radius.

In the shock frame, the upstream gas flows into the shock with speed
:math:`u_0` and the downstream gas exits with speed :math:`u_1`. Mass
conservation across the shock implies

.. math::

    \rho_0 u_0 = \rho_1 u_1,

so

.. math::

    u_1
    =
    \frac{\rho_0}{\rho_1}u_0
    =
    \frac{\gamma - 1}{\gamma + 1}u_0.

The shock speed in the lab frame is the difference between the upstream and
downstream velocities in the shock frame:

.. math::

    \dot{R}
    =
    u_0 - u_1
    =
    \frac{2}{\gamma + 1}u_0.

Equivalently,

.. math::

    u_0
    =
    \frac{\gamma + 1}{2}\dot{R}.

The momentum carried by the swept-up shell is approximately

.. math::

    p_{\rm shell}
    =
    \left(4\pi R^2D\right)\rho_1\dot{R}
    =
    \frac{4\pi}{3}R^3\rho_0\dot{R}.

Since the shell sweeps up more material as it expands, its momentum increases.
This requires a pressure force from the hot interior. We parameterize the
interior pressure by

.. math::

    P_{\rm in} = \alpha p_1,

where :math:`p_1` is the immediate post-shock pressure and :math:`\alpha` is a
dimensionless constant to be fixed by energy conservation.

For a strong adiabatic shock with negligible upstream pressure,

.. math::

    p_1
    =
    \frac{2}{\gamma+1}\rho_0 u_0^2.

The outward pressure force on the shell is then

.. math::

    F
    =
    4\pi R^2 P_{\rm in}
    =
    4\pi R^2 \alpha p_1.

Using the expression for :math:`p_1`, and writing the shell momentum in terms
of :math:`u_0`, one obtains the approximate equation of motion

.. math::

    3\alpha R^2 u_0^2
    =
    \frac{d}{dt}\left(u_0R^3\right).

Taking :math:`u_0 \sim \dot{R}` for the purpose of determining the scaling,
this gives

.. math::

    3\alpha R^2 \dot{R}^2
    =
    \frac{d}{dt}\left(R^3\dot{R}\right)
    =
    3R^2\dot{R}^2 + R^3\ddot{R}.

Now assume a power-law solution,

.. math::

    R(t) \propto t^\beta.

Substitution gives

.. math::

    \beta
    =
    \frac{1}{4 - 3\alpha}.

Therefore,

.. math::

    R(t) \propto t^{1/(4-3\alpha)}

and

.. math::

    \dot{R}(t) \propto t^{(3\alpha - 3)/(4-3\alpha)}.

To determine :math:`\alpha`, we impose energy conservation. The kinetic energy
of the swept-up shell scales as

.. math::

    K
    \sim
    \rho_0 R^3 \dot{R}^2.

The internal energy of the hot post-shock gas is

.. math::

    U
    =
    \frac{1}{\gamma - 1}
    \int p\,dV.

In the thin-shell approximation, this also scales as

.. math::

    U
    \sim
    R^3 P_{\rm in}
    \sim
    R^3 \rho_0 \dot{R}^2.

Thus the total energy scales as

.. math::

    E_0
    \sim
    \rho_0 R^3 \dot{R}^2.

Since :math:`E_0` is conserved,

.. math::

    R^3\dot{R}^2 = \text{constant}.

For :math:`R \propto t^\beta`, this implies

.. math::

    t^{5\beta - 2} = \text{constant},

and therefore

.. math::

    \beta = \frac{2}{5}.

Hence the characteristic Sedov--Taylor scalings are

.. math::

    \boxed{
    R(t) \propto t^{2/5}
    }

and

.. math::

    \boxed{
    \dot{R}(t) \propto t^{-3/5}
    }.

Since the post-shock pressure scales as

.. math::

    p_1 \sim \rho_0 \dot{R}^2,

we also have

.. math::

    \boxed{
    p_1(t) \propto t^{-6/5}
    }.

Dimensional Derivation
----------------------

The same result follows immediately from dimensional analysis. During the
energy-conserving phase, the only dimensional quantities controlling the shock
radius are

.. math::

    E_0, \qquad \rho_0, \qquad t.

Their dimensions are

.. math::

    [E_0] = M L^2 T^{-2},
    \qquad
    [\rho_0] = M L^{-3},
    \qquad
    [t] = T.

The unique combination with dimensions of length is

.. math::

    \left(\frac{E_0 t^2}{\rho_0}\right)^{1/5}.

Therefore the blast-wave radius must take the form

.. math::

    \boxed{
    R(t)
    =
    \beta(\gamma)
    \left(\frac{E_0 t^2}{\rho_0}\right)^{1/5}
    }

where :math:`\beta(\gamma)` is a dimensionless constant determined by the full
similarity solution and depends only on :math:`\gamma`. The shock velocity is
then

.. math::

    \boxed{
    \dot{R}(t)
    =
    \frac{2}{5}\frac{R}{t}
    }.

Post-Shock Jump Conditions
--------------------------

Immediately behind the shock, the Rankine--Hugoniot conditions for a strong
shock give

.. math::

    \boxed{
    \rho_1
    =
    \rho_0
    \frac{\gamma + 1}{\gamma - 1}
    }

and

.. math::

    \boxed{
    p_1
    =
    \frac{2}{\gamma + 1}
    \rho_0 \dot{R}^2
    }.

The downstream gas velocity in the lab frame is

.. math::

    \boxed{
    v_1
    =
    \frac{2}{\gamma + 1}\dot{R}
    }.

These provide the boundary conditions for the self-similar flow just behind the
shock.

Similarity Solution
-------------------

The full Sedov--Taylor solution resolves the spatial structure of the shocked
gas behind the blast wave. In spherical symmetry, the adiabatic Euler equations
are

.. math::

    \frac{\partial v}{\partial t}
    +
    v\frac{\partial v}{\partial r}
    =
    -\frac{1}{\rho}\frac{\partial p}{\partial r},

.. math::

    \frac{\partial \rho}{\partial t}
    +
    \frac{1}{r^2}
    \frac{\partial}{\partial r}
    \left(r^2\rho v\right)
    =
    0,

and

.. math::

    \left(
    \frac{\partial}{\partial t}
    +
    v\frac{\partial}{\partial r}
    \right)
    \log\left(\frac{p}{\rho^\gamma}\right)
    =
    0.

The last equation expresses adiabatic entropy conservation along streamlines
behind the shock.

Because the problem contains no intrinsic length scale, the flow variables must
depend on radius and time through the dimensionless similarity coordinate

.. math::

    \boxed{
    \xi
    =
    \frac{r}{R(t)}
    }.

The shock front is located at :math:`\xi = 1` and the center of the explosion
at :math:`\xi = 0`. We write the hydrodynamic variables in self-similar form as

.. math::

    v(r,t)
    =
    \dot{R}(t) V(\xi),

.. math::

    \rho(r,t)
    =
    \rho_0 G(\xi),

and

.. math::

    p(r,t)
    =
    \rho_0 \dot{R}^2(t) P(\xi).

The functions :math:`V(\xi)`, :math:`G(\xi)`, and :math:`P(\xi)` describe the
dimensionless velocity, density, and pressure profiles inside the blast wave.
At the shock, the Rankine--Hugoniot conditions impose the boundary conditions

.. math::

    V(1)
    =
    \frac{2}{\gamma+1},
    \qquad
    G(1)
    =
    \frac{\gamma+1}{\gamma-1},
    \qquad
    P(1)
    =
    \frac{2}{\gamma+1}.

Substituting the similarity ansatz into the Euler equations reduces the
original partial differential equations to a coupled system of ordinary
differential equations in :math:`\xi`. These ODEs are integrable in closed form
for general :math:`\gamma` (see :footcite:t:`landau1987fluid`, §106;
:footcite:t:`sedov1946propagation`), yielding the explicit parametric solution
described in the following section.

.. _sedov_taylor_normalization:

Normalization Coefficient
-------------------------

The normalization :math:`\beta(\gamma)` is not fixed by self-similarity alone;
it is determined by requiring that the total energy contained within the blast
equals the injected energy :math:`E_0`. This section derives the energy
integral and documents the explicit similarity profiles used in its evaluation.

Energy Normalization Condition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The total kinetic plus thermal energy inside the shock is

.. math::

    E_0
    =
    4\pi
    \int_0^{R(t)}
    \left[
        \frac{1}{2}\rho v^2
        +
        \frac{p}{\gamma - 1}
    \right]
    r^2\,dr.

Substituting the similarity ansatz and changing the integration variable from
:math:`r` to :math:`\xi = r/R`, this becomes

.. math::

    E_0
    =
    4\pi \rho_0 \dot{R}^2 R^3
    \int_0^1
    \left[
        \frac{1}{2}G(\xi)V^2(\xi)
        +
        \frac{P(\xi)}{\gamma - 1}
    \right]
    \xi^2\,d\xi.

Using the conventional notation from the similarity solution, it is more
natural to replace the pressure profile :math:`P(\xi)` with the dimensionless
squared sound speed :math:`Z(\xi)`, defined by

.. math::

    c_s^2
    =
    \frac{\gamma p}{\rho}
    \equiv
    \left(\frac{2r}{5t}\right)^2 Z(\xi),

so that :math:`P(\xi) = G(\xi) V(\xi)^2 Z(\xi) / \gamma`. The energy integral
then reads

.. math::

    E_0
    =
    \frac{16\pi}{25}
    \rho_0 \dot{R}^2 R^3
    \underbrace{
    \int_0^1
    G(\xi)
    \left[
        \frac{V(\xi)^2}{2}
        +
        \frac{Z(\xi)}{\gamma(\gamma-1)}
    \right]
    \xi^4\,d\xi
    }_{C_E(\gamma)/({16\pi}/{25})}.

Since :math:`\dot{R}^2 R^3 = (2R/5t)^2 R^3 = (4/25) R^5/t^2`, and
:math:`R^5/t^2 = \beta^5 E_0/\rho_0`, the energy conservation condition
reduces to

.. math::

    1 = \beta^5 C_E(\gamma),

hence

.. math::

    \boxed{
    \beta(\gamma)
    =
    C_E(\gamma)^{-1/5},
    \qquad
    C_E(\gamma)
    =
    \frac{16\pi}{25}
    \int_0^1
    G(\xi)
    \left[
        \frac{V(\xi)^2}{2}
        +
        \frac{Z(\xi)}{\gamma(\gamma-1)}
    \right]
    \xi^4\,d\xi.
    }

.. note::

    Some references write :math:`R(t) = (E_0 t^2 / \alpha_{\rm ST} \rho_0)^{1/5}`,
    in which case :math:`\alpha_{\rm ST} = \beta^{-5} = C_E`. Both conventions
    are in common use; Trilobite uses the :math:`\beta` form.

Explicit Parametric Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The similarity ODEs for :math:`V`, :math:`G`, and :math:`Z` as functions of
:math:`\xi` are integrable in closed form. Following
:footcite:t:`sedov1946propagation` and :footcite:t:`landau1987fluid`, the
explicit solution uses the **dimensionless velocity** :math:`V` as a
parametric variable in place of :math:`\xi`. Define the auxiliary combinations

.. math::

    A \;=\; \tfrac{\gamma+1}{2}\,V,
    \qquad
    B \;=\; \frac{\gamma+1}{7-\gamma}\bigl[5 - (3\gamma-1)V\bigr],

.. math::

    C \;=\; \frac{\gamma+1}{\gamma-1}\,({\gamma V - 1}),
    \qquad
    D \;=\; \frac{\gamma+1}{\gamma-1}\,(1 - V),

and the five exponents

.. math::

    \nu_1
    =
    -\frac{13\gamma^2 - 7\gamma + 12}{(3\gamma-1)(2\gamma+1)},
    \qquad
    \nu_2
    =
    \frac{5(\gamma-1)}{2\gamma+1},
    \qquad
    \nu_3
    =
    \frac{3}{2\gamma+1},

.. math::

    \nu_4
    =
    -\frac{\nu_1}{2-\gamma},
    \qquad
    \nu_5
    =
    -\frac{2}{2-\gamma}.

.. important::

    The formulae above are singular at :math:`\gamma = 2`, where several
    exponents diverge. Trilobite explicitly rejects this value.

The explicit parametric profiles are then

.. math::

    \xi(V)
    =
    \left(A^{-2} B^{\nu_1} C^{\nu_2}\right)^{1/5},

.. math::

    G(V)
    =
    \frac{\gamma+1}{\gamma-1}\,C^{\nu_3} B^{\nu_4} D^{\nu_5},

.. math::

    Z(V)
    =
    \frac{\gamma(\gamma-1)(1-V)\,V^2}{2(\gamma V - 1)}.

The parameter :math:`V` ranges from

.. math::

    V_0 = \frac{1}{\gamma}
    \qquad\text{(origin,}\ \xi = 0\text{)}

to

.. math::

    V_s = \frac{2}{\gamma+1}
    \qquad\text{(shock,}\ \xi = 1\text{)}.

The lower endpoint :math:`V_0` is numerically singular (several powers of
:math:`C` and :math:`D` vanish there), and the integration begins at
:math:`V_0 + \varepsilon\,\Delta V` with a small offset :math:`\varepsilon`
to avoid the singularity.

Change of Variable for Numerical Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because the similarity profiles are expressed parametrically in :math:`V`, the
energy integral is evaluated by changing the integration variable from
:math:`\xi` to :math:`V`:

.. math::

    C_E(\gamma)
    =
    \frac{16\pi}{25}
    \int_{V_0}^{V_s}
    G(V)
    \left[
        \frac{V^2}{2}
        +
        \frac{Z(V)}{\gamma(\gamma-1)}
    \right]
    \xi(V)^4
    \frac{d\xi}{dV}\,dV.

The Jacobian is obtained by differentiating :math:`\log\xi(V)`:

.. math::

    \frac{d\xi}{dV}
    =
    \xi(V)
    \times
    \frac{1}{5}
    \left[
        -\frac{2}{V}
        \;-\;
        \frac{\nu_1(3\gamma-1)}{5-(3\gamma-1)V}
        \;+\;
        \frac{\nu_2\,\gamma}{\gamma V - 1}
    \right].

The integral is then evaluated numerically using
:func:`scipy.integrate.quad`. For :math:`\gamma = 5/3` (monatomic ideal gas),

.. math::

    \beta\!\left(\tfrac{5}{3}\right)
    \approx 1.1517,
    \qquad
    C_E\!\left(\tfrac{5}{3}\right)
    =
    \beta^{-5}
    \approx 2.026,

and for :math:`\gamma = 7/5 = 1.4` (diatomic ideal gas),

.. math::

    \beta(1.4)
    \approx 1.033,
    \qquad
    C_E(1.4)
    \approx 1.18.

.. note::

    :func:`~trilobite.dynamics.shocks.sedov_taylor.sedov_taylor_beta` computes
    :math:`\beta(\gamma)` once at class instantiation and caches it on the
    :class:`~trilobite.dynamics.shocks.sedov_taylor.SedovTaylorShockEngine`.
    The quadrature tolerances and the singularity offset :math:`\varepsilon` are
    all user-controllable via keyword arguments.

Implementation Notes
--------------------

The derivations above are directly reflected in the Trilobite implementation:

- :func:`~trilobite.dynamics.shocks.sedov_taylor.sedov_taylor_beta` evaluates
  :math:`C_E(\gamma)` via :func:`scipy.integrate.quad` over the parametric
  :math:`V`-interval and returns :math:`\beta = C_E^{-1/5}`.

- :class:`~trilobite.dynamics.shocks.sedov_taylor.SedovTaylorShockEngine`
  stores :math:`\beta` and evaluates the shock kinematics

  .. math::

      R_s(t) = \beta\!\left(\frac{E_0\,t^2}{\rho_0}\right)^{1/5},
      \qquad
      v_s(t) = \frac{2}{5}\frac{R_s}{t},

  together with the immediate post-shock thermodynamics from the
  strong-shock Rankine--Hugoniot conditions.

- Post-shock temperature is computed from the ideal-gas relation

  .. math::

      T_s = \frac{p_s\,\mu\,m_p}{\rho_s\,k_B},

  where :math:`\mu` is the mean molecular weight (in units of the proton mass
  :math:`m_p`) and :math:`k_B` is Boltzmann's constant. For a fully ionized
  hydrogen plasma, :math:`\mu = 0.5`.

.. seealso::

    - :ref:`jump_conditions_theory` for the Rankine--Hugoniot conditions used
      to set the post-shock boundary values.
    - :class:`~trilobite.dynamics.shocks.sedov_taylor.SedovTaylorShockEngine`
      for the full API reference.

References
----------
.. footbibliography::
