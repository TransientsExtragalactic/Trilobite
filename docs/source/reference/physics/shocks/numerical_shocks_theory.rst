.. _numeric_shocks_theory:

==========================================
Theory: Numerical Shock Engines
==========================================

In many scenarios, the evolution of transient shocks is not well approximated by self-similar solutions or
other toy models. In these cases, one requires a more flexible approach to model the shock evolution. Many different
approaches have been taken in the literature ... (we should say more here).

.. contents::
    :local:
    :depth: 2

.. _numeric_thin_shell_shocks:
Thin Shell Shocks (Non-Relativistic)
-------------------------------------

One downside of the self-similar approach is that it relies on a number of assumptions about the ejecta and CSM density
profiles, which may **not always be valid**. To address this, we implement a purely numerical thin-shell model which
permits much more general density profiles for both the ejecta and CSM.

In this approach, we imagine the shock as a **thin shell** with an equation of motion dictated by the conservation of
momentum. The **thin shell** is the region between the forward and reverse shocks (regions 2 and 3 in our standard
notation). We denote the mass in this shell as :math:`M_{\rm sh} = M_2 + M_3`, where :math:`M_2` is the
mass swept up from the ejecta and :math:`M_3` is the mass swept up from the CSM. The velocity of the shell is
the shock velocity, :math:`v_{\rm sh} = \dot{R}_{\rm sh}`, where :math:`R_{\rm sh}` is the shock radius.

.. hint::

    In these models, we assume that the forward and reverse shocks are very close together such that
    :math:`R_{\rm sh} \approx R_{\rm fs} \approx R_{\rm rs}`.

The Equations of Motion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The core of the thin-shell model is the equation for the conservation of momentum in the shell. To avoid internal
inconsistencies, we define the "shell" in this scenario to be the region between the forward and reverse shocks which
has fully thermalized (i.e., shocked). We assume that, once thermalized behind the forward / reverse shock, the material
all moves together at the shock velocity, :math:`v_{\rm sh}`. Treating the shell as a contact discontinuity, there
is no mass transfer across the shell. Therefore, the only way for momentum to change in the shell is via pressure
forces from the shocked ejecta and CSM.

In this formalism, the Cauchy-Euler equations reduce to a single equation for the conservation of momentum:

.. math::

    \frac{\partial {\bf p}}{\partial t} + \nabla \cdot ({\bf p} \otimes {\bf u}) + \nabla \cdot \boldsymbol{\sigma} = 0,

where :math:`{\bf p} = \rho {\bf u}` is the momentum density and :math:`\boldsymbol{\sigma}` is the stress tensor.
Again, we treat the shell as pressure supported with no momentum transport across the shell, so the stress tensor
reduces to the pressure term, :math:`\boldsymbol{\sigma} = P {\bf I}`, and the ram pressure term
becomes 0. This ensures that we do not double-count contributions to the momentum from mass fluxes across the shell.

Integrating over the volume of the thin shell, we have

.. math::

    M_{\rm sh} \frac{\partial}{\partial t}\left(v_{\rm sh}\right) = 4\pi R_{\rm sh}^2 (P_2-P_3)

Shell Pressure
~~~~~~~~~~~~~~~

The pressures behind the forward and reverse shocks follow from the strong-shock Rankine--Hugoniot jump conditions.
Denoting the compression ratio :math:`\chi = (\gamma+1)/(\gamma-1)` and assuming a cold upstream medium on both
sides (:math:`P_{\rm upstream} = 0`), the pressures immediately behind each shock are

.. math::

    P_2 = \rho_{\rm csm}\,v_{\rm sh}^2\!\left(1 - \frac{1}{\chi}\right),

.. math::

    P_3 = \rho_{\rm ej}\,(v_{\rm ej} - v_{\rm sh})^2\!\left(1 - \frac{1}{\chi}\right).

We define :math:`\Delta \equiv v_{\rm ej} - v_{\rm sh}` as the relative velocity between the upstream ejecta and
the shell, so the net pressure force on the shell is

.. math::

    4\pi R_{\rm sh}^2 (P_2 - P_3)
    =
    4\pi R_{\rm sh}^2 \!\left(1 - \frac{1}{\chi}\right)
    \!\left(\rho_{\rm csm}\,v_{\rm sh}^2 - \rho_{\rm ej}\,\Delta^2\right).

.. dropdown:: Derivation: pressure from Rankine-Hugoniot

    Momentum conservation across a shock (in the shock frame) requires

    .. math::

        P_1 + \rho_1 v_1^2 = P_2 + \rho_2 v_2^2,

    where subscripts 1 and 2 denote upstream and downstream quantities and velocities are measured in the shock
    frame. For a strong shock the downstream velocity satisfies :math:`v_2 = v_1/\chi`.

    **Forward shock.** The upstream medium is the CSM with :math:`P_1 = 0` and upstream velocity
    :math:`v_1 = v_{\rm sh}`. Substituting:

    .. math::

        \rho_{\rm csm}\,v_{\rm sh}^2
        =
        P_2 + \rho_{\rm csm}\,\frac{v_{\rm sh}^2}{\chi^2}\cdot\chi
        =
        P_2 + \frac{\rho_{\rm csm}\,v_{\rm sh}^2}{\chi},

    .. math::

        P_2
        =
        \rho_{\rm csm}\,v_{\rm sh}^2\!\left(1 - \frac{1}{\chi}\right).

    **Reverse shock.** The upstream medium is the ejecta; in the shock frame the upstream velocity is
    :math:`v_1 = v_{\rm ej} - v_{\rm sh} = \Delta`, with :math:`P_1 = 0`. The same algebra gives

    .. math::

        P_3
        =
        \rho_{\rm ej}\,\Delta^2\!\left(1 - \frac{1}{\chi}\right).

Shell Mass Evolution
~~~~~~~~~~~~~~~~~~~~~~

It is easy to show that the shell masses can be obtained in quadrature by integrating the density profiles;
however, this leads to a numerical scheme that requires time-step integration to obtain the mass evolution. Instead,
we derive differential equations for the mass evolution that can be solved simultaneously with the equation of motion.

The approach exploits **homologous expansion**: because the ejecta expand freely before the shock sweeps through them,
the velocity and radius of a given mass shell are related by :math:`v = r/t`. If the ejecta carry a mass
distribution :math:`dM_{\rm ej}/dv = f(v)`, the density takes the self-similar form

.. math::

    \rho_{\rm ej}(r,t) = t^{-3}\,G\!\left(v = \frac{r}{t}\right),
    \qquad
    G(v) = \frac{f(v)}{4\pi v^2}.

.. hint::

    In the self-similar model we allow only scenarios where :math:`f(v) \propto v^{-n}`; however, here we can
    consider arbitrary distributions.

At time :math:`t`, the mass of swept-up ejecta (region 2) is the mass at velocities
:math:`v > R_{\rm sh}/t \equiv v_{\rm ej}`:

.. math::

    M_2(t) = \int_{v_{\rm ej}}^{\infty} f(v)\,dv.

The CSM mass (region 3) sweeps up straightforwardly:

.. math::

    \frac{dM_3}{dt} = 4\pi\,\rho_{\rm csm}(R_{\rm sh})\,R_{\rm sh}^2\,v_{\rm sh}.

.. dropdown:: Derivation: :math:`dM_2/dt` via Leibniz rule

    Differentiating the integral for :math:`M_2` with respect to time, with the lower limit
    :math:`v_{\rm ej}(t) = R_{\rm sh}(t)/t` depending on time:

    .. math::

        \frac{dM_2}{dt}
        =
        -f(v_{\rm ej})\,\frac{dv_{\rm ej}}{dt}.

    The negative sign arises because the lower limit moves upward as the shock expands, reducing the
    integral. Computing :math:`dv_{\rm ej}/dt`:

    .. math::

        \frac{dv_{\rm ej}}{dt}
        =
        \frac{d}{dt}\!\left(\frac{R_{\rm sh}}{t}\right)
        =
        \frac{\dot{R}_{\rm sh}}{t} - \frac{R_{\rm sh}}{t^2}
        =
        \frac{v_{\rm sh} - v_{\rm ej}}{t}
        =
        -\frac{\Delta}{t}.

    Substituting, and using :math:`f(v_{\rm ej}) = 4\pi v_{\rm ej}^2\,G(v_{\rm ej}) = 4\pi (R_{\rm sh}/t)^2 G(v_{\rm ej})`:

    .. math::

        \frac{dM_2}{dt}
        =
        -f(v_{\rm ej})\left(-\frac{\Delta}{t}\right)
        =
        4\pi\,\frac{R_{\rm sh}^2}{t^3}\,G(v_{\rm ej})\,\Delta.

Combining, the full ODE system in the original variables :math:`(R_{\rm sh}, v_{\rm sh}, M_{\rm sh})` is

.. math::

    \begin{aligned}
    \frac{dR_{\rm sh}}{dt} &= v_{\rm sh},\\[4pt]
    \frac{dv_{\rm sh}}{dt} &=
        \frac{-4\pi R_{\rm sh}^2}{M_{\rm sh}}\!\left(1-\frac{1}{\chi}\right)
        \!\left(\rho_{\rm csm}\,v_{\rm sh}^2 - t^{-3}\,G(v_{\rm ej})\,\Delta^2\right),\\[4pt]
    \frac{dM_{\rm sh}}{dt} &=
        4\pi R_{\rm sh}^2
        \!\left\{\rho_{\rm csm}\,v_{\rm sh} + t^{-3}\,G(v_{\rm ej})\,\Delta\right\}.
    \end{aligned}

Numerical Stability and Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The system above is numerically challenging: it is stiff, and when :math:`\Delta \to 0` (shock velocity
approaching ejecta velocity) small errors can produce unphysical results. We remove the dominant explicit
time dependence associated with homologous expansion by introducing the dimensionless variables

.. math::

    \xi \equiv \frac{R_{\rm sh}}{t}, \qquad
    \tau \equiv \ln t, \qquad
    \Delta \equiv \xi - v_{\rm sh},

where :math:`\xi` is the homologous ejecta velocity coordinate at the shock radius and :math:`\Delta` is the
relative velocity between the upstream ejecta and the shell.

The transformed, numerically stable ODE system in :math:`(\xi, \Delta, M_{\rm sh})` is

.. math::

    \boxed{
    \begin{aligned}
        \frac{d\xi}{d\tau} &= -\Delta,\\[4pt]
        \frac{d\Delta}{d\tau} &=
            -\Delta
            + \frac{4\pi\xi^2}{M_{\rm sh}}\!\left(1-\frac{1}{\chi}\right)
            \!\left(t^3\rho_{\rm csm}(\xi t)\,(\xi-\Delta)^2
                  - G(\xi)\,\Delta^2\right),\\[6pt]
        \frac{dM_{\rm sh}}{d\tau} &=
            4\pi\xi^2
            \!\left[t^3\rho_{\rm csm}(\xi t)\,(\xi-\Delta)
                  + G(\xi)\,\Delta\right].
    \end{aligned}
    }

.. dropdown:: Derivation: change of variables to :math:`(\xi,\tau,\Delta)`

    **Equation for** :math:`\xi`. Since :math:`R_{\rm sh} = t\,\xi`:

    .. math::

        v_{\rm sh}
        =
        \frac{dR_{\rm sh}}{dt}
        =
        \xi + t\,\frac{d\xi}{dt}.

    Rearranging and using :math:`d\tau = dt/t`:

    .. math::

        t\,\frac{d\xi}{dt}
        =
        v_{\rm sh} - \xi
        =
        -\Delta
        \quad\Rightarrow\quad
        \frac{d\xi}{d\tau} = -\Delta.

    **Equation for** :math:`v_{\rm sh}`. From the equation of motion,

    .. math::

        \frac{dv_{\rm sh}}{dt}
        =
        -\frac{4\pi R_{\rm sh}^2}{M_{\rm sh}}\!\left(1-\frac{1}{\chi}\right)
        \!\left(\rho_{\rm csm}(R_{\rm sh})\,v_{\rm sh}^2 - t^{-3}G(\xi)\,\Delta^2\right).

    Multiplying by :math:`t`, substituting :math:`R_{\rm sh} = \xi t` and :math:`v_{\rm sh} = \xi - \Delta`:

    .. math::

        \frac{dv_{\rm sh}}{d\tau}
        =
        -\frac{4\pi\xi^2}{M_{\rm sh}}\!\left(1-\frac{1}{\chi}\right)
        \!\left(t^3\rho_{\rm csm}(\xi t)\,(\xi-\Delta)^2 - G(\xi)\,\Delta^2\right).

    **Equation for** :math:`\Delta`. Since :math:`\Delta = \xi - v_{\rm sh}`:

    .. math::

        \frac{d\Delta}{d\tau}
        =
        \frac{d\xi}{d\tau} - \frac{dv_{\rm sh}}{d\tau}
        =
        -\Delta - \frac{dv_{\rm sh}}{d\tau},

    and substituting the expression for :math:`dv_{\rm sh}/d\tau` obtained above yields the boxed result.

    **Equation for** :math:`M_{\rm sh}`. Starting from

    .. math::

        \frac{dM_{\rm sh}}{dt}
        =
        4\pi R_{\rm sh}^2\!\left[\rho_{\rm csm}(R_{\rm sh})\,v_{\rm sh} + t^{-3}G(\xi)\,\Delta\right],

    multiplying by :math:`t` and substituting :math:`R_{\rm sh} = \xi t` and :math:`v_{\rm sh} = \xi - \Delta`
    gives the boxed result directly.

----

.. _relativistic_numeric_thin_shell_shocks:
Thin Shell Shocks (Relativistic)
-------------------------------------

We now consider the extended case in which the shock velocity approaches the
speed of light. In this regime, we must carefully account for relativistic
effects in both the shock jump conditions and the shell inertia. Unlike the
non-relativistic case, we will directly evolve the lab-frame energy
:math:`E_{\rm sh}` and radial momentum :math:`p_{\rm sh}` of the shell. The
baryonic rest mass is still conserved, but it is no longer by itself the full
measure of the shell inertia, since internal energy and pressure also contribute
to the stress-energy tensor.

The shell velocity is obtained from the shell four-momentum.

.. math::

    \beta_{\rm sh}
    =
    \frac{p_{\rm sh}c}{E_{\rm sh}},
    \qquad
    v_{\rm sh}
    =
    \frac{p_{\rm sh}c^2}{E_{\rm sh}}.

To formulate the relativistic thin-shell model, we work in the lab frame and
consider a shocked region near :math:`R_{\rm sh}(t)` separating the ejecta from
the CSM. As in the non-relativistic case, we assume that the shocked region is
thin compared with the overall dynamical scale, so that

.. math::

    R_{\rm sh} \approx R_{\rm fs} \approx R_{\rm rs}.

Here :math:`R_{\rm fs}` is the forward-shock radius and :math:`R_{\rm rs}` is
the reverse-shock radius.

The upstream states are specified as functions of lab-frame position and time.
For the pre-shock ejecta, we prescribe

.. math::

    \rho_{\rm ej}(r,t),
    \qquad
    U_{{\rm int},{\rm ej}}(r,t),
    \qquad
    v_{\rm ej}(r,t).

For the pre-shock CSM, we likewise prescribe

.. math::

    \rho_{\rm csm}(r,t),
    \qquad
    U_{{\rm int},{\rm csm}}(r,t),
    \qquad
    v_{\rm csm}(r,t).

The coordinates :math:`(r,t)` are **lab-frame coordinates**. However, the
thermodynamic quantities :math:`\rho` and :math:`U_{\rm int}` are proper
quantities: they are measured in the local comoving frame of the fluid element
located at :math:`(r,t)`. By contrast, :math:`v(r,t)` is the velocity of that
fluid element measured in the lab frame.

The corresponding lab-frame rest-mass density is Lorentz enhanced:

.. math::

    \rho_{\rm lab} = \gamma \rho,

where

.. math::

    \gamma = \frac{1}{\sqrt{1-\beta^2}},
    \qquad
    \beta = \frac{v}{c}.

We assume a perfect-fluid equation of state,

.. math::

    P = (\hat{\gamma}-1)U_{\rm int},

so that the proper total energy density is

.. math::

    e = \rho c^2 + U_{\rm int} = \rho c^2 + \frac{P}{\hat{\gamma}-1},

and the proper enthalpy density is

.. math::

    w = e + P = \rho c^2 + \hat{\gamma}U_{\rm int} = \rho c^2 + \frac{\hat{\gamma}}{\hat{\gamma}-1}P.

Equations of Motion
^^^^^^^^^^^^^^^^^^^^

In the relativistic case, the equations of motion are obtained from the Euler equations in relativistic form:

.. math::

    \begin{aligned}
    \nabla_\mu T^{\mu\nu} &= 0,\\
    \nabla_\mu (\rho u^\mu) &= 0,
    \end{aligned}

where :math:`T^{\mu\nu}` is the stress-energy tensor and :math:`u^\mu` is the four-velocity of the fluid. The first
equation encodes the conservation of energy and momentum, while the second encodes the conservation of baryon number
(rest mass). By integrating these equations over the volume of the thin shell, we can derive the equations of motion
for the shell's energy and momentum.

On either side of the shocked region, we have

.. math::

    T_i^{\mu\nu} = \frac{w_i}{c^2}u_i^\mu u_i^\nu + P_i g^{\mu\nu},

which in spherical coordinates (and assuming spherical symmetry), requires that

.. math::

    T_i = \begin{bmatrix}
    \frac{w_i}{c^2}\gamma_i^2 - P_i & \frac{w_i}{c^2}\gamma_i^2\beta_i\\
    \frac{w_i}{c^2}\gamma_i^2\beta_i & \frac{w_i}{c^2}\gamma_i^2\beta_i^2 + P_i\\
    \end{bmatrix}.

The first of our equations of motion is the **energy equation**, which we derive below

.. dropdown:: The Energy Equation

    We consider a thin shocked shell bounded by :math:`R_-` and :math:`R_+`,
    both of which are approximately equal to the shock radius :math:`R_s`.

    The **energy in the shell** is

    .. math::

        E_s(t)
        =
        4\pi
        \int_{R_-}^{R_+}
        T^{00}(r,t) r^2 \, dr.

    Differentiating this expression using the Leibniz rule gives

    .. math::

        \frac{dE_s}{dt}
        =
        4\pi
        \int_{R_-}^{R_+}
        \partial_t T^{00} \, r^2 \, dr
        +
        4\pi R_+^2 \dot{R}_+ T^{00}(R_+, t)
        -
        4\pi R_-^2 \dot{R}_- T^{00}(R_-, t).

    From energy-momentum conservation,

    .. math::

        \nabla_\mu T^{\mu\nu} = 0,

    the :math:`\nu = 0` component in spherical symmetry gives

    .. math::

        \frac{\partial T^{00}}{\partial t}
        +
        \frac{1}{r^2}
        \frac{\partial}{\partial r}
        \left(
            r^2 T^{r0}
        \right)
        =
        0.

    Substituting this into the time derivative of :math:`E_s` gives

    .. math::

        \frac{dE_s}{dt}
        =
        4\pi R^2
        \left(
            \mathcal{F}_-^0
            -
            \mathcal{F}_+^0
        \right),

    where the energy flux through a moving shell boundary is

    .. math::

        \mathcal{F}_\pm^0
        =
        T^{r0}
        -
        \beta_s T^{00}.

    Therefore, the shell energy equation is

    .. math::

        \boxed{
        \frac{dE_s}{dt}
        =
        4\pi R^2
        \left[
            w_0 \gamma_0^2(\beta_0-\beta_s)
            -
            w_1 \gamma_1^2(\beta_1-\beta_s)
            +
            (P_0-P_1)\beta_s
        \right]
        }.

    In terms of the internal energy density,

    .. math::

        \frac{dE_s}{dt}
        =
        4\pi R^2
        \left[
            \left(\rho_0 c^2 + \hat{\gamma} U_{{\rm int},0}\right) \gamma_0^2(\beta_0-\beta_s)
            -
            \left(\rho_1 c^2 + \hat{\gamma} U_{{\rm int},1}\right) \gamma_1^2(\beta_1-\beta_s)
            +
            (\hat{\gamma}-1)(U_{{\rm int},0}-U_{{\rm int},1})\beta_s
        \right].

Conservation of momentum follows in a very similar fashion:

.. dropdown:: The Momentum Equation

    The lab-frame radial momentum in the shell is

    .. math::

        P_s(t)
        =
        4\pi
        \int_{R_-}^{R_+}
        T^{0r}(r,t)r^2\,dr.

    Differentiating,

    .. math::

        \frac{dP_s}{dt}
        =
        4\pi
        \int_{R_-}^{R_+}
        \partial_tT^{0r}\,r^2dr
        +
        4\pi R_+^2\dot R_+T^{0r}(R_+,t)
        -
        4\pi R_-^2\dot R_-T^{0r}(R_-,t).

    Unlike the energy equation, the radial momentum equation contains a geometric
    source term in spherical coordinates:

    .. math::

        \frac{\partial T^{0r}}{\partial t}
        +
        \frac{1}{r^2}
        \frac{\partial}{\partial r}
        \left(r^2T^{rr}\right)
        -
        \frac{2P}{r}
        =
        0.

    Therefore,

    .. math::

        \frac{dP_s}{dt}
        =
        4\pi R_-^2
        \left(
        T^{rr}_--\dot R_-T^{0r}_-
        \right)
        -
        4\pi R_+^2
        \left(
        T^{rr}_+-\dot R_+T^{0r}_+
        \right)
        +
        8\pi
        \int_{R_-}^{R_+}
        Pr\,dr.

    In units :math:`c=1`, :math:`\dot R_\pm=\beta_\pm`. Defining

    .. math::

        \mathcal{F}^r
        \equiv
        T^{rr}-\beta_bT^{0r},

    we have

    .. math::

        \frac{dP_s}{dt}
        =
        4\pi R_-^2\mathcal{F}^r_-
        -
        4\pi R_+^2\mathcal{F}^r_+
        +
        8\pi
        \int_{R_-}^{R_+}
        Pr\,dr.

    Using

    .. math::

        T^{rr}=w\gamma^2\beta^2+P,
        \qquad
        T^{0r}=w\gamma^2\beta,

    the moving-boundary radial momentum flux is

    .. math::

        \mathcal{F}^r
        =
        w\gamma^2\beta(\beta-\beta_b)+P.

    In the thin-shell limit,

    .. math::

        R_-\simeq R_+\simeq R_s,
        \qquad
        \beta_-\simeq\beta_+\simeq\beta_s,

    so

    .. math::

        \frac{dP_s}{dt}
        =
        4\pi R_s^2
        \left(
        \mathcal{F}_-^r-\mathcal{F}_+^r
        \right)
        +
        8\pi
        \int_{R_-}^{R_+}
        Pr\,dr.

    If the inner side is labeled :math:`0` and the outer side is labeled :math:`1`,
    then

    .. math::

        \boxed{
        \frac{dP_s}{dt}
        =
        4\pi R_s^2
        \left[
        w_0\gamma_0^2\beta_0(\beta_0-\beta_s)
        -
        w_1\gamma_1^2\beta_1(\beta_1-\beta_s)
        +
        (P_0-P_1)
        \right]
        +
        8\pi
        \int_{R_-}^{R_+}
        Pr\,dr.
        }

    At leading order in the thin-shell approximation, the final geometric term is
    of order :math:`\ell/R_s` relative to the boundary pressure force and is often
    neglected, giving

    .. math::

        \boxed{
        \frac{dP_s}{dt}
        \simeq
        4\pi R_s^2
        \left[
        w_0\gamma_0^2\beta_0(\beta_0-\beta_s)
        -
        w_1\gamma_1^2\beta_1(\beta_1-\beta_s)
        +
        (P_0-P_1)
        \right].
        }

As such, our two fundamental conservation laws are

.. math::

    \boxed{
    \begin{aligned}
    \frac{dE_s}{dt}
    &=
    4\pi R^2
    \left[
        w_0 \gamma_0^2(\beta_0-\beta_s)
        -
        w_1 \gamma_1^2(\beta_1-\beta_s)
        +
        (P_0-P_1)\beta_s
    \right],\\[6pt]
    \frac{dP_s}{dt}
    &\simeq
    4\pi R^2
    \left[
        w_0\gamma_0^2\beta_0(\beta_0-\beta_s)
        -
        w_1\gamma_1^2\beta_1(\beta_1-\beta_s)
        +
        (P_0-P_1)
    \right].
    \end{aligned}
    }

These equations alone are sufficient to evolve the dynamics; however, we are also interested in understanding
the density / mass evolution of the shell, which requires also tracking the baryonic rest mass. The corresponding
conservation law is

.. math::

    \frac{dM_s}{dt} = 4\pi R^2 \left[\rho_0 \gamma_0 (\beta_0 - \beta_s) + \rho_1 \gamma_1 (\beta_1 - \beta_s)\right].
