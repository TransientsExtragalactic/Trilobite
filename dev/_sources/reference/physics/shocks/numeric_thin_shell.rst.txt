.. _numeric_thin_shell_shocks:
==========================================
Numeric Thin-Shell Shock Models
==========================================

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
------------------------

The core of the thin-shell model is the equation for the conservation of momentum in the shell. To avoid internal
inconsistencies, we define the "shell" in this scenario to be the region between the forward and reverse shocks which
has fully thermalized (i.e., shocked). We assume that, one thermalized behind the forward / reverse shock, the material
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
^^^^^^^^^^^^^^^

In order to leverage the above equation of motion, we need expressions for the pressures behind
the forward and reverse shocks, for which we can use the Rankine-Hugoniot jump conditions. Momentum conservation across
the shock requires that

.. math::

    P_1 + \rho_1 v_1^2 = P_2 + \rho_2 v_2^2,

where :math:`P`, :math:`\rho`, and :math:`v` are the pressure, density, and velocity in the shock frame, respectively.

For the forward shock, we assume a cold CSM relative to the post-shock material, allowing us to set :math:`P_1=0`.
In the shock frame, the upstream velocity is :math:`v_1 = v_{\rm sh}` and the downstream velocity is,
using the strong shock condition,

.. math::

    v_2 = \frac{1}{\chi}(v_{\rm sh}),

where :math:`\chi = (\gamma +1)/(\gamma -1)` is the compression ratio. Thus, the pressure behind the forward shock is

.. math::

    P_2 = \rho_{\rm CSM} v_{\rm sh}^2 \left(1 - \frac{1}{\chi}\right).

For the reverse shock, we again assume a cold upstream medium, allowing us to set :math:`P_1=0`. In the shock frame,
the upstream velocity is :math:`v_1 = v_{\rm ej} - v_{\rm sh}`, :math:`v_{\rm ej}` being the ejecta
velocity at the shock radius. The resulting pressure behind the reverse shock is then

.. math::

    P_3 = \rho_{\rm ej} (v_{\rm ej} - v_{\rm sh})^2 \left(1 - \frac{1}{\chi}\right).

The pressure term in the equation of motion is therefore

.. math::

    4\pi R_{\rm sh}^2 (P_2-P_3) = 4\pi R_{\rm sh}^2 \left(1 - \frac{1}{\chi}\right)\left(\rho_{\rm csm} v_{\rm sh}^2 - \rho_{\rm ej}
    (v_{\rm ej} - v_{\rm sh})^2\right).

We refer to the velocity difference between the ejecta and shock as :math:`\Delta = v_{\rm ej} - v_{\rm sh}` going
forward.

Shell Mass Evolution
^^^^^^^^^^^^^^^^^^^^^

It is easy to show that the masses can be obtained in quadrature by integrating the density profiles; however,
this leads to a numerical scheme that requires time-step integration to obtain the mass evolution. Instead, we
can derive differential equations for the mass evolution that can be solved simultaneously with the equation of motion.

A more mathematically rich approach relies again on the **homologous expansion** in the early phase of the supernova
expansion. Because this is true regardless of the ejecta profile, it remains the case that if the ejecta have some
distribution of velocities such that

.. math::

    \frac{dM_{\rm ej}}{dv} = f(v),


then the density must be of the form

.. math::

    \rho_{\rm ej}(r,t) = t^{-3} G(v = r/t),

where

.. math::

    G(v) = \frac{f(v)}{4\pi v^2}.

.. hint::

    In the self-similar model, we allow only scenarios where :math:`f(v) \propto v^{-n}`; however, here we can
    consider arbitrary distributions.

Now, at a given time :math:`t`, the mass swept up in region 2 (**post-shock ejecta**) is the same as all of
the mass with

.. math::

    r>R_{\rm sh} \iff v> R_{\rm sh}/t = v_{\rm ej},

Thus,

.. math::

    M_2(t) = \int_{v_{\rm ej}}^{\infty} f(v) dv,

which means that (using Leibniz rule), noting that

.. math::

    \frac{dv_{\rm ej}}{dt} = \frac{d}{dt}\left(\frac{R_{\rm sh}}{t}\right)
     = \frac{\dot{R}_{\rm sh}}{t} - \frac{R_{\rm sh}}{t^2},

the rate of change of :math:`M_2` is

.. math::

    \frac{dM_2}{dt} = \frac{dM_2}{dv_{\rm ej}} \frac{dv_{\rm ej}}{dt} = + 4\pi \frac{R_{\rm sh}^2}{t^3}
    G\left[v_{\rm ej}\right] \Delta.

We can, of course, find the rate of change of :math:`M_3` by considering the mass swept up from the CSM: a much
simpler undertaking than the ejecta case. We have

.. math::

    \frac{dM_3}{dt} = 4\pi \rho_{\rm csm}[R_{\rm sh}] R_{\rm sh}^2 v_{\rm sh},

We do have to keep track of the shock mass as it evolves, which is given by the equations derived above
for the post-shock ejecta and CSM mass evolution. The combined set of ODEs is therefore

.. math::

    \begin{aligned}
    \frac{dR_{\rm sh}}{dt} &= v_{\rm sh}\\
    \frac{dv_{\rm sh}}{dt} &= \frac{-4\pi R_{\rm sh}^2}{ M_{\rm sh}}\left(1-\frac{1}{\chi}\right)
    \left(\rho_{\rm csm} v_{\rm sh}^2 - t^{-3} G[v_{\rm ej}] \Delta^2\right)\\
    \frac{dM_{\rm sh}}{dt} &= 4\pi R_{\rm sh}^2 \left\{\rho_{\rm csm} v_{\rm sh} + t^{-3} G[v_{\rm ej}] \Delta\right\}
    \end{aligned}

Numerical Stability and Implementation
--------------------------------------

The equations above can be solved numerically using standard ODE solvers. Care must be taken to ensure numerical
stability, particularly in regimes where the shock velocity approaches the ejecta velocity (i.e., :math:`\Delta \to 0`).
In these cases, small numerical errors can lead to unphysical results, such as negative shock velocities or masses.
Likewise, the scheme is somewhat stiff and therefore requires an appropriate solver (e.g., implicit methods or adaptive
step-size control).

To mitigate these issues, we introduce dimensionless variables that remove the
dominant explicit time dependence associated with homologous expansion. Define

.. math::

    \xi \equiv \frac{R_{\rm sh}}{t}, \qquad
    \tau \equiv \ln t, \qquad
    \Delta \equiv \xi - v_{\rm sh}.

Note that :math:`\xi` is the homologous ejecta velocity coordinate at the shock
radius (:math:`v_{\rm ej} = R_{\rm sh}/t`), and :math:`\Delta` is the relative
velocity between the upstream ejecta and the shell.

Since :math:`R_{\rm sh} = t\,\xi`, we have

.. math::

    \frac{dR_{\rm sh}}{dt} = \xi + t\frac{d\xi}{dt}.

But :math:`dR_{\rm sh}/dt = v_{\rm sh}`, so

.. math::

    v_{\rm sh} = \xi + t\frac{d\xi}{dt}
    \quad\Rightarrow\quad
    t\frac{d\xi}{dt} = v_{\rm sh} - \xi = -\Delta.

Using :math:`d\tau = dt/t`, this becomes

.. math::

    \boxed{\frac{d\xi}{d\tau} = -\Delta.}

Next, since :math:`\Delta = \xi - v_{\rm sh}`, differentiation yields

.. math::

    \frac{d\Delta}{d\tau}
    = \frac{d\xi}{d\tau} - \frac{dv_{\rm sh}}{d\tau}
    = -\Delta - \frac{dv_{\rm sh}}{d\tau}.

From the thin-shell equation of motion derived above,

.. math::

    \frac{dv_{\rm sh}}{dt}
    =
    -\frac{4\pi R_{\rm sh}^2}{ M_{\rm sh}}\left(1-\frac{1}{\chi}\right)
    \left(\rho_{\rm csm}(R_{\rm sh})\, v_{\rm sh}^2
          - t^{-3}G(\xi)\,\Delta^2\right).

Multiplying by :math:`t` and using :math:`R_{\rm sh} = \xi t` and
:math:`v_{\rm sh} = \xi - \Delta`, we obtain

.. math::

    \boxed{
    \frac{dv_{\rm sh}}{d\tau}
    =
    -\frac{4\pi \xi^2}{M_{\rm sh}}\left(1-\frac{1}{\chi}\right)
    \left(t^3\rho_{\rm csm}(\xi t)\,(\xi-\Delta)^2
          - G(\xi)\,\Delta^2\right).
    }

Substituting into :math:`d\Delta/d\tau = -\Delta - dv_{\rm sh}/d\tau` gives

.. math::

    \boxed{
    \frac{d\Delta}{d\tau}
    =
    -\Delta
    + \frac{4\pi \xi^2}{M_{\rm sh}} \left(1-\frac{1}{\chi}\right)
    \left(t^3\rho_{\rm csm}(\xi t)\,(\xi-\Delta)^2
          - G(\xi)\,\Delta^2\right).
    }

Finally, the shell-mass evolution equation

.. math::

    \frac{dM_{\rm sh}}{dt}
    =
    4\pi R_{\rm sh}^2\left[
        \rho_{\rm csm}(R_{\rm sh})\,v_{\rm sh}
        + t^{-3}G(\xi)\,\Delta
    \right]

becomes, after multiplying by :math:`t` and substituting :math:`R_{\rm sh}=\xi t`,

.. math::

    \boxed{
    \frac{dM_{\rm sh}}{d\tau}
    =
    4\pi \xi^2
    \left[
        t^3\rho_{\rm csm}(\xi t)\,(\xi-\Delta)
        + G(\xi)\,\Delta
    \right].
    }

The numerically stable ODE system in :math:`(\xi,\Delta,M_{\rm sh})` is therefore

.. math::

    \boxed{
    \begin{aligned}
        \frac{d\xi}{d\tau} &= -\Delta,\\[4pt]
        \frac{d\Delta}{d\tau} &=
        -\Delta
        + \frac{4\pi \xi^2}{M_{\rm sh}} \left(1-\frac{1}{\chi}\right)
        \left(t^3\rho_{\rm csm}(\xi t)\,(\xi-\Delta)^2
              - G(\xi)\,\Delta^2\right),\\[6pt]
        \frac{dM_{\rm sh}}{d\tau} &=
        4\pi \xi^2
        \left[t^3\rho_{\rm csm}(\xi t)\,(\xi-\Delta)
              + G(\xi)\,\Delta\right].
    \end{aligned}
    }
