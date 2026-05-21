.. _relativistic_jump_conditions_theory:
==============================================
Methods: (Relativistic) Jump Conditions
==============================================

Relativistic shocks arise in flows where the bulk velocity approaches the speed
of light and the internal energy becomes comparable to (or exceeds) the rest-mass
energy of the fluid. In this regime, the classical Rankine–Hugoniot relations are
no longer sufficient, and the full machinery of relativistic fluid dynamics must
be employed.

Compared to the non-relativistic case, several key features change:

- Energy and momentum are unified through the stress-energy tensor.
- The inertia of the fluid includes both rest-mass and internal energy.
- Velocities are bounded by the speed of light, introducing Lorentz factors.
- Thermodynamic quantities are defined in the local fluid rest frame, while
  fluxes are evaluated in the shock frame.

These effects are essential in high-energy astrophysical systems, including
gamma-ray bursts (GRBs), relativistic jets, and compact object outflows, where
shocks play a central role in energy dissipation and radiation production.

In our context, robust relativistic shock solvers are required to accurately
capture these transitions across a wide range of regimes, from mildly relativistic
flows to ultra-relativistic blast waves.

.. contents::
   :local:
   :depth: 2

----


The Relativistic Fluid Equations
--------------------------------

The dynamics of a relativistic fluid are governed by conservation laws expressed
in covariant form.

**Baryon number conservation** is given by

.. math::

    \nabla_\mu (n u^\mu) = 0,

where :math:`n` is the proper (rest-frame) number density and :math:`u^\mu` is the
fluid four-velocity.

**Energy-momentum conservation** is expressed through the stress-energy tensor
:math:`T^{\mu\nu}`:

.. math::

    \nabla_\mu T^{\mu\nu} = 0.

For a perfect fluid, the stress-energy tensor is

.. math::

    T^{\mu\nu} = w u^\mu u^\nu + P g^{\mu\nu},

where:

- :math:`P` is the pressure,
- :math:`w = e + P` is the enthalpy density,
- :math:`e` is the total energy density (including rest-mass energy),
- :math:`g^{\mu\nu}` is the metric tensor.

To close the system, we adopt an ideal equation of state:

.. math::

    w = \rho c^2 + \frac{\gamma_{\rm ad}}{\gamma_{\rm ad} - 1} P,

where :math:`\rho = n m` is the rest-mass density and :math:`\gamma_{\rm ad}` is
the adiabatic index.

The Continuity Conditions
-------------------------

We now derive the jump conditions for a steady, planar shock. As in the
non-relativistic derivation, we work in the rest frame of the shock. In this
frame the shock is stationary, and the upstream and downstream fluids flow
through it with velocities :math:`\beta_1 c` and :math:`\beta_2 c`.

The four-velocity of the fluid is

.. math::

    u^\mu = \Gamma(c, v, 0, 0),

or, using :math:`\beta = v/c`,

.. math::

    u^\mu = \Gamma c(1, \beta, 0, 0).

The Lorentz factor is

.. math::

    \Gamma = \frac{1}{\sqrt{1-\beta^2}}.

Since the shock is planar and steady, only the flux through the shock surface is
relevant. Taking the shock normal to lie along the :math:`x` direction, baryon
conservation requires continuity of :math:`N^x`:

.. math::

    N_1^x = N_2^x.

Because :math:`N^x = n\Gamma v`, this gives

.. math::

    n_1\Gamma_1 v_1 = n_2\Gamma_2 v_2.

Equivalently, in terms of rest-mass density and dimensionless velocity,

.. math::

    \rho_1\Gamma_1\beta_1 = \rho_2\Gamma_2\beta_2.

This is the relativistic continuity condition.

Next consider energy-momentum conservation. The momentum flux normal to the shock
is the :math:`xx` component of the stress-energy tensor:

.. math::

    T^{xx} = w \Gamma^2 \beta^2 + P.

Continuity of normal momentum flux therefore gives

.. math::

    w_1\Gamma_1^2\beta_1^2 + P_1
    =
    w_2\Gamma_2^2\beta_2^2 + P_2.

Similarly, the energy flux through the shock is the :math:`0x` component,

.. math::

    T^{0x} = w\Gamma^2\beta,

up to the common factor of :math:`c` set by convention. Continuity of energy flux
gives

.. math::

    w_1\Gamma_1^2\beta_1
    =
    w_2\Gamma_2^2\beta_2.

Thus the relativistic jump conditions are

.. math::
    :label: continuity_conditions

    \begin{aligned}
    \rho_1 \Gamma_1 \beta_1 &= \rho_2 \Gamma_2 \beta_2, \\
    w_1 \Gamma_1^2 \beta_1^2 + P_1
        &= w_2 \Gamma_2^2 \beta_2^2 + P_2, \\
    w_1 \Gamma_1^2 \beta_1
        &= w_2 \Gamma_2^2 \beta_2.
    \end{aligned}

These equations have the same structure as the classical Rankine--Hugoniot
conditions, but with two essential relativistic modifications. First, the fluxes
carry factors of :math:`\Gamma` because densities and thermodynamic quantities are
measured in the comoving frame while the fluxes are measured in the shock frame.
Second, the mass density in the momentum and energy fluxes is replaced by the
enthalpy density :math:`w`, because pressure and internal energy contribute to
the inertia of the fluid.

A useful way to remember the frame conventions is:

- :math:`\rho`, :math:`n`, :math:`P`, :math:`e`, and :math:`w` are comoving
  fluid-frame quantities.
- :math:`\beta` and :math:`\Gamma` describe the motion of each fluid state in the
  shock frame.
- The conserved fluxes are evaluated through the stationary shock surface.

This distinction is not optional. Mixing lab-frame densities with shock-frame
velocities gives incorrect jump conditions.

In the non-relativistic limit, :math:`\beta \ll 1` and :math:`\Gamma \to 1`.
The rest-mass contribution dominates the enthalpy, so :math:`w \approx \rho c^2`.
After removing the common rest-energy flux from the energy equation, the system
reduces to the classical Rankine--Hugoniot relations,

.. math::

    \rho_1 u_1 = \rho_2 u_2,

.. math::

    P_1 + \rho_1 u_1^2 = P_2 + \rho_2 u_2^2,

together with the usual non-relativistic energy condition. Thus the classical
jump conditions are not a separate theory; they are the low-velocity limit of the
relativistic flux conservation laws.

The ultra-relativistic strong-shock limit gives another important check. For a
cold upstream medium, :math:`P_1 \to 0` and :math:`w_1 \approx \rho_1 c^2`. If
the downstream gas is relativistically hot, :math:`\gamma_{\rm ad}=4/3`. In the
limit :math:`\Gamma_1 \gg 1`, the downstream velocity in the shock frame tends to

.. math::

    \beta_2 \to \frac{1}{3}.

This is the classic strong relativistic shock result used in relativistic blast
wave theory and GRB afterglow modeling.

Solving the Relativistic Jump Conditions
-----------------------------------------

Unlike the classical case, the relativistic jump conditions do not admit a simple
closed-form solution in general. Instead, the system must be reduced to a single
nonlinear equation in one unknown. To do this, we recognize that :math:`\beta_2` can be expressed
entirely in terms of the upstream state via an implicit expression and then the downstream
quantities may be derived from :math:`\beta_2`. Formally, we begin from the shock
conditions, :eq:`continuity_conditions`. in the form

.. math::

    \begin{aligned}
    \rho_2(\beta_2) &= \rho_1 \frac{\Gamma_1 \beta_1}{\Gamma_2 \beta_2}, \\
    w_2(\beta_2) &= w_1 \frac{\Gamma_1^2 \beta_1}{\Gamma_2^2 \beta_2}, \\
    P_2(\beta_2) - P_1 &= w_1 \Gamma_1^2 \beta_1^2 \left(1- \frac{\beta_2}{\beta_1}\right).
    \end{aligned}

From the equation of state,

.. math::

    P_2(\beta_2) = \frac{\hat{\gamma} - 1}{\hat{\gamma}} \left(w_2 - \rho_2 c^2\right),

which may be expressed in terms of the existing shock conditions such that

.. math::

    P_2(\beta_2) = \frac{\hat{\gamma} - 1}{\hat{\gamma}} \frac{\Gamma_1 \beta_1}{\Gamma_2 \beta_2}\left[w_1 \frac{\Gamma_1}{\Gamma_2} - \rho c^2\right].

This results in a single fundamental equation for :math:`\beta_2`:

.. math::

    \boxed{
    \left(\frac{\hat{\gamma} - 1}{\hat{\gamma}}\right)
    \frac{\Gamma_1 \beta_1}{\Gamma_2 \beta_2}
    \left[w_1 \frac{\Gamma_1}{\Gamma_2} - \rho c^2\right] - w_1 \Gamma_1^2 \beta_1^2 \left(1- \frac{\beta_2}{\beta_1}\right) - P_1 = 0
    }

To avoid numerical issues, we substitute :math:`\Pi_i = P_i / (\rho_i c^2)` and :math:`\Omega_i = w_i / (\rho_i c^2)`, which gives

.. math::
    :label: eq:relativistic_jump_condition_final

    \boxed{
    \left(\frac{\hat{\gamma} - 1}{\hat{\gamma}}\right)
    \frac{\Gamma_1 \beta_1}{\Gamma_2 \beta_2}
    \left[\Omega_1 \frac{\Gamma_1}{\Gamma_2} - 1\right] - \Omega_1 \Gamma_1^2 \beta_1^2 \left(1- \frac{\beta_2}{\beta_1}\right) - \Pi_1 = 0
    }

.. hint::

    The physical meaning of these parameters is quite intuitive:

    - The factor :math:`\Pi_1` is the **dimensionless pressure**, representing the continuum from :math:`\Pi_1 \approx 0` for
      a cold shock to :math:`\Pi_1 \gg 1` for a hot shock.
    - The factor :math:`\Omega_1` is the **dimensionless enthalpy**, representing the continuum from :math:`\Omega_1 \approx 1` for a
      non-relativistic shock to :math:`\Omega_1 \gg 1` for a relativistically hot shock.

Analytic Scenarios
^^^^^^^^^^^^^^^^^^^

In most cases, these equations are solved numerically to determine the post-shock state of the gas; however, there
are a few special cases that admit closed-form solutions. From

.. math::

    \left(\frac{\hat{\gamma} - 1}{\hat{\gamma}}\right)
    \frac{\Gamma_1 \beta_1}{\Gamma_2 \beta_2}
    \left[\Omega_1 \frac{\Gamma_1}{\Gamma_2} - 1\right] - \Omega_1 \Gamma_1^2 \beta_1^2 \left(1- \frac{\beta_2}{\beta_1}\right) - \Pi_1 = 0,

we can see that in the **ultra-relativistic, strong shock limit** in a **cold medium**, we have :math:`\Omega_1 \sim 1`
and :math:`\Pi_1 \approx 0`. Using the fact that :math:`\Omega_i (\Gamma_1/\Gamma_2) \gg 1` and :math:`\beta_1 \sim 1` in this limit,

.. math::

    \left(\frac{\hat{\gamma} - 1}{\hat{\gamma}}\right)
    \frac{\Gamma_1 }{\Gamma_2 \beta_2}
    \left[\Omega_1 \frac{\Gamma_1}{\Gamma_2}\right] = \Omega_1 \Gamma_1^2  \left(1- \beta_2\right).

After simplification, this requires that

.. math::

    \beta_2(1-\beta_2) = \frac{\hat{\gamma} - 1}{\hat{\gamma}}\left(1-\beta_2^2\right),

which has the solution :math:`\beta_2 = 1/3` for :math:`\hat{\gamma} = 4/3`. This is the classic result for a
strong relativistic shock in a cold medium, which is commonly used in GRB afterglow modeling.

Numerical Scenarios
^^^^^^^^^^^^^^^^^^^

In the case of general shock scenarios, the relations above are not analytically tractable; however, they can be
solved efficiently with a simple numerical solver. Consider explicit solutions to :eq:`eq:relativistic_jump_condition_final`
under the condition that :math:`\beta_2 < \beta_1` (i.e. the shock is compressive). This condition is equivalent
to requiring that the shock is physical and not a rarefaction wave, which would have :math:`\beta_2 > \beta_1`.

In this case, we can solve for :math:`\beta_2` using a root-finding method. For our purposes in Triceratops, we elect
to use the bracketed root-finding method of :footcite:t:`brent2013algorithms`, which is a robust method for this
type of problem (see also :footcite:t:`2007nras.book.....P` for a pedagogical introduction).

State Reconstruction
^^^^^^^^^^^^^^^^^^^^

Once :math:`\beta_2` has been determined from the pre-shock state, the rest of the post-shock state must be reconstructed
from the remaining jump conditions. One immediately obtains the Lorentz factor :math:`\Gamma_2(\beta_2)` from

.. math::

    \Gamma_2 = \frac{1}{\sqrt{1-\beta_2^2}}.

Likewise, both the enthalpy and density may be reconstructed from the continuity and energy-momentum conditions:

.. math::

    \rho_2 = \rho_1 \frac{\Gamma_1 \beta_1}{\Gamma_2 \beta_2},

and

.. math::

    w_2 = w_1 \frac{\Gamma_1^2 \beta_1}{\Gamma_2^2 \beta_2}.

Finally, the pressure is obtained from the equation of state:

.. math::

    P_2 = \frac{\hat{\gamma} - 1}{\hat{\gamma}} \left(w_2 - \rho_2 c^2\right).

The (proper) internal energy density is then

.. math::

    U_{\rm int, 2} = \frac{P_2}{\hat{\gamma} - 1}.

and the (proper) total energy density is

.. math::

    e_2 = \rho_2 c^2 + U_{\rm int, 2}.

Assigning Post-Shock Temperatures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to assign a temperature to the fluid, one must invoke an equation of state for the fluid. A standard
Maxwellian distribution is no longer a sufficient choice for the relativistic fluid as the post-shock temperatures
will likely be relativistic. We therefore adopt a Maxwell-Juttner distribution
(see :footcite:p:`1996ApJ...465..327M, margalitThermalElectronsMildlyrelativistic2021` for a discussion; the original
paper :footcite:p:`juttner1911maxwellsche` is in German):

.. math::

    f(\gamma) = n \frac{\gamma^2 \beta}{\Theta K_2(1/\Theta)} \exp\!\left(-\frac{\gamma}{\Theta}\right),

where :math:`\Theta = k_B T / (m c^2)` is the dimensionless temperature, :math:`K_2` is the modified Bessel function
of the second kind, and :math:`N_0` is a normalization constant.

The internal energy density of a relativistic Maxwell-Juttner
distribution is given by\ :footcite:p:`1998ApJ...498..313G, 2000ApJ...541..234O, chandrasekhar1957introduction`

.. math::

    U_{\rm int} = \rho c^2 \left[\frac{3K_3(1/\Theta) + K_1(1/\Theta)}{4K_2(1/\Theta)} -1 \right],

which, in turn, is well approximated by\ :footcite:p:`1998ApJ...498..313G`

.. math::

    U_{\rm int}  \approx \rho c^2 \Theta \left(\frac{6+15\Theta}{4+5\Theta}\right)

To determine the temperature to assign to the post-shock fluid, we may invert this expression. Letting
:math:`\xi = U_{\rm int} / (\rho c^2)`, we have

.. math::

    \Theta = \frac{5\xi - 6 + \sqrt{(6-5\xi)^2 + 120\xi}}{30}.

.. note::

    Some sources in the literature (e.g. :footcite:t:`margalitThermalElectronsMildlyrelativistic2021`) instead
    invoke the classical shock jumps to assign relativistic temperatures, which is valid when the shock itself is
    only mildly relativistic, but the post-shock material is relativistically hot. This is a reasonable approximation
    in some cases, but the full relativistic jump conditions are required to assign accurate temperatures
    in the general case.


References
-----------

.. footbibliography::
