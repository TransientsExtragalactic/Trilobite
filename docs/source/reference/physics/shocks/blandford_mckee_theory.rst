.. _blandford_mckee_theory:

=====================================
Theory: The Blandford--McKee Solution
=====================================

The Blandford--McKee solution describes the self-similar evolution of an
ultra-relativistic blastwave expanding into an external medium\ :footcite:p:`1976PhFl...19.1130B`.
It is the relativistic analog of the
Sedov--Taylor blastwave\ :footcite:p:`sedov1946propagation`, but with an important physical difference: the shocked
material is compressed into a very thin shell behind the shock front.

This solution is especially useful for modeling relativistic explosions such as
gamma-ray burst afterglows and other transients in which the shocked material
moves with Lorentz factor :math:`\Gamma \gg 1`.

.. contents:: On This Page
   :local:
   :depth: 2

----

Overview and Assumptions
------------------------

We consider a **relativistic blastwave** produced by an explosion with (lab-frame) energy
:math:`E`, expanding into an initially **cold external medium**. The blastwave is
relativistic when the explosion energy greatly exceeds the rest-mass energy of
the swept-up material,

.. math::

   E \gg \left(M_{\rm ej} + \rho V\right)c^2.

Equivalently, most of the energy is carried as bulk kinetic energy and internal
energy of relativistically moving shocked gas.

.. admonition:: Model Assumptions

   - The upstream external medium is **cold** and **initially at rest** in the lab
     frame.

   - The blastwave is **adiabatic**: there is no radiative energy loss and no
     external work other than sweeping up ambient material.

   - The shocked gas is **ultra-relativistically hot**, so its equation of state is

     .. math::

        e = 3p,
        \qquad
        w=e+p=4p,

     where :math:`e` is the proper internal energy density, :math:`p` is the
     pressure, and :math:`w` is the relativistic enthalpy density.

   - The shock Lorentz factor satisfies :math:`\Gamma \gg 1`.

   - The **shocked region is geometrically thin**,

     .. math::

        \Delta R \sim \frac{R}{\Gamma^2} \ll R.

The solution is self-similar because, in the ultra-relativistic limit, the
structure behind the shock depends only on the distance behind the shock in
units of the relativistic shell thickness.

Relativistic Shock Jump Conditions
----------------------------------

Let :math:`\Gamma` be the Lorentz factor of the shock measured in the rest
frame of the unshocked upstream gas. In the strong, ultra-relativistic limit,
the relativistic Rankine--Hugoniot conditions
(see :ref:`relativistic_jump_conditions_theory`) imply that the downstream gas
immediately behind the shock has

.. math::

   \gamma_2^2 \simeq \frac{\Gamma^2}{2},

.. math::

   p_2 \simeq \frac{2}{3}w_1\Gamma^2,

and

.. math::

   n_2' \simeq 2\Gamma^2 n_1.

Here :math:`\gamma_2` is the Lorentz factor of the shocked gas in the lab
frame, :math:`p_2` is the downstream pressure, :math:`w_1` is the upstream
enthalpy density, :math:`n_1` is the upstream proper number density, and
:math:`n_2'` is the downstream lab-frame number density. For a cold upstream
medium,

.. math::

   w_1 \simeq \rho_1 c^2.

These relations provide the natural normalizations for the Blandford--McKee
self-similar profiles. Their derivation, including the full set of relativistic
flux-conservation equations and the analytic :math:`\beta_2 = 1/3` result for a
cold upstream medium, is worked through in :ref:`relativistic_jump_conditions_theory`.

Relativistic Hydrodynamic Equations
-----------------------------------

The dynamics of the shocked flow follow from stress-energy conservation,

.. math::

   \nabla_\mu T^{\mu\nu}=0.

For a perfect fluid,

.. math::

   T^{\mu\nu}
   =
   w u^\mu u^\nu + p g^{\mu\nu},

where

.. math::

   w=e+p,
   \qquad
   u^\mu=\gamma(1,\beta,0,0),
   \qquad
   \gamma=\frac{1}{\sqrt{1-\beta^2}}.

In spherical symmetry, and using units with :math:`c=1`, energy conservation is

.. math::

   \frac{\partial}{\partial t}
   \left[
      (e+p)\gamma^2-p
   \right]
   +
   \frac{1}{r^2}
   \frac{\partial}{\partial r}
   \left[
      r^2(e+p)\gamma^2\beta
   \right]
   =
   0,

while radial momentum conservation is

.. math::

   \frac{\partial}{\partial t}
   \left[
      (e+p)\gamma^2\beta
   \right]
   +
   \frac{1}{r^2}
   \frac{\partial}{\partial r}
   \left[
      r^2(e+p)\gamma^2\beta^2
   \right]
   +
   \frac{\partial p}{\partial r}
   =
   0.

For the ultra-relativistic equation of state,

.. math::

   e=3p,
   \qquad
   w=4p,

these equations may be recast into the useful Lagrangian form

.. math::

   \frac{d}{dt}\left(p\gamma^4\right)
   =
   \gamma^2\frac{\partial p}{\partial t},

and

.. math::

   \frac{d}{dt}
   \log\left(p^3\gamma^4\right)
   =
   -\frac{4}{r^2}
   \frac{\partial}{\partial r}
   \left(r^2\beta\right),

where

.. math::

   \frac{d}{dt}
   =
   \frac{\partial}{\partial t}
   +
   \beta\frac{\partial}{\partial r}

is the derivative following the flow.

Baryon number conservation gives

.. math::

   \nabla_\mu(nu^\mu)=0.

In spherical symmetry,

.. math::

   \frac{\partial n'}{\partial t}
   +
   \frac{1}{r^2}
   \frac{\partial}{\partial r}
   \left(r^2 n'\beta\right)
   =
   0,

where

.. math::

   n'=\gamma n

is the lab-frame number density and :math:`n` is the proper comoving density.
Combining baryon conservation with the ultra-relativistic equation of state
gives

.. math::

   \frac{d}{dt}
   \left(
      \frac{p}{n^{4/3}}
   \right)
   =
   0.

This is simply conservation of entropy along each fluid worldline.

.. dropdown:: Derivation: Relativistic Conservation Equations
   :icon: book-open

   The stress-energy tensor of a perfect fluid is

   .. math::

      T^{\mu\nu}
      =
      w u^\mu u^\nu + p g^{\mu\nu}.

   In flat spacetime and spherical symmetry, the relevant components are

   .. math::

      T^{00}
      =
      w\gamma^2-p,

   .. math::

      T^{0r}
      =
      w\gamma^2\beta,

   and

   .. math::

      T^{rr}
      =
      w\gamma^2\beta^2+p.

   Energy conservation is

   .. math::

      \partial_t T^{00}
      +
      \frac{1}{r^2}
      \partial_r\left(r^2T^{0r}\right)
      =
      0.

   Substituting the tensor components gives

   .. math::

      \frac{\partial}{\partial t}
      \left[
         w\gamma^2-p
      \right]
      +
      \frac{1}{r^2}
      \frac{\partial}{\partial r}
      \left[
         r^2w\gamma^2\beta
      \right]
      =
      0.

   Momentum conservation is

   .. math::

      \partial_t T^{0r}
      +
      \frac{1}{r^2}
      \partial_r\left(r^2T^{rr}\right)
      -
      \frac{2p}{r}
      =
      0.

   Since

   .. math::

      \frac{1}{r^2}
      \partial_r\left(r^2p\right)
      -
      \frac{2p}{r}
      =
      \partial_r p,

   the momentum equation becomes

   .. math::

      \frac{\partial}{\partial t}
      \left[
         w\gamma^2\beta
      \right]
      +
      \frac{1}{r^2}
      \frac{\partial}{\partial r}
      \left[
         r^2w\gamma^2\beta^2
      \right]
      +
      \frac{\partial p}{\partial r}
      =
      0.

   For an ultra-relativistic gas, :math:`w=4p`. The resulting equations can be
   combined into the Lagrangian forms used by Blandford and McKee.

Thickness of the Relativistic Shell
-----------------------------------

The **first major difference between the Blandford--McKee solution and the
Sedov--Taylor solution is the thickness of the shocked region**.

In a non-relativistic blastwave, the shocked region occupies an order-unity
fraction of the shock radius,

.. math::

   \Delta R \sim R.

In an ultra-relativistic blastwave, **the shocked material is compressed into a
thin shell**. A simple mass-conservation argument gives this thickness.

The swept-up rest mass inside radius :math:`R` is

.. math::

   M_{\rm sw}
   \sim
   \frac{4\pi}{3}R^3\rho_1.

If this material is concentrated into a shell of lab-frame thickness
:math:`\Delta R`, then

.. math::

   M_{\rm sw}
   \sim
   4\pi R^2\Delta R\,\rho_2',

where :math:`\rho_2'` is the downstream lab-frame density. Since the shock jump
conditions imply

.. math::

   \rho_2' \sim \Gamma^2\rho_1,

we obtain

.. math::

   \frac{4\pi}{3}R^3\rho_1
   \sim
   4\pi R^2\Delta R\,\Gamma^2\rho_1.

Thus,

.. math::

   \Delta R
   \sim
   \frac{R}{\Gamma^2}.

.. admonition:: Key Result: Relativistic Shell Thickness

   .. math::

      \Delta R \sim \frac{R}{\Gamma^2}\ll R.

This thin-shell structure motivates a similarity coordinate that resolves
distances behind the shock in units of :math:`R/\Gamma^2`, rather than in units
of :math:`R`.

Deceleration Law
----------------

For an adiabatic relativistic blastwave, the total energy is conserved. The
swept-up mass scales as

.. math::

   M_{\rm sw}\sim \rho_1 R^3.

For an ultra-relativistic shock, the energy per unit swept-up rest mass is
enhanced by a factor of order :math:`\Gamma^2`, so

.. math::

   E
   \sim
   \Gamma^2 M_{\rm sw}c^2
   \sim
   \Gamma^2\rho_1R^3c^2.

For a uniform external medium and fixed explosion energy,

.. math::

   \Gamma^2R^3 = \text{constant}.

Since the shock is ultra-relativistic,

.. math::

   R\simeq ct,

and therefore

.. math::

   \Gamma^2 \propto t^{-3}.

More generally, we write

.. math::

   \Gamma^2 \propto t^{-m},

where :math:`m` is a deceleration index. The adiabatic, energy-conserving
blastwave in a uniform external medium has

.. math::

   m=3.

The shock radius may be written, to the order needed to resolve the shell, as

.. math::

   R(t)
   =
   ct
   \left[
      1-\frac{1}{2(m+1)\Gamma^2}
   \right].

In units with :math:`c=1`,

.. math::

   R(t)
   =
   t
   \left[
      1-\frac{1}{2(m+1)\Gamma^2}
   \right].

The correction is small, of order :math:`\Gamma^{-2}`, but it is the same order
as the fractional shell thickness and therefore must be retained in the
self-similar coordinate.

----

The Blandford--McKee Similarity Variable
----------------------------------------

A Sedov--Taylor-like coordinate,

.. math::

   \xi_{\rm ST}=\frac{r}{R(t)},

is not useful in the relativistic problem because nearly all of the shocked
fluid lies in an extremely narrow region near :math:`\xi_{\rm ST}=1`.

Instead, define a coordinate that measures distance behind the shock in units of
the relativistic shell thickness:

.. math::

   \xi
   =
   \left(1-\frac{r}{R}\right)\Gamma^2.

Since

.. math::

   1-\frac{r}{R}
   =
   \frac{R-r}{R},

we have

.. math::

   \xi
   =
   \frac{R-r}{R/\Gamma^2}.

Thus, :math:`\xi` is order unity throughout the shocked shell.

Blandford and McKee define the closely related coordinate

.. math::

   \chi
   =
   1+2(m+1)\Gamma^2
   \left(1-\frac{r}{R}\right).

The shock front is located at

.. math::

   r=R
   \qquad\Longrightarrow\qquad
   \chi=1.

Material farther behind the shock has larger :math:`\chi`.

.. admonition:: Key Result: Blandford--McKee Similarity Coordinate

   .. math::

      \chi
      =
      1+2(m+1)\Gamma^2
      \left(1-\frac{r}{R}\right).

The numerical factor :math:`2(m+1)` is chosen for algebraic convenience. The
essential physical point is that :math:`\chi` resolves the thin relativistic
shell rather than the full blastwave radius.


Self-Similar Ansatz
-------------------

The shock jump conditions provide natural normalizations for the downstream
profiles. We introduce dimensionless functions :math:`f(\chi)`,
:math:`g(\chi)`, and :math:`h(\chi)` by

.. math::

   p(r,t)
   =
   \frac{2}{3}w_1\Gamma^2(t) f(\chi),

.. math::

   \gamma^2(r,t)
   =
   \frac{1}{2}\Gamma^2(t) g(\chi),

and

.. math::

   n'(r,t)
   =
   2n_1\Gamma^2(t) h(\chi).

The shock boundary conditions are then

.. math::

   f(1)=g(1)=h(1)=1.

Substituting this ansatz into the relativistic hydrodynamic equations reduces
the partial differential equations to ordinary differential equations in
:math:`\chi`.

For the adiabatic energy-conserving solution in a uniform external medium,
:math:`m=3`, the Blandford--McKee profiles are

.. math::

   g(\chi)=\chi^{-1},

.. math::

   f(\chi)=\chi^{-17/12},

and

.. math::

   h(\chi)=\chi^{-7/4}.

Thus,

.. math::

   \gamma^2(r,t)
   =
   \frac{1}{2}\Gamma^2\chi^{-1},

.. math::

   p(r,t)
   =
   \frac{2}{3}w_1\Gamma^2\chi^{-17/12},

and

.. math::

   n'(r,t)
   =
   2n_1\Gamma^2\chi^{-7/4}.

.. dropdown:: Derivation: Self-Similar Profiles for the Uniform Medium Case
   :icon: book-open

   The ansatz

   .. math::

      p =
      \frac{2}{3}w_1\Gamma^2 f(\chi),
      \qquad
      \gamma^2 =
      \frac{1}{2}\Gamma^2 g(\chi),
      \qquad
      n' =
      2n_1\Gamma^2h(\chi)

   is normalized so that :math:`f(1)=g(1)=h(1)=1` at the shock.

   For an energy-conserving blastwave in a uniform external medium,

   .. math::

      \Gamma^2\propto t^{-3},

   so :math:`m=3`. The relativistic hydrodynamic equations reduce to a set of
   ordinary differential equations in :math:`\chi`. Solving those equations
   with the shock boundary conditions gives

   .. math::

      g(\chi)=\chi^{-1},

   .. math::

      f(\chi)=\chi^{-17/12},

   and

   .. math::

      h(\chi)=\chi^{-7/4}.

   These power laws describe how the Lorentz factor, pressure, and lab-frame
   density decline moving inward from the shock front.

----

Range of the Similarity Variable
--------------------------------

The shocked region has thickness

.. math::

   \Delta R \sim \frac{R}{\Gamma^2}.

Therefore, throughout the main shocked shell,

.. math::

   1-\frac{r}{R}
   \sim
   \frac{1}{\Gamma^2}.

The similarity coordinate then satisfies

.. math::

   \chi-1
   =
   2(m+1)\Gamma^2
   \left(1-\frac{r}{R}\right)
   \sim
   2(m+1).

For the adiabatic uniform-medium solution, :math:`m=3`, so

.. math::

   \chi \sim 1+8

across the main shell, up to factors of order unity. The formal similarity
solution may be evaluated to larger :math:`\chi`, but most of the energy is
concentrated in the relativistically thin region behind the shock.

----

Energy Normalization
--------------------

The total blastwave energy is obtained by integrating the lab-frame energy
density over the shocked region. For an ultra-relativistic gas,

.. math::

   T^{00}
   =
   (e+p)\gamma^2-p
   =
   4p\gamma^2-p
   \simeq
   4p\gamma^2,

because :math:`\gamma^2\gg1`.

The energy is therefore

.. math::

   E
   =
   4\pi
   \int_0^R
   4p\gamma^2r^2\,dr.

Using the similarity forms,

.. math::

   p =
   \frac{2}{3}w_1\Gamma^2 f(\chi),

and

.. math::

   \gamma^2 =
   \frac{1}{2}\Gamma^2 g(\chi),

we find

.. math::

   4p\gamma^2
   =
   \frac{4}{3}w_1\Gamma^4 f(\chi)g(\chi).

The coordinate transformation is, to leading order,

.. math::

   d\chi
   =
   -2(m+1)\Gamma^2\frac{dr}{R},

so

.. math::

   dr
   =
   -\frac{R}{2(m+1)\Gamma^2}d\chi.

Because the shell is thin, :math:`r\simeq R` inside the integral. Therefore,

.. math::

   E
   \simeq
   \frac{8\pi}{3(m+1)}
   w_1\Gamma^2R^3
   \int_1^{\chi_{\rm max}}
   f(\chi)g(\chi)\,d\chi.

For the adiabatic uniform-medium solution,

.. math::

   f(\chi)g(\chi)
   =
   \chi^{-17/12}\chi^{-1}
   =
   \chi^{-29/12}.

The integral converges:

.. math::

   \int_1^\infty \chi^{-29/12}d\chi
   =
   \frac{12}{17}.

Since :math:`m=3`, we obtain

.. math::

   E
   =
   \frac{8\pi}{17}
   w_1\Gamma^2R^3.

For a cold upstream medium,

.. math::

   w_1\simeq\rho_1c^2,

so

.. math::

   E
   =
   \frac{8\pi}{17}
   \rho_1c^2\Gamma^2R^3.

Solving for the shock Lorentz factor gives

.. math::

   \Gamma^2(R)
   =
   \frac{17E}{8\pi\rho_1c^2R^3}.

Since :math:`R\simeq ct`, this reproduces the scaling

.. math::

   \Gamma^2\propto t^{-3}.

.. admonition:: Key Result: Blandford--McKee Energy Normalization

   .. math::

      E
      =
      \frac{8\pi}{17}
      \rho_1c^2\Gamma^2R^3.

   Equivalently,

   .. math::

      \Gamma^2(R)
      =
      \frac{17E}{8\pi\rho_1c^2R^3}.

.. dropdown:: Derivation: Energy Integral
   :icon: book-open

   The lab-frame energy density is

   .. math::

      T^{00}
      =
      (e+p)\gamma^2-p.

   For an ultra-relativistically hot gas,

   .. math::

      e=3p,
      \qquad
      e+p=4p.

   Therefore,

   .. math::

      T^{00}
      =
      4p\gamma^2-p.

   In the ultra-relativistic limit, :math:`\gamma^2\gg1`, so

   .. math::

      T^{00}\simeq4p\gamma^2.

   The energy is

   .. math::

      E
      =
      4\pi
      \int_0^R
      T^{00}r^2dr
      \simeq
      4\pi
      \int_0^R
      4p\gamma^2r^2dr.

   Substituting

   .. math::

      p =
      \frac{2}{3}w_1\Gamma^2 f(\chi),
      \qquad
      \gamma^2 =
      \frac{1}{2}\Gamma^2g(\chi),

   gives

   .. math::

      4p\gamma^2
      =
      \frac{4}{3}w_1\Gamma^4f(\chi)g(\chi).

   Since

   .. math::

      \chi =
      1+2(m+1)\Gamma^2
      \left(1-\frac{r}{R}\right),

   we have

   .. math::

      dr
      =
      -\frac{R}{2(m+1)\Gamma^2}d\chi.

   The shocked shell is thin, so :math:`r^2\simeq R^2`. Hence

   .. math::

      E
      \simeq
      \frac{8\pi}{3(m+1)}
      w_1\Gamma^2R^3
      \int_1^\infty f(\chi)g(\chi)d\chi.

   For the uniform-medium Blandford--McKee solution,

   .. math::

      f(\chi)g(\chi)=\chi^{-29/12}.

   Thus,

   .. math::

      \int_1^\infty \chi^{-29/12}d\chi
      =
      \frac{12}{17}.

   Setting :math:`m=3` gives

   .. math::

      E
      =
      \frac{8\pi}{17}
      w_1\Gamma^2R^3.

References
----------
.. footbibliography::
