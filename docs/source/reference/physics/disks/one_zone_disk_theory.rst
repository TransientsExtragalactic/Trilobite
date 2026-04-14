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

.. hint::

    Throughout our implementation, we follow the conventions of :footcite:t:`metzgerTimeDependentModelsAccretion2008`
    and :footcite:t:`piroLatetimeEvolutionInstabilities2025`.

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

enforces a zero-torque inner boundary condition (:math:`F_0 = 1.6` in :footcite:p:`metzgerTimeDependentModelsAccretion2008`
).  Matter draining through the inner edge at
:math:`R_{\rm in}` carries the local Keplerian specific angular momentum
:math:`\ell_{\rm in} = \sqrt{G M_{\rm BH} R_{\rm in}}`, so

.. math::

    \dot{J}_{\rm acc} = \dot{M}_{\rm acc}\,\sqrt{G M_{\rm BH} R_{\rm in}}.

Because :math:`\ell_{\rm in} \ll J_D / M_D` when :math:`R_{\rm in} \ll R_D`,
angular momentum drains more slowly than mass and the disk outer radius grows over
time — the standard viscous spreading behaviour.

Timestepping
^^^^^^^^^^^^

In order to correctly capture the evolution of the disk in these various scenarios, we follow the prescription of
:footcite:t:`2025ApJ...985...77P` and utilize an adaptive timestep which effectively requires that the mass not evolve
too greatly between timesteps:

.. math::

    \Delta t = \epsilon \min\left(\frac{M_D}{|\dot{M}_D|}, \frac{J_D}{|\dot{J}_D|}, t_{\rm visc}\right).

----

Thermodynamics
--------------

The above equations would be entirely sufficient to specify the state of the disk were it not for the
dependence on :math:`t_{\rm visc}`, which depends on the viscosity :math:`\nu` and therefore on the thermodynamic
state of the disk. Because of this, it becomes **necessary to treat (in some detail) the microphysics of the disk**. Many
options exist in the literature for how to do this in different scenarios and regimes; however, Triceratops
aims to be as flexible as possible in this regard, and so the thermodynamic closure is implemented as a modular
component that can be swapped out for different physics choices.  The following sections describe
the general framework for the thermodynamic closure and the specific closures currently implemented in Triceratops.

Overview
^^^^^^^^

In order to fully specify the evolution of a disk, choices must be made for each of the following properties
of the disk:

- **The Viscosity**: the parameterization of the viscosity (e.g. the :math:`\alpha`-prescription) and the physical
  processes that contribute to it (e.g. turbulence, magnetic fields, self-gravity). This should result in a
  function :math:`\nu(\boldsymbol{\Theta}_D)` that gives the viscosity as a function of the disk state.
- **The Equation of State (EoS)**: the relation between the thermodynamic variables (e.g. pressure, temperature, density)
  that defines the sound speed :math:`c_s` and therefore the disk scale height :math:`H = c_s / \Omega`. This should
  produce a function :math:`c_s(\boldsymbol{\Theta}_D)` that gives the sound speed as a function of the disk state as
  well as a function :math:`P(\boldsymbol{\Theta}_D)` that gives the pressure as a function of the disk state.
- **The Opacity**: the relation between the thermodynamic variables that defines the opacity :math:`\kappa` and
  therefore the optical depth :math:`\tau = \kappa \Sigma`. This should produce a function
  :math:`\kappa(\boldsymbol{\Theta}_D)` that gives the opacity as a function of the disk state.
- **The Heating and Cooling Processes**: the physical processes that contribute to the heating and cooling of the disk
  (e.g. viscous heating, radiative cooling, neutrino cooling) and therefore the thermal structure of the disk. This should
  produce functions :math:`Q^+(\boldsymbol{\Theta}_D)` and :math:`Q^-(\boldsymbol{\Theta}_D)` that give the
  heating and cooling rates as a function of the disk state.

Once each of these processes has been specified, the thermal structure of the disk can (in principle) be solved for. In
some sense, this is the artful element of constructing these models: the choices made for each of these processes will
determine the physical regimes that the model can capture and the computational cost of solving for the thermal structure.

In the following sections, we will describe the closures which are included explicitly in Triceratops, which are
designed to capture the physics of accretion disks in a variety of regimes while still being computationally efficient.
We will also describe the general framework for how these closures are implemented
in the code, and how users can extend this framework to include their own closures if desired.

Viscosity
^^^^^^^^^

The viscous parameterization is generally the first choice to be made when constructing a disk model as it determines
the set of available relations between various disk properties. By default, Triceratops only provides the **alpha-prescription** for viscosity,
which assumes that the viscosity can be expressed as

.. math::
    :label: eq:alpha_viscosity

    \boxed{
    \nu = \alpha\, c_s\, H,
    }

where :math:`\alpha` is a dimensionless parameter that encapsulates the efficiency of angular momentum transport in
the disk, :math:`c_s` is the sound speed, and :math:`H` is the disk scale height.
This prescription is widely used in the literature due to its simplicity and ability to capture a wide range
of physical processes that can contribute to viscosity (e.g. turbulence, magnetic fields, self-gravity) without
needing to model them explicitly.

In most cases, the scale height is not a primitive variable of the model but is instead derived from the sound speed and
the local Keplerian angular velocity on the basis of vertical hydrostatic equilibrium:

.. math::
    :label: eq:scale_height

    H = \frac{c_s}{\Omega}.

.. note::

    This assertion is generally well supported on the basis that :math:`t_{\rm thermal} \ll t_{\rm visc}`, so the disk can
    be assumed to be in thermal equilibrium at all times. However, this may not hold in some regimes (e.g. during rapid
    transitions between different accretion states) and so users should be aware of
    this assumption when constructing their models.

Equation of State
^^^^^^^^^^^^^^^^^

By default, Triceratops provides two closures for the equation of state:

1. **Gas Pressure Only** — assumes that the pressure is dominated by an ideal gas of mean molecular weight :math:`\mu`.
2. **Gas + Radiation Pressure** — includes both the ideal gas pressure and the radiation pressure.

.. tab-set::

    .. tab-item:: Gas Pressure Only

        The equation of state is

        .. math::

            P = \frac{\rho k_B T_c}{\mu m_p},

        where :math:`\rho` is the midplane density, :math:`T_c` is the midplane temperature, and
        :math:`\mu` is the mean molecular weight.  The sound speed is then

        .. math::

            c_s^2 = \frac{P}{\rho} = \frac{k_B T_c}{\mu m_p}.

        This is very convenient because the temperature can be deduced from heating equilibrium and the sound
        speed follows directly in a manner which does not require any root-finding.

    .. tab-item:: Gas + Radiation Pressure

        When the ideal gas is modified to include both the gas pressure contribution and the
        radiation pressure contribution, the equation of state becomes

        .. math::

            P = \frac{\rho k_B T_c}{\mu m_p} + \frac{a T_c^4}{3},

        where :math:`a` is the radiation constant.  The sound speed is then made considerably more complicated to
        obtain. Dividing both sides by :math:`\rho` gives

        .. math::

            c_s^2 = \frac{P}{\rho} = \frac{k_B T_c}{\mu m_p} + \frac{a T_c^4}{3 \rho}.

        the second term on the RHS can be replaced with the disk properties (:math:`\Sigma` and :math:`\Omega`) using
        vertical hydrostatic equilibrium:

        .. math::

            \rho = \frac{\Sigma}{\sqrt{2\pi} H} = \frac{\Sigma\,\Omega}{\sqrt{2\pi} \,c_s}.

        This then produces a quadratic equation for :math:`c_s` in terms of :math:`T_c`:

        .. math::
            :label: eq:cs_quadratic

            c_s^2 -
            \underbrace{\frac{\sqrt{2\pi}}{3}\frac{a T_c^4}{ \Sigma \Omega }}_{A(T)} c_s -
            \underbrace{\frac{k_B T_c}{\mu m_p}}_{B(T)} = 0.

        The physically meaningful (positive) root of this equation gives the sound speed. To ensure
        that the root is found robustly in numerical implementations, the quadratic formula is evaluated in
        a manner that avoids catastrophic cancellation:

        .. math::

            c_s(T_c)
            =
            \frac{A(T_c) + \sqrt{A(T_c)^2 + 4 B(T_c)}}{2}.

        This closure is more computationally expensive than the gas-only case because it requires
        root-finding to solve for the sound speed, but it is necessary to capture the physics of disks in
        regimes where radiation pressure becomes important (e.g. at high temperatures or low densities).

Energy Balance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the core of every closure is a simple energy balance condition which requires that the local heating
and cooling be in equilibrium at the midplane temperature :math:`T_c`:

.. math::
    :label: eq:energy_balance

    Q^+(T_c) = Q^-(T_c).

Depending on the choices made for the viscosity, equation of state, and opacity, this condition can be reduced
to a single algebraic equation in :math:`T_c` that can be solved for the thermal
structure of the disk at each timestep. Several heating terms appear in various different disk closures:

.. tab-set::

    .. tab-item:: Viscous Heating

        The viscous heating rate per unit area is (in the alpha prescription with vertical hydrostatic equilibrium)

        .. math::
            :label: eq:q_plus

            Q^+ = \frac{9}{8}\,\nu\,\Sigma\,\Omega^2
                = \frac{9}{8}\,\alpha\,\frac{c_s^2}{\Omega}\,\Sigma\,\Omega^2
                = \frac{9}{8}\,\alpha\,c_s^2\,\Sigma\,\Omega,

        In this form, this is a function only of the thermodynamics and the structure of the disk and is
        therefore easily evaluated once the sound speed is known.  This is the dominant heating term in most
        disk models, but other terms can be added to capture additional physics (e.g. neutrino heating,
        nuclear burning) if desired.

    .. tab-item:: Radiative Cooling

        Radiative cooling can be done in a number of ways with varying levels of sophistication. Tradiationally,
        one takes some prescription for the Rosseland opacity :math:`\kappa_R(T_c, \rho_c)` and utilizes radiative
        diffusion to approximate the cooling rate. This, of course, presumes that the disk is optically thick
        and that the diffusion approximation is valid, which may not hold in all regimes.

        When radiative diffusion is relevant, we presume that the surface radiates at some blackbody temperature
        :math:`T_{\rm eff}` and that the midplane temperature is related to the effective temperature through the
        optical depth of the disk:

        .. math::
            :label: eq:tc_teff

            T_c^4 = \frac{3}{4}\,\tau\,T_{\rm eff}^4,

        where the optical depth is given by :math:`\tau = \kappa_R \Sigma`.  The radiative cooling rate per unit area
        is then

        .. math::

            Q^- = \sigma_{\rm SB}\,T_{\rm eff}^4 =
            \frac{4 \sigma_{\rm SB}}{3 \tau} T_c^4 =
            \frac{16 \sigma_{\rm SB}}{3 \kappa_R \Sigma} T_c^4.

        .. note::

            The :math:`\frac{16}{3}` factor corresponds to our choice to use :math:`\Sigma = \sqrt{2\pi} \rho H` as the
            surface density in the optical depth, which is motivated by the fact that the disk photosphere is located at a height
            of order the scale height above the midplane.  Other choices are also present in the literature.

        The choice of opacity is also an important consideration. At this point, we implement only electron scattering
        opacity; however, we will eventually support both Kramer's opacity and the more sophisticated OPAL opacity tables,
        which will allow us to capture a wider range of physical regimes.

    .. tab-item:: Advection


        In some regimes, advection can become an important cooling term. This is particularly relevant in the context of
        radiatively inefficient accretion flows (RIAFs), where the disk is optically thin and the cooling timescale is long
        compared to the accretion timescale. In this case, the advective cooling rate per unit area can be approximated as

        .. math::

            Q^-_{\rm adv} = \xi \frac{\dot{M}}{2\pi R_D^2} c_s^2,

        where :math:`\dot{M}` is the mass accretion rate and :math:`\xi` is the logarithmic entropy gradient and is
        typically of order unity (1.5 is typical, see :footcite:t:`2007PASJ...59..443W`).
        This term can be added to the total cooling rate to capture the physics of RIAFs and other
        regimes where advection is important.


Once all of these specifications have been made, the energy balance condition can be solved for the midplane
temperature :math:`T_c` at each timestep, which then allows for the evaluation of the sound speed, scale height,
viscosity, and therefore the viscous timescale.  This closes the system of equations and allows for the integration
of the disk evolution over time.

Additional Physics
------------------

As we have described in the preceding section, there are many modifications which may be made to the canonical
disk scenarios to implement new physics. In this section, we'll describe the theory behind a few of the common
modifications that are implemented in Triceratops.
Fallback
^^^^^^^^

In many astrophysical systems, the accretion disk is continuously supplied with mass
from an external reservoir. In the context of tidal disruption events (TDEs), this
corresponds to the fallback of stellar debris onto the disk. This process modifies the
standard viscous evolution by introducing source terms in both the mass and angular
momentum equations.

For a parabolic TDE, the late-time fallback rate follows a characteristic power-law decay,
set by the orbital period of the most bound debris:

.. math::

    \dot{M}_{\rm FB}
    =
    \frac{M_\star}{5 t_{\rm fb}}
    \left(\frac{t}{t_{\rm fb}}\right)^{-5/3},

where the fallback timescale is

.. math::

    t_{\rm fb}
    =
    \frac{\pi R_t^3}{\sqrt{2 G M_\bullet} R_\star^{3/2}}
    \approx
    3.5\times 10^6\;{\rm s}\;
    M_6^{1/2} m_\star^{-1} r_\star^{3/2}.

The fallback material carries angular momentum corresponding to the circularization
radius, which we approximate as

.. math::

    \ell_{\rm FB} = \sqrt{G M_\bullet R_{\rm circ}}.

To allow for greater flexibility, Triceratops adopts a generalized parameterization of
the fallback rate:

.. math::

    \begin{aligned}
    \dot{M}_{\rm FB} &= \dot{M}_0
    \left(\frac{t}{t_0}\right)^{-\alpha_{\rm FB}}, \\
    \dot{J}_{\rm FB} &= \sqrt{G M_\bullet R_{\rm circ}}\;\dot{M}_{\rm FB},
    \end{aligned}

where :math:`\dot{M}_0`, :math:`t_0`, :math:`\alpha_{\rm FB}`, and :math:`R_{\rm circ}`
are free parameters that can be tuned to represent a range of fallback scenarios.

.. note::

    The current implementation does not include heating associated with fallback
    dissipation. This contribution can, in principle, be incorporated as an additional
    term in the energy balance condition. See :footcite:t:`2025ApJ...985...77P` for
    a discussion of the potential importance of fallback heating in TDE disks.


Wind Launching
^^^^^^^^^^^^^^

In many accretion disks, the accretion rate can exceed the Eddington limit, leading to
the launching of powerful radiatively driven winds. These winds remove both mass and
angular momentum from the disk, modifying the standard viscous evolution through
additional sink terms.

A common parameterization assumes that the accretion rate decreases with radius due
to cumulative mass loss (e.g. :footcite:t:`begelmanCanSphericallyAccreting1979`):

.. math::

    \dot{M}(r)
    =
    \dot{M}_{\rm sph}
    \left(\frac{r}{R_{\rm sph}}\right)^p,

where :math:`\dot{M}_{\rm sph}` is the accretion rate at the spherization radius
:math:`R_{\rm sph}`, and :math:`p` controls the strength of the wind.

The spherization radius marks the transition where radiation pressure becomes dynamically
important and can be approximated as

.. math::

    R_{\rm sph}
    =
    \min\left(
        R_D,
        \frac{\kappa \dot{M}}{4\pi c}
    \right).

Integrating the radial mass loss profile yields the total mass outflow rate:

.. math::

    \dot{M}_{\rm out}
    =
    \left[
        1 - \left(\frac{R_{\rm in}}{R_{\rm sph}}\right)^p
    \right]
    \dot{M}_{\rm sph}.

The associated angular momentum loss is obtained by weighting the outflow by the
local Keplerian specific angular momentum, giving

.. math::

    \frac{dJ_D}{dt}
    =
    -\frac{2p}{2p+1}
    \sqrt{G M_{\rm BH} R_{\rm sph}}\;
    \dot{M}_{\rm sph}.

Irradiation
^^^^^^^^^^^

In many accretion flows, radiation produced near the central object can illuminate
the outer disk and modify its thermal structure. This scenario is referred to as an
**irradiated disk**, and is particularly relevant in systems with persistent high-energy
emission (e.g. X-ray binaries, active galactic nuclei, and disks surrounding neutron stars).
In such cases, irradiation can contribute significantly to the local heating budget.

We model the central radiation field as a point source of luminosity :math:`L`. At the
characteristic disk radius :math:`R_D`, the incident flux is

.. math::

    F_{\rm irr} = \frac{L}{4\pi R_D^2} \cos\psi,

where :math:`\psi` is the angle between the disk surface normal and the direction to the
central source. Only a fraction of this flux is absorbed by the disk, with the remainder
reflected. The corresponding heating rate per unit area is therefore

.. math::

    Q^+_{\rm irr} = (1 - \beta)\, F_{\rm irr},

where :math:`\beta` is the disk albedo :footcite:p:`frank2002accretion`.

The geometric factor :math:`\cos\psi` determines how efficiently the disk intercepts
the central radiation. In general, this depends on the global disk structure, including
its scale height and radial flaring. For a disk surface described by a height profile
:math:`H(R)`, the relevant angles can be written as

.. math::

    \psi = \frac{\pi}{2} - \theta + \phi,

where

.. math::

    \tan \theta = \frac{dH}{dR}, \qquad \tan \phi = \frac{H}{R}.

In the thin-disk limit (:math:`H \ll R`), these angles are small, and we may approximate

.. math::

    \cos\psi \approx \frac{dH}{dR} - \frac{H}{R}.

This expression highlights an important physical point: irradiation requires disk
flaring. A purely flat disk (:math:`H \propto R`) intercepts no central radiation,
while a flared disk (:math:`dH/dR > H/R`) presents a non-zero projected area to the
source and can therefore be efficiently heated.

Because the one-zone model does not resolve the radial structure of the disk, we
parametrize this geometric factor. Motivated by analytic thin-disk solutions, we assume
a power-law scaling of the form

.. math::

    H \propto R^{n_{\rm irr}},

which implies

.. math::

    \frac{dH}{dR} = n_{\rm irr} \frac{H}{R}.

Substituting into the expression for :math:`\cos\psi`, we obtain

.. math::

    \cos\psi \approx (n_{\rm irr} - 1)\,\frac{H}{R}.

The irradiating flux can therefore be written as

.. math::

    F_{\rm irr}
    =
    \frac{L}{4\pi R_D^2}\,(n_{\rm irr} - 1)\,\frac{H}{R}.

Including the albedo, the final expression for the irradiative heating term becomes

.. math::

    Q^+_{\rm irr}
    =
    (1 - \beta_{\rm irr})\,
    \frac{L}{4\pi R_D^2}\,(n_{\rm irr} - 1)\,\frac{H}{R}.

Typical values are :math:`n_{\rm irr} \approx 9/8` for viscously heated thin disks and
:math:`n_{\rm irr} \approx 9/7` for irradiation-dominated disks
:footcite:p:`frank2002accretion`. In practice, we treat :math:`n_{\rm irr}` as a free
parameter that encapsulates the uncertain large-scale geometry of the disk.

Specific Disk Models
--------------------
Having now covered the detailed theoretical framework for constructing one-zone disk models, we can now describe the
specific disk models that are currently implemented in Triceratops.  Each of these models is designed
to capture the physics of accretion disks in a particular regime while still being computationally efficient.

Standard Disks
^^^^^^^^^^^^^^

In this section, we describe the theoretical elements of our "standard" disk models

.. tab-set::

    .. tab-item:: gP + ES

        This model includes gas pressure and electron scattering opacity.  The heating is dominated by viscous
        dissipation and the cooling is dominated by radiative diffusion.  This is a very common model in the literature
        and captures the physics of many accretion disks in a variety of regimes.

        A fortunate feature of this disk is that the temperature can be computed from the energy balance condition
        analytically. The heating balance equation takes the form

        .. math::

            \frac{9}{8}\,\alpha\,\frac{k_B T_c}{\mu m_p}\,\Sigma\,\Omega =
            \frac{16 \sigma_{\rm SB}}{3 \kappa_{\rm es} \Sigma} T_c^4,

        which results in

        .. math::
            :label: eq:gas_pressure_temperature

            \boxed{
            T_c^3 = \frac{27}{128}\,\frac{\kappa_{\rm es} k_B}{\sigma_{\rm SB} \mu m_p}\,\alpha\,\Sigma^2\,\Omega.
            }

    .. tab-item:: igP + ES

        This model includes gas pressure and radiation pressure in a single equation of state, as well
        as electron scattering opacity.  The heating is dominated by viscous dissipation and the cooling
        is dominated by radiative diffusion.  This model is necessary to capture the physics of disks in
        regimes where radiation pressure becomes important (e.g. at high temperatures or low densities).

        Once again,

        .. math::

            \frac{9}{8}\,\alpha\,\frac{k_B T_c}{\mu m_p}\,\Sigma\,\Omega =
            \frac{16 \sigma_{\rm SB}}{3 \kappa_{\rm es} \Sigma} T_c^4,

        however, the sound speed is now a non-trivial function of the temperature. We therefore utilize a
        root finding algorithm which occurs in two steps:

        1. For a given proposal temperature, the sound speed is computed by solving the quadratic equation (eq:cs_quadratic).
        2. The energy balance condition is evaluated and the proposal temperature is updated until the balance condition
           is satisfied to within some tolerance.

        This then produces the correct balancing temperature.

Modified Dynamical Disks
^^^^^^^^^^^^^^^^^^^^^^^^

In disks with advection of heat as well as the standard balance between viscous heating and radiative cooling,
the energy balance condition becomes

.. math::

    \frac{9}{8}\,\alpha\,c_s^2\,\Sigma\,\Omega =
    \frac{16 \sigma_{\rm SB}}{3 \kappa_R \Sigma} T_c^4 +
    \xi \frac{\dot{M}}{2\pi R_D^2} c_s^2.

where the second term on the RHS captures the advective cooling.  This model is necessary to capture
the physics of radiatively inefficient accretion flows (RIAFs) and other regimes where advection is important.

In this case, the procedure largely mirrors that of the standard disks with the additional complication of
having additional terms which must be computed. Using the standard closure, we use

.. math::

    \dot{M} = f_D\,\frac{M_D}{t_{\rm visc}} = f_D\,\nu\,\frac{M_D}{R_D^2} = f_D\,\alpha\,\frac{c_s^2}{\Omega}\,\frac{M_D}{R_D^2},

to compute the mass accretion rate, which then allows for the evaluation of the advective cooling term.  This is then
added to the radiative cooling term and the energy balance condition is solved for the temperature as before.

Fallback Disks
^^^^^^^^^^^^^^^

In many astrophysical systems, the accretion disk is continuously supplied with mass from an external reservoir.
In the context of tidal disruption events (TDEs), this corresponds to the fallback of stellar debris onto the disk.
This mass supply modifies the standard viscous evolution by introducing a source term in the mass and angular
momentum equations.

We model the fallback accretion rate using a simple power-law prescription:

.. math::

    \dot{M}_{\rm fb}(t) = \dot{M}_0 \left(\frac{t}{t_0}\right)^{-5/3},

where :math:`\dot{M}_0` is the normalization of the fallback rate at a reference time :math:`t_0`.

The disk mass evolution equation then becomes

.. math::

    \dot{M}_D =
    -\dot{M}_{\rm acc}
    + \dot{M}_{\rm fb}
    - \dot{M}_{\rm outflow}.

The fallback material also carries angular momentum. Assuming the material circularizes at a characteristic
radius :math:`R_{\rm circ}`, the specific angular momentum of the inflow is

.. math::

    \ell_{\rm fb} = \sqrt{G M_{\rm BH} R_{\rm circ}}.

The angular momentum evolution equation is therefore

.. math::

    \dot{J}_D =
    -\dot{J}_{\rm acc}
    + \dot{M}_{\rm fb}\,\ell_{\rm fb}
    - \dot{J}_{\rm outflow}.

In the simplest implementation, the fallback radius is assumed to be comparable to the disk radius
(:math:`R_{\rm circ} \sim R_D`), so that the injected material has a similar specific angular momentum
to the bulk of the disk. More detailed treatments may instead track the circularization radius explicitly.

The inclusion of fallback fundamentally alters the disk evolution. At early times, when
:math:`\dot{M}_{\rm fb} \gg \dot{M}_{\rm acc}`, the disk mass grows and the disk can enter
high-accretion states. At late times, as the fallback rate declines, the disk transitions
to a viscously dominated evolution where

.. math::

    \dot{M}_D \approx -\dot{M}_{\rm acc},

and the system asymptotically approaches the standard spreading-disk solution.

----

.. footbibliography::
