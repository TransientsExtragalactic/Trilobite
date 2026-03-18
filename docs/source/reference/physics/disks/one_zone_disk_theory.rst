.. _one_zone_disk_theory:
=============================================
One-Zone Accretion Disk Models: Theory
=============================================

A common challenge in accretion disk modeling is that the characteristic timescale for disk evolution is the
**viscous timescale**,

.. math::
    :label: eq:viscous_timescale

    t_{\rm visc} = \alpha^{-1} \left(\frac{R}{H}\right)^2 \Omega^{-1},


where :math:`\alpha` is the Shakura--Sunyaev viscosity parameter :footcite:p:`frank2002accretion, 1973A&A....24..337S`,
:math:`H` is the disk scale height, and :math:`\Omega` is the orbital frequency. For a disk with a fixed aspect ratio
:math:`(H/R)`, the viscous timescale scales with radius as :math:`t_{\rm visc} \propto R^{3/2}`.

This strong radial dependence implies that different parts of the disk evolve on
widely different timescales: the inner disk evolves rapidly, while the outer disk
changes much more slowly. In principle, one could resolve this hierarchy of timescales
using adaptive timestepping or full numerical simulations. However, for many
applications it is sufficient to construct a simplified model that captures the
global evolution of the disk to order of magnitude.

This motivates the idea of a **one-zone accretion model**, in which we treat the entire disk at a single radius
and make the approximation that the disk properties at this radius are representative of the entire disk. This allows
for very efficient modeling of the disk evolution and inclusion of non-standard physical processes without too much
complexity or computational overhead. Of course, the clear downside of this approach is that it cannot capture the
detailed radial structure of the disk, and may not be accurate for all applications. However, it can be a useful
tool for gaining insight into the global behavior of accretion disks and for making order-of-magnitude
estimates of disk properties and evolution.

A number of one-zone disk models are provided in Triceratops, which can be used for a variety of modeling applications.
This document provides a brief overview of the theory behind one-zone disk models, and the assumptions and approximations
that go into them. For details on the usage of the one-zone disk models in Triceratops, see :ref:`one_zone_disk`.

.. contents::

----

The Core Model
----------------

.. note::

    We follow here the development of the one-zone formalism as described in
    :footcite:t:`metzgerTimeDependentModelsAccretion2008`.

We will begin by introducing the central tenants of the core model. These are features of one-zone disks which are
present in effectively all of the one-zone implementations provided in Triceratops.

At its surface, the one-zone model is motivated by the observation that the viscous timescale :math:`t_{\rm visc}` partitions
the disk into 3 regions:

- The **inner disk** where :math:`t_{\rm visc} \ll t` and the disk is effectively steady and is subdominant in mass because
  mass is rapidly accreted onto the central object.
- The **outer disk** where :math:`t_{\rm visc} \gg t` where the disk has yet to have time to evolve away from its
  initial conditions.
- The **middle disk** where :math:`t_{\rm visc} \sim t` where the disk is evolving on a timescale comparable to
  the dynamical timescale, and is dominant in mass.

We therefore construct one-zone models to track the evolution of the entire disk using this middle disk region as a
proxy. In this formalism, the problem of evolution becomes effectively one of energy and momentum conservation. The
mass of the disk is

.. math::

    M_D = A \pi \Sigma_D R_D^2,

where :math:`A=3.62` is a correction constant used to ensure the model matches the exact Green's function solution.
Likewise, the angular momentum of the disk is

.. math::

    J_D = B(G M_D R_D)^{1/2} \pi R_D^2 \Sigma_D,

where :math:`B=3.24` is another correction constant.

At its core, the one-zone model is quite simple: we track the evolution in time of the **disk mass** and the
**disk angular momentum**:

.. math::
    :label: eq:disk_evolution

    \begin{aligned}
    \dot{M}_D &= -\dot{M}_{\rm acc} + \dot{M}_{\rm inflow} - \dot{M}_{\rm outflows},\\
    \dot{J}_D &= -\dot{J}_{\rm acc} + \dot{J}_{\rm inflow} - \dot{J}_{\rm outflows}.
    \end{aligned}

The exact nature of the source terms on the RHS of the equation is determined by the model-specific physics.

The radius of the disk is determined by

.. math::

    R_D = \frac{1}{GM_{\rm BH}} \left(\frac{J_D}{\xi M_D}\right)^2, \;\; \xi = \frac{B}{A}.

and the surface density of the disk is simply

.. math::

    \Sigma_D = \frac{M_D}{A \pi R_D^2}.

In order to produce interesting behavior in the model, one must specify the relevant terms on the RHS of
:ref:`eq:disk_evolution`. At a minimum, one must specify the accretion terms (otherwise the disk would simply not
evolve). We again follow :footcite:t:`metzgerTimeDependentModelsAccretion2008` in adopting the form

.. math::

    \dot{M}_{\rm acc} = f \frac{M_D}{t_{\rm visc}} = f \nu \frac{M_D}{R_D^2},

where :math:`\nu` is the viscosity and :math:`f` is a correction factor of order unity.

.. note::

    Unlike :math:`A` and :math:`B`, the correction factor may have some model dependence. For the standard spreading
    ring solution (Green's function solution), one requires :math:`f=1.6`; however, we also generally want a zero-torque
    inner boundary condition, which requires

    .. math::

        f = \frac{1.6}{1 - \sqrt{R_{\rm in}/R_D}}.

    For a steady state disk,

    .. math::

        f = 3/A \sim 0.83.

At its most basic, this is the **one-zone model framework**; however, as will be made clear throughout this document,
quite a bit of interesting physics and behavior can be captured with this model.

Model Architecture
------------------

.. note::

    This section introduces some of the basics of the model architecture; however, a detailed discussion of the
    implementation and usage of the one-zone disk models in Triceratops is deferred to :ref:`one_zone_disk`. This is
    also where one should look for instructions on implementing new one-zone disk models.

At its core, each one-zone model is a coupled set of ODEs for the disk mass and the angular momentum. Thus,
for each timestep, one computes the relevant derivatives and then integrates forward in time.

.. figure:: disk_diagram_1

A complicating factor is that, in general, one needs to compute additional properties of the disk reliant on
a secondary closure. For example, the viscous timescale :math:`t_{\rm visc}` requires a closure for the viscosity :math:`\nu`,
which in turn requires a closure for the disk thermodynamics. As such, we insert an additional layer in the timestepping
for each timestep in which we first compute the disk properties (e.g. :math:`t_{\rm visc}`) using the current disk mass and angular
momentum, and then use these properties to compute the derivatives of the disk mass and angular momentum. This allows
for a more modular architecture in which the disk properties and the disk evolution are decoupled, and allows for more flexibility in the implementation of different models.

In the sections below, we describe a few specific disk models which are implemented in Triceratops.

.. figure:: disk_diagram_2


Thermodynamics
^^^^^^^^^^^^^^^

In order to determine the viscosity, one must specify a thermodynamic closure for the disk. In most cases of interest,
we assume that the disk is optically thick and scattering dominated, such that energy is transported via radiative diffusion.

The thermal structure of the disk is determined by balancing viscous heating and radiative cooling:

.. math::

    Q^+ + Q^+_{(\text{ext. heating)} = Q^-_{(\text{radiative)} + Q^-_{(\text{advected)}},

where the viscous heating rate is

.. math::

    Q^+ = \frac{9}{8} \nu \Sigma \Omega^2 = \frac{9}{8fA} \frac{GM\dot{M}_D}{\pi R_D^3} f_D^4,
    \; f_D^4 = 1 - \sqrt{R_{\rm in}/R_D},

(see Eq. 12 of :footcite:t:`metzgerTimeDependentModelsAccretion2008`), and the radiative cooling rate is

.. math::

    Q^- = \sigma_{\rm SB} T_{\rm eff}^4.

For a geometrically thin, optically thick disk, the effective temperature is related to the midplane temperature
through radiative diffusion:

.. math::

    T_c^4 = \frac{3}{4} \tau T_{\rm eff}^4,

where the optical depth is given by

.. math::

    \tau = \kappa(\rho, T_c)\, \Sigma.

Here, :math:`\kappa` is the opacity, which may depend on both density and temperature.

The thermodynamic closure is completed by specifying an equation of state for the disk.
In general, the total pressure is composed of multiple contributions:

.. math::

    P = P_{\rm gas} + P_{\rm rad} + P_{\rm mag}.

In the gas pressure dominated regime,

.. math::

    P_{\rm gas} = \frac{\rho k_B T_c}{\mu m_p},

while in the radiation pressure dominated regime,

.. math::

    P_{\rm rad} = \frac{a T_c^4}{3}.

Additional contributions, such as magnetic pressure, may also be included depending on the model.

The sound speed is then given by

.. math::

    c_s^2 = \frac{P}{\rho},

and the disk scale height follows from vertical hydrostatic equilibrium:

.. math::

    H = \frac{c_s}{\Omega}.


The viscosity is typically parameterized using the :math:`\alpha`-prescription:

.. math::

    \nu = \alpha c_s H.

Solution Strategy
^^^^^^^^^^^^^^^^^

In general, the solution strategy cannot be performed analytically. One must therefore procure numerical solutions
to the relevant system of equations. However, in certain limits, one can obtain analytic scalings for the disk properties.

In the general case, Triceratops provides 3 **"entry points"** for control of the thermodynamic closure:

1. **Disk Property Closure**: This is the first closure to be computed and must take the disk state :math:`{\bf S}_D` (:math:`M_D` and :math:`J_D`)
   as inputs
   and provide from them as set of disk properties :math:`\boldsymbol{\Theta}_D` (e.g. :math:`\Sigma`, :math:`R_D`, etc.).
2. **Thermodynamic Closure**: This closure is performed second and may utilize both :math:`\boldsymbol{\Theta}_D` and :math:`{\bf S}_D` as inputs,
   but must provide the thermodynamic state of the disk :math:`\boldsymbol{\Psi}_D` (e.g. :math:`T_c`, :math:`P`, etc.). This is where
   computations involving the opacity and EOS should be performed.
3. **Viscosity Closure**: This is the final closure and must take as input the thermodynamic state :math:`\boldsymbol{\Psi}_D`,
   the disk properties :math:`\boldsymbol{\Theta}_D`, and the disk state :math:`{\bf S}_D` and provide the viscous timescale :math:`t_{\rm visc}`.

   Once this has been accomplished, the solver may iterate using any of the computed values (thus, they may be used in
   additional evolution terms).

This structure provides a lot of flexibility in the implementation of the different models.

We now briefly describe a few specific closures which are built into Triceratops by default.


Gas Pressure Dominated, Electron Scattering Opacity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this regime, we assume that the **pressure is dominated by the gas component** and that
the opacity is **dominated by electron scattering**:

.. math::

    P \approx P_{\rm gas}, \qquad \kappa = \kappa_{\rm es} = \text{constant}.

The solution proceeds as follows:

1. (Thermodynamic Closure) Use the dissipation balance to determine the effective temperature:

   .. math::

       \sigma_{\rm SB} T_{\rm eff}^4 = \frac{9}{8fA} \frac{GM\dot{M}_D}{\pi R_D^3} f_D^4,

2. (Thermodynamic Closure) Compute the optical depth:

   .. math::

       \tau = \kappa_{\rm es} \Sigma.

3. (Thermodynamic Closure) Relate the midplane temperature to the effective temperature:

   .. math::

       T_c^4 = \frac{3}{4} \tau T_{\rm eff}^4.

4. (Thermodynamic Closure) Use the ideal gas law:

   .. math::

       c_s^2 = \frac{k_B T_c}{\mu m_p},

6. (Viscous Closure) Compute the viscosity:

   .. math::

       \nu = \alpha c_s^2 \Omega^{-1}.

This system can be solved explicitly, yielding analytic scalings for all quantities.


Radiation Pressure Dominated, Electron Scattering Opacity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this regime, the pressure is dominated by radiation:

.. math::

    P \approx P_{\rm rad} = \frac{a T_c^4}{3}, \qquad \kappa = \kappa_{\rm es}.

The solution proceeds similarly:

1. (Thermodynamic Closure) Use the dissipation balance to determine the effective temperature:

   .. math::

       \sigma_{\rm SB} T_{\rm eff}^4 = \frac{9}{8fA} \frac{GM\dot{M}_{\rm acc}}{\pi R_D^3} f_D^4,

2. (Thermodynamic Closure) Compute the optical depth:

   .. math::

       \tau = \kappa_{\rm es} \Sigma.

3. (Thermodynamic Closure) Relate the midplane temperature to the effective temperature:

   .. math::

       T_c^4 = \frac{3}{4} \tau T_{\rm eff}^4.

4. (Thermodynamic Closure) Using the equation of state,

   .. math::

        P = \frac{a T_c^4}{3}.

   Because we do not immediately know :math:`\rho`, we cannot immediately determine the sound speed. Fortunately,
   we use the hydrostatic equilibrium of the atmosphere to determine the scale height:

   .. math::

        H = \sqrt{2\pi} P \Sigma^{-1} \Omega^{-2}.

   With that, the density is

   .. math::

        \rho = \frac{\Sigma}{\sqrt{2\pi} H} = \frac{\Sigma^2 \Omega^2}{2\pi P}.

   The sound speed is therefore

   .. math::

        c_s^2 = \frac{P}{\rho} = 2\pi \frac{P^2}{\Sigma^2 \Omega^2}.

4. (Viscous Closure) Using the alpha prescription, the viscosity is

   .. math::

    \nu = \alpha c_s^2 \Omega^{-1} = 2\pi \frac{\alpha}{\Omega^3} \Sigma^{-2} P^2.

Gas Pressure Dominated, Kramers Opacity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

    TODO


Ideal Gas and Generic Opacities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

    TODO

Magnetically Supported Disks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

    TODO
