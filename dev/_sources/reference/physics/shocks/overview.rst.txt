.. _shock_overview:
===================================
Shocks in Triceratops
===================================

One of the most common sources of astrophysical radio emission is synchrotron radiation from electrons accelerated
at shock fronts. Triceratops provides a suite of tools for modeling the dynamics of these shocks and the resulting
radiation. This section provides an overview of the shock physics modules available in Triceratops.

.. hint::

    Throughout this document, we relegate detailed discussion of the relevant shock theory to the dropdowns,
    but do nonetheless encourage the reader to expand and read through these sections for a more complete
    understanding of the underlying physics.

.. contents::


Overview
========

In practice, synchrotron modeling in Triceratops proceeds in a layered fashion:

- First the shock dynamics are computed to determine the evolution of key shock properties such as radius, velocity,
  and post-shock density and energy.
- Next, the microphysical parameters governing particle acceleration and magnetic field amplification are applied
  to determine the energy distribution of accelerated electrons and the strength of the magnetic field in the shocked
  region.
- Finally, the synchrotron emission and absorption processes are computed based on the shock properties and
  microphysical parameters to generate the predicted radio observables.

In this document, we'll focus on the first step in this process. There are a number of different types of shocks
which are relevant to different astrophysical scenarios, and Triceratops provides modular implementations of several
common shock models.

Shock Engines
=============

At the core of every shock model in Triceratops is a **shock engine** (:class:`dynamics.shock_engine.ShockEngine`),
which is responsible for computing the dynamical evolution of the shock front over time. Each implementation of a
shock engine provides (at least) a method

.. code-block:: python

    def compute_shock_properties(self, time, **parameters) -> dict:
        # Compute the shock properties at a given time ``time`` given a set of
        # parameters ``parameters``.

and a corresponding low-level (CGS only) version of the same method:

.. code-block:: python

    def _compute_shock_properties_cgs(self, time_cgs, **parameters_cgs) -> dict:
        # Compute the shock properties at a given time ``time_cgs`` (in CGS units)
        # given a set of parameters ``parameters_cgs`` (also in CGS units).

The output of these methods is a dictionary containing the key shock properties at the specified time.

.. important::

    The specific properties which are computed may vary between different shock engine implementations. It is important
    to consult the documentation for each specific shock engine to understand which properties are available.

    In practice, if you are using an established model, the shock engine is internal and not something you need to
    interact with directly. However, if you are implementing a new model or extending an existing one, you may need to
    create a custom shock engine.

.. note::

    Like :class:`~models.core.base.Model` the :class:`~dynamics.shock_engine.ShockEngine` and its subclasses are not
    state-carrying objects. Instead, they are stateless computational engines that take in parameters and return computed
    properties. This design allows for easy integration into larger modeling frameworks and ensures that
    shock engines can be reused across different models without concern for internal state management.

Many shock engines are already implemented in Triceratops and provide an excellent opportunity to discuss the various
types of shocks and their relevant physics.

Rankine–Hugoniot Jump Conditions
================================

.. hint::

    Canned routines for performing Rankine–Hugoniot calculations are available in the :mod:`~dynamics.rankine_hugoniot`
    module.

The `Rankine–Hugoniot <https://en.wikipedia.org/wiki/Rankine%E2%80%93Hugoniot_conditions>`__ jump conditions describe
the relationships between the physical properties of a fluid
on either side of a shock front. These conditions are derived from the conservation laws of mass, momentum, and energy
across the shock discontinuity. They are fundamental to understanding shock dynamics and are widely used in
astrophysics to model shock waves in various contexts, including supernova remnants, stellar winds, and jet interactions.

.. note::

    For detailed references, we suggest :footcite:t:`landau1987fluid`, :footcite:t:`thorne2017modern`, and
    :footcite:t:`clarke2007principles`.

Classical Shock Relations
-------------------------

In the classical (non-relativistic) regime, the Rankine–Hugoniot conditions relate the upstream (pre-shock) and downstream
(post-shock) states of the fluid. The key relations are as follows:

.. math::

    \begin{aligned}
    \rho_1 u_1 &= \rho_2 u_2 \quad &\text{(Mass Conservation)} \\
    P_1 + \rho_1 u_1^2 &= P_2 + \rho_2 u_2^2 \quad &\text{(Momentum Conservation)} \\
    \frac{1}{2} u_1^2 + \frac{c_1^2}{\Gamma-1} &= \frac{1}{2} u_2^2 + \frac{c_2^2}{\Gamma-1}
    \quad &\text{(Energy Conservation)},
    \end{aligned}

where we have assumed an ideal gas with polytropic equation of state :math:`P = K \rho^{\Gamma}`. From these, a number
of useful relations can be derived relating the thermodynamic quantities in the upstream and downstream regions. We
further refine based on two factors:

1. Whether the shock is weak or strong: For strong shocks (:math:`M_1 \to \infty`), the relations further simplify.
2. For shocks into **cold media** (i.e., :math:`P_1 \approx 0`), which is often the case in astrophysical contexts.

We follow the standard naming convention

.. code-block:: python

    # High-Level API
    def compute_<strong/weak>_<cold/warm>_shock_<quantity>(...):
        # Compute the specified shock quantity based on the shock type.

    # Low Level API
    def _compute_<s/w>_<h/c>_shock_<quantity>_cgs(...):
        # Compute the specified shock quantity in CGS units.

to distinguish between the different cases.

.. rubric:: API Methods

*current module*: :mod:`dynamics.rankine_hugoniot`

.. currentmodule:: dynamics.rankine_hugoniot

.. tab-set::

    .. tab-item:: High-Level API

        The high-level API is designed to provide unit handling and guardrails for common use cases which
        are not performance critical. The high-level functions accept parameters with units and perform
        the necessary conversions internally.

        .. autosummary::

            compute_strong_cold_shock_magnetic_field
            compute_strong_cold_shock_pressure
            compute_strong_cold_shock_temperature
            compute_strong_shock_velocity
            compute_strong_shock_density

    .. tab-item:: Low-Level API

        The low-level API functions are designed for performance-critical applications where unit
        conversions are handled externally. These functions accept parameters in CGS units only.

        .. autosummary::

            _compute_s_c_shock_magnetic_field_cgs
            _compute_s_c_shock_pressure_cgs
            _compute_s_c_shock_temperature_cgs
            _compute_s_shock_velocity_cgs
            _compute_s_density_cgs

Relavitistic Shock Relations
----------------------------

.. hint::

    Coming soon!

Types of Shocks
================

A number of different shock models are available in Triceratops, each relevant to different astrophysical scenarios.
Wherever possible, the relevant implementation is provided in the dynamics module for the corresponding transient.
Nonetheless, some shock models are more general and can be used across multiple transient types. In those cases, we
provide the shock engine in the module it most closely fits with.

In the sections below, we'll discuss the currently implemented shock engines available in Triceratops.

Self-Similar Shocks
--------------------

The canonical approach to modeling the shock dynamics is to
use self-similar solutions as described by :footcite:t:`ChevalierXRayRadioEmission1982`,
:footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`, :footcite:t:`sedov1946propagation`
and :footcite:t:`taylor1950formation`. These solutions often require some set of simplifying assumptions
about the structure of the ejecta and CSM density profiles, but provide a computationally efficient
means of calculating the shock evolution over time analytically. Several different self-similar solutions are implemented
in Triceratops, each relevant to different astrophysical scenarios.

Chevalier Self-Similar Engines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Chevalier self-similar solution
(:class:`~dynamics.supernovae.shock_dynamics.ChevalierSelfSimilarShockEngine`)
is a widely used model for shocks produced by supernovae and other explosive
transients interacting with a surrounding circumstellar medium (CSM).

Originally derived by :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982` to describe
radio and X-ray emission from supernovae, this solution has since become a
standard framework for modeling shock-powered emission in a broad range of
astrophysical transients.

The Chevalier model describes the interaction between expanding supernova
ejecta and an ambient medium under the assumption of self-similar evolution.
Specifically, it adopts the following assumptions:

.. dropdown:: Model Assumptions

    - The ejecta density follows a power-law distribution in velocity space:

      .. math::

            \rho(r, t) = A_1\, t^{-3} \left(\frac{r}{t}\right)^{-n},

    - The circumstellar medium (CSM) density follows a power-law distribution in
      radius:

      .. math::

            \rho_{\rm csm}(r) = A_4\, r^{-s},

    - The shock dynamics are self-similar, such that the shock radius and velocity
      evolve as power laws in time.

    - The shocks are strong and adiabatic, allowing the use of the
      Rankine–Hugoniot jump conditions to relate upstream and downstream
      thermodynamic quantities.

    - The forward and reverse shocks are sufficiently close that the shocked
      region may be approximated as a thin shell when computing synchrotron
      emission.

      While the original work of :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`
      solved the full self-similar ordinary differential equation system to obtain
      the detailed internal structure of the shocked region, it has become standard
      practice to adopt the thin-shell approximation for radiative calculations,
      following :footcite:t:`ChevalierXRayRadioEmission1982`.

Within Triceratops, Chevalier self-similar engines provide an efficient and
physically transparent description of shock dynamics for supernova-like
outflows and serve as a foundation for subsequent synchrotron emission
modeling.

To see the derivation of the relevant scalings and normalizations used by the
Chevalier shock engines in Triceratops, see :ref:`chevalier_theory`.

Usage
~~~~~

To use a Chevalier self-similar shock engine in Triceratops, you can choose between either
the standard implementation
(:class:`~dynamics.supernovae.shock_dynamics.ChevalierSelfSimilarShockEngine`)
or the wind-specific implementation
(:class:`~dynamics.supernovae.shock_dynamics.ChevalierSelfSimilarWindShockEngine`). In the
latter case, the CSM density profile is fixed to a wind-like profile with :math:`s=2`. See :ref:`chevalier_theory` for
details on the differences between these two implementations.

In either case, there are a few parameters that are required by the shock engine to compute the relevant shock
properties:

.. tab-set::

    .. tab-item:: Standard

        .. list-table::
            :header-rows: 1

            * - Parameter
              - Description
              - CGS Units
            * - ``E_ej``
              - Total kinetic energy of the ejecta. For a typical supernova, this is on the order
                of :math:`10^{51}` erg.
              - :math:`{\rm erg}`
            * - ``M_ej``
              - Total mass of the ejecta. Generally, for a supernova, this is on the order of a few
                solar masses.
              - :math:`g`
            * - ``n``
              - Power-law index of the outer ejecta density profile. This should be greater than 5
                for the self-similar solution to be valid. In most scenarios, :math:`n` is in the range
                of 6 to 12. ``10`` is a common choice for Type Ia supernovae, while core-collapse supernovae
                often have :math:`n` between 7 and 12.
              - dimensionless
            * - ``s``
              - Power-law index of the CSM density profile. This parameter describes how the density of the
                circumstellar medium falls off with radius. Common choices are :math:`s=0` for a uniform
                medium and :math:`s=2` for a wind-like medium. This must be between 0 and 3 for the self-similar
                solution to be valid.
              - dimensionless
            * - ``A_csm``
              - Normalization of the CSM density profile.
              - :math:`{\rm g\, cm^{s-3}}`
            * - ``delta``
              - Power-law index of the inner ejecta density profile. This parameter describes the density
                profile of the inner ejecta, which is typically flatter than the outer profile. Common choices
                are :math:`\delta=0` (constant density core) or :math:`\delta=1`.
              - dimensionless

    .. tab-item:: Wind CSM

        .. list-table::
            :header-rows: 1

            * - Parameter
              - Description
              - CGS Units
            * - ``E_ej``
              - Total kinetic energy of the ejecta. For a typical supernova, this is on the order
                of :math:`10^{51}` erg.
              - :math:`{\rm erg}`
            * - ``M_ej``
              - Total mass of the ejecta. Generally, for a supernova, this is on the order of a few
                solar masses.
              - :math:`g`
            * - ``M_dot``
              - Mass-loss rate of the progenitor star prior to explosion. This parameter sets the density
                normalization of the wind-like CSM profile. Typical values depend on the type of progenitor
                but are often in the range of :math:`10^{-6}` to :math:`10^{-4} M_{\odot}\, {\rm yr^{-1}}`.
              - :math:`M_{\odot}\, {\rm yr^{-1}}`
            * - ``v_w``
              - Wind velocity of the progenitor star. This parameter, together with ``M_dot`
                sets the density normalization of the wind-like CSM profile. Typical values are
                on the order of :math:`10^{6}` to :math:`10^{8}\, {\rm cm\, s^{-1}}`.
              - :math:`{\rm cm\, s^{-1}}`
            * - ``n``
              - Power-law index of the outer ejecta density profile. This should be greater than 5
                for the self-similar solution to be valid. In most scenarios, :math:`n` is in the range
                of 6 to 12. ``10`` is a common choice for Type Ia supernovae, while core-collapse supernovae
                often have :math:`n` between 7 and 12.
              - dimensionless
            * - ``delta``
              - Power-law index of the inner ejecta density profile. This parameter describes the density
                profile of the inner ejecta, which is typically flatter than the outer profile. Common choices
                are :math:`\delta=0` (constant density core) or :math:`\delta=1`.
              - dimensionless

.. dropdown:: Example: Chevalier Self-Similar Wind Shock Engine

    As a classic example of the Chevalier self-similar solution, consider a supernova explosion
    with the following parameters:

    - Ejecta kinetic energy: :math:`E_{\rm ej} = 10^{51}\, {\rm erg}`
    - Ejecta mass: :math:`M_{\rm ej} = 1.4\, M_{\odot}`
    - Mass loss rate: :math:`\dot{M} = 10^{-5}\, M_{\odot}\, {\rm yr^{-1}}`
    - Wind velocity: :math:`v_w = 10^{6}\, {\rm cm\, s^{-1}}`
    - Outer ejecta density index: :math:`n = 10`
    - Inner ejecta density index: :math:`\delta = 0`

    We can use the engine to compute the shock properties as a function of time after the explosion:

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from triceratops.dynamics.supernovae.shock_dynamics import ChevalierSelfSimilarWindShockEngine
        from triceratops.utils.plot_utils import set_plot_style
        from astropy import units as u

        # Define the shock engine
        shock_engine = ChevalierSelfSimilarWindShockEngine()

        # Define parameters
        params = {
            "E_ej": 1e51 * u.erg,
            "M_ej": 1.4 * u.M_sun,
            "M_dot": 1e-5 * u.M_sun / u.yr,
            "v_w": 1e6 * u.cm / u.s,
            "n": 10,
            "delta": 0,
        }

        # Time array (days)
        time = np.geomspace(1, 1000, 500) * u.day

        # Compute the shock properties.
        shock_properties = shock_engine.compute_shock_properties(
            time,
            E_ej=params["E_ej"],
            M_ej=params["M_ej"],
            M_dot=params["M_dot"],
            v_wind=params["v_w"],
            n=params["n"],
            delta=params["delta"],
        )

        # Extract the shock radius and velocity
        shock_radius = shock_properties["radius"]
        shock_velocity = shock_properties["velocity"]

        # Plotting
        set_plot_style()

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot Shock Radius
        axes[0].loglog(time.to_value(u.day), shock_radius.to_value(u.pc), label="Shock Radius", color="blue")
        axes[0].set_xlabel("Time since explosion (days)")
        axes[0].set_ylabel("Shock Radius (pc)")
        axes[0].set_title("Shock Radius and Velocity")
        axes[0].legend()

        # Plot Shock Velocity
        axes[1].loglog(time.to_value(u.day), shock_velocity.to_value(u.km / u.s), label="Shock Velocity", color="red")
        axes[1].set_xlabel("Time since explosion (days)")
        axes[1].set_ylabel("Shock Velocity (km/s)")
        axes[1].legend()
        plt.tight_layout()
        plt.show()


Numerical Shock Engines
-----------------------

While self-similar solutions provide an efficient means of computing shock dynamics, they often rely on simplifying
assumptions about the ejecta and CSM density profiles. In scenarios where these assumptions do not hold,
or where more complex density structures are present, numerical shock engines can be employed to compute the
shock evolution by directly integrating the equations of motion. These models can be powerful, but are also
more computationally intensive and often require a bit more domain expertise to implement correctly.

The Thin Shell Approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most common numerical shock engine implemented in Triceratops is the **thin-shell approximation** engine
(:class:`~dynamics.supernovae.shock_dynamics.NumericalThinShellShockEngine`), which models the shock as a thin shell
whose dynamics are governed by the conservation of momentum. This approach allows for arbitrary density profiles
for both the ejecta and CSM, enabling the modeling of more complex shock scenarios. This boils down (under the hood)
to numerically integrating the equation of motion for the shock front given specified density profiles.

.. note::

    For a detailed description of the theory, see :ref:`numeric_thin_shell_shocks`.

Usage
~~~~~

Stemming from the base-class :class:`~dynamics.supernovae.shock_dynamics.NumericalThinShellShockEngine`, there are
many subclasses which implement different specific channels all of which share the same underlying numerical thin-shell
framework. The various parameters for these engines will depend on the specific subclass being used. Use the tab
set below to explore the parameters for some of the more common numerical shock engines.

.. tab-set::

    .. tab-item:: Generic Numerical Thin-Shell

        In the most general case, the numerical solver requires the user to provide **two functions**:

        1. ``rho_csm(r)``: A function defining the CSM density profile as a function of radius. This **must be
           in CGS units** (i.e., ``g/cm^3``).
        2. ``G_ej(v)``: A function defining the ejecta density as a function of velocity. This **must be in CGS units**.

        The :math:`G_{\rm ej}(v)` function is defined by the fact that homologous expansion requires that the
        ejecta density have the form

        .. math::

            \rho_{\rm ej}(r, t) = t^{-3} G_{\rm ej}(r/t).

        Thus, instead of allowing the user to provide an explicit density profile, we instead ask for the
        :math:`G_{\rm ej}(v)` function which implicitly defines the density profile and allows for a more
        efficient numerical implementation.

        .. list-table::
            :header-rows: 1

            * - Parameter
              - Description
              - CGS Units
            * - ``rho_csm``
              - The function :math:`\rho_{\rm CSM}(r)` which returns the CSM density at radius ``r`` in CGS units.
                This should be a function which takes as input a float or array-like of radii in ``cm`` and returns
                the corresponding CSM density in ``g/cm^3``.
              - :math:`g/cm^{3}`
            * - ``G_ej``
              - The function :math:`G(v)` which returns the ejecta density profile function at
                velocity ``v`` in CGS units. This should be a function which takes as input a float or
                array-like of velocities in ``cm/s`` and returns
                the corresponding ejecta density profile function in ``g * s^3 / cm^3``.
              - :math:`{\rm cm\, s^{-1}}`
            * - ``R_0``
              - Initial radius of the shock at the start of the simulation.
              - :math:`{\rm cm}`
            * - ``v_0``
              - Initial velocity of the shock at the start of the simulation.
              - :math:`{\rm cm\, s^{-1}}`
            * - ``t_0``
              - Initial time at the start of the simulation.
              - :math:`{\rm s}`
            * - ``M_0``
              - Initial swept-up mass of the shock at the start of the simulation.
              - :math:`{\rm g}`

.. dropdown:: Example: Generic Numerical Thin-Shell Shock Engine

    As an example of using the generic numerical thin-shell shock engine, we can model a standard supernova ejecta
    profile expanding into a broken CSM, where we have a wind-driven profile close to the progenitor which then
    transitions to a uniform density ISM at larger radii.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from triceratops.dynamics.supernovae.shock_dynamics import NumericalThinShellShockEngine
        from triceratops.dynamics.supernovae.profiles import get_broken_power_law_ejecta_kernel_func
        from triceratops.utils.plot_utils import set_plot_style
        from astropy import units as u

        # Define parameters for the CSM and ejecta profiles.
        M_dot = 1e-5 * u.M_sun / u.yr  # Mass-loss rate
        v_wind = 1e5 * u.cm / u.s  # Wind velocity
        t_wind = 1e8 * u.s  # Duration of wind phase
        rho_ism = 1e-21 * u.g / u.cm ** 3  # ISM density
        E_ej = 1e48 * u.erg  # Ejecta energy
        M_ej = 10 * u.M_sun  # Ejecta mass
        R_wind = v_wind * t_wind  # Radius of wind termination shock

        # Derive CGS parameters for the wind profile.
        rho_0_cgs = (M_dot / (4 * np.pi * R_wind ** 2 * v_wind)).to_value(u.g / u.cm ** 3)
        R_wind_cgs = R_wind.to_value(u.cm)
        rho_ISM_cgs = rho_ism.to_value(u.g / u.cm ** 3)


        # Create the rho_csm function so that it can be passed to the
        # integrator.
        def rho_csm(r):
            """Broken CSM density profile: wind-like close in, uniform ISM far out."""
            return np.where(
                r < R_wind_cgs,
                rho_0_cgs * r ** -2,
                rho_ISM_cgs
            )

        # Get the ejecta density kernel from the pre-built Chevalier-style broken power-law
        # scenario.
        G_v = get_broken_power_law_ejecta_kernel_func(E_ej,M_ej,n=10,delta=0)

        # Define the shock engine
        shock_engine = NumericalThinShellShockEngine()

        # Define parameters
        params = {
            "rho_csm": rho_csm,
            "G_ej": G_v,
            "R_0": 1e10 * u.cm,
            "v_0": 1e9 * u.cm / u.s,
            "t_0": 1e1 * u.s,
            "M_0": 1e-4 * u.M_sun,
        }

        # Time array (days)
        time = np.geomspace(1, 1000, 500) * u.day

        # Compute the shock properties.
        shock_properties = shock_engine.compute_shock_properties(
            time,
            **params)

        # Calculate the homologous expansion rate.
        R_homologous = (1e9 * u.cm/u.s) * time

        # Create the plot of the radius and velocity of the shock as a function
        # of time.
        set_plot_style()
        fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        ax[0].loglog(time.to_value(u.day), shock_properties["radius"].to_value(u.cm), label="Numerical Shock Radius")
        ax[0].loglog(time.to_value(u.day), R_homologous.to_value(u.cm), ls="--", label="Homologous Expansion")
        ax[0].set_ylabel("Shock Radius (cm)")
        ax[0].grid(True, which="both", ls="--", alpha=0.5)
        ax[1].loglog(time.to_value(u.day), shock_properties["velocity"].to_value(u.cm / u.s))
        ax[1].set_xlabel("Time (days)")
        ax[1].set_ylabel("Shock Velocity (cm/s)")
        ax[1].grid(True, which="both", ls="--", alpha=0.5)
        ax[2].loglog(time.to_value(u.day), shock_properties["mass"].to_value(u.M_sun))
        ax[2].set_xlabel("Time (days)")
        ax[2].set_ylabel(r"Swept-up Mass (M$_\odot$)")
        ax[2].grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

References
----------
.. footbibliography::
