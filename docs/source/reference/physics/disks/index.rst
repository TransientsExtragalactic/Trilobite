.. _accretion_disks:

=========================================
Accretion Disks
=========================================

Accretion disks are a central component of many astrophysical transient systems.
In the context of tidal disruption events (TDEs), disrupted stellar debris
circularises and forms a disk around the central black hole whose thermal emission
powers much of the observed UV and optical flux.  Disk winds and state transitions
also drive the formation of radio-emitting outflows whose synchrotron emission
Trilobite is designed to model.  More broadly, whenever a compact object accretes
from a disrupted body or a companion, the disk sets the long-term light-curve
envelope, determines the seed photon field for inverse-Compton cooling of shock-accelerated
electrons, and regulates the rate at which matter and angular momentum are deposited
onto the central object.

Trilobite provides a flexible framework for **computationally lightweight** disk
models that are nonetheless physically self-consistent.  The design priorities are:

- **Efficiency** — models must be fast enough to run inside Bayesian inference pipelines.
- **Modularity** — new physical processes (magnetic pressure, mass injection, wind-driven
  outflows) can be layered onto the shared infrastructure without rewriting the
  integrator.
- **Extensibility** — adding a new thermodynamic closure requires implementing two
  methods and a Cython extension type; the base class handles everything else.

.. note::

    Trilobite does not aim to replace global MHD simulations or full
    radiation-hydrodynamics codes.  The models here are effective descriptions
    designed to match the accuracy requirements of transient light-curve and
    spectral fitting. Depending on context, models provided by Trilobite may range
    from simple toy models to 1-D hydrodynamics; however, the underlying philosophy is to capture
    the essential physics with minimal free parameters and computational overhead.

To this end, a number of different types of disk models are provided for different applications, each
with its own benefits and limitations.  The following sections provide an overview of the available
models and link to more detailed documentation on the theory, implementation, and usage of each.

.. seealso::

     Link this to the gallery, and to the API documentation for the disk module.

.. contents::
    :local:
    :depth: 1

Steady-State Analytical Models
-------------------------------

There are relatively few accretion scenarios which may be solved analytically; however,
when applicable, such models are generally the **most efficient** option for coupling to
other physical processes / modeling. They are appropriate for spectral fitting, checking structural
scalings against analytic theory, and computing multi-colour blackbody SEDs.

.. grid:: 2
    :gutter: 2

    .. grid-item-card:: User Guide
        :link: thin_disk
        :link-type: ref

        How to evaluate the Shakura-Sunyaev disk structure, compute the effective
        temperature profile, generate multi-colour blackbody SEDs, and obtain the
        analytic bolometric luminosity.

.. toctree::
    :maxdepth: 1
    :hidden:

    one_zone_disk
    one_zone_disk_theory
    one_zone_disk_dev
    thin_disk


One-Zone Accretion Disk Models
-------------------------------

One-zone models describe the entire disk as a **single spatial zone**, tracking the
total disk mass :math:`M_D` and angular momentum :math:`J_D` as a function of time.
This approach is highly efficient, reduces the disk evolution to a two-component ODE
system, and captures the global viscous spreading behaviour with minimal free
parameters.  The ODE system is integrated by a compiled Cython explicit-Euler kernel
that evaluates all thermodynamic and structural quantities at each step inside a
GIL-free hot loop.

.. note::

    One-zone models are ideal for coupling to other physical processes (e.g. shock
    cooling, wind-driven outflows) and for performing large parameter sweeps; however,
    they do not capture the radial structure of the disk and are not suitable for
    spectral fitting. We base our formalism for one-zone modeling on
    :footcite:t:`piroLatetimeEvolutionInstabilities2025`.

.. grid:: 3
    :gutter: 2

    .. grid-item-card:: User Guide
        :link: one_zone_disk
        :link-type: ref

        How to instantiate disk models, configure initial conditions, run the solver,
        access results, perform parameter-grid sweeps, and save/load HDF5 files.

    .. grid-item-card:: Theory
        :link: one_zone_disk_theory
        :link-type: ref

        The one-zone formalism, viscous evolution equations, thermodynamic closures
        (gas pressure, radiation pressure, advection, fallback accretion), and the
        energy-balance structure.

    .. grid-item-card:: Developer Guide
        :link: one_zone_disk_dev
        :link-type: ref

        Step-by-step instructions for adding a new Cython thermodynamic closure,
        declaring the Python model class, passing parameters through the integration
        layer, and registering tests.

References
----------
.. footbibliography::
