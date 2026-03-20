.. _accretion_disks:
=========================================
Accretion Disks
=========================================

Accretion disks are a central component of many astrophysical transient systems.
In the context of tidal disruption events (TDEs), disrupted stellar debris
circularises and forms a disk around the central black hole whose thermal emission
powers much of the observed UV and optical flux.  Disk winds and state transitions
also drive the formation of radio-emitting outflows whose synchrotron emission
Triceratops is designed to model.  More broadly, whenever a compact object accretes
from a disrupted body or a companion, the disk sets the long-term light-curve
envelope, determines the seed photon field for inverse-Compton cooling of shock-accelerated
electrons, and regulates the rate at which matter and angular momentum are deposited
onto the central object.

Triceratops provides a flexible framework for **computationally lightweight** disk
models that are nonetheless physically self-consistent.  The design priorities are:

- **Efficiency** — models must be fast enough to run inside Bayesian inference pipelines.
- **Modularity** — new physical processes (magnetic pressure, mass injection, wind-driven
  outflows) can be layered onto the shared infrastructure without rewriting the
  integrator.
- **Extensibility** — adding a new thermodynamic closure requires implementing two
  methods and a Cython extension type; the base class handles everything else.

.. note::

    Triceratops does not aim to replace global MHD simulations or full
    radiation-hydrodynamics codes.  The models here are effective descriptions
    designed to match the accuracy requirements of transient light-curve and
    spectral fitting.

.. contents::
    :local:
    :depth: 1

One-Zone Accretion Disk Models
-------------------------------

One-zone models describe the entire disk as a **single spatial zone**, tracking the
total disk mass :math:`M_D` and angular momentum :math:`J_D` as a function of time.
This approach is highly efficient, reduces the disk evolution to a two-component ODE
system, and captures the global viscous spreading behaviour with minimal free
parameters.  The ODE system is integrated by a compiled Cython explicit-Euler kernel
that evaluates all thermodynamic and structural quantities at each step inside a
GIL-free hot loop.

*Full user guide:* :ref:`one_zone_disk`

*Theory note:* :ref:`one_zone_disk_theory`

.. toctree::
    :maxdepth: 1
    :hidden:

    one_zone_disk
    one_zone_disk_theory
