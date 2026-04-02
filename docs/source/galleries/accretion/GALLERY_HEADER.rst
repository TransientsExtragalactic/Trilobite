.. _accretion_gallery:

*****************************
Accretion Disk Models
*****************************

These examples demonstrate the one-zone time-dependent accretion disk models in Triceratops.
The integrator evolves disk temperature and surface density through the viscous timescale using
a compiled Cython explicit-Euler solver, supporting a range of equations of state and external
mass supply terms (e.g., fallback accretion from a tidal disruption event).

Examples progress from a minimal quickstart through physical regime comparisons (gas-pressure
vs. radiation-pressure dominated, advective disks), thermal instability S-curves, and
observationally motivated TDE disk scenarios.

.. rubric:: What you'll find here

- Quickstart: running the one-zone disk integrator
- Comparing equations of state (gas-pressure vs. radiation-pressure dominated)
- Mapping the thermal S-curve and limit-cycle behavior
- Modeling a fallback accretion disk after a TDE
- Advection-dominated disk solutions
- Generating observable light curves from disk evolution

.. rubric:: Theory and API reference

:ref:`one_zone_disk_theory` — theoretical background

:mod:`~triceratops.dynamics.accretion.one_zone` — full API reference: :ref:`one_zone_disk`
