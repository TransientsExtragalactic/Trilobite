.. _accretion_gallery:

*****************************
Accretion Disk Models
*****************************

These examples cover both steady-state and time-dependent accretion disk models in Trilobite.

**Steady-state thin disks** (:class:`~trilobite.dynamics.accretion.AlphaDisk`) evaluate the
Shakura-Sunyaev (SS73) radial structure and multi-colour blackbody SED at a fixed accretion
rate — useful for spectral fitting and structural validation.

**One-zone time-dependent disks** evolve disk mass and angular momentum through the viscous
timescale using a compiled Cython explicit-Euler solver, supporting a range of equations of
state and external mass-supply terms (e.g., fallback accretion from a TDE).

Examples progress from the steady-state SS73 scalings through physical regime comparisons
(gas-pressure vs. radiation-pressure dominated, advective disks), thermal instability S-curves,
and observationally motivated TDE disk scenarios.

.. rubric:: What you'll find here

- Sunyaev-Shakura alpha-disk: radial structure, scalings, and multi-colour SED
- Late-time broadband SED evolution of a TDE accretion disk
- Quickstart: running the one-zone disk integrator
- Comparing equations of state (gas-pressure vs. radiation-pressure dominated)
- Mapping the thermal S-curve and limit-cycle behavior
- Modeling a fallback accretion disk after a TDE
- Advection-dominated disk solutions
- Generating observable light curves from disk evolution

.. rubric:: Theory and API reference

:ref:`thin_disk` — steady-state thin-disk user guide and API

:ref:`one_zone_disk_theory` — one-zone theoretical background

:ref:`one_zone_disk` — one-zone user guide and API
