.. _dynamics_gallery:

*****************************
Shock Dynamics
*****************************

These examples demonstrate the low-level shock physics machinery in Trilobite. They cover
the numerical shock engine that evolves blast-wave radius, velocity, and swept-up mass through
an arbitrary density profile, using Rankine–Hugoniot jump conditions at each step.

These examples are most useful when working with the :mod:`~trilobite.dynamics` subpackage
directly (e.g., to implement a new shock model) rather than through the higher-level model
interface in :mod:`~trilobite.models`.

.. rubric:: What you'll find here

- Rankine--Hugoniot jump conditions from the Newtonian to the ultra-relativistic regime
- Configuring and running the numerical shock engine
- Shock evolution in power-law circumstellar density profiles
- Visualizing shock radius, velocity, and column density over time

.. rubric:: API reference

:mod:`trilobite.dynamics.shocks`
