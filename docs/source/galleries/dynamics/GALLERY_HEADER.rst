.. _dynamics_gallery:

*****************************
Shock Dynamics
*****************************

These examples demonstrate the low-level shock physics machinery in Triceratops. They cover
the numerical shock engine that evolves blast-wave radius, velocity, and swept-up mass through
an arbitrary density profile, using Rankine–Hugoniot jump conditions at each step.

These examples are most useful when working with the :mod:`~triceratops.dynamics` subpackage
directly (e.g., to implement a new shock model) rather than through the higher-level model
interface in :mod:`~triceratops.models`.

.. rubric:: What you'll find here

- Configuring and running the numerical shock engine
- Shock evolution in power-law circumstellar density profiles
- Visualizing shock radius, velocity, and column density over time

.. rubric:: API reference

:mod:`triceratops.dynamics` — :class:`~triceratops.dynamics.shocks.ShockEngine`,
:mod:`~triceratops.dynamics.shocks.rankine_hugoniot`
