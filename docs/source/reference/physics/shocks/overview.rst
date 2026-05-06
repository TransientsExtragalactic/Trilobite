.. _shock_overview:

===================================
Shock Physics in Triceratops
===================================

One of the most ubiquitous sources of radiation in astrophysical transients is from shocks created
by the interaction of outflows with the surrounding medium. These shocks can accelerate particles to relativistic
energies and amplify magnetic fields, leading to the production of synchrotron radiation that
can be observed across the electromagnetic spectrum, particularly in the radio. Shock heating can also
produce thermal emission that can be observed in the X-ray, UV, and optical.

The shock module provided in Triceratops is designed to allow users to efficiently design models of
transients of interest using a variety of different shock physics routines ranging from fully realized shock solutions
to general jump-conditions. This page is intended to provide an entry point for users to get familiar with the
shock physics we make available in Triceratops and to understand how to use these tools to build their own models.
Each section starts with an overview that introduces the physical context, maps out the module structure,
and provides minimal working examples. The usage references and theory documents linked from it go
into the full API detail and derivations.

----

Core Shock Physics
------------------

At the lowest-level of the shock physics module, Triceratops provides :mod:`~triceratops.dynamics.shocks.core`, which
contains a variety of core physics routines for computing shock properties and performing shock-related calculations.
Unlike other parts of the codebase, these core routines are not full-fledged shock engines, but instead intended to
allow users to design their own shock engines and models by providing the necessary building blocks
for performing shock calculations.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Usage Reference

        - :ref:`rankine_hugoniot_overview`

    .. grid-item-card:: Theory

        - :ref:`rankine_hugoniot_theory`
        - :ref:`relativistic_jump_conditions_theory`

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Shocks-Core

    jump_conditions_overview
    jump_conditions_theory
    relativistic_jump_conditions_theory


----

Shock Engines
--------------

At the core of the Triceratops shock physics module are the shock engines, which are responsible for
computing the dynamical evolution of shocks in a variety of well-described physical scenarios. These engines
range from numerical calculations to fully analytical self-similar solutions, and are designed to be modular
and extensible so that users can easily implement new shock models or extend existing ones to fit their specific needs.


References
----------
.. footbibliography::
