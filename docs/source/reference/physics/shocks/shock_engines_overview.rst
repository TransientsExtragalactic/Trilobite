.. _shock_engines:
===========================================
Shock Engines
===========================================

Introduce the interest in having shock engines to model the behavior of
various different transients.

Introduce the need for a variety of different shocks (radiative, adiabatic, relativistic, etc.) and
the need for modularity.

Introduce the shock engine object: it wraps the physics and provides you with the shock properties.

.. contents::
    :local:
    :depth: 2

Quickstart
--------------

Show off the simple case with the Chevalier self-similar engine. Give a really short
introduction: "We want to evolve a Chevalier shock with ... parameters", use the
normalization functions to get the parameters, then evolve and produce the plot.

.. plot::
    :include-source:

Types of Shock Engines
----------------------

- Introduce the various types of shock engine.

Self-Similar Shock Engines
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

    Pointer to the specific self-similar docs.

- What these are and what assumptions they generally make.
- We want a directory of these.

Numerical Shock Engines
^^^^^^^^^^^^^^^^^^^^^^^^

.. seealso::

    Pointer to the specific numerical docs.

- Introduce these.
- Explain how the evolution behaves.

Custom Shock Engines
--------------------

- Introduce the relevance of custom shock engines.

The Shock Engine Class
^^^^^^^^^^^^^^^^^^^^^^^^

In depth discussion of the ABC contract in the class.

.. example:: Chevalier Shock Engine

    Show off the Chevalier shock engine as an example of how to use the shock engine class.
