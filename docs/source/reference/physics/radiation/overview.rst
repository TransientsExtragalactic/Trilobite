.. _radiation_overview:

==================================
Radiation Processes in Triceratops
==================================

Foundational to any model of emission from an astrophysical source are the radiation
processes driving it.  The vast majority of models in Triceratops rely on a small set
of core radiation processes which, together with the underlying source dynamics,
determine the observed emission.  This page provides an entry point to each of those
processes.  Each section starts with an overview that introduces the physical context,
maps out the module structure, and provides minimal working examples.  The usage
references and theory documents linked from it go into the full API detail and
derivations.

----

Synchrotron Radiation
---------------------

Synchrotron radiation is emitted by relativistic charged particles (typically
electrons) spiraling around magnetic field lines.  This process is responsible for a
wide range of astrophysical phenomena, from radio emission in supernova remnants to
the jets of active galactic nuclei.  Most importantly for Triceratops, synchrotron
emission is the dominant radiation mechanism in a variety of astrophysical transients
such as gamma-ray bursts (GRBs) and radio outflows in TDEs.

Triceratops provides a comprehensive suite of tools for modeling synchrotron
radiation, covering the full hierarchy from single-electron physics through
population-averaged spectra to broadband multi-component SED fitting.

.. grid:: 1
    :gutter: 3

    .. grid-item-card:: Overview
        :link: synchrotron_overview
        :link-type: ref

        **Start here.** Introduces the module structure, provides physical context,
        and includes minimal worked examples for each layer of the synchrotron stack.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Usage Reference

        - :ref:`synchrotron_core` -- kernel functions, single-electron spectra
        - :ref:`synchrotron_microphysics` -- distributions, equipartition closure
        - :ref:`synchrotron_seds` -- one-zone SED models and spectrum inversion
        - :ref:`synchrotron_cooling` -- radiative cooling engines

    .. grid-item-card:: Theory

        - :ref:`synchrotron_theory`
        - :ref:`synch_sed_theory`
        - :ref:`synchrotron_cooling_theory`
        - :ref:`synchrotron_cooling_closure`
        - :ref:`stratified_absorption`

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Synchrotron Radiation

    synchrotron/synchrotron_overview
    synchrotron/synchrotron_core
    synchrotron/synchrotron_microphysics
    synchrotron/synchrotron_seds
    synchrotron/synchrotron_cooling
    synchrotron/synchrotron_theory
    synchrotron/synchrotron_sed_theory
    synchrotron/synchrotron_cooling_theory
    synchrotron/synchrotron_cooling_closure
    synchrotron/synchrotron_stratified_absorption

----

Free-Free Radiation
--------------------

Thermal free-free (bremsstrahlung) emission and absorption from ionized plasmas.
This is the primary opacity source in radio supernova CSM and HII regions modeled by
Triceratops.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Usage Guide
        :link: free_free_emission
        :link-type: ref

        Tools for computing free-free emission and absorption coefficients, Gaunt
        factors, and optical depths in ionized plasma.

    .. grid-item-card:: Theory
        :link: free_free_theory
        :link-type: ref

        Derivation of the free-free emissivity, the Gaunt factor approximations
        used in Triceratops, and their range of validity.

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Free-Free Radiation

    free_free/free_free_emission
    free_free/free_free_theory

----

Grey Opacity Laws
-----------------

The :mod:`triceratops.radiation.opacity` module provides a set of grey
(frequency-independent) opacity laws for use across Triceratops: as the Rosseland
mean opacity driving radiative diffusion in accretion disks, as an effective
photon-absorption cross-section in CSM optical-depth calculations, and as a building
block in any custom thermal solver.  The module exposes a uniform interface for
evaluating :math:`\kappa(\rho, T)` and its logarithmic derivatives, and is designed
to be context-agnostic so the same opacity object can be dropped into any part of the
codebase that needs it.

Implemented laws include constant (grey), electron-scattering (Thomson), free-free
and bound-free Kramers power laws, combined Kramers + electron-scattering forms, and
OPAL numerical opacity tables.

.. grid:: 1 1 3 3
    :gutter: 3

    .. grid-item-card:: Usage Guide
        :link: opacity_user_guide
        :link-type: ref

        How to instantiate opacity laws, evaluate :math:`\kappa(\rho, T)`, compute
        logarithmic derivatives, and load OPAL numerical tables.

    .. grid-item-card:: Theory
        :link: opacity_theory
        :link-type: ref

        Derivations of each implemented opacity law and discussion of the physical
        regimes in which each applies.

    .. grid-item-card:: Developer Guide
        :link: opacity_dev_guide
        :link-type: ref

        How to implement a new opacity law and plug it into the Triceratops opacity
        registry.

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Grey Opacity Laws

    opacity/opacity_user_guide
    opacity/opacity_theory
    opacity/opacity_dev_guide

----

Inverse Compton Scattering
--------------------------

.. important::

    Coming Soon!
