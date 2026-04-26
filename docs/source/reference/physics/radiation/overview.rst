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
        - :ref:`barnoil_duran`

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
    synchrotron/synchrotron_barniol_duran

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

Blackbody Radiation
--------------------

Thermal emission from a blackbody in local thermodynamic equilibrium.
The :mod:`triceratops.radiation.blackbody` module provides the Planck
function (per unit frequency and per unit wavelength), the Wien displacement
laws, and the Stefan-Boltzmann bolometric flux, all following the standard
Triceratops two-level API so the same functions work in interactive analysis
and in MCMC hot loops.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Usage Guide
        :link: blackbody_user_guide
        :link-type: ref

        Evaluate :math:`B_\nu`, :math:`B_\lambda`, Wien peak frequency and
        wavelength, and bolometric flux.  Includes notes on using the fast
        CGS kernels directly in performance-critical code.

    .. grid-item-card:: Theory
        :link: blackbody_theory
        :link-type: ref

        Derivations of the Planck function, the Wien displacement law, and
        the Stefan-Boltzmann relation, with notes on numerical stability.

.. toctree::
    :hidden:
    :maxdepth: 1
    :caption: Blackbody Radiation

    blackbody/blackbody

----

Opacity Laws
------------

The :mod:`triceratops.radiation.opacity` module provides opacity laws for use across
Triceratops. This includes common analytic forms for the Rosseland and Planck mean opacities as
well as numerical tables for various purposes. In many cases, these are implemented at the C-level to
ensure seamless performance in hot loops.

.. note::

    Currently, all of the available opacity laws are gray (frequency-independent), Rosseland means
    or Planck means.  Future work will expand this to include frequency-dependent opacities and
    potentially more specialized forms (e.g. line opacities for specific ions); however, current
    implementations are sufficient for the majority of applications in Triceratops.

    Triceratops is **not** a spectral synthesis code, so we do not implement detailed
    frequency-dependent opacities.

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
    :caption: Opacity Laws

    opacity/opacity_user_guide
    opacity/opacity_theory
    opacity/opacity_dev_guide

----

Inverse Compton Scattering
--------------------------

.. important::

    Coming Soon!
