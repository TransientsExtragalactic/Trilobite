.. _radiation_overview:
==================================
Radiation Processes in Triceratops
==================================

Foundational to any model of radio emission from an astrophysical source are the radiation processes driving
the emission. The vast majority of the models implemented in Triceratops rely on a number of core radiation
processes which, along with the underlying dynamics of their systems, determine the observed emission. This guide
discusses the various radiative processes implemented in Triceratops, how to use them, and provides references and
readings on the theory behind them wherever possible.

Synchrotron Radiation
---------------------
The most foundational radiation process implemented in Triceratops is synchrotron radiation. Synchrotron radiation
is emitted by relativistic charged particles (typically electrons) spiraling around magnetic field lines. This
process is responsible for a wide range of astrophysical phenomena, from radio emission in supernova remnants
to the jets of active galactic nuclei.

Most importantly, synchrotron emission is the dominant radiation mechanism in a variety of astrophysical transients
such as gamma-ray bursts (GRBs) and tidal disruption events (TDEs).

Because of its relevance to so many astrophysical systems, Triceratops provides a comprehensive suite of tools for
modeling synchrotron radiation. This includes modules for calculating synchrotron spectra, light curves, and
polarization, as well as tools for incorporating synchrotron self-absorption and cooling effects. The documents below
are designed to provide a thorough description of the available tools and their implementation; however, we recognize
that the theory of synchrotron radiation can be complex. Therefore, we also provide a detailed overview of the underlying
theory in the :ref:`synchrotron_theory` section of the documentation.

.. toctree::
    :maxdepth: 1
    :caption: Synchrotron Radiation Topics

    synchrotron/synchrotron_core
    synchrotron/synchrotron_microphysics
    synchrotron/synchrotron_seds
    synchrotron/synchrotron_cooling

Additionally, there are a few additional theory documents which provide further context for the implementation of
synchrotron radiation in Triceratops:

.. toctree::
    :maxdepth: 1
    :caption: Synchrotron Theory References

    synchrotron/synchrotron_theory
    synchrotron/synchrotron_sed_theory
    synchrotron/synchrotron_cooling_theory

Free-Free Radiation
--------------------

.. important::

    Coming Soon!


Inverse Compton Scattering
--------------------------

.. important::

    Coming Soon!
