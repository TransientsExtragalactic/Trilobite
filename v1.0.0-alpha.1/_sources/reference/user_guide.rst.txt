.. _user_guide:

========================
Trilobite User Guide
========================

Welcome to the **Trilobite User Guide**! This page is the location of all technical documentation for the
trilobite library and is generally the first place to look when seeking information on how to use the library. In
addition to the resources on this page, the API reference (:ref:`api`) gives in-depth information about call signatures
and code structure. The developer guide (:ref:`developer_guide`) is a resource for those who want to contribute to the library or
understand the codebase in detail.

Topics in this guide are broken down by subject area, with each section providing
a comprehensive overview of the relevant components and their usage.

.. figure:: ../images/index/TrilobiteCodeStructure.svg
    :align: center
    :alt: Trilobite Code Structure

    The core structure of TRILOBITE, from data and physics modules to a final inference product. The typical workflow
    in Trilobite will touch each of these 4 layers, but the user can also interact with each layer independently.
----

Getting Started
---------------

To get started with trilobite, check out :ref:`getting_started` guide which walks through installation,
basic usage, and a simple example. This is a great place to begin if you're new to trilobite or scientific
Python in general. You can also explore the :ref:`examples` section for more in-depth tutorials and use cases.

Data Loading, Handling, and Visualization
-----------------------------------------

In most workflows, the first step is to load and inspect data. Trilobite supports photometric data from
**radio to X-ray** frequencies, and provides tools for handling and visualizing these data. Below are guides
that cover how to load and manipulate data, how to get data ready for analysis, how to inspect data, and
how to visualize data:

.. toctree::
   :maxdepth: 1

   data/overview
   data/photometry
   data/optical_photometry
   data/filter_photometry
   data/light_curve
   data/inference_data


Trilobite Models
------------------

The core functionality of trilobite is its ability to create physical models of radio sources and their
environments. These models are built using a combination of modular building blocks that represent different physical
processes and components. The following guides provide an overview of the modeling capabilities of trilobite:

.. toctree::
    :maxdepth: 1

    models/overview
    models/directory

More than any other part of the Trilobite library, the models section is designed for extensibility, even by every
day users of the code. It is important to be able to correctly leverage the capabilities of the building blocks and
correctly implement the models relevant to your science case.

Inference and Statistical Analysis
-----------------------------------

Trilobite models are designed to very easily plug into inference pipelines to perform parameter estimation
and model comparison. The inference modules provide tools for setting up and running inference analyses using
various sampling algorithms. The following guides cover the relevant functionality:

.. toctree::
    :maxdepth: 1

    inference/overview


Building Blocks
---------------

Behind every Trilobite model are a set of modular building blocks that define the physical processes and components
of the system being modeled. These building blocks can be combined in various ways to create complex models that
capture the nuances of radio observations. Each major division of the building-block layer has its own landing page
that provides an overview of the available modules and links to detailed user, theory, and developer documentation:

.. grid:: 3
    :gutter: 2

    .. grid-item-card:: Shocks
        :link: shock_overview
        :link-type: ref

        Shock dynamics, Rankine--Hugoniot jump conditions, self-similar and numerical shock engines.

    .. grid-item-card:: Radiation
        :link: radiation_overview
        :link-type: ref

        Synchrotron emission and self-absorption, free-free (bremsstrahlung) emission, and grey
        opacity laws.

    .. grid-item-card:: Accretion Disks
        :link: accretion_disks
        :link-type: ref

        Time-dependent one-zone disk models: thermodynamic closures, viscous spreading, fallback
        accretion, and the advective RIAF regime.

.. toctree::
    :maxdepth: 1
    :hidden:

    physics/shocks/overview
    physics/radiation/overview
    physics/disks/index


Parallel Computing
------------------

Trilobite supports parallel execution across multiple backends — from single-process
serial execution to shared-memory multiprocessing and distributed MPI clusters. The parallel
module provides a unified pool abstraction that works transparently with MCMC samplers,
parameter grid sweeps, and other compute-intensive workflows.

.. toctree::
   :maxdepth: 1

   parallel/index

Extensions and 3rd Party Libraries
----------------------------------


Configuration and Setup
------------------------

Analysis Guides
----------------
Having now covered all of the core infrastructure of the Trilobite library, we can now turn to the analysis pipelines that
leverage these components to perform scientific inference and other important workflows. The goal of the guides in this section
is to provide a comprehensive overview of the standard methods used in the literature to analyze various types of radio transients
and how to perform those analyses using Trilobite. We break down the guides by system.
