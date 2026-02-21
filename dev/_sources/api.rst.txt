.. image:: ./images/logo.png
   :width: 200px
   :align: center

.. _api:

API
===

This page provides the complete reference for all public classes, functions, and modules in the Triceratops codebase.

Triceratops is modular by design, and its API is organized into several key modules.
Below is an overview of the main modules available:

Triceratops Models
------------------
The most relevant modules for typical users working with Triceratops is the set of model implementations. These
are all located in the :mod:`models` subpackage and include various forward models of radio emission
in different scenarios. These models can be used to simulate observations, fit data, and perform inference.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    models.supernovae
    models.core
    models.generic
    models.SEDs

Data Modules
------------
The data modules provide classes and functions for handling observational data, including loading, processing,
and visualizing data sets.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    data.core
    data.light_curve
    data.spectra
    data.photometry

Inference Modules
-----------------

The inference modules contain tools and algorithms for fitting models to observational data, performing parameter
estimation, and conducting statistical analysis.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    inference.prior
    inference.likelihood
    inference.sampling
    inference.problem



Physics and Computation Modules
-------------------------------
Underlying all of the models and inference pipelines in Triceratops are a set of physics and computation modules.
These modules implement the core algorithms for simulating plasma dynamics, radiation processes, and numerical methods.

Physics
^^^^^^^
The physics modules implement the core physical processes modeled in Triceratops, including shock dynamics,
radiation mechanisms, and ejecta evolution.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    radiation
    dynamics

Computation
^^^^^^^^^^^
The computation modules provide the numerical methods and algorithms used throughout Triceratops. This includes solvers
for differential equations, interpolation routines, and data handling utilities.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    parallel

Utilities
---------
The utilities modules contain helper functions and classes that support the main functionality of Triceratops.
These include data I/O, configuration management, and common mathematical operations.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    utils.plot_utils
    utils.io_utils
    utils.log
    utils.config
