.. _data_overview:

=========================================
Data Loading, Handling, and Visualization
=========================================

The :mod:`triceratops.data` module provides the foundational
abstractions used to represent observational data throughout
the Triceratops library. These objects form the boundary between
raw measurements and the modeling and inference systems.

At a high level, the data layer serves three purposes:

1. Provide structured, validated containers for observational data.
2. Preserve physical units and enforce consistent schemas.
3. Translate observational tables into the numerical format required by inference.

The design is intentionally layered. User-facing containers wrap
:class:`astropy.table.Table` objects and provide semantic meaning
to specific columns. These containers can then generate
:class:`~triceratops.data.InferenceData` objects, which are the
sole data representation recognized by likelihood classes.


Architecture of the Data Layer
------------------------------

There are three principal observational container types in Triceratops:

- :class:`~triceratops.data.light_curve.RadioLightCurveContainer`
- :class:`~triceratops.data.photometry.RadioPhotometryContainer`
- :class:`~triceratops.data.spectra` (spectral containers)

Each container enforces a schema, ensures unit compatibility,
and provides convenience accessors tailored to the structure of
the underlying data.

These containers are **not** themselves used directly in inference.
Instead, they provide a bridge to the inference system via
the :class:`InferenceData` object.

The flow is therefore:

.. code-block:: text

    Astropy Table
          ↓
    DataContainer (LightCurve / Photometry / Spectra)
          ↓
    InferenceData
          ↓
    Likelihood
          ↓
    Inference Problem

Light Curves
------------

Light curves represent single-frequency time-series data:

.. math::

    F_\nu(t)

They are implemented in the
:mod:`triceratops.data.light_curve` module via
:class:`~triceratops.data.light_curve.RadioLightCurveContainer`.

A light curve container wraps a validated
:class:`astropy.table.Table`, enforces required columns
(e.g. time, flux density, uncertainty, upper limits),
and stores the observing frequency as metadata.

Light curves are ideal for modeling temporal evolution
at a fixed observing frequency. They expose convenience
properties such as:

- Detection and non-detection masks
- Unit-aware column access
- Conversion to dense CGS arrays

For full details, see:

.. toctree::
   :maxdepth: 1

   light_curve


Photometry Tables
-----------------

Photometry containers generalize light curves to heterogeneous
multi-frequency observations. They are implemented in
:mod:`triceratops.data.photometry` via
:class:`~triceratops.data.photometry.RadioPhotometryContainer`.

Unlike light curves, photometry tables may contain multiple
observing frequencies and may group measurements into epochs.

These containers:

- Enforce a strict schema for time, frequency, flux density,
  uncertainty, and upper limits.
- Support detection and non-detection separation.
- Provide utilities for grouping observations into epochs.
- Preserve auxiliary metadata columns without interpretation.

Photometry containers are the most flexible observational format
and are typically used when fitting broadband spectral energy
distributions or multi-frequency datasets.

For full details, see:

.. toctree::
   :maxdepth: 1

   photometry


Spectra
-------

Spectral containers represent flux density as a function of
frequency at a fixed time:

.. math::

    F_\nu(\nu)

They are implemented in :mod:`triceratops.data.spectra`.

While conceptually similar to photometry tables, spectral
containers assume that all rows correspond to a single epoch.
They are optimized for modeling instantaneous broadband spectra
rather than time evolution.

(Full documentation forthcoming.)


Inference Data
--------------

The :class:`~triceratops.data.InferenceData` object is the
numerical representation of observational data used by the
inference layer.

Unlike light curve or photometry containers, `InferenceData`
contains **only validated NumPy arrays**. It performs no unit
conversion, no column resolution, and no schema enforcement.
Those responsibilities belong to the observational containers.

`InferenceData` provides:

- Deterministic stacking of independent variables
- Deterministic stacking of observables
- Symmetric uncertainty arrays
- Upper and lower limit arrays (with NaN fill where appropriate)

Likelihood classes operate exclusively on this structure.

An `InferenceData` object can be generated in three ways:

From a Data Container
^^^^^^^^^^^^^^^^^^^^^

Every observational container provides a
:meth:`to_inference_data` method:

.. code-block:: python

    inference_data = photometry_container.to_inference_data(model)

This is the most common pathway and ensures correct
column mapping and unit coercion.

From an Astropy Table
^^^^^^^^^^^^^^^^^^^^^

If working directly with tables:

.. code-block:: python

    inference_data = InferenceData.from_table(
        model,
        table,
        variables={...},
        observables={...},
    )

From Raw Arrays
^^^^^^^^^^^^^^^

For programmatic workflows:

.. code-block:: python

    inference_data = InferenceData.from_arrays(
        model,
        x={...},
        y={...},
        y_err={...},
    )

This pathway assumes that arrays are already in the model’s
expected base units.

For complete documentation, see:

.. toctree::
   :maxdepth: 1

   inference_data


Design Philosophy
-----------------

The data module intentionally separates:

- **Observational semantics** (handled by containers)
- **Numerical inference representation** (handled by InferenceData)
- **Statistical assumptions** (handled by likelihood classes)

This separation provides several benefits:

- Clean, testable interfaces between layers
- Deterministic stacking and ordering
- Unit safety at ingestion time
- Minimal overhead during likelihood evaluation

Once data have been converted to `InferenceData`,
likelihood evaluation becomes purely numerical and free
of schema or unit concerns.


Visualization
-------------

Observational containers retain their underlying
:class:`astropy.table.Table` structure and can therefore
be used directly with standard Astropy or Matplotlib tools.

For example:

.. code-block:: python

    import matplotlib.pyplot as plt

    plt.errorbar(
        light_curve.time,
        light_curve.flux_density,
        yerr=light_curve.flux_density_error,
        fmt="o",
    )

Because containers preserve units, plotting and
unit conversions remain explicit and transparent.


Summary
-------

The data module provides structured, unit-aware containers
for observational data and a clean translation layer into
the inference system.

In typical workflows:

1. Load or construct a light curve or photometry container.
2. Convert it to :class:`InferenceData`.
3. Pass the inference data to a likelihood object.
4. Evaluate models against the data.

This layered design ensures clarity, reproducibility,
and extensibility across the entire modeling pipeline.
