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

Triceratops provides the following observational container types:

**Radio containers**

- :class:`~triceratops.data.photometry.RadioPhotometryContainer` — multi-epoch, multi-frequency radio photometry
- :class:`~triceratops.data.photometry.RadioPhotometryEpoch` — single-epoch radio SED (frequency as independent variable)
- :class:`~triceratops.data.light_curve.RadioLightCurveContainer` — single-frequency radio time-series

**Optical containers**

- :class:`~triceratops.data.optical_photometry.OpticalPhotometryContainer` — multi-epoch, multi-band optical photometry
- :class:`~triceratops.data.optical_photometry.OpticalPhotometryEpoch` — single-epoch optical SED (band as independent variable)
- :class:`~triceratops.data.light_curve.OpticalLightCurveContainer` — single-band optical time-series

Each container enforces a schema, ensures unit compatibility,
and provides convenience accessors tailored to the structure of
the underlying data.

These containers are **not** themselves used directly in inference.
Instead, they provide a bridge to the inference system via
the :class:`~triceratops.data.core.InferenceData` object.

The flow is therefore:

.. code-block:: text

    Astropy Table
          ↓
    DataContainer (RadioPhotometry / OpticalPhotometry / LightCurve / ...)
          ↓
    container.to_inference_data(model)
          ↓
    InferenceData
          ↓
    Likelihood
          ↓
    Inference Problem

.. tip::

    See :ref:`data_to_inference` for a complete step-by-step guide on
    moving from raw data to an inference-ready problem.


Radio Light Curves
------------------

:class:`~triceratops.data.light_curve.RadioLightCurveContainer` represents single-frequency time-series radio data:

.. math::

    F_\nu(t)

The observing frequency is stored as metadata (not a column) and is
not treated as a variable in inference — it is a fixed property of
the container. Detection status is inferred from the ``flux_upper_limit``
column (NaN = detection, finite = non-detection).


Optical Light Curves
--------------------

:class:`~triceratops.data.light_curve.OpticalLightCurveContainer` is the optical analog:
a single-band time-series in AB magnitudes or flux density units.
The band name is stored as metadata and resolved to a model band index
at inference time via the model's
:class:`~triceratops.utils.phot_utils.FilterBundle`.

Observations may be supplied as flux densities, AB magnitudes, or both.
All representations are always accessible via properties regardless of
the input format.

.. toctree::
   :maxdepth: 1

   light_curve


Radio Photometry
----------------

:class:`~triceratops.data.photometry.RadioPhotometryContainer` generalizes
light curves to heterogeneous multi-frequency, multi-epoch radio photometry.
Observations may contain multiple frequencies and can be grouped into epochs.

:class:`~triceratops.data.photometry.RadioPhotometryEpoch` represents a
single-epoch radio SED where frequency is the independent variable — designed
for broadband spectral fitting.

.. toctree::
   :maxdepth: 1

   photometry
   radio_photometry_epoch


Optical Photometry
------------------

:class:`~triceratops.data.optical_photometry.OpticalPhotometryContainer`
handles multi-epoch, multi-band optical photometry with dual flux/magnitude
representation.

:class:`~triceratops.data.optical_photometry.OpticalPhotometryEpoch` is the
optical SED analog of :class:`~triceratops.data.photometry.RadioPhotometryEpoch`:
a snapshot across multiple bands for broadband SED fitting.

.. toctree::
   :maxdepth: 1

   optical_photometry
   optical_photometry_epoch


Inference Data
--------------

The :class:`~triceratops.data.InferenceData` object is the
numerical representation of observational data used by the
inference layer.

Unlike observational containers, :class:`~triceratops.data.core.InferenceData`
contains **only validated NumPy arrays**. It performs no unit
conversion, no column resolution, and no schema enforcement.
Those responsibilities belong to the containers.

Every observational container provides a :meth:`to_inference_data` method:

.. code-block:: python

    inference_data = container.to_inference_data(model)
    print(inference_data.describe())  # inspect the result

.. toctree::
   :maxdepth: 1

   inference_data
   data_to_inference


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

Once data have been converted to ``InferenceData``,
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
2. Call :meth:`to_inference_data(model)` to produce an :class:`~triceratops.data.core.InferenceData`.
3. Pass the inference data to a likelihood object.
4. Evaluate models against the data.

This layered design ensures clarity, reproducibility,
and extensibility across the entire modeling pipeline.
