.. _data_overview:

=========================================
Data Loading, Handling, and Visualization
=========================================

The :mod:`trilobite.data` module is the boundary between raw observational
data and the Trilobite modeling and inference systems. It provides
schema-validated, unit-aware containers for every major data type encountered
in time-domain radio and optical astronomy, and a clean translation layer
into the numerical representation required by the inference pipeline.

.. tip::

   New to Trilobite data? The fastest route to an inference-ready dataset
   is the step-by-step guide: :ref:`data_to_inference`.


How the Data Layer Fits In
---------------------------

Every analysis in Trilobite follows the same four-stage pipeline.
The data module owns the first two stages.

.. code-block:: text

    ┌──────────────────────────────────────────────────────────────┐
    │  Stage 1 — Load                                              │
    │  Raw table (FITS, CSV, …) → DataContainer                    │
    │  Schema validation · unit enforcement · column semantics     │
    ├──────────────────────────────────────────────────────────────┤
    │  Stage 2 — Convert                                           │
    │  container.to_inference_data(model) → InferenceData          │
    │  Unit coercion · band mapping · error inference              │
    ├──────────────────────────────────────────────────────────────┤
    │  Stage 3 — Evaluate                                          │
    │  InferenceData → Likelihood                                  │
    │  Statistical noise model (Gaussian, censored, …)             │
    ├──────────────────────────────────────────────────────────────┤
    │  Stage 4 — Infer                                             │
    │  Likelihood → InferenceProblem → Sampler                     │
    │  Priors · MCMC · posterior analysis                          │
    └──────────────────────────────────────────────────────────────┘

The key architectural contract is that **InferenceData is the only object
the inference layer ever sees**. Once data cross that boundary, they are
purely numerical — no units, no column names, no schema. The container layer
owns all of that complexity.


Container Quick Reference
--------------------------

Navigate directly to the container you need:

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Radio Photometry Container
      :class-card: sd-shadow-sm sd-border-1
      :link: radio_photometry_container
      :link-type: ref

      Multi-epoch, multi-frequency radio photometry :math:`F_\nu(t, \nu)`.
      Supports epoch grouping, detection masking, and direct inference conversion.

   .. grid-item-card:: Radio Photometry Epoch
      :class-card: sd-shadow-sm sd-border-1
      :link: radio_photometry_epoch
      :link-type: ref

      Single-epoch radio SED. Frequency is the independent variable.
      Designed for broadband spectral fitting at one snapshot in time.

   .. grid-item-card:: Radio Light Curve
      :class-card: sd-shadow-sm sd-border-1
      :link: radio_light_curve
      :link-type: ref

      Single-frequency radio time-series :math:`F_\nu(t)`.
      Observing frequency stored as metadata, not a column.

   .. grid-item-card:: Optical Photometry Container
      :class-card: sd-shadow-sm sd-border-1
      :link: optical_photometry
      :link-type: ref

      Multi-epoch, multi-band optical photometry. Dual flux / AB magnitude
      representation. Band names resolved to model indices at inference time.

   .. grid-item-card:: Optical Photometry Epoch
      :class-card: sd-shadow-sm sd-border-1
      :link: optical_photometry_epoch
      :link-type: ref

      Single-epoch optical SED across multiple bands.
      Band index is the independent variable for broadband SED fitting.

   .. grid-item-card:: Optical Light Curve
      :class-card: sd-shadow-sm sd-border-1
      :link: optical_light_curve
      :link-type: ref

      Single-band optical time-series in flux or AB magnitudes.
      Band name stored as metadata and resolved at inference time.


Choosing the Right Container
------------------------------

The choice of container depends on two things: the wavelength range of the
data and the structure of the observation (time series vs. multi-frequency
snapshot). The table below maps common observational scenarios to the
appropriate class.

.. list-table::
   :header-rows: 1
   :widths: 38 32 30

   * - Observation type
     - Fixed quantity
     - Container
   * - Radio, multiple epochs and frequencies
     - —
     - :class:`~trilobite.data.photometry.RadioPhotometryContainer`
   * - Radio, single epoch, multiple frequencies
     - Time (single epoch)
     - :class:`~trilobite.data.photometry.RadioPhotometryEpoch`
   * - Radio, single frequency, multiple epochs
     - Frequency (metadata)
     - :class:`~trilobite.data.light_curve.RadioLightCurveContainer`
   * - Optical, multiple epochs and bands
     - —
     - :class:`~trilobite.data.optical_photometry.OpticalPhotometryContainer`
   * - Optical, single epoch, multiple bands
     - Time (single epoch)
     - :class:`~trilobite.data.optical_photometry.OpticalPhotometryEpoch`
   * - Optical, single band, multiple epochs
     - Band (metadata)
     - :class:`~trilobite.data.light_curve.OpticalLightCurveContainer`


Common Patterns
----------------

The six containers share a uniform interface. Once you know one, the others
are immediately familiar.

.. tab-set::

   .. tab-item:: Loading data

      Every container supports the same three construction paths:

      .. code-block:: python

          from trilobite.data import RadioPhotometryContainer
          from astropy import units as u

          # From an Astropy Table
          c = RadioPhotometryContainer(table)

          # From a table with column renaming
          c = RadioPhotometryContainer.from_table(
              table,
              column_map={"t": "time", "nu": "freq"},
          )

          # From a file (FITS, CSV, HDF5, …)
          c = RadioPhotometryContainer.from_file("photometry.fits")

      All three paths perform schema validation and unit coercion at
      construction time. If a required column is missing or carries
      incompatible units, a :exc:`ValueError` is raised immediately —
      not silently at inference time.

   .. tab-item:: Accessing data

      All containers expose unit-aware accessors as properties.
      The returned objects are :class:`astropy.units.Quantity` instances,
      so unit conversions are always explicit.

      .. code-block:: python

          c.time               # Quantity in days
          c.flux_density       # Quantity in Jy (radio) or erg/s/cm²/Hz (optical)
          c.n_obs              # total row count (int)
          c.n_detections       # rows where upper limit is NaN (int)
          c.n_non_detections   # rows where upper limit is finite (int)

      Standard table indexing is also supported:

      .. code-block:: python

          c[0]           # first row
          c[:10]         # first ten rows, returned as a new container
          c["freq"]      # raw column access

   .. tab-item:: Detection masking

      All containers classify rows as **detections** or **non-detections**
      (upper limits) based on the ``flux_upper_limit`` column:

      - If ``flux_upper_limit`` is ``NaN``, the row is a detection.
      - If ``flux_upper_limit`` is finite, the row is a non-detection.

      .. code-block:: python

          # Boolean masks
          c.detection_mask       # True = detection
          c.non_detection_mask   # True = upper limit

          # Convenience counts
          c.n_detections
          c.n_non_detections

          # Sub-containers
          det = c.apply_mask(c.detection_mask)
          ul  = c.apply_mask(c.non_detection_mask)

   .. tab-item:: Inference conversion

      Every container provides :meth:`to_inference_data`, which performs
      all unit coercion, column mapping, and validation in one call:

      .. code-block:: python

          inference_data = c.to_inference_data(model)

      Always inspect the result before proceeding:

      .. code-block:: python

          print(inference_data.describe())
          # InferenceData — 32 observations
          # ────────────────────────────────────────────
          # Independent Variables
          #   time  : min=0.10  max=1200.00  (shape=(32,))
          #   freq  : min=1.40  max=22.00    (shape=(32,))
          # ────────────────────────────────────────────
          # Observables
          #   flux_density
          #     detections   : 28
          #     upper limits : 4
          #     error present: True

      Optional parameters give fine-grained control:

      .. code-block:: python

          inference_data = c.to_inference_data(
              model,
              infer_errors=True,          # infer 1σ from upper limits
              detection_threshold=3.0,    # N-sigma assumed for upper limits
              mask=some_boolean_array,    # restrict to a subset of rows
          )


Radio Containers
-----------------

Radio containers all store flux densities in Jansky (Jy) and times in days.
Frequencies are stored in GHz. All three support detection masking and
direct inference conversion.

.. dropdown:: RadioPhotometryContainer — multi-epoch, multi-frequency

   :class:`~trilobite.data.photometry.RadioPhotometryContainer` is the
   most general radio container. Each row is one observation at a specific
   time and frequency:

   .. math::

      F_\nu(t,\, \nu)

   It supports **epoch grouping** — clustering simultaneous or
   near-simultaneous multi-frequency measurements into labelled epochs:

   .. code-block:: python

       # Automatic grouping: observations within 2 days are one epoch
       c.set_epochs_from_time_gaps(2.0 * u.day)

       print(c.n_epochs)      # number of unique epochs
       epoch_0 = c.get_epoch(0)   # rows in epoch 0

   Full reference: :ref:`radio_photometry_container`

.. dropdown:: RadioPhotometryEpoch — single-epoch SED

   :class:`~trilobite.data.photometry.RadioPhotometryEpoch` stores a
   multi-frequency snapshot at a single epoch, where **frequency is the
   independent variable**:

   .. math::

      F_\nu(\nu) \quad \text{at fixed } t

   This is the correct container for broadband radio SED fitting at one
   moment in time. Unlike :class:`~trilobite.data.photometry.RadioPhotometryContainer`,
   there is no time column — the epoch is implicit.

   Full reference: :ref:`radio_photometry_epoch`

.. dropdown:: RadioLightCurveContainer — single-frequency time series

   :class:`~trilobite.data.light_curve.RadioLightCurveContainer` stores
   time-series data at a **fixed observing frequency**:

   .. math::

      F_\nu(t) \quad \text{at fixed } \nu

   The observing frequency is stored as metadata (not a column) and is not
   passed to the model as a variable — it is a fixed property of the
   dataset. This keeps the inference variable set minimal: only ``time``
   varies.

   Full reference: :ref:`radio_light_curve`


Optical Containers
-------------------

Optical containers support **dual representation**: data may be supplied as
flux densities (F\ :sub:`ν` in erg/s/cm²/Hz), AB magnitudes, or both.
Whichever form is absent is computed on-the-fly. Band names are human-readable
strings (``"g"``, ``"r"``, etc.) that are resolved to integer indices at
inference time via the model's :class:`~trilobite.utils.phot_utils.FilterBundle`.

.. dropdown:: OpticalPhotometryContainer — multi-epoch, multi-band

   :class:`~trilobite.data.optical_photometry.OpticalPhotometryContainer`
   is the optical analog of
   :class:`~trilobite.data.photometry.RadioPhotometryContainer`.
   Each row is one observation in a specific band at a specific time:

   .. math::

      F_\nu(t,\, \mathrm{band})

   Band names (``"g"``, ``"r"``, …) are stored as strings in a ``band_name``
   column. At inference time they are mapped to integer ``band_idx`` values
   by looking them up in ``model.bundle.filter_names``.

   Dual-representation example:

   .. code-block:: python

       # Load from magnitude columns — flux is computed on demand
       c = OpticalPhotometryContainer.from_table(mag_table)
       c.flux    # F_nu in erg/s/cm^2/Hz  (converted automatically)
       c.mag     # AB magnitude            (stored directly)

   Full reference: :ref:`optical_photometry`

.. dropdown:: OpticalPhotometryEpoch — single-epoch SED

   :class:`~trilobite.data.optical_photometry.OpticalPhotometryEpoch`
   stores a snapshot across multiple bands at a single epoch, where
   **band index is the independent variable**:

   .. math::

      F_\nu(\mathrm{band}) \quad \text{at fixed } t

   This is the correct container for broadband optical SED fitting at one
   epoch. There is no time column.

   Full reference: :ref:`optical_photometry_epoch`

.. dropdown:: OpticalLightCurveContainer — single-band time series

   :class:`~trilobite.data.light_curve.OpticalLightCurveContainer` stores
   a time series in a **fixed photometric band**. Data may be supplied as
   flux density or AB magnitude; both forms are always accessible regardless
   of input format.

   The band name is stored as metadata. At inference time it is resolved to
   an integer index via ``model.bundle.filter_names`` and broadcast as a
   constant array — every observation has the same ``band_idx``.

   Full reference: :ref:`optical_light_curve`


InferenceData
--------------

:class:`~trilobite.data.core.InferenceData` is the numerical bridge between
the data layer and the inference layer. It contains only validated NumPy arrays
— no units, no column names, no schema.

.. code-block:: python

    inference_data = container.to_inference_data(model)

    # The two primary structures:
    inference_data.x                    # dict: variable name → array
    inference_data.observables          # dict: observable name → Observable

    # Stacked arrays for use in samplers:
    X      = inference_data.get_x_array()
    Y      = inference_data.get_y_array()
    Y_err  = inference_data.get_y_error_array()
    Y_ul   = inference_data.get_y_upper_array()

Once data are in ``InferenceData`` form, likelihood evaluation is a pure
numerical operation — no hidden unit coercion, no column resolution,
no ambiguity in array shape or ordering.

Full reference: :ref:`inference_data`

Full pipeline walkthrough: :ref:`data_to_inference`


Design Philosophy
------------------

The data module is deliberately layered around **three distinct responsibilities**:

.. list-table::
   :widths: 28 28 44
   :header-rows: 1

   * - Layer
     - Class
     - Responsibility
   * - Observational semantics
     - :class:`~trilobite.data.core.DataContainer` subclasses
     - Column names, units, detection logic, epoch grouping,
       dual flux/magnitude representation.
   * - Numerical inference representation
     - :class:`~trilobite.data.core.InferenceData`
     - Validated NumPy arrays matched to a model's declared
       variables and outputs. No schema, no units.
   * - Statistical assumptions
     - :class:`~trilobite.inference.likelihood.base.GaussianLikelihood` (and siblings)
     - Noise model, censoring, likelihood evaluation. Consumes
       InferenceData only.

Keeping these three responsibilities in separate objects makes each layer
independently testable, and ensures that likelihood evaluation is free of
data-handling concerns.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Light Curves

   light_curve

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Radio Photometry

   photometry
   radio_photometry_epoch

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Optical Photometry

   optical_photometry
   optical_photometry_epoch

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Inference Integration

   inference_data
   data_to_inference
