.. _optical_photometry:

============================
Optical Photometry
============================

The :class:`~triceratops.data.optical_photometry.OpticalPhotometryContainer` provides a
schema-validated, unit-aware interface to optical photometric observations from surveys such
as ZTF, LSST/Rubin, and DECam.  It is the optical counterpart to
:class:`~triceratops.data.photometry.RadioPhotometryContainer` and sits at the same tier in
the Triceratops pipeline: between raw tabular data and the likelihood evaluation layer.

Data may be provided as **AB magnitudes**, as physical **flux densities**
(F\ :sub:`ν` in erg/s/cm²/Hz), or both.  The container exposes either representation through
properties that compute the missing form on-the-fly, and always converts to physical flux units
when building an :class:`~triceratops.data.core.InferenceData` object for inference.

Band identification uses human-readable **band names** (e.g. ``"g"``, ``"r"``, ``"i"``)
stored directly in the table.  Integer indices — needed by the :class:`~triceratops.utils.phot_utils.FilterBundle`
inside the model — are resolved at :meth:`~triceratops.data.optical_photometry.OpticalPhotometryContainer.to_inference_data`
time by looking each name up in ``model.bundle.filter_names``.  This keeps the container
decoupled from any particular model or filter ordering convention.


.. rubric:: In this section

- `Schema`_
- `Construction`_
- `Dual-Representation Properties`_
- `Detection / Non-Detection Logic`_
- `Epoch Support`_
- `Connecting to Inference`_
- `API Reference`_

----

Schema
------

The following columns are recognised by the container.  At least one y-column group
(**flux** or **magnitude**) must be present.

.. list-table::
   :header-rows: 1
   :widths: 25 12 10 53

   * - Column
     - Type
     - Required
     - Description
   * - ``time``
     - ``float``
     - Yes
     - Relative time since some reference epoch [days].
   * - ``band_name``
     - ``str``
     - Yes
     - Survey band identifier (e.g. ``"g"``, ``"r"``, ``"i"``).
   * - ``flux_density``
     - ``float``
     - No†
     - Measured flux density F\ :sub:`ν` [erg/s/cm²/Hz] for detections.
   * - ``flux_density_error``
     - ``float``
     - No
     - 1-σ uncertainty on ``flux_density`` [erg/s/cm²/Hz].
   * - ``flux_upper_limit``
     - ``float``
     - No
     - Upper limit on F\ :sub:`ν` [erg/s/cm²/Hz] for non-detections; ``NaN`` for detections.
   * - ``mag_ab``
     - ``float``
     - No†
     - AB magnitude for detections (dimensionless).
   * - ``mag_ab_error``
     - ``float``
     - No
     - 1-σ uncertainty on ``mag_ab``.
   * - ``mag_ab_upper_limit``
     - ``float``
     - No
     - AB magnitude upper limit for non-detections; ``NaN`` for detections.
   * - ``obs_name``
     - ``str``
     - No
     - Free-form observation identifier.
   * - ``epoch_id``
     - ``int``
     - No
     - Integer epoch grouping label.

†At least one of ``flux_density`` or ``mag_ab`` must be present.

----

Construction
------------

The container is most commonly built via :meth:`~triceratops.data.optical_photometry.OpticalPhotometryContainer.from_table`
or :meth:`~triceratops.data.optical_photometry.OpticalPhotometryContainer.from_file`.

**From a table with flux columns:**

.. code-block:: python

    import numpy as np
    import astropy.units as u
    from astropy.table import Table
    from triceratops.data import OpticalPhotometryContainer

    t = Table({
        "time":                [1., 2., 3., 4., 5.] * u.day,
        "band_name":           ["g", "r", "i", "g", "r"],
        "flux_density":        [1e-27, 2e-27, 3e-27, 4e-27, np.nan] * u.Unit("erg/(s cm2 Hz)"),
        "flux_density_error":  [1e-28, 2e-28, 3e-28, 4e-28, np.nan] * u.Unit("erg/(s cm2 Hz)"),
        "flux_upper_limit":    [np.nan, np.nan, np.nan, np.nan, 1e-26] * u.Unit("erg/(s cm2 Hz)"),
    })

    c = OpticalPhotometryContainer.from_table(t)

**From a table with magnitude columns:**

.. code-block:: python

    t = Table({
        "time":               [1., 2., 3., 4., 5.] * u.day,
        "band_name":          ["g", "r", "i", "g", "r"],
        "mag_ab":             [20.1, 20.5, 21.0, 20.3, np.nan],
        "mag_ab_error":       [0.05, 0.06, 0.07, 0.05, np.nan],
        "mag_ab_upper_limit": [np.nan, np.nan, np.nan, np.nan, 23.0],
    })

    c = OpticalPhotometryContainer.from_table(t)

**Absolute time conversion:**

Observations stored as MJD (or any Astropy time format) can be converted to relative days
at load time:

.. code-block:: python

    from astropy.time import Time

    c = OpticalPhotometryContainer.from_table(
        table,
        internal_time_format="mjd",
        time_start=Time(59000.0, format="mjd"),
    )

**Column renaming:**

If your table uses non-standard column names, supply a ``column_map``:

.. code-block:: python

    c = OpticalPhotometryContainer.from_table(
        table,
        column_map={"filter": "band_name", "mjd": "time"},
    )

----

Dual-Representation Properties
-------------------------------

The container always exposes both representations, regardless of which columns are stored:

.. code-block:: python

    # If data was loaded as magnitudes:
    c.flux            # → F_nu as Quantity [erg/s/cm²/Hz]  (converted on-the-fly)
    c.flux_error      # → σ_F as Quantity  (σ_F = F * σ_m / 1.0857)
    c.flux_upper_limit # → upper limit as Quantity (NaN for detections)

    # If data was loaded as flux:
    c.mag             # → AB magnitude as ndarray  (converted on-the-fly)
    c.mag_error       # → σ_m as ndarray  (σ_m = 1.0857 * σ_F / F)
    c.mag_upper_limit  # → mag upper limit  (NaN for detections)

    # Always available:
    c.time            # → Quantity [days]
    c.band_name       # → str array of band identifiers

The conversion formulae are:

.. math::

    F_\nu = 3.631 \times 10^{-20} \cdot 10^{-0.4\,m_\mathrm{AB}} \;\text{erg/s/cm}^2/\text{Hz}

    \sigma_F = F_\nu \cdot \sigma_m / 1.0857

    m_\mathrm{AB} = -2.5 \log_{10}(F_\nu / 3.631 \times 10^{-20})

    \sigma_m = 1.0857 \cdot \sigma_F / F_\nu

----

Detection / Non-Detection Logic
---------------------------------

A row is classified as a **detection** when its upper-limit column contains ``NaN``.
``flux_upper_limit`` takes priority over ``mag_ab_upper_limit``; if neither is present
all rows are treated as detections.

.. code-block:: python

    c.detection_mask      # → bool array: True = detection
    c.non_detection_mask  # → bool array: True = upper limit
    c.n_detections        # → int
    c.n_non_detections    # → int

    det = c.detections    # → sub-container of detections only
    ul  = c.upper_limits  # → sub-container of upper limits only

----

Epoch Support
-------------

Observations can be grouped into epochs using the same API as
:class:`~triceratops.data.photometry.RadioPhotometryContainer`:

.. code-block:: python

    # Explicit assignment
    c.set_epochs_from_indices([0, 0, 1, 1, 2])

    # Automatic grouping by time gap
    c.set_epochs_from_time_gaps(3.0 * u.day)

    print(c.n_epochs)   # number of unique epochs
    print(c.epoch_ids)  # integer epoch ID per observation

----

Connecting to Inference
-----------------------

:meth:`~triceratops.data.optical_photometry.OpticalPhotometryContainer.to_inference_data`
bridges the container to the Triceratops inference pipeline.  It performs three key steps:

1. **Band mapping** — looks up each ``band_name`` in ``model.bundle.filter_names`` to produce
   the ``band_idx`` array stored in ``InferenceData.x``.
2. **Flux conversion** — converts magnitudes to F\ :sub:`ν` (if the data are in mag space).
3. **Error inference** — optionally fills missing uncertainties for non-detections from their
   upper limits (``infer_errors=True``, ``detection_threshold=3.0``).

The resulting :class:`~triceratops.data.core.InferenceData` has
``x = {"time": ..., "band_idx": ...}`` and ``y = {"flux_density": ...}``, which is exactly
what :class:`~triceratops.inference.likelihood.GaussianLikelihood` expects.

.. code-block:: python

    # Assuming `model` is an optical model with a FilterBundle attribute:
    idata = c.to_inference_data(model)

    # Subset to detections only, with explicit detection threshold:
    idata = c.to_inference_data(
        model,
        mask=c.detection_mask,
        infer_errors=False,
    )

**Optical model pattern** — the container is designed to work with models of the form:

.. code-block:: python

    class OpticalTransientModel(Model):
        VARIABLES = [ModelVariable("time", u.s), ModelVariable("band_idx", u.dimensionless_unscaled)]

        def __init__(self, bundle: FilterBundle, **kwargs):
            self._bundle = bundle
            super().__init__(**kwargs)

        def _forward_model(self, time, band_idx, **theta):
            F_nu_sed = self._compute_sed(time, **theta)       # (N_obs, N_grid)
            F_bands  = self._bundle.apply(F_nu_sed)           # (N_obs, N_filters)
            return F_bands[np.arrange(len(band_idx)), band_idx] # (N_obs,)

The ``FilterBundle``'s ``filter_names`` list is the single source of truth for the
``band_name → band_idx`` mapping.

----

API Reference
-------------

.. autoclass:: triceratops.data.optical_photometry.OpticalPhotometryContainer
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:


----


OpticalPhotometryEpoch
-----------------------

See :ref:`optical_photometry_epoch` for the full reference.

.. autoclass:: triceratops.data.optical_photometry.OpticalPhotometryEpoch
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
