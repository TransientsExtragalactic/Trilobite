.. _radio_photometry:

============================
Radio Photometry
============================

Triceratops provides two containers for radio photometric data:

- :class:`~triceratops.data.photometry.RadioPhotometryContainer` — multi-epoch,
  multi-frequency radio photometry (the most common case).
- :class:`~triceratops.data.photometry.RadioPhotometryEpoch` — single-epoch radio
  SED, where frequency is the independent variable. See :ref:`radio_photometry_epoch`.

Both containers wrap an :class:`astropy.table.Table` with an enforced schema,
provide unit-aware accessors, and expose a
:meth:`~triceratops.data.photometry.RadioPhotometryContainer.to_inference_data`
method for direct integration with the inference pipeline.


.. _radio_photometry_container:

RadioPhotometryContainer
-------------------------

:class:`~triceratops.data.photometry.RadioPhotometryContainer` stores
heterogeneous multi-frequency, multi-epoch radio observations:

.. math::

    F_\nu(t,\, \nu)

where each row represents one measurement at a given time and frequency.


Schema
^^^^^^

The container enforces the following column schema. Columns marked Required must
be present; the rest are optional but are recognised and unit-validated when
present.

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 10 46

   * - Column
     - Type
     - Unit
     - Required
     - Description
   * - ``time``
     - float
     - ``day``
     - Yes
     - Relative time of the observation (with respect to a user-defined reference).
   * - ``freq``
     - float
     - ``GHz``
     - Yes
     - Central observing frequency.
   * - ``flux_density``
     - float
     - ``Jy``
     - Yes
     - Measured flux density. Set to ``np.nan`` for non-detections.
   * - ``flux_density_error``
     - float
     - ``Jy``
     - Yes
     - 1σ uncertainty on ``flux_density``. Set to ``np.nan`` for non-detections.
   * - ``flux_upper_limit``
     - float
     - ``Jy``
     - Yes
     - Upper limit for non-detections. Set to ``np.nan`` for detections.
   * - ``obs_time``
     - float
     - ``day``
     - No
     - Total integration time of the observation.
   * - ``obs_name``
     - str
     - —
     - No
     - Free-form observation identifier (e.g. telescope + epoch label).
   * - ``band``
     - int
     - —
     - No
     - Instrument-specific integer band identifier.
   * - ``epoch_id``
     - int
     - —
     - No
     - Integer epoch grouping label (see :ref:`Epoch Support <radio_photometry_epochs>`).
   * - ``comments``
     - str
     - —
     - No
     - Free-form comments or metadata.

.. note::

    Time is always **relative** — it is measured with respect to some reference
    event (explosion time, trigger time, etc.) chosen by the user. The absolute
    reference time is not stored in the container.

    Units must be *compatible* with the declared units but do not need to match
    exactly. For example, ``freq`` can be supplied in MHz or Hz and will be
    converted automatically.

    For non-detections: set ``flux_density`` and ``flux_density_error`` to
    ``np.nan``, and provide the upper limit in ``flux_upper_limit``.


Construction
^^^^^^^^^^^^

From an Astropy Table
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from astropy.table import Table
    from astropy import units as u
    from triceratops.data import RadioPhotometryContainer

    t = Table({
        "time":                np.array([1., 5., 10., 20., 50.]) * u.day,
        "freq":                np.array([5.5, 8.5, 5.5, 8.5, 5.5]) * u.GHz,
        "flux_density":        np.array([1.2, 1.5, 1.1, 0.9, np.nan]) * u.Jy,
        "flux_density_error":  np.array([0.1, 0.1, 0.1, 0.1, np.nan]) * u.Jy,
        "flux_upper_limit":    np.array([np.nan, np.nan, np.nan, np.nan, 0.3]) * u.Jy,
    })

    c = RadioPhotometryContainer(t)
    print(c.n_obs, c.n_detections, c.n_non_detections)
    # → 5  4  1

If your table uses different column names, supply a ``column_map``:

.. code-block:: python

    c = RadioPhotometryContainer.from_table(t, column_map={
        "t_day":    "time",
        "nu_GHz":   "freq",
        "flux_Jy":  "flux_density",
    })

To convert from absolute times (MJD, Unix, etc.) to relative days at load time:

.. code-block:: python

    from astropy.time import Time

    c = RadioPhotometryContainer.from_table(
        t,
        internal_time_format="mjd",
        time_start=Time(59000.0, format="mjd"),
    )

From a File
~~~~~~~~~~~

Any format supported by :meth:`astropy.table.Table.read` works:

.. code-block:: python

    c = RadioPhotometryContainer.from_file("photometry.fits")
    c = RadioPhotometryContainer.from_file("photometry.csv", format="ascii.csv")


Data Access
^^^^^^^^^^^

All column accessors return :class:`astropy.units.Quantity` objects in the
units declared in the schema.

.. code-block:: python

    c.time                  # Quantity, units: day
    c.freq                  # Quantity, units: GHz
    c.flux_density          # Quantity, units: Jy
    c.flux_density_error    # Quantity, units: Jy
    c.flux_upper_limit      # Quantity, units: Jy

    c.n_obs                 # int: total row count
    c.n_detections          # int: rows where flux_upper_limit is NaN
    c.n_non_detections      # int: rows where flux_upper_limit is finite

Standard table indexing is also supported:

.. code-block:: python

    c[0]                    # first row as a Table row
    c[:5]                   # first five rows (returns a new container)
    c["flux_density"]       # raw column access


Detection and Non-Detection Masks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A row is a **detection** when its ``flux_upper_limit`` is ``NaN``.
A row is a **non-detection** (upper limit) when ``flux_upper_limit`` is finite.

.. code-block:: python

    c.detection_mask        # bool array: True = detection
    c.non_detection_mask    # bool array: True = upper limit
    c.detection_table       # sub-table of detections only
    c.non_detection_table   # sub-table of upper limits only

To extract a subset:

.. code-block:: python

    detections     = c.apply_mask(c.detection_mask)
    non_detections = c.apply_mask(c.non_detection_mask)
    first_ten      = c.apply_mask(np.arrange(10))


.. _radio_photometry_epochs:

Epoch Support
^^^^^^^^^^^^^

Observations may be grouped into **epochs** — sets of measurements that are
simultaneous or near-simultaneous (e.g. a multi-frequency observing block).

Assign epochs manually via the ``epoch_id`` column before construction:

.. code-block:: python

    t["epoch_id"] = [0, 0, 1, 1, 2]   # rows 0–1 in epoch 0, rows 2–3 in epoch 1, etc.
    c = RadioPhotometryContainer(t)

Or generate epochs automatically from the data:

.. code-block:: python

    # Group observations separated by less than 2 days into the same epoch
    c.set_epochs_from_time_gaps(2.0 * u.day)

    # Assign explicit epoch IDs by index
    c.set_epochs_from_indices([0, 0, 0, 1, 1, 1, 2, 2])

    # Bin into fixed-width time windows
    bins = np.arrange(0, 200, 10) * u.day
    c.set_epochs_from_bins(bins)

Once epochs are set:

.. code-block:: python

    c.has_epochs            # True
    c.n_epochs              # number of unique epochs
    c.epoch_ids             # integer epoch ID per observation

    epoch_0 = c.get_epoch(0)          # all rows in epoch 0
    mask    = c.get_epoch_mask(0)     # boolean mask for epoch 0
    table   = c.get_epoch_table(0)    # as a Table

Plotting
^^^^^^^^

.. code-block:: python

    fig, axes = c.plot_photometry()

.. plot::

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.table import Table
    from astropy import units as u
    from triceratops.data import RadioPhotometryContainer

    rng = np.random.default_rng(0)
    n = 24
    time = np.sort(rng.uniform(1, 500, n))
    freqs = np.tile([5.5, 8.5, 15.0], 8)
    flux = 5e-26 * (time / 10.0) ** -0.8 * rng.lognormal(0, 0.08, n)
    err = flux * 0.1

    flux_arr = np.where(time > 400, np.nan, flux)
    err_arr = np.where(time > 400, np.nan, err)
    ul = np.where(time > 400, 5e-28, np.nan)

    t = Table({
        "time": time * u.day,
        "freq": freqs * u.GHz,
        "flux_density": flux_arr * u.Jy,
        "flux_density_error": err_arr * u.Jy,
        "flux_upper_limit": ul * u.Jy,
    })
    c = RadioPhotometryContainer(t)

    colors = {5.5: "steelblue", 8.5: "tomato", 15.0: "seagreen"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for nu in [5.5, 8.5, 15.0]:
        mask = (c.freq.value == nu) & c.detection_mask
        ax.errorbar(c.time[mask].value, c.flux_density[mask].value,
                    yerr=c.flux_density_error[mask].value,
                    fmt="o", label=f"{nu} GHz", color=colors[nu])
        ul_mask = (c.freq.value == nu) & c.non_detection_mask
        if ul_mask.any():
            ax.errorbar(c.time[ul_mask].value, c.flux_upper_limit[ul_mask].value,
                        fmt="v", color=colors[nu], alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Flux density (Jy)")
    ax.set_title("Synthetic multi-frequency radio photometry")
    ax.legend()
    plt.tight_layout()


Inference Integration
^^^^^^^^^^^^^^^^^^^^^

:meth:`~triceratops.data.photometry.RadioPhotometryContainer.to_inference_data`
converts the container into an :class:`~triceratops.data.core.InferenceData`
object ready for likelihood evaluation.

.. code-block:: python

    inference_data = c.to_inference_data(model)
    print(inference_data.describe())

Optional parameters:

.. code-block:: python

    inference_data = c.to_inference_data(
        model,
        infer_errors=True,         # infer 1σ errors from upper limits
        detection_threshold=3.0,   # N-sigma assumed for upper limits
        mask=c.detection_mask,     # restrict to detections only
    )

If the model's variable names don't match the container's default column names
(``time`` and ``freq``), supply an explicit mapping:

.. code-block:: python

    inference_data = c.to_inference_data(
        model,
        variables={"frequency": "freq"},          # model name → container column
        observables={
            "flux_density": (
                "flux_density",
                "flux_density_error",
                "flux_upper_limit",
                None,                             # no lower limits
            )
        },
    )

See :ref:`data_to_inference` for the full step-by-step pipeline guide.


API Reference
^^^^^^^^^^^^^

.. autoclass:: triceratops.data.photometry.RadioPhotometryContainer
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:


----


RadioPhotometryEpoch
---------------------

See :ref:`radio_photometry_epoch` for the full reference.

.. autoclass:: triceratops.data.photometry.RadioPhotometryEpoch
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
