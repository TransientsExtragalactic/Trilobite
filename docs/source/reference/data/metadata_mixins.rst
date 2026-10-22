.. _data_metadata_mixins:

=====================================
Container Metadata and Mixin Defaults
=====================================

Every :class:`~trilobite.data.core.DataContainer` subclass stores an optional
dictionary of user-supplied metadata under ``container.meta``. This dictionary
is persisted inside the underlying Astropy table so it survives all I/O
round-trips. Beyond simple key–value storage, Trilobite uses this dictionary to
drive a set of **metadata mixins** — lightweight capability classes that
activate when specific keys are present. Mixins let you attach physical context
(cosmological distances, filter curves, detection thresholds) directly to a
container at construction time, rather than having to pass the same values
repeatedly to every downstream method.

Each mixin implements a ``_setup_mixin`` hook that is called automatically
during construction.  If the required key is absent, the hook exits silently
and the mixin is effectively a no-op; it never raises unless a malformed value
is provided.

----

Attaching Metadata to a Container
----------------------------------

Pass a ``metadata`` dict to any container constructor or ``from_table`` call.
Keys already present in the underlying Astropy table's ``meta`` block are not
overwritten.

.. code-block:: python

    from trilobite.data import RadioPhotometryContainer

    container = RadioPhotometryContainer.from_table(
        table,
        metadata={
            "redshift": 0.31,
            "n_sigma_det_thresh": 3.0,
        },
    )

For optical containers, use ``filters`` together with ``n_sigma_det_thresh``:

.. code-block:: python

    from trilobite.data import OpticalPhotometryContainer

    container = OpticalPhotometryContainer.from_table(
        table,
        metadata={
            "redshift": 0.031,
            "filters": {"g": "SDSS/SDSS.g", "r": "SDSS/SDSS.r"},
            "n_sigma_det_thresh": 5.0,
        },
    )

Metadata can also be inspected or updated at any time through the ``meta``
property, which returns a live reference to the underlying dict:

.. code-block:: python

    container.meta["n_sigma_det_thresh"] = 5.0

----

Available Mixins
-----------------

.. _cosmology_mixin:

CosmologyMixin
^^^^^^^^^^^^^^

:class:`~trilobite.data.metadata.CosmologyMixin` is active on **all**
:class:`~trilobite.data.core.DataContainer` subclasses. It resolves a full
set of cosmological distances from a single user-supplied distance measure.

**Trigger keys** (provide exactly one):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Value
   * - ``redshift``
     - Dimensionless float
   * - ``luminosity_distance``
     - Float in Mpc, or :class:`~astropy.units.Quantity`
   * - ``angular_diameter_distance``
     - Float in Mpc, or :class:`~astropy.units.Quantity`
   * - ``proper_distance``
     - Float in Mpc, or :class:`~astropy.units.Quantity`

**Optional key:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Value
   * - ``cosmology_spec``
     - Named Astropy cosmology string (e.g. ``"Planck18"``) or an
       :class:`~astropy.cosmology.FLRW` object. Defaults to the Trilobite
       configured cosmology.

When triggered, the mixin resolves all four distance measures and exposes them
as properties:

.. code-block:: python

    container = RadioPhotometryContainer.from_table(
        table, metadata={"redshift": 0.31}
    )

    container.redshift                    # 0.31 (float)
    container.luminosity_distance         # ~1650 Mpc (Quantity)
    container.angular_diameter_distance   # Quantity
    container.proper_distance             # Quantity
    container.cosmology                   # Planck18 (or configured default)

.. note::

   Cosmological distances are resolved at construction time using
   :func:`~trilobite.physics_utils.cosmology.resolve_cosmological_distances`.
   The resolved values are cached as private attributes and are not updated
   if ``meta`` is modified after construction.

.. _filter_mixin:

FilterMixin
^^^^^^^^^^^

:class:`~trilobite.data.metadata.FilterMixin` is active on all **optical**
containers:

- :class:`~trilobite.data.optical_photometry.OpticalPhotometryContainer`
- :class:`~trilobite.data.optical_photometry.OpticalPhotometryEpoch`
- :class:`~trilobite.data.light_curve.OpticalLightCurveContainer`

It stores a ``{band_name: svo_filter_id}`` mapping from metadata and provides
a :meth:`~trilobite.data.metadata.FilterMixin.load_svo_filters` method to
fetch transmission curves from the `SVO Filter Profile Service
<https://svo2.cab.inta-csic.es/svo/theory/fps/>`_ on demand.

**Trigger key:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Value
   * - ``filters``
     - ``dict`` of ``{band_name: svo_filter_id}``, e.g.
       ``{"g": "SDSS/SDSS.g", "r": "SDSS/SDSS.r"}``

At construction the mixin validates the ``filters`` dict and records the
band-to-SVO-ID mapping.  Transmission curves are **not** downloaded at this
stage; call :meth:`~trilobite.data.metadata.FilterMixin.load_svo_filters`
explicitly when you are ready to fetch them:

.. code-block:: python

    from trilobite.data import OpticalPhotometryContainer

    container = OpticalPhotometryContainer.from_table(
        table,
        metadata={"filters": {"g": "SDSS/SDSS.g", "r": "SDSS/SDSS.r"}},
    )

    # Inspect the registered band-to-SVO mapping (no network call)
    container.filter_specs
    # {"g": "SDSS/SDSS.g", "r": "SDSS/SDSS.r"}

    # Fetch transmission curves (makes an HTTP request on first call;
    # subsequent calls are served from astroquery's local cache)
    bundle = container.load_svo_filters(weighting="photon", cache=True)

    # bundle is stored and accessible as a property
    container.filter_bundle
    # FilterBundle(['g', 'r'], N_common=...)

.. tip::

   Save the returned :class:`~trilobite.utils.phot_utils.FilterBundle` to
   disk (``bundle.to_hdf5("filters.h5")``) so subsequent runs don't need a
   network connection. See :ref:`filter_photometry` for the full filter API.

**Integration with** ``to_inference_data``

Once a bundle is loaded, it becomes the **default** ``filters`` argument for
:meth:`~trilobite.data.optical_photometry.OpticalPhotometryContainer.to_inference_data`.
You no longer need to pass it explicitly:

.. code-block:: python

    # Before: filters must be passed every time
    idata = container.to_inference_data(filters=bundle)

    # After load_svo_filters: filters default to the loaded bundle
    container.load_svo_filters()
    idata = container.to_inference_data()   # equivalent

If ``filters=None`` and no bundle has been loaded, the method raises a
:exc:`TypeError` with a message directing you to call
:meth:`~trilobite.data.metadata.FilterMixin.load_svo_filters` first.

.. _detection_threshold_mixin:

DetectionThresholdMixin
^^^^^^^^^^^^^^^^^^^^^^^

:class:`~trilobite.data.metadata.DetectionThresholdMixin` is active on
**all six** data containers — both optical and radio.  It records an
:math:`N_\sigma` detection threshold that controls how upper-limit errors
are inferred when building :class:`~trilobite.data.core.InferenceData`.

**Trigger key:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Value
   * - ``n_sigma_det_thresh``
     - Positive float. An observation satisfying
       :math:`F / \sigma_F \ge N_\sigma` is treated as a detection.

The value is coerced to ``float`` and validated (must be :math:`> 0`) at
construction time:

.. code-block:: python

    container = RadioPhotometryContainer.from_table(
        table, metadata={"n_sigma_det_thresh": 3.0}
    )
    container.detection_threshold   # 3.0

**Integration with** ``to_inference_data``

When ``detection_threshold`` is not passed explicitly to
``to_inference_data``, the method falls back to
:attr:`~trilobite.data.metadata.DetectionThresholdMixin.detection_threshold`
from metadata.  If neither is available, the hard-coded default of ``3.0``
is used:

.. code-block:: python

    # Explicit override — always takes priority
    idata = container.to_inference_data(detection_threshold=5.0)

    # Metadata default — used when detection_threshold is not passed
    container.meta["n_sigma_det_thresh"] = 5.0
    idata = container.to_inference_data()   # uses 5.0

    # Neither provided — silently uses 3.0
    container2 = RadioPhotometryContainer.from_table(table)
    idata = container2.to_inference_data()  # uses 3.0

----

Combining Mixins
-----------------

All mixins compose cleanly.  A full optical workflow that attaches cosmology,
filters, and a detection threshold in one construction call looks like this:

.. code-block:: python

    from trilobite.data import OpticalPhotometryContainer

    container = OpticalPhotometryContainer.from_table(
        table,
        metadata={
            "redshift": 0.031,
            "filters": {
                "g": "SDSS/SDSS.g",
                "r": "SDSS/SDSS.r",
                "i": "SDSS/SDSS.i",
            },
            "n_sigma_det_thresh": 5.0,
        },
    )

    # Fetch filter curves once (cached to disk afterward)
    container.load_svo_filters()

    # All three mixin properties now available
    container.redshift            # 0.031
    container.luminosity_distance # ~133 Mpc
    container.filter_bundle       # FilterBundle(['g', 'r', 'i'])
    container.detection_threshold # 5.0

    # to_inference_data needs no extra arguments
    idata = container.to_inference_data()

The resolution order for each ``to_inference_data`` argument is:

1. Explicit keyword argument (highest priority).
2. Mixin metadata value (``filter_bundle`` or ``detection_threshold``).
3. Built-in default — ``3.0`` for ``detection_threshold``; ``TypeError`` for
   ``filters`` when no bundle has been loaded.

----

.. seealso::

    - :ref:`filter_photometry` — full reference for
      :class:`~trilobite.utils.phot_utils.PhotometryFilter` and
      :class:`~trilobite.utils.phot_utils.FilterBundle`
    - :ref:`data_to_inference` — end-to-end pipeline walkthrough
    - :class:`~trilobite.data.metadata.CosmologyMixin` — API reference
    - :class:`~trilobite.data.metadata.FilterMixin` — API reference
    - :class:`~trilobite.data.metadata.DetectionThresholdMixin` — API reference
