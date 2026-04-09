.. _optical_photometry_epoch:

========================
OpticalPhotometryEpoch
========================

:class:`~triceratops.data.optical_photometry.OpticalPhotometryEpoch` represents
a **single-epoch multi-band optical SED**. It stores flux density or AB
magnitude measurements taken at one snapshot in time across multiple
photometric bands, making it suitable for broadband optical SED fitting
where band index is the independent variable.

This is the optical analog of
:class:`~triceratops.data.photometry.RadioPhotometryEpoch`. The band name
(e.g., ``"g"``, ``"r"``) is stored as a string column and resolved to an
integer band index at inference time via the model's
:class:`~triceratops.utils.phot_utils.FilterBundle`.


Schema
------

.. list-table::
    :header-rows: 1
    :widths: 25 45 15 15

    * - Column Name
      - Description
      - Default Unit
      - Required
    * - ``band_name``
      - Photometric band name (e.g. ``"g"``, ``"r"``)
      - None (string)
      - Yes
    * - ``flux_density``
      - Measured flux density (detections; NaN for upper limits)
      - ``erg s⁻¹ cm⁻² Hz⁻¹``
      - No *
    * - ``flux_density_error``
      - 1σ uncertainty on flux density
      - ``erg s⁻¹ cm⁻² Hz⁻¹``
      - No *
    * - ``flux_upper_limit``
      - Upper limit on flux density (non-detections; NaN for detections)
      - ``erg s⁻¹ cm⁻² Hz⁻¹``
      - No *
    * - ``mag_ab``
      - AB magnitude (detections; NaN for upper limits)
      - dimensionless
      - No *
    * - ``mag_ab_error``
      - 1σ uncertainty on AB magnitude
      - dimensionless
      - No *
    * - ``mag_ab_upper_limit``
      - Upper limit on AB magnitude (non-detections; NaN for detections)
      - dimensionless
      - No *
    * - ``obs_name``
      - Optional observation identifier
      - None
      - No

.. note::

    At least one of the flux density or magnitude column groups must be present.
    If only magnitudes are provided, flux is computed on-the-fly from the AB
    magnitude convention. ``band_name`` is always required.


Construction
------------

From an Astropy Table
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from astropy.table import Table
    from astropy import units as u
    from triceratops.data import OpticalPhotometryEpoch

    # From flux density columns
    table = Table({
        "band_name": ["g", "r", "i", "z"],
        "flux_density": [1.2e-28, 2.3e-28, 2.8e-28, np.nan] * u.Unit("erg/(s cm2 Hz)"),
        "flux_density_error": [0.1e-28, 0.2e-28, 0.2e-28, np.nan] * u.Unit("erg/(s cm2 Hz)"),
        "flux_upper_limit": [np.nan, np.nan, np.nan, 1.0e-28] * u.Unit("erg/(s cm2 Hz)"),
    })

    epoch = OpticalPhotometryEpoch(table)

From AB magnitudes:

.. code-block:: python

    table = Table({
        "band_name": ["g", "r", "i"],
        "mag_ab": [22.1, 21.5, 21.0],
        "mag_ab_error": [0.05, 0.04, 0.05],
        "mag_ab_upper_limit": [np.nan, np.nan, np.nan],
    })

    epoch = OpticalPhotometryEpoch(table)

A :meth:`from_table` classmethod is also available:

.. code-block:: python

    epoch = OpticalPhotometryEpoch.from_table(table, column_map={
        "band": "band_name",
        "flux": "flux_density",
    })

From a File
^^^^^^^^^^^

.. code-block:: python

    epoch = OpticalPhotometryEpoch.from_file("optical_epoch.fits")


Dual Flux/Magnitude Representation
------------------------------------

Both representations are always accessible regardless of input format:

.. code-block:: python

    # Flux density (erg/s/cm^2/Hz)
    epoch.flux            # with magnitude→flux conversion if needed
    epoch.flux_error
    epoch.flux_upper_limit

    # AB magnitudes
    epoch.mag             # with flux→magnitude conversion if needed
    epoch.mag_error
    epoch.mag_upper_limit

    # Band names
    epoch.band_name       # array of strings, e.g. ['g', 'r', 'i']


Detection Semantics
--------------------

.. code-block:: python

    epoch.detection_mask        # True = detection
    epoch.non_detection_mask    # True = upper limit (non-detection)
    epoch.n_detections
    epoch.n_non_detections
    epoch.detection_table
    epoch.non_detection_table

    epoch.apply_mask(mask)      # returns a new OpticalPhotometryEpoch


Inference Integration
----------------------

.. code-block:: python

    inference_data = epoch.to_inference_data(model)

This call:

1. Inspects ``model.bundle.filter_names`` (e.g., ``['g', 'r', 'i', 'z']``).
2. Resolves each string in the ``band_name`` column to an integer index.
3. Uses the integer ``band_idx`` array as the independent variable.
4. Returns a validated :class:`~triceratops.data.core.InferenceData` object.

Before converting, you can inspect the model's registered bands:

.. code-block:: python

    print(model.bundle.filter_names)   # e.g., ['g', 'r', 'i', 'z']

If a band name in your container is not found in the model's filter bundle,
a :exc:`KeyError` is raised with a diagnostic message.

Optional parameters:

.. code-block:: python

    inference_data = epoch.to_inference_data(
        model,
        infer_errors=True,          # infer 1σ errors from upper limits
        detection_threshold=3.0,    # N-sigma assumed for upper limits
        mask=some_boolean_array,    # select a subset of rows
    )

After conversion, inspect the result:

.. code-block:: python

    print(inference_data.describe())

See :ref:`data_to_inference` for the full pipeline walkthrough.


----

API Reference
-------------

.. autoclass:: triceratops.data.optical_photometry.OpticalPhotometryEpoch
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
