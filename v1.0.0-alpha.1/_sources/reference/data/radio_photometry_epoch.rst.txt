.. _radio_photometry_epoch:

=======================
RadioPhotometryEpoch
=======================

:class:`~trilobite.data.photometry.RadioPhotometryEpoch` represents a
**single-epoch radio spectral energy distribution (SED)**. It stores
multi-frequency flux density measurements taken at one snapshot in time,
making it suitable for broadband spectral fitting where frequency is the
independent variable.

This is the spectral complement to
:class:`~trilobite.data.light_curve.RadioLightCurveContainer` (which
fixes frequency and varies time). In a ``RadioPhotometryEpoch``, time is
fixed metadata (implicitly a single epoch) and frequency is the
x-axis variable.

.. note::

    This class was previously named ``RadioPhotometryEpochContainer``.
    The old name is still importable but will emit a :exc:`DeprecationWarning`.
    Update your code to use ``RadioPhotometryEpoch``.


Schema
------

.. list-table::
    :header-rows: 1
    :widths: 25 45 15 15

    * - Column Name
      - Description
      - Default Unit
      - Required
    * - ``freq``
      - Central observing frequency
      - ``GHz``
      - Yes
    * - ``flux_density``
      - Measured flux density (detections; NaN for upper limits)
      - ``Jy``
      - Yes
    * - ``flux_density_error``
      - 1σ uncertainty on flux density
      - ``Jy``
      - Yes
    * - ``flux_upper_limit``
      - Upper limit for non-detections (NaN for detections)
      - ``Jy``
      - Yes
    * - ``obs_name``
      - Optional observation identifier
      - None
      - No
    * - ``comments``
      - Optional free-form metadata
      - None
      - No

**Detection convention**: An observation is a detection if ``flux_upper_limit``
is NaN. It is a non-detection (upper limit) if ``flux_upper_limit`` is finite.


Construction
------------

From an Astropy Table
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    from astropy.table import Table
    from astropy import units as u
    from trilobite.data import RadioPhotometryEpoch

    table = Table({
        "freq": [1.4, 5.5, 8.5, 15.0, 22.0] * u.GHz,
        "flux_density": [1.2, 1.8, 1.6, np.nan, np.nan] * u.Jy,
        "flux_density_error": [0.1, 0.1, 0.1, np.nan, np.nan] * u.Jy,
        "flux_upper_limit": [np.nan, np.nan, np.nan, 0.4, 0.6] * u.Jy,
    })

    epoch = RadioPhotometryEpoch(table)

A :meth:`from_table` classmethod with column renaming is also available:

.. code-block:: python

    epoch = RadioPhotometryEpoch.from_table(table, column_map={
        "frequency": "freq",
        "flux":      "flux_density",
    })

From a File
^^^^^^^^^^^

.. code-block:: python

    epoch = RadioPhotometryEpoch.from_file("sed_epoch1.fits")

Any format supported by :meth:`astropy.table.Table.read` is accepted.


Data Access
-----------

.. code-block:: python

    epoch.freq                    # astropy.Quantity in GHz
    epoch.flux_density            # astropy.Quantity in Jy
    epoch.flux_density_error      # astropy.Quantity in Jy
    epoch.flux_upper_limit        # astropy.Quantity in Jy

    epoch.n_obs                   # total number of rows
    epoch.n_detections            # rows with finite flux_density
    epoch.n_non_detections        # rows with finite flux_upper_limit

    epoch.detection_mask          # boolean array, True = detection
    epoch.non_detection_mask      # boolean array, True = upper limit
    epoch.detection_table         # sub-table of detections only
    epoch.non_detection_table     # sub-table of upper limits only

    epoch.apply_mask(mask)        # returns a new RadioPhotometryEpoch


Inference Integration
---------------------

.. code-block:: python

    inference_data = epoch.to_inference_data(model)

This call maps the ``freq`` column to the model variable automatically.
Both ``"freq"`` and ``"frequency"`` are recognized as the container's
frequency column name; the actual model variable name must be one of these
two strings.

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


Deprecated Alias
----------------

For backward compatibility, ``RadioPhotometryEpochContainer`` is available
as a deprecated alias:

.. code-block:: python

    # Still works, but emits DeprecationWarning
    from trilobite.data import RadioPhotometryEpochContainer

    epoch = RadioPhotometryEpochContainer(table)
    # DeprecationWarning: RadioPhotometryEpochContainer is deprecated.
    # Use RadioPhotometryEpoch instead.

Update your code to import ``RadioPhotometryEpoch`` directly.


----

API Reference
-------------

.. autoclass:: trilobite.data.photometry.RadioPhotometryEpoch
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
