.. _light_curves:

============================
Light Curves
============================

Trilobite provides two light curve container types:

- :class:`~trilobite.data.light_curve.RadioLightCurveContainer` — single-frequency radio time-series (:math:`F_\nu(t)`)
- :class:`~trilobite.data.light_curve.OpticalLightCurveContainer` — single-band optical time-series (:math:`F_\nu(t)` or :math:`m_\mathrm{AB}(t)`)

Both containers wrap an :class:`astropy.table.Table` with an enforced schema,
providing a validated, unit-aware, and immutable interface for downstream
modeling and inference. Both expose a :meth:`to_inference_data` method for
seamless integration with the Trilobite inference pipeline.


.. _radio_light_curve:

RadioLightCurveContainer
-------------------------

The radio light curve container is designed for **single-frequency radio
time-series data**, where all measurements correspond to a fixed observing
frequency:

.. math::

    F_\nu(t)

The observing frequency is stored as **metadata** (not a column) — it is a
fixed property of the dataset, not a per-observation variable.


Schema
^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 45 15 15

    * - Column Name
      - Description
      - Default Unit
      - Required
    * - ``time``
      - Observation time (relative to some reference)
      - ``day``
      - Yes
    * - ``flux_density``
      - Measured flux density (detections only; NaN for upper limits)
      - ``Jy``
      - Yes
    * - ``flux_density_error``
      - 1σ uncertainty on flux density
      - ``Jy``
      - Yes
    * - ``flux_upper_limit``
      - Upper limit on flux density (non-detections only; NaN for detections)
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
^^^^^^^^^^^^

From an Astropy Table:

.. code-block:: python

    import numpy as np
    from astropy.table import Table
    from astropy import units as u
    from trilobite.data import RadioLightCurveContainer

    table = Table({
        "time": [0, 10, 30, 100] * u.day,
        "flux_density": [1.2, 1.5, 1.1, np.nan] * u.Jy,
        "flux_density_error": [0.1, 0.1, 0.1, np.nan] * u.Jy,
        "flux_upper_limit": [np.nan, np.nan, np.nan, 0.3] * u.Jy,
    })

    lc = RadioLightCurveContainer(table, frequency=8.5 * u.GHz)

From a file (any format supported by :meth:`astropy.table.Table.read`):

.. code-block:: python

    lc = RadioLightCurveContainer.from_file("lightcurve.csv", frequency=8.5)

If the frequency is passed as a plain float, GHz is assumed.

To rename columns on loading:

.. code-block:: python

    lc = RadioLightCurveContainer.from_table(
        table,
        frequency=8.5 * u.GHz,
        column_map={
            "t": "time",
            "flux": "flux_density",
        },
    )


Data Access
^^^^^^^^^^^

Unit-aware accessors:

.. code-block:: python

    lc.time                  # astropy.Quantity in day
    lc.flux_density          # astropy.Quantity in Jy
    lc.flux_density_error    # astropy.Quantity in Jy
    lc.flux_upper_limit      # astropy.Quantity in Jy
    lc.frequency             # astropy.Quantity in GHz

Detection utilities:

.. code-block:: python

    lc.detection_mask        # boolean array, True = detection
    lc.non_detection_mask    # boolean array, True = upper limit
    lc.n_detections
    lc.n_non_detections
    lc.detection_table       # sub-table of detections only
    lc.non_detection_table   # sub-table of upper limits only


Inference Integration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    inference_data = lc.to_inference_data(model)

The container maps ``time`` to the model variable automatically. The observing
frequency is **not** included in the independent variable array — it is fixed
metadata and is not passed to the model as a variable.

Optional parameters:

.. code-block:: python

    inference_data = lc.to_inference_data(
        model,
        infer_errors=True,          # infer 1σ from upper limits
        detection_threshold=3.0,    # N-sigma assumed for upper limits
        mask=some_boolean_array,    # select a subset of rows
    )

See :ref:`data_to_inference` for the full pipeline walkthrough.


.. _optical_light_curve:

OpticalLightCurveContainer
---------------------------

The optical light curve container handles **single-band optical time-series**
data. Unlike the radio container, observations may be supplied as flux
densities, AB magnitudes, or both. The band name is stored as metadata and
resolved to a model band index at inference time.

.. math::

    F_\nu(t) \quad \text{or} \quad m_\mathrm{AB}(t)


Schema
^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 25 45 15 15

    * - Column Name
      - Description
      - Default Unit
      - Required
    * - ``time``
      - Observation time (relative to some reference)
      - ``day``
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
    * - ``comments``
      - Optional free-form metadata
      - None
      - No

.. note::

    At least one of the flux or magnitude column groups must be present.
    If only magnitudes are provided, flux is computed on-the-fly via
    the AB magnitude conversion formula.


Construction
^^^^^^^^^^^^

The ``band`` keyword argument is required:

.. code-block:: python

    from trilobite.data import OpticalLightCurveContainer

    # From flux density columns
    lc = OpticalLightCurveContainer(table, band="g")

    # From a file
    lc = OpticalLightCurveContainer.from_file("optical_g.fits", band="g")

    # From a table with column renaming
    lc = OpticalLightCurveContainer.from_table(table, band="g", column_map={
        "t_day": "time",
        "f_nu":  "flux_density",
    })


Dual Flux/Magnitude Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All four representations are always accessible regardless of the input format:

.. code-block:: python

    # Flux density (erg/s/cm^2/Hz)
    lc.flux           # flux density, with magnitude→flux conversion if needed
    lc.flux_error
    lc.flux_upper_limit

    # AB magnitudes
    lc.mag            # AB magnitude, with flux→magnitude conversion if needed
    lc.mag_error
    lc.mag_upper_limit

If only flux columns were supplied, magnitude properties are computed on the fly.
If only magnitude columns were supplied, flux properties are computed on the fly.

The band name is accessible as:

.. code-block:: python

    lc.band   # e.g., "g"


Detection Semantics
^^^^^^^^^^^^^^^^^^^^

Detection status is derived from the ``flux_upper_limit`` column:

.. code-block:: python

    lc.detection_mask        # True = detection
    lc.non_detection_mask    # True = upper limit (non-detection)
    lc.n_detections
    lc.n_non_detections
    lc.detection_table
    lc.non_detection_table


Inference Integration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    inference_data = lc.to_inference_data(model)

This call:

1. Resolves ``lc.band`` → an integer band index via ``model.bundle.filter_names``.
2. Passes ``time`` as the temporal independent variable.
3. Broadcasts the band index as a constant array (same value for every row).
4. Returns a validated :class:`~trilobite.data.core.InferenceData` object.

.. code-block:: python

    # Inspect the model's registered filter names before converting:
    print(model.bundle.filter_names)   # e.g., ['g', 'r', 'i', 'z']

    inference_data = lc.to_inference_data(
        model,
        infer_errors=True,
        mask=some_boolean_array,
    )

If the band name stored in the container is not found in
``model.bundle.filter_names``, a :exc:`KeyError` is raised with a clear
diagnostic message.

See :ref:`data_to_inference` for the full pipeline walkthrough.


----

API Reference
-------------

.. autoclass:: trilobite.data.light_curve.RadioLightCurveContainer
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

----

.. autoclass:: trilobite.data.light_curve.OpticalLightCurveContainer
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
