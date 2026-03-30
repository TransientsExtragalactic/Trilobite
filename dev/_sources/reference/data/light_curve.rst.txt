.. _light_curves:

============================
Light Curves
============================

Light curve data in Triceratops is represented by the
:class:`~triceratops.data.light_curve.RadioLightCurveContainer`
class. This container is designed for **single-band radio time-series data**,
where all measurements correspond to a fixed observing frequency.

Unlike the photometry container, which may store heterogeneous
multi-frequency observations, the light curve container assumes that
all data belong to a single band and are intended to be modeled as
a single temporal sequence.

The container wraps an :class:`astropy.table.Table` with an enforced
schema, providing a validated, unit-aware, and immutable interface
for downstream modeling and inference.


Conceptual Overview
-------------------

A radio light curve represents flux density measurements taken
at a fixed observing frequency as a function of time:

.. math::

    F_\nu(t)

where the observing frequency :math:`\nu` is constant across all rows.

In this container:

- Time is stored explicitly as a table column.
- Flux densities and uncertainties are stored as table columns.
- The observing frequency is stored as metadata, not as a column.
- Detection status is inferred from the ``flux_upper_limit`` column.

This structure allows clean integration with inference pipelines
while preserving the clarity and simplicity of a time-series dataset.


The Light Curve Schema
----------------------

The underlying :class:`astropy.table.Table` must satisfy a strict schema.
The required and optional columns are defined in the class-level
``COLUMNS`` attribute :contentReference[oaicite:1]{index=1}.

The required columns are:

.. list-table::
    :header-rows: 1
    :widths: 25 45 15 15

    * - Column Name
      - Description
      - Default Unit
      - Required
    * - ``time``
      - Observation time (relative)
      - ``day``
      - Yes
    * - ``flux_density``
      - Measured flux density (detections)
      - ``Jy``
      - Yes
    * - ``flux_density_error``
      - 1σ uncertainty on flux density
      - ``Jy``
      - Yes
    * - ``flux_upper_limit``
      - Upper limit for non-detections
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

A few important details follow directly from the implementation:

- The container always expects ``flux_upper_limit`` to be present.
- Detection status is inferred from whether the upper-limit entry is finite.
- Units must be convertible to the declared units (e.g., days → seconds, Jy → CGS).


Detection Semantics
-------------------

Detection and non-detection status is derived automatically.

An observation is considered:

- A **detection** if the upper limit is NaN.
- A **non-detection** (upper limit) if the upper-limit value is finite.

The following properties are provided:

.. code-block:: python

    light_curve.detection_mask
    light_curve.non_detection_mask
    light_curve.n_detections
    light_curve.n_non_detections

Tables containing only detections or only upper limits can be
accessed via:

.. code-block:: python

    light_curve.detection_table
    light_curve.non_detection_table


Observing Frequency
-------------------

The observing frequency is not stored as a table column.
Instead, it is stored as metadata on the container instance.

It is provided during construction:

.. code-block:: python

    from triceratops.data.light_curve import RadioLightCurveContainer
    from astropy.table import Table
    import astropy.units as u

    table = Table({
        "time": [0, 1, 2],
        "flux_density": [1.2, 1.5, 1.1],
        "flux_density_error": [0.1, 0.1, 0.1],
        "flux_upper_limit": [np.nan, np.nan, np.nan],
    })

    lc = RadioLightCurveContainer(
        table,
        frequency=8.5 * u.GHz,
    )

The frequency is internally stored in GHz and exposed as:

.. code-block:: python

    lc.frequency

If a unitless float is provided, GHz is assumed.


Creating a Light Curve Container
--------------------------------

From an Astropy Table
^^^^^^^^^^^^^^^^^^^^^

The simplest construction path is directly from an
:class:`astropy.table.Table`:

.. code-block:: python

    lc = RadioLightCurveContainer(
        table,
        frequency=8.5 * u.GHz,
    )

A convenience constructor allows column renaming:

.. code-block:: python

    lc = RadioLightCurveContainer.from_table(
        table,
        frequency=8.5 * u.GHz,
        column_map={
            "t": "time",
            "flux": "flux_density",
            "flux_err": "flux_density_error",
            "flux_ul": "flux_upper_limit",
        },
    )

From a File
^^^^^^^^^^^

Files readable by :meth:`astropy.table.Table.read` may be loaded directly:

.. code-block:: python

    lc = RadioLightCurveContainer.from_file(
        "lightcurve.csv",
        frequency=8.5,
    )

If the frequency is passed as a float, GHz is assumed.


Accessing Data
--------------

The container provides unit-aware accessors:

.. code-block:: python

    lc.time
    lc.flux_density
    lc.flux_density_error
    lc.flux_upper_limit

Each of these returns an :class:`astropy.units.Quantity`.

Standard table indexing also works:

.. code-block:: python

    lc[0]
    lc[:5]
    lc["flux_density"]


Numerical Backend Representation
--------------------------------

For low-level numerical workflows, the container provides
:meth:`to_cgs_array` :contentReference[oaicite:2]{index=2}.

This returns a dense NumPy array of shape:

.. math::

    (N, 4)

containing:

.. math::

    [t, F_\nu, \sigma, F_{\nu,\mathrm{ul}}]

all converted to CGS units:

- Time → seconds
- Flux → erg s⁻¹ cm⁻² Hz⁻¹
- Error → erg s⁻¹ cm⁻² Hz⁻¹
- Upper limit → erg s⁻¹ cm⁻² Hz⁻¹

Example:

.. code-block:: python

    array = lc.to_cgs_array()

This method is particularly useful when interfacing with
external numerical solvers or compiled backends.


Immutability and Design Philosophy
-----------------------------------

The light curve container is immutable after construction.
The underlying table cannot be modified in-place.

This design ensures:

- Reproducibility during inference
- Stable behavior when passed to likelihood objects
- Clear separation between raw data preparation and modeling

If modifications are required, users should modify the original
:class:`astropy.table.Table` and construct a new container.

The result is a compact, well-defined abstraction for single-band
radio time-series data that integrates seamlessly with the
Triceratops inference framework.
