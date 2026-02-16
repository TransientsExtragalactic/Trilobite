.. _data_overview:
=========================================
Data Loading, Handling, and Visualization
=========================================

The data modules in Triceratops provide tools for loading, processing, and visualizing observational data. Most
importantly, these structures are the entry point to the library, providing a consistent interface for working
with different types of data in our model and inference pipelines.

There are **three core data types** that appear in Triceratops:

1. **Light Curves**: Time series data representing the brightness of an object over time at one or more frequencies.
   Light curves are typically used to study the temporal evolution of radio sources. These are implemented in the
   :mod:`data.light_curve` module.
2. **Spectra**: Frequency-dependent data representing the flux density of an object at a specific time or over a
   range of times. Spectra are used to analyze the frequency characteristics of radio sources. These are implemented in the
   :mod:`data.spectra`.
3. **Photometry Tables**: Tabular data containing measurements of flux densities at various times and frequencies.
   Photometry tables provide a structured way to store and access observational data. These are implemented in the
   :mod:`data.photometry` module.

Each of these data types comes with a set of methods for loading data from common file formats (e.g., CSV, FITS),
processing the data (e.g., filtering, interpolation), and visualizing the results (e.g., plotting light curves and spectra).


Photometric Data
-----------------

Photometry data in Triceratops is handled through the :class:`~data.photometry.RadioPhotometryContainer` class, which
is effectively a wrapper around a standard :class:`astropy.table.Table` object with an enforced schema dictating the
required columns and their meanings. This structure allows for easy loading, manipulation, and access to photometric
data. The photometry container includes methods for common operations such as filtering data by time or frequency,
interpolating missing values, and exporting data to various formats. It is also immediately compatible with
the inference pipelines in Triceratops, allowing users to seamlessly integrate their observational data into
model fitting and analysis workflows.

The Photometry Table
^^^^^^^^^^^^^^^^^^^^

Underlying the photometry container is an :class:`astropy.table.Table` object with a specific schema. This schema breaks
columns into 3 categories:

1. **Required Columns**: These columns must be present in the table for it to be considered valid photometry data.
   They include essential information such as time, frequency, flux density, and measurement uncertainties.
2. **Optional Columns**: These columns provide additional information that can enhance the analysis but are not strictly
   necessary. Examples include upper limits, measurement methods, and observational metadata.
3. **Auxiliary Columns**: These are columns that you, as the user, may wish to include for your own purposes. They are not
   interpreted by Triceratops in any way, but are preserved when saving and loading photometry data.

The set of **required** and **optional** columns are as follows:

.. list-table::
    :header-rows: 1
    :widths: 20 50 15 15

    * - Column Name
      - Description
      - CGS-Equivalent Unit
      - Data Type
    * - ``time``
      - Canonical time of the observation used in analysis.
      - ``s``
      - float
    * - ``freq``
      - Central observing frequency.
      - ``Hz``
      - float
    * - ``flux_density``
      - Measured flux density for detections. Should be ``np.nan`` for non-detections.
      - ``erg s^-1 cm^-2 Hz^-1``
      - float
    * - ``flux_density_error``
      - 1σ uncertainty on ``flux_density``.
      - ``erg s^-1 cm^-2 Hz^-1``
      - float
    * - ``flux_upper_limit``
      - Upper limit on flux density for non-detections. Should be ``np.nan`` for detections.
      - ``erg s^-1 cm^-2 Hz^-1``
      - float
    * - ``obs_time``
      - Total integration time of the observation.
      - ``s``
      - float
    * - ``obs_name``
      - Observation identifier (e.g. telescope + epoch).
      - ``None``
      - str
    * - ``band``
      - Integer band identifier (instrument-specific).
      - ``None``
      - int
    * - ``epoch_id``
      - Integer epoch identifier (user-defined).
      - ``None``
      - int
    * - ``comments``
      - Free-form comments or metadata.
      - ``None``
      - str

.. hint::

    There are a couple of important notes regarding the photometry table schema:

    - **Time** is always a relative measurement with respect to some reference time (e.g., explosion time, trigger time).
      The actual reference time is not stored in the photometry table itself, but should be tracked separately by the user.
    - **Units**: Columns must be *compatible* with the specified CGS-equivalent units, but do not need to be in those exact units.
      For example, frequency can be provided in GHz as long as it can be converted to Hz.
    - **Non-detections**: For non-detections, the ``flux_density`` and ``flux_density_error`` columns should be set to ``np.nan``,
      and the ``flux_upper_limit`` column should contain the upper limit value.

Once the photometry table has been created, it is **immutable**; that is, you cannot add or remove rows or columns directly.
You can, of course, modify the progenitor :class:`astropy.table.Table` before creating the photometry container, or create a new
photometry container from modified data. The reason for the immutability is to ensure data integrity and consistency when using
the photometry container in analysis and modeling.

Creating a Photometry Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create a :class:`~data.photometry.RadioPhotometryContainer`, you can either load data from a file or create it
directly from an :class:`astropy.table.Table` object.

From a Table
~~~~~~~~~~~~

To create a photometry container from an existing :class:`astropy.table.Table`, ensure that the table contains
the required columns as per the schema described above. Then, you can instantiate the container as follows:

.. code-block:: python

    from astropy.table import Table
    from triceratops.data.photometry import RadioPhotometryContainer

    # Create an Astropy Table with the required columns
    data = Table({
        'time': [0.0, 1.0, 2.0],
        'freq': [1e9, 1e9, 1e9],
        'flux_density': [1e-26, 2e-26, 1.5e-26],
        'flux_density_error': [1e-27, 1e-27, 1e-27],
        'flux_upper_limit': [np.nan, np.nan, np.nan],
        'obs_time': [100.0, 100.0, 100.0],
        'obs_name': ['obs1', 'obs2', 'obs3'],
        'band': [1, 1, 1],
        'comments': ['', '', '']
    })

    # Create the RadioPhotometryContainer
    photometry_container = RadioPhotometryContainer(data)

Alternatively, there is the :meth:`~data.photometry.RadioPhotometryContainer.from_table` class method:

.. code-block:: python

    photometry_container = RadioPhotometryContainer.from_table(data)

This method has 2 additional features beyond the standard constructor:

1. It can accept a parameter ``column_map``, which is a dictionary mapping the required column names to
   alternative names in the provided table. This allows you to create a photometry container from a table
   that uses different column names.
2. It can accept a ``time_starts`` parameter, which allows the ``time`` column to be specified in absolute terms
   (e.g., MJD, JD, or Unix time). If provided, the values in the ``time`` column will be converted to relative
   times with respect to the specified reference time.

From a File
~~~~~~~~~~~

To create a photometry container from a file, you can use the :meth:`~data.photometry.RadioPhotometryContainer.from_file`
class method. This method supports loading data from common file formats such as CSV and FITS.
Here is an example of how to load photometry data from a CSV file:

.. code-block:: python

    from triceratops.data.photometry import RadioPhotometryContainer

    # Load photometry data from a CSV file
    photometry_container = RadioPhotometryContainer.from_file('photometry_data.csv')

This is a thin wrapper around the :meth:`astropy.table.Table.read` method, so any file format supported by Astropy
can be used. Similar to the ``from_table`` method, you can also provide a ``column_map`` and ``time_starts`` parameter
to customize the loading process.

Accessing Photometry Data
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the photometry container has been created, you can access the underlying data and perform various operations.
Accessing the data behaves much like the underlying :class:`astropy.table.Table`, with some additional convenience
methods provided by the photometry container.

Indexing and Slicing
~~~~~~~~~~~~~~~~~~~~~~

The indexing and slicing behavior of the photometry container is identical to that of an :class:`astropy.table.Table`.
You can use standard indexing and slicing techniques to access rows and columns of the data. For example:

.. code-block:: python

    # Access the first row
    first_row = photometry_container[0]

    # Access the 'flux_density' column
    flux_densities = photometry_container['flux_density']

    # Slice the first three rows
    first_three_rows = photometry_container[:3]

Special Column Access
~~~~~~~~~~~~~~~~~~~~~~

In addition to the standard indexing methods, the photometry container provides convenience properties for accessing
the required columns directly. For example:

.. code-block:: python

    # Access the time column
    times = photometry_container.time

    # Access the frequency column
    frequencies = photometry_container.freq

    # Access the flux density column
    flux_densities = photometry_container.flux_density

These are returned as :class:`astropy.units.Quantity` objects with the appropriate units.

Detections and Non-Detections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The photometry container provides methods to easily separate detections from non-detections. A non-detection is defined
as an observation where the ``flux_density`` is ``np.nan`` and the ``flux_upper_limit`` is a valid number. Likewise,
a detection is an observation where the ``flux_density`` is a valid number. On initialization, the
:attr:`~data.photometry.RadioPhotometryContainer.detection_mask` and
:attr:`~data.photometry.RadioPhotometryContainer.non_detection_mask` boolean masks are created to
to allow rapid filtering of detections and non-detections. You can use these masks to filter the data as follows:

.. code-block:: python

    # Get all detections
    detections = photometry_container[photometry_container.detection_mask]

    # Get all non-detections
    non_detections = photometry_container[photometry_container.non_detection_mask]

Likewise, convenience properties are provided to directly access the detections and non-detections:

.. code-block:: python

    # Access detections
    detections = photometry_container.detection_table

    # Access non-detections
    non_detections = photometry_container.non_detection_table

and to access the counts of each:

.. code-block:: python

    # Count of detections
    num_detections = photometry_container.n_detections

    # Count of non-detections
    num_non_detections = photometry_container.n_non_detections


Epochs
^^^^^^

Another very useful feature of the photometry container is the ability to group observations into epochs.
An epoch is defined as a set of observations that are considered to be simultaneous or nearly simultaneous.
This is particularly useful for multi-frequency observations taken at the same time.

In some situations, models may require data to contain epochs. The photometry container provides methods
to automatically generate epochs based on time proximity, as well as to manually specify epochs.

Manually Specifying Epochs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To directly specify epochs, simply specify the ``epoch_id`` column in the underlying table before creating
the photometry container. Observations with the same ``epoch_id`` will be grouped into the same epoch. These should
be integers with no particular meaning beyond grouping. The mapping of each epoch to a specific time is determined
by taking the weighted mean of the observation times within that epoch.

Generating Epochs
~~~~~~~~~~~~~~~~~

If your data does not already contain an ``epoch_id`` column, you can use the
:meth:`~data.photometry.RadioPhotometryContainer.set_epochs_from_indices`,
:meth:`~data.photometry.RadioPhotometryContainer.set_epochs_from_time_gaps`,
or :meth:`~data.photometry.RadioPhotometryContainer.set_epochs_from_bins` methods to automatically generate epochs.

The :meth:`~data.photometry.RadioPhotometryContainer.set_epochs_from_indices` method allows you
to directly specify the ``epoch_id`` values for each observation
by providing a list or array of integers.

.. code-block:: python

    # Set epochs using specified indices
    photometry_container.set_epochs_from_indices([0, 0, 1, 1, 2])

The :meth:`~data.photometry.RadioPhotometryContainer.set_epochs_from_time_gaps` method generates epochs based
on gaps in observation times. You can specify
a time gap threshold, and observations separated by a gap larger than this threshold will be assigned to different epochs.

.. code-block:: python

    # Set epochs based on time gaps of 1 day
    photometry_container.set_epochs_from_time_gaps(86400 * u.day)

The :meth:`~data.photometry.RadioPhotometryContainer.set_epochs_from_bins` method allows you to bin observations
into epochs based on fixed time intervals.
You can specify the bin size, and observations falling within the same time bin will be grouped into the same epoch.

.. code-block:: python

    # Set epochs using time bins of 2 days
    bins = np.arrange(0, 10, 2) * u.day
    photometry_container.set_epochs_from_bins(bins)

Light Curves
------------

.. important::

    Not yet implemented.

Spectra
-------

.. important::

    Not yet implemented.
