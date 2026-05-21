.. _filter_photometry:

============================
Optical Filters and Photometry
============================

Trilobite includes a complete system for filter-based optical photometry. Whether you are working
with archival ZTF or LSST observations, or building a multi-band light curve model for inference, this
module provides efficient tools for filter convolution, magnitude conversions, and filter I/O.

The central objects are:

- :class:`~trilobite.utils.phot_utils.PhotometryFilter` — a single bandpass filter
- :class:`~trilobite.utils.phot_utils.FilterBundle` — a collection of filters optimised for MCMC hot loops
- Magnitude conversion utilities: :func:`~trilobite.utils.phot_utils.flux_to_ab_mag`,
  :func:`~trilobite.utils.phot_utils.ab_mag_to_flux`, etc.

----

PhotometryFilter
----------------

A :class:`~trilobite.utils.phot_utils.PhotometryFilter` wraps a filter transmission curve and
precomputes integration weights so that filter convolution is a single dot product at evaluation time.

Construction
^^^^^^^^^^^^

Filters accept wavelength as either a bare NumPy array **in cm** or as an
:class:`astropy.units.Quantity` in any compatible unit (Å, nm, μm, …). Transmission values are
dimensionless.

.. code-block:: python

    import numpy as np
    import astropy.units as u
    from trilobite.utils.phot_utils import PhotometryFilter

    # From an Astropy Quantity (recommended)
    lam = np.linspace(5500, 7500, 300) * u.AA
    T   = np.exp(-0.5 * ((lam.value - 6500) / 500) ** 2)
    r_filt = PhotometryFilter(lam, T, name="r-band", weighting="photon")

    # From a bare array already in cm
    lam_cm = np.linspace(5.5e-5, 7.5e-5, 300)   # cm
    r_filt = PhotometryFilter(lam_cm, T, name="r-band")

The ``weighting`` keyword controls how the detector samples incoming photons:

- ``"photon"`` (default) — appropriate for CCDs and most photon-counting detectors
- ``"energy"`` — appropriate for bolometers and energy-integrating detectors

Useful Properties
^^^^^^^^^^^^^^^^^

.. code-block:: python

    print(r_filt)
    # PhotometryFilter('r-band', lam_eff=6408.5 AA, weighting='photon', N=300)

    r_filt.effective_wavelength   # pivot wavelength in cm
    r_filt.effective_frequency    # pivot frequency in Hz
    r_filt.filter_width_lambda    # equivalent width in cm
    r_filt.wavelength_bounds      # (lambda_min, lambda_max) in cm
    r_filt.frequency_bounds       # (nu_min, nu_max) in Hz
    r_filt.weights                # precomputed normalised integration weights

Convolution
^^^^^^^^^^^

There are three convolution entry points, trading off convenience against speed:

.. code-block:: python

    # Case 1: F_nu already on the filter's own grid (fastest — no interpolation)
    F_nu_on_grid = model.evaluate(r_filt.frequency)
    F_band = r_filt.apply(F_nu_on_grid)

    # Case 2: F_nu on an arbitrary frequency grid (interpolates internally)
    nu  = np.logspace(13, 16, 1000)   # Hz
    F_nu = my_sed_model(nu)
    F_band = r_filt.convolve_nu(nu, F_nu)

    # Case 3: F_lambda on a wavelength grid
    lam_cm = np.linspace(4e-5, 1e-4, 1000)
    F_lam  = my_lambda_model(lam_cm)
    F_band = r_filt.convolve_lambda(lam_cm, F_lam)

All convolution methods accept a batch dimension: passing ``F_nu`` with shape ``(N_t, N_nu)``
returns a result of shape ``(N_t,)`` — one band flux per time step.

The filter object is also callable as a shorthand for :meth:`~trilobite.utils.phot_utils.PhotometryFilter.convolve_nu`:

.. code-block:: python

    F_band = r_filt(nu, F_nu)   # equivalent to r_filt.convolve_nu(nu, F_nu)

Visualisation
^^^^^^^^^^^^^

.. code-block:: python

    fig, ax = r_filt.plot(wavelength_unit="AA", label="r-band", color="firebrick")
    ax.set_title("SDSS r-band transmission")

----

FilterBundle
------------

When running MCMC inference over a multi-band dataset, calling individual filters in a loop is
wasteful. :class:`~trilobite.utils.phot_utils.FilterBundle` addresses this by precomputing a
**weight matrix** of shape ``(N_filters, N_common)`` at construction time, reducing all-band
convolution to a single matrix multiply at inference time.

Construction
^^^^^^^^^^^^

.. code-block:: python

    from trilobite.utils.phot_utils import FilterBundle

    bundle = FilterBundle({
        "g": g_filt,
        "r": r_filt,
        "i": i_filt,
    })

    print(bundle)
    # FilterBundle(['g', 'r', 'i'], N_common=600)

The **common frequency grid** is the sorted union of all individual filter grids. The
``weight_matrix`` property exposes the precomputed ``(N_filters, N_common)`` matrix.

MCMC Hot-Loop Pattern
^^^^^^^^^^^^^^^^^^^^^

The recommended workflow for MCMC evaluation is:

1. Build the ``FilterBundle`` once, before the sampler starts.
2. Obtain the :attr:`~trilobite.utils.phot_utils.FilterBundle.frequency_grid` and evaluate
   your model on that grid.
3. Call :meth:`~trilobite.utils.phot_utils.FilterBundle.apply` — a pure matrix multiply with
   no interpolation overhead.

.. code-block:: python

    # Setup (once, before MCMC)
    bundle = FilterBundle({"g": g_filt, "r": r_filt, "i": i_filt})
    nu_eval = bundle.frequency_grid   # shape (N_common,)

    # Inside the likelihood (called millions of times)
    def log_likelihood(theta):
        F_nu = my_model(nu_eval, *theta)       # shape (N_common,)
        F_bands = bundle.apply(F_nu)           # shape (N_filters,)  — one matrix multiply
        mags = flux_to_ab_mag(F_bands)
        return -0.5 * np.sum(((mags - obs_mags) / obs_errors) ** 2)

For batched model evaluation (e.g. ``(N_walkers, N_common)``):

.. code-block:: python

    F_nu_batch = my_model(nu_eval, theta_batch)   # shape (N_walkers, N_common)
    F_bands    = bundle.apply(F_nu_batch)          # shape (N_walkers, N_filters)

If your model is not already on the common grid, use
:meth:`~trilobite.utils.phot_utils.FilterBundle.convolve_nu` which adds one interpolation step:

.. code-block:: python

    F_bands = bundle.convolve_nu(nu_arb, F_nu_arb)

Dict-like Interface
^^^^^^^^^^^^^^^^^^^

:class:`~trilobite.utils.phot_utils.FilterBundle` supports a dictionary-like interface:

.. code-block:: python

    bundle["r"]                          # retrieve a filter by name
    "g" in bundle                        # membership test
    for name in bundle: ...              # iteration over filter names
    len(bundle)                          # number of filters

Mutation
^^^^^^^^

Filters can be added or removed at any time; the common grid and weight matrix are automatically
rebuilt:

.. code-block:: python

    z_filt = PhotometryFilter(...)
    bundle.add_filter("z", z_filt)

    old_filt = bundle.remove_filter("i")   # returns the removed filter

Visualisation
^^^^^^^^^^^^^

.. code-block:: python

    fig, ax = bundle.plot(wavelength_unit="AA")

Each filter is drawn in a distinct colour sampled from the default colourmap and a legend is
added automatically (pass ``legend=False`` to suppress it).

----

Loading Filters
---------------

From a File
^^^^^^^^^^^

Two-column ASCII files (wavelength, transmission) can be loaded directly.
By default the wavelength column is assumed to be in Ångström:

.. code-block:: python

    from trilobite.utils.phot_utils import load_filter_from_file

    filt = load_filter_from_file("sdss_r.dat")                       # Å default
    filt = load_filter_from_file("sdss_r.dat", wavelength_unit="nm") # nm

The filter name defaults to the file stem.

From speclite
^^^^^^^^^^^^^

`speclite <https://speclite.readthedocs.io>`_ provides a large registry of standard photometric
filters (SDSS, ZTF, LSST, DECam, HST WFC3, Bessell, Gaia, and more). Install the optional
dependency with:

.. code-block:: bash

    pip install trilobite[optical]

Then load any filter by its speclite name:

.. code-block:: python

    from trilobite.utils.phot_utils import load_filter_from_speclite, list_speclite_filters

    # List all available filters (or a subset by prefix)
    all_filters  = list_speclite_filters()
    sdss_filters = list_speclite_filters(group="sdss")

    # Load by name
    r_sdss = load_filter_from_speclite("sdss-r")
    r_ztf  = load_filter_from_speclite("ztf-r")
    r_lsst = load_filter_from_speclite("lsst2016-r")

Common filter name prefixes include: ``sdss-``, ``ztf-``, ``lsst2016-``, ``decam-``,
``hst_wfc3_ir-``, ``bessell-``, ``gaia2r-``.

----

Magnitude Systems
-----------------

AB Magnitudes
^^^^^^^^^^^^^

The **AB magnitude** system defines magnitude zero as a constant spectrum with
``F_ν = 3.631 × 10⁻²⁰ erg/s/cm²/Hz``:

.. math::

    m_{\rm AB} = -2.5 \log_{10}\!\left(\frac{F_\nu}{3.631 \times 10^{-20}
    \;{\rm erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}}\right)

.. code-block:: python

    from trilobite.utils.phot_utils import flux_to_ab_mag, ab_mag_to_flux

    mag = flux_to_ab_mag(3.631e-20)   # → 0.0
    F   = ab_mag_to_flux(0.0)         # → 3.631e-20

Both functions operate on scalars or arrays and handle non-positive fluxes
gracefully (returning ``np.inf``).

ST Magnitudes
^^^^^^^^^^^^^

The **ST magnitude** system is defined in wavelength space:

.. math::

    m_{\rm ST} = -2.5 \log_{10}(F_\lambda) - 21.1

where ``F_λ`` is in erg/s/cm²/Å.

.. code-block:: python

    from trilobite.utils.phot_utils import flux_lambda_to_st_mag, st_mag_to_flux_lambda

    mag = flux_lambda_to_st_mag(3.631e-9)      # erg/s/cm²/Å
    F_lambda = st_mag_to_flux_lambda(mag)

Filter-Convolved Magnitudes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~trilobite.utils.phot_utils.filter_to_ab_mag` combines filter convolution and magnitude
conversion into a single call:

.. code-block:: python

    from trilobite.utils.phot_utils import filter_to_ab_mag

    # Single filter
    mag_r = filter_to_ab_mag(r_filt, nu, F_nu)   # scalar

    # FilterBundle → all bands at once
    mags = filter_to_ab_mag(bundle, nu, F_nu)     # shape (N_filters,)

    # Batched (N_t time steps)
    mags_batch = filter_to_ab_mag(bundle, nu, F_nu_batch)  # shape (N_t, N_filters)

----

I/O Reference
-------------

:class:`~trilobite.utils.phot_utils.PhotometryFilter` supports the following I/O formats:

.. list-table::
    :header-rows: 1
    :widths: 20 35 25 20

    * - Format
      - Write
      - Read (classmethod)
      - Notes
    * - Dictionary
      - ``to_dict()``
      - ``from_dict(d)``
      - Pure Python; JSON-serialisable
    * - JSON file
      - ``to_json(path)``
      - ``from_json(path)``
      - Human-readable; portable
    * - HDF5 file
      - ``to_hdf5(path, key="filter")``
      - ``from_hdf5(path, key="filter")``
      - Binary; supports multiple filters per file via ``key``
    * - Astropy Table
      - ``to_table()``
      - ``from_table(table)``
      - Integrates with Astropy ecosystem (FITS, CSV, …)
    * - NumPy array
      - ``to_array()``
      - ``from_array(arr)``
      - Shape ``(2, N)``; row 0 = wavelength (cm), row 1 = transmission

:class:`~trilobite.utils.phot_utils.FilterBundle` provides the same formats at the bundle level:
``to_dict`` / ``from_dict``, ``to_json`` / ``from_json``, and ``to_hdf5`` / ``from_hdf5``. In the
HDF5 file each filter occupies a separate group and the original filter order is preserved.

.. code-block:: python

    # Save and reload a bundle
    bundle.to_hdf5("my_filters.h5")
    bundle2 = FilterBundle.from_hdf5("my_filters.h5")

    # Or via JSON (useful for version control)
    bundle.to_json("my_filters.json")
    bundle2 = FilterBundle.from_json("my_filters.json")
