.. _data_to_inference:

=====================================
From Data to Inference: A Complete Guide
=====================================

This guide walks through the full chain from raw observational data to a
sampler-ready inference problem. It is the single document you need if you
are setting up a new inference analysis in Trilobite.

The pipeline has four steps, each with a single responsibility:

1. **Load your data into a container** — schema validation, unit enforcement, column semantics.
2. **Understand what your model expects** — match variables and observables.
3. **Call** ``to_inference_data(model)`` — produce a validated numerical dataset.
4. **Wire into likelihood and InferenceProblem** — specify statistical assumptions and priors.


Step 1: Load Your Data into a Container
---------------------------------------

Every piece of observational data in Trilobite is loaded into a
:class:`~trilobite.data.core.DataContainer` subclass. The container enforces
a column schema, validates units, and provides domain-specific accessors.

Choose the container that matches your data:

.. tab-set::

    .. tab-item:: Radio light curve

        .. code-block:: python

            from trilobite.data import RadioLightCurveContainer

            container = RadioLightCurveContainer.from_file(
                "lightcurve.fits",
                frequency=5.5,   # GHz
            )

        The frequency is metadata — it is stored on the container, not as a
        per-row column. Use this for a single-frequency time series.

    .. tab-item:: Radio photometry (multi-epoch)

        .. code-block:: python

            from trilobite.data import RadioPhotometryContainer

            container = RadioPhotometryContainer.from_file("photometry.fits")

        Use this for heterogeneous multi-frequency, multi-epoch radio data.

    .. tab-item:: Optical photometry (multi-epoch)

        .. code-block:: python

            from trilobite.data import OpticalPhotometryContainer

            container = OpticalPhotometryContainer.from_file("optical.fits")

        Band names (e.g. ``"g"``, ``"r"``) are stored in a ``band_name`` column
        and resolved to model band indices at inference time.

    .. tab-item:: Optical light curve

        .. code-block:: python

            from trilobite.data import OpticalLightCurveContainer

            container = OpticalLightCurveContainer.from_file(
                "optical_g.fits",
                band="g",
            )

        The band is metadata. Data may be supplied in AB magnitudes,
        flux densities, or both.

    .. tab-item:: Single-epoch SED (radio)

        .. code-block:: python

            from trilobite.data import RadioPhotometryEpoch

            container = RadioPhotometryEpoch.from_file("epoch_sed.fits")

        Frequency is the independent variable. Use this for SED fitting at
        a single snapshot in time.

    .. tab-item:: Single-epoch SED (optical)

        .. code-block:: python

            from trilobite.data import OpticalPhotometryEpoch

            container = OpticalPhotometryEpoch.from_file("optical_epoch.fits")

        Band is the independent variable. Use this for broadband optical
        SED fitting at a single epoch.

Once loaded, you can inspect your data:

.. code-block:: python

    print(container.n_obs)           # total number of rows
    print(container.n_detections)    # rows with finite flux
    print(container.n_non_detections)  # rows with upper limits only


Step 2: Understand What Your Model Expects
------------------------------------------

Every model in Trilobite declares:

- **VARIABLES** — the independent variables it accepts as input (e.g. time, frequency).
- **PARAMETERS** — the physical parameters that are fitted during inference.
- **OUTPUTS** (observables) — what the model predicts (e.g. flux density).

You can inspect these declarations at runtime:

.. code-block:: python

    print(model.variable_names)    # e.g. ['time', 'freq']
    print(model.output_names)      # e.g. ['flux_density']

The :meth:`to_inference_data` method uses these declarations to validate and
coerce the container's data. If your container's column names don't match what
the model expects, you can provide an explicit mapping (see Step 3).

For optical models, the model also exposes a
:class:`~trilobite.utils.phot_utils.FilterBundle`:

.. code-block:: python

    print(model.bundle.filter_names)   # e.g. ['g', 'r', 'i', 'z']

Band names in the container are looked up in this list at inference time. The
integer index (position in the list) becomes the ``band_idx`` variable passed
to the model.


Step 3: Call to_inference_data()
---------------------------------

The central step. Every container provides this method:

.. code-block:: python

    inference_data = container.to_inference_data(model)

This single call performs:

- Mapping container columns to model variable names.
- Unit coercion (e.g. time from days to seconds, flux from Jy to CGS).
- Band-name resolution for optical containers.
- Magnitude-to-flux conversion where needed.
- Optional error inference for non-detections (from upper limits).
- Shape validation of all resulting arrays.

**Always inspect the result before proceeding:**

.. code-block:: python

    inference_data.describe()

This prints a summary like:

.. code-block:: text

    InferenceData — 32 observations
    ──────────────────────────────────────────────────
    Independent Variables
      time        : min=0.10  max=1200.00  (shape=(32,))
      freq        : min=1.40  max=22.00    (shape=(32,))
    ──────────────────────────────────────────────────
    Observables
      flux_density
        detections    : 28
        upper limits  : 4
        lower limits  : 0
        value range   : 1.23e-28 … 8.45e-26
        error present : True

Verify that the number of detections and upper limits match your expectations.

**Optional parameters:**

.. code-block:: python

    inference_data = container.to_inference_data(
        model,
        infer_errors=True,         # infer 1-sigma errors from upper limits
        detection_threshold=3.0,   # N-sigma assumed for upper limits
        mask=some_boolean_array,   # subset of rows to include
    )

**Explicit variable/observable mapping** (advanced):

If the model's variable names don't match the container's column names, you can
provide an explicit mapping:

.. code-block:: python

    inference_data = container.to_inference_data(
        model,
        variables={"frequency": "freq"},   # model name → container column
        observables={
            "flux_density": (
                "flux_density",         # value column
                "flux_density_error",   # error column
                "flux_upper_limit",     # upper limit column
                None,                   # lower limit column (None if absent)
            )
        },
    )


Step 4: Wire into Likelihood and InferenceProblem
--------------------------------------------------

With a valid :class:`~trilobite.data.core.InferenceData` in hand, the
remainder of the pipeline is straightforward:

.. code-block:: python

    from trilobite.inference.likelihood import GaussianLikelihood
    from trilobite.inference.problem import InferenceProblem
    from astropy import units as u

    # Build the likelihood — binds model, data, and noise assumption
    likelihood = GaussianLikelihood(model=model, data=inference_data)

    # Create the inference problem — manages parameters and priors
    problem = InferenceProblem(likelihood)

    # Set priors on all free parameters
    problem.set_prior("B",          "log_uniform", lower=0.01 * u.G,   upper=100 * u.G)
    problem.set_prior("n0",         "log_uniform", lower=0.001 / u.cm**3, upper=100 / u.cm**3)
    problem.set_prior("epsilon_e",  "uniform",     lower=0.0,           upper=1.0)
    problem.set_prior("epsilon_B",  "uniform",     lower=0.0,           upper=1.0)

    # Verify the problem is well-posed
    print(problem.initial_log_posterior)   # should be finite

Once the inference problem is configured, pass it to a sampler:

.. code-block:: python

    from trilobite.inference.sampling import EmceeSampler

    sampler = EmceeSampler(problem)
    result = sampler.run(n_walkers=32, n_steps=2000)


Common Errors and How to Fix Them
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Error
     - Cause
     - Fix
   * - ``ValueError: Model declares variables ['frequency'] but x has keys ['freq', 'time']``
     - Container's default mapping uses ``'freq'`` but the model expects ``'frequency'``
     - Pass ``variables={"frequency": "freq"}`` to ``to_inference_data()``
   * - ``AttributeError: Model '...' does not have a 'bundle' attribute``
     - Optical container used with a non-optical model
     - Ensure your model exposes a :class:`~trilobite.utils.phot_utils.FilterBundle` via ``model.bundle``
   * - ``KeyError: Band name 'z' is not in the model's FilterBundle``
     - A band name in the container is not registered in the model's filter bundle
     - Check ``model.bundle.filter_names``; ensure band names match exactly
   * - ``ValueError: Mask must have the same length as the number of observations``
     - Boolean mask has wrong shape
     - Check ``mask.shape == (container.n_obs,)``
   * - ``ValueError: ... shapes are inconsistent``
     - Arrays in ``x`` or ``y`` have different lengths
     - Ensure all arrays share the same shape before constructing InferenceData manually
   * - ``Warning: flux_density_error contains NaN values``
     - Non-detections lack explicit error estimates
     - Use ``infer_errors=True`` to infer errors from upper limits


Building InferenceData Manually (Advanced)
-------------------------------------------

For synthetic data, testing, or containers not yet covered by Trilobite,
you can build :class:`~trilobite.data.core.InferenceData` directly from arrays:

.. code-block:: python

    import numpy as np
    from astropy import units as u
    from trilobite.data import InferenceData

    inference_data = InferenceData.from_arrays(
        model=model,
        x={
            "time": np.linspace(1, 1000, 50) * u.day,
            "freq": np.full(50, 5.5) * u.GHz,
        },
        y={
            "flux_density": np.random.lognormal(-60, 0.5, 50) * u.Unit("erg/(s cm2 Hz)"),
        },
        y_err={
            "flux_density": np.random.uniform(1e-29, 1e-28, 50) * u.Unit("erg/(s cm2 Hz)"),
        },
    )

The keys in ``x`` must exactly match ``model.variable_names``. The keys in
``y`` must exactly match ``model.output_names``. Units are coerced automatically
if :class:`astropy.units.Quantity` objects are provided.

For the full API reference, see :ref:`inference_data`.
