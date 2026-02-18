.. _inference_data:
================================
Inference Data Containers
================================

The :class:`~triceratops.data.core.InferenceData` class defines the numerical
interface between the :mod:`data` layer and the :mod:`inference` layer.

It is the only data object recognized by likelihood classes.

While :class:`DataContainer` subclasses are responsible for interpreting
tables, resolving column names, and coercing units, an
:class:`InferenceData` instance contains only validated NumPy arrays that are
guaranteed to be consistent with a specific model. Once constructed, it is
immutable, unit-consistent, and suitable for direct statistical evaluation.

Inference Data Overview
-----------------------

Every model declares a set of independent variables and a set of
observables, along with the base units expected for each quantity.
The role of :class:`InferenceData` is to take real data and transform it into
a structure that matches this declaration exactly.

In practical terms, the transformation is conceptual rather than
statistical. The object converts

``(variable_1, variable_2, ...)`` into a structured mapping of independent
variables, and converts ``(observable_1, observable_2, ...)`` into a
collection of observable arrays, optionally including uncertainties and
one-sided limits.

No likelihood assumptions are made at this stage. The object merely
guarantees that:

* All required variables are present,
* All observables match the model declaration,
* Units are consistent with the model,
* All arrays share identical shape.

Once these guarantees are satisfied, the data are ready to be consumed by a
likelihood object.

Generating Inference Data
-------------------------

An :class:`InferenceData` instance can be constructed in three canonical ways.
The appropriate method depends on how your data are stored and how much
automation you require.

From Data Containers
^^^^^^^^^^^^^^^^^^^^

The most common workflow is to construct inference-ready data directly from
a :class:`DataContainer`. Since data containers already understand their own
schema and units, they can perform the conversion with minimal user input.

.. code-block:: python

    inference_data = container.to_inference_data(model)

Optional column overrides may be supplied if the container’s default
column names do not match the model’s expectations. In typical use,
however, no manual intervention is required.

This approach is recommended for real observational datasets.

From Arrays
^^^^^^^^^^^

For synthetic data, simulations, or programmatic workflows,
:meth:`InferenceData.from_arrays` provides a direct constructor.

.. code-block:: python

    inference_data = InferenceData.from_arrays(
        model=model,
        x={"frequency": freq_array},
        y={"flux_density": flux_array},
        y_err={"flux_density": flux_error_array},
    )

Arrays may be plain NumPy arrays or `astropy.units.Quantity` objects.
If quantities are provided, they are coerced to the model’s declared
base units. Shape validation is enforced automatically.

This method is particularly useful in testing and controlled experiments,
where the data are already in memory and do not originate from a table.

From Astropy Tables
^^^^^^^^^^^^^^^^^^^

When working directly with an :class:`astropy.table.Table`, use
:meth:`InferenceData.from_table`.

.. code-block:: python

    inference_data = InferenceData.from_table(
        model=model,
        table=table,
        variables={"frequency": "freq"},
        observables={
            "flux_density": (
                "flux",
                "flux_error",
                "flux_upper_limit",
                None,
            )
        },
    )

Column specifications may be provided explicitly or allowed to default
to model-declared names. Unit conversion and shape validation are
performed during construction.

This method is appropriate when ingesting external catalogues or survey
products without first wrapping them in a custom data container.

Working With Inference Data
---------------------------

Once constructed, an :class:`InferenceData` object exposes two primary
structures: a mapping of independent variables and a mapping of
observables.

Independent variables are stored as a dictionary mapping variable names
to NumPy arrays. Observables are stored as
:class:`~triceratops.data.core.Observable` objects, each of which contains
a value array and optional arrays for uncertainties and one-sided limits.

All arrays share the same base shape. This consistency is enforced at
construction time and is guaranteed thereafter.

Extracting Numerical Arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Likelihood implementations and numerical backends often require stacked
arrays rather than dictionaries. The following convenience methods provide
deterministic stacking in model-defined order:

.. code-block:: python

    X = data.get_x_array()
    Y = data.get_y_array()
    Y_err = data.get_y_error_array()
    Y_upper = data.get_y_upper_array()
    Y_lower = data.get_y_lower_array()

By default, the returned arrays have shape ``(*base_shape, n_quantities)``.
If ``flatten=True`` is supplied, the arrays are reshaped to
``(n_points, n_quantities)``.

Missing upper or lower limits are represented by ``np.nan``.
If uncertainties are required but absent, the corresponding accessor
raises an exception rather than silently returning incomplete data.

Design Philosophy
-----------------

The inference system is deliberately layered.

The data layer interprets tables and manages units.
The model layer defines physical structure.
The likelihood layer defines statistical assumptions.
The :class:`InferenceData` object sits between them and guarantees that
data entering the statistical layer are already numerically valid.

This separation ensures that likelihood evaluation is a pure numerical
operation with no hidden unit coercion, no column resolution logic,
and no ambiguity in array shape or ordering.
