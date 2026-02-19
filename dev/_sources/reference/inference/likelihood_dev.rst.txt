.. _likelihood_dev:

===============================
Likelihood Development Guide
===============================

Likelihoods are the bridge between physics and inference in Triceratops.
They bind a concrete :class:`~triceratops.models.core.base.Model` to a
:class:`~triceratops.data.core.InferenceData` object under a specific statistical
assumption and produce a scalar log-likelihood suitable for samplers,
optimizers, or evidence estimators.

This guide is written for developers. It does not assume that you merely want
to *use* a likelihood — it assumes you want to *build one correctly*.

The likelihood system is deliberately layered. That layering is not incidental;
it is what keeps statistical mathematics testable, model semantics explicit,
and inference backends fast. Understanding that architecture is the key to
extending the framework safely.

.. contents::
    :local:
    :depth: 2

----

A Conceptual Overview
---------------------

A Triceratops likelihood is not a monolithic object. It consists of two
conceptual layers:

1. A **numerical backend** that encodes pure statistics.
2. A **likelihood class** that binds a forward model to validated numerical data.

The separation between these layers is enforced by
:class:`~triceratops.inference.likelihood.base.Likelihood`. That base class
defines the initialization workflow, compatibility contract, and evaluation
interface.

Before diving into extension patterns, we examine each layer in turn.

Low-Level Numerical Likelihoods
-------------------------------

At the foundation of the Triceratops likelihood system lie the **pure
numerical backends**. These are ordinary Python functions — typically located
in submodules of :mod:`triceratops.inference.likelihood` such as
:mod:`triceratops.inference.likelihood.gaussian` — whose sole responsibility
is to evaluate a statistical expression in array space.

These functions operate directly on NumPy arrays and return a scalar
log-likelihood value. They are deliberately unaware of:

- physical models,
- data container abstractions,
- unit systems,
- inference engines.

Their role is purely mathematical.

For example,
:func:`~triceratops.inference.likelihood.gaussian.gaussian_loglikelihood`
computes the standard Gaussian log-likelihood from arrays of observed values,
model predictions, and uncertainties. It assumes that the arrays are already
compatible, correctly shaped, and expressed in consistent units. It does not
ask where those arrays came from, nor does it attempt to reshape, validate, or
interpret them.

In other words, this layer encodes

.. math::

    \ln \mathcal{L}(\mathbf{y} \mid \boldsymbol{\mu}, \boldsymbol{\sigma})

and nothing else.

This separation is intentional and foundational. By isolating statistical
formulas from model and data semantics, the numerical layer becomes:

- straightforward to unit-test in isolation,
- easy to reason about analytically,
- reusable across multiple likelihood classes,
- safe to use inside tight sampling loops.

.. important::

   A low-level numerical likelihood function must:

   - Accept only NumPy arrays and simple scalars,
   - Return a scalar log-likelihood value,
   - Perform no unit coercion,
   - Perform no model evaluation,
   - Inspect no container objects,
   - Avoid implicit reshaping beyond what NumPy enforces.

   If a function violates these constraints, it does not belong in the
   numerical backend layer.

----

The Likelihood Base Class
-------------------------

The structural core of the likelihood system is
:class:`~triceratops.inference.likelihood.base.Likelihood`.

This abstract base class does not implement a specific statistical model.
Instead, it defines the lifecycle and contract that all concrete likelihoods
must obey. Its purpose is architectural: it ensures that model binding,
compatibility validation, and evaluation occur in a consistent and
disciplined manner.

A likelihood instance represents the binding of:

- one :class:`~triceratops.models.core.base.Model`,
- one :class:`~triceratops.data.core.InferenceData`,
- one statistical assumption.

Once constructed, it becomes a callable object capable of evaluating
log-likelihoods efficiently and safely.


Initialization Lifecycle
^^^^^^^^^^^^^^^^^^^^^^^^^

All likelihood subclasses inherit the constructor defined in
:class:`~triceratops.inference.likelihood.base.Likelihood`. That constructor
enforces a strict sequence of steps:

1. Bind the model and inference data.
2. Invoke :meth:`_configure`.
3. Validate compatibility via :meth:`_validate_model_and_data`.

Unlike earlier versions of the framework, there is no separate data
preprocessing stage. Any one-time array extraction or transformation must
occur inside :meth:`_configure`.

Each step has a narrowly defined responsibility.


Model and Data Binding
^^^^^^^^^^^^^^^^^^^^^^

The constructor requires:

- a :class:`Model`,
- an :class:`InferenceData`.

Basic interface checks ensure that the objects conform to the expected
abstract base classes before proceeding.

At this stage, no structural validation is performed. The objects are
stored internally as ``self._model`` and ``self._data_container``.


Configuration Hook: ``_configure``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :meth:`_configure` method runs once during initialization.

This method is responsible for:

- Extracting stacked NumPy arrays from :class:`InferenceData`,
- Caching independent variables,
- Constructing masks (for censoring or selection),
- Precomputing anything that would otherwise occur inside the hot loop.

For example:

.. code-block:: python

    def _configure(self, **kwargs):

        self._x = self._data_container.x

        self._data_y = self._data_container.get_y_array(flatten=False)
        self._y_err = self._data_container.get_y_error_array(flatten=False)

Configuration should not:

- Evaluate the forward model,
- Perform statistical calculations,
- Modify the inference data object.

Its sole purpose is one-time numerical preparation.


Compatibility Validation: ``_validate_model_and_data``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After configuration, subclasses validate structural assumptions.

Typical checks include:

- Required uncertainty arrays exist,
- Model dimensionality matches observable dimensionality,
- Censoring arrays are structurally consistent,
- No unsupported data features are present.

Validation verifies assumptions.
It does not transform arrays.


Likelihood Evaluation
^^^^^^^^^^^^^^^^^^^^^

Evaluation is intentionally split into two methods.

The public method
:meth:`~triceratops.inference.likelihood.base.Likelihood.log_likelihood`
accepts parameters in any format supported by the model and coerces them
into the model’s internal representation.

It then delegates to
:meth:`~triceratops.inference.likelihood.base.Likelihood._log_likelihood`,
which assumes parameters are already in raw format.

A typical implementation looks like:

.. code-block:: python

    def _log_likelihood(self, parameters):

        model_tuple = self._model._forward_model_tupled(
            self._x,
            parameters,
        )

        model_y = np.stack(model_tuple, axis=-1)

        return gaussian_loglikelihood(
            data_y=self._data_y,
            model_y=model_y,
            y_err=self._y_err,
        )

At evaluation time, the likelihood reduces to:

1. Forward model evaluation,
2. Lightweight array stacking,
3. Delegation to the numerical backend,
4. Scalar return.

Nothing else.

.. admonition:: Performance Principle

   Any operation that can be performed once during initialization must not
   be repeated inside :meth:`_log_likelihood`. Sampling loops may invoke that
   method millions of times.


Concrete Likelihood Classes
---------------------------

Concrete likelihood classes implement a specific statistical assumption
for a given :class:`InferenceData` object.

Because :class:`InferenceData` already guarantees:

- validated shapes,
- consistent stacking,
- resolved variable and observable names,

the likelihood layer does not interpret tables, units, or schema.

Instead, it operates purely on stacked arrays.

A complete Gaussian likelihood under the current architecture may look like:

.. code-block:: python

    class GaussianLikelihood(Likelihood):

        def _configure(self, **kwargs):

            self._x = self._data_container.x

            self._data_y = self._data_container.get_y_array(flatten=False)
            self._y_err = self._data_container.get_y_error_array(flatten=False)

            if self._y_err is None:
                raise ValueError(
                    "GaussianLikelihood requires observable uncertainties."
                )

        def _validate_model_and_data(self):
            pass

        def _log_likelihood(self, parameters):

            model_tuple = self._model._forward_model_tupled(
                self._x,
                parameters,
            )

            model_y = np.stack(model_tuple, axis=-1)

            return gaussian_loglikelihood(
                data_y=self._data_y,
                model_y=model_y,
                y_err=self._y_err,
            )

The censored case follows the same pattern, with additional arrays and
masks cached in :meth:`_configure`.

----

Summary
-------

Under the current architecture:

- Numerical backends implement pure statistics.
- :class:`InferenceData` defines the numerical contract.
- Likelihood classes bind models to numerical arrays.
- All preprocessing occurs once in :meth:`_configure`.
- The hot loop performs only forward evaluation and backend delegation.

This separation ensures that:

- Statistical mathematics remains testable,
- Data semantics remain isolated,
- Inference backends remain fast,
- The system remains extensible without architectural drift.
