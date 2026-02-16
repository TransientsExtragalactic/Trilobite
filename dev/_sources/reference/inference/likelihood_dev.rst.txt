.. _likelihood_dev:

===============================
Likelihood Development Guide
===============================

Likelihoods are the bridge between physics and inference in Triceratops.
They bind a concrete :class:`~models.core.base.Model` to a
:class:`~data.core.DataContainer` under a specific statistical
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

A Triceratops likelihood is not a monolithic object. It is composed of three
conceptual layers:

1. A **numerical backend** that encodes pure statistics.
2. A **likelihood stencil** that links a forward model to that backend.
3. A **concrete likelihood class** that adapts a specific data container.

The separation between these layers is enforced by
:class:`~inference.likelihood.base.Likelihood`. That base class defines the
initialization workflow, compatibility contract, and evaluation interface.

Before diving into extension patterns, we examine each layer in turn.

Low-Level Numerical Likelihoods
-------------------------------

At the foundation of the Triceratops likelihood system lie the **pure
numerical backends**. These are ordinary Python functions — typically located
in submodules of :mod:`inference.likelihood` such as
:mod:`inference.likelihood.gaussian` — whose sole responsibility is to evaluate
a statistical expression in array space.

These functions operate directly on NumPy arrays and return a scalar
log-likelihood value. They are deliberately unaware of:

- physical models,
- data container abstractions,
- unit systems,
- masking semantics beyond what is passed explicitly,
- inference engines.

Their role is purely mathematical.

For example,
:func:`~inference.likelihood.gaussian.gaussian_loglikelihood`
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
   - Inspect no data container objects,
   - Avoid implicit reshaping or structural validation beyond what NumPy enforces.

   If a function violates these constraints, it does not belong in the
   numerical backend layer.

.. rubric:: Examples

The Gaussian case is familiar, but the pattern generalizes immediately to
other statistical models.

**Poisson Likelihood**

Suppose observations consist of integer event counts :math:`k_i`, and the
model predicts expected rates :math:`\lambda_i`. The Poisson log-likelihood is

.. math::

    \ln \mathcal{L}
    =
    \sum_i
    \left[
        k_i \ln \lambda_i
        - \lambda_i
        - \ln(k_i!)
    \right].

A suitable numerical backend would look like:

.. code-block:: python

    import numpy as np
    from scipy.special import gammaln

    def poisson_loglikelihood(data_k, model_rate):
        return np.sum(
            data_k * np.log(model_rate)
            - model_rate
            - gammaln(data_k + 1)
        )

This function:

- assumes ``data_k`` and ``model_rate`` are already compatible arrays,
- does not verify non-negativity,
- does not inspect the model,
- does not perform unit handling.

All semantic validation belongs elsewhere.

**Binomial Likelihood**

Now suppose each observation consists of a number of trials :math:`n_i`
and observed successes :math:`k_i`, while the model predicts success
probabilities :math:`p_i`. The Binomial log-likelihood is

.. math::

    \ln \mathcal{L}
    =
    \sum_i
    \left[
        \ln \binom{n_i}{k_i}
        + k_i \ln p_i
        + (n_i - k_i)\ln(1 - p_i)
    \right].

A corresponding numerical backend might be written as:

.. code-block:: python

    import numpy as np
    from scipy.special import gammaln

    def binomial_loglikelihood(k, n, p):
        log_binom_coeff = (
            gammaln(n + 1)
            - gammaln(k + 1)
            - gammaln(n - k + 1)
        )

        return np.sum(
            log_binom_coeff
            + k * np.log(p)
            + (n - k) * np.log(1.0 - p)
        )

Again, this function:

- assumes valid inputs,
- performs no structural checks,
- does not enforce bounds on ``p``,
- does not validate integer constraints.

Those responsibilities lie in higher layers of the likelihood stack.

----

The Likelihood Base Class
-------------------------

The structural core of the likelihood system is
:class:`~inference.likelihood.base.Likelihood`.

This abstract base class does not implement a specific statistical model.
Instead, it defines the lifecycle and contract that all concrete likelihoods
must obey. Its purpose is architectural: it ensures that model binding,
compatibility validation, preprocessing, and evaluation occur in a consistent
and disciplined manner.

A likelihood instance represents the binding of:

- one :class:`~triceratops.models.core.base.Model`,
- one :class:`~triceratops.data.core.DataContainer`,
- one statistical assumption.

Once constructed, it becomes a callable object capable of evaluating
log-likelihoods efficiently and safely.


Initialization Lifecycle
^^^^^^^^^^^^^^^^^^^^^^^^^

All likelihood subclasses inherit the constructor defined in
:class:`~inference.likelihood.base.Likelihood`. That constructor enforces
a strict sequence of steps:

1. Bind the model and data container.
2. Invoke :meth:`_configure`.
3. Validate compatibility via :meth:`_validate_model_and_data`.
4. Preprocess input data via :meth:`_process_input_data`.

Each step has a narrowly defined responsibility. Understanding those
responsibilities is essential when implementing a new likelihood.


Model and Data Binding
^^^^^^^^^^^^^^^^^^^^^^

The constructor requires a model instance and a data container instance.
Basic interface checks ensure that the objects conform to the expected
abstract base classes before proceeding.

At this stage, no structural validation is performed. The objects are
simply stored internally as ``self._model`` and ``self._data_container``.

The purpose of this early binding is to guarantee that all subsequent
validation and preprocessing logic has access to both components.


Configuration Hook: ``_configure``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :meth:`_configure` method provides an optional configuration hook that
runs before validation and preprocessing.

This method is typically used to parse keyword arguments or toggle internal
options that influence how the likelihood behaves. Many likelihoods require
no configuration and may simply implement:

.. code-block:: python

    def _configure(self, **kwargs):
        pass

However, a likelihood supporting, for example, optional robust weighting
might implement:

.. code-block:: python

    def _configure(self, robust=False, **kwargs):
        self._robust = robust

Configuration logic should remain lightweight. It should not:

- Access or reshape data arrays,
- Perform unit coercion,
- Evaluate the forward model.

Those steps belong later in the lifecycle.


Compatibility Validation: ``_validate_model_and_data``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every likelihood class declares compatibility at the class level via
``COMPATIBLE_MODELS`` and ``COMPATIBLE_DATA_CONTAINERS``. The base class
verifies these declarations when the subclass is defined and checks them
again during initialization.

The default implementation ensures that:

- The model instance is of an allowed type.
- The data container instance is of an allowed type.

Concrete subclasses may extend :meth:`_validate_model_and_data`
to enforce additional structural requirements.

For example, a Gaussian likelihood for XY data might require:

.. code-block:: python

    def _validate_model_and_data(self):
        super()._validate_model_and_data()

        if not self._data_container.has_column("y_err"):
            raise ValueError(
                "GaussianLikelihoodXY requires a 'y_err' column."
            )

        if self._model.n_independent_variables != 1:
            raise ValueError(
                "GaussianLikelihoodXY requires a single independent variable."
            )

Validation should check structure and semantic assumptions,
but it should not transform data.

.. admonition:: Separation of Concerns

   Validation verifies that assumptions are satisfied.
   It does not perform unit coercion, reshaping, or array extraction.
   Those operations belong in :meth:`_process_input_data`.


One-Time Data Preprocessing: ``_process_input_data``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After compatibility is confirmed, the constructor calls
:meth:`_process_input_data`.

This method performs all transformations required to convert a data container
into the numerical form expected by the likelihood backend.

It must return a :class:`types.SimpleNamespace` containing only NumPy arrays
and any masks required for evaluation.

A minimal Gaussian example might look like:

.. code-block:: python

    from types import SimpleNamespace

    def _process_input_data(self):
        x = self._data_container.get_column("x").to_value(
            self._model.x_unit
        )

        y = self._data_container.get_column("y").to_value(
            self._model.y_unit
        )

        y_err = self._data_container.get_column("y_err").to_value(
            self._model.y_unit
        )

        return SimpleNamespace(
            x=x,
            y=y,
            y_err=y_err,
        )

This method is called exactly once during initialization. Its output is stored
internally as ``self._data``.

After this point, likelihood evaluation should never access
``self._data_container`` again.

.. admonition:: Performance Principle

   Any operation that can be performed once during initialization must not
   be repeated inside :meth:`_log_likelihood`. Sampling loops may invoke that
   method millions of times.


Likelihood Evaluation
^^^^^^^^^^^^^^^^^^^^^

Evaluation is intentionally split into two methods.

The public method
:meth:`~inference.likelihood.base.Likelihood.log_likelihood`
accepts parameters in any format supported by the model and coerces them
into the model’s internal representation.

It then delegates to
:meth:`~inference.likelihood.base.Likelihood._log_likelihood`,
which assumes parameters are already in raw format.

A typical implementation of ``_log_likelihood`` might look like:

.. code-block:: python

    def _log_likelihood(self, parameters):
        model_y = self._model._forward_model_tupled(
            self._data.x,
            parameters,
        )[0]

        return gaussian_loglikelihood(
            data_y=self._data.y,
            model_y=model_y,
            y_err=self._data.y_err,
        )

Notice the absence of:

- Unit conversion,
- Mask construction,
- Structural checks.

At evaluation time, the likelihood should reduce to:
1. Evaluate the forward model.
2. Pass arrays to the numerical backend.
3. Return a scalar.

Inference engines should call :meth:`_log_likelihood` directly when operating
in tight loops to avoid repeated parameter coercion overhead.

Likelihood Stencils
-------------------

A likelihood stencil is an abstract subclass of
:class:`~inference.likelihood.base.Likelihood` that defines the *structural*
relationship between model predictions and a numerical backend.

Where the low-level functions encode mathematics, and concrete classes handle
data semantics, a stencil encodes the *wiring* between them.

More precisely, a stencil answers the question:

    Given preprocessed NumPy arrays and a forward model,
    how is this likelihood evaluated?

A stencil therefore defines the shape of ``_log_likelihood``. It assumes that
:meth:`_process_input_data` has already produced a
:class:`types.SimpleNamespace` containing the necessary arrays, and it
assumes that a numerical backend exists to compute the log-likelihood from
those arrays.

What a Stencil Does
^^^^^^^^^^^^^^^^^^^

A stencil typically performs exactly two operations:

1. Evaluate the forward model using the independent variables stored in
   ``self._data``.
2. Delegate to a low-level numerical likelihood function.

For example,
:class:`~inference.likelihood.base.GaussianLikelihoodStencil`
computes model predictions and passes them to
:func:`~inference.likelihood.gaussian.gaussian_loglikelihood`.

A minimal stencil might look like:

.. code-block:: python

    class GaussianLikelihoodStencil(Likelihood, ABC):

        def _log_likelihood(self, parameters):
            model_y = self._model._forward_model_tupled(
                self._data.x,
                parameters,
            )[0]

            return gaussian_loglikelihood(
                data_y=self._data.y,
                model_y=model_y,
                y_err=self._data.y_err,
            )

Notice what is absent:

- No statistical expressions.
- No container access.
- No unit conversion.
- No mask construction.

All of those responsibilities belong to other layers.

.. admonition:: Responsibility Boundary

   If you find yourself re-deriving Gaussian formulas inside a stencil,
   that logic belongs in :mod:`inference.likelihood.gaussian`.

   If you find yourself extracting columns from a data container,
   that logic belongs in a concrete likelihood class.

Stencils as Reusable Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stencils are especially valuable when multiple concrete likelihood classes
share the same statistical structure but differ in their data sources.

For example, a Gaussian stencil could support:

- XY data containers,
- Spectral data containers,
- Time-series containers,

as long as each concrete class produces the same numerical namespace
(``x``, ``y``, ``y_err``, etc.).

The stencil isolates the statistical structure from the data representation.
It is the middle layer of the architecture: neither purely mathematical nor
purely semantic.


Concrete Likelihood Classes
---------------------------

Concrete likelihood classes complete the stack. They adapt a specific
:class:`~triceratops.data.core.DataContainer` to the numerical contract
expected by a stencil.

This is the layer where data semantics live.

While a stencil assumes arrays already exist, a concrete class is responsible
for producing those arrays correctly and consistently.

In practice, this means:

- Interpreting container columns,
- Enforcing structural requirements,
- Coercing units into the model’s base units,
- Constructing masks or censoring arrays,
- Returning a clean numerical namespace.

A typical implementation combines compatibility validation and preprocessing.

For example,
:class:`~inference.likelihood.base.GaussianLikelihoodXY`
may implement:

.. code-block:: python

    class GaussianLikelihoodXY(GaussianLikelihoodStencil):

        COMPATIBLE_MODELS = (Model,)
        COMPATIBLE_DATA_CONTAINERS = (XYDataContainer,)

        def _validate_model_and_data(self):
            super()._validate_model_and_data()

            if not self._data_container.has_column("y_err"):
                raise ValueError(
                    "GaussianLikelihoodXY requires a 'y_err' column."
                )

        def _process_input_data(self):
            from types import SimpleNamespace

            x = self._data_container.get_column("x").to_value(
                self._model.x_unit
            )

            y = self._data_container.get_column("y").to_value(
                self._model.y_unit
            )

            y_err = self._data_container.get_column("y_err").to_value(
                self._model.y_unit
            )

            return SimpleNamespace(
                x=x,
                y=y,
                y_err=y_err,
            )

After this preprocessing step, the original data container is no longer
accessed during likelihood evaluation. The stencil operates purely on
``self._data``.

.. admonition:: Semantic Isolation

   The numerical backend must remain completely ignorant of how the data were
   stored. The stencil must remain ignorant of how the data were extracted.
   Only the concrete class may interpret container semantics.
