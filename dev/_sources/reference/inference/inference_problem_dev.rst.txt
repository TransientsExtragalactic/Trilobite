.. _inference_problem_dev:

========================================
Inference Problem Developer Guide
========================================

This document describes how to extend and customize the
:class:`~triceratops.inference.problem.InferenceProblem` framework.

If you are writing new likelihoods, you should read
:ref:`likelihood_dev` first. This guide focuses specifically on:

- how parameters are wrapped and managed,
- how priors and transforms are handled,
- how packing/unpacking works internally,
- and how to safely subclass :class:`InferenceProblem`.

The reference implementation lives in
``triceratops/inference/problem.py``.

Design Goals
------------

The :class:`InferenceProblem` class serves as the **statistical glue layer**
between:

- a physical :class:`~models.core.base.Model`,
- a :class:`~inference.likelihood.base.Likelihood`,
- and external sampling backends.

Its responsibilities are strictly limited to:

- parameter bookkeeping,
- prior management,
- log-probability evaluation,
- validation and consistency checks,
- sampler-facing vector interfaces.

It is intentionally:

- **model-agnostic**,
- **likelihood-agnostic**,
- **sampler-agnostic**.

No sampling logic belongs in this class.

Core Architecture
-----------------

InferenceParameter
^^^^^^^^^^^^^^^^^^

Each model parameter is wrapped in an
:class:`~triceratops.inference.problem.InferenceParameter`.

This dataclass stores:

- the underlying :class:`~models.core.parameters.ModelParameter`,
- freeze status,
- prior object,
- optional transforms,
- initial value (in model base units).

These objects are automatically constructed during
:class:`InferenceProblem` initialization:

.. code-block:: python

    self.__parameters__ = {
        model_parameter.name: InferenceParameter(
            model_parameter=model_parameter,
            freeze=False,
            prior=None,
            transform=None,
            inverse_transform=None,
            initial_value=None,
        )
        for model_parameter in self.__likelihood__._model.PARAMETERS
    }

Developers should **never manually construct**
``InferenceParameter`` objects outside this initialization pathway.

Parameter Ordering
^^^^^^^^^^^^^^^^^^

A critical invariant of the inference system is that **parameter ordering
is deterministic and inherited from the model definition**:

.. code-block:: python

    self.__parameter_order__ = tuple(
        [p.name for p in self.__likelihood__._model.PARAMETERS]
    )

This ordering controls:

- vector packing,
- unpacking,
- sampler interfaces,
- dimensionality checks.

Never override ordering unless you fully understand the downstream
consequences for samplers.

Probability Evaluation Flow
---------------------------

The evaluation chain is:

1. Sampler provides ``theta_free`` (free parameters only).
2. ``_log_free_posterior`` expands to full vector.
3. ``_log_posterior`` computes:
   - ``_log_prior``
   - ``_log_likelihood``
4. Likelihood delegates to its internal optimized backend.

Key methods:

.. code-block:: python

    def _log_posterior(self, theta):
        logp = self._log_prior(theta)
        if not np.isfinite(logp):
            return -np.inf

        logl = self._log_likelihood(theta)
        if not np.isfinite(logl):
            return -np.inf

        return logp + logl

Important design rule:

- Always short-circuit on non-finite priors.
- Never allow NaNs to propagate into samplers.

High-Level vs Low-Level API
---------------------------

The class provides two interfaces:

Dictionary API (user-facing)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    inf_prob.log_posterior({
        "B": 0.3,
        "n0": 1.0,
    })

This API:

- coerces units,
- inserts frozen parameters,
- validates structure.

Vector API (sampler-facing)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    theta = inf_prob.initial_theta
    logp = inf_prob(theta)

This path:

- assumes ordering is correct,
- assumes floats in base units,
- performs no expensive coercion.

All samplers should use the vector API.

Packing and Unpacking
---------------------

The inference problem provides four key methods:

- ``pack_parameters``
- ``unpack_parameters``
- ``pack_free_parameters``
- ``unpack_free_parameters``

These guarantee:

- deterministic ordering,
- frozen parameter insertion,
- dimension consistency.

Developers extending the class must preserve these guarantees.

Transforms
----------

Transforms allow parameters to be sampled in a different space
(e.g., log-space).

If a transform is provided:

- ``transform``
- ``inverse_transform``
- optionally ``transform_jacobian``

must be defined consistently.

Invariant:

.. code-block:: python

    x = p.initial_value
    y = p.transform(x)
    x_back = p.inverse_transform(y)

``x_back`` must equal ``x`` within floating precision.

Transforms are applied during prior

Sources

You said:
Let's use more prose and fewer bulleted list. I want to focus on a nice readable guide with good examples and complete sentence strings.

ChatGPT said:
.. _inference_problem_dev:

========================================
Inference Problem Developer Guide
========================================

This guide describes the internal structure and intended extension points of
:class:`~triceratops.inference.problem.InferenceProblem`. It is written for
developers who wish to customize parameter handling, introduce new inference
workflows, or build higher-level abstractions on top of the core inference layer.

If you are implementing a new likelihood, see :ref:`likelihood_dev` first.
The likelihood defines *how* data and models are compared. The inference problem
defines *how parameters are managed and how probabilities are evaluated*.

The reference implementation can be found in
``triceratops/inference/problem.py`` :contentReference[oaicite:0]{index=0}.

Overview and Philosophy
-----------------------

The :class:`InferenceProblem` class exists to separate physics from statistics
and statistics from numerics. The physical model defines parameters and produces
predictions. The likelihood defines how those predictions are compared to data.
The inference problem binds those pieces together and exposes a clean,
sampler-ready interface.

Crucially, the inference problem does *not* implement any sampling algorithm.
It does not run MCMC, nested sampling, or optimization routines. Instead, it
provides a stable and predictable API that samplers can call repeatedly and
efficiently.

An inference problem is therefore best understood as a *probability evaluation
engine* with strong bookkeeping guarantees.

Parameter Wrapping and Internal Representation
----------------------------------------------

When an :class:`InferenceProblem` is initialized, it receives a fully
constructed :class:`~triceratops.inference.likelihood.base.Likelihood`
instance. From this likelihood, it extracts the underlying model and reads
the model's declared parameter definitions.

Each model parameter is wrapped in an
:class:`~triceratops.inference.problem.InferenceParameter` object. This wrapper
stores:

- whether the parameter is frozen,
- which prior distribution is attached,
- any optional transforms,
- the initial value in base units.

Developers should never manually construct these wrappers. They are created
automatically during initialization to ensure consistency with the model’s
parameter definitions.

A minimal internal initialization flow looks like:

.. code-block:: python

    problem = InferenceProblem(likelihood)

    # Parameters are auto-generated
    print(problem.parameters)

The order of parameters is inherited directly from the model definition. This
ordering is deterministic and must not be altered arbitrarily. Samplers rely on
this order when packing and unpacking parameter vectors.

Understanding Parameter Packing
-------------------------------

Samplers operate on NumPy arrays rather than dictionaries. For this reason,
the inference problem provides explicit packing and unpacking utilities.

Consider the following example:

.. code-block:: python

    params = {
        "B": 0.3,
        "n0": 1.0,
        "epsilon_e": 0.1,
    }

    theta = problem.pack_parameters(params)
    params_back = problem.unpack_parameters(theta)

The returned vector ``theta`` is ordered according to the model’s internal
parameter definition. This guarantees that repeated likelihood evaluations
are unambiguous.

When sampling, only *free* parameters are included in the vector representation:

.. code-block:: python

    theta_free = problem.initial_theta
    logpost = problem(theta_free)

Frozen parameters are automatically reinserted before likelihood evaluation.
This design ensures that samplers never need to reason about frozen parameters
or dictionary bookkeeping.

Probability Evaluation Internals
--------------------------------

The log-posterior is evaluated in three steps:

1. Compute the log-prior.
2. If the prior is finite, compute the log-likelihood.
3. Return their sum.

The implementation short-circuits immediately if the prior evaluates to
``-np.inf``. This prevents invalid parameter regions from ever reaching the
likelihood evaluation stage.

A simplified version of the internal logic is:

.. code-block:: python

    def _log_posterior(self, theta):
        logp = self._log_prior(theta)
        if not np.isfinite(logp):
            return -np.inf

        logl = self._log_likelihood(theta)
        if not np.isfinite(logl):
            return -np.inf

        return logp + logl

The likelihood evaluation itself is delegated entirely to the likelihood
object. The inference problem does not modify model predictions or data.

High-Level and Low-Level Interfaces
-----------------------------------

Two complementary APIs are provided.

The dictionary-based API is intended for debugging and interactive use:

.. code-block:: python

    logpost = problem.log_posterior({
        "B": 0.3,
        "n0": 1.0,
    })

This interface performs unit coercion and full parameter validation.

The vectorized API is intended for samplers:

.. code-block:: python

    theta = problem.initial_theta
    logpost = problem(theta)

This path assumes parameters are already in base units and correctly ordered.
It avoids expensive validation overhead and should be used inside tight loops.

Parameter Transforms
--------------------

Inference parameters may optionally define transforms. This allows sampling
in an unconstrained or reparameterized space, such as log-space for strictly
positive parameters.

A valid transform configuration must define:

- ``transform``
- ``inverse_transform``
- optionally ``transform_jacobian``

The forward and inverse transforms must be mathematically consistent:

.. code-block:: python

    x = p.initial_value
    y = p.transform(x)
    x_back = p.inverse_transform(y)

    assert np.allclose(x, x_back)

Transforms are applied before prior evaluation. If a Jacobian correction is
required for the chosen sampling scheme, it must be included explicitly.

Validation and Safety Checks
----------------------------

Before sampling begins, the inference problem should be validated:

.. code-block:: python

    problem.validate_for_inference()

This routine verifies that:

- at least one parameter is free,
- all free parameters have priors,
- initial values are finite and within support,
- transforms are internally consistent,
- packing and unpacking behave correctly,
- likelihood evaluation succeeds at the initial point.

This validation step is intentionally strict. It is better to fail immediately
than to waste computational resources on an ill-defined inference problem.

Subclassing InferenceProblem
----------------------------

In most cases, subclassing is not necessary. The base class is sufficient for
nearly all workflows.

However, you may wish to subclass if you need to:

- enforce custom parameter initialization rules,
- automatically freeze or transform certain parameters,
- implement specialized validation logic,
- provide convenience constructors for domain-specific inference setups.

When subclassing, follow these guidelines:

1. Always call ``super().__init__(likelihood)``.
2. Do not alter ``__parameter_order__`` unless absolutely necessary.
3. Preserve the packing/unpacking interface.
4. Never embed sampling logic inside the inference problem.

The inference problem must remain a clean, stateless probability evaluation
layer.
