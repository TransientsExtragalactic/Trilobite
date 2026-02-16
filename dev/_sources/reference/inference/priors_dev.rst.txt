.. _priors_dev:

========================
Prior Development Guide
========================

This document describes how prior distributions are implemented in
Triceratops and how developers can extend the prior system safely
and consistently.

The reference implementation lives in
``triceratops/inference/prior.py`` :contentReference[oaicite:0]{index=0}.

Overview
--------

In Triceratops, priors are lightweight, stateless, callable objects
that evaluate the **log-prior probability** of a parameter value in
physical parameter space. The only quantity required by the inference
engine is:

.. math::

    \log p(x)

Everything else — PDFs, CDFs, transforms, sampling-space changes —
is optional or handled elsewhere.

The prior system is intentionally:

- sampler-agnostic,
- model-agnostic,
- lightweight,
- easy to extend,
- explicit about parameter support.

A prior must always return ``-np.inf`` outside of its support.

The Prior Base Class
--------------------

All priors inherit from the abstract base class
:class:`~triceratops.inference.prior.Prior`.

At construction time, a prior stores its defining parameters and
generates callable evaluation functions:

.. code-block:: python

    class NormalPrior(Prior):
        def __init__(self, mean, sigma):
            super().__init__(mean=mean, sigma=sigma)

The base class automatically builds:

- ``self._log_prior`` (required),
- ``self._prior`` (optional PDF),
- ``self._cum_prior`` (optional CDF).

The only method subclasses are required to implement is:

.. code-block:: python

    def _generate_log_prior(self, **parameters):
        ...

This method must return a callable of the form:

.. code-block:: python

    def logp(x: float) -> float:
        ...

The callable must return ``-np.inf`` outside of its support.

The Public Interface
--------------------

Priors expose a minimal evaluation interface:

.. code-block:: python

    prior = NormalPrior(mean=0.0, sigma=1.0)

    logp = prior(0.3)      # alias for prior.logp(0.3)

Optional PDF and CDF evaluation are also supported:

.. code-block:: python

    p = prior.pdf(0.3)
    F = prior.cdf(0.3)

If a subclass does not implement a PDF or CDF, calling these methods
raises ``NotImplementedError``.

Importantly, priors operate strictly in **physical parameter space**.
Any transformations (e.g., log-sampling) are handled by
:class:`InferenceParameter` inside the inference problem layer.

Serialization
-------------

Priors are serializable for reproducibility and metadata storage.
Each prior implements:

.. code-block:: python

    prior_dict = prior.to_dict()
    prior2 = Prior.from_dict(prior_dict)

This mechanism works for all built-in priors.

Custom callable priors (wrapped by :class:`CallablePrior`) are marked
as non-reconstructible because arbitrary Python callables cannot be
serialized safely.

If you introduce new prior subclasses, ensure they are discoverable
via the ``Prior._all_subclasses()`` mechanism so that deserialization
continues to work automatically.

Implementing a New Prior
------------------------

To implement a new prior, subclass :class:`Prior` and define
``_generate_log_prior``.

For example, suppose we want a simple exponential prior:

.. code-block:: python

    class ExponentialPrior(Prior):

        def __init__(self, rate: float):
            if rate <= 0:
                raise ValueError("Rate must be positive.")
            super().__init__(rate=float(rate))

        def _generate_log_prior(self, *, rate: float):

            def logp(x: float) -> float:
                if x >= 0:
                    return np.log(rate) - rate * x
                return -np.inf

            return logp

There are a few important design rules:

1. Validate parameters at initialization time.
2. Enforce support explicitly.
3. Always return ``-np.inf`` outside the support.
4. Avoid stateful behavior.
5. Avoid dependence on model or sampler internals.

If your prior has finite normalization and analytic PDF/CDF expressions,
you may optionally implement:

.. code-block:: python

    def _generate_prior(self, **parameters):
        ...

    def _generate_cum_prior(self, **parameters):
        ...

These are not required for MCMC or nested sampling but may be useful
for diagnostics or prior predictive checks.

Callable Priors
---------------

Users may supply arbitrary log-prior callables without subclassing:

.. code-block:: python

    def custom_log_prior(x):
        if x > 0:
            return -x
        return -np.inf

    prior = CallablePrior(custom_log_prior)

This wrapper allows flexibility without requiring users to implement
a full class. However, such priors cannot be serialized and are marked
as non-reconstructible.

Support and Stability Guarantees
--------------------------------

All priors in Triceratops must satisfy the following invariants:

- They operate on scalar floats in physical units.
- They are deterministic and side-effect free.
- They return finite log-probabilities inside support.
- They return ``-np.inf`` outside support.
- They do not perform parameter transformations.
- They are compatible with any sampler.

Improper priors are strongly discouraged. If introduced, they should
be clearly documented and must not silently produce non-finite
normalization behavior.

Interaction with InferenceProblem
---------------------------------

The :class:`InferenceProblem` attaches priors to parameters and evaluates
them during posterior computation. From the prior's perspective, it simply
receives a float and returns a float.

For example:

.. code-block:: python

    problem.set_prior("B", "loguniform", lower=1e-3, upper=10.0)

Internally, the inference problem handles:

- unit coercion,
- parameter freezing,
- transform inversion,
- Jacobian corrections (if applicable).

The prior itself remains unaware of all of these details.

Final Remarks
-------------

The prior system in Triceratops is intentionally simple and explicit.
Its design prioritizes clarity, extensibility, and numerical stability
over abstraction or implicit behavior.

When implementing new priors, focus on:

- clear mathematical definition,
- strict support enforcement,
- numerical robustness,
- and consistency with the log-probability interface.

If those principles are respected, priors will integrate cleanly
with the broader inference framework.
