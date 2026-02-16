.. _samplers_dev:

==========================
Sampler Development Guide
==========================

This document describes how sampling algorithms are integrated into the
Triceratops inference framework and how to implement new samplers
in a way that is consistent, robust, and fully compatible with the
existing infrastructure.

The core sampler interface is defined in
``triceratops/inference/sampling/base.py``
An example concrete implementation using ``emcee`` can be found in
``triceratops/inference/sampling/mcmc.py``
Sampling outputs are handled by classes in
``triceratops/inference/sampling/result.py``
Supporting statistical utilities live in
``triceratops/inference/sampling/utils.py``

Overview and Design Philosophy
-------------------------------

In Triceratops, a sampler is responsible for *exploring* the posterior
distribution defined by an :class:`InferenceProblem`. It does not define
the posterior, it does not manage priors, and it does not interact directly
with the physical model or dataset. All of that logic is encapsulated inside
the inference problem layer.

A sampler therefore has a very narrow and well-defined responsibility:

1. Repeatedly evaluate the posterior through the inference problem.
2. Manage algorithm-specific state (walkers, chains, live points, etc.).
3. Return results in a standardized :class:`SamplingResult` object.

This separation ensures that new sampling strategies can be added without
modifying any model or likelihood code.

The Sampler Base Class
----------------------

All samplers must inherit from
:class:`~triceratops.inference.sampling.base.Sampler`

The base class defines a minimal interface:

- It stores a reference to an :class:`InferenceProblem`.
- It exposes a public :meth:`run` method (abstract).
- It provides a hook for validating compatibility.

A minimal skeleton looks like:

.. code-block:: python

    from triceratops.inference.sampling.base import Sampler
    from triceratops.inference.sampling.result import SamplingResult

    class MySampler(Sampler):

        def _validate_inference_problem(self):
            # Perform compatibility checks here
            self.problem.validate_for_inference()

        def run(self, **kwargs) -> SamplingResult:
            # Implement sampling logic
            ...
            return SamplingResult(...)

The constructor receives an :class:`InferenceProblem` instance and
stores it internally. This problem object is the only interface
a sampler needs in order to evaluate the posterior.

Interacting with the InferenceProblem
-------------------------------------

Samplers must use the **vectorized API** of the inference problem.
Specifically, they should evaluate:

.. code-block:: python

    logp = self.problem._log_free_posterior(theta)

or, equivalently:

.. code-block:: python

    logp = self.problem(theta)

The vector ``theta`` must represent only the free parameters, ordered
according to the model definition. The inference problem handles:

- reinserting frozen parameters,
- evaluating priors,
- delegating likelihood evaluation,
- short-circuiting invalid regions.

Samplers must never manipulate priors or likelihoods directly.

Validation
----------

Before sampling begins, the sampler should ensure that the inference
problem is fully specified and valid. The standard approach is:

.. code-block:: python

    self.problem.validate_for_inference()

The base class provides a hook method
:meth:`_validate_inference_problem` for algorithm-specific constraints.

For example, the ``EmceeSampler`` verifies that:

- the problem has at least one free parameter,
- the inference problem passes validation,
- the number of walkers is sufficient.

See the implementation in
``triceratops/inference/sampling/mcmc.py``

Implementing a New Sampler
--------------------------

When implementing a new sampler, follow these principles:

1. Accept an :class:`InferenceProblem` in the constructor.
2. Perform inference validation during initialization.
3. Use only the vectorized posterior API.
4. Keep algorithm-specific state isolated.
5. Return a properly constructed :class:`SamplingResult`.

A minimal example might look like:

.. code-block:: python

    class RandomWalkSampler(Sampler):

        def _validate_inference_problem(self):
            self.problem.validate_for_inference()

        def run(self, n_steps=1000):

            theta = self.problem.initial_theta
            samples = []

            for _ in range(n_steps):
                proposal = theta + 0.1 * np.random.randn(len(theta))
                logp_new = self.problem(proposal)
                logp_old = self.problem(theta)

                if np.log(np.random.rand()) < logp_new - logp_old:
                    theta = proposal

                samples.append(theta)

            samples = np.array(samples)

            return SamplingResult(
                samples=samples,
                parameter_metadata=[
                    p.to_metadata()
                    for p in self.problem.parameters.values()
                ],
                sampler_class_name=self.__class__.__name__,
                likelihood_class_name=self.problem.likelihood.__class__.__name__,
                model_class_name=self.problem.model.__class__.__name__,
                data_container_class_name=self.problem.data.__class__.__name__,
                inference_problem_class_name=self.problem.__class__.__name__,
            )

While simplistic, this illustrates the core contract.

SamplingResult Objects
----------------------

All samplers must return a subclass of
:class:`SamplingResult`

This object stores:

- posterior samples,
- optional log-prior / log-likelihood / log-posterior arrays,
- parameter metadata,
- sampler and model identifiers.

For MCMC-based samplers, the
:class:`MCMCSamplingResult` subclass should be used. It expects
samples to be shaped as:

.. code-block:: text

    (n_steps, n_walkers, n_dim)

This class provides:

- flattening utilities,
- autocorrelation time estimation,
- effective sample size computation,
- Vehtari-style split R-hat,
- trace plots,
- corner plots.

If you are implementing a nested sampler or variational method,
you may instead return a plain :class:`SamplingResult`.

Convergence Diagnostics and Utilities
-------------------------------------

Common diagnostics are implemented in
``sampling/utils.py``

For example, the Gelman–Rubin statistic:

.. code-block:: python

    from triceratops.inference.sampling.utils import compute_gelman_rubin_rhat

These utilities are intentionally separated from sampler implementations
so that they can be reused across algorithms.

Metadata and Reproducibility
----------------------------

Each sampling result stores:

- sampler class name,
- likelihood class name,
- model class name,
- data container class name,
- inference problem class name.

This allows partial reproducibility and introspection even without
reconstructing the full inference stack.

Results can be serialized to HDF5 via:

.. code-block:: python

    result.to_hdf5("chains.h5")

and reconstructed using:

.. code-block:: python

    SamplingResult.from_hdf5("chains.h5")

Guidelines and Best Practices
------------------------------

When developing new samplers, keep the following principles in mind:

- Never bypass the inference problem.
- Never manipulate priors or likelihoods directly.
- Never assume parameter ordering beyond what the inference problem defines.
- Always validate before sampling.
- Keep the sampler stateless beyond algorithm state.
- Ensure returned samples have consistent dimensionality.

The sampler layer must remain numerically focused and free of
model-specific assumptions.

If these invariants are respected, new sampling strategies can be
added safely and will integrate cleanly with the broader
Triceratops inference ecosystem.
