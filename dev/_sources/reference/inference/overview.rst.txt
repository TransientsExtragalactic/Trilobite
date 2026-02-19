.. _inference:
========================================
Parameter Inference and Model Comparison
========================================

Triceratops is designed to seamlessly connect the physical models defined in
:mod:`models` with modern Bayesian inference pipelines. In practice, this means
you can take any compatible model and:

- perform **parameter estimation** using MCMC or nested sampling,
- compute **posterior distributions** and credible intervals,
- carry out **Bayesian model comparison**,
- and explore structured parameter spaces with custom priors and constraints.

The :mod:`inference` subpackage provides a flexible and modular framework for
constructing and running inference analyses. It supports:

- integration with third-party samplers such as :mod:`emcee`,
  :mod:`dynesty`, and :mod:`bilby`,
- user-defined likelihood functions and prior distributions,
- consistent parameter management and unit handling,
- and configurable workflows tailored to specific scientific goals.

The overall design philosophy is separation of concerns:

- **Models** describe the physics.
- **Likelihoods** describe the statistical assumptions.
- **Inference problems** combine models, data, and priors into a sampler-ready object.
- **Samplers** explore the resulting posterior distribution.

This modular structure allows users to move cleanly from forward modeling
to statistically rigorous parameter estimation and model comparison,
without rewriting core logic for each new scientific application.

The Triceratops Inference Pipeline
-----------------------------------

.. graphviz:: ../../images/inference/inference_diagram.dot

----

As with any robust statistical pipeline, there are a lot of options and configurations that make the inference
pipeline look more complicated than it actually is. Before introducing any of the specifics about inference,
its worth taking a step back and looking at the overall structure of the inference pipeline in Triceratops.
The diagram above provides a high-level overview of the inference pipeline in Triceratops. We'll discuss them
step by step:

1. **The Model**: The first step in any inference pipeline in Triceratops is the model. This can either be
   a model that's already built into the :mod:`models` module, or it can be a custom model. In effect, this provides
   a mapping from some input variables :math:`{\bf x}` and some set of parameters :math:`\boldsymbol{\Theta}` to
   some set of observables :math:`{\bf y}`:

   .. math::

         {\bf y} = \mathcal{M}({\bf x}; \boldsymbol{\Theta})

   The parameters of each model are defined by the model class.

2. **The Dataset**: With the model in hand, we also need the data we want to fit too. The idea is to find the
   parameters of the model which best predict the dataset. Thus, the dataset provides a set of observed
   :math:`{\bf x}_{\rm obs}` and :math:`{\bf y}_{\rm obs}` values. These are typically loaded using the
   :mod:`data` module.

3. **The Likelihood Function**: The likelihood function quantifies how well the model predictions match
   the observed data for a given set of parameters. In Triceratops, these are :class:`inference.likelihood.base.Likelihood`
   objects, which provide a function :math:`\mathcal{L}(\boldsymbol{\Theta} | {\bf x}_{\rm obs}, {\bf y}_{\rm obs})` that
   computes the likelihood of the observed data given the model parameters. Triceratops provides a variety of
   built-in likelihood functions, and users can also define custom likelihoods as needed.

4. **Inference Problems**: The likelihood alone isn't enough to completely specify an inference problem. We also need
   to provide priors on the model parameters. In Triceratops, an inference problem is defined by a combination of a
   model, a dataset, a likelihood function, and a set of priors. This is encapsulated in the
   :class:`~inference.problem.InferenceProblem`.

5. **Samplers**: With the inference problem defined, we can now use a sampling algorithm to explore the parameter
   space and estimate the posterior distribution of the model parameters. Triceratops integrates with several
   third-party sampling libraries, including :mod:`emcee`, :mod:`dynesty`, and :mod:`bilby`. Each of these
   samplers has its own strengths and weaknesses, and Triceratops provides a unified interface for using them through the
   :class:`~inference.sampling.base.Sampler` class and its subclasses.

6. **Results**: After running the sampler, we obtain a set of samples from the posterior distribution of the
   model parameters. These samples can be analyzed to compute summary statistics, generate plots, and perform
   model comparison. Triceratops provides tools for working with the results of inference analyses, which are specific
   to each sampler used.

This is the standard workflow for performing inference in Triceratops. The following sections will provide more details
on each of these components, along with examples of how to use them in practice.

----
.. _likelihoods:

Likelihoods
-----------

*Module:* :mod:`inference.likelihood`

The likelihood is where the statistical assumptions of the inference problem are encoded. It defines how well
the model predictions match the observed data for a given set of parameters. In Triceratops,
likelihoods are implemented as structured objects that bind together a model,
a validated numerical dataset, and a noise/statistical assumption.

The Likelihood Class
^^^^^^^^^^^^^^^^^^^^

.. hint::

        For more information about likelihood development, implementation details,
        and architectural principles, see :ref:`likelihood_dev`.

In Triceratops, **likelihood functions** quantify how well a physical model reproduces a
dataset under a particular statistical noise model. Conceptually, a likelihood defines

.. math::

    \mathcal{L}(\boldsymbol{\Theta}\mid \mathcal{D}),

the probability of observing the dataset :math:`\mathcal{D}` given model parameters
:math:`\boldsymbol{\Theta}`.

Rather than treating likelihoods as opaque black boxes, Triceratops implements them as
**structured objects** that explicitly bind together:

- a **model** (from :mod:`models`),
- an :class:`~triceratops.data.core.InferenceData` object (from :mod:`data`),
- and a **noise/statistical assumption** (implemented by the likelihood subclass).

All likelihoods inherit from :class:`~inference.likelihood.base.Likelihood`.

.. currentmodule:: inference.likelihood.base

Creating a Likelihood Object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A likelihood object is constructed by combining a model instance with an
:class:`InferenceData` object. In most workflows, you will construct the likelihood once
and reuse it repeatedly during sampling.

A typical initialization pattern looks like:

.. code-block:: python

    from triceratops.models import MyModel
    from triceratops.data import InferenceData
    from triceratops.inference.likelihood import GaussianLikelihood

    model = MyModel(...)

    inference_data = InferenceData.from_table(
        model,
        table,
        variables={"time": "time"},
        observables={"flux": ("flux", "flux_err", None, None)},
    )

    like = GaussianLikelihood(model=model, data=inference_data)

During initialization, likelihoods perform three core operations:

1. **Compatibility validation**

   The likelihood verifies that the model and dataset are structurally compatible.
   For example, a Gaussian likelihood requires uncertainties to be present.
   Validation failures are intentionally eager — it is better to fail at construction
   time than after thousands of sampler calls.

2. **Numerical array extraction**

   The :class:`InferenceData` object already stores validated, stacked numerical arrays.
   The likelihood extracts and caches those arrays once during
   :meth:`Likelihood._configure`.

3. **Preparation for fast evaluation**

   Any masks (e.g., for censored data), stacked observable arrays, or frequently
   accessed structures are prepared once during initialization.
   The goal is that likelihood evaluation inside a sampling loop performs no
   structural work beyond model evaluation and lightweight stacking.

The Likelihood Function
~~~~~~~~~~~~~~~~~~~~~~~

Likelihood evaluation is split into two layers: a public wrapper and a
performance-oriented backend.

.. tab-set::

    .. tab-item:: High-Level API

        The high-level interface is :meth:`Likelihood.log_likelihood`. This method is designed
        to be called by user code. It accepts model parameters in any format supported by the model,
        performs coercion into the model’s internal representation, and then evaluates the likelihood.

        .. code-block:: python

            lnL = like.log_likelihood(
                {
                    "theta_E": 0.12,
                    "epsilon_B": 1e-2,
                    "n0": 0.5,
                }
            )

        This is the recommended interface for interactive use or exploratory work.
        If you are evaluating the likelihood inside a tight sampling loop,
        you may prefer the low-level API to avoid repeated parameter coercion.

    .. tab-item:: Low-Level API

        The low-level backend is :meth:`Likelihood._log_likelihood`. This method is written
        for performance. It assumes that:

        - parameters are already coerced into the model’s raw internal format,
        - data arrays have already been extracted and cached during initialization.

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

        The pattern is always the same:

        1. Evaluate the forward model.
        2. Stack outputs to match the observable array shape.
        3. Delegate to a numerical backend.
        4. Return a scalar log-likelihood.

        No unit conversion, table parsing, or structural validation occurs here.

While it is not common to need to implement a custom likelihood, we recognize that some users
may wish to do so. For that purpose, see the developer guide:
:ref:`likelihood_dev`.

Existing Likelihoods
^^^^^^^^^^^^^^^^^^^^

In most cases, a likelihood function has already been implemented for your use case.
Below are the likelihood classes currently available in Triceratops.

.. currentmodule:: triceratops.inference.likelihood

.. autosummary::
    :toctree:

    base.Likelihood
    base.GaussianLikelihood
    base.GaussianCensoredLikelihood

----

Inference Problems
------------------

.. hint::

    For more information about inference problem development, implementation details, and advanced
    parameter management, see :ref:`inference_problem_dev`.

Once you have defined a **model**, selected a **dataset**, and constructed a
**likelihood function**, the next step is to combine these components into an
**inference problem**.

An inference problem represents a *fully specified statistical model* ready to
be explored by a sampler. It bundles together:

- the **physical model** to be evaluated,
- the **likelihood** describing how model predictions are compared to data,
- the **set of parameters** to be inferred,
- and the **priors, bounds, and transformations** that define the allowed
  parameter space.

In Triceratops, this functionality is encapsulated by the
:class:`~inference.problem.InferenceProblem` class.

Conceptually, an inference problem defines the posterior distribution

.. math::

    \log p(\boldsymbol{\Theta} \mid \mathcal{D})
    =
    \log \mathcal{L}(\boldsymbol{\Theta} \mid \mathcal{D})
    +
    \log p(\boldsymbol{\Theta}),

and provides a **standardized interface** for evaluating the log-prior,
log-likelihood, and log-posterior in a way that is compatible with a wide range
of sampling algorithms.

A key design principle in Triceratops is that **samplers never interact directly
with models or data containers**. Instead, they operate exclusively on an
:class:`~inference.problem.InferenceProblem`, which exposes a clean,
validated, and performance-oriented API for statistical evaluation.

The Parts of an Inference Problem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An inference problem in Triceratops is intentionally designed to be a **single point of
coordination** for all objects and assumptions required to perform statistical inference.
Rather than passing models, datasets, priors, and parameter metadata separately to a sampler,
all of these components are collected and validated in one place.

Specifically, an :class:`~inference.problem.InferenceProblem` defines and manages:

- **The model** (:attr:`~inference.problem.InferenceProblem.model`)

  A physical or phenomenological model that maps inputs (e.g., time, frequency)
  and parameters :math:`\boldsymbol{\Theta}` to predicted observables. The model
  determines *what* is being fit.

- **The likelihood** (:attr:`~inference.problem.InferenceProblem.likelihood`)

  A :class:`~inference.likelihood.base.Likelihood` instance that encodes
  *how* model predictions are compared to the data, including assumptions about
  measurement uncertainties, noise properties, and censoring.

- **The data** (:attr:`~inference.problem.InferenceProblem.data`)

  The dataset itself, provided indirectly through the likelihood via a validated
  data container (e.g., a light curve or photometry table). The inference problem
  does not manipulate raw data directly, but ensures it is consistently tied to
  the likelihood.


- **The parameters** (:attr:`~inference.problem.InferenceProblem.parameters`)

  A structured collection of inference parameters, including:

  - parameter names and current values,
  - prior distributions,
  - bounds and constraints,
  - fixed vs. free parameters,
  - optional transformations between *physical* and *sampling* parameter spaces.

  These are typically represented internally using
  :class:`~inference.problem.InferenceParameter` objects.

All of these components are defined, validated, and stored within the
:class:`~inference.problem.InferenceProblem` class. This ensures that
once an inference problem is constructed, it represents a **self-consistent and
sampler-ready statistical model**.

By centralizing this information, Triceratops enables:

- clean separation between physics, statistics, and numerics,
- reproducible inference configurations,
- and seamless interoperability with multiple sampling backends.

Creating Inference Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An :class:`~inference.problem.InferenceProblem` is typically created
by supplying a fully constructed likelihood object (:class:`~inference.likelihood.base.Likelihood`).
In most workflows, this is the *only* required input.

Because the likelihood **already binds together the model, data, and statistical
assumptions**, the inference problem can infer everything it needs from this
single object.

A minimal example looks like:

.. code-block:: python

    from triceratops.inference.problem import InferenceProblem
    from triceratops.inference.likelihood import GaussianLikelihood
    from triceratops.models import MyModel
    from triceratops.data import RadioLightCurveContainer

    # Construct model and data
    model = MyModel(...)
    data = RadioLightCurveContainer.from_file(
        "lightcurve.fits",
        frequency=6.0,  # GHz if unitless
    )

    # Build likelihood
    likelihood = GaussianLikelihood(
        model=model,
        data=data,
    )

    # Create inference problem
    problem = InferenceProblem(likelihood)

At initialization time, the inference problem performs several tasks:

- The object will **register the likelihood** and pull all the relevant parameters therein
  so that it knows about the data, the model, etc.
- Using the model (:attr:`~models.core.base.Model`), it will **extract parameters** and set
  up the internal :class:`~inference.problem.InferenceParameter` objects for the problem.
- validates that priors, bounds, and parameter names are consistent,
- prepares the problem for use by sampling backends.

In most cases, no additional arguments are required. Optional keyword arguments
may be used to override default behavior (e.g., parameter initialization or
validation settings), but these are typically unnecessary for standard workflows.

Once constructed, the inference problem serves as the **single source of truth**
for everything required during sampling: parameters, priors, likelihood
evaluation, and bookkeeping.

Inference Parameters
^^^^^^^^^^^^^^^^^^^^^

Once an :class:`~triceratops.inference.problem.InferenceProblem` has been created,
it exposes a unified **parameter interface** that controls how the model is
evaluated during inference.

Inference parameters represent the quantities over which the posterior
distribution is defined. Each parameter may have:

- an initial value,
- a prior distribution,
- optional bounds,
- and a status indicating whether it is free or fixed.
- A transformation and inverse transformation if supported by the sampler.

Together, these properties define the *search space* explored by the sampler.

Accessing Parameters
~~~~~~~~~~~~~~~~~~~~~

Once an inference problem has been created, its parameters can be accessed
via the :attr:`~inference.problem.InferenceProblem.parameters` attribute. For
example, given a model with a ``B`` parameter, we can access the corresponding
:class:`~inference.problem.InferenceParameter` like so:

.. code-block::

    inf_prob = InferenceProblem(likelihood)
    B_param = inf_prob.parameters["B"]

Likewise, parameters can be accessed by indexing

.. code-block::

    info_prob = InferenceProblem(likelihood)
    B_param = inf_prob.parameters['B']

Parameters also have :class:`dict`-like methods for iteration, listing keys,
and checking membership:

- :meth:`~inference.problem.InferenceProblem.keys`
- :meth:`~inference.problem.InferenceProblem.values`
- :meth:`~inference.problem.InferenceProblem.items`

Setting Initial Values
~~~~~~~~~~~~~~~~~~~~~~

Each parameter in an inference problem has an associated initial value, which
determines where the sampler starts exploring the parameter space. By default, these initial values
are taken from the underlying model at the time the inference problem is created. However, users may wish to override these defaults
to improve convergence or explore different regions of parameter space.

To see a parameter's initial value, access the :attr:`~inference.problem.InferenceParameter.initial_value` attribute:

.. code-block::

    B_param = inf_prob.parameters["B"]
    print("Initial value of B:", B_param.initial_value)

.. hint::

    The :attr:`~inference.problem.InferenceParameter.initial_value` attribute is a ``float`` representing the
    initial value of the parameter in physical units (i.e., the units used by the model). To see the corresponding
    unit-bearing quantity, use the :attr:`~inference.problem.InferenceParameter.initial_quantity` attribute instead.

You can also set a new initial value by assigning to this attribute:

.. code-block::

    B_param.initial_value = 0.5  # Set new initial value for B

This will also automatically coerce units for you, so you can also do:

.. code-block::

    from astropy import units as u
    B_param.initial_quantity = 0.5 * u.G  # Set new initial value for B with units


Setting Priors
~~~~~~~~~~~~~~

.. hint::

    For more information about priors, including custom priors, see :ref:`priors_dev`.

A critical component of every inference problem is the **prior distribution** assigned
to each parameter. Priors encode our knowledge (or lack thereof) about plausible
parameter values before considering the data. Before you'll be able to perform inference,
**all free parameters** must have a valid prior assigned.

There are a number of ways to assign priors; however, the easiest is to use the
:class:`~inference.problem.InferenceProblem` API directly, which provides the
:meth:`~inference.problem.InferenceProblem.set_prior` method.

.. code-block::

    inf_prob = InferenceProblem(likelihood)

    # Set a uniform prior on parameter 'B' between 0.1 and 1.0
    inf_prob.set_prior('B', 'uniform', lower=0.1*u.G, upper=1.0*u.G)

When you use this method, the inference problem will automatically validate that the prior
is compatible with the parameter (e.g., units, bounds, etc.) and will raise an error if not. It
will also **automatically convert parameter units** as needed, so you can specify bounds
using any compatible unit.

.. important::

    This is the **only unit aware method** for setting priors on parameters.

An alternative approach is to simply set the :attr:`~inference.problem.InferenceParameter.prior` attribute
directly. However, when doing so, you are responsible for ensuring that the prior is compatible
with the parameter (e.g., units, bounds, etc.). For example:

.. code-block::

    from triceratops.inference.priors import UniformPrior

    B_param = inf_prob.parameters['B']
    B_param.prior = UniformPrior(lower=0.1, upper=1.0) # In gauss because those are the base units.

All priors are instances of :class:`~inference.prior.Prior`, and Triceratops provides a variety of
built-in prior classes for common distributions (uniform, log-uniform, normal, etc.). It is also
possible to define custom priors and to implement new prior classes; see :ref:`priors_dev` for more details.

Freezing and Free Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, all parameters in an inference problem are considered **free**, meaning that
the sampler is allowed to vary them during inference. However, in some cases, you may wish to
**freeze** certain parameters, keeping them fixed at their initial values throughout the sampling
process. This can be useful for testing, debugging, or when certain parameters are known
to be well-constrained by prior knowledge.

To freeze a parameter, set its :attr:`~inference.problem.InferenceParameter.freeze` attribute to ``True``:

.. code-block::

    B_param = inf_prob.parameters['B']
    B_param.freeze = True  # Freeze parameter B

Alternatively, you can use the :meth:`~inference.problem.InferenceProblem.freeze_parameters`.

Computing the Prior and Posterior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once an inference problem has been fully specified (i.e., all free parameters
have priors and initial values), it provides a consistent interface for
evaluating the **log-prior**, **log-likelihood**, and **log-posterior**.

Triceratops exposes *two complementary APIs* for probability evaluation:

- a **high-level, dictionary-based interface** intended for user interaction
  and debugging,
- and a **low-level, vectorized interface** designed for samplers and numerical
  optimizers.

These APIs are fully consistent with one another and differ only in how
parameters are represented.

.. tab-set::

    .. tab-item:: Dictionary-Based API (User-Facing)

        This interface accepts parameter values as **dictionaries**, keyed by
        parameter name. It is the most readable and safest way to evaluate
        probabilities by hand.

        .. code-block:: python

            params = {
                "B": 0.3 * u.G,
                "n0": 1.0 / u.cm**3,
                "epsilon_e": 0.1,
            }

            logp  = inf_prob.log_prior(params)
            logl  = inf_prob.log_likelihood(params)
            logpost = inf_prob.log_posterior(params)

        Under the hood, these methods:

        - coerce unit-bearing quantities into model base units,
        - insert frozen parameters automatically,
        - pack parameters into the model-defined order,
        - and dispatch to the optimized internal backend.

        This interface is ideal for:

        - sanity checks,
        - debugging likelihood behavior,
        - inspecting priors and posteriors interactively.

        .. note::

            All dictionary-based methods expect **full parameter dictionaries**.
            Frozen parameters may be included or omitted — their stored values
            will be used automatically.

    .. tab-item:: Vectorized API (Sampler-Facing)

        Samplers and optimizers operate most efficiently on **NumPy arrays**.
        For this reason, :class:`~inference.problem.InferenceProblem` provides a low-level API that
        works directly with parameter vectors.

        In this representation:

        - parameters are ordered according to the underlying model definition,
        - only **free parameters** are included,
        - all values are plain floats in base units.

        The core entry point is calling the inference problem itself:

        .. code-block:: python

            theta = inf_prob.initial_theta
            logpost = inf_prob(theta)

        This is equivalent to calling:

        .. code-block:: python

            logpost = inf_prob._log_free_posterior(theta)

        Internally, the inference problem will:

        - unpack the free parameter vector,
        - reinsert frozen parameters,
        - evaluate the log-prior and log-likelihood,
        - and return their sum.

        This interface is used automatically by all built-in samplers.

Parameter Packing and Unpacking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To support seamless transitions between user-friendly dictionaries and
sampler-friendly vectors, :class:`~inference.problem.InferenceProblem` provides explicit utilities
for **packing** and **unpacking** parameters.

Packing Full Parameter Sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The methods :meth:`~inference.problem.InferenceProblem.pack_parameters` and
:meth:`~inference.problem.InferenceProblem.unpack_parameters` convert between
dictionaries and full parameter vectors (including frozen parameters).

.. code-block:: python

    full_params = {
        name: p.initial_value
        for name, p in inf_prob.parameters.items()
    }

    theta = inf_prob.pack_parameters(full_params)
    params_back = inf_prob.unpack_parameters(theta)

Packing Free Parameters Only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Samplers operate only on **free parameters**. The inference problem therefore
provides dedicated helpers:

- :meth:`~inference.problem.InferenceProblem.pack_free_parameters`
- :meth:`~inference.problem.InferenceProblem.unpack_free_parameters`

.. code-block:: python

    theta_free = inf_prob.pack_free_parameters({
        name: p.initial_value
        for name, p in inf_prob.free_parameters.items()
    })

    full_params = inf_prob.unpack_free_parameters(theta_free)

These methods guarantee that:

- frozen parameters are inserted consistently,
- parameter ordering is unambiguous,
- dimensionality matches what the sampler expects.

.. important::

    Samplers should **never** need to reason about frozen parameters,
    unit conversions, or parameter names. All such bookkeeping is handled
    exclusively by the :class:`inference.problem.InferenceProblem`.

Initial Probability Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For convenience, the inference problem exposes properties that allow quick
inspection of the initial configuration:

.. code-block:: python

    inf_prob.initial_theta
    inf_prob.initial_log_posterior

These are useful for:

- validating that the problem is well-posed,
- checking prior support at the starting point,
- debugging sampler failures before runtime.

Samplers
--------

.. hint::

    For more information about sampler development, implementation, and integration, see :ref:`samplers_dev`.

What is a sampler: a way to explore the parameter space and find the minimum of the posterior distribution.
Triceratops provides interfaces to several popular sampling libraries, including :mod:`emcee`, :mod:`dynesty`, and :mod:`bilby`.
