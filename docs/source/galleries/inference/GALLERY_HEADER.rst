.. _inference_gallery:

***************************************
Inference and Parameter Estimation
***************************************

These examples demonstrate end-to-end Bayesian parameter inference with Triceratops. Starting
from observational data (real or simulated), they walk through constructing a likelihood,
defining priors, running an MCMC sampler, and interpreting the resulting posteriors.

The :mod:`~triceratops.inference` subpackage ties a physical model to a dataset via a
:class:`~triceratops.inference.InferenceProblem`. The sampler backend (``emcee``) handles
the MCMC, and result containers provide convergence diagnostics, posterior summaries, and
corner plots.

.. rubric:: What you'll find here

- Fitting a single-epoch synchrotron SED (with and without upper limits)
- Fitting an evolving, multi-epoch SED time series
- Recovering shock physical parameters from a simulated radio light curve
- Propagating SED posteriors through closure relations to obtain source-size posteriors

.. rubric:: API reference

:mod:`triceratops.inference` — :class:`~triceratops.inference.InferenceProblem`,
:class:`~triceratops.data.InferenceData`,
:class:`~triceratops.inference.GaussianCensoredLikelihood`
