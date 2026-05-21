"""
Sampling integrations for Trilobite inference problems.

This module provides classes and functions to perform sampling-based inference
on astrophysical models using the Trilobite library. It includes support for
various sampling algorithms, such as Markov Chain Monte Carlo (MCMC) and nested
sampling, allowing users to estimate model parameters and their uncertainties
based on observational data.
"""

__all__ = [
    "base",
    "result",
    "mcmc",
    "EmceeSampler",
    "SamplingResult",
    "MCMCSamplingResult",
]

from . import base, mcmc, result
from .mcmc import EmceeSampler
from .result import MCMCSamplingResult, SamplingResult
