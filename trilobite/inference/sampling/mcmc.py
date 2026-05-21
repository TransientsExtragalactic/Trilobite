"""
MCMC Sampling integrations for Trilobite.

This module provides classes and functions to perform MCMC-based inference
on astrophysical models using the Trilobite library. It includes support for
various MCMC algorithms, allowing users to estimate model parameters and their
uncertainties based on observational data.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from trilobite.utils.config import trilobite_config

from .base import Sampler
from .result import MCMCSamplingResult

if TYPE_CHECKING:
    from ..problem import InferenceProblem

import emcee


class EmceeSampler(Sampler):
    """
    MCMC sampler using the ``emcee`` library.

    This sampler utilizes the affine-invariant ensemble sampler provided by the
    ``emcee`` library to perform MCMC sampling of the posterior distribution defined
    by a given inference problem. It is designed to work seamlessly with the
    Trilobite inference framework, allowing users to easily integrate MCMC sampling
    into their analysis workflows.
    """

    # -------------------------------------- #
    # Initialization                         #
    # -------------------------------------- #
    def __init__(
        self,
        inference_problem: "InferenceProblem",
        n_walkers: int = 100,
        ensemble_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the :class:`EmceeSampler` from a given inference problem.

        This sets up the sampler with the necessary configuration to perform MCMC
        sampling on the provided inference problem using the ``emcee`` library.

        Parameters
        ----------
        inference_problem : InferenceProblem
            The inference problem to sample.
        n_walkers : int
            Number of walkers / chains to be generated.
        ensemble_kwargs : dict, optional
            Extra keyword arguments passed to `emcee.EnsembleSampler`. This can include multi-processing
            pools or other advanced configurations.
        """
        # Call the base class initializer to set the self.__problem__ attribute.
        super().__init__(inference_problem)

        # With self._problem set, we now need to store the walker info and
        # any extra ensemble kwargs. We can then generate the ensemble itself at
        # runtime.
        self.n_walkers = int(n_walkers)
        """int: Number of walkers / chains to be generated."""
        self.ensemble_kwargs = ensemble_kwargs or {}
        """dict: Extra keyword arguments passed to `emcee.EnsembleSampler`."""

    # -------------------------------------- #
    # Validation                             #
    # -------------------------------------- #
    def _validate_inference_problem(self):
        """Ensure the inference problem is compatible with emcee."""
        # First, call the base class validation.
        super()._validate_inference_problem()

        # validate the base problem
        self._problem.validate_for_inference()

        # Ensure that the problem is valid for inference.
        if self._problem.n_free_parameters <= 0:
            raise ValueError("InferenceProblem must have at least one free parameter.")

    # -------------------------------------- #
    # Sampling                               #
    # -------------------------------------- #
    def generate_initial_state(
        self,
        *,
        scale: float = 1e-3,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Generate an initial walker ensemble around the problem's initial point.

        Parameters
        ----------
        scale : float
            Fractional scale of the log-space Gaussian perturbation applied to
            each component of the initial parameter vector.  Walkers are placed
            at ``theta0 * exp(scale * N(0, 1))``, which is scale-invariant and
            works correctly for parameters that span many orders of magnitude.
            For any component whose initial value is exactly zero an absolute
            perturbation of ``scale`` is used instead.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        np.ndarray
            Initial walker positions with shape ``(n_walkers, n_dim)``.
        """
        # Grab the RNG if it is not given to us.
        rng = rng or np.random.default_rng()

        # Grab the initial theta and dimensionality
        theta0 = self.problem.initial_theta
        ndim = theta0.size

        noise = scale * rng.standard_normal(size=(self.n_walkers, ndim))

        # Use a multiplicative (log-space) perturbation so that the spread is
        # proportional to the magnitude of each parameter.  This avoids the
        # near-zero absolute perturbations that arise when theta0 contains
        # values spanning many orders of magnitude.
        nonzero = theta0 != 0.0
        positions = np.where(nonzero, theta0 * np.exp(noise), noise)

        return positions

    def run(
        self,
        n_steps: int,
        initial_state: Optional[np.ndarray] = None,
        *,
        progress: bool = True,
        pool=None,
    ) -> "MCMCSamplingResult":
        """
        Run the MCMC sampler.

        This method executes the MCMC sampling procedure using the ``emcee`` library
        on the configured inference problem. It initializes the walker ensemble,
        runs the MCMC chains for the specified number of steps, and collects the
        results into a :class:`~trilobite.inference.sampling.result.MCMCSamplingResult` object.

        Parameters
        ----------
        n_steps : int
            Number of steps per walker.
        initial_state : np.ndarray, optional
            Initial walker positions. If not provided, they will be generated
            automatically via the :meth:`generate_initial_state` method.
        progress : bool
            Whether to display the emcee progress bar.
        pool : multiprocessing pool, optional
            Pool for parallel likelihood evaluation.

        Returns
        -------
        SamplingResult
            Sampling result object containing raw chains and metadata.
        """
        # Validate the walkers and the dimensionality before
        # proceeding. Emcee places a lower limit on the number of walkers.
        ndim = self.problem.n_dim

        if self.n_walkers < 2 * ndim:
            raise ValueError(
                f"n_walkers={self.n_walkers} is too small for n_dim={ndim}. emcee requires at least 2 * ndim walkers."
            )

        # Configure the initial state of the walkers. We may be handed this
        # or we may need to generate it.
        if initial_state is None:
            initial_state = self.generate_initial_state()
        else:
            initial_state = np.asarray(initial_state)

        # Ensure that the initial state has the correct shape.
        if initial_state.shape != (self.n_walkers, ndim):
            raise ValueError(f"initial_state must have shape (n_walkers, n_dim) = ({self.n_walkers}, {ndim})")

        # Build the sampler.
        sampler = emcee.EnsembleSampler(
            self.n_walkers,
            ndim,
            self.problem._log_free_posterior,
            pool=pool,
            **self.ensemble_kwargs,
        )
        sampler.run_mcmc(
            initial_state,
            n_steps,
            progress=progress,
            progress_kwargs={
                "bar_format": trilobite_config["system.appearance.progress_bar_format"],
                "desc": "Sampling Posterior Distribution",
            },
        )

        # --- Post Processing ------------------------------------- #
        # We now need to extract the results and build the SamplingResult
        # object to return.

        # Extract the chain and log probabilities from the
        # sampler.
        chain = sampler.get_chain()  # (steps, walkers, dim)
        logp = sampler.get_log_prob()  # (steps, walkers)

        # --------------------------------------------------
        # Build result
        # --------------------------------------------------
        return MCMCSamplingResult(
            samples=chain,
            problem=self.problem,
            log_posterior=logp,
        )
