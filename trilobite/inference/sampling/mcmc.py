"""
MCMC Sampling integrations for Trilobite.

This module provides classes and functions to perform MCMC-based inference
on astrophysical models using the Trilobite library. It includes support for
various MCMC algorithms, allowing users to estimate model parameters and their
uncertainties based on observational data.
"""

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

from trilobite.utils.config import trilobite_config

from .base import Sampler
from .result import MCMCSamplingResult
from .utils import find_posterior_maximum

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

    Parameters
    ----------
    inference_problem : InferenceProblem
        The inference problem to sample.
    n_walkers : int
        Number of walkers / chains to be generated.
    ensemble_kwargs : dict, optional
        Extra keyword arguments passed to `emcee.EnsembleSampler`. This can include multi-processing pools or other
        advanced configurations.
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
        scale: float = 1e-2,
        optimize: bool = True,
        rng: Optional[np.random.Generator] = None,
        max_attempts: int = 100,
        from_prior: bool = False,
        n_candidates: int = 400,
    ) -> np.ndarray:
        """
        Generate an initial walker ensemble.

        By default the walkers are scattered around the problem's initial point. If ``optimize`` is set, the
        initial point is first moved to a nearby maximum of the posterior
        (see :func:`~trilobite.inference.sampling.utils.find_posterior_maximum`) and the walkers are
        scattered around that. This matters when the initial guess is poor: from a point where the
        log-posterior is, say, :math:`-10^8`, a sizeable fraction of walkers can fall into spurious
        corners of the prior and never reach the posterior bulk.

        If ``from_prior`` is set, no starting point is needed. Instead, ``n_candidates`` points are drawn from
        the priors (:meth:`~trilobite.inference.problem.InferenceProblem.sample_prior`) and the ``n_walkers``
        with the highest log-posterior are kept. This is a good choice when the priors are wide, the posterior
        is narrow and there is no reliable guess. Selecting the best candidates does not guarantee that they
        all lie in the same posterior mode: check the result for stranded walkers
        (:meth:`~trilobite.inference.sampling.result.MCMCSamplingResult.find_stranded_walkers`).

        Parameters
        ----------
        scale : float
            Fractional scale of the log-space Gaussian perturbation applied to
            each component of the starting point.  Walkers are placed
            at ``theta0 * exp(scale * N(0, 1))``, which is scale-invariant and
            works correctly for parameters that span many orders of magnitude.
            For any component whose starting value is exactly zero an absolute
            perturbation of ``scale`` is used instead. Ignored if ``from_prior`` is set.
        optimize : bool
            Whether to optimize the initial point before scattering the walkers. This costs a
            few thousand posterior evaluations. Default ``True``. Ignored if ``from_prior`` is set.
        rng : ~numpy.random.Generator, optional
            Random number generator.
        max_attempts : int
            Walkers whose proposed starting point has a non-finite log-posterior
            (e.g. outside the prior support) are redrawn up to this many times. With ``from_prior``, this is
            the maximum number of batches of ``n_candidates`` prior draws.
        from_prior : bool
            Initialize from the best of ``n_candidates`` prior draws rather than around the initial point.
            Default ``False``.
        n_candidates : int
            Number of prior draws per batch evaluated when ``from_prior`` is set. Must be at least
            ``n_walkers``. Each costs one log-posterior evaluation. Default 400.

        Returns
        -------
        numpy.ndarray
            Initial walker positions with shape ``(n_walkers, n_dim)``.

        Raises
        ------
        ValueError
            If ``from_prior`` is set and ``n_candidates < n_walkers``.
        RuntimeError
            If some walkers still have a non-finite log-posterior after ``max_attempts`` redraws, or, with
            ``from_prior``, if fewer than ``n_walkers`` finite candidates were found.
        """
        # Grab the RNG if it is not given to us.
        rng = rng if rng is not None else np.random.default_rng()

        if from_prior:
            return self._initial_state_from_prior(n_candidates, rng, max_attempts)

        # Grab the initial position and dimensionality. The log-posterior is evaluated in sampling space, so the
        # walkers must start there too (this equals ``initial_theta`` for parameters without a transform).
        theta0 = self.problem.initial_z
        if optimize:
            theta0 = find_posterior_maximum(self.problem._log_free_posterior, theta0)
        ndim = theta0.size

        # Use a multiplicative (log-space) perturbation so that the spread is
        # proportional to the magnitude of each parameter.  This avoids the
        # near-zero absolute perturbations that arise when theta0 contains
        # values spanning many orders of magnitude.
        nonzero = theta0 != 0.0

        def _draw(n: int) -> np.ndarray:
            noise = scale * rng.standard_normal(size=(n, ndim))
            return np.where(nonzero, theta0 * np.exp(noise), noise)

        positions = _draw(self.n_walkers)

        # A wide ball can step outside the prior support (e.g. near a bound), which
        # emcee cannot start from. Redraw those walkers.
        for _ in range(max_attempts):
            invalid = ~np.isfinite([self.problem._log_free_posterior(pos) for pos in positions])
            if not invalid.any():
                return positions
            positions[invalid] = _draw(int(invalid.sum()))

        raise RuntimeError(
            f"Could not place all {self.n_walkers} walkers at a finite log-posterior after {max_attempts} attempts "
            f"with scale={scale}. Reduce `scale` or move the initial values away from the prior bounds."
        )

    def _initial_state_from_prior(self, n_candidates: int, rng: np.random.Generator, max_attempts: int) -> np.ndarray:
        """Keep the ``n_walkers`` highest-posterior points among batches of prior draws."""
        n_candidates = int(n_candidates)
        if n_candidates < self.n_walkers:
            raise ValueError(f"n_candidates={n_candidates} must be at least n_walkers={self.n_walkers}.")

        positions = np.empty((0, self.problem.n_free_parameters))
        log_post = np.empty(0)

        for _ in range(max(int(max_attempts), 1)):
            candidates = self.problem.sample_prior(n_candidates, rng=rng)
            lp = np.array([self.problem._log_free_posterior(c) for c in candidates])
            finite = np.isfinite(lp)
            positions = np.vstack([positions, candidates[finite]])
            log_post = np.concatenate([log_post, lp[finite]])
            if log_post.size >= self.n_walkers:
                best = np.argsort(log_post)[::-1][: self.n_walkers]
                return positions[best]

        raise RuntimeError(
            f"Found only {log_post.size} of the {self.n_walkers} required prior draws with a finite log-posterior "
            f"after {max_attempts} batches of {n_candidates}. Increase `n_candidates` or check that the priors "
            "overlap the region where the model is finite."
        )

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
        result = MCMCSamplingResult(
            samples=chain,
            problem=self.problem,
            log_posterior=logp,
        )

        # Walkers that never reach the bulk of the posterior silently bias flat-chain statistics,
        # so flag them rather than leaving the user to discover it in a corner plot.
        stranded = result.find_stranded_walkers()
        if stranded.size:
            warnings.warn(
                f"{stranded.size} of {self.n_walkers} walkers (indices {stranded.tolist()}) are stranded far below "
                "the rest of the ensemble in log-posterior. They bias the posterior statistics; exclude them with "
                "`get_flat_samples(exclude_walkers=...)` or restart from a better initial state.",
                RuntimeWarning,
                stacklevel=2,
            )

        return result
