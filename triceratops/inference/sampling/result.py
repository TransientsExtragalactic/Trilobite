"""
Sampling output handling for Triceratops.

This module contains the :class:`~triceratops.inference.sampling.result.SamplingResult` class,
which is used to store and manage
the results of sampling procedures performed during model inference. It provides methods
to access and analyze the sampled parameter distributions, as well as to visualize the
results.
"""

import json
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import emcee
import h5py
import numpy as np

from .utils import compute_gelman_rubin_rhat

if TYPE_CHECKING:
    from triceratops.data.core import InferenceData
    from triceratops.models.core.base import Model

    from ..likelihood.base import Likelihood
    from ..problem import InferenceProblem


# ------------------------------------------------ #
# BASE CLASS FOR SAMPLING RESULTS                  #
# ------------------------------------------------ #
@dataclass
class SamplingResult(ABC):
    r"""
    Abstract base class representing the result of a sampling procedure.

    A :class:`~triceratops.inference.sampling.result.SamplingResult` stores posterior samples along with the
    associated :class:`~triceratops.inference.problem.InferenceProblem` required
    to fully reconstruct the inference configuration.

    The result is designed to be:

    - Fully serializable
    - Self-consistent
    - Reproducible without requiring the original sampler instance
    - Lightweight with respect to inference infrastructure

    Numerical arrays (samples and optional log-probabilities) are written
    to an HDF5 file, while structural metadata (the inference problem and
    sampler metadata) are written to a companion JSON file.

    Notes
    -----
    - The final axis of ``samples`` must correspond to parameter space
      dimensionality.
    - Subclasses may impose additional structure (e.g. MCMC chain layouts).
    - The associated :class:`InferenceProblem` contains the model, priors,
      transforms, and data required for posterior evaluation.
    """

    # ------------------------------------------------ #
    # PARAMETERS / INPUTS                              #
    # ------------------------------------------------ #
    # Various elements of the sampling result and other sampling infrastructure
    # may be included in the result object for convenience; however, care should
    # be taken to ensure that SamplingResult can be read / written to disk without
    # requiring the full sampler infrastructure to be present. This is why we don't
    # initialize on the Sampler class itself.
    samples: np.ndarray
    """numpy.ndarray
    Sampled parameter values.

    The final dimension must correspond to parameter space dimensionality.
    Subclasses may use higher-dimensional layouts (e.g., chains × steps × dim).
    """
    problem: "InferenceProblem"
    """InferenceProblem: The inference problem associated with this sampling result."""

    # --- Log-probability values --- #
    log_likelihood: Optional[np.ndarray] = None
    """numpy.ndarray: The log-likelihood values corresponding to each sample.
    This may or may not be present depending on the sampling algorithm used.
    """
    log_prior: Optional[np.ndarray] = None
    """numpy.ndarray: The log-prior values corresponding to each sample.
    This may or may not be present depending on the sampling algorithm used.
    """
    log_posterior: Optional[np.ndarray] = None
    """numpy.ndarray: The log-posterior values corresponding to each sample.
    This may or may not be present depending on the sampling algorithm used.
    """
    sampler_metadata: Optional[dict] = None
    """dict: Metadata about the sampling procedure.

    This may be quite generic, but could include information such as:

    - Sampler type (e.g., "MCMC", "Nested Sampling")
    - Sampler configuration (e.g., number of walkers, step size)
    - Inference problem metadata (e.g., parameter names, prior types)
    """

    # ------------------------------------------------ #
    # Initialization                                   #
    # ------------------------------------------------ #
    def __post_init__(self):
        self.samples = np.asarray(self.samples)

    # ------------------------------------------------ #
    # Properties                                       #
    # ------------------------------------------------ #
    @property
    def n_samples(self) -> int:
        """
        int: Total number of samples.

        For multi-dimensional layouts (e.g., MCMC chains),
        this may correspond only to the leading axis. Subclasses
        may override this behavior.
        """
        return self.samples.shape[0]

    @property
    def n_dim(self) -> int:
        """int: Dimensionality of parameter space."""
        return self.samples.shape[-1]

    @property
    def likelihood(self) -> "Likelihood":
        """Likelihood: The likelihood object associated with the underlying :class:`InferenceProblem`."""

    @property
    def model(self) -> "Model":
        """Model: The model associated with this sampling result."""
        return self.problem.likelihood._model

    @property
    def data(self) -> "InferenceData":
        """Data: The data associated with this sampling result."""
        return self.problem.likelihood._data

    @property
    def parameters(self):
        """dict: The parameters of the inference problem."""
        return self.problem.parameters

    # ------------------------------------------------ #
    # IO                                               #
    # ------------------------------------------------ #
    def to_hdf5(
        self,
        path: Union[str, Path],
        overwrite: bool = False,
    ) -> None:
        """
        Serialize the sampling result to disk.

        This method writes two files:

        1. ``<path>`` — HDF5 file containing numeric arrays:
           - ``samples``
           - ``log_likelihood`` (if present)
           - ``log_prior`` (if present)
           - ``log_posterior`` (if present)

        2. ``<path>.meta`` — JSON file containing:
           - Serialized :class:`InferenceProblem`
           - Sampler metadata
           - Result class information

        Parameters
        ----------
        path : str or Path
            Output path for the HDF5 file.
        overwrite : bool, optional
            If ``True``, overwrite an existing file. Default is ``False``.

        Raises
        ------
        FileExistsError
            If ``path`` already exists and ``overwrite`` is ``False``.

        Notes
        -----
        The JSON metadata file is required to reconstruct the full inference
        problem and must remain alongside the HDF5 file.
        """
        path = Path(path)

        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists.")

        meta_path = Path(str(path) + ".meta")

        # ----------------------------
        # Write numeric arrays
        # ----------------------------
        with h5py.File(path, "w") as f:
            f.create_dataset("samples", data=self.samples)

            if self.log_likelihood is not None:
                f.create_dataset("log_likelihood", data=self.log_likelihood)

            if self.log_prior is not None:
                f.create_dataset("log_prior", data=self.log_prior)

            if self.log_posterior is not None:
                f.create_dataset("log_posterior", data=self.log_posterior)

        # ----------------------------
        # Write metadata JSON
        # ----------------------------
        metadata = {
            "format": "SamplingResult",
            "class": self.__class__.__name__,
            "problem": self.problem.to_dict(),
            "sampler_metadata": self.sampler_metadata or {},
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def from_hdf5(
        cls,
        path: Union[str, Path],
    ) -> "SamplingResult":
        """
        Reconstruct a :class:`~triceratops.inference.sampling.result.SamplingResult` from disk.

        This method expects:

        - ``<path>`` — HDF5 file containing numeric arrays.
        - ``<path>.meta`` — JSON file containing serialized metadata.

        The inference problem is reconstructed using
        :meth:`~triceratops.inference.problem.InferenceProblem.from_dict`.

        Parameters
        ----------
        path : str or Path
            Path to the HDF5 file containing numeric sampling results.

        Returns
        -------
        SamplingResult
            Reconstructed sampling result instance. If the metadata
            specifies a subclass of :class:`~triceratops.inference.sampling.result.SamplingResult`, that subclass
            will be instantiated.

        Raises
        ------
        FileNotFoundError
            If either the HDF5 file or its metadata file is missing.
        ValueError
            If the metadata format is invalid or incomplete.

        Notes
        -----
        The metadata JSON file must be present and valid in order to
        reconstruct the associated :class:`InferenceProblem`.
        """
        path = Path(path)
        meta_path = Path(str(path) + ".meta")

        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")

        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        # --------------------------------------------------
        # Load numeric arrays
        # --------------------------------------------------
        with h5py.File(path, "r") as f:
            samples = f["samples"][...]

            log_likelihood = f["log_likelihood"][...] if "log_likelihood" in f else None
            log_prior = f["log_prior"][...] if "log_prior" in f else None
            log_posterior = f["log_posterior"][...] if "log_posterior" in f else None

        # --------------------------------------------------
        # Load metadata JSON
        # --------------------------------------------------
        with open(meta_path) as f:
            metadata = json.load(f)

        if metadata.get("format") != "SamplingResult":
            raise ValueError("Invalid SamplingResult metadata file.")

        result_class_name = metadata.get("class")
        sampler_metadata = metadata.get("sampler_metadata", {})
        problem_dict = metadata.get("problem")

        if problem_dict is None:
            raise ValueError("Metadata file missing 'problem' field.")

        # --------------------------------------------------
        # Reconstruct InferenceProblem
        # --------------------------------------------------
        from triceratops.inference import InferenceProblem

        problem = InferenceProblem.from_dict(problem_dict)

        # --------------------------------------------------
        # Determine correct result class
        # --------------------------------------------------
        result_cls = cls

        if result_class_name is not None and result_class_name != cls.__name__:
            # Attempt dynamic subclass resolution
            subclass = globals().get(result_class_name)
            if subclass is None:
                raise ValueError(f"Unknown SamplingResult subclass '{result_class_name}'.")
            result_cls = subclass

        # --------------------------------------------------
        # Construct result
        # --------------------------------------------------
        return result_cls(
            samples=samples,
            problem=problem,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
            log_posterior=log_posterior,
            sampler_metadata=sampler_metadata,
        )


# ------------------------------------------------ #
# MCMC SAMPLING RESULT                             #
# ------------------------------------------------ #
# This is a concrete implementation of SamplingResult for MCMC samplers. The
# only core difference is that we respect the chain structure of MCMC samples.
@dataclass
class MCMCSamplingResult(SamplingResult):
    r"""
    Sampling result for MCMC samplers.

    This subclass of :class:`~triceratops.inference.sampling.result.SamplingResult` stores samples with explicit
    chain structure. Samples must be provided in the form:

        (n_steps, n_walkers, n_dim)

    or

        (n_samples, n_dim)

    In the latter case, the samples are interpreted as a single-chain
    result and reshaped internally to:

        (n_samples, 1, n_dim)
    """

    acceptance_fraction: Optional[np.ndarray] = None
    """numpy.ndarray: Acceptance fraction for each walker, if available."""

    # ------------------------------------------------ #
    # Initialization                                   #
    # ------------------------------------------------ #
    def __post_init__(self):
        super().__post_init__()

        if self.samples.ndim == 2:
            n_samples, n_dim = self.samples.shape
            self.samples = self.samples.reshape((n_samples, 1, n_dim))

        elif self.samples.ndim != 3:
            raise ValueError(
                "MCMCSamplingResult samples must have shape (n_steps, n_walkers, n_dim) or (n_samples, n_dim)."
            )

        self._autocorr_time_cache: Optional[np.ndarray] = None

    # ------------------------------------------------ #
    # Core Properties                                  #
    # ------------------------------------------------ #
    @property
    def n_steps(self) -> int:
        """int: Number of MCMC steps."""
        return self.samples.shape[0]

    @property
    def n_walkers(self) -> int:
        """int: Number of MCMC walkers."""
        return self.samples.shape[1]

    @property
    def n_samples(self) -> int:
        """int: Total number of MCMC samples (n_steps × n_walkers)."""
        return self.n_steps * self.n_walkers

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Tuple[str]: Names of inferred parameters."""
        return tuple(self.problem.free_parameter_names)

    # ------------------------------------------------ #
    # Flattening                                       #
    # ------------------------------------------------ #
    def get_flat_samples(
        self,
        burn: int = 0,
        thin: int = 1,
    ) -> np.ndarray:
        """
        Return flattened MCMC samples with optional burn-in and thinning.

        Parameters
        ----------
        burn : int, optional
            Number of initial steps to discard.
        thin : int, optional
            Thinning factor.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_effective_samples, n_dim).
        """
        samples = self.samples[burn::thin, :, :]
        n_steps, n_walkers, n_dim = samples.shape

        return samples.reshape((n_steps * n_walkers, n_dim))

    # ------------------------------------------------ #
    # Goodness of Fit Statistics                       #
    # ------------------------------------------------ #
    def compute_autocorr_time(self, cache: bool = False, **kwargs) -> np.ndarray:
        r"""
        Estimate the integrated autocorrelation time for each parameter.

        This uses :func:`emcee.autocorr.integrated_time` to estimate the
        autocorrelation time :math:`\tau` for each parameter dimension.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_dim,)`` containing the estimated autocorrelation
            time for each parameter.
        cache : bool, optional
            Whether to cache the computed autocorrelation time for future reference.
        **kwargs:
            Additional keyword arguments to pass to :func:`emcee.autocorr.integrated_time`.

        Notes
        -----
        - Requires a sufficient number of effective samples to converge.
        - May raise ``emcee.autocorr.AutocorrError`` if the chain is too short.
        """
        auto_corr_time = emcee.autocorr.integrated_time(
            self.samples,
            **kwargs,
        )
        if cache:
            self._autocorr_time_cache = auto_corr_time
        return auto_corr_time

    def compute_effective_sample_size(self, use_cache: bool = True, **kwargs) -> np.ndarray:
        r"""
        Compute the effective sample size (ESS) for each parameter.

        The ESS is estimated as:

        .. math::

            \mathrm{ESS} = \frac{N}{\tau}

        where :math:`N` is the total number of MCMC samples and :math:`\tau`
        is the integrated autocorrelation time.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_dim,)`` containing the effective sample size
            for each parameter.
        use_cache : bool, optional
            Whether to use the cached autocorrelation time if available.
        **kwargs:
            Additional keyword arguments to pass to :meth:`compute_autocorr_time`
        """
        if use_cache:
            if self._autocorr_time_cache is not None:
                auto_corr_time = self._autocorr_time_cache
            else:
                auto_corr_time = self.compute_autocorr_time(cache=True, **kwargs)
        else:
            auto_corr_time = self.compute_autocorr_time(**kwargs)

        return self.n_samples / auto_corr_time

    def compute_mcse(
        self,
        burn: int = 0,
        thin: int = 1,
        use_cache: bool = True,
    ) -> np.ndarray:
        r"""
        Compute the Monte Carlo standard error (MCSE) of the posterior mean for each parameter.

        The MCSE is estimated as:

        .. math::

            \mathrm{MCSE} = \frac{\sigma}{\sqrt{\mathrm{ESS}}}

        where :math:`\sigma` is the posterior standard deviation and :math:`\mathrm{ESS}` is the effective sample
        size accounting for autocorrelation.

        Parameters
        ----------
        burn : int, optional
            Number of initial steps to discard as burn-in.
        thin : int, optional
            Thinning factor applied to the chain.
        use_cache : bool, optional
            Whether to use cached autocorrelation time / ESS if available.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_dim,)`` containing the MCSE for each parameter.
        """
        # Flatten samples after burn-in / thinning
        flat_samples = self.get_flat_samples(burn=burn, thin=thin)

        # Posterior standard deviation
        posterior_sd = np.std(flat_samples, axis=0, ddof=1)

        # Effective sample size
        ess = self.compute_effective_sample_size(use_cache=use_cache)

        return posterior_sd / np.sqrt(ess)

    def compute_relative_mcse(
        self,
        burn: int = 0,
        thin: int = 1,
        use_cache: bool = True,
    ) -> np.ndarray:
        r"""
        Compute the relative Monte Carlo standard error (MCSE) of the posterior mean for each parameter.

        The MCSE of the posterior mean is:

        .. math::

            \mathrm{MCSE} = \frac{\sigma}{\sqrt{\mathrm{ESS}}}

        where :math:`\sigma` is the posterior standard deviation and
        :math:`\mathrm{ESS}` is the effective sample size.

        The relative MCSE is defined as:

        .. math::

            \frac{\mathrm{MCSE}}{\sigma} = \frac{1}{\sqrt{\mathrm{ESS}}}

        This provides a dimensionless measure of Monte Carlo error relative to
        posterior uncertainty.

        Parameters
        ----------
        burn : int, optional
            Number of initial steps to discard as burn-in.
        thin : int, optional
            Thinning factor applied to the chain.
        use_cache : bool, optional
            Whether to use cached autocorrelation time / ESS if available.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_dim,)`` containing the relative MCSE for each parameter.
        """
        ess = self.compute_effective_sample_size(use_cache=use_cache)
        return 1.0 / np.sqrt(ess)

    def compute_vehtari_rhat(
        self,
        burn: int = 0,
    ) -> np.ndarray:
        r"""
        Compute the rank-normalized, split R-hat diagnostic following Vehtari et al. (2021).

        This diagnostic assesses MCMC convergence by comparing within-chain and between-chain
        variances after rank-normalizing the samples. It is more robust to non-normality and
        heavy tails than the traditional Gelman-Rubin R-hat. For more detail, see
        :footcite:t:`vehtari2021rank`.

        This implementation performs:

        1. Chain splitting
        2. Rank normalization
        3. Bulk R-hat computation
        4. Tail folding
        5. Tail R-hat computation

        The final R-hat is the maximum of bulk and tail R-hat for each parameter.

        Parameters
        ----------
        burn : int, optional
            Number of initial steps to discard as burn-in.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_dim,)`` containing Vehtari R-hat values.
        """
        from scipy.stats import norm, rankdata

        # Perform the burn-in step and discard initial samples
        samples = self.samples[burn:, :, :]  # (steps, chains, dim)
        n_steps, n_chains, n_dim = samples.shape

        if n_steps < 4:
            raise ValueError("Not enough samples to compute split R-hat.")

        # Perform the chain splitting step of the analysis. We split each chain
        # in half to create 2 * n_chains shorter chains. To ensure all chains
        # are of equal length, we may need to discard the last sample if n_steps
        # is odd.
        half_n_steps = n_steps // 2
        samples = samples[: 2 * half_n_steps, :, :]  # enforce even split

        # Construct the first and second half cuts.
        first_half_chains = samples[:half_n_steps, :, :]
        second_half_chains = samples[half_n_steps:, :, :]

        # Now restack the chains to create the split chain array.
        split_samples = np.concatenate([first_half_chains, second_half_chains], axis=1)
        # shape: (half, 2 * chains, dim)

        n, m, _ = split_samples.shape

        # We now perform rank normalization across all chains for each parameter. This means
        # that for each parameter dimension, we replace the samples with their ranks
        # across all chains, and then transform these ranks to standard normal quantiles.
        flat = split_samples.reshape(-1, n_dim)  # (n * m, dim)

        z = np.empty_like(flat)
        for d in range(n_dim):
            ranks = rankdata(flat[:, d], method="average")
            z[:, d] = norm.ppf((ranks - 0.5) / len(ranks))

        z = z.reshape(n, m, n_dim)

        # Compute the bulk r_hat by processing the rank-normalized samples.
        rhat_bulk = compute_gelman_rubin_rhat(z)

        # Perform the rhat tail folding step. We compute the median
        # of the rank-normalized samples for each parameter, and then fold
        # the samples about this median.
        z_median = np.median(z, axis=(0, 1))
        z_folded = np.abs(z - z_median)

        rhat_tail = compute_gelman_rubin_rhat(z_folded)

        # Take the maximum of the two.
        return np.maximum(rhat_bulk, rhat_tail)

    # -------------------------------------------------- #
    # Plotting Methods                                   #
    # -------------------------------------------------- #
    def trace_plot(
        self,
        parameter: str,
        burn: int = 0,
        thin: int = 1,
        fig=None,
        axes=None,
        **kwargs,
    ):
        """
        Plot the trace (chain evolution) for a single parameter.

        Parameters
        ----------
        parameter : str
            Name of the parameter to plot. This must be a valid
            parameter name in the inference problem.
        burn : int, optional
            Number of initial steps to discard as burn-in. By default,
            this is 0.
        thin : int, optional
            Thinning factor to apply to the samples. By default, this is 1
            (no thinning).
        fig : matplotlib.figure.Figure, optional
            Existing figure to plot into.
        axes : matplotlib.axes.Axes, optional
            Existing axes to plot into.
        **kwargs: Additional keyword arguments passed to the plotting function.

        Returns
        -------
        fig, ax
            The matplotlib figure and axes.
        """
        from triceratops.utils.plot_utils import (
            get_cmap,
            resolve_fig_axes,
            set_plot_style,
        )

        # Ensure that the plot style is enabled.
        set_plot_style()

        # Ensure that the parameter is in the list and obtain
        # its index so that we can extract the samples.
        if parameter not in self.parameter_names:
            raise ValueError(f"Unknown parameter '{parameter}'")
        pidx = self.parameter_names.index(parameter)

        # Extract samples
        samples = self.samples[burn::thin, :, pidx]  # (steps, walkers)
        n_steps, n_walkers = samples.shape

        # Resolve the figure and the axes.
        fig, axes = resolve_fig_axes(fig, axes)

        # Posterior mean (flattened)
        flat = samples.reshape(-1)
        axes.axhline(
            np.mean(flat),
            color=kwargs.pop("mean_color", "black"),
            lw=kwargs.pop("mean_lw", 1.5),
            linestyle=kwargs.pop("mean_ls", "--"),
            label="mean",
            zorder=10,
        )

        # Handle the color of the walkers. By default, we'll use a colormap
        # to differentiate them (tab20). If kwargs['cmap'] is provided, we use that instead.
        cmap = kwargs.pop("cmap", "tab20")
        cmap = get_cmap(cmap)

        # Determine the color we're going to assign to each of the
        # walkers.
        colors = [cmap(i / n_walkers) for i in range(n_walkers)]

        # Cycle through each of the walkers and plot their traces.
        for _walker in range(n_walkers):
            axes.plot(
                np.arange(n_steps),
                samples[:, _walker],
                color=colors[_walker],
                **kwargs,
            )

        axes.set_xlabel("Step")
        axes.set_ylabel(parameter)
        axes.set_title(f"Trace plot: {parameter}")

        axes.legend(frameon=False)

        return fig, axes

    def corner_plot(
        self,
        burn: int = 0,
        thin: int = 1,
        parameters: Optional[list[str]] = None,
        truths: Optional[dict[str, float]] = None,
        fig=None,
        **kwargs,
    ):
        """
        Generate a corner plot of the posterior samples.

        Parameters
        ----------
        burn : int, optional
            Number of initial steps to discard as burn-in.
        thin : int, optional
            Thinning factor applied to the samples for visualization only.
        parameters : list of str, optional
            Subset of parameter names to include in the plot. If None,
            all parameters are plotted.
        truths : dict, optional
            Dictionary mapping parameter names to reference values
            (e.g. injected or true values) to overlay on the plot.
        fig : matplotlib.figure.Figure, optional
            Existing figure to draw into.
        **kwargs
            Additional keyword arguments passed to `corner.corner`.

        Returns
        -------
        fig
            The matplotlib figure containing the corner plot.
        """
        import corner

        from triceratops.utils.plot_utils import set_plot_style

        # Ensure plotting style consistency
        set_plot_style()

        # Resolve which parameters to plot
        if parameters is None:
            parameters = list(self.parameter_names)
        else:
            for p in parameters:
                if p not in self.parameter_names:
                    raise ValueError(f"Unknown parameter '{p}'")

        # Indices of selected parameters
        pidx = [self.parameter_names.index(p) for p in parameters]

        # Extract flattened samples
        flat_samples = self.get_flat_samples(burn=burn, thin=thin)
        flat_samples = flat_samples[:, pidx]

        # Resolve truth values in correct order
        truth_values = None
        if truths is not None:
            truth_values = [truths.get(p, None) for p in parameters]

        # Generate the corner plot

        fig = corner.corner(
            flat_samples,
            labels=parameters,
            truths=truth_values,
            fig=fig,
            show_titles=True,
            title_fmt=kwargs.pop("title_fmt", ".3g"),
            quantiles=kwargs.pop("quantiles", [0.16, 0.5, 0.84]),
            levels=kwargs.pop("levels", (0.68, 0.95)),
            **kwargs,
        )

        return fig
