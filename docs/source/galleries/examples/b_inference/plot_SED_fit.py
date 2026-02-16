"""
====================================
Fit Radio Photometry to a BPL
====================================

In this example, we'll do the simplest possible SED fit to radio photometry data using a broken power-law (BPL) model.
This is an excellent place to start for understanding how to fit models to data using the Triceratops library.

We'll use the :class:`~models.emission.synchrotron.Synchrotron_SSA_SBPL_SED` model to generate some synthetic
data with a fixed noise threshold and then we'll invert that data to recover the original parameters using MCMC.
This will take us through the basic steps of generating the model, defining the dataset, setting up the likelihood,
and running the inference with correct priors.
"""

import numpy as np
from astropy import units as u

# %%
# Setup
# -----
# First, we need to import the necessary libraries
from triceratops.models.SEDs.synchrotron import Synchrotron_SSA_SBPL_Model

# %%
# Now, the :class:`~models.SEDs.synchrotron.Synchrotron_SSA_SBPL_Model` model produces a synchrotron SED
# with the form:
#
# .. math::
#
#    F_{\nu} = F_{\nu,0}
#    \left[
#        \left( \frac{\nu}{\nu_{\rm break}} \right)^{\alpha_{\rm thick}/s}
#        +
#        \left( \frac{\nu}{\nu_{\rm break}} \right)^{\alpha_{\rm thin}/s}
#    \right]^s,
#
# where :math:`F_{\nu,0}` is the normalization at the break frequency
# :math:`\nu_{\rm break}`. The spectral indices are tied to the electron energy
# distribution power-law index :math:`p` via
#
# .. math::
#
#    \alpha_{\rm thick} = \frac{5}{2},
#    \qquad
#    \alpha_{\rm thin} = -\frac{p - 1}{2}.
#
# This choice reproduces the canonical optically thick and optically thin synchrotron
# spectral slopes expected for a homogeneous emitting region with a power-law electron
# population.
#
# In order to produce the synthetic data, we'll first define a set of model parameters and then forward model their
# SED with a Gaussian noise with a standard deviation proportional to the flux density.

# Generate the forward model object.
sed_model = Synchrotron_SSA_SBPL_Model()

# Create a parameter dictionary with our preferred true values.
true_params = {"norm": 5.0 * u.mJy, "nu_break": 4.0 * u.GHz, "p": 3.0, "s": -1.0}

# Define the noise level as a fraction of the flux density.
noise_fraction = 0.1  # 5% noise

# Generate the frequencies to use for the synthetic data.
frequencies = u.Quantity([0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0], u.GHz)

# Generate the synthetic flux densities with noise.
synthetic_flux = sed_model.forward_model({"frequency": frequencies}, true_params).flux_density
synthetic_flux += (
    np.random.normal(size=synthetic_flux.shape, scale=noise_fraction * synthetic_flux.to_value("mJy")) * u.mJy
)

# %%
# Let's go ahead and look at the plot of the true model and the synthetic data points.
import matplotlib.pyplot as plt

# Create a figure and axis for the plot.
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the true model.
freqs_plot = np.logspace(8, 11, 100) * u.Hz
true_flux_plot = sed_model.forward_model({"frequency": freqs_plot}, true_params).flux_density
ax.plot(freqs_plot.to_value(u.GHz), true_flux_plot.to_value(u.mJy), label="True Model", color="black")

# Add our synthetic data points.
ax.errorbar(
    frequencies.to_value(u.GHz),
    synthetic_flux.to_value(u.mJy),
    yerr=noise_fraction * synthetic_flux.to_value(u.mJy),
    fmt="o",
    label="Synthetic Data",
    color="red",
)

ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Flux Density [mJy]")
ax.set_xscale("log")
ax.set_yscale("log")

plt.show()

# %%
# Inference
# ---------
# Now that we have our synthetic dataset, we can set up the inference to recover the original parameters.
# We'll use MCMC for this purpose, which requires us to define a likelihood function and priors for the parameters.
#
# To start, we'll need to create a data container object to hold our synthetic data. We'll use
# a :class:`~astropy.table.Table` object and then feed it into the
# :class:`~data.photometry.RadioPhotometryEpochContainer`.
from astropy.table import Table

from triceratops.data.photometry import RadioPhotometryEpochContainer

# Create an Astropy Table with the synthetic data. We'll have everything happen at the same
# epoch in this case (dummy time column), and we'll set the upper limits to NaN since we have detections.
data_table = Table()
data_table["freq"] = frequencies
data_table["flux_density"] = synthetic_flux
data_table["flux_density_error"] = noise_fraction * synthetic_flux
data_table["flux_upper_limit"] = np.full((frequencies.size,), np.nan) * u.mJy  # No upper limits

# Create the RadioPhotometryContainer from the table.
photometry_data = RadioPhotometryEpochContainer(data_table)

# %%
# Now that we have the data container, we can set up the likelihood function. We'll use the
# :class:`inference.likelihood.base.GaussianLikelihoodXY` for this purpose.
#
# This likelihood works with single-epoch photometry data and assumes Gaussian errors on the flux densities.
from triceratops.inference.likelihood.base import GaussianLikelihoodXY

# Create the likelihood object.
likelihood = GaussianLikelihoodXY(
    model=sed_model,
    data=photometry_data,
)

# Print the current log likelihood value for the true parameters.
log_likelihood_true = likelihood.log_likelihood(true_params)
print(f"Log Likelihood at True Parameters: {log_likelihood_true}")

# %%
# Next, we need to define priors for the parameters we want to infer and generate an
# inference problem (:class:`inference.problem.InferenceProblem`). We'll use uniform priors for simplicity.
from triceratops.inference.prior import UniformPrior
from triceratops.inference.problem import InferenceProblem
from triceratops.inference.sampling.mcmc import EmceeSampler

# Generate the inference problem
problem = InferenceProblem(
    likelihood=likelihood,
)

# Set the priors for the parameters.
problem.set_prior("norm", "uniform", lower=1e-3 * u.Jy, upper=10.0 * u.Jy)
problem.set_prior("nu_break", "uniform", lower=1 * u.GHz, upper=50 * u.GHz)
problem.set_prior("p", "uniform", lower=2.0, upper=5.0)

# Fix the 's' parameter since we don't want to infer it in this example.
problem.parameters["s"].initial_value = true_params["s"]
problem.parameters["s"].freeze = True

# Create the sampler.
sampler = EmceeSampler(problem, n_walkers=32, ensemble_kwargs=dict())

# Run MCMC
result = sampler.run(10_000, progress=True)
samples = result.get_flat_samples(burn=1000, thin=10)

# %%
# Finally, we can analyze the MCMC results to see how well we recovered the original parameters.
# Frequencies for plotting
freqs_plot = np.logspace(8, 11, 200) * u.Hz

fig, ax = plt.subplots(figsize=(10, 6))

# Plot data
ax.errorbar(
    frequencies.to_value(u.GHz),
    synthetic_flux.to_value(u.mJy),
    yerr=noise_fraction * synthetic_flux.to_value(u.mJy),
    fmt="o",
    color="red",
    label="Data",
)

# Draw a subset of posterior samples
n_draw = 100
idx = np.random.choice(samples.shape[0], n_draw, replace=False)

for i in idx:
    theta_free = samples[i]
    params = problem.unpack_free_parameters(theta_free)
    flux = sed_model.forward_model({"frequency": freqs_plot}, params).flux_density

    ax.plot(freqs_plot.to_value(u.GHz), flux.to_value(u.mJy), color="C0", alpha=0.05)

# Plot true model
true_flux_plot = sed_model.forward_model({"frequency": freqs_plot}, true_params).flux_density

ax.plot(freqs_plot.to_value(u.GHz), true_flux_plot.to_value(u.mJy), color="black", lw=2, label="True Model")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Flux Density [mJy]")
ax.legend()

plt.show()

# %%
# It's also always good practice to look at some diagnostic plots to ensure that the MCMC
# chains have converged and that the posterior distributions look reasonable. We'll use the built-in plotting utilities
# in Triceratops for this.
#
# Let's start by looking at the trace plots for the :math:`\nu_{\rm brk}` parameter.
fig, ax = result.trace_plot("nu_break", burn=1000, thin=5)
plt.show()

# %%
# Let's also look at the corner plot for all the parameters to see their posterior distributions and covariances.
fig = result.corner_plot(burn=1000, thin=5, parameters=["norm", "nu_break", "p"])
plt.show()
