"""
Fitting a Time-Evolving Synchrotron SED
=======================================

This example demonstrates a complete end-to-end inference workflow in
Triceratops:

- Loading radio photometry data using the :class:`~data.photometry.RadioPhotometryContainer`, including
  some special configurations for analysis.
- Use an evolving SED model from :mod:`models.SEDs` to model the radio emission.
- Running MCMC with :class:`~inference.sampling.mcmc.EmceeSampler`.
- Visualizing posterior predictive spectra.

We model the dataset using a :class:`~models.SEDs.evolving_sed.PL_Evolving_SSA_SED_Model`,
which describes a broken power-law spectrum whose break frequency and amplitude
evolve in time.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.time import Time

from triceratops.data.photometry import RadioPhotometryContainer
from triceratops.inference import GaussianCensoredLikelihood, InferenceProblem
from triceratops.inference.sampling import EmceeSampler
from triceratops.models.SEDs import PL_Evolving_SSA_SED_Model
from triceratops.utils.plot_utils import set_plot_style

set_plot_style()
np.random.seed(42)

# %%
# Loading and Inspecting the Data
# --------------------------------
#
# We begin by loading a synthetic radio photometry dataset bundled with
# the documentation. The file is nearly in the correct format for direct
# ingestion, but requires two small adjustments:
#
# 1. The column ``time_midobs`` is remapped to the canonical name ``time``.
# 2. The times in the file are absolute (Julian Date), so we convert them
#    to relative time measured from a physically meaningful reference epoch.
#
# For AT2018COW, from which this dataset was randomly generated,
# we adopt the optical peak time reported in the literature
# as the reference epoch. Internally, the
# :class:`~data.photometry.RadioPhotometryContainer`
# converts all times to **relative days since the reference epoch**, which is
# the canonical time representation used throughout the modeling and
# inference pipelines.
#
# The resulting object is a validated, immutable data container that
# enforces schema consistency, unit correctness, and detection/upper-limit
# bookkeeping.
#
# .. note::
#
#     A more detailed discussion of the various options for loading data can be found
#     in the documentation for :class:`~data.photometry.RadioPhotometryContainer`.

# Set the data file.
data_file = "../../../../../docs/source/_data/example_photometry.fits"

# Set the peak time from the literature.
optical_peak_time = Time(58284.3, format="mjd", scale="utc")

photometry_data = RadioPhotometryContainer.from_file(
    data_file,
    column_map={"time_midobs": "time"},
    time_starts=optical_peak_time.jd1 * u.day,
    internal_time_scale="utc",
    internal_time_format="jd",
)

print(f"Loaded {photometry_data.n_obs} observations.")

# Group observations separated by more than 3 days into distinct epochs
photometry_data.set_epochs_from_time_gaps(max_gap=3.0 * u.day)

# %%
# The photometry container includes a built-in plotting routine
# that automatically handles detections and upper limits.
fig, ax = plt.subplots(figsize=(8, 6))
photometry_data.plot_photometry(fig=fig, axes=ax)
plt.show()

# %%
# Constructing the Model and Likelihood
# --------------------------------------
# For this example, we'll use the :class:`~models.SEDs.evolving_sed.PL_Evolving_SSA_SED_Model`, which describes provides
# a phenomenological description of generic synchrotron SEDs from time evolving transients. Generically, we
# assume that each epoch has an SED described by a smooth broken power-law with fixed slopes :math:`\alpha_1`
# and :math:`\alpha_2` below and above a break frequency :math:`\nu_{\rm brk}(t)`, respectively.
# The break frequency evolves in time as
#
# .. math::
#     \nu_{\rm brk}(t) = \nu_{\rm brk,0} \left( \frac{t}{t_0} \right)^{\beta},
#
# where :math:`\nu_{\rm brk,0}` is the break frequency at a reference time :math:`t_0` and :math:`\beta` is
# the power-law index describing the temporal evolution of the break frequency.
# The flux density at the break also evolves in time as
#
# .. math::
#     F_{\rm brk}(t) = F_{\rm brk,0} \left( \frac{t}{t_0} \right)^{\gamma},
#
# where :math:`F_{\rm brk,0}` is the flux density at the break at the reference time and :math:`\gamma`
# is the power-law index describing the temporal evolution of the break flux density.
# The parameter :math:`s` controls the smoothness of the break, and is fixed to a value of 0.5 for this analysis.
# The reference time :math:`t_0` is fixed to 100 days, which is a reasonable timescale
# for the evolution of the SED in this dataset. The model is implemented in physical units,
# so all parameters have associated units that are automatically
# handled throughout the inference process.
#
# More detailed information can be found in the API documentation for
# :class:`~models.SEDs.evolving_sed.PL_Evolving_SSA_SED_Model`.

# Generate the model object.
model = PL_Evolving_SSA_SED_Model()

# Convert the photometry data to the format required for inference. This standardizes
# the data so that it is more easily ingested by the likelihood and model, and also ensures that all units are
# properly handled and consistent with the model's base units.
inference_data = photometry_data.to_inference_data(model=model)

# Construct the likelihood, which automatically handles detections and upper limits.
likelihood = GaussianCensoredLikelihood(
    model=model,
    data=inference_data,
)

# Finally, we construct the inference problem by combining the likelihood with the model and data.
problem = InferenceProblem(likelihood=likelihood)

# %%
# Defining Priors
# ---------------
#
# We assign broad uniform priors to the spectral parameters.
# Units are automatically handled and coerced into model base units.

problem.set_prior("F_brk_0", "uniform", lower=1e-3 * u.mJy, upper=1e3 * u.mJy)
problem.set_prior("nu_brk_0", "uniform", lower=0.01 * u.GHz, upper=50 * u.GHz)
problem.set_prior("alpha_1", "uniform", lower=0.0, upper=5.0)
problem.set_prior("alpha_2", "uniform", lower=-5.0, upper=0.0)
problem.set_prior("beta", "uniform", lower=-5.0, upper=5.0)
problem.set_prior("gamma", "uniform", lower=-5.0, upper=5.0)

# Fix nuisance parameters
problem.parameters["s"].initial_value = 0.5
problem.parameters["s"].freeze = True

problem.parameters["t_0"].initial_value = (100 * u.day).to(u.s)
problem.parameters["t_0"].freeze = True

# Set reasonable initial values for the free parameters to help with MCMC convergence.
problem.parameters["F_brk_0"].initial_value = 0.01  # Jy
problem.parameters["nu_brk_0"].initial_value = 1.1e10  # Hz
problem.parameters["alpha_1"].initial_value = 1
problem.parameters["alpha_2"].initial_value = -1.0
problem.parameters["beta"].initial_value = 2
problem.parameters["gamma"].initial_value = 1

# %%
# Running MCMC
# ------------
#
# We now sample the posterior using an ensemble sampler.

sampler = EmceeSampler(problem, n_walkers=16)

result = sampler.run(20_000, progress=True)

samples = result.get_flat_samples(burn=4000, thin=10)


# %%
# Posterior Predictive Spectra
# ----------------------------
#
# We now visualize the posterior predictive model at representative
# epoch times. For each epoch, we:
#
# 1. Compute spectra from posterior samples
# 2. Calculate the median model
# 3. Compute the 16th–84th percentile credible region
# 4. Overlay this on the observed photometry
#
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

freqs_plot = np.logspace(8, 11, 300) * u.Hz

fig, ax = plt.subplots(figsize=(8, 6))

cmap = plt.get_cmap("jet")
norm = LogNorm(vmin=np.amin(photometry_data.time.to_value(u.day)), vmax=np.amax(photometry_data.time.to_value(u.day)))

for epoch_id in np.unique(photometry_data.epoch_ids):
    # Get the epoch time.
    epoch_time = np.mean(photometry_data.time[photometry_data.get_epoch_mask(epoch_id)].to_value(u.day))
    color = cmap(norm(epoch_time))

    photometry_data.plot_epoch(
        epoch_id,
        fig=fig,
        axes=ax,
        color=color,
        detection_style=dict(marker="o", markersize=5, markeredgecolor="black"),
        upper_limit_style=dict(marker="o", markersize=5, markeredgecolor="black"),
    )

    mask = photometry_data.get_epoch_mask(epoch_id)
    t_epoch = np.mean(photometry_data.time[mask])

    # Collect model draws for this epoch
    model_fluxes = []

    for theta in samples[np.random.choice(len(samples), 200, replace=False)]:
        params = problem.unpack_free_parameters(theta)

        model_output = model.forward_model(
            {"frequency": freqs_plot, "time": t_epoch},
            params,
        )

        model_fluxes.append(model_output.flux_density.to_value(u.Jy))

    model_fluxes = np.array(model_fluxes)

    # Compute summary statistics
    median = np.median(model_fluxes, axis=0)
    lower = np.percentile(model_fluxes, 16, axis=0)
    upper = np.percentile(model_fluxes, 84, axis=0)

    # Plot median curve
    ax.plot(
        freqs_plot.to_value(u.Hz),
        median,
        color=color,
        lw=2,
        label=f"{int(epoch_time)} days",
    )

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Flux Density [mJy]")
ax.legend(title="Posterior Median + 68% CI", ncol=2)

plt.tight_layout()
# sphinx_gallery_thumbnail_number = -3
plt.show()


# %%
# Diagnostic Plots
# ----------------
#
# Finally, we inspect sampler diagnostics.

result.trace_plot("alpha_1", burn=1000, thin=5)
plt.show()

# %%
# The resulting corner plot is as follows:
#
result.corner_plot(
    burn=1000,
    thin=5,
    parameters=["F_brk_0", "nu_brk_0", "alpha_1", "alpha_2", "beta", "gamma"],
)
plt.show()
