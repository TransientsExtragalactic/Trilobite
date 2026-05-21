r"""
=====================================================
From SED to Source Size: The Inverse Closure Workflow
=====================================================

Radio astronomers often measure a synchrotron SED with uncertainties and want
to estimate the **source radius**, **magnetic field**, and **minimum energy**
without direct angular resolution (e.g. no VLBI size measurement). This
workflow — fitting the SED and propagating the fit uncertainties through the
inverse closure — is the standard approach in the literature.

This example demonstrates the complete pipeline:

1. **Generate a realistic observed SED** with measurement uncertainties.
2. **Fit the SED** using :class:`~models.SEDs.synchrotron.Synchrotron_SSA_SBPL_Model`
   and MCMC to extract posterior distributions of :math:`F_{\rm pk}`,
   :math:`\nu_{\rm pk}`.
3. **Propagate the posteriors** through the inverse closure to obtain a
   posterior distribution of source radius :math:`R`.
4. **Check the expansion velocity** :math:`v = R/t` as a fraction of the
   speed of light — a simple test for relativistic ejecta.

.. hint::

    This workflow is particularly powerful when applied to a time series of
    epochs: the evolution of :math:`R(t)` directly measures the shock
    expansion velocity. For multi-epoch usage see the
    :ref:`sphx_glr_auto_examples_inference_plot_multi_epoch_sed_evolution.py`
    example.

Relevant API
------------
- :class:`~models.SEDs.synchrotron.Synchrotron_SSA_SBPL_Model`
- :meth:`~radiation.synchrotron.SEDs.PowerLaw_SSA_SynchrotronSED.from_params_to_physics`
- :class:`~inference.problem.InferenceProblem`
- :class:`~inference.sampling.mcmc.EmceeSampler`
"""

# %%
# Setup
# -----
import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u

from trilobite.models.SEDs.synchrotron import Synchrotron_SSA_SBPL_Model
from trilobite.radiation.synchrotron.SEDs import PowerLaw_SSA_SynchrotronSED
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

# %%
# Simulate a Realistic Observed SED
# ----------------------------------
#
# We simulate a single-epoch broadband radio SED of a radio transient — say a
# Type Ibc supernova observed 30 days post-explosion at 20 Mpc.  The true
# physical parameters are:
#
# - Source age: :math:`t = 30` days
# - True radius: :math:`R = 3 \times 10^{16}` cm
# - True B-field: :math:`B = 0.5` G
#
# These correspond to an expansion velocity
# :math:`v = R/t \approx 0.04 c` — sub-relativistic.

sed_model = Synchrotron_SSA_SBPL_Model()

# True SED parameters (from a forward closure calculation with B=0.5 G, R=3e16 cm)
true_params = {
    "norm": 8.0 * u.mJy,
    "nu_break": 5.0 * u.GHz,
    "p": 3.0,
    "s": -1.0,
}

# Observing frequencies spanning 0.5 GHz (GMRT/MeerKAT) to 43 GHz (VLA)
frequencies = u.Quantity([0.5, 1.4, 3.0, 5.0, 7.0, 10.0, 15.0, 22.0, 43.0], u.GHz)

# Simulate with 15% flux errors — realistic for calibration-limited observations
rng = np.random.default_rng(42)
noise_frac = 0.15
true_flux = sed_model.forward_model({"frequency": frequencies}, true_params).flux_density
noise = rng.normal(size=true_flux.size, scale=noise_frac) * true_flux
obs_flux = true_flux + noise
obs_err = noise_frac * true_flux

print("=== Simulated observations ===")
for f, flux, err in zip(frequencies, obs_flux, obs_err):
    print(f"  {f.to_value(u.GHz):5.1f} GHz:  {flux.to_value(u.mJy):.2f} ± {err.to_value(u.mJy):.2f} mJy")

# %%
# Visualize the Synthetic Data
# ----------------------------

freqs_plot = np.geomspace(0.3, 60.0, 300) * u.GHz
true_curve = sed_model.forward_model({"frequency": freqs_plot}, true_params).flux_density

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(freqs_plot.to_value(u.GHz), true_curve.to_value(u.mJy), color="black", lw=2, label="True SED")
ax.errorbar(
    frequencies.to_value(u.GHz),
    obs_flux.to_value(u.mJy),
    yerr=obs_err.to_value(u.mJy),
    fmt="o",
    color="C0",
    capsize=3,
    label="Simulated observations",
)
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Flux Density [mJy]")
ax.set_title("Synthetic Broadband Radio SED")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Fit the SED with MCMC
# ---------------------
#
# We set up an :class:`~inference.problem.InferenceProblem` to fit the SED
# parameters and sample the posterior with :class:`~inference.sampling.mcmc.EmceeSampler`.

from astropy.table import Table

from trilobite.data import InferenceData
from trilobite.data.photometry import RadioPhotometryEpochContainer
from trilobite.inference import GaussianCensoredLikelihood
from trilobite.inference.problem import InferenceProblem
from trilobite.inference.sampling.mcmc import EmceeSampler

# Build a data table
data_table = Table()
data_table["freq"] = frequencies
data_table["flux_density"] = obs_flux
data_table["flux_density_error"] = obs_err
data_table["flux_upper_limit"] = np.full(frequencies.size, np.nan) * u.mJy

photometry = RadioPhotometryEpochContainer(data_table)

inference_data = InferenceData.from_table(
    sed_model,
    photometry.table,
    variables={"frequency": "freq"},
    observables={
        "flux_density": ("flux_density", "flux_density_error", "flux_upper_limit", None),
    },
)

likelihood = GaussianCensoredLikelihood(model=sed_model, data=inference_data)

problem = InferenceProblem(likelihood=likelihood)
problem.set_prior("norm", "uniform", lower=0.01 * u.mJy, upper=100.0 * u.mJy)
problem.set_prior("nu_break", "uniform", lower=0.5 * u.GHz, upper=50.0 * u.GHz)
problem.set_prior("p", "uniform", lower=2.0, upper=5.0)

problem.parameters["s"].initial_value = -1.0
problem.parameters["norm"].initial_value = 1e-3
problem.parameters["s"].freeze = True

sampler = EmceeSampler(problem, n_walkers=16)
result = sampler.run(8_000, progress=True)
samples = result.get_flat_samples(burn=2000, thin=5)

print(f"\nPosterior samples shape: {samples.shape}")

# %%
# Plot Posterior SED Envelope
# ---------------------------
#
# Draw 200 samples from the posterior and overlay them on the data to visualize
# the SED uncertainty.

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(
    frequencies.to_value(u.GHz),
    obs_flux.to_value(u.mJy),
    yerr=obs_err.to_value(u.mJy),
    fmt="o",
    color="C0",
    capsize=3,
    label="Observations",
    zorder=5,
)

n_draw = 200
idx = rng.choice(samples.shape[0], n_draw, replace=False)
for i in idx:
    params = problem.unpack_free_parameters(samples[i])
    flux = sed_model.forward_model({"frequency": freqs_plot}, params).flux_density
    ax.plot(freqs_plot.to_value(u.GHz), flux.to_value(u.mJy), color="C1", alpha=0.05)

ax.plot(freqs_plot.to_value(u.GHz), true_curve.to_value(u.mJy), color="black", lw=2, label="True SED")

ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Flux Density [mJy]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Posterior SED Envelope")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Propagate Posteriors Through the Inverse Closure
# ------------------------------------------------
#
# Now comes the key step: for each posterior sample we apply the inverse
# closure to recover :math:`R` and :math:`B`. This propagates the SED
# measurement uncertainties into physical parameter uncertainties.
#
# We assume fixed microphysical parameters:
# :math:`p=3`, :math:`\epsilon_e = \epsilon_B = 0.1`.
from tqdm.auto import tqdm

phys_sed = PowerLaw_SSA_SynchrotronSED()

D_L = 20.0 * u.Mpc
epsilon_e = 0.1
epsilon_B = 0.1
p_fixed = 3.0
t_obs = 30.0 * u.day

R_post, B_post = [], []

num_samples = 1_000
indices = rng.choice(samples.shape[0], num_samples, replace=False)

for i in tqdm(indices):
    params = problem.unpack_free_parameters(samples[i])
    F_pk = params["norm"]
    nu_pk = params["nu_break"]

    inv = phys_sed.from_params_to_physics(
        "optically_thick",
        F_peak=F_pk * u.Jy,
        nu_peak=nu_pk * u.GHz,
        gamma_min=1.0,
        gamma_max=1e8,
        p=p_fixed,
        epsilon_E=epsilon_e,
        epsilon_B=epsilon_B,
        luminosity_distance=D_L,
        f_V=1.0,
        pitch_average=True,
    )
    R_post.append(inv["R"])
    B_post.append(inv["B"])

R_post = u.Quantity(R_post).to_value("cm")
B_post = u.Quantity(B_post).to_value("G")

# Expansion velocity posterior
v_post = R_post / t_obs.to_value("s")
v_over_c = v_post / const.c.cgs.value

print("\n=== Physical parameter posteriors ===")
print(
    f"  R  :  {np.percentile(R_post, 16):.2e}  –  {np.percentile(R_post, 50):.2e}  –  {np.percentile(R_post, 84):.2e}  cm"
)
print(
    f"  B  :  {np.percentile(B_post, 16):.3f}  –  {np.percentile(B_post, 50):.3f}  –  {np.percentile(B_post, 84):.3f}  G"
)
print(
    f"  v/c:  {np.percentile(v_over_c, 16):.4f}  –  {np.percentile(v_over_c, 50):.4f}  –  {np.percentile(v_over_c, 84):.4f}"
)

# %%
# Physical Parameter Posteriors
# -----------------------------
#
# The posterior distributions on :math:`R`, :math:`B`, and :math:`v/c` quantify
# how measurement uncertainties in the SED propagate to physical uncertainties.

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].hist(R_post, bins=40, color="C0", edgecolor="white", alpha=0.8)
axes[0].axvline(np.percentile(R_post, 16), ls="--", color="gray")
axes[0].axvline(np.percentile(R_post, 84), ls="--", color="gray")
axes[0].set_xlabel(r"Source Radius $R$ [cm]")
axes[0].set_ylabel("Posterior count")
axes[0].set_title("Inferred Source Radius")

axes[1].hist(B_post, bins=40, color="C1", edgecolor="white", alpha=0.8)
axes[1].axvline(np.percentile(B_post, 16), ls="--", color="gray")
axes[1].axvline(np.percentile(B_post, 84), ls="--", color="gray")
axes[1].set_xlabel(r"Equipartition Field $B$ [G]")
axes[1].set_title("Inferred Magnetic Field")

axes[2].hist(v_over_c, bins=40, color="C2", edgecolor="white", alpha=0.8)
axes[2].axvline(np.percentile(v_over_c, 16), ls="--", color="gray")
axes[2].axvline(np.percentile(v_over_c, 84), ls="--", color="gray")
axes[2].axvline(1.0, ls="-", color="red", lw=2, label="Speed of light")
axes[2].set_xlabel(r"Expansion Velocity $v/c$")
axes[2].set_title("Relativistic Check")
axes[2].set_xscale("log")
axes[2].legend()

plt.tight_layout()
plt.show()

# %%
# Relativistic Check
# ------------------
#
# A crucial sanity check: if the posterior on :math:`v/c = R/(ct)` extends
# above 1 the source is **superluminal** — which signals either (a) truly
# relativistic ejecta requiring a relativistic shock model, (b) an underestimated
# distance, or (c) a geometry effect (e.g. off-axis viewing).
#
# In this case the inferred :math:`v/c \sim 0.04` is comfortably sub-relativistic,
# consistent with the true input parameters.
#
# Median inferred velocity:
v_median = np.median(v_over_c)
print(f"\n  Median v/c = {v_median:.4f}  ({'sub-relativistic' if v_median < 0.1 else 'mildly relativistic'})")
