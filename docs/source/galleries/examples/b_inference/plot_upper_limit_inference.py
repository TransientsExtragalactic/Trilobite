r"""
======================================================
Upper Limit Constraints: Non-Detections as Information
======================================================

In radio surveys, newly discovered transients are often not detected at early
times. Rather than discarding these non-detections, they can be incorporated as
**upper limits** that constrain the parameter space just as powerfully as
detections — sometimes more so.

This example demonstrates:

1. **The astrophysical context**: early radio upper limits from a supernova
   search constrain the CSM wind density and ejecta energy before any
   detection is possible.
2. **How Triceratops handles upper limits**: using
   :class:`~inference.likelihood.base.GaussianCensoredLikelihood`, which
   implements a censored Gaussian likelihood that treats non-detections as
   one-sided constraints.
3. **The informational value of upper limits**: by comparing posteriors with
   and without upper limits, we show that including the non-detections
   significantly tightens the constraints on physical parameters.

The Physics
-----------

We model the radio emission using a
:class:`~models.SEDs.synchrotron.Synchrotron_SSA_SBPL_Model` evaluated at
a single epoch. The two main parameters of interest are:

- The **peak flux** :math:`F_{\rm pk}` — a proxy for the total energy in the
  emitting plasma (related to :math:`E_{\rm ej}` and :math:`\epsilon_e`).
- The **break frequency** :math:`\nu_{\rm pk}` — related to the CSM column
  density (related to :math:`\dot{M}/v_{\rm w}`).

Early upper limits in a low-frequency band (e.g. 1.4 GHz) directly constrain
whether the source has already peaked at low frequencies, ruling out models
with high CSM densities.

Relevant API
------------
- :class:`~inference.likelihood.base.GaussianCensoredLikelihood`
- :class:`~data.photometry.RadioPhotometryEpochContainer`
- :class:`~inference.problem.InferenceProblem`
- :class:`~inference.sampling.mcmc.EmceeSampler`
"""

# %%
# Setup
# -----
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from triceratops.models.SEDs.synchrotron import Synchrotron_SSA_SBPL_Model
from triceratops.utils.plot_utils import set_plot_style

set_plot_style()

rng = np.random.default_rng(17)

# %%
# Scenario: A Radio Supernova Caught at Early Times
# --------------------------------------------------
#
# We imagine a scenario where a Type Ibc SN was discovered optically and radio
# observations began immediately. At the first epoch (15 days post-explosion)
# the source is detected at high frequencies but not yet at low frequencies.
#
# True model parameters (representing a moderate CSM density):
#
# - Peak flux: 3 mJy at :math:`\nu_{\rm pk} = 8` GHz
# - The low-frequency emission is still rising (absorbed by SSA)

sed_model = Synchrotron_SSA_SBPL_Model()

true_params = {
    "norm": 3.0 * u.mJy,
    "nu_break": 8.0 * u.GHz,
    "p": 3.0,
    "s": -1.0,
}

# Observing frequencies
# Low-frequency bands (non-detections expected): 0.6, 1.4, 3 GHz
# High-frequency bands (detections expected): 7, 10, 15 GHz
all_frequencies = u.Quantity([0.6, 1.4, 3.0, 7.0, 10.0, 15.0], u.GHz)
flux_limit = 0.4 * u.mJy  # 3-sigma detection threshold

# True fluxes
true_flux = sed_model.forward_model({"frequency": all_frequencies}, true_params).flux_density
noise_frac = 0.15
noise = rng.normal(size=true_flux.size, scale=noise_frac) * true_flux
obs_flux = true_flux + noise

# Apply flux limit: detections where obs_flux > flux_limit, upper limits otherwise
is_detection = obs_flux > flux_limit
is_upper_limit = ~is_detection

print("=== Observations ===")
for f, fl, det in zip(all_frequencies, obs_flux, is_detection):
    status = "DETECTED" if det else "UPPER LIMIT"
    print(f"  {f.to_value(u.GHz):4.1f} GHz: {fl.to_value(u.mJy):.3f} mJy  [{status}]")

# %%
# Visualize the Scenario
# ----------------------
#
# Plot the true SED and the observations, showing which are detections and
# which are upper limits.

freqs_plot = np.geomspace(0.3, 25.0, 300) * u.GHz
true_curve = sed_model.forward_model({"frequency": freqs_plot}, true_params).flux_density

fig, ax = plt.subplots(figsize=(9, 5))
ax.loglog(freqs_plot.to_value(u.GHz), true_curve.to_value(u.mJy), color="black", lw=2, ls="--", label="True SED")
ax.axhline(flux_limit.to_value(u.mJy), color="gray", ls=":", lw=1.5, label=f"Detection limit ({flux_limit:.1f})")

# Detections
ax.errorbar(
    all_frequencies[is_detection].to_value(u.GHz),
    obs_flux[is_detection].to_value(u.mJy),
    yerr=noise_frac * obs_flux[is_detection].to_value(u.mJy),
    fmt="o",
    color="C0",
    capsize=3,
    ms=7,
    label="Detections",
)

# Upper limits (triangles pointing down)
ax.scatter(
    all_frequencies[is_upper_limit].to_value(u.GHz),
    flux_limit.to_value(u.mJy) * np.ones(is_upper_limit.sum()),
    marker="v",
    s=80,
    color="C3",
    zorder=5,
    label="Upper limits",
)

ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Flux Density [mJy]")
ax.set_title("Radio Observations at 15 Days Post-Explosion")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Build the Two Inference Problems
# --------------------------------
#
# We now run two inference problems:
#
# 1. **Detections only** — uses only the detected (high-frequency) points.
# 2. **Detections + upper limits** — includes the low-frequency non-detections
#    as censored data points.
#
# By comparing the resulting posteriors, we can directly quantify the
# information content of the upper limits.

from astropy.table import Table

from triceratops.data import InferenceData
from triceratops.data.photometry import RadioPhotometryEpochContainer
from triceratops.inference import GaussianCensoredLikelihood
from triceratops.inference.problem import InferenceProblem
from triceratops.inference.sampling.mcmc import EmceeSampler


def _build_problem(use_upper_limits: bool):
    """Build an InferenceProblem with or without upper limits."""
    if use_upper_limits:
        freqs = all_frequencies
        fluxes = np.where(is_detection, obs_flux.to_value(u.mJy), np.nan) * u.mJy
        errs = noise_frac * np.abs(obs_flux)
        errs[is_upper_limit] = flux_limit / 3.0  # small dummy error
        upper_lims = np.where(is_upper_limit, flux_limit.to_value(u.mJy), np.nan) * u.mJy
    else:
        freqs = all_frequencies[is_detection]
        fluxes = obs_flux[is_detection]
        errs = noise_frac * obs_flux[is_detection]
        upper_lims = np.full(is_detection.sum(), np.nan) * u.mJy

    data_table = Table()
    data_table["freq"] = freqs
    data_table["flux_density"] = fluxes
    data_table["flux_density_error"] = errs
    data_table["flux_upper_limit"] = upper_lims

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
    problem.set_prior("norm", "uniform", lower=0.1 * u.mJy, upper=30.0 * u.mJy)
    problem.set_prior("nu_break", "uniform", lower=0.5 * u.GHz, upper=50.0 * u.GHz)
    problem.set_prior("p", "uniform", lower=2.0, upper=5.0)
    problem.parameters["norm"].initial_value = 3.0 * u.mJy
    problem.parameters["nu_break"].initial_value = 8.0 * u.GHz
    problem.parameters["p"].initial_value = 3.0
    problem.parameters["s"].initial_value = -1.0
    problem.parameters["s"].freeze = True
    return problem


# Run both inferences
print("Running inference: detections only...")
problem_det = _build_problem(use_upper_limits=False)
sampler_det = EmceeSampler(problem_det, n_walkers=32)
result_det = sampler_det.run(8_000, progress=True)
samples_det = result_det.get_flat_samples(burn=2000, thin=5)

print("Running inference: detections + upper limits...")
problem_ul = _build_problem(use_upper_limits=True)
sampler_ul = EmceeSampler(problem_ul, n_walkers=32)
result_ul = sampler_ul.run(8_000, progress=True)
samples_ul = result_ul.get_flat_samples(burn=2000, thin=5)

# %%
# Compare the Posteriors
# ----------------------
#
# We compare the marginal posteriors on the two key parameters:
# peak normalization (related to CSM density) and break frequency.

param_names = [p for p, par in problem_ul.parameters.items() if not par.freeze]

fig, axes = plt.subplots(1, len(param_names), figsize=(5 * len(param_names), 4))

param_labels = {
    "norm": r"Peak Flux $F_{\rm pk}$ [mJy]",
    "nu_break": r"Break Frequency $\nu_{\rm pk}$ [GHz]",
    "p": r"Electron index $p$",
}
true_vals = {
    "norm": true_params["norm"].to_value(u.mJy),
    "nu_break": true_params["nu_break"].to_value(u.GHz),
    "p": true_params["p"],
}

for ax, name in zip(axes, param_names):
    col = param_names.index(name)
    s_det = samples_det[:, col]
    s_ul = samples_ul[:, col]

    ax.hist(s_det, bins=40, alpha=0.5, color="C0", density=True, label="Detections only")
    ax.hist(s_ul, bins=40, alpha=0.5, color="C1", density=True, label="Detections + ULs")
    ax.axvline(true_vals[name], color="black", lw=2, ls="--", label="True value")
    ax.set_xlabel(param_labels.get(name, name))
    ax.set_ylabel("Posterior density")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.3)

plt.suptitle("Posterior Comparison: With and Without Upper Limits", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Quantify the Improvement
# ------------------------
#
# Compute the 68% credible interval width for each parameter in both runs.
# A narrower interval means tighter constraints.

print("\n=== Posterior 68% credible interval widths ===")
print(f"{'Parameter':<20}  {'Det only':>12}  {'Det + ULs':>12}  {'Improvement':>14}")
print("-" * 65)
for name in param_names:
    col = param_names.index(name)
    s_det = samples_det[:, col]
    s_ul = samples_ul[:, col]
    w_det = np.percentile(s_det, 84) - np.percentile(s_det, 16)
    w_ul = np.percentile(s_ul, 84) - np.percentile(s_ul, 16)
    improvement = (w_det - w_ul) / w_det * 100
    print(f"  {name:<18}  {w_det:>12.3g}  {w_ul:>12.3g}  {improvement:>12.1f}%")

# %%
# Posterior SED Envelopes
# -----------------------
#
# Finally, plot the posterior SED envelopes from both runs to visually
# confirm that the upper limits exclude SEDs with peaks below the detection
# threshold at low frequencies.

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, (samples, problem, label, color) in zip(
    axes,
    [
        (samples_det, problem_det, "Detections only", "C0"),
        (samples_ul, problem_ul, "Detections + Upper Limits", "C1"),
    ],
):
    idx = rng.choice(samples.shape[0], 200, replace=False)
    for i in idx:
        params = problem.unpack_free_parameters(samples[i])
        flux = sed_model.forward_model({"frequency": freqs_plot}, params).flux_density
        ax.plot(freqs_plot.to_value(u.GHz), flux.to_value(u.mJy), color=color, alpha=0.04)

    ax.loglog(freqs_plot.to_value(u.GHz), true_curve.to_value(u.mJy), color="black", lw=2, label="True SED")
    ax.axhline(flux_limit.to_value(u.mJy), color="gray", ls=":", label="Detection limit")

    ax.errorbar(
        all_frequencies[is_detection].to_value(u.GHz),
        obs_flux[is_detection].to_value(u.mJy),
        yerr=noise_frac * obs_flux[is_detection].to_value(u.mJy),
        fmt="o",
        color="C0",
        capsize=3,
        label="Detections",
    )
    ax.scatter(
        all_frequencies[is_upper_limit].to_value(u.GHz),
        flux_limit.to_value(u.mJy) * np.ones(is_upper_limit.sum()),
        marker="v",
        s=80,
        color="C3",
        label="Upper limits",
    )

    ax.set_xlabel("Frequency [GHz]")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)

axes[0].set_ylabel("Flux Density [mJy]")
plt.suptitle("SED Posteriors: Upper Limits Exclude Low-Frequency Models", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# The comparison shows clearly that including upper limits removes SED solutions
# that would place the spectral peak at low frequency (high CSM column density),
# ruling out models where the absorption turnover sits well below 3 GHz at this
# epoch. This is the key information content of early radio non-detections.
