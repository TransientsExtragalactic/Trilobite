r"""
Fitting a Multi-Band Optical Light Curve with MCMC
====================================================

This example performs a complete round-trip:

1. **Forward model** — generate synthetic SDSS *ugriz* photometry from a
   :class:`~trilobite.models.generic.optical_photometry.FREDBlackbodyModel`
   with known parameters.
2. **Load** — package the noisy observations into an
   :class:`~trilobite.data.optical_photometry.OpticalPhotometryContainer`,
   demonstrating the standard data pipeline described in :ref:`data_overview`.
3. **Inspect** — verify the container and convert to
   :class:`~trilobite.data.core.InferenceData` via
   :meth:`~trilobite.data.optical_photometry.OpticalPhotometryContainer.to_inference_data`.
4. **Infer** — configure a
   :class:`~trilobite.inference.likelihood.base.GaussianCensoredLikelihood`,
   set log-uniform priors, and run ``emcee`` MCMC via
   :class:`~trilobite.inference.sampling.mcmc.EmceeSampler`.
5. **Diagnose** — inspect chains, acceptance fraction, and corner plot; verify
   the recovered posteriors bracket the true parameters.
6. **Posterior predictive check** — draw light curves from the posterior and
   overlay them on the data in all five bands.

The coupled optical model architecture is detailed in
:ref:`sphx_glr_galleries_modeling_plot_fred_blackbody_optical_lc.py`.

See Also
--------
- :class:`trilobite.models.generic.optical_photometry.FREDBlackbodyModel`
- :class:`trilobite.data.optical_photometry.OpticalPhotometryContainer`
- :class:`trilobite.inference.likelihood.base.GaussianCensoredLikelihood`
- :class:`trilobite.inference.sampling.mcmc.EmceeSampler`
- :ref:`data_overview` — data pipeline overview
- :ref:`data_to_inference` — step-by-step guide to ``to_inference_data()``
"""

# %%
# Step 1 — Forward Model
# -----------------------
#
# We assemble the SDSS *ugriz* filter bundle and instantiate the
# :class:`~trilobite.models.generic.optical_photometry.FREDBlackbodyModel`.
# The bundle's ``filter_names`` list defines the mapping
# ``band_name → band_idx`` that is used automatically when the data container
# calls :meth:`~trilobite.data.optical_photometry.OpticalPhotometryContainer.to_inference_data`.
#
# Trilobite loads real SDSS response curves via ``speclite``.  See
# :ref:`sphx_glr_galleries_photometry_plot_filter_bundle.py` for details
# on filter construction.

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import Table

from trilobite.models.generic.optical_photometry import FREDBlackbodyModel
from trilobite.utils.phot_utils import FilterBundle, flux_to_ab_mag, load_filter_from_speclite

# ---- Build SDSS ugriz filter bundle ----------------------------------------
SDSS_BANDS = {
    "u": "sdss2010-u",
    "g": "sdss2010-g",
    "r": "sdss2010-r",
    "i": "sdss2010-i",
    "z": "sdss2010-z",
}
filters = {short: load_filter_from_speclite(sp) for short, sp in SDSS_BANDS.items()}
bundle = FilterBundle(filters)
model = FREDBlackbodyModel(bundle)

print(f"Model variables : {model.variable_names}")
print(f"Model parameters: {model.parameter_names}")
print(f"Model outputs   : {model.output_names}")

# ---- True parameters for the synthetic transient ---------------------------
#
# All physical time parameters are stored in seconds internally (the model's
# base units for ``time``).  We express them as days for readability.
TRUE_PARAMS = {
    "t_0": 1.0 * 86400,  # onset at day 1  [s]
    "tau_r": 2.0 * 86400,  # 2-day rise       [s]
    "tau_d": 10.0 * 86400,  # 10-day decay     [s]
    "T_eff": 12_000.0,  # colour temperature [K]
    "amplitude": 5e-28,  # peak F_nu at V-pivot [erg/s/cm²/Hz]
}

# %%
# Observation grid: 9 epochs × 5 bands (ugriz) = 45 synthetic measurements.
# We observe at the same five bands per epoch — a realistic cadence for
# multi-band transient follow-up.

rng = np.random.default_rng(42)

# Observing epochs in days (post-onset)
t_epochs = np.array([1.5, 2.5, 4.0, 6.0, 9.0, 13.0, 20.0, 30.0, 42.0])
BAND_LIST = bundle.filter_names  # ['u', 'g', 'r', 'i', 'z']
n_epochs = len(t_epochs)
n_bands = len(BAND_LIST)

# Tile so every epoch has all five bands
t_all_days = np.repeat(t_epochs, n_bands)  # shape (45,)
band_names_all = np.tile(BAND_LIST, n_epochs)  # shape (45,)
band_idx_all = np.tile(np.arange(n_bands), n_epochs)  # shape (45,)

t_all_s = t_all_days * 86400.0

# ---- Evaluate the forward model to get noise-free fluxes -------------------
variables_true = {"time": t_all_s, "band_idx": band_idx_all}
flux_true = model.forward_model(variables_true, TRUE_PARAMS).flux_density.value

# Add 5 % Gaussian noise
noise_frac = 0.05
flux_obs = flux_true + rng.normal(0.0, noise_frac * flux_true)
flux_err = noise_frac * flux_true

print(f"\n{len(t_all_days)} synthetic observations generated across {n_epochs} epochs and {n_bands} bands.")
print(f"Noise level: {noise_frac * 100:.0f}% Gaussian (S/N ~ {1 / noise_frac:.0f})")

# %%
# Step 2 — Load into OpticalPhotometryContainer
# ----------------------------------------------
#
# :class:`~trilobite.data.optical_photometry.OpticalPhotometryContainer` is
# the standard entry point for multi-band optical photometry.  It validates the
# schema (required columns, units, ``band_name`` string column) and provides
# detection masking, dual flux/magnitude representation, and a direct path to
# the inference layer.
#
# The ``flux_upper_limit`` column is ``NaN`` for all rows because all
# observations are detections.  For censored data workflows, see
# :ref:`sphx_glr_galleries_inference_plot_censored_SED_fit.py`.

from trilobite.data import OpticalPhotometryContainer

obs_table = Table(
    {
        "time": t_all_days * u.day,
        "band_name": band_names_all,
        "flux_density": flux_obs * u.Unit("erg/(s cm2 Hz)"),
        "flux_density_error": flux_err * u.Unit("erg/(s cm2 Hz)"),
        "flux_upper_limit": np.full(len(t_all_days), np.nan) * u.Unit("erg/(s cm2 Hz)"),
    }
)

container = OpticalPhotometryContainer(obs_table)

print(f"\nOpticalPhotometryContainer")
print(f"  Observations : {container.n_obs}")
print(f"  Detections   : {container.n_detections}")
print(f"  Upper limits : {container.n_non_detections}")

# %%
# Quick look at the data before inference.

BAND_COLORS = {"u": "#8b5cf6", "g": "#3b82f6", "r": "#22c55e", "i": "#f59e0b", "z": "#ef4444"}

# Dense time grid for the true model curve
t_dense_s = np.linspace(0.1 * 86400, 50 * 86400, 400)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax_flux, ax_mag = axes

for band_name in BAND_LIST:
    idx = BAND_LIST.index(band_name)
    color = BAND_COLORS[band_name]

    # True model
    out_dense = model.forward_model(
        {"time": t_dense_s, "band_idx": np.full(len(t_dense_s), idx, dtype=int)}, TRUE_PARAMS
    )
    f_dense = out_dense.flux_density.value

    # Observed data for this band
    mask = band_names_all == band_name
    t_obs = t_all_days[mask]
    f_obs = flux_obs[mask]
    f_e = flux_err[mask]

    ax_flux.plot(t_dense_s / 86400, f_dense, color=color, lw=1.5, alpha=0.7)
    ax_flux.errorbar(t_obs, f_obs, yerr=f_e, fmt="o", color=color, ms=5, label=rf"${band_name}$")

    ax_mag.plot(t_dense_s / 86400, flux_to_ab_mag(f_dense), color=color, lw=1.5, alpha=0.7)
    ax_mag.errorbar(
        t_obs,
        flux_to_ab_mag(f_obs),
        yerr=2.5 * f_e / (np.log(10) * f_obs),
        fmt="o",
        color=color,
        ms=5,
        label=rf"${band_name}$",
    )

ax_flux.set_xlabel("Time [days]")
ax_flux.set_ylabel(r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_flux.set_title("Synthetic photometry — flux")
ax_flux.legend(ncol=2, fontsize=9)

ax_mag.invert_yaxis()
ax_mag.set_xlabel("Time [days]")
ax_mag.set_ylabel(r"AB magnitude")
ax_mag.set_title("Synthetic photometry — AB magnitude")
ax_mag.legend(ncol=2, fontsize=9)

fig.suptitle(
    rf"FREDBlackbodyModel  ($T_\mathrm{{eff}} = {TRUE_PARAMS['T_eff'] / 1000:.0f}$ kK, "
    rf"$\tau_r = {TRUE_PARAMS['tau_r'] / 86400:.0f}$ d, "
    rf"$\tau_d = {TRUE_PARAMS['tau_d'] / 86400:.0f}$ d)",
    y=1.01,
)
plt.tight_layout()
plt.show()

# %%
# Step 3 — Convert to InferenceData
# ----------------------------------
#
# :meth:`~trilobite.data.optical_photometry.OpticalPhotometryContainer.to_inference_data`
# performs three operations in a single call:
#
# 1. Resolves each ``band_name`` string to an integer ``band_idx`` by looking
#    up ``model.bundle.filter_names``.
# 2. Converts time to seconds (the model's ``base_units`` for ``time``).
# 3. Returns an :class:`~trilobite.data.core.InferenceData` object containing
#    only validated NumPy arrays — no units, no column names, no schema —
#    ready for likelihood evaluation.
#
# See :ref:`inference_data` for a full description of the ``InferenceData``
# interface.

inference_data = container.to_inference_data(model)
print(inference_data.describe())

# %%
# Step 4 — Likelihood and InferenceProblem
# -----------------------------------------
#
# :class:`~trilobite.inference.likelihood.base.GaussianCensoredLikelihood`
# evaluates
#
# .. math::
#
#     \ln \mathcal{L} = -\frac{1}{2} \sum_i
#     \left(\frac{F_{\nu,i}^\mathrm{obs} - F_{\nu,i}^\mathrm{model}}
#                {\sigma_i}\right)^2
#
# for detections (rows where the upper limit column is ``NaN``).  Upper
# limits would contribute a complementary CDF term, but this dataset contains
# only detections.
#
# :class:`~trilobite.inference.problem.InferenceProblem` wraps the likelihood
# together with priors on the free parameters.  All five parameters are
# sampled; log-uniform (``loguniform``) priors are appropriate for
# scale-invariant quantities.

from trilobite.inference.likelihood.base import GaussianCensoredLikelihood
from trilobite.inference.problem import InferenceProblem
from trilobite.inference.sampling.mcmc import EmceeSampler

likelihood = GaussianCensoredLikelihood(model=model, data=inference_data)

# Verify the log-likelihood is finite at the true parameters before sampling.
ll_true = likelihood.log_likelihood(TRUE_PARAMS)
print(f"log-likelihood at true parameters: {ll_true:.1f}")

problem = InferenceProblem(likelihood=likelihood)

# Log-uniform priors spanning a generous range around each parameter.
# All bounds can be expressed as Quantity objects — unit conversion is automatic.
problem.set_prior("t_0", "loguniform", lower=0.1 * u.day, upper=30.0 * u.day)
problem.set_prior("tau_r", "loguniform", lower=0.1 * u.day, upper=20.0 * u.day)
problem.set_prior("tau_d", "loguniform", lower=0.5 * u.day, upper=100.0 * u.day)
problem.set_prior("T_eff", "loguniform", lower=1_000.0 * u.K, upper=50_000.0 * u.K)
problem.set_prior(
    "amplitude", "loguniform", lower=1e-30 * u.Unit("erg/(s cm2 Hz)"), upper=1e-24 * u.Unit("erg/(s cm2 Hz)")
)

print(f"\nFree parameters  : {problem.free_parameter_names}")
print(f"Initial log-posterior: {problem.initial_log_posterior:.1f}")

# %%
# Step 5 — Run MCMC
# ------------------
#
# We use :class:`~trilobite.inference.sampling.mcmc.EmceeSampler`, a thin
# wrapper around `emcee <https://emcee.readthedocs.io>`_ that handles
# parameter packing/unpacking and result serialization.
#
# **Burn-in strategy**: the first 400 steps are discarded as burn-in; the
# remaining 800 steps are thinned by 10 to reduce autocorrelation.  Run
# longer chains (≥ 5 000 steps) for publication-quality posteriors.

N_WALKERS = 64
N_STEPS = 1_200
N_BURN = 400
N_THIN = 10

sampler = EmceeSampler(problem, n_walkers=N_WALKERS)
result = sampler.run(N_STEPS, progress=True)

af = result.acceptance_fraction
if af is not None:
    print(f"\nMean acceptance fraction: {np.mean(af):.3f}  (ideal: 0.2–0.5)")
else:
    print("\nAcceptance fraction: not recorded by sampler.")

# %%
# A well-tuned ensemble sampler typically achieves acceptance fractions between
# 0.2 and 0.5.  Values outside this range suggest that the walkers are taking
# steps that are too large or too small relative to the posterior width.

# %%
# Trace Plots — Chain Convergence
# --------------------------------
#
# Trace plots show the walker positions as a function of step number.  A
# converged chain looks like a "fuzzy caterpillar" with walkers mixing well
# across the posterior.  Clear trends or poor mixing indicate the burn-in
# was too short.

fig, result.trace_plot("T_eff", burn=0, thin=1)
plt.suptitle(r"Trace plot — $T_\mathrm{eff}$  (all steps, no burn-in removed)", y=1.01)
plt.tight_layout()
plt.show()

# %%
# After the burn-in phase (first ~400 steps) the chains settle and mix well.
# The dotted region before the dashed vertical line is discarded in all
# subsequent analyses.

fig, result.trace_plot("T_eff", burn=N_BURN, thin=N_THIN)
plt.suptitle(
    rf"Trace plot — $T_\mathrm{{eff}}$  (burn={N_BURN}, thin={N_THIN})",
    y=1.01,
)
plt.tight_layout()
plt.show()

# %%
# Corner Plot — Posterior Distributions
# --------------------------------------
#
# The corner plot shows marginal and joint posterior distributions for all
# five free parameters.  The dashed lines mark the true values; the recovered
# posteriors should bracket them.

# Convert true parameter values to the units used internally for display.
# The corner_plot ``truths`` dict expects values in base units (seconds, K, CGS).
truths_display = {
    "t_0": TRUE_PARAMS["t_0"],
    "tau_r": TRUE_PARAMS["tau_r"],
    "tau_d": TRUE_PARAMS["tau_d"],
    "T_eff": TRUE_PARAMS["T_eff"],
    "amplitude": TRUE_PARAMS["amplitude"],
}

fig = result.corner_plot(
    burn=N_BURN,
    thin=N_THIN,
    truths=truths_display,
)
plt.suptitle("Corner plot — posterior distributions with true values (dashed)", y=1.02)
plt.show()

# %%
# All five posteriors are unimodal and centred near the true values, confirming
# that the model is identifiable from five-band light curve data at 5 % noise.

# %%
# Posterior Summary
# ------------------
#
# We compute the median and 68 % credible interval for each parameter.

flat_samples = result.get_flat_samples(burn=N_BURN, thin=N_THIN)
param_names = result.parameter_names

print("\nPosterior summary (median ± 1σ):")
print(f"{'Parameter':<14} {'True':>12} {'Median':>12} {'−1σ':>10} {'+1σ':>10}")
print("-" * 60)

true_vals_in_order = [TRUE_PARAMS[p] for p in param_names]

for i, (name, true_val) in enumerate(zip(param_names, true_vals_in_order)):
    col = flat_samples[:, i]
    med = np.median(col)
    lo, hi = np.percentile(col, [16, 84])
    print(f"{name:<14} {true_val:>12.3e} {med:>12.3e} {med - lo:>10.3e} {hi - med:>10.3e}")

# %%
# Step 6 — Posterior Predictive Check
# -------------------------------------
#
# A posterior predictive check (PPC) draws model realisations from the
# posterior and overlays them on the data.  Good model–data agreement
# across all bands validates both the forward model and the inference setup.
#
# We draw 200 random posterior samples and evaluate the model in each band at
# a dense time grid.

N_DRAW = 200
rng_ppc = np.random.default_rng(0)
idx_draw = rng_ppc.choice(flat_samples.shape[0], N_DRAW, replace=False)

t_ppc_s = np.linspace(0.2 * 86400, 50 * 86400, 300)

fig, axes = plt.subplots(len(BAND_LIST), 1, figsize=(10, 14), sharex=True, sharey=False)

for ax, band_name in zip(axes, BAND_LIST):
    band_idx_val = BAND_LIST.index(band_name)
    color = BAND_COLORS[band_name]

    # Posterior predictive envelope
    for k in idx_draw:
        theta = flat_samples[k]
        params_k = problem.unpack_free_parameters(theta)
        out_k = model.forward_model(
            {"time": t_ppc_s, "band_idx": np.full(len(t_ppc_s), band_idx_val, dtype=int)},
            params_k,
        )
        ax.plot(t_ppc_s / 86400, flux_to_ab_mag(out_k.flux_density.value), color=color, alpha=0.04, lw=0.8)

    # True model
    out_true = model.forward_model(
        {"time": t_ppc_s, "band_idx": np.full(len(t_ppc_s), band_idx_val, dtype=int)},
        TRUE_PARAMS,
    )
    ax.plot(
        t_ppc_s / 86400, flux_to_ab_mag(out_true.flux_density.value), color="black", lw=2, zorder=5, label="True model"
    )

    # Data
    mask = band_names_all == band_name
    t_obs = t_all_days[mask]
    f_obs_band = flux_obs[mask]
    f_e_band = flux_err[mask]
    mag_obs = flux_to_ab_mag(f_obs_band)
    mag_err = 2.5 * f_e_band / (np.log(10) * f_obs_band)
    ax.errorbar(t_obs, mag_obs, yerr=mag_err, fmt="o", color="black", ms=5, zorder=6, label="Data")

    ax.invert_yaxis()
    ax.set_ylabel(rf"$m_{band_name}$ [AB mag]", fontsize=11)
    ax.text(0.02, 0.08, rf"${band_name}$ band", transform=ax.transAxes, fontsize=11, color=color, fontweight="bold")
    if band_name == BAND_LIST[0]:
        ax.legend(fontsize=9, loc="upper right")

axes[-1].set_xlabel("Time since reference [days]", fontsize=11)
fig.suptitle(
    "Posterior predictive check — 200 posterior draws (shaded) vs. data",
    fontsize=13,
    y=1.01,
)
plt.tight_layout()
plt.show()

# %%
# The shaded region represents the 68 % predictive interval from the posterior.
# It narrows around the peak, where the data constrain the model most tightly,
# and broadens at early and late times where fewer observations were made.
#
# Key takeaways
# ~~~~~~~~~~~~~
#
# - A :class:`~trilobite.models.generic.optical_photometry.FREDBlackbodyModel`
#   with just five parameters (three temporal, two spectral) recovers the
#   true light curves in all five *ugriz* bands simultaneously.
# - Multi-band coverage strongly constrains :math:`T_\mathrm{eff}`: the colour
#   information from five bands breaks the degeneracy between temperature and
#   amplitude that would be present in single-band data.
# - The :ref:`data_overview` pipeline — container → ``to_inference_data`` →
#   likelihood — requires no manual unit conversions or column remapping; the
#   ``model.bundle`` attribute carries all the band-resolution logic.
#
# Next steps
# ~~~~~~~~~~
#
# - Replace the synthetic table with real photometry from a FITS file:
#   ``container = OpticalPhotometryContainer.from_file("phot.fits")``.
# - Add upper limits by setting ``flux_density = NaN`` and populating
#   ``flux_upper_limit`` for non-detections.
# - Try :class:`~trilobite.models.generic.optical_photometry.GeneralizedFREDBlackbodyModel`
#   for a more flexible temporal shape, or
#   :class:`~trilobite.models.generic.optical_photometry.GaussianBlackbodyModel`
#   for symmetric transients.
