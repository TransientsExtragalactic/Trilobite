r"""
===========================================================
Inferring Physical Shock Parameters from a Radio Supernova
===========================================================

This example demonstrates the **full science-grade inference workflow** for
recovering the astrophysical parameters of a radio supernova from multi-frequency
radio observations.

The goal is to directly infer the progenitor's **mass-loss history** and the
supernova's **explosion energy** from radio data — parameters that probe the
progenitor star decades before the explosion.

The Physical Model
------------------

We use :class:`~models.supernovae.chevalier_shock.ChevalierShockModel`, which
implements the Chevalier (1982) self-similar solution for a supernova shock
expanding into a wind-stratified CSM. The model computes the synchrotron flux
density at a given frequency and time directly from six physical parameters:

**Explosion / ejecta parameters**:

- :math:`E_{\rm ej}` — total kinetic energy of the ejecta [erg]
- :math:`M_{\rm ej}` — ejecta mass [:math:`M_\odot`]
- :math:`n` — outer ejecta density power-law index

**CSM parameters** (encoding the progenitor wind):

- :math:`\dot{M}` — progenitor mass-loss rate [:math:`M_\odot/{\rm yr}`]
- :math:`v_{\rm w}` — progenitor wind speed [km/s]
  (together these give the wind density :math:`\rho_w \propto \dot{M}/v_w r^{-2}`)

**Microphysical parameters**:

- :math:`\epsilon_e` — electron energy fraction
- :math:`\epsilon_B` — magnetic energy fraction
- :math:`p` — electron energy distribution power-law index

For this example we fix :math:`n = 10` (canonical for Type Ibc supernovae),
:math:`M_{\rm ej} = 2 M_\odot`, and :math:`p = 3`, and infer the remaining
four parameters.

Overview
--------

1. **Simulate synthetic multi-frequency radio light curves** using known
   true parameter values.
2. **Set up an InferenceProblem** with physically motivated priors.
3. **Run MCMC** to sample the posterior.
4. **Analyze the results** — corner plot, posterior predictive check, and
   physical interpretation.

Relevant API
------------
- :class:`~models.supernovae.chevalier_shock.ChevalierShockModel`
- :class:`~inference.problem.InferenceProblem`
- :class:`~inference.sampling.mcmc.EmceeSampler`
"""

# %%
# Setup
# -----
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from triceratops.models.supernovae.chevalier_shock import ChevalierShockModel
from triceratops.utils.plot_utils import set_plot_style

set_plot_style()

rng = np.random.default_rng(42)

# %%
# Define True Parameters and Generate Synthetic Observations
# -----------------------------------------------------------
#
# We simulate observations of a radio-bright Type Ibc supernova at 10 Mpc.
# The true parameters are chosen to represent a moderate mass-loss rate progenitor,
# similar to observed events like SN 2011dh or SN 2002ap.
#
# We observe at 4 frequencies spanning VLA bands (3, 7, 15, 22 GHz) and 8 epochs
# from 10 to 200 days post-explosion.

shock_model = ChevalierShockModel()

# ------ True physical parameters ------
true_params = {
    "E_ej": 1.0e51 * u.erg,  # Kinetic energy
    "M_ej": 2.0 * u.Msun,  # Ejecta mass (fixed in inference)
    "n": 10.0,  # Outer ejecta index (fixed)
    "s": 2.0,  # CSM power-law index (wind; fixed)
    "rho_0": 5e-19 * u.g / u.cm**3,  # CSM density at 1e14 cm
    "delta": 0.0,  # Inner ejecta index (fixed)
    "epsilon_e": 0.1,  # Electron energy fraction
    "epsilon_B": 0.1,  # Magnetic energy fraction
    "p": 3.0,  # Electron index (fixed)
    "gamma_min": 1.0,
    "gamma_max": 1e7,
    "f": 0.5,
    "theta": np.pi / 2,
    "smoothing_s": -0.5,
    "D": 10.0,  # Distance in Mpc
}

# Observing setup
frequencies = u.Quantity([3.0, 7.0, 15.0, 22.0], u.GHz)
epochs = u.Quantity([15.0, 25.0, 40.0, 60.0, 90.0, 130.0, 180.0, 250.0], u.day)

noise_frac = 0.15

# Generate synthetic light curves
syn_data = {}
for freq in frequencies:
    fluxes_t = []
    for t in epochs:
        variables = {"frequency": freq.to(u.Hz), "time": t.to(u.s)}
        flux = shock_model.forward_model(variables, true_params).flux_density.to(u.mJy)
        noise = rng.normal() * noise_frac * flux
        fluxes_t.append((flux + noise).to(u.mJy))
    syn_data[freq.to_value(u.GHz)] = u.Quantity(fluxes_t)

# %%
# Visualize the Synthetic Light Curves
# -------------------------------------

colors = ["C0", "C1", "C2", "C3"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

t_dense = np.geomspace(10.0, 300.0, 200) * u.day

for ax, (freq, color) in zip(axes, zip(frequencies, colors)):
    # True model
    true_lc = []
    for t in t_dense:
        v = {"frequency": freq.to(u.Hz), "time": t.to(u.s)}
        true_lc.append(shock_model.forward_model(v, true_params).flux_density.to_value(u.mJy))
    ax.loglog(t_dense.to_value(u.day), true_lc, color="black", lw=1.5, ls="--", label="True model")

    # Observations
    obs = syn_data[freq.to_value(u.GHz)]
    errs = noise_frac * obs
    ax.errorbar(
        epochs.to_value(u.day),
        obs.to_value(u.mJy),
        yerr=errs.to_value(u.mJy),
        fmt="o",
        color=color,
        capsize=3,
        ms=6,
        label="Synthetic data",
    )
    ax.set_title(f"{freq.to_value(u.GHz):.0f} GHz")
    ax.set_ylabel("Flux Density [mJy]")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)

axes[2].set_xlabel("Time post-explosion [days]")
axes[3].set_xlabel("Time post-explosion [days]")
plt.suptitle("Synthetic Radio Light Curves (ChevalierShockModel)", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Set Up the Inference Problem
# -----------------------------
#
# We infer four physical parameters:
#
# - :math:`E_{\rm ej}` — total ejecta kinetic energy [erg], **log-uniform prior**
#   (scale-free quantity, uncertain over orders of magnitude)
# - :math:`\rho_0` — CSM density at 1e14 cm [g/cm³], **log-uniform prior**
#   (directly related to mass-loss rate)
# - :math:`\epsilon_e` — electron energy fraction, **log-uniform prior**
#   (typical range 0.01–0.5)
# - :math:`\epsilon_B` — magnetic energy fraction, **log-uniform prior**
#
# Log-uniform (Jeffreys) priors are appropriate for scale-free parameters
# whose order of magnitude is uncertain.
#
# The remaining parameters are fixed at their true values.

from astropy.table import Table, vstack

from triceratops.data import InferenceData
from triceratops.data.photometry import RadioPhotometryEpochContainer
from triceratops.inference import GaussianCensoredLikelihood
from triceratops.inference.likelihood.base import GaussianLikelihood
from triceratops.inference.problem import InferenceProblem
from triceratops.inference.sampling.mcmc import EmceeSampler

# Build a stacked multi-frequency, multi-epoch data table
tables = []
for freq in frequencies:
    for i, t in enumerate(epochs):
        row = Table()
        row["freq"] = [freq.to(u.Hz)]
        row["time"] = [t.to(u.s)]
        obs_f = syn_data[freq.to_value(u.GHz)][i]
        row["flux_density"] = [obs_f]
        row["flux_density_error"] = [noise_frac * obs_f]
        row["flux_upper_limit"] = [np.nan * u.mJy]
        tables.append(row)

data_table = vstack(tables)
photometry = RadioPhotometryEpochContainer(data_table)

inference_data = InferenceData.from_table(
    shock_model,
    photometry.table,
    variables={"frequency": "freq", "time": "time"},
    observables={
        "flux_density": ("flux_density", "flux_density_error", "flux_upper_limit", None),
    },
)

likelihood = GaussianCensoredLikelihood(model=shock_model, data=inference_data)
problem = InferenceProblem(likelihood=likelihood)

# Log-uniform priors on scale-free parameters
problem.set_prior("E_ej", "loguniform", lower=1e49 * u.erg, upper=1e53 * u.erg)
problem.set_prior("rho_0", "loguniform", lower=1e-21 * u.g / u.cm**3, upper=1e-17 * u.g / u.cm**3)
problem.set_prior("epsilon_e", "loguniform", lower=0.01, upper=0.5)
problem.set_prior("epsilon_B", "loguniform", lower=0.01, upper=0.5)

# Set initial values within the prior bounds (required for sampler validation)
problem.parameters["E_ej"].initial_value = 1e51 * u.erg
problem.parameters["rho_0"].initial_value = 5e-19 * u.g / u.cm**3
problem.parameters["epsilon_e"].initial_value = 0.1
problem.parameters["epsilon_B"].initial_value = 0.1

# Fix all other parameters.  Pass Quantities directly so that the initial_value
# setter can convert to the model's base units (e.g. M_ej in g, not M_sun).
for name in ["M_ej", "n", "s", "delta", "p", "gamma_min", "gamma_max", "f", "theta", "smoothing_s", "D"]:
    problem.parameters[name].initial_value = true_params[name]
    problem.parameters[name].freeze = True

print("Free parameters:")
for name, par in problem.parameters.items():
    if not par.freeze:
        print(f"  {name}")

# %%
# Run MCMC
# ---------
#
# We run the MCMC with 32 walkers for 15,000 steps. In a real analysis you
# would check convergence using the trace plots and autocorrelation lengths.

sampler = EmceeSampler(problem, n_walkers=32)
result = sampler.run(15_000, progress=True)
samples = result.get_flat_samples(burn=5000, thin=10)

print(f"\nPosterior samples: {samples.shape[0]}")

# %%
# Corner Plot: Posterior Distributions
# -------------------------------------
#
# The corner plot shows the 1D and 2D marginal posteriors for all free
# parameters. Vertical lines mark the true input values.

free_params = [name for name, par in problem.parameters.items() if not par.freeze]
true_vals = [
    np.log10(true_params["E_ej"].to_value(u.erg)),
    np.log10(true_params["rho_0"].to_value(u.g / u.cm**3)),
    np.log10(true_params["epsilon_e"]),
    np.log10(true_params["epsilon_B"]),
]

fig = result.corner_plot(burn=5000, thin=10, parameters=free_params)
plt.suptitle("Posterior Distributions: Chevalier Shock Parameters", fontsize=13, y=1.02)
plt.show()

# %%
# Print Posterior Summaries
# --------------------------
#
# Report the median and 68% credible intervals for each inferred parameter.

print("\n=== Posterior Summary ===")
print(f"{'Parameter':<15}  {'16th':>12}  {'50th (med)':>12}  {'84th':>12}  {'True':>12}")
print("-" * 70)
for i, name in enumerate(free_params):
    lo = np.percentile(samples[:, i], 16)
    med = np.percentile(samples[:, i], 50)
    hi = np.percentile(samples[:, i], 84)
    print(f"  {name:<13}  {lo:>12.3f}  {med:>12.3f}  {hi:>12.3f}  {true_vals[i]:>12.3f}")

# %%
# Posterior Predictive Check
# ---------------------------
#
# Draw light curves from the posterior and overlay them on the data at each
# frequency. Good model fit means the posterior envelope covers the data points.

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

t_dense = np.geomspace(10.0, 300.0, 200) * u.day
idx = rng.choice(samples.shape[0], 80, replace=False)

for ax, (freq, color) in zip(axes, zip(frequencies, colors)):
    # Posterior predictive samples
    for i in idx:
        p_sample = problem.unpack_free_parameters(samples[i])
        lc = []
        for t in t_dense:
            v = {"frequency": freq.to(u.Hz), "time": t.to(u.s)}
            lc.append(shock_model.forward_model(v, p_sample).flux_density.to_value(u.mJy))
        ax.loglog(t_dense.to_value(u.day), lc, color=color, alpha=0.05)

    # True model
    true_lc = []
    for t in t_dense:
        v = {"frequency": freq.to(u.Hz), "time": t.to(u.s)}
        true_lc.append(shock_model.forward_model(v, true_params).flux_density.to_value(u.mJy))
    ax.loglog(t_dense.to_value(u.day), true_lc, color="black", lw=2, ls="--", label="True")

    # Data
    obs = syn_data[freq.to_value(u.GHz)]
    ax.errorbar(
        epochs.to_value(u.day),
        obs.to_value(u.mJy),
        yerr=(noise_frac * obs).to_value(u.mJy),
        fmt="o",
        color=color,
        capsize=3,
        ms=6,
        label="Data",
    )

    ax.set_title(f"{freq.to_value(u.GHz):.0f} GHz")
    ax.set_ylabel("Flux Density [mJy]")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.3)

axes[2].set_xlabel("Time post-explosion [days]")
axes[3].set_xlabel("Time post-explosion [days]")
plt.suptitle("Posterior Predictive Check", fontsize=13)
plt.tight_layout()
plt.show()

# %%
# Physical Interpretation
# ------------------------
#
# The inferred :math:`\rho_0` can be directly converted to the progenitor mass-loss
# rate using the standard wind density relation:
#
# .. math::
#
#     \rho_{\rm w}(r) = \frac{\dot{M}}{4\pi v_{\rm w} r^2}
#     = \rho_0 \left(\frac{r}{r_0}\right)^{-2}
#
# at fiducial radius :math:`r_0 = 10^{14}` cm.

r0 = 1e14 * u.cm
v_wind = 10.0 * u.km / u.s  # assumed wind velocity

rho0_med = 10 ** np.median(samples[:, 1]) * u.g / u.cm**3
Mdot = (4 * np.pi * r0**2 * rho0_med * v_wind.to(u.cm / u.s)).to(u.Msun / u.yr)

print(f"\n=== Physical parameter interpretation ===")
print(f"  Inferred rho_0   : {rho0_med:.2e}")
print(f"  => M_dot / v_w   : {Mdot:.2e}")
print(f"  (assuming v_w = {v_wind})")
print(f"\n  True rho_0       : {true_params['rho_0']:.2e}")
Mdot_true = (4 * np.pi * r0**2 * true_params["rho_0"].to(u.g / u.cm**3) * v_wind.to(u.cm / u.s)).to(u.Msun / u.yr)
print(f"  True M_dot/v_w   : {Mdot_true:.2e}")
