"""
=====================================================
Fit a Free Systematic Error Term (Nuisance Parameter)
=====================================================

Reported uncertainties are often too small: calibration errors, unmodeled variability, or scatter that the
model does not capture all add noise that the quoted error bars do not include. If we fit such data with the
quoted errors alone, the posterior is too narrow and the fit is dragged around by individual points.

A standard remedy is to let the data determine an additional error scale. In Trilobite this is a
*nuisance parameter*: a parameter that belongs to the **likelihood** rather than to the physical model. It is
sampled alongside the model parameters, but the model never sees it.

In this example we generate a synthetic radio SED whose points scatter by about 0.1 dex around the model even
though the quoted errors are only a few percent. We then fit it with
:class:`~trilobite.inference.likelihood.base.LogGaussianCensoredSysLikelihood`, which adds a free systematic
term, ``sys_dex``, in quadrature to every uncertainty, and recover the hidden scatter.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import Table

from trilobite.data import InferenceData
from trilobite.data.photometry import RadioPhotometryEpochContainer
from trilobite.inference import LogGaussianCensoredSysLikelihood
from trilobite.inference.problem import InferenceProblem
from trilobite.inference.sampling.mcmc import EmceeSampler
from trilobite.inference.transform import Log10Transform
from trilobite.models.SEDs.synchrotron import Synchrotron_SSA_SBPL_Model
from trilobite.utils.plot_utils import set_plot_style

# %%
# Synthetic data with underestimated errors
# -----------------------------------------
# We use the same single-epoch synchrotron SED model as in the other inference examples. The measured fluxes
# are the model flux multiplied by a log-normal factor with a standard deviation of ``true_scatter`` dex, but
# we only *report* a 5% uncertainty. One channel falls below a flux floor and becomes an upper limit.
rng = np.random.default_rng(42)

sed_model = Synchrotron_SSA_SBPL_Model()
true_params = {"norm": 5.0 * u.mJy, "nu_break": 4.0 * u.GHz, "p": 3.0, "s": -1.0}

frequencies = u.Quantity([0.3, 0.6, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0], u.GHz)
true_flux = sed_model.forward_model({"frequency": frequencies}, true_params).flux_density

true_scatter = 0.1  # dex, hidden from the likelihood
reported_fraction = 0.05  # the (too small) quoted fractional error
flux_floor = 0.4 * u.mJy

flux = true_flux * 10 ** rng.normal(0.0, true_scatter, size=true_flux.shape)
is_limit = flux < flux_floor

# %%
# The data go into an :class:`~trilobite.data.core.InferenceData` object exactly as in the other examples.
# Detections carry the reported 5% error, and the limit is stored as an upper limit.
table = Table()
table["freq"] = frequencies
table["flux_density"] = flux
table["flux_density_error"] = reported_fraction * flux
table["flux_upper_limit"] = np.full(frequencies.shape, np.nan) * u.mJy
table["flux_upper_limit"][is_limit] = flux_floor
table["flux_density"][is_limit] = np.nan * u.mJy
table["flux_density_error"][is_limit] = flux_floor * reported_fraction

photometry = RadioPhotometryEpochContainer(table)
data = InferenceData.from_table(
    sed_model,
    photometry.table,
    variables={"frequency": "freq"},
    observables={"flux_density": ("flux_density", "flux_density_error", "flux_upper_limit", None)},
)

set_plot_style()
fig, ax = plt.subplots(figsize=(8, 5))
grid = np.logspace(8.3, 10.8, 200) * u.Hz
ax.plot(
    grid.to_value(u.GHz), sed_model.forward_model({"frequency": grid}, true_params).flux_density.to_value(u.mJy), "k"
)
ax.errorbar(
    frequencies[~is_limit].to_value(u.GHz),
    flux[~is_limit].to_value(u.mJy),
    yerr=(reported_fraction * flux[~is_limit]).to_value(u.mJy),
    fmt="o",
    color="C3",
    label="Data (reported 5% errors)",
)
ax.scatter(frequencies[is_limit].to_value(u.GHz), flux[is_limit].to_value(u.mJy), marker="v", color="C0", label="Limit")
ax.set(xscale="log", yscale="log", xlabel="Frequency [GHz]", ylabel="Flux density [mJy]")
ax.legend()
plt.show()

# %%
# The points clearly scatter by much more than their error bars allow.

# %%
# A likelihood with a free systematic term
# ----------------------------------------
# :class:`~trilobite.inference.likelihood.base.LogGaussianCensoredSysLikelihood` works in :math:`\log_{10}` of the
# flux, so a single dex-valued term describes scatter across a spectrum that spans decades. Its total
# uncertainty for each point is
#
# .. math::
#
#    \sigma_i^2 = \sigma_{{\rm meas},i}^2 + \sigma_{\rm sys}^2,
#
# where :math:`\sigma_{{\rm meas},i}` is the reported error converted to dex and :math:`\sigma_{\rm sys}` is the
# nuisance parameter ``sys_dex``. The same term widens the censored limit. Because :math:`\sigma_i` depends on a
# parameter, the :math:`\ln(2\pi\sigma_i^2)` normalization stays in the likelihood, and the sampler finds the
# error scale that the data support without any reduced-:math:`\chi^2=1` condition.
likelihood = LogGaussianCensoredSysLikelihood(model=sed_model, data=data)

print("Nuisance parameters:", [p.name for p in likelihood.nuisance_parameters])

# %%
# The :class:`~trilobite.inference.problem.InferenceProblem` lists the nuisance parameter after the model's
# own parameters, and it is configured in exactly the same way: it needs a prior and can be frozen or
# transformed. Since ``sys_dex`` must be positive and its scale is unknown, a log-uniform prior and a
# :class:`~trilobite.inference.transform.Log10Transform` are natural choices.
problem = InferenceProblem(likelihood=likelihood)

problem.set_prior("norm", "uniform", lower=1e-3 * u.Jy, upper=10.0 * u.Jy)
problem.set_prior("nu_break", "uniform", lower=1 * u.GHz, upper=50 * u.GHz)
problem.set_prior("p", "uniform", lower=2.0, upper=5.0)
problem.set_prior("sys_dex", "loguniform", lower=1e-3, upper=1.0)
problem.parameters["sys_dex"].transform = Log10Transform()

# The shape parameter is not inferred here.
problem.parameters["s"].initial_value = true_params["s"]
problem.parameters["s"].freeze = True

print(problem.summary())

# %%
# Sampling
# --------
# The sampler is configured exactly as for any other problem; the walkers are initialized in a small ball around
# the best-fit point, and ``sys_dex`` is sampled in log space because of its transform.
sampler = EmceeSampler(problem, n_walkers=32)
result = sampler.run(4_000, progress=False)

samples = result.get_flat_samples(burn=1_000, thin=10)
names = result.parameter_names
sys_dex = samples[:, names.index("sys_dex")]

lo, med, hi = np.percentile(sys_dex, [16, 50, 84])
print(f"sys_dex = {med:.3f} (+{hi - med:.3f} / -{med - lo:.3f}) dex   [true scatter: {true_scatter} dex]")

# %%
# The posterior on ``sys_dex`` is centered on the scatter that we injected, even though no error bar hinted at
# it. The reported errors alone correspond to only about 0.02 dex.
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(sys_dex, bins=40, color="C0", alpha=0.7, density=True)
ax.axvline(true_scatter, color="k", ls="--", label="Injected scatter")
ax.set(xlabel=r"$\sigma_{\rm sys}$ [dex]", ylabel="Posterior density")
ax.legend()
plt.show()

# %%
# Because the nuisance parameter is an ordinary member of the problem, it appears in the trace and corner
# plots alongside the model parameters. Samples are returned in physical space by default.
fig = result.corner_plot(burn=1_000, thin=10, parameters=["norm", "nu_break", "p", "sys_dex"], truths={"p": 3.0})
plt.show()

# %%
# Using the posterior
# -------------------
# A problem's full parameter dictionary, nuisance values included, can be passed directly to the model: the
# model reads only its own parameters. Here we draw posterior samples and overlay the predicted spectra.
fig, ax = plt.subplots(figsize=(8, 5))
for index in rng.choice(samples.shape[0], 80, replace=False):
    params = problem.unpack_free_parameters(samples[index])
    ax.plot(
        grid.to_value(u.GHz),
        sed_model.forward_model({"frequency": grid}, {**problem.initial_full_params, **params}).flux_density.to_value(
            u.mJy
        ),
        color="C0",
        alpha=0.08,
    )

ax.errorbar(
    frequencies[~is_limit].to_value(u.GHz),
    flux[~is_limit].to_value(u.mJy),
    yerr=(reported_fraction * flux[~is_limit]).to_value(u.mJy),
    fmt="o",
    color="C3",
    label="Data",
)
ax.set(xscale="log", yscale="log", xlabel="Frequency [GHz]", ylabel="Flux density [mJy]")
ax.legend()
plt.show()
