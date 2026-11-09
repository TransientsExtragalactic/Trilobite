r"""
Fitting a Magnetar-Powered Superluminous Supernova to Swift/UVOT Photometry
============================================================================

This example fits the Swift/UVOT light curves of the Type I superluminous supernova SN 2020znr with
:class:`~trilobite.models.supernovae.arnett.MagnetarArnettCutoffModel`, set up like the magnetar model of MOSFiT
(:footcite:t:`Nicholl_2017`), and checks the result against the published MOSFiT fit of the same object. The photometry is
from :footcite:t:`Hinkle_2025` ("A Swift Fix II: Physical Parameters of Type I Superluminous Supernovae",
arXiv:2309.03270), who re-reduced the UVOT data with the updated calibration, subtracted the host, and fit 26
superluminous supernovae with MOSFiT. SN 2020znr is one of the best sampled: six filters, 22 epochs each, from the rise through the peak and
about four months of decline.

We will:

1. load the photometry into an
   :class:`~trilobite.data.optical_photometry.OpticalPhotometryContainer`,
2. build the model and sample the posterior with ``emcee``,
3. compare the posterior with the published parameters, and check that the published parameters themselves
   reproduce the data, and
4. see how much the UV suppression of the spectrum matters.

.. note::

    Loading the Swift filter curves from the Spanish Virtual Observatory needs the optional dependency ``astroquery``
    (``pip install trilobite[optical]``).
"""

# %%
# The Data
# --------
#
# The photometry is host-subtracted and corrected for Galactic extinction, in the AB system. Every point is a detection.
# We measure time in days since the first Swift epoch.

from pathlib import Path

import astropy.constants as const
import corner
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.time import Time
from scipy.optimize import differential_evolution

from trilobite.data.optical_photometry import OpticalPhotometryContainer
from trilobite.inference import LogGaussianCensoredLikelihood
from trilobite.inference.problem import InferenceProblem
from trilobite.inference.sampling.mcmc import EmceeSampler
from trilobite.models.supernovae import MagnetarArnettCutoffModel, MagnetarArnettModel
from trilobite.utils.phot_utils import FilterBundle, load_filter_from_svo
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

# sphinx-gallery runs an example from its own directory and does not define ``__file__``.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA = HERE.parents[1] / "_data" / "sn2020znr_swift_uvot.csv"
table = Table.read(DATA, format="ascii.csv", comment="#")
table.rename_columns(["mjd", "band"], ["time", "band_name"])
table["mag_ab_upper_limit"] = np.nan  # no non-detections for this object

MJD_0 = table["time"].min()
photometry = OpticalPhotometryContainer.from_table(
    table, internal_time_format="mjd", time_start=Time(MJD_0, format="mjd")
)
print(f"{photometry.n_obs} observations, {photometry.n_detections} detections")
print(f"Time span: {photometry.time.min():.1f} to {photometry.time.max():.1f} after MJD {MJD_0:.1f}")

# %%
# The Model
# ---------
#
# :class:`~trilobite.models.supernovae.arnett.MagnetarArnettCutoffModel` computes the bolometric luminosity of a
# magnetar-heated ejecta with the one-zone diffusion engine of :ref:`arnett`. The photosphere expands at a constant
# velocity until it cools to a temperature floor and then recedes, and it radiates a blackbody whose flux below 3000
# Angstrom is suppressed (:ref:`blackbody_cutoff`), as is seen in the spectra of superluminous supernovae. The spectrum
# is convolved with the Swift filters of a :class:`~trilobite.utils.phot_utils.FilterBundle`.
#
# To follow MOSFiT, we use the redshift and cosmology of the paper (:math:`z = 0.1`,
# :math:`H_0 = 69.6\,\mathrm{km\,s^{-1}\,Mpc^{-1}}`, :math:`\Omega_m = 0.29`), and ``trapping_mode="emergence"``, which
# applies the leakage of the magnetar's radiation to the luminosity at the time of observation, as MOSFiT does
# (see :ref:`arnett_trapping_mode`).
#
# The explosion epoch is not known, so it is fit: the parameter ``t_exp`` is the explosion epoch on the time axis of
# the data. MOSFiT defines the magnetar through its rotational energy and spin-down time (Eqs. 3 and 4 of Nicholl et
# al.), for a 1.4 solar mass neutron star: :math:`E_{\rm mag} = 2.6\times10^{52}\,(P/1\,{\rm ms})^{-2}` erg and
# :math:`t_{\rm mag} = 1.3\times10^{5}\,(P/1\,{\rm ms})^{2}\,(B_\perp/10^{14}\,{\rm G})^{-2}` s. Here the magnetar is
# defined by a moment of inertia and a radius, which we choose to reproduce those two relations, so that the
# spin period and the magnetic field mean the same thing in both.

BANDS = ["V", "B", "U", "UVW1", "UVM2", "UVW2"]
bundle = FilterBundle({band: load_filter_from_svo(f"Swift/UVOT.{band}") for band in BANDS})
cosmology = FlatLambdaCDM(H0=69.6, Om0=0.29)

omega_1ms = 2.0 * np.pi / 1e-3
I_NS = 2.0 * 2.6e52 / omega_1ms**2  # E_mag = I omega^2 / 2
R_NS = (3.0 * I_NS * const.c.cgs.value**3 / ((1e14) ** 2 * omega_1ms**2 * 1.3e5)) ** (1.0 / 6.0)  # t_mag, for n = 3
print(f"I_ns = {I_NS:.3e} g cm^2, R_ns = {R_NS:.3e} cm")

model = MagnetarArnettCutoffModel(bundle, redshift=0.1, cosmology=cosmology, trapping_mode="emergence")
print(f"Luminosity distance: {model.luminosity_distance.to(u.Mpc):.0f}")

# %%
# The Likelihood and Priors
# -------------------------
#
# The container converts the magnitudes to flux densities and
# :meth:`~trilobite.data.optical_photometry.OpticalPhotometryContainer.to_inference_data` pairs each with its filter in
# the bundle. We compare model and data in :math:`\log_{10}` flux with
# :class:`~trilobite.inference.likelihood.base.LogGaussianCensoredLikelihood`, with a systematic floor of 0.05 dex
# (0.12 mag), which is the median white-noise term that MOSFiT fits for these objects.
#
# The priors are those of :footcite:t:`Nicholl_2017`: flat in the spin period (0.7 to 20 ms), flat in the logarithm of the
# magnetic field and of the ejecta mass, and flat in the explosion epoch (up to 100 days before the first Swift
# epoch). The kinetic energy takes the place of their ejecta velocity, and has a wide logarithmic prior. MOSFiT's
# Gaussian prior on the final temperature (6000 K) helps it fit the late ground-based photometry that we do not have,
# so we use a flat prior over their allowed range, 3000 to 10000 K.
#
# Two parameters are fixed. The opacity of the ejecta, :math:`\kappa = 0.1\,\mathrm{cm^2\,g^{-1}}`, is degenerate with
# the ejecta mass, which is set by their product. And the opacity to the magnetar's radiation,
# :math:`\kappa_\gamma = 0.1\,\mathrm{cm^2\,g^{-1}}`, is degenerate with the spin-down: both make the light curve fall
# as :math:`t^{-2}` at late times, so with only these data, freeing :math:`\kappa_\gamma` lets the chain wander along a
# ridge towards fast spins and weak fields that the data cannot tell apart. (MOSFiT breaks the degeneracy with a longer
# light curve.)

data = photometry.to_inference_data(model)
likelihood = LogGaussianCensoredLikelihood(model=model, data=data, sys_dex=0.05)
problem = InferenceProblem(likelihood=likelihood)

problem.set_prior("M_ej", "loguniform", lower=0.1 * u.Msun, upper=100 * u.Msun)
problem.set_prior("E_ej", "loguniform", lower=1e50 * u.erg, upper=1e53 * u.erg)
problem.set_prior("P_spin", "uniform", lower=0.7 * u.ms, upper=20 * u.ms)
problem.set_prior("B_dipole", "loguniform", lower=1e12 * u.G, upper=1e15 * u.G)
problem.set_prior("T_floor", "uniform", lower=3000 * u.K, upper=10000 * u.K)
problem.set_prior("t_exp", "uniform", lower=-100 * u.day, upper=0 * u.day)

initial = {
    "M_ej": 15 * u.Msun,
    "E_ej": 5e51 * u.erg,
    "P_spin": 2.5 * u.ms,
    "B_dipole": 3e13 * u.G,
    "T_floor": 9000 * u.K,
    "t_exp": -25 * u.day,
}
fixed = {
    "kappa": 0.1 * u.cm**2 / u.g,
    "kappa_magnetar": 0.1 * u.cm**2 / u.g,
    "braking_index": 3.0,
    "I_ns": I_NS * u.g * u.cm**2,
    "R_ns": R_NS * u.cm,
    "lambda_cut": 3000 * u.AA,
}
for name, value in initial.items():
    problem.parameters[name].initial_value = value
for name, value in fixed.items():
    problem.parameters[name].initial_value = value
    problem.parameters[name].freeze = True

print("Free parameters:", problem.free_parameter_names)

# %%
# Sampling
# --------
#
# The model costs well under a millisecond per evaluation, so a 24-walker chain of 4000 steps takes a few minutes. That
# is enough for the medians and 68% intervals, but a publication-quality run would be longer: we print how many
# autocorrelation times the chain spans after burn-in.

result = EmceeSampler(problem, n_walkers=24).run(4000, progress=True)

BURN, THIN = 1500, 10
names = list(result.parameter_names)
tau = result.compute_autocorr_time(quiet=True)
print(f"Autocorrelation times (steps): {dict(zip(names, np.round(tau)))}")
print(f"Post burn-in steps per autocorrelation time: {(4000 - BURN) / tau.max():.0f}")

# %%
# The Posterior
# -------------
#
# The sampler works in CGS units, so we convert to the units used in the literature before looking at the posterior.
# The velocity that MOSFiT calls :math:`v_{\rm phot}` is, here, :math:`v_{\rm sc} = \sqrt{10 E_{\rm ej} / 3 M_{\rm ej}}`,
# the velocity implied by the kinetic energy of a uniform sphere, which sets the diffusion time, the leakage, and the
# radius of the photosphere.

samples = result.get_flat_samples(burn=BURN, thin=THIN)
column = {name: samples[:, i] for i, name in enumerate(names)}

M_ej = column["M_ej"] / u.Msun.to(u.g)  # solar masses
E_ej = column["E_ej"] / 1e51  # 1e51 erg
P_spin = column["P_spin"] * 1e3  # ms
B_dipole = column["B_dipole"] / 1e14  # 1e14 G
t_exp = column["t_exp"] / 86400.0  # days before the first epoch
T_floor = column["T_floor"]  # K
v_sc = np.sqrt(10.0 * column["E_ej"] / (3.0 * column["M_ej"])) / 1e8  # 1e3 km/s

fig = corner.corner(
    np.column_stack([M_ej, E_ej, P_spin, B_dipole, T_floor, t_exp]),
    labels=[
        r"$M_{\rm ej}\ [M_\odot]$",
        r"$E_{\rm ej}\ [10^{51}\,{\rm erg}]$",
        r"$P\ [{\rm ms}]$",
        r"$B\ [10^{14}\,{\rm G}]$",
        r"$T_{\rm floor}\ [{\rm K}]$",
        r"$t_{\rm exp}\ [{\rm d}]$",
    ],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
)
plt.show()

# %%
# Comparison with MOSFiT
# ----------------------
#
# Hinkle et al. report the medians and 68% intervals of their MOSFiT fit to these Swift data and to ground-based light
# curves. The spin period, magnetic field and ejecta mass mean the same thing in both models.

mosfit = {  # median, +err, -err
    "P (ms)": (2.79, 0.25, 0.50),
    "B (1e14 G)": (0.46, 1.33, 0.27),
    "M_ej (Msun)": (16.22, 1.98, 1.76),
    "v (1e3 km/s)": (5.18, 0.10, 0.10),
}
ours = {"P (ms)": P_spin, "B (1e14 G)": B_dipole, "M_ej (Msun)": M_ej, "v (1e3 km/s)": v_sc}

print(f"{'':16s}  {'This fit':>24s}  {'MOSFiT (Hinkle+25)':>24s}")
for key, (mid, up, down) in mosfit.items():
    lo, med, hi = np.percentile(ours[key], [16, 50, 84])
    print(f"{key:16s}  {med:8.2f} +{hi - med:5.2f} -{med - lo:5.2f}  {mid:8.2f} +{up:5.2f} -{down:5.2f}")

# %%
# The spin period, the magnetic field and the ejecta mass fall within the MOSFiT intervals. The velocity is higher by
# about a quarter: it is correlated with the opacity and the ejecta mass, which we fix and MOSFiT fits, and MOSFiT also
# has a spectroscopic prior on it. The final temperature, :math:`T_{\rm floor}`, is also higher than the 6000 K that
# MOSFiT favors, because the Swift data end before the light curve reaches its plateau and prefer a hotter photosphere.
# We should not expect better agreement from three months of Swift data alone.
#
# Do the Published Parameters Reproduce the Data?
# -----------------------------------------------
#
# A direct test of whether our model is the same as theirs is to fix the parameters that the paper reports (the spin
# period, the field, the neutron star mass, the ejecta mass, and the velocity) at their medians, and see whether the
# model still fits the Swift photometry. The paper does not give the remaining parameters, which are the opacity, the
# final temperature and the explosion epoch, so we fit those with a quick global optimization. If the models were
# different, the published parameters would not reproduce the light curves.


def best_fit(fit_model, bounds, fixed_parameters):
    """The maximum-likelihood parameters (CGS) within ``bounds``, and the log-likelihood there."""
    fit_likelihood = LogGaussianCensoredLikelihood(
        model=fit_model, data=photometry.to_inference_data(fit_model), sys_dex=0.05
    )
    free_names = list(bounds)

    def negative(x):
        value = fit_likelihood.log_likelihood({**fixed_parameters, **dict(zip(free_names, x))})
        return -value if np.isfinite(value) else 1e30

    optimum = differential_evolution(negative, list(bounds.values()), seed=1, popsize=12, maxiter=200, tol=1e-8)
    return dict(zip(free_names, optimum.x)), -optimum.fun


M_SUN = u.Msun.to(u.g)
published = {
    "M_ej": 16.22 * M_SUN,
    "E_ej": 0.3 * 16.22 * M_SUN * (5.18e8) ** 2,  # v_sc = v_phot
    "P_spin": 2.79e-3,
    "B_dipole": 0.46e14,
    "I_ns": I_NS * (1.70 / 1.4) ** 1.5,  # their neutron star mass of 1.70 solar masses
    "R_ns": R_NS,
    "braking_index": 3.0,
    "kappa_magnetar": 0.1,
    "lambda_cut": 3000.0,
}
published_fit, published_lnL = best_fit(
    model,
    {"kappa": (0.05, 0.2), "T_floor": (3000.0, 10000.0), "t_exp": (-100 * 86400.0, 0.0)},
    published,
)

fixed_values = {name: problem.parameters[name].initial_value for name in fixed}
median = dict(zip(names, np.median(samples, axis=0)))
our_lnL = likelihood.log_likelihood({**median, **fixed_values})
print(f"ln L, published parameters (opacity, T_floor and t_exp fit): {published_lnL:8.1f}")
print(f"ln L, this fit (posterior median):                           {our_lnL:8.1f}")
print(f"The published parameters need kappa = {published_fit['kappa']:.2f}, T_floor = {published_fit['T_floor']:.0f} K")

# %%
# With five parameters fixed at values that came from a fit to different data, the published parameters give a
# log-likelihood about 20 lower than our own fit. The light curves that they predict (red dashed lines, below) have the
# right shape and timescale in every filter, but peak a little later and fall more slowly in the ultraviolet than the
# data. That is agreement in the physics of the model, but it is not an exact reproduction, which we should not expect
# from different data, different priors and a different implementation. In the figure we draw 50 samples from the
# posterior and plot the model in each filter, along with the data, as a function of MJD.

rng = np.random.default_rng(0)
draws = samples[rng.choice(len(samples), size=50, replace=False)]

mjd_grid = np.linspace(MJD_0 - 5.0, MJD_0 + 145.0, 120)
t_grid = (mjd_grid - MJD_0) * 86400.0


def ab_mag(flux_density):
    return -2.5 * np.log10(flux_density / 3.631e-20)


fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for i, (band, ax) in enumerate(zip(BANDS, axes.flat)):
    variables = {"band_index": np.full(t_grid.size, i), "time": t_grid}
    for theta in draws:
        parameters = dict(zip(names, theta), **fixed_values)
        ax.plot(mjd_grid, ab_mag(model.forward_model(variables, parameters).flux.value), color="C0", alpha=0.1, lw=1)
    parameters = {**published, **published_fit}
    ax.plot(mjd_grid, ab_mag(model.forward_model(variables, parameters).flux.value), color="C3", ls="--", lw=1.5)
    selected = photometry.band_name == band
    ax.errorbar(
        MJD_0 + photometry.time[selected].to_value(u.day),
        photometry.mag[selected],
        yerr=photometry.mag_error[selected],
        fmt="o",
        color="k",
        ms=3,
    )
    ax.set_title(band)
    ax.invert_yaxis()
    ax.set_xlim(mjd_grid[0], mjd_grid[-1])
for ax in axes[1]:
    ax.set_xlabel("MJD")
for ax in axes[:, 0]:
    ax.set_ylabel("AB magnitude")
plt.tight_layout()
plt.show()

# %%
# How Much Does the UV Suppression Matter?
# ----------------------------------------
#
# Superluminous supernovae are fainter in the ultraviolet than a blackbody, and the Swift filters reach well into the
# ultraviolet, so the data should be able to tell a modified blackbody from a plain one. We fit the same model with a
# plain blackbody photosphere, :class:`~trilobite.models.supernovae.arnett.MagnetarArnettModel`, with the same fixed
# parameters, and compare the best log-likelihood of the two, and the typical residual in each filter, in magnitudes
# (positive means that the model is fainter than the data).

bounds = {
    "M_ej": (3 * M_SUN, 40 * M_SUN),
    "E_ej": (5e50, 2e52),
    "T_floor": (4000.0, 10000.0),
    "t_exp": (-60 * 86400.0, -5 * 86400.0),
    "P_spin": (1e-3, 6e-3),
    "B_dipole": (5e12, 2e14),
}
fixed_cgs = {
    "kappa": 0.1,
    "kappa_magnetar": 0.1,
    "braking_index": 3.0,
    "I_ns": I_NS,
    "R_ns": R_NS,
    "lambda_cut": 3000.0,
}
plain_model = MagnetarArnettModel(bundle, redshift=0.1, cosmology=cosmology, trapping_mode="emergence")
fits = {
    "plain blackbody": (plain_model, *best_fit(plain_model, bounds, fixed_cgs)),
    "UV-suppressed": (model, *best_fit(model, bounds, fixed_cgs)),
}

band_index = np.asarray(data.x["band_index"])
print(f"{'':16s} {'ln L':>8s}   " + "  ".join(f"{band:>6s}" for band in BANDS))
for label, (fitted_model, parameters, lnL) in fits.items():
    flux = fitted_model.forward_model(
        {"band_index": band_index, "time": data.x["time"]}, {**fixed_cgs, **parameters}
    ).flux.value
    residual = ab_mag(flux) - photometry.mag
    medians = [np.median(residual[band_index == i]) for i in range(len(BANDS))]
    print(f"{label:16s} {lnL:8.1f}   " + "  ".join(f"{m:+6.2f}" for m in medians))

# %%
# The UV-suppressed spectrum improves the log-likelihood by about 30, and reduces the offsets in V and U by about a third,
# but it does not remove them: a blackbody with a linear UV cutoff is still only an approximation of the spectrum, and the
# systematic floor of 0.05 dex that we gave the likelihood is there for what remains. This is why MOSFiT models the
# spectrum in this way.
#
# %%
# The Photosphere
# ---------------
#
# The model also predicts the evolution of the bolometric luminosity and of the photosphere. At first the photosphere
# is the surface of the ejecta, and cools as the luminosity falls and the radius grows. Once it reaches the temperature floor,
# the photosphere stops cooling and the radius shrinks as the luminosity declines.

t_since_explosion = np.linspace(2.0, 160.0, 160) * u.day
histories = []
for theta in draws:
    parameters = dict(zip(names, theta), **fixed_values)
    photosphere = model.compute_photosphere(t_since_explosion + parameters["t_exp"] * u.s, parameters)
    histories.append(
        [photosphere.luminosity.value, photosphere.temperature.value, photosphere.radius.to_value(u.cm) / 1e15]
    )
histories = np.array(histories)  # (draws, quantity, time)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
for ax, curves, label in zip(
    axes,
    np.moveaxis(histories, 1, 0),
    [r"$L_{\rm bol}\ [{\rm erg\,s^{-1}}]$", r"$T\ [{\rm K}]$", r"$R\ [10^{15}\,{\rm cm}]$"],
):
    ax.plot(t_since_explosion.value, curves.T, color="C0", alpha=0.1, lw=1)
    ax.set_xlabel("Time since explosion [d]")
    ax.set_ylabel(label)
axes[0].set_yscale("log")
plt.tight_layout()
plt.show()

# %%
# .. footbibliography::
