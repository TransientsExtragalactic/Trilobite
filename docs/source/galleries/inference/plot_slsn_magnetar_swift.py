r"""
Fitting a Magnetar-Powered Superluminous Supernova to Swift/UVOT Photometry
============================================================================

This example fits the Swift/UVOT light curves of the Type I superluminous supernova SN 2020znr with
:class:`~trilobite.models.supernovae.arnett.MagnetarArnettModel`, and checks the result against the
published fit of the same data. The photometry is from Hinkle et al. (2023), "A Swift Fix II: Physical Parameters of
Type I Superluminous Supernovae" (arXiv:2309.03270), who re-reduced the UVOT data with the updated calibration, subtracted
the host, and fit the light curves of 26 of them with the magnetar model in MOSFiT. SN 2020znr is one of the best
sampled: six filters, 22 epochs each, from the rise through the peak and about four months of decline.

We will:

1. load the photometry into an
   :class:`~trilobite.data.optical_photometry.OpticalPhotometryContainer`,
2. build the model with the Swift filter curves and the paper's cosmology,
3. sample the posterior with ``emcee``, and
4. compare the posterior with the paper's MOSFiT parameters and the model with the data.

.. note::

    Loading the Swift filter curves from the Spanish Virtual Observatory needs the optional dependency ``astroquery``
    (``pip install trilobite[optical]``).
"""

# %%
# The Data
# --------
#
# The photometry is host-subtracted and corrected for Galactic extinction, in the AB system. Every point is a detection.
# A time axis is needed in days since some reference, so we measure time from the first Swift epoch.

from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy.time import Time

from trilobite.data.optical_photometry import OpticalPhotometryContainer
from trilobite.inference import LogGaussianCensoredLikelihood
from trilobite.inference.problem import InferenceProblem
from trilobite.inference.sampling.mcmc import EmceeSampler
from trilobite.models.core.parameters import ModelParameter
from trilobite.models.supernovae import MagnetarArnettModel
from trilobite.utils.phot_utils import FilterBundle, load_filter_from_svo
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

DATA = Path(__file__).resolve().parents[2] / "_data" / "sn2020znr_swift_uvot.csv"
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
# :class:`~trilobite.models.supernovae.arnett.MagnetarArnettModel` computes the bolometric luminosity of a magnetar-heated
# ejecta with the one-zone diffusion engine of :ref:`arnett`, radiates it from a blackbody photosphere, and convolves
# the spectrum with the filters of a :class:`~trilobite.utils.phot_utils.FilterBundle`. We build the bundle from the
# UVOT response curves, and use the redshift and cosmology of the paper (:math:`z = 0.1`,
# :math:`H_0 = 69.6\,\mathrm{km\,s^{-1}\,Mpc^{-1}}`, :math:`\Omega_m = 0.29`).
#
# The model takes the time since explosion, but here the explosion epoch is unknown and has to be fit. We add it as a
# parameter in a small subclass, which shifts the observation times before they reach the model.

BANDS = ["V", "B", "U", "UVW1", "UVM2", "UVW2"]
bundle = FilterBundle({band: load_filter_from_svo(f"Swift/UVOT.{band}") for band in BANDS})


class SLSNMagnetarModel(MagnetarArnettModel):
    """:class:`MagnetarArnettModel` with an explosion epoch, so that ``time`` can be measured from any reference."""

    PARAMETERS = MagnetarArnettModel.PARAMETERS + (
        ModelParameter("t_exp", 0.0, base_units=u.s, description="Explosion epoch.", latex=r"$t_{\rm exp}$"),
    )

    def _compute_sed(self, nu_grid, t_unique, parameters):
        return super()._compute_sed(nu_grid, t_unique - parameters["t_exp"], parameters)


model = SLSNMagnetarModel(bundle, redshift=0.1, cosmology=FlatLambdaCDM(H0=69.6, Om0=0.29))
print(f"Luminosity distance: {model.luminosity_distance.to(u.Mpc):.0f}")

# %%
# The Likelihood and Priors
# -------------------------
#
# The container converts the magnitudes to flux densities and
# :meth:`~trilobite.data.optical_photometry.OpticalPhotometryContainer.to_inference_data` pairs each with its filter in the
# bundle. We compare model and data in :math:`\log_{10}` flux with
# :class:`~trilobite.inference.likelihood.base.LogGaussianCensoredLikelihood`, which treats a factor of two too bright
# the same as a factor of two too faint. A blackbody photosphere cannot reproduce every feature of the spectrum, so
# we add a systematic floor of 0.08 dex (20%) in quadrature, which stops the many high signal-to-noise points from
# demanding more of the model than it can deliver.
#
# Six parameters are free: the ejecta mass and kinetic energy, the spin period and magnetic field of the magnetar, the
# temperature floor of the photosphere, and the explosion epoch (days before the first Swift epoch). The remaining
# parameters are fixed: a typical ejecta opacity, :math:`\kappa = 0.1\,\mathrm{cm^2\,g^{-1}}`, a magnetic dipole
# (:math:`n = 3`), and a standard neutron star. We also fix the opacity to the magnetar's hard radiation,
# because the late-time :math:`t^{-2}` decline of a magnetar is degenerate with the leakage of its radiation, and
# these data cannot separate the two.

data = photometry.to_inference_data(model)
likelihood = LogGaussianCensoredLikelihood(model=model, data=data, sys_dex=0.08)
problem = InferenceProblem(likelihood=likelihood)

problem.set_prior("M_ej", "loguniform", lower=1 * u.Msun, upper=100 * u.Msun)
problem.set_prior("E_ej", "loguniform", lower=1e50 * u.erg, upper=1e53 * u.erg)
problem.set_prior("P_spin", "loguniform", lower=0.7 * u.ms, upper=15 * u.ms)
problem.set_prior("B_dipole", "loguniform", lower=1e12 * u.G, upper=1e15 * u.G)
problem.set_prior("T_floor", "uniform", lower=4000 * u.K, upper=10000 * u.K)
problem.set_prior("t_exp", "uniform", lower=-100 * u.day, upper=0 * u.day)

initial = {
    "M_ej": 20 * u.Msun,
    "E_ej": 8e51 * u.erg,
    "P_spin": 2.2 * u.ms,
    "B_dipole": 3.5e13 * u.G,
    "T_floor": 8000 * u.K,
    "t_exp": -25 * u.day,
}
fixed = {
    "kappa": 0.1 * u.cm**2 / u.g,
    "kappa_magnetar": 0.1 * u.cm**2 / u.g,
    "braking_index": 3.0,
    "I_ns": 1e45 * u.g * u.cm**2,
    "R_ns": 1e6 * u.cm,
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

samples = result.get_flat_samples(burn=BURN, thin=THIN)
column = {name: samples[:, i] for i, name in enumerate(names)}

M_ej = column["M_ej"] / u.Msun.to(u.g)  # solar masses
E_ej = column["E_ej"] / 1e51  # 1e51 erg
P_spin = column["P_spin"] * 1e3  # ms
B_dipole = column["B_dipole"] / 1e14  # 1e14 G
t_exp = column["t_exp"] / 86400.0  # days before the first epoch
T_floor = column["T_floor"]  # K
v_ej = np.sqrt(2.0 * column["E_ej"] / column["M_ej"]) / 1e8  # 1e3 km/s

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
# Hinkle et al. report the medians and 68% intervals of the MOSFiT magnetar fit to these Swift data and to the
# ground-based optical light curves. The spin period, magnetic field and ejecta mass mean the same thing in both models.
# The two models do not define the ejecta velocity in the same way, so we compare the velocity
# :math:`v = \sqrt{2E_{\rm ej}/M_{\rm ej}}` only roughly.

mosfit = {  # median, +err, -err
    "P (ms)": (2.79, 0.25, 0.50),
    "B (1e14 G)": (0.46, 1.33, 0.27),
    "M_ej (Msun)": (16.22, 1.98, 1.76),
    "v_ej (1e3 km/s)": (5.18, 0.10, 0.10),
}
ours = {"P (ms)": P_spin, "B (1e14 G)": B_dipole, "M_ej (Msun)": M_ej, "v_ej (1e3 km/s)": v_ej}

print(f"{'':18s}  {'This fit':>24s}  {'MOSFiT (Hinkle+23)':>24s}")
for key, (mid, up, down) in mosfit.items():
    lo, med, hi = np.percentile(ours[key], [16, 50, 84])
    print(f"{key:18s}  {med:8.2f} +{hi - med:5.2f} -{med - lo:5.2f}  {mid:8.2f} +{up:5.2f} -{down:5.2f}")

# %%
# The spin period, magnetic field and ejecta mass land close to the MOSFiT values: the field is consistent, the
# period is slightly below the MOSFiT interval, and the ejecta mass is within about one standard deviation of ours. The
# velocities differ by about 20%, which is no surprise given the different definitions. We should not expect better
# agreement. The MOSFiT fit also uses ground-based optical light curves, which pin down the rise, and fits the opacity,
# while the diffusion time depends only on the product of the opacity and the ejecta mass. We fix the opacity and
# only have Swift data, which begin when the supernova is already bright. The uncertainties here are also purely
# statistical, and are narrower than the systematic differences between two one-zone models.
#
# The Light Curves
# ----------------
#
# Now compare the model with the data. We draw 50 samples from the posterior and plot the model in each filter,
# along with the data, as a function of MJD.

rng = np.random.default_rng(0)
draws = samples[rng.choice(len(samples), size=50, replace=False)]
fixed_values = {name: problem.parameters[name].initial_value for name in fixed}

mjd_grid = np.linspace(MJD_0 - 5.0, MJD_0 + 145.0, 120)
t_grid = (mjd_grid - MJD_0) * 86400.0


def ab_mag(flux_density):
    return -2.5 * np.log10(flux_density / 3.631e-20)


fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for i, (band, ax) in enumerate(zip(BANDS, axes.flat)):
    for theta in draws:
        parameters = dict(zip(names, theta), **fixed_values)
        flux = model.forward_model({"band_index": np.full(t_grid.size, i), "time": t_grid}, parameters).flux
        ax.plot(mjd_grid, ab_mag(flux.value), color="C0", alpha=0.1, lw=1)
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
# The model follows the rise, the peak and the decline in all six filters, from the V band to the far ultraviolet.
# What it cannot do is make a blackbody into the real spectrum. The median residuals of each band, below, show where it
# falls short: the model is too bright in V and too faint in U, by amounts comparable to the 0.08 dex systematic floor
# that we gave the likelihood. Real SLSNe are also fainter in the far ultraviolet than a blackbody, because of
# line blanketing, which the paper's MOSFiT model includes as a linear UV suppression and this model does not.
# The temperature of the fit is a compromise between the bands.

parameters = dict(zip(names, np.median(samples, axis=0)), **fixed_values)
flux_model = model.forward_model({"band_index": data.x["band_index"], "time": data.x["time"]}, parameters).flux.value
residual = ab_mag(flux_model) - photometry.mag  # positive: the model is fainter than the data
for band in BANDS:
    selected = photometry.band_name == band
    print(
        f"{band:5s}  median residual {np.median(residual[selected]):+.2f} mag,  scatter {np.std(residual[selected]):.2f} mag"
    )

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
    photosphere = model.compute_photosphere(t_since_explosion, parameters)
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
