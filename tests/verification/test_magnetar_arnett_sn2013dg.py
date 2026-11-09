"""
Verification of the magnetar Arnett model against the MOSFiT fit of SN 2013dg.

:footcite:t:`Nicholl_2017` fitted a magnetar-powered diffusion model to the multicolour light curves of 38 superluminous
supernovae with MOSFiT, an MCMC code. This test repeats their fit of SN 2013dg, whose photometry is from
:footcite:t:`Nicholl_2014`, with :class:`~trilobite.models.supernovae.arnett.MagnetarArnettCutoffModel` and the
priors of their Table 1, and checks that the posterior medians agree with their published values.

The model of the library is used unchanged. Only the *parameterization* differs: MOSFiT samples the photospheric
velocity and the neutron-star mass instead of the kinetic energy, the moment of inertia and the radius, so a small
subclass maps one to the other (``_MOSFiTParameterization``). The first two tests check that this mapping is a pure
change of variables and that the spin-down law of the library is that of the paper.

Differences from the paper, none of which can be removed from the published information:

* There is no host extinction (``A_V``, published as 0.07 +0.10 -0.06) and no extra-variance parameter (``sigma``,
  published as 0.01 +0.02 -0.01 mag); the likelihood uses the catalogue errors alone.
* The two constraints of the paper (energy conservation and an optically thick time of at least 100 d) are not applied.
* The Galactic extinction coefficients are those of Schlafly & Finkbeiner (2011), and the data are all the detections
  of the published table, including four measurements that the authors flag as possibly host-contaminated. The
  upper limits are not used.

The MCMC is not seeded (``emcee`` offers no way to do so through the interface of Trilobite), so the chain length is
chosen to make the Monte Carlo error small compared with the tolerance; the test checks that it is.

.. footbibliography::
"""

from pathlib import Path

import emcee
import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15
from astropy.table import QTable, Table

from verification.registry import verifies

pytest.importorskip("speclite")

from trilobite.data import OpticalPhotometryContainer  # noqa: E402
from trilobite.inference.likelihood.base import GaussianCensoredLikelihood  # noqa: E402
from trilobite.inference.problem import InferenceProblem  # noqa: E402
from trilobite.inference.sampling.mcmc import EmceeSampler  # noqa: E402
from trilobite.inference.transform import Log10Transform  # noqa: E402
from trilobite.models.core.parameters import ModelParameter  # noqa: E402
from trilobite.models.supernovae.arnett import MagnetarArnettCutoffModel  # noqa: E402
from trilobite.radiation.arnett import _magnetar_spin_down_cgs  # noqa: E402
from trilobite.utils.phot_utils import FilterBundle, ab_mag_to_flux, load_filter_from_speclite  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "sn2013dg"

_C = const.c.cgs.value
_MSUN = const.M_sun.cgs.value
_MJD_FIRST = 56425.2  # first detection; the time origin of the fit
_COSMOLOGY = Planck15  # MOSFiT adopts a Planck cosmology
_SDSS = {"g": "sdss2010-g", "r": "sdss2010-r", "i": "sdss2010-i", "z": "sdss2010-z"}
# Schlafly & Finkbeiner (2011), A_lambda / E(B-V) for R_V = 3.1.
_A_OVER_EBV = {"g": 3.303, "r": 2.285, "i": 1.698, "z": 1.263}

# Nicholl et al. (2017), eq. 2 and 3, for a neutron star of mass M_NS, spin period P_ms and field B_14 (in 1e14 G):
#   E_mag = 2.6e52 (M_NS / 1.4)^1.5 P_ms^-2 erg,   t_mag = 1.3e5 (M_NS / 1.4)^1.5 P_ms^2 B_14^-2 s.
# The first fixes the moment of inertia (E_mag = I omega^2 / 2). The second, with the spin-down law of the library
# (t_sd = 3 I c^3 / (B^2 R^6 omega^2) for a dipole), fixes the radius, which the paper does not state.
_OMEGA_1MS = 2.0 * np.pi / 1.0e-3
_I_REF = 2.0 * 2.6e52 / _OMEGA_1MS**2
_R_NS = (3.0 * _I_REF * _C**3 / (1.0e14**2 * _OMEGA_1MS**2 * 1.3e5)) ** (1.0 / 6.0)

N_SIGMA = 2.0  # a parameter agrees if its median is within this many combined standard deviations of the paper's
N_WALKERS = 40
N_STEPS = 20_000
BURN_FRACTION = 0.5
# The standard error of a median is about 1.25 sigma / sqrt(N_eff). This many effective samples keep it below 0.1 sigma
# of the posterior, small compared with the tolerance, which is about 2.8 sigma.
MIN_EFFECTIVE_SAMPLES = 200
MAX_CHI2_PER_POINT = 1.5  # the paper's extra variance of 0.01 mag means a reduced chi-squared of about 1


class _MOSFiTParameterization(MagnetarArnettCutoffModel):
    """The library model, with ``v_phot`` and ``M_NS`` in place of ``E_ej``, ``I_ns`` and ``R_ns``."""

    PARAMETERS = tuple(p for p in MagnetarArnettCutoffModel.PARAMETERS if p.name not in ("E_ej", "I_ns", "R_ns")) + (
        ModelParameter("v_phot", 1.0e9, base_units=u.cm / u.s, description="Photospheric velocity.", bounds=(0, None)),
        ModelParameter("M_NS", 1.8 * u.Msun, base_units=u.g, description="Neutron-star mass.", bounds=(0, None)),
    )

    def _compute_photosphere_cgs(self, t_obs, parameters):
        p = dict(parameters)
        p["E_ej"] = 0.3 * p["M_ej"] * p["v_phot"] ** 2  # so that the velocity sqrt(10 E / 3 M) of the model is v_phot
        p["I_ns"] = _I_REF * (p["M_NS"] / _MSUN / 1.4) ** 1.5
        p["R_ns"] = _R_NS
        return super()._compute_photosphere_cgs(t_obs, p)


def _model(cls=_MOSFiTParameterization):
    bundle = FilterBundle({band: load_filter_from_speclite(name) for band, name in _SDSS.items()})
    return cls(bundle, redshift=0.265, cosmology=_COSMOLOGY, trapping_mode="emergence")


def _observations():
    """Table: the detections of SN 2013dg as AB flux densities, corrected for Galactic extinction."""
    photometry = QTable.read(DATA / "photometry.ecsv", format="ascii.ecsv")
    photometry = photometry[~photometry["upper_limit"]]
    band = np.array(photometry["band"], dtype=str)
    ebv = photometry.meta["ebv_mw"]
    mag = photometry["mag"].value - np.array([_A_OVER_EBV[b] for b in band]) * ebv
    flux = np.asarray(ab_mag_to_flux(mag), dtype=float)
    unit = u.Unit("erg/(s cm2 Hz)")
    return Table(
        {
            "time": (photometry["mjd"].value - _MJD_FIRST) * u.day,
            "band_name": band,
            "flux_density": flux * unit,
            "flux_density_error": flux * np.log(10.0) / 2.5 * photometry["mag_err"].value * unit,
            "flux_upper_limit": np.full(len(band), np.nan) * unit,
        }
    )


def _problem(model, observations):
    """Build the inference problem with the priors of Table 1 of Nicholl et al. (2017), except for A_V and sigma."""
    data = OpticalPhotometryContainer(observations).to_inference_data(model)
    problem = InferenceProblem(likelihood=GaussianCensoredLikelihood(model=model, data=data))
    cm2g = u.cm**2 / u.g
    problem.set_prior("P_spin", "uniform", lower=0.7e-3 * u.s, upper=20e-3 * u.s)
    problem.set_prior("B_dipole", "loguniform", lower=0.01e14 * u.G, upper=10e14 * u.G)
    problem.set_prior("M_ej", "loguniform", lower=0.1 * u.Msun, upper=100.0 * u.Msun)
    problem.set_prior("v_phot", "truncatednormal", mean=1.47e9, sigma=4.3e8, lower=1.0e8, upper=3.0e9)
    problem.set_prior("kappa", "uniform", lower=0.05 * cm2g, upper=0.2 * cm2g)
    problem.set_prior("kappa_magnetar", "loguniform", lower=0.01 * cm2g, upper=100.0 * cm2g)
    problem.set_prior("M_NS", "uniform", lower=1.4 * u.Msun, upper=2.2 * u.Msun)
    problem.set_prior("T_floor", "truncatednormal", mean=6000.0, sigma=1000.0, lower=3000.0, upper=10000.0)
    problem.set_prior("t_exp", "uniform", lower=-100.0 * u.day, upper=0.0 * u.day)
    for name in ("M_ej", "B_dipole", "kappa_magnetar"):
        problem.parameters[name].transform = Log10Transform()
    problem.parameters["t_exp"].initial_value = -50.0 * u.day  # the sampler wants the initial point inside the prior
    for name, value in {"braking_index": 3.0, "lambda_cut": 3000.0}.items():  # dipole spin-down; MOSFiT's UV cutoff
        problem.parameters[name].initial_value = value
        problem.parameters[name].freeze = True
    return problem


# =========================================================================== #
# Tests                                                                       #
# =========================================================================== #
@verifies(MagnetarArnettCutoffModel)
def test_parameterization_is_a_change_of_variables(verify):
    """The subclass used for the fit gives the fluxes of the library model at the mapped parameters."""
    times = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0]) * 86400.0
    bands = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    mapped = {
        "M_ej": 2.75 * _MSUN,
        "kappa": 0.12,
        "T_floor": 5070.0,
        "t_exp": -50.0 * 86400.0,
        "P_spin": 3.5e-3,
        "B_dipole": 1.56e14,
        "kappa_magnetar": 0.04,
    }
    v_phot, M_NS = 8.38e8, 1.8 * _MSUN
    flux = (
        _model()
        .forward_model({"time": times, "band_index": bands}, {**mapped, "v_phot": v_phot, "M_NS": M_NS})
        .flux.value
    )
    reference = (
        _model(MagnetarArnettCutoffModel)
        .forward_model(
            {"time": times, "band_index": bands},
            {
                **mapped,
                "E_ej": 0.3 * mapped["M_ej"] * v_phot**2,
                "I_ns": _I_REF * (M_NS / _MSUN / 1.4) ** 1.5,
                "R_ns": _R_NS,
            },
        )
        .flux.value
    )
    verify.agrees(flux, reference, tol=1e-10, metric="rel", label="flux")


@verifies(MagnetarArnettCutoffModel)
@pytest.mark.parametrize("period_ms, field_14, mass", [(3.5, 1.56, 1.8), (1.0, 1.0, 1.4), (2.2, 0.33, 2.0)])
def test_spin_down_matches_the_published_equations(verify, period_ms, field_14, mass):
    """The spin-down law of the library gives the rotational energy and spin-down time of Nicholl et al. (2017)."""
    I_ns = _I_REF * (mass / 1.4) ** 1.5
    L_0, t_sd, _ = _magnetar_spin_down_cgs(period_ms * 1e-3, field_14 * 1e14, 3.0, I_ns, _R_NS)
    verify.agrees(L_0 * t_sd, 2.6e52 * (mass / 1.4) ** 1.5 / period_ms**2, tol=1e-6, label=f"E_mag, P={period_ms} ms")
    verify.agrees(
        t_sd, 1.3e5 * (mass / 1.4) ** 1.5 * period_ms**2 / field_14**2, tol=1e-6, label=f"t_mag, P={period_ms} ms"
    )


@verifies(MagnetarArnettCutoffModel)
def test_reproduces_the_published_fit_of_sn2013dg(verify):
    """The posterior of the fit of SN 2013dg agrees with the published medians and fits the data to its errors."""
    model = _model()
    observations = _observations()
    problem = _problem(model, observations)

    result = EmceeSampler(problem, n_walkers=N_WALKERS).run(
        N_STEPS, initial_state=problem.sample_prior(N_WALKERS, rng=np.random.default_rng(0)), progress=False
    )
    names = result.parameter_names
    burn = int(BURN_FRACTION * N_STEPS)
    chain = result.get_flat_samples(burn=0, thin=1, physical=True).reshape(N_STEPS, N_WALKERS, -1)[burn:]
    log_posterior = np.asarray(result.log_posterior)[burn:]

    # The comparison is only meaningful if the chain holds enough independent samples. The log-spaced parameters are
    # sampled in log space, where the chain is well behaved.
    log_spaced = [names.index(n) for n in ("M_ej", "B_dipole", "kappa_magnetar")]
    sampled = chain.copy()
    sampled[..., log_spaced] = np.log10(sampled[..., log_spaced])
    tau = float(np.max(emcee.autocorr.integrated_time(sampled, tol=0)))
    verify.within(N_WALKERS * chain.shape[0] / tau, MIN_EFFECTIVE_SAMPLES, np.inf, label="effective samples")

    # Posterior medians against the published ones, in the units of the published table.
    column = {n: chain[..., names.index(n)].ravel() for n in names}
    ours = {
        "P": column["P_spin"] * 1e3,
        "B_perp": column["B_dipole"] / 1e14,
        "M_ej": column["M_ej"] / _MSUN,
        "v_phot": column["v_phot"] / 1e8,
        "kappa": column["kappa"],
        "kappa_gamma": column["kappa_magnetar"],
        "M_NS": column["M_NS"] / _MSUN,
        "T_floor": column["T_floor"] / 1e3,
    }
    published = QTable.read(DATA / "published_fit.ecsv", format="ascii.ecsv")
    for row in published:
        if row["name"] not in ours:
            continue
        low, median, high = np.percentile(ours[row["name"]], [16, 50, 84])
        sigma = np.hypot(0.5 * (high - low), 0.5 * (row["err_up"] + row["err_down"]))
        verify.within(
            median,
            row["value"] - N_SIGMA * sigma,
            row["value"] + N_SIGMA * sigma,
            label=f"{row['name']} [{row['unit']}], published {row['value']}",
        )

    # The best sample fits the data to the quoted errors, as the published extra variance of 0.01 mag implies.
    best = np.unravel_index(np.argmax(log_posterior), log_posterior.shape)
    bands = model.bundle.filter_names
    flux = model.forward_model(
        {
            "time": np.asarray(observations["time"], dtype=float) * 86400.0,
            "band_index": np.array([bands.index(b) for b in observations["band_name"]]),
        },
        dict(zip(names, chain[best])),
    ).flux.value
    observed = np.asarray(observations["flux_density"], dtype=float)
    error = np.asarray(observations["flux_density_error"], dtype=float)
    chi2 = np.sum(((flux - observed) / error) ** 2) / len(observed)
    verify.within(chi2, 0.0, MAX_CHI2_PER_POINT, label="chi-squared per data point of the best sample")
