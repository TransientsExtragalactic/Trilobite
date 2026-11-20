r"""
From a Radio SED to Source Size and Velocity
=============================================

**Objective:** turn a single-epoch radio SED with error bars into a measurement of a supernova's radius,
magnetic field and expansion speed, with uncertainties, and check that the shock is sub-relativistic.

The workflow has three steps:

1. fit the SED with a broken power law to get the break frequency and flux and their covariance,
2. draw samples from that covariance and invert each with
   :func:`~trilobite.radiation.synchrotron.spectral_inversion.invert_ssa_Diggins26`,
3. divide the radius by the age to get the velocity.

.. note::

    The analytic one-zone SED classes are being phased out in favor of the numerical engines, which
    are more accurate. The "observed" SED here is generated with the numerical engine from a source of known
    properties, so the recovered values can be compared with the truth.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.optimize import curve_fit

from trilobite.radiation.synchrotron import invert_ssa_Diggins26
from trilobite.radiation.synchrotron.electron_distributions import PowerLaw
from trilobite.radiation.synchrotron.SEDs.numerical import NonRelativisticSphericalSynchrotronEngine
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()
rng = np.random.default_rng(42)

# %%
# The Observation
# ---------------
#
# A Type Ibc supernova at 20 Mpc is observed 30 days after explosion at ten frequencies from 1.4 to 45 GHz,
# each with a 10% flux error. The true shell has :math:`B = 1` G and :math:`R = 10^{16}` cm, and the electrons
# have :math:`p = 3` and :math:`\epsilon_E = \epsilon_B = 0.1`.

B_true, R_true, t_obs, D = 1.0 * u.G, 1.0e16 * u.cm, 30.0 * u.day, 20.0 * u.Mpc
micro = dict(epsilon_E=0.1, epsilon_B=0.1, gamma_min=1.0, gamma_max=1e6)
geometry = dict(f_A=0.5, f_V=0.5)

engine = NonRelativisticSphericalSynchrotronEngine()
engine.load_avg_first_kernel()
gamma = np.geomspace(micro["gamma_min"], micro["gamma_max"], 400)
N0 = PowerLaw.normalize_from_magnetic_field(B_true, p=3.0, **micro)
N = N0.to_value(u.cm**-3) * PowerLaw.pdf(gamma, p=3.0, gamma_min=gamma[0], gamma_max=gamma[-1])


def numerical_sed(nu):
    return engine.compute_flux_density(nu, R=R_true, B=B_true, N=N, gamma=gamma, luminosity_distance=D, **geometry).to(
        u.mJy
    )


nu_obs = np.array([1.4, 2.5, 4.0, 6.0, 8.0, 11.0, 15.0, 22.0, 33.0, 45.0]) * u.GHz
F_obs = numerical_sed(nu_obs)
F_err = 0.1 * F_obs
F_obs = F_obs + rng.normal(scale=F_err.value) * u.mJy

# %%
# Fitting the Break
# -----------------
#
# We fit a smoothly broken power law with a :math:`\nu^{5/2}` thick slope. The free parameters are the break
# flux :math:`F_{\rm brk}`, the break frequency :math:`\nu_{\rm brk}`, the thin slope :math:`\beta` and the
# smoothing :math:`s`. The fit returns their covariance, which carries the measurement errors forward.


def sbpl(log_nu, log_F, log_nu_brk, beta, s):
    x = (log_nu - log_nu_brk) * np.log(10)
    return log_F + s * np.logaddexp(2.5 * x / s, beta * x / s) / np.log(10)


log_nu, log_F = np.log10(nu_obs.to_value(u.Hz)), np.log10(F_obs.to_value(u.Jy))
log_err = (F_err / F_obs).decompose().value / np.log(10)
guess = [log_F.max(), log_nu[np.argmax(log_F)], -1.0, -0.7]
fit, cov = curve_fit(
    sbpl, log_nu, log_F, p0=guess, sigma=log_err, absolute_sigma=True, bounds=([-20, 7, -4, -3], [5, 12, 0, -0.1])
)

nu_grid = np.geomspace(1.0, 60.0, 200) * u.GHz
fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
for draw in rng.multivariate_normal(fit, cov, size=100):
    ax.loglog(nu_grid.to_value(u.GHz), 10 ** sbpl(np.log10(nu_grid.to_value(u.Hz)), *draw) * 1e3, "C1", alpha=0.08)
ax.loglog(nu_grid.to_value(u.GHz), numerical_sed(nu_grid).to_value(u.mJy), "k", lw=1.5, label="True SED")
ax.errorbar(nu_obs.to_value(u.GHz), F_obs.to_value(u.mJy), F_err.to_value(u.mJy), fmt="o", capsize=3, label="Observed")
ax.set(
    xlabel="Frequency [GHz]",
    ylabel="Flux density [mJy]",
    title="Broken power-law fit (orange: draws from the covariance)",
)
ax.legend()
plt.show()

# %%
# Propagating the Uncertainty
# ---------------------------
#
# Each draw gives a break frequency, a break flux and a slope. The slope sets the electron index through
# :math:`p = 1 - 2\beta`, and the inversion accepts arrays, so all 2000 draws are inverted in one call.

draws = rng.multivariate_normal(fit, cov, size=2000)
nu_brk, F_brk, p_draw = 10 ** draws[:, 1] * u.Hz, 10 ** draws[:, 0] * u.Jy, 1 - 2 * draws[:, 2]
ok = p_draw > 2.05  # the inversion needs p > 2
out = invert_ssa_Diggins26(nu_brk[ok], F_brk[ok], D, p=p_draw[ok], pitch_average=True, **micro, **geometry)

B_post, R_post = out["B"], out["R"]
v_post = (R_post / t_obs / const.c).decompose()

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
panels = [
    (B_post.to_value(u.G), B_true.value, r"$B$ [G]"),
    (R_post.to_value(u.cm), R_true.value, r"$R$ [cm]"),
    (v_post.value, (R_true / t_obs / const.c).decompose().value, r"$v/c$"),
]
for ax, (samples, truth, label) in zip(axes, panels):
    lo, med, hi = np.percentile(samples, [16, 50, 84])
    ax.hist(samples, bins=40, color="C0", alpha=0.8)
    ax.axvline(truth, color="k", lw=2, label="True")
    ax.axvspan(lo, hi, color="C1", alpha=0.25, label=r"$\pm1\sigma$")
    ax.set(xlabel=label, title=rf"{med:.3g} $^{{+{hi - med:.2g}}}_{{-{med - lo:.2g}}}$")
axes[0].legend()
plt.show()

# %%
# **Outcome:** the radius, field and speed are recovered with the true values inside the :math:`1\sigma`
# range. The expansion speed is :math:`v/c \approx 0.13`, well below the speed of light. A posterior that reached
# :math:`v/c \sim 1` would instead signal relativistic ejecta, an underestimated distance or a non-spherical
# geometry, all of which the single-zone inversion does not describe.
