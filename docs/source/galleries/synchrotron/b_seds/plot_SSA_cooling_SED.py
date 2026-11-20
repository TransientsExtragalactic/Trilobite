r"""
Synchrotron Spectra With Cooling and SSA
==========================================

In practice, synchrotron spectra are considerably more complex than the
idealized scenarios discussed in textbooks such as :footcite:t:`RybickiLightman` or
:footcite:t:`1970ranp.book.....P`. In the modern synchrotron modeling literature for
supernovae and GRBs, both synchrotron self-absorption (SSA) and radiative
cooling are routinely included to reproduce observed spectra
(:footcite:t:`GranotSari2002SpectralBreaks`).

:class:`~trilobite.radiation.synchrotron.SEDs.one_zone.seds.PowerLaw_Cooling_SSA_SynchrotronSED`
implements the complete one-zone phenomenological SED including both effects.
This example shows how varying the peak flux density — which shifts the
self-absorption frequency :math:`\nu_a` — modifies the spectrum in each of
the three cooling regimes.

.. hint::

    For a detailed discussion of the relevant theory and its implementation
    in Trilobite, see :ref:`synchrotron_theory`.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from trilobite.radiation.synchrotron.SEDs import PowerLaw_Cooling_SSA_SynchrotronSED
from trilobite.utils.plot_utils import set_plot_style

# %%
# Parameters
# ----------
# We fix the injection frequency :math:`\nu_m`, the high-frequency cutoff
# :math:`\nu_{\rm max}`, and the SED shape parameters (:math:`p`, :math:`s`,
# :math:`\gamma_m`, :math:`\omega`).  The colormap sweeps over peak flux densities
# :math:`F_{\nu,\rm pk}`, which shifts :math:`\nu_a` while leaving the optically
# thin structure unchanged.

sed = PowerLaw_Cooling_SSA_SynchrotronSED()

nu = np.geomspace(1e7, 1e20, 1000) * u.Hz
log_nu = np.log10(nu.to_value(u.Hz))

nu_m = 1e7 * u.Hz
nu_max = 1e18 * u.Hz
nu_c_ref = 1e14 * u.Hz
F_norm = 100 * u.mJy

p = 3.0
s = -0.01
gamma_m = 1
omega = 0.5 * np.pi * (1e16 * u.cm) ** 2 / (1 * u.Mpc) ** 2

F_peaks = np.geomspace(1e-1, 1e12, 10) * u.mJy
cmap = plt.cm.viridis

# %%
# A Representative SED
# ---------------------
# We begin with a single slow-cooling spectrum (:math:`\nu_m \ll \nu_c`) to
# orient the reader.  The upper panel shows :math:`F_\nu` and the lower shows
# the log–log slope :math:`d\log F_\nu / d\log\nu`, which separates the
# spectral segments.

flux_ref = sed.sed(nu, nu_m, nu_c_ref, F_norm, nu_max=nu_max, omega=omega, p=p, s=s, gamma_m=gamma_m)

set_plot_style()

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
)

ax_top.loglog(nu, flux_ref.to_value("mJy"), color="C0", lw=2, label="Cooling + SSA SED")
for nu_val, label, color in [
    (nu_m, r"$\nu_m$", "C1"),
    (nu_c_ref, r"$\nu_c$", "C2"),
    (nu_max, r"$\nu_{\rm max}$", "C3"),
]:
    ax_top.axvline(nu_val.value, color=color, ls="--", lw=1.2, label=label)
ax_top.set_ylim([1e-10, 1e1])
ax_top.set_ylabel(r"$F_\nu$ [mJy]")
ax_top.set_title("Synchrotron SED with Cooling and SSA")
ax_top.legend()
ax_top.grid(True, which="both", ls="--", alpha=0.3)

ax_bot.semilogx(nu, np.gradient(np.log10(flux_ref.to_value(u.mJy)), log_nu), color="C0", lw=2)
for nu_val, color in [(nu_m, "C1"), (nu_c_ref, "C2"), (nu_max, "C3")]:
    ax_bot.axvline(nu_val.value, color=color, ls="--", lw=1.2)
ax_bot.set_xlabel(r"Frequency [Hz]")
ax_bot.set_ylabel(r"$d\log F_\nu / d\log\nu$")
ax_bot.set_ylim([-5, 5])
ax_bot.grid(True, which="both", ls="--", alpha=0.3)

plt.show()

# %%
# The Regimes of the Spectrum
# ----------------------------
#
# The spectral shape depends primarily on whether the electron population
# cools efficiently.  We divide the parameter space into three broad classes:
#
# 1. **Non-cooling** (:math:`\nu_c \gg \nu_{\rm max}`) — injected power law intact
# 2. **Fast-cooling** (:math:`\nu_c \ll \nu_m`) — all electrons cool; spectrum steepens
# 3. **Slow-cooling** (:math:`\nu_m \ll \nu_c \ll \nu_{\rm max}`) — partial cooling above :math:`\nu_c`
#
# Within each class we sweep over :math:`F_{\nu,\rm pk}` (colormap) to shift
# :math:`\nu_a`, illustrating how SSA modifies the low-frequency turnover.

# %%
# I. Non-Cooling Spectra
# ^^^^^^^^^^^^^^^^^^^^^^
#
# With :math:`\nu_c \gg \nu_{\rm max}` radiative losses are negligible.
# The optically thin spectrum peaks at :math:`\nu_m` with slopes
# :math:`F_\nu \propto \nu^{1/3}` below and :math:`\nu^{-(p-1)/2}` above.
# SSA introduces a break at :math:`\nu_a < \nu_m` below which
# :math:`F_\nu \propto \nu^{5/2}`.

nu_c_nc = 1e20 * u.Hz

set_plot_style()
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
)
norm = LogNorm(vmin=F_peaks.min().value, vmax=F_peaks.max().value)

for Fp in F_peaks:
    flux_i = sed.sed(nu, nu_m, nu_c_nc, Fp, nu_max=nu_max, omega=omega, p=p, s=s, gamma_m=gamma_m)
    color = cmap(norm(Fp.value))
    ax_top.loglog(nu, flux_i.to_value(u.mJy), color=color, lw=2)
    ax_bot.semilogx(nu, np.gradient(np.log10(flux_i.to_value(u.mJy)), log_nu), color=color, lw=1.5)

for ax in (ax_top, ax_bot):
    ax.axvline(nu_m.value, color="k", ls="--", alpha=0.4)
    ax.grid(True, which="both", ls="--", alpha=0.3)
ax_top.set_ylabel(r"$F_\nu$ [mJy]")
ax_top.set_title("Non-Cooling Synchrotron Spectra with SSA")
ax_bot.set_xlabel(r"Frequency [Hz]")
ax_bot.set_ylabel(r"$d\log F_\nu / d\log\nu$")
ax_bot.set_ylim(-3, 3)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=[ax_top, ax_bot], pad=0.02, label=r"$F_{\nu,\rm pk}\ [\rm mJy]$")
plt.show()

# %%
# II. Fast-Cooling Spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# When :math:`\nu_c \ll \nu_m` electrons cool efficiently below the injection
# Lorentz factor.  Ignoring SSA, the optically thin slopes are:
#
# - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu < \nu_c`
# - :math:`F_\nu \propto \nu^{-1/2}` for :math:`\nu_c < \nu < \nu_m`
# - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_m`
#
# We focus on the ordering :math:`\nu_a < \nu_c < \nu_m`.

nu_c_fc = 1e9 * u.Hz

set_plot_style()
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
)
norm = LogNorm(vmin=F_peaks.min().value, vmax=F_peaks.max().value)

for Fp in F_peaks:
    flux_i = sed.sed(nu, nu_m, nu_c_fc, Fp, nu_max=nu_max, omega=omega, p=p, s=s, gamma_m=gamma_m)
    color = cmap(norm(Fp.value))
    ax_top.loglog(nu, flux_i.to_value(u.mJy), color=color, lw=2)
    ax_bot.semilogx(nu, np.gradient(np.log10(flux_i.to_value(u.mJy)), log_nu), color=color, lw=1.5)

for ax in (ax_top, ax_bot):
    ax.axvline(nu_c_fc.value, color="k", ls="--", alpha=0.4)
    ax.axvline(nu_m.value, color="k", ls="--", alpha=0.4)
    ax.grid(True, which="both", ls="--", alpha=0.3)
ax_top.set_ylabel(r"$F_\nu$ [mJy]")
ax_top.set_title("Fast-Cooling Synchrotron Spectra with SSA")
ax_bot.set_xlabel(r"Frequency [Hz]")
ax_bot.set_ylabel(r"$d\log F_\nu / d\log\nu$")
ax_bot.set_ylim(-4, 3)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=[ax_top, ax_bot], pad=0.02, label=r"$F_{\nu,\rm pk}\ [\rm mJy]$")
plt.show()

# %%
# III. Slow-Cooling Spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When :math:`\nu_m \ll \nu_c \ll \nu_{\rm max}` radiative losses affect only
# the highest-energy electrons.  The optically thin spectrum has three segments:
#
# - :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu < \nu_m`
# - :math:`F_\nu \propto \nu^{-(p-1)/2}` for :math:`\nu_m < \nu < \nu_c`
# - :math:`F_\nu \propto \nu^{-p/2}` for :math:`\nu > \nu_c`
#
# We focus on the ordering :math:`\nu_a < \nu_m < \nu_c`.

nu_c_sc = 1e16 * u.Hz

set_plot_style()
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
)
norm = LogNorm(vmin=F_peaks.min().value, vmax=F_peaks.max().value)

for Fp in F_peaks:
    flux_i = sed.sed(nu, nu_m, nu_c_sc, Fp, nu_max=nu_max, omega=omega, p=p, s=s, gamma_m=gamma_m)
    color = cmap(norm(Fp.value))
    ax_top.loglog(nu, flux_i.to_value(u.mJy), color=color, lw=2)
    ax_bot.semilogx(nu, np.gradient(np.log10(flux_i.to_value(u.mJy)), log_nu), color=color, lw=1.5)

for ax in (ax_top, ax_bot):
    ax.axvline(nu_m.value, color="k", ls="--", alpha=0.4)
    ax.axvline(nu_c_sc.value, color="k", ls="--", alpha=0.4)
    ax.grid(True, which="both", ls="--", alpha=0.3)
ax_top.set_ylabel(r"$F_\nu$ [mJy]")
ax_top.set_title("Slow-Cooling Synchrotron Spectra with SSA")
ax_bot.set_xlabel(r"Frequency [Hz]")
ax_bot.set_ylabel(r"$d\log F_\nu / d\log\nu$")
ax_bot.set_ylim(-4, 3)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=[ax_top, ax_bot], pad=0.02, label=r"$F_{\nu,\rm pk}\ [\rm mJy]$")
plt.show()
