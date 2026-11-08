r"""
Equipartition Analysis: What Does a Radio Peak Say About Source Energy?
========================================================================

**Objective:** estimate the magnetic field, radius and total energy of a radio supernova from nothing but its
SSA peak, and see how those estimates depend on the assumed energy partition.

The standard equipartition analysis
(:footcite:t:`readheadEquipartitionBrightnessTemperature1994`) assumes that fixed fractions
:math:`\epsilon_E` and :math:`\epsilon_B` of the post-shock energy go into relativistic electrons and magnetic
fields. The peak frequency and flux then fix :math:`B` and :math:`R`, and the energy in the emitting region follows,

.. math::

    U = \frac{B^2}{8\pi}\left(1 + \frac{\epsilon_E}{\epsilon_B}\right) V,
    \qquad V = \tfrac{4}{3}\pi R^3 f_V .

.. note::

    The analytic one-zone SED classes are being phased out in favor of the numerical engines, which
    are more accurate. Here the SED is generated with the numerical engine, the peak is extracted by a
    fit, and only then is it inverted with
    :func:`~trilobite.radiation.synchrotron.spectral_inversion.invert_ssa_Diggins26`.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.optimize import curve_fit

from trilobite.radiation.synchrotron import invert_ssa_Diggins26
from trilobite.radiation.synchrotron.electron_distributions import PowerLaw
from trilobite.radiation.synchrotron.SEDs.numerical import NonRelativisticSphericalSynchrotronEngine
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

# %%
# The Source
# ----------
#
# We take a Type Ibc supernova at 20 Mpc, 30 days after explosion, with :math:`B = 0.5` G and
# :math:`R = 10^{16}` cm, an expansion speed of about :math:`0.13c`. The electrons carry
# :math:`\epsilon_E = 0.1` and the field :math:`\epsilon_B = 0.1` of the post-shock energy.

B_true, R_true, D, p = 0.5 * u.G, 1.0e16 * u.cm, 20.0 * u.Mpc, 3.0
epsilon_E, epsilon_B_true = 0.1, 0.1
micro = dict(gamma_min=1.0, gamma_max=1e6)
geometry = dict(f_A=0.5, f_V=0.5)

engine = NonRelativisticSphericalSynchrotronEngine()
engine.load_avg_first_kernel()

nu = np.geomspace(0.1, 300.0, 60) * u.GHz
gamma = np.geomspace(micro["gamma_min"], micro["gamma_max"], 400)

N0 = PowerLaw.normalize_from_magnetic_field(B_true, epsilon_B_true, epsilon_E, p=p, **micro)
N = N0.to_value(u.cm**-3) * PowerLaw.pdf(gamma, p=p, gamma_min=gamma[0], gamma_max=gamma[-1])
flux = engine.compute_flux_density(nu, R=R_true, B=B_true, N=N, gamma=gamma, luminosity_distance=D, **geometry)

# %%
# Extracting the Peak
# -------------------
#
# A smoothly broken power law with a :math:`\nu^{5/2}` thick slope and a free thin slope gives the peak
# frequency :math:`\nu_{\rm pk}` and the asymptotic peak flux :math:`F_{\rm pk}`.


def sbpl(log_nu, log_F, log_nu_pk, beta, s):
    x = (log_nu - log_nu_pk) * np.log(10)
    return log_F + s * np.logaddexp(2.5 * x / s, beta * x / s) / np.log(10)


log_nu, log_F = np.log10(nu.to_value(u.Hz)), np.log10(flux.to_value(u.Jy))
guess = [log_F.max(), log_nu[np.argmax(log_F)], -1.0, -1.0]
fit, _ = curve_fit(sbpl, log_nu, log_F, p0=guess, bounds=([-20, 7, -4, -10], [5, 12, 0, -0.05]))
nu_pk, F_pk = 10 ** fit[1] * u.Hz, 10 ** fit[0] * u.Jy

fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
ax.loglog(nu.to_value(u.GHz), flux.to_value(u.mJy), "o", ms=4, label="Numerical SED")
ax.plot(nu_pk.to_value(u.GHz), F_pk.to_value(u.mJy), "r*", ms=14, label=r"$(\nu_{\rm pk}, F_{\rm pk})$")
ax.set(xlabel="Frequency [GHz]", ylabel="Flux density [mJy]", title="Observed SED and extracted peak")
ax.legend()
plt.show()

# %%
# Sweeping the Energy Partition
# -----------------------------
#
# The observables are fixed, so we invert them for a range of :math:`\epsilon_B` at fixed
# :math:`\epsilon_E = 0.1`. All inputs broadcast, so the sweep is one call.

epsilon_B = np.geomspace(1e-3, 1.0, 200)
out = invert_ssa_Diggins26(
    nu_pk, F_pk, D, p=p, epsilon_E=epsilon_E, epsilon_B=epsilon_B, pitch_average=True, **micro, **geometry
)

B, R = out["B"], out["R"]
V = 4 / 3 * np.pi * R**3 * geometry["f_V"]
U = (B.to_value(u.G) ** 2 / (8 * np.pi) * (1 + epsilon_E / epsilon_B) * V.to_value(u.cm**3)) * u.erg
i_min = np.argmin(U)

fig, (ax_BR, ax_U) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

ax_BR.semilogx(epsilon_B, (B / B_true).decompose(), label=r"$B_{\rm rec}/B_{\rm true}$")
ax_BR.semilogx(epsilon_B, (R / R_true).decompose(), label=r"$R_{\rm rec}/R_{\rm true}$")
ax_BR.axhline(1.0, color="k", lw=1)
ax_BR.set(xlabel=r"Assumed $\epsilon_B$", ylabel="Recovered / true", title="Field and radius")
ax_BR.legend()

ax_U.loglog(epsilon_B, U.to_value(u.erg))
ax_U.plot(epsilon_B[i_min], U[i_min].to_value(u.erg), "ro", label=rf"Minimum at $\epsilon_B={epsilon_B[i_min]:.2f}$")
ax_U.set(xlabel=r"Assumed $\epsilon_B$", ylabel="Total energy $U$ [erg]", title="Energy in the emitting region")
ax_U.legend()

for ax in (ax_BR, ax_U):
    ax.axvline(epsilon_B_true, color="0.5", ls="--", lw=1)
plt.show()

# %%
# **Outcome:** the inferred radius is nearly independent of :math:`\epsilon_B` (within 20% over three decades),
# while the field changes by a factor of about four. The energy is minimized near equipartition, at
# :math:`\epsilon_B \approx 0.07` for :math:`\epsilon_E = 0.1`, with :math:`U_{\rm min} \approx 4\times10^{46}`
# erg. It rises by a factor of 2 to 3.5 at the extremes of the sweep, so the estimate is a *minimum* energy.
# The dashed line marks the true :math:`\epsilon_B`.
#
# .. footbibliography::
