r"""
Recovering B and R from an SSA Peak: the Diggins26 Inversion
=============================================================

**Objective:** test how accurately
:func:`~trilobite.radiation.synchrotron.spectral_inversion.invert_ssa_Diggins26`
recovers the magnetic field :math:`B` and radius :math:`R` of a source from
its self-absorption peak.

.. note::

    The analytic one-zone SED classes are being phased out in favor of the numerical engines, which
    are more accurate. Here the SED is generated with the numerical engine, the break is extracted by a
    fit, and only then is it inverted. The inversion is therefore tested against an independent calculation,
    not against its own forward model.

The inversion needs two numbers from the SED, the break frequency :math:`\nu_{\rm brk}` and the
asymptotic flux :math:`F_{\rm brk}` at the break, plus the electron index :math:`p` and the
microphysics :math:`(\epsilon_E, \epsilon_B)`. The optically thin slope measures :math:`p`.
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
# A Source and Its Numerical SED
# ------------------------------
#
# We take a supernova shell with :math:`B = 1` G and :math:`R = 10^{16}` cm at 20 Mpc. The electron
# normalization is fixed by equipartition, so :math:`B` and :math:`R` are the only free source properties.

engine = NonRelativisticSphericalSynchrotronEngine()
engine.load_avg_first_kernel()

B_true, R_true, D = 1.0 * u.G, 1.0e16 * u.cm, 20.0 * u.Mpc
micro = dict(epsilon_E=0.1, epsilon_B=0.1, gamma_min=1.0, gamma_max=1e6)
geometry = dict(f_A=0.5, f_V=0.5)

nu = np.geomspace(0.1, 300.0, 60) * u.GHz
gamma = np.geomspace(micro["gamma_min"], micro["gamma_max"], 400)


def numerical_sed(p):
    """Numerical flux density of the source for electron index ``p``."""
    N0 = PowerLaw.normalize_from_magnetic_field(B_true, p=p, **micro)
    N = N0.to_value(u.cm**-3) * PowerLaw.pdf(gamma, p=p, gamma_min=gamma[0], gamma_max=gamma[-1])
    return engine.compute_flux_density(nu, R=R_true, B=B_true, N=N, gamma=gamma, luminosity_distance=D, **geometry).to(
        u.mJy
    )


# %%
# Extracting the Break
# --------------------
#
# We fit a smoothly broken power law, with a fixed :math:`\nu^{5/2}` optically thick slope and a free thin
# slope :math:`\beta`. The fitted intersection of the two asymptotes is :math:`(\nu_{\rm brk}, F_{\rm brk})`,
# and for uncooled electrons :math:`p = 1 - 2\beta`. The intercept lies above the smoothed peak of the
# data, because it is the flux of the optically thick asymptote at :math:`\nu_{\rm brk}`, which is the
# convention the inversion expects.


def sbpl(log_nu, log_F, log_nu_brk, beta, s):
    x = (log_nu - log_nu_brk) * np.log(10)
    return log_F + s * np.logaddexp(2.5 * x / s, beta * x / s) / np.log(10)


def extract_break(flux):
    """Fit the SED and return ``(nu_brk, F_brk, p)``."""
    log_nu, log_F = np.log10(nu.to_value(u.Hz)), np.log10(flux.to_value(u.Jy))
    guess = [log_F.max(), log_nu[np.argmax(log_F)], -1.0, -1.0]
    fit, _ = curve_fit(sbpl, log_nu, log_F, p0=guess, bounds=([-20, 7, -4, -10], [5, 12, 0, -0.05]))
    return 10 ** fit[1] * u.Hz, 10 ** fit[0] * u.Jy, 1 - 2 * fit[2], fit


flux = numerical_sed(p=3.0)
nu_brk, F_brk, p_meas, fit = extract_break(flux)

fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
ax.loglog(nu.to_value(u.GHz), flux.to_value(u.mJy), "o", ms=4, label="Numerical SED")
ax.loglog(
    nu.to_value(u.GHz), 10 ** sbpl(np.log10(nu.to_value(u.Hz)), *fit) * 1e3, "k-", lw=1.5, label="Broken power-law fit"
)
ax.plot(nu_brk.to_value(u.GHz), F_brk.to_value(u.mJy), "r*", ms=14, label=r"$(\nu_{\rm brk}, F_{\rm brk})$")
ax.set(xlabel="Frequency [GHz]", ylabel="Flux density [mJy]", title=rf"Extracted break ($p_{{\rm meas}}={p_meas:.2f}$)")
ax.legend()
plt.show()

# %%
# Inversion Across Electron Index
# -------------------------------
#
# The inversion depends on :math:`p` through the exponents of its power laws. We repeat the round trip for
# :math:`2.2 \le p \le 3.8`, each time inverting with the :math:`p` measured from the fitted slope.

p_true = np.linspace(2.2, 3.8, 9)
B_ratio, R_ratio = [], []

for p in p_true:
    nu_brk, F_brk, p_meas, _ = extract_break(numerical_sed(p))
    out = invert_ssa_Diggins26(nu_brk, F_brk, D, p=p_meas, pitch_average=True, **micro, **geometry)
    B_ratio.append((out["B"] / B_true).decompose().value)
    R_ratio.append((out["R"] / R_true).decompose().value)

fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
ax.axhspan(0.95, 1.05, color="0.9", label=r"$\pm 5\%$")
ax.plot(p_true, B_ratio, "o-", label=r"$B_{\rm rec}/B_{\rm true}$")
ax.plot(p_true, R_ratio, "s-", label=r"$R_{\rm rec}/R_{\rm true}$")
ax.axhline(1.0, color="k", lw=1)
ax.set(xlabel="Electron index $p$", ylabel="Recovered / true", title="Diggins26 round trip", ylim=(0.94, 1.06))
ax.legend()
plt.show()

# %%
# **Outcome:** across :math:`2.2 \le p \le 3.8` the inversion recovers both :math:`B` and :math:`R` to within
# about 4% of the true values.
