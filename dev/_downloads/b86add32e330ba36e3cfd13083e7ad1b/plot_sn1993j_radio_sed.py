r"""
A Radio Supernova in a Red-Supergiant Wind: SN 1993J
=====================================================

**Objective:** build the radio SEDs and light curves of a supernova from a handful of physical parameters, with the
shock, the magnetic field and the electron population all tied to the same wind and shock speed.

The pipeline has four stages, each handled by one Trilobite component:

1. the shock radius :math:`R(t)` and speed :math:`V(t)`,
2. the strong-shock jump conditions, which give the post-shock thermal energy density and, with a magnetic energy
   fraction :math:`\epsilon_B`, the field :math:`B(t)`,
3. the cooling engine, which gives the cooling Lorentz factor :math:`\gamma_c(t)`,
4. the numerical synchrotron engine, which integrates the radiative transfer for the SED at each epoch.

.. note::

    All parameters are from Fransson & Bjornsson (1998) for SN 1993J. The distance is 3.63 Mpc, the wind has
    :math:`\dot M = 5\times10^{-5}\,M_\odot\,{\rm yr^{-1}}` at 10 km/s, and the shock moves at
    :math:`2.2\times10^4` km/s for 100 days and then decelerates as :math:`R \propto t^{0.74}`. The injected
    electrons have :math:`p = 2.1`. The paper finds a magnetic energy density of 0.14 and a relativistic electron
    energy density of :math:`5\times10^{-4}` (extrapolating the spectrum to :math:`\gamma = 1`) relative to the
    thermal energy density, which it takes as :math:`\rho V^2`. The jump conditions give a thermal energy
    density of :math:`\tfrac{9}{8}\rho V^2`, so both fractions are scaled by 8/9 here. Free-free absorption
    and Coulomb losses, which the paper also includes, are not modeled.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from trilobite.dynamics.shocks.core import StrongColdShockConditions
from trilobite.radiation.synchrotron.cooling import SynchrotronRadiativeCoolingEngine
from trilobite.radiation.synchrotron.electron_distributions import PowerLaw
from trilobite.radiation.synchrotron.SEDs.numerical import NonRelativisticSphericalSynchrotronEngine
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

# %%
# Parameters
# ----------
#
# The energy fractions are the paper's, scaled from its thermal energy density :math:`\rho V^2` to the
# :math:`\tfrac{9}{8}\rho V^2` used by the jump conditions. The area filling factor :math:`f_A = 0.1` is an
# effective value, not a measurement: a thin emitting shell is optically thick only near its limb, so the
# emitting area is a fraction of :math:`\pi R^2`, which the one-zone engine represents with :math:`f_A`. The
# optically thick flux scales linearly with it, so it sets the overall flux level.

D = 3.63 * u.Mpc
M_dot, v_wind = 5.0e-5 * u.Msun / u.yr, 10.0 * u.km / u.s
V_0, t_p, m = 22_000.0 * u.km / u.s, 100.0 * u.day, 0.74

p, gamma_min, gamma_max = 2.1, 1.0, 1.0e6
epsilon_B = 0.14 * 8 / 9
epsilon_E = 5.1e-4 * 8 / 9
f_V, f_A = 0.5, 0.1

# %%
# Shock and Magnetic Field
# ------------------------
#
# The shock speed is constant at :math:`V_0` until :math:`t_p` and then falls as :math:`t^{m-1}`, so the radius is
# the integral of that speed. The upstream density is the wind density at the shock radius, and the post-shock
# thermal energy density sets :math:`B = \sqrt{8\pi\,\epsilon_B\,e_{\rm th}}`.

t = np.geomspace(10.0, 1300.0, 40) * u.day
x = (t / t_p).decompose().value

V = V_0 * np.where(x < 1, 1.0, x ** (m - 1))
R = np.where(x < 1, V_0 * t, V_0 * t_p * (1 + (x**m - 1) / m)).to(u.cm)

rho_wind = (M_dot / (4 * np.pi * R**2 * v_wind)).to(u.g / u.cm**3)
e_th = StrongColdShockConditions.compute_post_shock_thermal_energy_density(shock_velocity=V, upstream_density=rho_wind)
B = np.sqrt(8 * np.pi * epsilon_B * e_th.to_value(u.erg / u.cm**3)) * u.G

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for ax, y, label in zip(
    axes, [R.value, V.to_value(u.km / u.s), B.value], ["Shock radius [cm]", "Shock speed [km/s]", "Magnetic field [G]"]
):
    ax.loglog(t.value, y, lw=2)
    ax.set(xlabel="Time since explosion [day]", ylabel=label)
plt.show()

# %%
# Because the same shock speed and wind density enter both :math:`B` and the energy density, the field falls as
# :math:`B \propto V/R \approx t^{-1}`. At day 10 it is about 34 G, the value adopted in the paper.
#
# SEDs From the Numerical Engine
# ------------------------------
#
# Electrons above the cooling Lorentz factor :math:`\gamma_c` have lost their energy. We use
# :math:`N \propto \gamma^{-p}` below :math:`\gamma_c` and :math:`\gamma^{-(p+1)}` above it, normalized by the
# energy fraction :math:`\epsilon_E`, and let the numerical engine do the radiative transfer, including
# self-absorption.

engine = NonRelativisticSphericalSynchrotronEngine()
engine.load_avg_first_kernel()
cooling = SynchrotronRadiativeCoolingEngine()

nu = np.geomspace(0.3, 300.0, 120) * u.GHz
gamma = np.geomspace(gamma_min, gamma_max, 500)

gamma_c = cooling.compute_cooling_gamma(B=B, t=t)
flux = np.empty((t.size, nu.size)) * u.mJy
for i in range(t.size):
    N0 = PowerLaw.normalize_from_magnetic_field(
        B[i], epsilon_B, epsilon_E, p=p, gamma_min=gamma_min, gamma_max=gamma_max
    ).to_value(u.cm**-3)
    N = N0 * np.where(gamma < gamma_c[i], gamma**-p, gamma_c[i] * gamma ** (-p - 1))
    flux[i] = engine.compute_flux_density(
        nu, R=R[i], B=B[i], N=N, gamma=gamma, f_V=f_V, f_A=f_A, luminosity_distance=D
    ).to(u.mJy)

# %%
# SED Evolution and Light Curves
# ------------------------------
#
# The SED peak moves to lower frequency as the shock expands, which delays the light curves at low frequencies.

fig, (ax_sed, ax_lc) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

norm = LogNorm(t.value.min(), t.value.max())
cmap = plt.cm.viridis
for i in range(t.size):
    ax_sed.loglog(nu.value, flux[i].value, color=cmap(norm(t[i].value)), lw=1.2)
ax_sed.set(xlabel="Frequency [GHz]", ylabel="Flux density [mJy]", title="SED evolution", ylim=(1e-2, 2e3))
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax_sed, label="Time since explosion [day]")

for freq, color in zip([1.4, 5.0, 22.0], ["C3", "C1", "C0"]):
    j = np.argmin(np.abs(nu.value - freq))
    ax_lc.loglog(t.value, flux[:, j].value, color=color, lw=2, label=f"{freq:g} GHz")
ax_lc.set(xlabel="Time since explosion [day]", ylabel="Flux density [mJy]", title="Light curves", ylim=(1e-1, 2e3))
ax_lc.legend()
plt.show()
