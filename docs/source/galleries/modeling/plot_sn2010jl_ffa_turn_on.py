r"""
Free-Free Turn-On in a Dense Wind: SN 2010jl
=============================================

**Objective:** produce the radio light curves of a supernova inside an extremely dense circumstellar wind, where
free-free absorption (FFA) in the unshocked wind hides the synchrotron emission until the wind thins out.

The source is SN 2010jl, a luminous Type IIn supernova with a progenitor wind of roughly
:math:`0.1\,M_\odot\,{\rm yr^{-1}}` at 49 Mpc. The model combines:

1. a numerical thin-shell shock in a broken power-law wind (:math:`\rho \propto r^{-2}` inside a break radius,
   steeper outside),
2. synchrotron emission from the shocked gas, from the numerical SED engine, and
3. free-free absorption through the unshocked wind, from
   :func:`~trilobite.radiation.free_free.absorption.compute_ff_RJ_optical_depth_from_quadrature`.

.. note::

    The shock and wind parameters are those of Chandra et al. (2015): :math:`D = 49` Mpc,
    :math:`E_{\rm ej} = 2.1\times10^{51}` erg, :math:`M_{\rm ej} = 8\,M_\odot`, a wind with
    :math:`4\pi r^2 \rho = 4.1\times10^{17}\,{\rm g\,cm^{-1}}` (:math:`\dot M \approx 0.065\,M_\odot\,{\rm yr^{-1}}`
    at 100 km/s) inside :math:`1.29\times10^{16}` cm and :math:`\rho \propto r^{-4.45}` outside, a wind
    temperature of :math:`10^4` K and a hydrogen ionization fraction of about 0.1. The paper does not specify the
    synchrotron microphysics, so we assume :math:`\epsilon_B = 0.1`, :math:`p = 3` and
    :math:`\epsilon_e = 5\times10^{-4}`.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u

from trilobite.dynamics.profiles import BrokenPowerLawCSMProfile, BrokenPowerLawEjectaProfile
from trilobite.dynamics.shocks.core import StrongColdShockConditions
from trilobite.dynamics.shocks.numerical import PressureDrivenThinShellShockEngine
from trilobite.dynamics.shocks.utils import make_homologous_stationary_sources
from trilobite.radiation.free_free.absorption import compute_ff_RJ_optical_depth_from_quadrature
from trilobite.radiation.synchrotron.cooling import SynchrotronRadiativeCoolingEngine
from trilobite.radiation.synchrotron.electron_distributions import PowerLaw
from trilobite.radiation.synchrotron.SEDs.numerical import NonRelativisticSphericalSynchrotronEngine
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

# %%
# The Shock
# ---------
#
# The ejecta have the published energy and mass and a steep outer density profile (:math:`n = 8`). The wind
# density profile is the published broken power law. The thin-shell engine needs an initial radius and speed,
# which we set to 2 days and 1.5 times the transition velocity of the ejecta profile.

E_ej, M_ej, n_ej = 2.1e51 * u.erg, 8.0 * u.Msun, 8
r_break, w_1, slope_out = 1.29e16 * u.cm, 4.1e17 * u.g / u.cm, 4.45
rho_break = w_1 / (4 * np.pi * r_break**2)

wind = BrokenPowerLawCSMProfile.as_optimized_callable(
    density_break=rho_break, radius_break=r_break, slope_inner=2.0, slope_outer=slope_out
)
K, v_t = BrokenPowerLawEjectaProfile.normalize(E_ej, M_ej, n=n_ej, delta=0)
ejecta = BrokenPowerLawEjectaProfile.as_optimized_callable(K=K, v_t=v_t, n=n_ej, delta=0)
rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(ejecta, wind)

t = np.geomspace(30.0, 1200.0, 160) * u.day
t_0, v_0 = 2.0 * u.day, 1.5 * v_t
shock = PressureDrivenThinShellShockEngine().compute_shock_properties(
    t, rho_1=rho_1, rho_4=rho_4, u_1=u_1, u_4=u_4, R_0=(v_0 * t_0).to(u.cm), v_0=v_0, t_0=t_0, M_0=1e-6 * u.Msun
)
R, v = shock.radius.to(u.cm), shock.velocity.to(u.km / u.s)

fig, (ax_R, ax_v) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
ax_R.loglog(t.value, R.value, lw=2)
ax_R.axhline(r_break.value, color="0.5", ls=":", label="Wind break radius")
ax_R.set(xlabel="Time since explosion [day]", ylabel="Shock radius [cm]", title="Shock radius")
ax_R.legend()
ax_v.loglog(t.value, v.value, lw=2)
ax_v.set(xlabel="Time since explosion [day]", ylabel="Shock speed [km/s]", title="Shock speed")
plt.show()

# %%
# Synchrotron Emission
# --------------------
#
# The post-shock thermal energy density sets :math:`B`. The numerical engine integrates the emission and
# self-absorption of a cooled electron distribution in the shocked shell. For the first year :math:`\gamma_c < 1`,
# so every electron is cooled and the emission is strongly suppressed. We assume that electrons carry
# :math:`\epsilon_e = 5\times10^{-4}` of the post-shock energy.

epsilon_e, epsilon_B, p = 5.0e-4, 0.1, 3.0
D, f_V, gamma_min, gamma_max = 49.0 * u.Mpc, 0.5, 1.0, 1.0e6

rho_wind = wind(R.value, 0.0) * u.g / u.cm**3
e_th = StrongColdShockConditions.compute_post_shock_thermal_energy_density(shock_velocity=v, upstream_density=rho_wind)
B = np.sqrt(8 * np.pi * epsilon_B * e_th.to_value(u.erg / u.cm**3)) * u.G
with warnings.catch_warnings():
    # In this dense wind gamma_c < 1 for the first year: every electron is cooled and the engine warns.
    warnings.simplefilter("ignore", RuntimeWarning)
    gamma_c = SynchrotronRadiativeCoolingEngine().compute_cooling_gamma(B=B, t=t)

bands = [5.5, 9.0, 21.2] * u.GHz
engine = NonRelativisticSphericalSynchrotronEngine()
engine.load_avg_first_kernel()
gamma = np.geomspace(gamma_min, gamma_max, 400)

synch = np.empty((t.size, bands.size)) * u.mJy
for i in range(t.size):
    N0 = PowerLaw.normalize_from_magnetic_field(
        B[i], epsilon_B, epsilon_e, p=p, gamma_min=gamma_min, gamma_max=gamma_max
    ).to_value(u.cm**-3)
    N = N0 * np.where(gamma < gamma_c[i], gamma**-p, gamma_c[i] * gamma ** (-p - 1))
    synch[i] = engine.compute_flux_density(
        bands, R=R[i], B=B[i], N=N, gamma=gamma, f_V=f_V, f_A=1.0, luminosity_distance=D
    ).to(u.mJy)

# %%
# Free-Free Absorption
# --------------------
#
# The unshocked wind outside the shock absorbs the synchrotron emission. For a wind that is a fraction
# :math:`x` ionized, :math:`n_e = n_i = x\,\rho/(1.4\,m_p)`, and the optical depth is integrated from the shock
# radius outward. A fully ionized wind absorbs everything, so we use the published :math:`x = 0.1`.

x_ion, T_wind = 0.1, 1.0e4


def wind_density(r):
    """Electron and ion number density of the partly ionized wind [cm^-3]."""
    return x_ion * wind(r, 0.0) / (1.4 * const.m_p.cgs.value)


tau_ff = np.empty_like(synch.value)
for i in range(t.size):
    tau_ff[i] = compute_ff_RJ_optical_depth_from_quadrature(
        frequency=bands.to(u.Hz),
        r=R[i],
        n_e=wind_density,
        n_i=wind_density,
        temperature=lambda r: T_wind,
        r_max=1e19 * u.cm,
    )
observed = synch.value * np.exp(-tau_ff)

# The turn-on time of each band is when the wind becomes optically thin to free-free absorption, tau_ff = 1.
for j, band in enumerate(bands.value):
    t_on = np.interp(1.0, tau_ff[::-1, j], t.value[::-1])
    print(f"{band:>5g} GHz: tau_ff = 1 at day {t_on:.0f}")

# %%
# The Light Curves
# ----------------
#
# The solid curves are the synchrotron emission after free-free absorption and the dashed curves are the emission
# without it. The three bands switch on one after another, highest frequency first.

fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
for j, (band, color) in enumerate(zip(bands.value, ["C3", "C1", "C0"])):
    ax.loglog(t.value, synch[:, j].to_value(u.uJy), "--", color=color, alpha=0.6)
    ax.loglog(t.value, observed[:, j] * 1e3, color=color, lw=2.5, label=f"{band:g} GHz")
ax.set(xlabel="Time since explosion [day]", ylabel=r"Flux density [$\mu$Jy]", ylim=(1e-2, 3e4), xlim=(30, 1200))
ax.plot([], [], "k-", lw=2.5, label="With FFA")
ax.plot([], [], "k--", alpha=0.6, label="Without FFA")
ax.legend(loc="upper left")
plt.show()

# %%
# **Outcome:** the wind becomes transparent to free-free absorption (:math:`\tau_{\rm ff} = 1`) at day 527, 688 and
# 801 for 21.2, 9 and 5.5 GHz, so the bands switch on in order of frequency. The unabsorbed synchrotron emission
# peaks near :math:`10^3\,\mu{\rm Jy}` within the first year, and the absorption hides it until then. A fully
# ionized wind would hide the source for years.
