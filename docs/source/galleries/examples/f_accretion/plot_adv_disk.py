r"""
Advective Cooling in One-Zone Accretion Disks
==============================================

Viscously heated accretion disks do not always radiate all of their dissipated
energy locally.  When the accretion rate is high (or the disk is geometrically
thick) a substantial fraction of the viscous heating can be carried inward with
the accreting gas rather than radiated away — a process called *advective
cooling*.

:class:`~triceratops.dynamics.accretion.one_zone.igP_es_advDisk` extends the
standard :class:`~triceratops.dynamics.accretion.one_zone.igP_esDisk` by
splitting the viscous dissipation rate into a radiated component and an
advected component:

.. math::

    q_{\rm visc} = q_{\rm rad} + q_{\rm adv},\quad
    q_{\rm adv}  = q_{\rm visc}\,B\,c_s^2,

where

.. math::

    B = \frac{4}{9\pi}\,\xi\,F_0\,\alpha\,\frac{M_D}{R_D^4\,\Omega^2\,\Sigma}.

The dimensionless entropy-gradient parameter :math:`\xi` controls the
advective fraction.  In the standard ADAF (Advection-Dominated Accretion
Flow) scaling the advective fraction is proportional to :math:`\xi\alpha(H/R)^2`.

Setting :math:`\xi \to 0` recovers the non-advective
:class:`~triceratops.dynamics.accretion.one_zone.igP_esDisk` limit.

Relevant API References
-----------------------
- :class:`~triceratops.dynamics.accretion.one_zone.igP_es_advDisk`
- :class:`~triceratops.dynamics.accretion.one_zone.igP_esDisk`
- :meth:`~triceratops.dynamics.accretion.one_zone.OneZoneAccretionDiskBase.solve`

.. hint::

    For the derivation of the advective energy balance and the B factor see
    :ref:`one_zone_disk_theory`.
"""

# %%
# Setup
# -----

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u

from triceratops.dynamics.accretion.one_zone import igP_esDisk, igP_es_advDisk
from triceratops.utils.plot_utils import set_plot_style

set_plot_style()

# %%
# Shared Parameters
# -----------------
#
# We use a 3 M☉ BH with a 0.1 M☉ initial disk — a configuration representative
# of a compact-object TDE or a neutron-star merger remnant.

M_BH = 3.0 * const.M_sun
R_in = 3.0e6 * u.cm
alpha = 0.1
mu = 0.62

# All models share the same geometry constants so we can use any of them
# to generate consistent initial conditions.
_ref_disk = igP_es_advDisk(mu=mu, xi=0.5)

ic = _ref_disk.generate_initial_conditions(
    M_BH=M_BH,
    M_D_0=0.1 * const.M_sun,
    R_D_0=3.0e13 * u.cm,
)

run_params = {"M_BH": M_BH, "R_in": R_in, "alpha": alpha}
t_span = (1.0e6 * u.s, 1.0e9 * u.s)
max_steps = 50_000

# %%
# Effect of the ξ Parameter
# --------------------------
#
# We run ``igP_es_advDisk`` for three values of ξ to show how the advective
# fraction Q_adv / Q_visc scales with the entropy-gradient parameter.

xi_values = [0.1, 0.5, 1.0]
adv_results = {}

for xi_val in xi_values:
    disk = igP_es_advDisk(mu=mu, xi=xi_val)
    adv_results[xi_val] = disk.solve(ic, run_params, t_span, max_steps=max_steps)

# %%
# Advective Fraction vs. Time
# ----------------------------
#
# The advective fraction Q_adv / Q_visc shows the fraction of viscous heating
# that is carried inward rather than radiated.  For thin disks (H/R ≪ 1) this
# fraction is small; it grows as the disk thickens at early times.

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for xi_val, res in adv_results.items():
    t = res.data["t"].to(u.day).value
    frac = (res.data["Q_adv"] / res.data["Q_visc"]).value
    axes[0].semilogx(t, frac, label=rf"$\xi = {xi_val}$")

axes[0].set_xlabel("Time [days]")
axes[0].set_ylabel(r"$q_{\rm adv} / q_{\rm visc}$")
axes[0].set_title("Advective Fraction")
axes[0].legend()
axes[0].grid(True, which="both", ls="--", alpha=0.4)

# ---- Aspect ratio (H/R) — governs how thick the disk is ----
for xi_val, res in adv_results.items():
    t = res.data["t"].to(u.day).value
    H_over_R = res.data["H_over_R"].value
    axes[1].loglog(t, H_over_R, label=rf"$\xi = {xi_val}$")

axes[1].set_xlabel("Time [days]")
axes[1].set_ylabel(r"$H/R$")
axes[1].set_title("Disk Aspect Ratio")
axes[1].legend()
axes[1].grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.show()

# %%
# Temperature Suppression by Advection
# --------------------------------------
#
# Because advection carries away a fraction of the viscous heating, less energy
# is available for radiation.  The energy-balance equation
#
# .. math::
#
#     1 = A\,c_s^{-2}\,T_c^4 + B\,c_s^2
#
# has a lower root :math:`T_c` when :math:`B > 0` compared with the pure
# radiation solution :math:`T_c^{(0)} \propto c_s (A^{-1})^{1/4}`.
#
# Here we compare the midplane temperature T_c for the non-advective baseline
# (``igP_esDisk``, ξ = 0) and two advective runs.

disk_no_adv = igP_esDisk(mu=mu)
result_no_adv = disk_no_adv.solve(ic, run_params, t_span, max_steps=max_steps)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ---- Midplane temperature ----
t_ref = result_no_adv.data["t"].to(u.day).value
T_ref = result_no_adv.data["T_c"].to(u.K).value
axes[0].loglog(t_ref, T_ref, "k--", lw=1.5, label=r"$\xi = 0$ (no advection)")

for xi_val in [0.1, 1.0]:
    res = adv_results[xi_val]
    t = res.data["t"].to(u.day).value
    T = res.data["T_c"].to(u.K).value
    axes[0].loglog(t, T, label=rf"$\xi = {xi_val}$")

axes[0].set_xlabel("Time [days]")
axes[0].set_ylabel(r"$T_c\;[\mathrm{K}]$")
axes[0].set_title("Midplane Temperature")
axes[0].legend()
axes[0].grid(True, which="both", ls="--", alpha=0.4)

# ---- Accretion rate ----
axes[1].loglog(
    t_ref,
    result_no_adv.data["mdot"].to(u.Msun / u.yr).value,
    "k--",
    lw=1.5,
    label=r"$\xi = 0$ (no advection)",
)
for xi_val in [0.1, 1.0]:
    res = adv_results[xi_val]
    t = res.data["t"].to(u.day).value
    mdot = res.data["mdot"].to(u.Msun / u.yr).value
    axes[1].loglog(t, mdot, label=rf"$\xi = {xi_val}$")

axes[1].set_xlabel("Time [days]")
axes[1].set_ylabel(r"$\dot{M}\;[M_\odot\,\mathrm{yr}^{-1}]$")
axes[1].set_title("Accretion Rate")
axes[1].legend()
axes[1].grid(True, which="both", ls="--", alpha=0.4)

# sphinx_gallery_thumbnail_number = -1
plt.tight_layout()
plt.show()
