r"""
Bolometric Synchrotron Emissivity and Equipartition
===================================================

The **bolometric synchrotron emissivity** is the total radiated power per unit
volume integrated over all frequencies. For a power-law electron distribution

.. math::

    N(\gamma) = N_0 \gamma^{-p}

in a magnetic field :math:`B`, the total emissivity follows from integrating
the single-electron synchrotron power

.. math::

    P(\gamma) =
    \frac{4}{3}\sigma_T c U_B \gamma^2,

where :math:`U_B = B^2/(8\pi)` is the magnetic energy density.

The bolometric emissivity therefore becomes

.. math::

    j_{\rm bol}
    =
    \frac{4}{3}\sigma_T c \frac{B^2}{8\pi}
    \int_{\gamma_{\min}}^{\gamma_{\max}}
    N(\gamma)\gamma^2 d\gamma.

For a fixed electron normalization :math:`N_0` this yields the scaling

.. math::

    j_{\rm bol} \propto N_0 B^2.

Equipartition
-------------

In many astrophysical systems the magnetic field and electron population are
tied to the post-shock thermal energy density :math:`u_{\rm therm}` through
equipartition fractions

.. math::

    u_B = \epsilon_B u_{\rm therm},
    \qquad
    u_e = \epsilon_E u_{\rm therm}.

This implies

.. math::

    B = \sqrt{8\pi\,\epsilon_B\,u_{\rm therm}},
    \qquad
    N_0 \propto u_e \propto u_{\rm therm}.

Since :math:`u_{\rm therm} \propto B^2`, the electron normalization obeys

.. math::

    N_0 \propto B^2.

Combining this with the intrinsic synchrotron scaling
:math:`j_{\rm bol} \propto N_0 B^2` yields

.. math::

    j_{\rm bol} \propto B^4.

This scaling is independent of the power-law index :math:`p`.

"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from trilobite.radiation.synchrotron.electron_distributions import (
    BrokenPowerLaw,
    PowerLaw,
    equipartition_magnetic_field,
)
from trilobite.utils.plot_utils import set_plot_style

# %%
# Section 1: Bolometric Emissivity vs Magnetic Field
# --------------------------------------------------
#
# At fixed electron normalization :math:`N_0`, the bolometric synchrotron
# emissivity scales as
#
# .. math::
#
#     j_{\rm bol} \propto B^2.
#
# However, if the electron normalization is determined by equipartition,
# then :math:`N_0 \propto B^2`. The resulting emissivity scaling becomes
#
# .. math::
#
#     j_{\rm bol} \propto B^4.
#
# This scaling is independent of the power-law index :math:`p`.  Below we
# compute the emissivity for several values of :math:`p` to illustrate
# that the slope is unchanged.

set_plot_style()

B_arr = np.geomspace(0.01, 10.0, 200) * u.G

p_values = [2.0, 2.5, 3.0, 3.5]

gamma_min = 100
gamma_max = 1e6

epsilon_E = 0.1
epsilon_B = 0.1

fig, ax = plt.subplots(figsize=(8, 5))

for p in p_values:
    j = PowerLaw.bol_emiss_from_magnetic_field(
        B_arr,
        epsilon_B=epsilon_B,
        epsilon_E=epsilon_E,
        p=p,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    ax.loglog(
        B_arr.to_value(u.G),
        j.to_value(u.erg / u.s / u.cm**3),
        lw=2,
        label=rf"$p={p}$",
    )

# reference B^4 guide
B_ref = 0.1
j_ref = PowerLaw.bol_emiss_from_magnetic_field(
    B_ref * u.G,
    epsilon_B=epsilon_B,
    epsilon_E=epsilon_E,
    p=2.5,
    gamma_min=gamma_min,
    gamma_max=gamma_max,
)

guide = j_ref.to_value(u.erg / u.s / u.cm**3) * (B_arr.value / B_ref) ** 4

ax.loglog(B_arr.value, guide, "k--", lw=1, label=r"$\propto B^4$")

ax.set_xlabel(r"Magnetic field $B$ [G]")
ax.set_ylabel(r"$j_{\rm bol}$ [erg s$^{-1}$ cm$^{-3}$]")
ax.set_title("Bolometric Synchrotron Emissivity vs Magnetic Field")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# Section 2: Power-Law vs Broken Power-Law Emissivity
# ---------------------------------------------------
#
# Radiative cooling modifies the electron distribution. A cooled population
# is described by a broken power-law
#
# .. math::
#
#     N(\gamma) \propto
#     \begin{cases}
#         \gamma^{-p}, & \gamma < \gamma_c \\
#         \gamma^{-(p+1)}, & \gamma > \gamma_c
#     \end{cases}
#
# The steepening above :math:`\gamma_c` reduces the contribution of
# high-energy electrons to the bolometric emissivity.  The ratio
#
# .. math::
#
#     j_{\rm BPL}/j_{\rm PL}
#
# therefore measures the suppression caused by cooling.

B_demo = 0.5 * u.G
p = 2.5

gamma_c = np.geomspace(200, 5e5, 200)

j_pl = PowerLaw.bol_emiss_from_magnetic_field(
    B_demo,
    epsilon_B=epsilon_B,
    epsilon_E=epsilon_E,
    p=p,
    gamma_min=gamma_min,
    gamma_max=gamma_max,
)

j_bpl = np.array(
    [
        BrokenPowerLaw.bol_emiss_from_magnetic_field(
            B_demo,
            epsilon_B=epsilon_B,
            epsilon_E=epsilon_E,
            p1=p,
            p2=p + 1,
            gamma_c=float(gc),
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        ).to_value(u.erg / u.s / u.cm**3)
        for gc in gamma_c
    ]
) * (u.erg / u.s / u.cm**3)

fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

axes[0].loglog(
    gamma_c,
    j_bpl.to_value(u.erg / u.s / u.cm**3),
    lw=2,
    label="Cooled BPL",
)

axes[0].axhline(
    j_pl.to_value(u.erg / u.s / u.cm**3),
    ls="--",
    color="k",
    label="Uncooled PL",
)

axes[0].set_ylabel(r"$j_{\rm bol}$")
axes[0].set_title("Cooling Suppression of Bolometric Emissivity")
axes[0].legend()
axes[0].grid(True, which="both", ls="--", alpha=0.3)

axes[1].semilogx(
    gamma_c,
    (j_bpl / j_pl).value,
    lw=2,
)

axes[1].axhline(1, ls="--", color="k")
axes[1].set_xlabel(r"Cooling Lorentz factor $\gamma_c$")
axes[1].set_ylabel(r"$j_{\rm BPL}/j_{\rm PL}$")
axes[1].grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# Section 3: Equipartition Magnetic Field
# ---------------------------------------
#
# Solving :math:`u_B = B^2/(8\pi) = \epsilon_B u_{\rm therm}` for :math:`B`
# gives :math:`B_{\rm eq} \propto u_{\rm therm}^{1/2}`, independent of
# :math:`\epsilon_B` in slope (only the intercept shifts).  Below we confirm
# this for several values of :math:`\epsilon_B`.

u_therm = np.geomspace(1e-4, 1e6, 200) * u.erg / u.cm**3

eps_vals = [0.01, 0.1, 0.33, 1]

fig, ax = plt.subplots(figsize=(8, 5))

for eps in eps_vals:
    B_eq = equipartition_magnetic_field(
        u_therm=u_therm,
        epsilon_B=eps,
    )

    ax.loglog(
        u_therm.to_value(u.erg / u.cm**3),
        B_eq.to_value(u.G),
        lw=2,
        label=rf"$\epsilon_B={eps}$",
    )

# guide slope
B_ref_eq = equipartition_magnetic_field(
    u_therm=1 * u.erg / u.cm**3,
    epsilon_B=0.1,
)

guide = B_ref_eq.value * (u_therm.value) ** 0.5

ax.loglog(u_therm.value, guide, "k--", label=r"$\propto u_{\rm therm}^{1/2}$")

ax.set_xlabel(r"$u_{\rm therm}$ [erg cm$^{-3}$]")
ax.set_ylabel(r"$B_{\rm eq}$ [G]")
ax.set_title("Equipartition Magnetic Field Scaling")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.show()
