r"""
Equipartition Analysis from a Radio SED
========================================

The **equipartition analysis** is the standard first-pass method for extracting
physical parameters from a synchrotron radio SED. It was popularized by
:footcite:t:`readheadEquipartitionBrightnessTemperature1994` and is now
used routinely in the analysis of radio supernovae, GRBs, and TDEs.

The key insight is that the total energy in relativistic electrons and magnetic
fields is minimized — and therefore the source is in **equipartition** — when
the energy is roughly equally shared between particles and fields. Under this
assumption, only two observed SED parameters are needed to infer the underlying
source physics:

- The **peak flux density** :math:`F_{\rm pk}` [Jy], and
- The **peak (self-absorption) frequency** :math:`\nu_{\rm pk}` [GHz].

Together these constrain the source radius :math:`R`, magnetic field :math:`B`,
and minimum energy :math:`U_{\rm min}` through the **inverse closure relations**
built into Trilobite.

In this example we carry out a complete equipartition analysis of a synthetic
radio SED:

1. **Construct a representative SED** using the forward closure.
2. **Apply the inverse closure** via
   :meth:`~trilobite.radiation.synchrotron.SEDs.one_zone.seds.PowerLaw_SynchrotronSED.from_params_to_physics`
   to recover :math:`R`, :math:`B`, and :math:`U_{\rm min}`.
3. **Sweep over** :math:`\epsilon_B` to show how inferred quantities scale with
   microphysical assumptions.

.. hint::

    The equipartition analysis yields **minimum-energy estimates**. The true
    source energy is always :math:`\geq U_{\rm min}`. If the source is not in
    equipartition the inferred parameters are still useful fiducial values.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from trilobite.radiation.synchrotron.SEDs import PowerLaw_SynchrotronSED
from trilobite.utils.plot_utils import set_plot_style

# %%
# Generate a Synthetic SED
# ------------------------
#
# We start from a representative synchrotron source and use the forward closure
# to compute the observable peak flux and frequency.

sed = PowerLaw_SynchrotronSED()

R_true = 5e16 * u.cm
B_true = 3000 * u.G
D_L = 10.0 * u.Mpc
p = 3.0
epsilon_e = 0.1
epsilon_B = 0.1
gamma_min = 1.0

fwd = sed.from_physics_to_params(
    B=B_true,
    R=R_true,
    gamma_min=gamma_min,
    gamma_max=1e8,
    p=p,
    epsilon_E=epsilon_e,
    epsilon_B=epsilon_B,
    luminosity_distance=D_L,
    f_V=1.0,
    pitch_average=True,
)

F_pk = fwd["F_peak"]
nu_pk = fwd["nu_m"]

freqs = np.geomspace(1e-2, 1000.0, 300) * u.GHz
flux = sed.sed(freqs, fwd["F_norm"], nu_m=fwd["nu_m"], nu_max=fwd["nu_max"], p=p, s=-0.01)

set_plot_style()

fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(freqs.to_value("GHz"), flux.to_value("Jy"), lw=2)
ax.scatter([nu_pk.to_value("GHz")], [F_pk.to_value("Jy")], color="C1", zorder=5, label="Peak")
ax.set_xlabel("Frequency [GHz]")
ax.set_ylabel("Flux Density [Jy]")
ax.set_title("Synthetic Synchrotron SED")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Inverse Closure
# ---------------
#
# Recover physical parameters from the peak observables.  The round-trip
# should reproduce the input :math:`B` and :math:`R` to numerical precision.

inv = sed.from_params_to_physics(
    F_peak=F_pk,
    nu_peak=nu_pk,
    gamma_min=gamma_min,
    gamma_max=1e8,
    p=p,
    epsilon_E=epsilon_e,
    epsilon_B=epsilon_B,
    luminosity_distance=D_L,
    f_V=1.0,
    pitch_average=True,
)

R_rec = inv["R"]
B_rec = inv["B"]

# %%
# Scaling with Microphysical Parameters
# -------------------------------------
#
# The inferred :math:`B`, :math:`R`, and minimum energy :math:`U_{\rm min}`
# all depend on the assumed :math:`\epsilon_B`.  Varying it while holding
# the observables fixed traces the degeneracy between microphysical
# assumptions and inferred source parameters.

epsilon_B_vals = np.geomspace(1e-3, 0.5, 50)

R_arr, B_arr, U_arr = [], [], []

for eB in epsilon_B_vals:
    res = sed.from_params_to_physics(
        F_peak=F_pk,
        nu_peak=nu_pk,
        gamma_min=gamma_min,
        gamma_max=1e8,
        p=p,
        epsilon_E=epsilon_e,
        epsilon_B=eB,
        luminosity_distance=D_L,
        f_V=1.0,
        pitch_average=True,
    )
    Ri = res["R"]
    Bi = res["B"].to(u.G)
    Ui = Bi.value**2 / (8.0 * np.pi) * (4.0 / 3.0) * np.pi * Ri.to(u.cm).value ** 3 * (1 + epsilon_e / eB)

    R_arr.append(Ri.to(u.cm).value)
    B_arr.append(Bi.value)
    U_arr.append(Ui)

R_arr = np.array(R_arr)
B_arr = np.array(B_arr)
U_arr = np.array(U_arr)

set_plot_style()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].loglog(epsilon_B_vals, B_arr, lw=2, color="C0")
axes[0].axvline(epsilon_B, ls="--", color="gray", alpha=0.7)
axes[0].set_xlabel(r"$\epsilon_B$")
axes[0].set_ylabel(r"Inferred $B$ [G]")
axes[0].set_title(r"Magnetic Field vs $\epsilon_B$")
axes[0].grid(True, which="both", ls="--", alpha=0.3)

axes[1].loglog(epsilon_B_vals, R_arr, lw=2, color="C1")
axes[1].axvline(epsilon_B, ls="--", color="gray", alpha=0.7)
axes[1].set_xlabel(r"$\epsilon_B$")
axes[1].set_ylabel(r"Inferred $R$ [cm]")
axes[1].set_title(r"Source Radius vs $\epsilon_B$")
axes[1].grid(True, which="both", ls="--", alpha=0.3)

axes[2].loglog(epsilon_B_vals, U_arr, lw=2, color="C2")
axes[2].axvline(epsilon_B, ls="--", color="gray", alpha=0.7, label=r"True $\epsilon_B$")
axes[2].set_xlabel(r"$\epsilon_B$")
axes[2].set_ylabel(r"$U_{\rm min}$ [erg cm$^{-3}$ cm$^3$]")
axes[2].set_title(r"Minimum Energy vs $\epsilon_B$")
axes[2].legend()
axes[2].grid(True, which="both", ls="--", alpha=0.3)

plt.suptitle("Equipartition Scaling with Microphysical Assumptions")
plt.tight_layout()
plt.show()
