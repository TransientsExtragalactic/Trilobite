r"""
Free-Free Flux Density from a Hot Optically-Thin Plasma
========================================================

In the optically-thin limit the observed flux density from a homogeneous
spherical plasma of radius :math:`R` at distance :math:`d` is

.. math::

    F_\nu = \frac{j_\nu\,V}{d^2},
    \qquad V = \tfrac{4}{3}\pi R^3,

where :math:`j_\nu` [erg s\ :sup:`-1` cm\ :sup:`-3` Hz\ :sup:`-1` sr\ :sup:`-1`]
is the monochromatic bremsstrahlung emissivity returned by
:func:`~triceratops.radiation.free_free.core.compute_ff_emissivity`.  The factor of
:math:`4\pi` sr from integrating over the full sphere cancels the :math:`4\pi d^2`
in the denominator, leaving exactly one steradian to multiply against
:math:`j_\nu` (which carries sr\ :sup:`-1`).

The free-free spectrum is nearly flat below the thermal cutoff
:math:`h\nu \lesssim k_B T` and falls exponentially above it, so the
characteristic rolloff energy :math:`E = k_B T` is directly readable from
the spectrum.

Relevant API
------------
- :func:`~triceratops.radiation.free_free.core.compute_ff_emissivity`
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u

from triceratops.radiation.free_free import compute_ff_emissivity
from triceratops.utils.plot_utils import set_plot_style

set_plot_style()

# %%
# Physical Setup
# ---------------
#
# A hot diffuse plasma — typical of an intracluster or coronal region —
# at three representative temperatures spanning the soft X-ray band.

n_e = 1e-3 / u.cm**3  # electron density
n_i = 1e-3 / u.cm**3  # ion density (fully ionized H)
Z = 1
R = 100 * u.kpc  # source radius
d = 100 * u.Mpc  # luminosity distance

V = (4 / 3) * np.pi * R**3

# Frequency grid: 0.1 – 20 keV in photon energy
E_keV = np.geomspace(0.1, 20, 300) * u.keV
nu = (E_keV / const.h).to(u.Hz)

# %%
# Emissivity and Flux Density
# ----------------------------

TEMPS = [5e6, 2e7, 1e8] * u.K

fig, ax = plt.subplots(figsize=(7, 4.5))

for T in TEMPS:
    j_nu = compute_ff_emissivity(nu, n_e=n_e, n_i=n_i, Z=Z, T=T)
    F_nu = (j_nu * V * u.sr / d**2).to(u.erg / u.s / u.cm**2 / u.Hz)
    kT_keV = (const.k_B * T).to(u.keV).value
    ax.loglog(E_keV.value, F_nu.value, lw=2, label=rf"$T = {T.value:.0e}$ K  ($k_BT = {kT_keV:.2f}$ keV)")

ax.set(
    xlabel=r"Photon energy (keV)",
    ylabel=r"$F_\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)",
    title=rf"Free-free spectrum: $n_e = {n_e.value}$ cm$^{{-3}}$, $R = {R.value:.0f}$ kpc, $d = {d.value:.0f}$ Mpc",
)
ax.legend(fontsize=9)
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.show()
