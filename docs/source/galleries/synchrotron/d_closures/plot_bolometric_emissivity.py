r"""
How Much Power Does the Shocked Shell Radiate? Bolometric Synchrotron Emissivity
================================================================================

**Objective:** estimate the bolometric synchrotron luminosity of a supernova shell and find out how much
radiative cooling reduces it.

The power radiated per unit volume is the single-electron synchrotron power integrated over the electron
distribution,

.. math::

    j_{\rm bol} = \frac{4}{3}\sigma_T c\,\frac{B^2}{8\pi}\int N(\gamma)\,\gamma^2\,d\gamma ,

with the normalization of :math:`N(\gamma)` set by the equipartition fractions :math:`\epsilon_E` and
:math:`\epsilon_B`. The luminosity of the shell is :math:`L_{\rm bol} = j_{\rm bol} V`. The classes in
:mod:`trilobite.radiation.synchrotron.electron_distributions` provide :math:`j_{\rm bol}` directly through
:meth:`~trilobite.radiation.synchrotron.electron_distributions.ElectronDistribution.bol_emiss_from_magnetic_field`.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from trilobite.radiation.synchrotron.cooling import SynchrotronRadiativeCoolingEngine
from trilobite.radiation.synchrotron.electron_distributions import BrokenPowerLaw, PowerLaw
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

# %%
# The Shell
# ---------
#
# We take the shell of a Type Ibc supernova 30 days after explosion, with :math:`B = 1` G,
# :math:`R = 10^{16}` cm and a volume filling factor of 0.5. The electrons have :math:`p = 3`
# and equipartition fractions :math:`\epsilon_E = \epsilon_B = 0.1`.

B, R, t, f_V = 1.0 * u.G, 1.0e16 * u.cm, 30.0 * u.day, 0.5
V = 4 / 3 * np.pi * R**3 * f_V
kwargs = dict(epsilon_B=0.1, epsilon_E=0.1, gamma_min=1.0, gamma_max=1e6)

# %%
# Without Cooling
# ---------------
#
# For an uncooled power-law distribution the luminosity follows directly.

j_pl = PowerLaw.bol_emiss_from_magnetic_field(B, p=3.0, **kwargs)
print(f"Uncooled:  j_bol = {j_pl:.2e},  L_bol = {(j_pl * V).to(u.erg / u.s):.2e}")

# %%
# With Cooling
# ------------
#
# Over 30 days electrons above the cooling Lorentz factor :math:`\gamma_c` have radiated away much of their
# energy, which steepens the distribution to :math:`\gamma^{-(p+1)}` above :math:`\gamma_c`. We take
# :math:`\gamma_c` from
# :class:`~trilobite.radiation.synchrotron.cooling.SynchrotronRadiativeCoolingEngine` and use a broken
# power law.

gamma_c = SynchrotronRadiativeCoolingEngine().compute_cooling_gamma(B=B, t=t)
j_bpl = BrokenPowerLaw.bol_emiss_from_magnetic_field(B, p1=3.0, p2=4.0, gamma_c=float(gamma_c), **kwargs)
print(f"Cooled:    j_bol = {j_bpl:.2e},  L_bol = {(j_bpl * V).to(u.erg / u.s):.2e}  (gamma_c = {gamma_c:.0f})")

# %%
# The cooled luminosity depends on where the break falls, so we repeat the calculation over a range of
# :math:`\gamma_c` and mark the value for this shell.

gamma_c_grid = np.geomspace(2, 1e5, 100)
ratio = [
    (BrokenPowerLaw.bol_emiss_from_magnetic_field(B, p1=3.0, p2=4.0, gamma_c=g, **kwargs) / j_pl).decompose().value
    for g in gamma_c_grid
]

fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
ax.semilogx(gamma_c_grid, ratio, lw=2)
ax.plot(gamma_c, (j_bpl / j_pl).decompose(), "ro", ms=9, label=rf"This shell ($\gamma_c={gamma_c:.0f}$)")
ax.axhline(1.0, color="k", ls="--", lw=1, label="Uncooled")
ax.set(xlabel=r"Cooling Lorentz factor $\gamma_c$", ylabel=r"$L_{\rm bol}$ (cooled) / $L_{\rm bol}$ (uncooled)")
ax.legend()
plt.show()

# %%
# **Outcome:** the shell radiates about :math:`1.5\times10^{39}` erg/s if the electrons are uncooled. With the
# cooling expected after 30 days the luminosity is about half of that, :math:`7.7\times10^{38}` erg/s.
# Neglecting cooling would overestimate the radiated power by roughly a factor of two for this source.
