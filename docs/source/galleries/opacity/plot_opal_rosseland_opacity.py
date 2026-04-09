r"""
OPAL Rosseland Mean Opacity: :math:`\kappa(T,\rho)` for Solar Composition
==========================================================================

The OPAL project (Badnell et al. 2005, MNRAS 360:458) provides tables of the
Rosseland mean opacity :math:`\kappa_R(T,\rho)` for a wide range of stellar
compositions.  Triceratops ships a bundled table for solar composition
(:math:`X=0.70,\;Z=0.02`) from which opacity and its log-space derivatives are
evaluated via bilinear interpolation.

This example performs the canonical sanity check: plotting
:math:`\kappa_R` as a function of temperature for several densities,
reproducing the characteristic features of stellar-interior opacity:

- The **electron-scattering plateau** at high :math:`T` (:math:`\kappa \approx 0.34\;\mathrm{cm^2\,g^{-1}}`),
- The **Kramers rise** at intermediate temperatures (:math:`\kappa \propto \rho\,T^{-3.5}`),
- The **opacity peak** near :math:`T \sim 10^5\;\mathrm{K}` driven by iron-group line opacity,
- The rapid **drop at low** :math:`T` as hydrogen and helium become neutral.

.. note::

   The OPAL tables cover :math:`3.75 \le \log_{10}(T\,[\mathrm{K}]) \le 8.70` and
   :math:`-8 \le \log_{10}(R) \le 1`, where
   :math:`R \equiv \rho\,T_6^{-3}` and :math:`T_6 = 10^{-6}\,T`.
   Points that fall outside the tabulated domain—or in cells marked invalid by
   the OPAL authors—are shown as gaps.

Relevant API references
-----------------------
- :func:`~triceratops.radiation.opacity.models.core.load_opal_opacity`
- :class:`~triceratops.radiation.opacity.models.core.OPALOpacity`
- :class:`~triceratops.radiation.opacity.tables.opacity_table.OPALOpacityTable`
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from triceratops.radiation.opacity.models.core import load_opal_opacity
from triceratops.utils.plot_utils import set_plot_style

# %%
# Load the Default Solar Opacity
# ------------------------------
#
# :func:`~triceratops.radiation.opacity.models.core.load_opal_opacity` loads
# table index 72 from the bundled HDF5 file (X = 0.70, Z = 0.02).
# We use ``out_of_bounds='nan'`` so that points outside the tabulated domain
# are silently returned as ``NaN`` and appear as gaps in the plot.

op = load_opal_opacity(72, out_of_bounds="nan")

# %%
# Temperature and Density Grids
# -----------------------------
#
# We sweep temperature from 10³ K to 5×10⁸ K — intentionally a wider range than the
# table domain (:math:`10^{3.75} \approx 5600\;\mathrm{K}` to :math:`5\times10^8\;\mathrm{K}`)
# so the plot shows where the curves begin and end.  Points outside the tabulated range
# are returned as ``NaN`` by the opacity object and appear as gaps.
#
# Because OPAL tables are stored on a :math:`(\log_{10}T,\,\log_{10}R)` grid where
# :math:`R = \rho / T_6^3`, a fixed density :math:`\rho` corresponds to a *varying*
# :math:`\log_{10}R` as temperature changes.  High-density curves therefore go
# out-of-bounds in the :math:`R` direction at low :math:`T`, producing gaps even
# within the nominal temperature range.

n_T = 800
T = np.geomspace(1e3, 5e8, n_T) * u.K

log10_rho_vals = np.array([-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0])
rho_vals = (10.0**log10_rho_vals) * u.Unit("g cm-3")

# %%
# Evaluate :math:`\kappa_R(T,\rho)`
# ----------------------------------
#
# For each density we broadcast the opacity call over the full temperature
# array.  Out-of-domain or NaN-cell results are returned as ``NaN`` and
# masked before plotting.

kappa_curves = []
for rho in rho_vals:
    kappa = op.opacity(rho * np.ones(n_T), T).to_value(u.Unit("cm2 g-1"))
    kappa_curves.append(kappa)

# %%
# Plot
# ----
#
# Lines are colour-coded by :math:`\log_{10}(\rho\,[\mathrm{g\,cm^{-3}}])`
# using a diverging colourmap centred near the stellar interior.  A shared
# colourbar on the right encodes the density axis.

set_plot_style()

cmap = plt.get_cmap("plasma")
norm = Normalize(vmin=log10_rho_vals.min(), vmax=log10_rho_vals.max())

fig, ax = plt.subplots(figsize=(9, 6))

for log10_rho, kappa in zip(log10_rho_vals, kappa_curves):
    color = cmap(norm(log10_rho))
    mask = np.isfinite(kappa) & (kappa > 0)
    ax.semilogy(
        T[mask].to_value(u.K),
        kappa[mask],
        color=color,
        lw=1.8,
    )

ax.set_xscale("log")
ax.set_xlabel(r"Temperature $T$ [K]")
ax.set_ylabel(r"Rosseland mean opacity $\kappa_R$ [cm$^2$ g$^{-1}$]")
ax.set_title(r"OPAL Opacity — Solar Composition ($X=0.70,\;Z=0.02$)")
ax.set_xlim(T.min().to_value(u.K), T.max().to_value(u.K))
ax.grid(True, which="both", ls="--", alpha=0.3)

# Reference line: electron-scattering approximation κ_es ≈ 0.2(1 + X) cm² g⁻¹
# (accurate to ~0.06% vs. the exact Thomson cross-section formula)
kappa_es = 0.2 * (1.0 + 0.70)
ax.axhline(kappa_es, color="k", ls=":", lw=1.2, alpha=0.6, label=rf"$\kappa_\mathrm{{es}}={kappa_es:.2f}$")
ax.legend(loc="upper right", fontsize=9)

# Colourbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r"$\log_{10}\!\left(\rho\;[\mathrm{g\,cm^{-3}}]\right)$")

plt.tight_layout()
plt.show()

# %%
# Discussion
# ----------
#
# The curves reproduce the well-known features of Rosseland mean opacity in
# stellar interiors:
#
# - **High** :math:`T` **plateau** — at :math:`T \gtrsim 10^7\;\mathrm{K}`, where
#   matter is fully ionised, opacity approaches the electron-scattering value
#   :math:`\kappa_\mathrm{es} = 0.2(1+X)\;\mathrm{cm^2\,g^{-1}} \approx 0.34`
#   (dashed line).
# - **Kramers regime** — at intermediate temperatures the opacity rises steeply
#   as :math:`\kappa \propto \rho\,T^{-3.5}`, driven by bound-free and free-free
#   absorption.  Higher-density curves are shifted upward by the :math:`\rho`
#   prefactor.
# - **Iron-group peak** — the prominent bump near
#   :math:`T \sim 1`–:math:`5 \times 10^5\;\mathrm{K}` originates from a forest
#   of bound–bound transitions of partially ionised iron and nickel.  This feature
#   is the driver of the *κ*-mechanism for stellar pulsations.
# - **Low-**:math:`T` **decline** — below :math:`\sim 10^4\;\mathrm{K}` the
#   dominant absorbers (H, He) recombine, sharply reducing the opacity; these
#   cells lie outside the OPAL table and appear as gaps.
#
# The density dependence is strongest in the Kramers regime and vanishes at the
# electron-scattering plateau, consistent with the analytic expectation.

sphinx_gallery_thumbnail_number = -1
