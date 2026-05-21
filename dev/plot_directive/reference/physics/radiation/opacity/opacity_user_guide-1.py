import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from triceratops.radiation.opacity import OPALOpacity, KramersOpacity, TOPSOpacity

# ---------------------------------------------------------------------
# Sampling grid
# ---------------------------------------------------------------------
rho = np.logspace(-6, 1, 120) * u.g / u.cm**3
T   = np.logspace(3.75, 9, 120) * u.K
RHO, TMP = np.meshgrid(rho, T, indexing="ij")

rho_q = RHO.ravel()
T_q   = TMP.ravel()

# ---------------------------------------------------------------------
# Evaluate opacity models
# ---------------------------------------------------------------------
opal = OPALOpacity.load_default(out_of_bounds="nan")
kram = KramersOpacity()
tops = TOPSOpacity.load_default(mean_type="rosseland", out_of_bounds="nan")

lg_opal = np.log10(opal.opacity(rho_q, T_q).reshape(RHO.shape).cgs.value)
lg_kram = np.log10(kram.opacity(rho_q, T_q).reshape(RHO.shape).cgs.value)
lg_tops = np.log10(tops.opacity(rho_q, T_q).reshape(RHO.shape).cgs.value)

# ---------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------
vmin, vmax = -2, 6

cmap = plt.get_cmap("plasma").copy()
cmap.set_bad(color="k")

fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)

panels = [
    (lg_opal, "OPAL (Rosseland)"),
    (lg_kram, "Kramers (analytic)"),
    (lg_tops, "TOPS (Rosseland)"),
]

# ---------------------------------------------------------------------
# Plot panels
# ---------------------------------------------------------------------
for ax, (surf, title) in zip(axes.flat, panels):
    im = ax.pcolormesh(
        T.value,
        rho.value,
        surf,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11)

# Hide unused panel
axes[1, 1].axis("off")

# Axis labels (only outer)
for ax in axes[1]:
    ax.set_xlabel("Temperature [K]")
for ax in axes[:, 0]:
    ax.set_ylabel(r"Density [g cm$^{-3}$]")

# ---------------------------------------------------------------------
# Inset colorbar
# ---------------------------------------------------------------------
cb_parent = axes[1, 1]
cb_ax = inset_axes(
    cb_parent,
    width="6%",
    height="95%",
    loc="center left",
    borderpad=1,
)

cbar = fig.colorbar(
    im,
    cax=cb_ax,
)
cbar.set_label(r"$\log_{10}\,\kappa\ \mathrm{[cm^2\,g^{-1}]}$")

# ---------------------------------------------------------------------
# Layout polish
# ---------------------------------------------------------------------
plt.tight_layout()
plt.show()