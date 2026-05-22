"""
Optical Light Curve Container
==============================

An :class:`~trilobite.data.light_curve.OpticalLightCurveContainer` holds
time-series photometry in a **single fixed optical band**.  Unlike
:class:`~trilobite.data.optical_photometry.OpticalPhotometryContainer`, the
band is stored as metadata rather than a per-row column — this is the natural
representation once you have isolated a single filter's light curve.

Both flux and magnitude representations are always accessible, regardless of
which columns were supplied at construction time.
"""

# %%
# Building the Container
# -----------------------
#
# We construct a synthetic g-band light curve that contains both detections
# and upper limits, supplying flux columns.  Magnitude properties are then
# computed on-the-fly by the container.

import numpy as np
from astropy import units as u
from astropy.table import Table

from trilobite.data import OpticalLightCurveContainer

rng = np.random.default_rng(42)
n = 22
time = np.sort(rng.uniform(5, 300, n)) * u.day

flux_unit = u.Unit("erg/(s cm2 Hz)")
flux = rng.lognormal(-63, 0.3, n) * flux_unit
err = flux * rng.uniform(0.05, 0.15, n)
ul = np.full(n, np.nan) * flux_unit

# Mark the last four rows as upper limits.
flux[-4:] = np.nan * flux_unit
err[-4:] = np.nan * flux_unit
ul[-4:] = rng.uniform(1e-30, 3e-30, 4) * flux_unit

table = Table({"time": time, "flux_density": flux, "flux_density_error": err, "flux_upper_limit": ul})
lc = OpticalLightCurveContainer(table, band="g")

print(f"Band         : {lc.band}")
print(f"Observations : {lc.n_obs}")
print(f"Detections   : {lc.n_detections}")
print(f"Upper limits : {lc.n_non_detections}")

# %%
# Flux and Magnitude Representations
# ------------------------------------
#
# Regardless of the input format, :attr:`~trilobite.data.light_curve.OpticalLightCurveContainer.flux`
# and :attr:`~trilobite.data.light_curve.OpticalLightCurveContainer.mag` are
# always available as complementary views of the same data.

det = lc.detection_mask
print("\nFlux density (first 3 detections):")
print(lc.flux[det][:3])

print("\nAB magnitude (first 3 detections, derived):")
print(lc.mag[det][:3])

# %%
# Plotting
# ---------
#
# The dual representation makes it easy to show both a linear-flux and a
# magnitude panel side-by-side.

import matplotlib.pyplot as plt

from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

ul = lc.non_detection_mask

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(f"Synthetic {lc.band!r}-band optical light curve")

# --- flux panel ---
ax = axes[0]
ax.errorbar(
    lc.time[det].to_value(u.day),
    lc.flux[det].to_value(flux_unit),
    yerr=lc.flux_error[det].to_value(flux_unit),
    fmt="o",
    color="steelblue",
    label="Detection",
)
ax.errorbar(
    lc.time[ul].to_value(u.day),
    lc.flux_upper_limit[ul].to_value(flux_unit),
    fmt="v",
    color="gray",
    alpha=0.7,
    label="Upper limit",
)
ax.set_yscale("log")
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"$F_\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
ax.legend()

# --- magnitude panel ---
ax = axes[1]
ax.errorbar(
    lc.time[det].to_value(u.day),
    lc.mag[det],
    yerr=lc.mag_error[det],
    fmt="o",
    color="steelblue",
    label="Detection",
)
ax.errorbar(
    lc.time[ul].to_value(u.day),
    lc.mag_upper_limit[ul],
    fmt="^",
    color="gray",
    alpha=0.7,
    label="Upper limit (mag)",
)
ax.invert_yaxis()
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"$m_\mathrm{AB}$ (mag)")
ax.legend()

plt.tight_layout()
plt.show()
