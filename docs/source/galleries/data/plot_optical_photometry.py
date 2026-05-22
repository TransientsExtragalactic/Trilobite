"""
Optical Photometry Container
=============================

An :class:`~trilobite.data.optical_photometry.OpticalPhotometryContainer`
holds multi-band optical photometry from survey instruments such as ZTF,
LSST/Rubin, or DECam.  Observations may be supplied as AB magnitudes, flux
densities, or a mix — the container exposes both representations on demand.
"""

# %%
# Building a Multi-Band Container
# --------------------------------
#
# The schema requires ``time`` and ``band_name`` columns, plus at least one
# y-column group: either flux (``flux_density``, ``flux_density_error``,
# ``flux_upper_limit``) or magnitude (``mag_ab``, ``mag_ab_error``,
# ``mag_ab_upper_limit``).
#
# Here we synthesise a four-band (g, r, i, z) light curve with a smooth
# power-law decline and scattered upper limits, as might be observed from a
# supernova at late times.

import numpy as np
from astropy import units as u
from astropy.table import Table

from trilobite.data import OpticalPhotometryContainer

rng = np.random.default_rng(99)

bands = ["g", "r", "i", "z"]
# Flux scales decrease toward redder bands (simple colour term).
band_scale = {"g": 1.0, "r": 0.75, "i": 0.55, "z": 0.40}
n_per_band = 20

rows = {k: [] for k in ["time", "band_name", "flux_density", "flux_density_error", "flux_upper_limit"]}

for band in bands:
    t = np.sort(rng.uniform(3, 250, n_per_band))
    # Power-law decline in flux (fading transient).
    f = 8e-28 * band_scale[band] * (t / 10.0) ** -1.1 * rng.lognormal(0, 0.15, n_per_band)
    err = f * rng.uniform(0.05, 0.12, n_per_band)

    # Last three epochs per band become upper limits.
    ul = np.full(n_per_band, np.nan)
    ul[-3:] = rng.uniform(1.5e-29, 4e-29, 3)
    f[-3:] = np.nan
    err[-3:] = np.nan

    rows["time"].extend(t)
    rows["band_name"].extend([band] * n_per_band)
    rows["flux_density"].extend(f)
    rows["flux_density_error"].extend(err)
    rows["flux_upper_limit"].extend(ul)

flux_unit = u.Unit("erg/(s cm2 Hz)")
table = Table(
    {
        "time": np.array(rows["time"]) * u.day,
        "band_name": np.array(rows["band_name"]),
        "flux_density": np.array(rows["flux_density"]) * flux_unit,
        "flux_density_error": np.array(rows["flux_density_error"]) * flux_unit,
        "flux_upper_limit": np.array(rows["flux_upper_limit"]) * flux_unit,
    }
)

phot = OpticalPhotometryContainer(table)

print(f"Observations : {phot.n_obs}")
print(f"Detections   : {phot.n_detections}")
print(f"Upper limits : {phot.n_non_detections}")
print(f"Bands        : {np.unique(phot.band_name).tolist()}")

# %%
# Flux and Magnitude Representations
# ------------------------------------
#
# Because we supplied flux columns, the container computes AB magnitudes
# on-the-fly via :math:`m_\mathrm{AB} = -2.5 \log_{10}(F_\nu / F_0)` where
# :math:`F_0 = 3.631 \times 10^{-20}` erg/s/cm²/Hz.

det = phot.detection_mask
print("\nFlux density (first 4 detections, CGS):")
print(phot.flux[det][:4])

print("\nAB magnitude (first 4 detections, computed):")
print(phot.mag[det][:4])

# %%
# Per-Band Light Curves
# ----------------------
#
# The most natural way to visualise multi-band optical data is one light
# curve per band.  We loop over bands, build a detection mask, and plot
# each in AB magnitudes (inverted y-axis).

import matplotlib.pyplot as plt

from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

band_colors = {"g": "#1f77b4", "r": "#d62728", "i": "#8c564b", "z": "#7f7f7f"}

fig, ax = plt.subplots(figsize=(9, 5))

for band in bands:
    band_mask = phot.band_name == band
    det_mask = band_mask & phot.detection_mask
    ul_mask = band_mask & phot.non_detection_mask
    col = band_colors[band]

    ax.errorbar(
        phot.time[det_mask].to_value(u.day),
        phot.mag[det_mask],
        yerr=phot.mag_error[det_mask],
        fmt="o",
        color=col,
        label=band,
    )
    if np.any(ul_mask):
        ax.errorbar(
            phot.time[ul_mask].to_value(u.day),
            phot.mag_upper_limit[ul_mask],
            fmt="^",
            color=col,
            alpha=0.5,
        )

ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"$m_\mathrm{AB}$ (mag)")
ax.set_title("Synthetic multi-band optical photometry")
ax.legend(title="Band")
plt.tight_layout()
plt.show()

# %%
# Epoching
# ---------
#
# The same epoching API available on radio containers works here too: a
# single call to
# :meth:`~trilobite.data.optical_photometry.OpticalPhotometryContainer.set_epochs_from_time_gaps`
# groups all bands together into shared temporal epochs.

phot.set_epochs_from_time_gaps(max_gap=8.0 * u.day)
print(f"\nEpochs identified: {phot.n_epochs}")

epoch_ids = np.unique(phot.epoch_ids)
g_mask = phot.band_name == "g"
print("g-band epoch membership:")
for eid in epoch_ids[:4]:
    t_epoch = phot.time[(phot.epoch_ids == eid) & g_mask]
    if len(t_epoch):
        print(f"  Epoch {eid}: {len(t_epoch)} g-band obs, t ~ {t_epoch.mean():.1f}")
