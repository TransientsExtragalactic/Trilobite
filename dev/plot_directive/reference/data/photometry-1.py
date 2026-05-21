import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
from triceratops.data import RadioPhotometryContainer

rng = np.random.default_rng(0)
n = 24
time = np.sort(rng.uniform(1, 500, n))
freqs = np.tile([5.5, 8.5, 15.0], 8)
flux = 5e-26 * (time / 10.0) ** -0.8 * rng.lognormal(0, 0.08, n)
err = flux * 0.1

flux_arr = np.where(time > 400, np.nan, flux)
err_arr = np.where(time > 400, np.nan, err)
ul = np.where(time > 400, 5e-28, np.nan)

t = Table({
    "time": time * u.day,
    "freq": freqs * u.GHz,
    "flux_density": flux_arr * u.Jy,
    "flux_density_error": err_arr * u.Jy,
    "flux_upper_limit": ul * u.Jy,
})
c = RadioPhotometryContainer(t)

colors = {5.5: "steelblue", 8.5: "tomato", 15.0: "seagreen"}
fig, ax = plt.subplots(figsize=(7, 4))
for nu in [5.5, 8.5, 15.0]:
    mask = (c.freq.value == nu) & c.detection_mask
    ax.errorbar(c.time[mask].value, c.flux_density[mask].value,
                yerr=c.flux_density_error[mask].value,
                fmt="o", label=f"{nu} GHz", color=colors[nu])
    ul_mask = (c.freq.value == nu) & c.non_detection_mask
    if ul_mask.any():
        ax.errorbar(c.time[ul_mask].value, c.flux_upper_limit[ul_mask].value,
                    fmt="v", color=colors[nu], alpha=0.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Time (days)"); ax.set_ylabel("Flux density (Jy)")
ax.set_title("Synthetic multi-frequency radio photometry")
ax.legend()
plt.tight_layout()