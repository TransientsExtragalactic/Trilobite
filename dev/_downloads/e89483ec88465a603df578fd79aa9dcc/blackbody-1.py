import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from trilobite.radiation.blackbody import planck_B, wien_peak_frequency

nu = np.geomspace(1e11, 1e17, 500) * u.Hz
temperatures = [1e4, 1e5, 1e6] * u.K
labels = [r"$10^4\,\mathrm{K}$", r"$10^5\,\mathrm{K}$", r"$10^6\,\mathrm{K}$"]

fig, ax = plt.subplots(figsize=(7, 4))

for T, label in zip(temperatures, labels):
    B = planck_B(T, nu=nu)
    line, = ax.loglog(nu.value, B.value, label=label)
    nu_peak = wien_peak_frequency(T)
    B_peak = planck_B(T, nu=nu_peak)
    ax.axvline(nu_peak.value, color=line.get_color(), linestyle="--", alpha=0.5)

ax.set_xlabel(r"Frequency $\nu$ [Hz]")
ax.set_ylabel(r"$B_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$]")
ax.set_title("Planck Function")
ax.legend()
fig.tight_layout()