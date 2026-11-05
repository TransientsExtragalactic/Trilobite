import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from trilobite.radiation.blackbody import planck_luminosity, planck_cutoff_fraction

T, R, cut = 15000 * u.K, 5e15 * u.cm, 3000 * u.AA
lam = np.geomspace(800, 12000, 400) * u.AA

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(lam.value, planck_luminosity(T, R, lam=lam).value, label="blackbody")
ax.plot(lam.value, planck_luminosity(T, R, lam=lam, lambda_cut=cut).value, label="UV-suppressed")
ax.axvline(cut.value, color="k", ls="--", alpha=0.4)
ax.set_xlabel(r"Rest-frame wavelength [$\AA$]")
ax.set_ylabel(r"$L_\lambda$ [erg s$^{-1}$ cm$^{-1}$]")
ax.set_title(f"Surviving fraction f = {planck_cutoff_fraction(T, cut):.2f}")
ax.legend()
fig.tight_layout()