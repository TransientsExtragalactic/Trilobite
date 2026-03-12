import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from triceratops.radiation.synchrotron import PowerLaw_Cooling_SynchrotronSED

sed = PowerLaw_Cooling_SynchrotronSED()
nu = np.logspace(9, 18, 500) * u.Hz

nu_m = 1e12 * u.Hz
nu_c = 1e15 * u.Hz
nu_max = 1e19 * u.Hz

Fnu = sed.sed(
    nu,
    F_norm=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
    nu_m=nu_m,
    nu_c=nu_c,
    nu_max=nu_max,
    p=2.5,
)

plt.loglog(nu, Fnu)
plt.axvline(nu_m.value, color="C1", ls="--", label=r"$\nu_m$")
plt.axvline(nu_c.value, color="C2", ls="--", label=r"$\nu_c$")
plt.axvline(nu_max.value, color="C3", ls="--", label=r"$\nu_{\max}$")
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
plt.legend()
plt.tight_layout()
plt.show()