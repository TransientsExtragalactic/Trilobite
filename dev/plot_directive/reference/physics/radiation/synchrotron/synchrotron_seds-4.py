from triceratops.radiation.synchrotron import PowerLaw_Cooling_SSA_SynchrotronSED
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

sed = PowerLaw_Cooling_SSA_SynchrotronSED()

nu = np.logspace(7, 20, 600) * u.Hz

Fnu = sed.sed(
    nu,
    nu_m=1e12 * u.Hz,
    nu_c=1e15 * u.Hz,
    nu_max=1e19 * u.Hz,
    F_peak=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
    omega=4 * np.pi,
    gamma_m=300,
    p=2.5,
    s=-0.05,
)

plt.loglog(nu, Fnu)
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$F_\nu$")
plt.show()