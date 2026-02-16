from triceratops.radiation.synchrotron import PowerLaw_SSA_SynchrotronSED
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

sed = PowerLaw_SSA_SynchrotronSED()

nu = np.logspace(7, 18, 500) * u.Hz

Fnu = sed.sed(
    nu,
    nu_m=1e11 * u.Hz,
    F_peak=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
    omega=4 * np.pi,
    gamma_m=300,
    p=2.5,
)

plt.loglog(nu, Fnu)
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$F_\nu$")
plt.show()