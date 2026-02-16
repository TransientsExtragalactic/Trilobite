from triceratops.radiation.synchrotron import PowerLaw_Cooling_SynchrotronSED
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

sed = PowerLaw_Cooling_SynchrotronSED()

nu = np.logspace(8, 20, 500) * u.Hz

Fnu = sed.sed(
    nu,
    nu_m=1e12 * u.Hz,
    nu_c=1e15 * u.Hz,
    nu_max=1e19 * u.Hz,
    F_peak=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
    p=2.5,
    s=-0.05,
)

plt.loglog(nu, Fnu)
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$F_\nu$")
plt.show()