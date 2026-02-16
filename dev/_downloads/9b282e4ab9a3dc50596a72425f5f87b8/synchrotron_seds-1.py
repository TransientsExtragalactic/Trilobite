import numpy as np
from astropy import units as u
from triceratops.radiation.synchrotron.SEDs import PowerLaw_Cooling_SynchrotronSED

sed = PowerLaw_Cooling_SynchrotronSED()
nu = np.logspace(9, 18, 500) * u.Hz

Fnu = sed.sed(
    nu,
    nu_m=1e12 * u.Hz,
    nu_c=1e15 * u.Hz,
    F_peak=1e-26 * u.erg / (u.s * u.cm ** 2 * u.Hz),
    p=2.5,
)

import matplotlib.pyplot as plt

plt.loglog(nu, Fnu)
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$F_\nu$")
plt.tight_layout()
plt.show()