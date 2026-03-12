import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from triceratops.radiation.synchrotron import PowerLaw_Cooling_SSA_SynchrotronSED

sed = PowerLaw_Cooling_SSA_SynchrotronSED()
nu = np.logspace(8, 18, 500) * u.Hz

# Convert physical parameters to SED parameters
D_L = 100 * u.Mpc
parameters = dict(
    B=0.5 * u.G,
    R=1e16 * u.cm,
    gamma_min=100.0,
    gamma_c=1e4,
    gamma_max=1e7,
    p=2.5,
    f_V=1.0,
    f_A=1.0,
    epsilon_E=0.1,
    epsilon_B=0.1,
    luminosity_distance=D_L,
    pitch_average=True,
)

norm = sed.from_physics_to_params(**parameters)

# Evaluate and plot
Fnu = sed.sed(nu, nu_m=norm['nu_m'],
              nu_c=norm['nu_c'],
              F_norm=norm['F_norm'],
              nu_max=norm['nu_max'],
              p=parameters['p'],
              omega=norm['omega'],
              gamma_m=parameters['gamma_min'])
plt.loglog(nu, Fnu)
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
plt.tight_layout()
plt.show()