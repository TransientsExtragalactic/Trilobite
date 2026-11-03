import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from trilobite.dynamics.profiles.csm import AxisymmetricCSMDensityProfile
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

params = dict(
    rho_int=2e-16, rho_ext=1e-18, r_ref=1e15, alpha_int=1.5, alpha_ext=2.0,
    r_trans=5e15, delta_trans=0.3, f_pole=0.2, aspect=2.0,
)
r = np.geomspace(1e14, 1e18, 500) * u.cm

fig, ax = plt.subplots()
for theta, label in [(np.pi / 2, "equator"), (0.0, "pole")]:
    rho = AxisymmetricCSMDensityProfile.eval(r, theta=theta, **params)
    ax.loglog(r.value, rho.value, label=label)
ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
ax.legend()
plt.tight_layout()