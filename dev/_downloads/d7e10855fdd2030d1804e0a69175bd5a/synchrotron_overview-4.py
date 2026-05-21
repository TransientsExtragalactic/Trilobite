import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from trilobite.radiation.synchrotron.core import compute_nu_critical
from trilobite.radiation.synchrotron.SEDs.numerical import NumericalSynchrotronEngine

B = 1.0 * u.G
gamma_min = 1e2
nu = np.logspace(7, 15, 400) * u.Hz
ps = [2.5, 3.0, 3.5]
colors = ['steelblue', 'darkorange', 'firebrick']

engine = NumericalSynchrotronEngine()
engine.load_avg_first_kernel()

fig, ax = plt.subplots(figsize=(7, 4))

for p, color in zip(ps, colors):
    def N(gamma, p=p):
        return np.where(gamma >= gamma_min, (gamma / gamma_min) ** (-p), 0.0)

    j_nu = engine.compute_emissivity(nu, B=B, N=N, gamma_min=gamma_min, gamma_max=1e8)

    ax.loglog(nu.to(u.GHz).value, j_nu.value, color=color, lw=2,
              label=rf'$p = {p}$')

nu_c_min = compute_nu_critical(gamma=gamma_min, B=B)
ax.axvline(nu_c_min.to(u.GHz).value, ls='--', color='gray', lw=1.2,
           label=rf'$\nu_c(\gamma_{{\min}})$')

ax.set_xlabel('Frequency [GHz]', fontsize=12)
ax.set_ylabel(r'$j_\nu$ [erg s$^{-1}$ cm$^{-3}$ Hz$^{-1}$ sr$^{-1}$]', fontsize=12)
ax.set_title(r'Population-averaged synchrotron emissivity, $B = 1$ G', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()