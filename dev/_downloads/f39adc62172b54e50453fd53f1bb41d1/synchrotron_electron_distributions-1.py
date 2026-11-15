import numpy as np
import matplotlib.pyplot as plt
from trilobite.radiation.synchrotron.electron_distributions import (
    PowerLaw, BrokenPowerLaw, MaxwellJuettner, MaxwellJuettnerPowerLaw,
)

# Define a range of Lorentz factors for the plot.
gamma = np.logspace(0, 5, 1000)

# Set the total number of electrons
N_total = 1e6

# Set the parameter dictionaries
params = [
    {'p': 2.5, 'gamma_min': 10.0, 'gamma_max': 1e5},
    {'p1': 2.0, 'p2': 3.5, 'gamma_c': 1e3, 'gamma_min': 10.0, 'gamma_max': 1e5},
    {'Theta': 2.0},
    {'epsilon_T': 0.4, 'epsilon_B': 0.1, 'epsilon_E': 0.2, 'Theta': 2.0, 'p': 2.5, 'gamma_min': 10.0, 'gamma_max': 1e5},
]

# Compute the norms for each distribution.
norms = [
    PowerLaw.normalize_from_n_total(N_total, **params[0]),
    BrokenPowerLaw.normalize_from_n_total(N_total, **params[1]),
    MaxwellJuettner.normalize_from_n_total(N_total, **params[2]),
    MaxwellJuettnerPowerLaw.normalize_from_n_total(N_total, **params[3]),
]
pdfs = [
    PowerLaw.pdf(gamma, **params[0], norm=norms[0].value),
    BrokenPowerLaw.pdf(gamma, **params[1], norm=norms[1].value),
    MaxwellJuettner.pdf(gamma, **params[2], norm=norms[2].value),
    MaxwellJuettnerPowerLaw.pdf(gamma, **params[3], norm=norms[3].value),
]

# Plot the distributions.
plt.figure(figsize=(10, 6))
labels = ['Power Law', 'Broken Power Law', 'Maxwell-Juettner', 'Maxwell-Juettner Power Law']
for pdf, label in zip(pdfs, labels):
    plt.plot(gamma, pdf, label=label)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Lorentz Factor, $\gamma$')
plt.ylabel(r'Distribution Function, $N(\gamma)$')
plt.title('Electron Energy Distributions')
plt.ylim([1e-6,1e6])
plt.legend()
plt.show()