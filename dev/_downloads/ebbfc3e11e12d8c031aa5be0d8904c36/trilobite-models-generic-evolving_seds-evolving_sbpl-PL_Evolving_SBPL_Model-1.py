import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from trilobite.models.SEDs.evolving_sed import PL_Evolving_SSA_SED_Model

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

rng = np.random.default_rng(42)

# --------------------------------------------------
# Instantiate model
# --------------------------------------------------

model = PL_Evolving_SSA_SED_Model()

# --------------------------------------------------
# Frequency grid
# --------------------------------------------------

nu = np.logspace(8, 11, 200) * u.Hz

# Choose several epochs
times = np.geomspace(1, 1000, 5) * u.day

# --------------------------------------------------
# Model parameters
# --------------------------------------------------

parameters = {
    "alpha_1": 5 / 2,  # self-absorbed slope
    "alpha_2": -1.0,  # optically thin slope
    "beta": 1.0,  # nu_brk evolution
    "gamma": 0.0,  # F_brk evolution
    "nu_brk_0": 1e10 * u.Hz,
    "F_brk_0": 1.0 * u.Jy,
    "t_0": 10 * u.day,
    "s": 0.3,
}

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for t in times:
    flux = model(
        {"frequency": nu, "time": t},
        parameters
    ).flux_density

    # Add 10% Gaussian noise
    noise = 0.1 * flux * rng.normal(size=flux.size)
    synthetic = flux + noise

    plt.loglog(nu, flux, lw=2, label=f"t = {t.value:.0f} d")
    plt.scatter(nu, synthetic, s=8, alpha=0.4)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Flux Density [Jy]")
plt.title("Time-Evolving Smoothed Broken Power-Law SED")
plt.legend()
plt.tight_layout()
plt.show()