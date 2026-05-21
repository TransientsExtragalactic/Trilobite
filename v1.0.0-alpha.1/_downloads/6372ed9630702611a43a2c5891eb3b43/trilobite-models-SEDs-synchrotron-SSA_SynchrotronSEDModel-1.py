import numpy as np
import matplotlib.pyplot as plt

from trilobite.models.SEDs.synchrotron import (
    SSA_SynchrotronSEDModel
)

rng = np.random.default_rng(42)

model = SSA_SynchrotronSEDModel(pitch_averaged=True)

nu = np.logspace(8, 16, 400)

parameters = {
    "p": 2.5,
    "epsilon_E": 0.1,
    "epsilon_B": 0.1,
    "log_gamma_min": np.log(10.0),
    "log_gamma_max": np.log(1e4),
    "gamma_bulk": 2.0,
    "alpha": np.pi / 2,
    "log_B": np.log(1.0),
    "log_R": np.log(1e16),
    "f_V": 1.0,
    "f_A": 1.0,
    "log_D_L": np.log(1e27),
    "log_D_A": np.log(1e27),
    "redshift": 0.01,
}

# Evaluate model
output = model({"log_nu": np.log(nu)}, parameters)
flux = np.exp(output.flux)

# Compute break frequencies
norm = model._sed._opt_from_physics_to_params(
    log_B=parameters["log_B"],
    log_R=parameters["log_R"],
    log_D_L=parameters["log_D_L"],
    log_D_A=parameters["log_D_A"],
    log_gamma_min=parameters["log_gamma_min"],
    log_gamma_max=parameters["log_gamma_max"],
    p=parameters["p"],
    epsilon_E=parameters["epsilon_E"],
    epsilon_B=parameters["epsilon_B"],
    alpha=parameters["alpha"],
    gamma_bulk=parameters["gamma_bulk"],
    redshift=parameters["redshift"],
    f_V=parameters["f_V"],
    f_A=parameters["f_A"],
    pitch_average=True,
)

# Synthetic data
synthetic = flux + 0.1 * flux * rng.normal(size=flux.size)

plt.figure(figsize=(8,6))
plt.loglog(nu, flux, lw=2, label="Model")
plt.scatter(nu, synthetic, s=10, alpha=0.4)

# Mark breaks
for key, label in [
    ("nu_a", r"$\nu_a$"),
    ("nu_m", r"$\nu_m$"),
    ("nu_max", r"$\nu_{\max}$"),
]:
    plt.axvline(np.exp(norm[key]), ls="--", alpha=0.6)
    plt.text(np.exp(norm[key]), np.amax(flux)*0.3, label, rotation=90)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Flux Density [cgs]")
plt.title("Synchrotron SED (SSA Only)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()