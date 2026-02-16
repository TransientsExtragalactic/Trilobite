import numpy as np
import matplotlib.pyplot as plt

from triceratops.models.SEDs.synchrotron import (
    SSA_Cooling_SynchrotronSEDModel
)

rng = np.random.default_rng(42)

model = SSA_Cooling_SynchrotronSEDModel(pitch_averaged=True)

nu = np.logspace(8, 16, 400)

parameters = {
    "p": 2.5,
    "epsilon_E": 0.1,
    "epsilon_B": 0.01,
    "log_gamma_min": np.log(10.0),
    "log_gamma_c": np.log(1e3),
    "log_gamma_max": np.log(1e4),
    "gamma_bulk": 2.0,
    "alpha": np.pi / 2,
    "log_B": np.log(1.0),
    "log_V_eff": np.log(1e55),
    "log_Omega": np.log(1e-10),
    "log_D_L": np.log(1e27),
    "redshift": 0.01,
}

# Evaluate model
output = model({"log_nu": np.log(nu)}, parameters)
flux = np.exp(output.flux)

# Compute break frequencies
norm = model._sed._opt_from_physics_to_params(
    log_B=parameters["log_B"],
    log_V=parameters["log_V_eff"],
    log_D_L=parameters["log_D_L"],
    log_Omega=parameters["log_Omega"],
    log_gamma_min=parameters["log_gamma_min"],
    log_gamma_c=parameters["log_gamma_c"],
    log_gamma_max=parameters["log_gamma_max"],
    p=parameters["p"],
    epsilon_E=parameters["epsilon_E"],
    epsilon_B=parameters["epsilon_B"],
    alpha=parameters["alpha"],
    gamma_bulk=parameters["gamma_bulk"],
    redshift=parameters["redshift"],
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
    ("nu_c", r"$\nu_c$"),
    ("nu_max", r"$\nu_{\max}$"),
]:
    plt.axvline(np.exp(norm[key]), ls="--", alpha=0.6)
    plt.text(np.exp(norm[key]), np.amax(flux)*0.3, label, rotation=90)

plt.xlabel("Frequency [Hz]")
plt.ylabel("Flux Density [cgs]")
plt.title("Synchrotron SED (Cooling + SSA)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()