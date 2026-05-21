import numpy as np
import matplotlib.pyplot as plt
from trilobite.models.generic.light_curve import GeneralizedFRED

rng = np.random.default_rng(42)

model = GeneralizedFRED()

t = np.linspace(-1, 10, 400)
flux = model({"t": t}, {}).flux

noise = 0.05 * np.max(flux) * rng.normal(size=t.size)
synthetic = flux + noise

plt.plot(t, flux, lw=2, label="Model")
plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

plt.xlabel("Time")
plt.ylabel("Flux")
plt.title("Generalized FRED Example")
plt.legend()
plt.show()