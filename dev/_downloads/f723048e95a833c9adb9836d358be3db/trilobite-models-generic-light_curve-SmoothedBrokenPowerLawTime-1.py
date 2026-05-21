import numpy as np
import matplotlib.pyplot as plt
from trilobite.models.generic.light_curve import SmoothedBrokenPowerLawTime

rng = np.random.default_rng(42)

model = SmoothedBrokenPowerLawTime()

t = np.logspace(-1, 2, 400)
flux = model({"t": t}, {}).flux

noise = 0.1 * flux * rng.normal(size=t.size)
synthetic = flux + noise

plt.loglog(t, flux, lw=2, label="Model")
plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

plt.xlabel("Time")
plt.ylabel("Flux")
plt.title("Smoothed Broken Power Law Time Example")
plt.legend()
plt.show()