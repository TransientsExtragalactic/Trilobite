import numpy as np
import matplotlib.pyplot as plt
from trilobite.models.generic.light_curve import FRED

rng = np.random.default_rng(42)

model = FRED()

t = np.linspace(-1, 10, 400)
flux = model({"t": t}, {}).flux

noise = 0.05 * np.max(flux) * rng.normal(size=t.size)
synthetic = flux + noise

plt.plot(t, flux, label="Model", lw=2)
plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

plt.xlabel("Time")
plt.ylabel("Flux")
plt.title("FRED Model Example")
plt.legend()
plt.show()