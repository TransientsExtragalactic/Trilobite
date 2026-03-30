import numpy as np
import matplotlib.pyplot as plt
from triceratops.models.generic.light_curve import GaussianPulse

rng = np.random.default_rng(42)

model = GaussianPulse()

t = np.linspace(-5, 5, 400)
flux = model({"t": t}, {}).flux

noise = 0.05 * np.max(flux) * rng.normal(size=t.size)
synthetic = flux + noise

plt.plot(t, flux, lw=2, label="Model")
plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

plt.xlabel("Time")
plt.ylabel("Flux")
plt.title("Gaussian Pulse Example")
plt.legend()
plt.show()