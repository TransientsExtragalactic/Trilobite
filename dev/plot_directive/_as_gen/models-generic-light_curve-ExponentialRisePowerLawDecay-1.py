import numpy as np
import matplotlib.pyplot as plt
from triceratops.models.generic.light_curve import ExponentialRisePowerLawDecay

rng = np.random.default_rng(42)

model = ExponentialRisePowerLawDecay()

t = np.logspace(-2, 2, 400)
flux = model({"t": t}, {}).flux

noise = 0.1 * flux * rng.normal(size=t.size)
synthetic = flux + noise

plt.loglog(t, flux, lw=2, label="Model")
plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

plt.xlabel("Time")
plt.ylabel("Flux")
plt.title("Exponential Rise Power-Law Decay Example")
plt.legend()
plt.show()