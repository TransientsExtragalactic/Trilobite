import numpy as np
import matplotlib.pyplot as plt
from triceratops.models.generic.bpl import SmoothedTripleBrokenPowerLaw

model = SmoothedTripleBrokenPowerLaw()

x = np.logspace(-2, 3, 500)
y = model({"x": x}, {}).y

plt.loglog(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Smoothed Triple Broken Power Law Example")
plt.show()