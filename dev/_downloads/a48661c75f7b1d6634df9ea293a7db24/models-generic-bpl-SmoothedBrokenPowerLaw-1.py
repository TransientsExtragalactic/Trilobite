import numpy as np
import matplotlib.pyplot as plt
from triceratops.models.generic.bpl import SmoothedBrokenPowerLaw

model = SmoothedBrokenPowerLaw()

x = np.logspace(-2, 2, 500)
y = model({"x": x}, {}).y

plt.loglog(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Smoothed Broken Power Law Example")
plt.show()