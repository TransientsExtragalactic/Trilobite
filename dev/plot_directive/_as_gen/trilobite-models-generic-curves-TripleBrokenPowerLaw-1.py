import numpy as np
import matplotlib.pyplot as plt
from trilobite.models.generic.bpl import TripleBrokenPowerLaw

model = TripleBrokenPowerLaw()

x = np.logspace(-2, 3, 500)
y = model({"x": x}, {}).y

plt.loglog(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Triple Broken Power Law Example")
plt.show()