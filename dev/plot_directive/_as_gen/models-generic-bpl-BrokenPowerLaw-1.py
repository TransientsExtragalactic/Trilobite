import numpy as np
import matplotlib.pyplot as plt
from triceratops.models.generic.bpl import BrokenPowerLaw

model = BrokenPowerLaw()

x = np.logspace(-2, 2, 500)
y = model({"x": x}, {}).y

plt.loglog(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Broken Power Law Example")
plt.show()