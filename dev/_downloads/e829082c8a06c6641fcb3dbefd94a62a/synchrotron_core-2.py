import numpy as np
import matplotlib.pyplot as plt
from triceratops.radiation.synchrotron.core import get_first_kernel_interpolator

# Create the interpolator
F_interp = get_first_kernel_interpolator(x_min=1e-4, x_max=10.0, num_points=500)

# Evaluate the interpolator
x = np.logspace(-5, 2, 200)
F_x = F_interp(x)

plt.plot(x, F_x, label="Interpolated F(x)")
plt.xlim([0,6])
plt.xlabel(r"$x = \nu / \nu_c$")
plt.ylabel(r"$F(x)$")
plt.title("First Synchrotron Kernel Interpolator")
plt.grid(True)
plt.legend()
plt.show()