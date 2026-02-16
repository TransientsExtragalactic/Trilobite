import numpy as np
import matplotlib.pyplot as plt
from triceratops.radiation.synchrotron.core import first_synchrotron_kernel

x = np.logspace(-5, 2, 200)
F_x = first_synchrotron_kernel(x)

plt.plot(x, F_x)
plt.xlim([0,6])
plt.xlabel(r"$x = \nu / \nu_c$")
plt.ylabel(r"$F(x)$")
plt.title("First Synchrotron Kernel")
plt.grid(True)
plt.show()