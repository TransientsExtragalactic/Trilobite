import numpy as np
import matplotlib.pyplot as plt
from triceratops.radiation.synchrotron.core import (
    compute_first_synchrotron_kernel,
    compute_averaged_first_synchrotron_kernel,
)

x = np.logspace(-3, 1, 300)
F     = compute_first_synchrotron_kernel(x)
F_avg = compute_averaged_first_synchrotron_kernel(x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(x, F,     lw=2, label=r'$F(x)$')
ax.loglog(x, F_avg, lw=2, label=r'$\bar{F}(x)$', ls='--')
ax.set_xlabel(r'$x = \nu / \nu_c$', fontsize=12)
ax.set_ylabel(r'Kernel value', fontsize=12)
ax.set_title('Synchrotron kernel functions', fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()