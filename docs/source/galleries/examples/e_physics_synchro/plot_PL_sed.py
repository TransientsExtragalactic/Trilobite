r"""
Synchrotron Spectra from Single Electrons and Electron Populations
==================================================================

This example demonstrates how Triceratops computes synchrotron emission
from both individual relativistic electrons and from a population of
electrons following a power-law energy distribution.

We illustrate:

1. The synchrotron spectrum emitted by a **single electron** at the
   minimum and maximum Lorentz factors of a distribution.
2. The **integrated synchrotron spectrum** produced by a power-law
   electron population.
3. How the population-level power-law spectrum emerges from the
   superposition of individual electron spectra.

For numerical efficiency, this example makes use of the interpolated
synchrotron kernel rather than the exact Bessel-function form.
For theoretical background, see :ref:`synchrotron_theory`.

Relevant API references
-----------------------
- :func:`radiation.synchrotron.core.compute_single_electron_power`
- :func:`radiation.synchrotron.core.compute_ME_spectrum_from_dist_function`
- :func:`radiation.synchrotron.core.get_first_kernel_interpolator`
- :func:`radiation.synchrotron.core.compute_nu_critical`
"""

from time import perf_counter

import matplotlib.pyplot as plt

# %%
# Imports
# -------
import numpy as np
from astropy import units as u

from triceratops.radiation.synchrotron.core import (
    compute_ME_spectrum_from_dist_function,
    compute_nu_critical,
    compute_single_electron_power,
    get_first_kernel_interpolator,
)
from triceratops.utils.plot_utils import set_plot_style

# %%
# Physical Setup
# --------------
#
# We define a simple synchrotron-emitting plasma consisting of:
#
# - A uniform magnetic field
# - A power-law electron energy distribution
# - A fixed pitch angle (π/2)

B = 1.0 * u.G
alpha = np.pi / 2

# Electron distribution parameters
p = 2.5
gamma_min = 10.0
gamma_max = 1e5

# %%
# Frequency Grid
# --------------
#
# The frequency range is chosen to comfortably encompass emission from
# both the lowest- and highest-energy electrons in the distribution.

nu = np.geomspace(1e6, 1e18, 500) * u.Hz

# %%
# Interpolated Synchrotron Kernel
# -------------------------------
#
# Evaluating the synchrotron kernel exactly involves modified Bessel
# functions and is computationally expensive. For population integrals,
# Triceratops therefore provides interpolated kernels that dramatically
# improve performance with negligible loss of accuracy.

# Start the timer.
start_time = perf_counter()
kernel = get_first_kernel_interpolator(x_min=1e-5, x_max=1e2)
end_time = perf_counter()

dt_interp = end_time - start_time

print(f"Interpolated kernel setup time: {end_time - start_time:.4f} seconds")
# %%
# Single-Electron Synchrotron Spectra
# -----------------------------------
#
# We first compute the synchrotron spectrum emitted by *individual*
# electrons at the minimum and maximum Lorentz factors of the population.

P_nu_gamma_min = compute_single_electron_power(
    nu=nu,
    gamma=gamma_min,
    B=B,
    alpha=alpha,
    kernel_function=kernel,
)

P_nu_gamma_max = compute_single_electron_power(
    nu=nu,
    gamma=gamma_max,
    B=B,
    alpha=alpha,
    kernel_function=kernel,
)

# Corresponding critical frequencies
nu_c_min = compute_nu_critical(gamma_min, B)
nu_c_max = compute_nu_critical(gamma_max, B)

# %%
# Power-Law Electron Distribution
# -------------------------------
#
# We define a simple power-law electron energy distribution
#
# .. math::
#
#     \frac{dN}{d\gamma} \propto \gamma^{-p}
#
# between ``gamma_min`` and ``gamma_max``.


def N_gamma(gamma):
    return gamma ** (-p)


# %%
# Population Synchrotron Spectrum
# -------------------------------
#
# Integrating the single-electron emissivity over the electron
# distribution yields the total synchrotron spectrum emitted per
# unit volume.
start_time = perf_counter()
P_nu_population = compute_ME_spectrum_from_dist_function(
    nu=nu,
    N_gamma_func=N_gamma,
    gamma_min=gamma_min,
    gamma_max=gamma_max,
    B=B,
    alpha=alpha,
    kernel_function=kernel,
)
end_time = perf_counter()

print(f"Population spectrum computation time: {end_time - start_time:.4f} seconds")
print(
    f"Predicted naive computation time without interpolation on (100,) gamma resolution: {dt_interp * 100:.4f} seconds."
)
print(f"Interpolation speedup factor: {(dt_interp * 100) / (end_time - start_time):.1f}x")

# %%
# Theoretical Optically-Thin Slope
# --------------------------------
#
# For a power-law electron distribution with index ``p``, the optically
# thin synchrotron spectrum obeys
#
# .. math::
#
#     P_\nu \propto \nu^{-(p-1)/2}
#
# away from spectral breaks.

slope = -(p - 1) / 2

# Normalize spectra for shape comparison
Pmin_norm = P_nu_gamma_min / np.nanmax(P_nu_gamma_min)
Pmax_norm = P_nu_gamma_max / np.nanmax(P_nu_gamma_max)
Ppop_norm = P_nu_population / np.nanmax(P_nu_population)

# Choose a frequency range safely within the power-law regime
nu_ref = 1e12 * u.Hz
mask = (nu > 3 * nu_c_min) & (nu < 0.03 * nu_c_max)

# Normalize the guide line at a reference frequency
P_ref = np.interp(
    nu_ref.to_value(u.Hz),
    nu.to_value(u.Hz),
    Ppop_norm.value,
)

guide_line = P_ref * (nu[mask] / nu_ref) ** slope

# %%
# Plotting
# --------
#
# We compare:
#
# - Single-electron emission at ``gamma_min`` and ``gamma_max``
# - The integrated population spectrum
# - The theoretical optically thin synchrotron slope

set_plot_style()

fig, ax = plt.subplots(figsize=(8, 6))

ax.loglog(
    nu.to_value(u.Hz),
    Pmin_norm.value,
    ls="--",
    label=rf"Single Electron ($\gamma_{{\min}}={gamma_min:.0f}$)",
)

ax.loglog(
    nu.to_value(u.Hz),
    Pmax_norm.value,
    ls="--",
    label=rf"Single Electron ($\gamma_{{\max}}={gamma_max:.0e}$)",
)

ax.loglog(
    nu.to_value(u.Hz),
    Ppop_norm.value,
    lw=2,
    label="Electron Population",
)

ax.loglog(
    nu[mask].to_value(u.Hz),
    guide_line,
    color="k",
    lw=2,
    alpha=0.7,
    label=rf"Theory: $P_\nu \propto \nu^{{{slope:.2f}}}$",
)

# Mark critical frequencies
ax.axvline(
    nu_c_min.to_value(u.Hz),
    color="gray",
    ls=":",
    alpha=0.5,
    label=r"$\nu_c(\gamma_{\min})$",
)

ax.axvline(
    nu_c_max.to_value(u.Hz),
    color="gray",
    ls=":",
    alpha=0.5,
    label=r"$\nu_c(\gamma_{\max})$",
)

ax.set_xlabel(r"Frequency $\nu$ [Hz]")
ax.set_ylabel(r"Normalized $P_\nu$")
ax.set_title("Synchrotron Spectra from Electrons and Electron Populations")
ax.set_ylim(1e-10, 1.5)
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Discussion
# ----------
#
# This example illustrates several fundamental aspects of synchrotron
# radiation:
#
# - A **single relativistic electron** emits a broad spectrum peaked near
#   its critical frequency :math:`\nu_c \propto \gamma^2 B`.
# - The **population spectrum** arises from superposing these single-
#   electron spectra, weighted by the electron energy distribution.
# - For a power-law electron population, the synchrotron spectrum
#   approaches a power law with slope :math:`-(p-1)/2` over a wide
#   frequency range, as indicated by the guide line.
#
# This clean separation between microphysical emission kernels and
# population-level integration is a core design principle of the
# Triceratops synchrotron module.
#
# More advanced examples—including cooling breaks, synchrotron
# self-absorption, and time-dependent spectra—can be found in:
#
# - :ref:`synchrotron_microphysics`
# - :ref:`synchrotron_core`
# - :ref:`synchrotron_theory`
