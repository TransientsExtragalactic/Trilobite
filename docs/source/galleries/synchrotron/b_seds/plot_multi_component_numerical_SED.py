"""
Multi-Component Synchrotron SEDs with the Numerical Engine
===========================================================

.. currentmodule:: trilobite.radiation.synchrotron.SEDs.numerical.core

The synchrotron spectra of astrophysical transients are often modelled with a simple
power-law electron distribution, but more complete shock-acceleration models
(e.g. :footcite:t:`2011ApJ...741...40T`) predict a **two-component** population: a
thermal (Maxwell-Jüttner) pool near the shock, and a non-thermal power-law tail
accelerated to high Lorentz factors.

Analytic broken power-law SED approximations are not adequate for this case, because the
transition between the two components is smooth and does not map cleanly onto a single
spectral break. The
:class:`NumericalSynchrotronEngine` provides a
fully numerical alternative: given any sampled electron distribution, it directly
evaluates the synchrotron emissivity integral and solves the radiative transfer equation
along a line-of-sight depth :math:`R`, without any piecewise approximation.

In this example we:

1. Construct a mixed **thermal + power-law** electron distribution.
2. Normalise each component to its share of the thermal energy budget and verify it.
3. Compute the resulting synchrotron SED and compare the contributions from each
   component.

.. hint::

    For a detailed description of the numerical integration algorithm and the synchrotron
    kernel used here, see :ref:`synchrotron_numerical_methods`.

Setup
------

We import the engine and initialise it with the **pitch-angle-averaged** synchrotron
kernel :math:`\\langle F \\rangle(x)`. This is the appropriate choice when the electron
pitch-angle distribution is isotropic.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.integrate import trapezoid

from trilobite.physics_utils import compute_thermal_electron_energy_fraction
from trilobite.radiation.constants import electron_rest_energy_cgs
from trilobite.radiation.synchrotron.electron_distributions import MaxwellJuettner, PowerLaw
from trilobite.radiation.synchrotron.SEDs.numerical import NonRelativisticSphericalSynchrotronEngine
from trilobite.utils.plot_utils import set_plot_style

engine = NonRelativisticSphericalSynchrotronEngine()
engine.load_avg_first_kernel()

# %%
# The Electron Distribution
# --------------------------
#
# The post-shock thermal energy density :math:`u_{\rm th}` is shared between ions, thermal electrons, non-thermal
# electrons and the magnetic field. Trilobite describes the split with three fractions:
#
# - :math:`\epsilon_B`: the fraction of :math:`u_{\rm th}` in the magnetic field,
# - :math:`\epsilon_E`: the fraction in the kinetic energy of the non-thermal electrons,
# - :math:`\epsilon_T`: the ratio of the thermal-electron to the ion energy.
#
# Removing the magnetic and non-thermal energies first, the thermal electrons carry
#
# .. math::
#
#     f_{\rm th} = \frac{\epsilon_T\,(1 - \epsilon_B - \epsilon_E)}{1 + \epsilon_T}
#
# of :math:`u_{\rm th}`. Electron energies are kinetic, :math:`\int N(\gamma-1)\,m_e c^2\,d\gamma`, and each component
# is normalized by equipartition with the magnetic field, so that it carries its fraction of
# :math:`B^2 / (8\pi \epsilon_B)`.
#
# **Thermal component** -- the Maxwell-Juettner distribution,
# :class:`~trilobite.radiation.synchrotron.electron_distributions.MaxwellJuettner`,
#
# .. math::
#
#     N_{\rm therm}(\gamma)
#         = N_{0,\rm therm}\,\frac{\gamma^2\beta}{\theta K_2(1/\theta)}\,e^{-\gamma/\theta},
#     \qquad \beta = \sqrt{1-\gamma^{-2}},
#
# where :math:`\theta = kT/m_e c^2` is the dimensionless electron temperature. For :math:`\theta \gg 1` this peaks near
# :math:`\gamma \approx 2\theta`.
#
# **Non-thermal power-law component** -- a truncated power law,
# :class:`~trilobite.radiation.synchrotron.electron_distributions.PowerLaw`,
#
# .. math::
#
#     N_{\rm PL}(\gamma) = N_{0,\rm PL}\,\gamma^{-p},
#     \qquad \gamma_{\min} \le \gamma \le \gamma_{\max}.

B = 0.1 * u.G  # magnetic field
epsilon_B, epsilon_E, epsilon_T = (
    0.1,
    0.005,
    0.006,
)  # magnetic, non-thermal electron, and thermal-electron/ion fractions
theta = 10  # dimensionless temperature kT / m_e c^2
p = 2.5  # non-thermal power-law index
gamma_min, gamma_max = 50, 1e6  # power-law Lorentz factor bounds

N0_therm = MaxwellJuettner.normalize_from_magnetic_field(
    B, epsilon_B, epsilon_T=epsilon_T, epsilon_E=epsilon_E, Theta=theta
)
N0_pl = PowerLaw.normalize_from_magnetic_field(B, epsilon_B, epsilon_E, p=p, gamma_min=gamma_min, gamma_max=gamma_max)


def N_therm(gamma):
    """Return the Maxwell-Juettner distribution of the thermal electrons."""
    return MaxwellJuettner.pdf(gamma, N0_therm.value, Theta=theta) * u.cm**-3


def N_pl(gamma):
    """Return the truncated power law of the non-thermal electrons."""
    return PowerLaw.pdf(gamma, N0_pl.value, p=p, gamma_min=gamma_min, gamma_max=gamma_max) * u.cm**-3


# %%
# We evaluate both components on a logarithmically-spaced :math:`\gamma` grid that spans non-relativistic through
# ultra-relativistic energies, with a node placed at :math:`\gamma_{\min}` so the sharp edge of the power law is
# sampled exactly. Integrating each component confirms that it carries its fraction of the thermal energy density
# :math:`u_{\rm th} = B^2 / (8\pi \epsilon_B)`.

gamma_array = np.unique(np.append(np.geomspace(1, gamma_max, 400), gamma_min))
u_th = B.value**2 / (8 * np.pi * epsilon_B)  # erg cm^-3
f_th = compute_thermal_electron_energy_fraction(epsilon_T, epsilon_B, epsilon_E)


def kinetic_energy_density(N):
    """Kinetic energy density in erg cm^-3 of a distribution sampled on ``gamma_array``."""
    return electron_rest_energy_cgs * trapezoid(N.to_value(u.cm**-3) * (gamma_array - 1), gamma_array)


N_therm_array = N_therm(gamma_array)
N_pl_array = N_pl(gamma_array)
N_total_array = N_therm_array + N_pl_array

print(f"thermal electrons: {kinetic_energy_density(N_therm_array) / u_th:.5f} of u_th (expected {f_th:.5f})")
print(f"non-thermal electrons: {kinetic_energy_density(N_pl_array) / u_th:.5f} of u_th (expected {epsilon_E:.5f})")

# %%
# Visualising the Distribution
# -----------------------------

set_plot_style()
fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(gamma_array, N_therm_array.to_value(u.cm**-3), color="C3", lw=2, label="Thermal")
ax.loglog(gamma_array, N_pl_array.to_value(u.cm**-3), color="C0", lw=2, label="Power-law")
ax.loglog(
    gamma_array,
    N_total_array.to_value(u.cm**-3),
    color="k",
    lw=1.5,
    ls="--",
    label="Total",
)

ax.axvline(2 * theta, color="C3", ls=":", lw=1, alpha=0.6, label=r"$\gamma_{\rm peak} = 2\theta$")
ax.axvline(gamma_min, color="C0", ls=":", lw=1, alpha=0.6, label=r"$\gamma_{\min}$")

ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$N(\gamma)\ [\mathrm{cm^{-3}}]$")
ax.set_title("Mixed Thermal + Power-Law Electron Distribution")
ax.set_xlim(1, 1e4)
ax.set_ylim(1e-9, 1)
ax.legend(ncol=2)
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.show()

# %%
# The thermal component peaks at :math:`\gamma \approx 2\theta = 20` and falls off
# exponentially above that, while the power-law component switches on at
# :math:`\gamma_{\min} = 50` and follows the :math:`\gamma^{-p}` slope. In the
# transition region :math:`20 \lesssim \gamma \lesssim 100`, both components
# contribute at a comparable level to the total distribution.
#
# Computing the SED
# ------------------
#
# We now pass the sampled distributions to
# :meth:`NumericalSynchrotronEngine.compute_specific_intensity`.
# The engine numerically integrates the synchrotron emissivity
#
# .. math::
#
#     j_\nu = \chi B \int_0^\infty
#             \langle F\rangle\!\left(\frac{\nu}{c_{1\gamma} B \gamma^2}\right)
#             N(\gamma)\,d\gamma
#
# over the supplied :math:`\gamma` grid, then solves the radiative transfer equation
# for a slab of line-of-sight depth :math:`R` to yield the specific intensity
# :math:`I_\nu`.  The ``z`` keyword applies a cosmological redshift to the frequency
# grid before evaluating the emissivity.

nu = np.geomspace(1e-2, 1e2, 100) * u.GHz

source_kwargs = dict(
    R=1e17 * u.cm,
    B=B,
    gamma=gamma_array,
    luminosity_distance=60 * u.Mpc,
)

F_total = engine.compute_flux_density(nu, N=N_total_array, **source_kwargs)
F_therm = engine.compute_flux_density(nu, N=N_therm_array, **source_kwargs)
F_pl = engine.compute_flux_density(nu, N=N_pl_array, **source_kwargs)

# %%
# With the three intensity curves in hand, we overlay them to compare the spectral
# contributions from each component.

set_plot_style()
fig, ax = plt.subplots(figsize=(9, 5.5))

F_unit = "mJy"

ax.loglog(
    nu.to_value(u.GHz),
    F_total.to_value(F_unit),
    color="k",
    lw=2,
    label="Total",
)
ax.loglog(
    nu.to_value(u.GHz),
    F_therm.to_value(F_unit),
    color="C3",
    lw=1.5,
    ls="--",
    label="Thermal",
)
ax.loglog(
    nu.to_value(u.GHz),
    F_pl.to_value(F_unit),
    color="C0",
    lw=1.5,
    ls="--",
    label="Power-law",
)
ax.set_ylim([1e-3, 1e1])
ax.set_xlim([1e-1, 1e2])
ax.set_xlabel(r"$\nu\ [\mathrm{GHz}]$")
ax.set_ylabel(r"$F_\nu\ [\mathrm{mJy}]$")
ax.set_title("Numerical Synchrotron SED: Thermal + Power-Law Distribution")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.show()

# %%
# Several features are visible in the combined spectrum:
#
# - The **thermal component** produces a broad, hump-shaped peak whose frequency is set
#   by the critical synchrotron frequency of electrons near the distribution peak,
#   :math:`\nu_c \propto B\,\gamma_{\rm peak}^2 \propto B\,\theta^2`.
# - The **power-law component** extends the spectrum to much higher frequencies with
#   the familiar optically thin slope :math:`I_\nu \propto \nu^{-(p-1)/2}`.
# - The **total spectrum** smoothly connects the thermal hump to the power-law tail
#   without any artificial spectral breaks or patching — a direct consequence of
#   numerically integrating the full :math:`N(\gamma)`.
#
# .. rubric:: References
#
# .. footbibliography::
