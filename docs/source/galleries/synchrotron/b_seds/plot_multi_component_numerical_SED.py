"""
Multi-Component Synchrotron SEDs with the Numerical Engine
===========================================================

The synchrotron spectra of astrophysical transients are often modelled with a simple
power-law electron distribution, but more complete shock-acceleration models
(e.g. :footcite:t:`2011ApJ...741...40T`) predict a **two-component** population: a
thermal (Maxwell-Jüttner) pool near the shock, and a non-thermal power-law tail
accelerated to high Lorentz factors.

Analytic broken power-law SED approximations are not adequate for this case, because the
transition between the two components is smooth and does not map cleanly onto a single
spectral break. The
:class:`~trilobite.radiation.synchrotron.SEDs.numerical.NumericalSynchrotronEngine` provides a
fully numerical alternative: given any sampled electron distribution, it directly
evaluates the synchrotron emissivity integral and solves the radiative transfer equation
along a line-of-sight depth :math:`R`, without any piecewise approximation.

In this example we:

1. Construct a mixed **thermal + power-law** electron distribution.
2. Verify the normalisation by checking that the components integrate to their target
   number densities.
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

from trilobite.radiation.synchrotron.SEDs.numerical import NonRelativisticSphericalSynchrotronEngine
from trilobite.utils.plot_utils import set_plot_style

engine = NonRelativisticSphericalSynchrotronEngine()
engine.load_avg_first_kernel()

# %%
# The Electron Distribution
# --------------------------
#
# We split the total electron number density :math:`n_e` between two components
# controlled by the **thermal fraction** :math:`\delta`:
#
# .. math::
#
#     n_e = \underbrace{\delta\, n_e}_{n_{\rm therm}}
#           \;+\;
#           \underbrace{(1-\delta)\, n_e}_{n_{\rm PL}}.
#
# **Thermal component** — relativistic Maxwell-Jüttner distribution:
#
# .. math::
#
#     N_{\rm therm}(\gamma)
#         = \frac{n_{\rm therm}}{2\theta^3}\,\gamma^2\,e^{-\gamma/\theta},
#
# where :math:`\theta = kT/m_e c^2` is the dimensionless temperature.  The prefactor
# :math:`1/(2\theta^3)` follows from
# :math:`\int_0^\infty \gamma^2 e^{-\gamma/\theta}\,d\gamma = 2\theta^3`, valid in
# the ultra-relativistic limit :math:`\theta \gg 1`. The distribution peaks at
# :math:`\gamma_{\rm peak} \approx 2\theta`.
#
# **Non-thermal power-law component** — truncated power law with index :math:`p`:
#
# .. math::
#
#     N_{\rm PL}(\gamma)
#         = \frac{(p-1)\,n_{\rm PL}}
#               {\gamma_{\min}^{1-p} - \gamma_{\max}^{1-p}}
#               \,\gamma^{-p},
#         \qquad \gamma_{\min} \le \gamma \le \gamma_{\max}.
#
# The prefactor is the reciprocal of
# :math:`\int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^{-p}\,d\gamma`, ensuring that
# :math:`\int_{\gamma_{\min}}^{\gamma_{\max}} N_{\rm PL}(\gamma)\,d\gamma = n_{\rm PL}`.

delta = 0.999  # thermal fraction of the total electron population
n_e = 1.0 * u.cm**-3  # total electron number density
p = 2.5  # non-thermal power-law index
theta = 1  # dimensionless temperature kT / m_e c^2
gamma_min, gamma_max = 6, 1e8  # power-law Lorentz factor bounds

n_therm = delta * n_e
n_pl = (1 - delta) * n_e

_pl_norm = (p - 1) / (gamma_min ** (1 - p) - gamma_max ** (1 - p))


def N_therm(gamma):
    """Maxwell-Jüttner distribution, normalised to n_therm."""
    return n_therm * gamma**2 * np.exp(-gamma / theta) / (2 * theta**3)


def N_pl(gamma):
    """Truncated power-law distribution, normalised to n_pl."""
    N = n_pl * _pl_norm * gamma ** (-p)
    N[gamma < gamma_min] = 0.0 * u.cm**-3
    return N


# %%
# Visualising the Distribution
# -----------------------------
#
# We evaluate both components on a logarithmically-spaced :math:`\gamma` grid that
# spans from non-relativistic through ultra-relativistic energies.

gamma_array = np.geomspace(1, 1e6, 200)

N_therm_array = N_therm(gamma_array)
N_pl_array = N_pl(gamma_array)
N_total_array = N_therm_array + N_pl_array

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
# :meth:`~trilobite.radiation.synchrotron.SEDs.numerical.NumericalSynchrotronEngine.compute_specific_intensity`.
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
    B=1 * u.G,
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
ax.set_ylim([1e-3, 1e0])
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
