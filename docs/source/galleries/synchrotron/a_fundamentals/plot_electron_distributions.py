r"""
Electron Energy Distributions
==============================

Trilobite's synchrotron framework is built on **encapsulated electron distribution
classes** that package the PDF, moments, normalization, and bolometric emissivity
for a given population model.  This example walks through the most common
workflows:

1. Evaluating and plotting the PDF for all five built-in distributions.
2. Computing statistical moments, the mean Lorentz factor, and variance.
3. Setting the amplitude :math:`N_0` from physical closure conditions.
4. Computing the bolometric synchrotron emissivity directly from the distribution.
5. Evaluating and plotting the CDF.

For the underlying theory see :ref:`synch_theory_populations`.

Relevant API References
-----------------------
- :class:`~trilobite.radiation.synchrotron.electron_distributions.PowerLaw`
- :class:`~trilobite.radiation.synchrotron.electron_distributions.BrokenPowerLaw`
- :class:`~trilobite.radiation.synchrotron.electron_distributions.MaxwellJuettner`
- :class:`~trilobite.radiation.synchrotron.electron_distributions.MaxwellJuettnerPowerLaw`
- :class:`~trilobite.radiation.synchrotron.electron_distributions.MaxwellJuettnerBrokenPowerLaw`
- :func:`~trilobite.radiation.synchrotron.electron_distributions.equipartition_magnetic_field`
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from trilobite.radiation.synchrotron import (
    BrokenPowerLaw,
    MaxwellJuettner,
    MaxwellJuettnerBrokenPowerLaw,
    MaxwellJuettnerPowerLaw,
    PowerLaw,
    equipartition_magnetic_field,
)
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

# %%
# The Five Built-in Distributions
# ---------------------------------
#
# All distribution classes are **stateless**: there are no instances to construct.
# Every method is a :obj:`classmethod` that accepts distribution parameters as
# keyword arguments.  The shape function :math:`f(\gamma)` is returned by
# :meth:`~trilobite.radiation.synchrotron.electron_distributions.ElectronDistribution.pdf`
# with ``norm=1`` (the default); multiply by :math:`N_0` to get physical number
# densities.

gamma = np.logspace(0, 6, 2000)

params_pl = {"p": 2.5, "gamma_min": 10.0, "gamma_max": 1e5}
params_bpl = {"p1": 2.0, "p2": 3.5, "gamma_c": 1e3, "gamma_min": 10.0, "gamma_max": 1e5}
params_mj = {"Theta": 2.0}
params_mjpl = {"delta": 0.5, "Theta": 2.0, "p": 2.5, "gamma_min": 10.0, "gamma_max": 1e5}
params_mjbpl = {
    "delta": 0.5,
    "Theta": 2.0,
    "p1": 2.0,
    "p2": 3.5,
    "gamma_c": 1e3,
    "gamma_min": 10.0,
    "gamma_max": 1e5,
}

N_total = 1.0

norms = {
    "pl": PowerLaw.normalize_from_n_total(N_total, **params_pl).value,
    "bpl": BrokenPowerLaw.normalize_from_n_total(N_total, **params_bpl).value,
    "mj": MaxwellJuettner.normalize_from_n_total(N_total, **params_mj).value,
    "mjpl": MaxwellJuettnerPowerLaw.normalize_from_n_total(N_total, **params_mjpl).value,
    "mjbpl": MaxwellJuettnerBrokenPowerLaw.normalize_from_n_total(N_total, **params_mjbpl).value,
}

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(gamma, PowerLaw.pdf(gamma, norms["pl"], **params_pl), label="Power law", lw=2)
ax.loglog(gamma, BrokenPowerLaw.pdf(gamma, norms["bpl"], **params_bpl), label="Broken power law", lw=2, ls="--")
ax.loglog(gamma, MaxwellJuettner.pdf(gamma, norms["mj"], **params_mj), label=r"Maxwell-Jüttner ($\Theta=2$)", lw=2)
ax.loglog(
    gamma,
    MaxwellJuettnerPowerLaw.pdf(gamma, norms["mjpl"], **params_mjpl),
    label=r"MJ + power law ($\delta=0.5$)",
    lw=2,
    ls=":",
)
ax.loglog(
    gamma,
    MaxwellJuettnerBrokenPowerLaw.pdf(gamma, norms["mjbpl"], **params_mjbpl),
    label=r"MJ + broken power law ($\delta=0.5$)",
    lw=2,
    ls="-.",
)
ax.set_xlim(1, 1e6)
ax.set_ylim(1e-8, 10)
ax.set_xlabel(r"Lorentz factor $\gamma$")
ax.set_ylabel(r"$N(\gamma)$ [normalized to $N_{\rm tot}=1$]")
ax.set_title("Electron energy distributions")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# %%
# Statistical Moments and Derived Quantities
# -------------------------------------------
#
# Every distribution exposes :meth:`~ElectronDistribution.moment`,
# :meth:`~ElectronDistribution.mean`, :meth:`~ElectronDistribution.var`, and
# :meth:`~ElectronDistribution.mean_energy`.  For :class:`PowerLaw` and
# :class:`BrokenPowerLaw` these are exact analytic integrals.  For
# :class:`MaxwellJuettner`, orders 0, 1, and 2 are closed-form Bessel
# expressions; higher orders fall back to numerical quadrature.

print("=== Statistical moments (shape function, norm=1) ===")
print(
    f"PowerLaw         M0={PowerLaw.moment(0, **params_pl):.4f}  "
    f"<gamma>={PowerLaw.mean(**params_pl):.2f}  "
    f"std={PowerLaw.std(**params_pl):.2f}"
)
print(
    f"BrokenPowerLaw   M0={BrokenPowerLaw.moment(0, **params_bpl):.4f}  "
    f"<gamma>={BrokenPowerLaw.mean(**params_bpl):.2f}  "
    f"std={BrokenPowerLaw.std(**params_bpl):.2f}"
)
print(
    f"MaxwellJuettner  M0={MaxwellJuettner.moment(0, **params_mj):.4f}  "
    f"<gamma>={MaxwellJuettner.mean(**params_mj):.2f}  "
    f"std={MaxwellJuettner.std(**params_mj):.2f}"
)

# %%
# Normalization from Physical Parameters
# ----------------------------------------
#
# Four normalization routes are available.  The most common in the synchrotron
# literature derives :math:`N_0` from a magnetic field strength via the
# equipartition fractions :math:`\varepsilon_e` and :math:`\varepsilon_B`.

B = 0.5 * u.G
eps_B = 0.1
eps_E = 0.1

N0_pl = PowerLaw.normalize_from_magnetic_field(B, eps_B, eps_E, **params_pl)
N0_mj = MaxwellJuettner.normalize_from_magnetic_field(B, eps_B, eps_E, **params_mj)

print(f"\n=== Normalization from B = {B}, eps_B = {eps_B}, eps_E = {eps_E} ===")
print(f"PowerLaw         N0 = {N0_pl:.3e}")
print(f"MaxwellJuettner  N0 = {N0_mj:.3e}")

# You can also normalize from the thermal energy density directly:
u_therm = 1e5 * u.erg / u.cm**3
N0_from_u = PowerLaw.normalize_from_energy_density(u_therm, eps_E, **params_pl)
print(f"PowerLaw from u_therm = {u_therm:.1e}:  N0 = {N0_from_u:.3e}")

# %%
# Bolometric Synchrotron Emissivity
# -----------------------------------
#
# Given equipartition parameters, the bolometric emissivity
#
# .. math::
#
#     j = \frac{4}{3}\,\sigma_T c\, U_B\, N_0\, M_2
#
# can be computed in a single call.  The result is in
# :math:`\mathrm{erg\,s^{-1}\,cm^{-3}}`.

p_vals = np.linspace(2.1, 4.5, 40)
j_pl = np.array(
    [PowerLaw.bol_emiss_from_magnetic_field(B, eps_B, eps_E, p=p, gamma_min=10.0, gamma_max=1e5).value for p in p_vals]
)

Theta_vals = np.logspace(-1, 1.5, 40)
j_mj = np.array([MaxwellJuettner.bol_emiss_from_magnetic_field(B, eps_B, eps_E, Theta=T).value for T in Theta_vals])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].semilogy(p_vals, j_pl, lw=2, color="C0")
axes[0].set_xlabel(r"Power-law index $p$")
axes[0].set_ylabel(r"Emissivity [erg s$^{-1}$ cm$^{-3}$]")
axes[0].set_title("Power-law electron distribution")
axes[0].grid(True, which="both", ls="--", alpha=0.4)

axes[1].loglog(Theta_vals, j_mj, lw=2, color="C1")
axes[1].set_xlabel(r"Dimensionless temperature $\Theta = k_BT / m_e c^2$")
axes[1].set_title(r"Maxwell-Jüttner electron distribution")
axes[1].grid(True, which="both", ls="--", alpha=0.4)

fig.suptitle(r"Bolometric synchrotron emissivity ($B=0.5\,\mathrm{G}$, $\varepsilon_e=\varepsilon_B=0.1$)")
plt.tight_layout()
plt.show()

# %%
# Cumulative Distribution Functions
# -----------------------------------
#
# :meth:`~ElectronDistribution.cdf` integrates the PDF from the lower support
# bound up to :math:`\gamma`.  For :class:`PowerLaw` and
# :class:`BrokenPowerLaw` this is analytic; for :class:`MaxwellJuettner` it
# uses vectorized trapezoidal integration on a log-spaced grid.

gamma_cdf = np.logspace(0, 5, 500)

fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(
    gamma_cdf, PowerLaw.cdf(gamma_cdf, 1.0, **params_pl) / PowerLaw.n_total(1.0, **params_pl), label="Power law", lw=2
)
ax.semilogx(
    gamma_cdf,
    BrokenPowerLaw.cdf(gamma_cdf, 1.0, **params_bpl) / BrokenPowerLaw.n_total(1.0, **params_bpl),
    label="Broken power law",
    lw=2,
    ls="--",
)
ax.semilogx(gamma_cdf, MaxwellJuettner.cdf(gamma_cdf, 1.0, **params_mj), label=r"Maxwell-Jüttner ($\Theta=2$)", lw=2)
ax.set_xlabel(r"Lorentz factor $\gamma$")
ax.set_ylabel(r"$F(\gamma) / N_{\rm tot}$")
ax.set_title("Normalized CDFs")
ax.legend()
ax.grid(True, ls="--", alpha=0.4)
plt.tight_layout()
plt.show()
