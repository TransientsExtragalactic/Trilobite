r"""
Comparing Free-Free Gaunt Factor Approximations
================================================

The free-free Gaunt factor :math:`g_{\rm ff}(\nu, T)` is a dimensionless
quantum-mechanical correction to the classical Larmor formula for
bremsstrahlung.  Its value is of order unity but depends weakly on both
frequency and temperature — well enough to matter when computing optical
depths or emissivities to better than ~10–20%.

Triceratops provides four approximations, selectable via the ``approx``
keyword of :func:`~triceratops.radiation.free_free.gaunt_factor.compute_ff_gaunt_factor`:

- ``"lu"`` — analytic fit from Lu (2000; *ApJS* 131, 499), the default.
- ``"draine"`` — analytic fit from :footcite:t:`draine2011physics`, accurate
  to a few percent over a wide range.
- ``"vanhoof"`` — bilinear interpolation on the non-relativistic tabulated
  grid of van Hoof et al. (2014; *MNRAS* 444, 420); valid for all
  astrophysically relevant temperatures below :math:`\sim 10^8` K.
- ``"vanhoof_rel"`` — trilinear interpolation on the *relativistic* 3-D
  table of van Hoof et al. (2014); required above :math:`T \gtrsim 10^7` K
  for :math:`Z \leq 36`.

This example compares all four approximations across frequency and temperature.

.. hint::

    For the physical derivation and the definitions of :math:`u = h\nu/k_B T`
    and :math:`\gamma^2 = k_B T / (Z^2\,{\rm Ry})`, see :ref:`free_free_theory`.

Relevant API
------------
- :func:`~triceratops.radiation.free_free.gaunt_factor.compute_ff_gaunt_factor`
- :class:`~triceratops.radiation.free_free.gaunt_factor.NonRelativisticGauntFactorInterpolator`
- :class:`~triceratops.radiation.free_free.gaunt_factor.RelativisticGauntFactorInterpolator`
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from triceratops.radiation.free_free import compute_ff_gaunt_factor
from triceratops.utils.plot_utils import set_plot_style

set_plot_style()

# %%
# Frequency Dependence at Fixed Temperature
# ------------------------------------------
#
# We first compare the three non-relativistic approximations (``"lu"``,
# ``"draine"``, and ``"vanhoof"``) across the radio-to-UV frequency range at
# two representative temperatures: a photoionized HII region
# (:math:`T = 10^4` K) and a hot X-ray emitting plasma (:math:`T = 10^6` K).
#
# The Gaunt factor increases logarithmically with decreasing frequency (or
# increasing :math:`T`) because slow electrons spend more time near an ion and
# therefore have a larger effective cross section.

nu = np.geomspace(1e7, 1e16, 600) * u.Hz

temps = [1e4, 1e6]
temp_labels = [r"$T = 10^4\ \mathrm{K}$", r"$T = 10^6\ \mathrm{K}$"]
temp_colors = ["C0", "C1"]

approx_styles = {
    "lu": ("Lu (2000)", "-", 1.8),
    "draine": ("Draine (2011)", "--", 1.8),
    "vanhoof": ("van Hoof NR (2014)", "-.", 1.8),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

for ax, T_val, label in zip(axes, temps, temp_labels):
    T = T_val * u.K
    for approx, (name, ls, lw) in approx_styles.items():
        g = compute_ff_gaunt_factor(nu, T, Z=1, approx=approx)
        ax.semilogx(nu.to(u.GHz), g, ls=ls, lw=lw, label=name)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_title(label)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=9)

axes[0].set_ylabel(r"$g_{\rm ff}$")
fig.suptitle(r"Gaunt factor vs. frequency at $Z = 1$", fontsize=12)
plt.tight_layout()
plt.show()

# %%
# The three non-relativistic methods agree at the percent level across most
# of the radio band. Differences grow at very high frequencies (near or above
# the Wien peak :math:`h\nu \sim k_B T`) where the exact asymptotic behaviour
# of the Gaunt factor is more sensitive to the approximation.

# %%
# Temperature Dependence at Fixed Frequency
# ------------------------------------------
#
# Next we hold the frequency fixed at three representative radio bands and
# sweep over temperature.  At each frequency the Gaunt factor rises as
# :math:`g_{\rm ff} \propto \ln T` until thermal broadening effects saturate.
# We also include the relativistic ``"vanhoof_rel"`` method at high
# temperatures, where relativistic electron velocities are no longer
# negligible.

T_arr = np.geomspace(1e3, 1e10, 400) * u.K

freq_cases = {
    "1 GHz": 1e9 * u.Hz,
    "10 GHz": 1e10 * u.Hz,
    "100 GHz": 1e11 * u.Hz,
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)

for ax, (freq_label, nu_fixed) in zip(axes, freq_cases.items()):
    for approx, (name, ls, lw) in approx_styles.items():
        g = compute_ff_gaunt_factor(nu_fixed, T_arr, Z=1, approx=approx)
        ax.semilogx(T_arr.to(u.K), g, ls=ls, lw=lw, label=name)

    # Relativistic correction — only valid above ~10^7 K for this method
    T_rel = T_arr[T_arr > 1e7 * u.K]
    g_rel = compute_ff_gaunt_factor(nu_fixed, T_rel, Z=1, approx="vanhoof_rel")
    ax.semilogx(T_rel.to(u.K), g_rel, ls=":", lw=2.2, color="C3", label="van Hoof Rel. (2014)")

    ax.axvline(1e7, color="grey", ls="--", lw=0.9, alpha=0.6)
    ax.text(
        1.2e7, ax.get_ylim()[0] + 0.3 if ax.get_ylim()[0] > 0 else 0.3, r"$10^7\ \mathrm{K}$", fontsize=7, color="grey"
    )

    ax.set_xlabel(r"Temperature (K)")
    ax.set_title(freq_label)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=8)

axes[0].set_ylabel(r"$g_{\rm ff}$")
fig.suptitle(r"Gaunt factor vs. temperature at $Z = 1$", fontsize=12)
plt.tight_layout()
plt.show()

# %%
# The relativistic ``"vanhoof_rel"`` method (dotted, orange) begins to deviate
# from the non-relativistic curves above :math:`T \sim 10^7` K (vertical grey
# dashed line), where electron thermal velocities become a non-negligible
# fraction of *c*. For hot plasmas such as the intracluster medium or
# accreting systems, use ``"vanhoof_rel"`` to avoid underestimating
# :math:`g_{\rm ff}`.

# %%
# Fractional Agreement Between Methods
# --------------------------------------
#
# Finally, we quantify the fractional difference between the analytic ``"lu"``
# and ``"draine"`` approximations at :math:`T = 10^4` K, which is the regime
# most relevant for radio SNe and HII regions.  Differences of order 5–10%
# propagate directly into emissivity and optical depth estimates.

T_ref = 1e4 * u.K

g_lu = compute_ff_gaunt_factor(nu, T_ref, Z=1, approx="lu")
g_draine = compute_ff_gaunt_factor(nu, T_ref, Z=1, approx="draine")
g_vanhoof = compute_ff_gaunt_factor(nu, T_ref, Z=1, approx="vanhoof")

frac_draine = 100 * (g_draine - g_lu) / g_lu
frac_vanhoof = 100 * (g_vanhoof - g_lu) / g_lu

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogx(nu.to(u.GHz), frac_draine, lw=2, color="C1", label="Draine vs. Lu")
ax.semilogx(nu.to(u.GHz), frac_vanhoof, lw=2, color="C2", ls="--", label="van Hoof NR vs. Lu")
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel(r"$(g_{\rm other} - g_{\rm Lu}) / g_{\rm Lu}\ [\%]$")
ax.set_title(
    r"Fractional difference in $g_{\rm ff}$ relative to Lu (2000)"
    "\n"
    r"$T = 10^4\ \mathrm{K},\ Z = 1$"
)
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# The analytic approximations agree to better than ~5% across the GHz
# frequency range at :math:`T = 10^4` K — well within the uncertainty of
# most astrophysical density and temperature estimates.  At very low or
# very high frequencies (outside the classical radio band) the deviations
# can reach ~10–20%.  The tabulated van Hoof interpolation is generally
# the most accurate and should be preferred when precision matters.
