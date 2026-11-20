r"""
Analytic Inverse Closure: Recovering B and R from an SED Peak
==============================================================

**Objective:** use the *analytic* inverse closure of
:class:`~trilobite.radiation.synchrotron.SEDs.one_zone.seds.PowerLaw_SSA_SynchrotronSED` to turn an observed
SED peak into a magnetic field :math:`B` and radius :math:`R`, and check the result against a source of known
properties.

.. note::

    This is the closure of the analytic one-zone SED classes, which are being phased out in favor of the numerical
    engines because the numerical engines are more accurate. To test the closure independently of its own forward
    model, the SED here is generated with the numerical engine, the peak is extracted by a fit, and the extracted
    peak is then inverted. For production analyses use the inversions in
    :mod:`trilobite.radiation.synchrotron.spectral_inversion`, which are tested in the same way in the
    :ref:`spectral inversion examples <sphx_glr_auto_examples_synchrotron_e_inversions>`.

The closure takes the peak flux and frequency, the electron index :math:`p`, the microphysics
:math:`(\epsilon_E, \epsilon_B)` and the filling factors, and returns :math:`B` and :math:`R`. Closure relations
assume a single homogeneous emission region, so the results are effective one-zone quantities.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.optimize import curve_fit

from trilobite.radiation.synchrotron.electron_distributions import PowerLaw
from trilobite.radiation.synchrotron.SEDs import PowerLaw_SSA_SynchrotronSED
from trilobite.radiation.synchrotron.SEDs.numerical import NonRelativisticSphericalSynchrotronEngine
from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

# %%
# Numerical SEDs for a Grid of Sources
# ------------------------------------
#
# We take nine supernova shells at 20 Mpc, with :math:`B` of 0.3, 1 and 3 G and :math:`R` of
# :math:`5\times10^{15}`, :math:`10^{16}` and :math:`3\times10^{16}` cm. The electron normalization is set by
# equipartition, so each :math:`(B, R)` pair gives one SED.

engine = NonRelativisticSphericalSynchrotronEngine()
engine.load_avg_first_kernel()

D, p = 20.0 * u.Mpc, 3.0
micro = dict(epsilon_E=0.1, epsilon_B=0.1, gamma_min=1.0, gamma_max=1e6)
geometry = dict(f_A=0.5, f_V=0.5)

nu = np.geomspace(0.1, 500.0, 60) * u.GHz
gamma = np.geomspace(micro["gamma_min"], micro["gamma_max"], 400)


def numerical_sed(B, R):
    """Numerical flux density of an uncooled shell."""
    N0 = PowerLaw.normalize_from_magnetic_field(B, p=p, **micro)
    N = N0.to_value(u.cm**-3) * PowerLaw.pdf(gamma, p=p, gamma_min=gamma[0], gamma_max=gamma[-1])
    return engine.compute_flux_density(nu, R=R, B=B, N=N, gamma=gamma, luminosity_distance=D, **geometry).to(u.mJy)


# %%
# Extracting the Peak
# -------------------
#
# A smoothly broken power law with a :math:`\nu^{5/2}` thick slope and a free thin slope gives the peak
# frequency and flux as the intersection of its asymptotes.


def sbpl(log_nu, log_F, log_nu_pk, beta, s):
    x = (log_nu - log_nu_pk) * np.log(10)
    return log_F + s * np.logaddexp(2.5 * x / s, beta * x / s) / np.log(10)


def extract_peak(flux):
    """Fit the SED and return ``(nu_pk, F_pk)``."""
    log_nu, log_F = np.log10(nu.to_value(u.Hz)), np.log10(flux.to_value(u.Jy))
    guess = [log_F.max(), log_nu[np.argmax(log_F)], -1.0, -1.0]
    fit, _ = curve_fit(sbpl, log_nu, log_F, p0=guess, bounds=([-20, 7, -4, -10], [5, 12, 0, -0.05]))
    return 10 ** fit[1] * u.Hz, 10 ** fit[0] * u.Jy


# %%
# Applying the Closure
# --------------------
#
# The source is uncooled with :math:`\nu_a > \nu_m`, which is the ``"optically_thick"`` regime of the SED class.

sed = PowerLaw_SSA_SynchrotronSED()

B_true = np.array([0.3, 1.0, 3.0]) * u.G
R_true = np.array([5e15, 1e16, 3e16]) * u.cm

B_in, R_in, B_out, R_out = [], [], [], []
for B in B_true:
    for R in R_true:
        nu_pk, F_pk = extract_peak(numerical_sed(B, R))
        inv = sed.from_params_to_physics(
            "optically_thick",
            F_peak=F_pk,
            nu_peak=nu_pk,
            p=p,
            luminosity_distance=D,
            pitch_average=True,
            **micro,
            **geometry,
        )
        B_in.append(B.value), R_in.append(R.value)
        B_out.append(inv["B"].to_value(u.G)), R_out.append(inv["R"].to_value(u.cm))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
for ax, x, y, label, unit in zip(axes, [B_in, R_in], [B_out, R_out], ["B", "R"], ["G", "cm"]):
    lim = [0.5 * min(x), 2 * max(x)]
    ax.loglog(x, y, "o")
    ax.loglog(lim, lim, "k--", lw=1, label="Recovered = true")
    ax.set(xlabel=rf"True ${label}$ [{unit}]", ylabel=rf"Recovered ${label}$ [{unit}]", title=f"Recovered ${label}$")
axes[0].legend()
plt.show()

# %%
# **Outcome:** the analytic closure does not reproduce the numerical sources. The recovered field is about
# 50% too high and the radius about 50% too low, by nearly the same factor for every source in the grid. The
# offset is a property of the analytic closure rather than of any one source, which is why the numerical
# engines and the inversions in :mod:`~trilobite.radiation.synchrotron.spectral_inversion` are preferred.
