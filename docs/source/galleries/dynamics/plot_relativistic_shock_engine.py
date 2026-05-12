"""
Relativistic Thin-Shell Shock in a Stellar Wind
================================================

This example demonstrates how to use the
:class:`~triceratops.dynamics.shocks.numerical.RelativisticNumericalThinShellShockEngine`
to model the mildly-relativistic evolution of a supernova shock expanding into a
stellar-wind circumstellar medium (CSM).

Unlike the non-relativistic thin-shell engine, this solver tracks the lab-frame
**energy** :math:`E_{\rm sh}` and **radial momentum** :math:`p_{\rm sh}` of the shell
alongside the baryonic rest mass :math:`M_s`. This makes it suitable for mildly to
ultra-relativistic shocks (e.g., GRBs, hypernovae, engine-driven transients) where
:math:`\\beta_{\\rm sh} = p_{\\rm sh}c / E_{\\rm sh}` may approach unity.

We first validate the engine against an **exact analytic solution** (a
momentum-conserving blast wave in a wind), then apply it to a full Chevalier-like
interaction with ejecta.
"""

# %%
# Setup
# -----

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from triceratops.dynamics.shocks import (
    RelativisticNumericalThinShellShockEngine,
    get_broken_power_law_ejecta_kernel_func,
    get_wind_csm_density_func,
)
from triceratops.radiation.constants import c_cgs
from triceratops.utils.plot_utils import set_plot_style

# %%
# Shared Wind CSM
# ---------------
#
# Both the validation and main example use the same wind CSM characterised by
# :math:`\rho_{\rm csm}(r) = A/r^2` with
# :math:`A = \dot{M}/(4\pi v_w)`.
# The relativistic engine requires callables :math:`\rho(r,t)`, so we wrap the
# (radius-only) helper to accept and ignore the time argument.

mass_loss_rate = 1e-3 * u.Msun / u.yr
wind_velocity = 100 * u.km / u.s

_rho_csm_1d = get_wind_csm_density_func(
    mass_loss_rate=mass_loss_rate,
    wind_velocity=wind_velocity,
)

rho_csm = lambda r, t: _rho_csm_1d(r)
v_csm = lambda r, t: 0.0
U_int_csm = lambda r, t: 0.0

# A-coefficient for the analytic solution (g/cm).
A_wind = mass_loss_rate.to(u.g / u.s).value / (4.0 * np.pi * wind_velocity.to(u.cm / u.s).value)

# %%
# Validation: Momentum-Conserving Blast Wave
# ------------------------------------------
#
# With no ejecta and a cold static CSM (:math:`P = 0`,
# :math:`\beta_{\rm csm} = 0`), the momentum equation gives
# :math:`dp_{\rm sh}/dt = 0`, so :math:`p_{\rm sh} = p_0 = \mathrm{const}`.
# The energy sweep-up rate :math:`dE/dt = 4\pi A c^4 p_0 / E`
# then integrates exactly to
#
# .. math::
#
#     E(t) = \sqrt{E_0^2 + 8\pi A c^4 p_0\,(t - t_0)},
#     \quad
#     \beta(t) = \frac{p_0 c}{E(t)},
#     \quad
#     M(t) = M_0 + \frac{E(t) - E_0}{c^2}.
#
# We compare this exact solution against the numerical integration with
# :math:`\rho_{\rm ej} = 0`.

beta_val = 0.5
gamma_val = 1.0 / np.sqrt(1.0 - beta_val**2)

t_val_0 = 1.0 * u.day
t_val_0_s = t_val_0.to(u.s).value

M_val_0 = 0.1 * u.Msun
M_val_0_g = M_val_0.to(u.g).value
E_val_0 = gamma_val * M_val_0_g * c_cgs**2  # erg
p_val_0 = gamma_val * beta_val * M_val_0_g * c_cgs  # g cm/s
R_val_0 = beta_val * c_cgs * t_val_0_s  # cm

time_val = np.geomspace(1, 200, 400) * u.day
time_val_s = time_val.to(u.s).value

# Analytic solution.
Dt = time_val_s - t_val_0_s
E_anal = np.sqrt(E_val_0**2 + 8.0 * np.pi * A_wind * c_cgs**4 * p_val_0 * Dt)
beta_anal = p_val_0 * c_cgs / E_anal
M_anal = M_val_0_g + (E_anal - E_val_0) / c_cgs**2

# Numerical solution (no ejecta).
engine = RelativisticNumericalThinShellShockEngine()

result_val = engine.compute_shock_properties(
    time_val,
    rho_ej=lambda r, t: 0.0,
    v_ej=lambda r, t: 0.0,
    U_int_ej=lambda r, t: 0.0,
    rho_csm=rho_csm,
    v_csm=v_csm,
    U_int_csm=U_int_csm,
    R_0=R_val_0 * u.cm,
    E_0=E_val_0 * u.erg,
    p_0=p_val_0 * (u.g * u.cm / u.s),
    M_0=M_val_0,
    t_0=t_val_0,
)

# %%
# Validation Plots
# ----------------

set_plot_style()

fig_val, axes_val = plt.subplots(1, 2, figsize=(10, 4))
t_days_val = time_val.to_value(u.day)

axes_val[0].semilogx(t_days_val, beta_anal, lw=2, label="Analytic")
axes_val[0].semilogx(t_days_val, result_val["beta"].value, ls="--", lw=1.5, label="Numerical")
axes_val[0].set_xlabel("Time (days)")
axes_val[0].set_ylabel(r"$\beta_{\rm sh}$")
axes_val[0].set_title("Shell velocity")
axes_val[0].legend()

axes_val[1].loglog(t_days_val, M_anal / 2e33, lw=2, label="Analytic")
axes_val[1].loglog(t_days_val, result_val["mass"].to_value(u.Msun), ls="--", lw=1.5, label="Numerical")
axes_val[1].set_xlabel("Time (days)")
axes_val[1].set_ylabel(r"$M_s\;(M_\odot)$")
axes_val[1].set_title("Swept-up mass")
axes_val[1].legend()

for ax in axes_val:
    ax.grid(True, which="both", ls="--", alpha=0.4)

fig_val.suptitle("Validation: momentum-conserving blast wave (no ejecta)", y=1.01)
fig_val.tight_layout()
plt.show()

# %%
# Main Example: Chevalier-Like Interaction
# ----------------------------------------
#
# We now include ejecta with a broken-power-law velocity profile
# (:math:`n=10`, :math:`\delta=0`, :math:`v_t \approx 0.22\,c`).
# Starting at :math:`\beta_0 = 0.30` (outer power-law regime), the swept
# CSM mass reaches :math:`M_{\rm ej}` by :math:`\sim 10^4` days, producing
# clear deceleration across the integration.

E_ej = 1e51 * u.erg
M_ej = 0.3 * u.Msun

_G_ej = get_broken_power_law_ejecta_kernel_func(E_ej, M_ej, n=10, delta=0)


def rho_ej(r, t):
    """Ejecta rest-mass density from homologous kernel."""
    return float(np.squeeze(_G_ej(np.atleast_1d(r / t)))) / t**3


v_ej = lambda r, t: r / t
U_int_ej = lambda r, t: 0.0

beta_0 = 0.30
gamma_0 = 1.0 / np.sqrt(1.0 - beta_0**2)

t_0 = 1.0 * u.day
R_0 = beta_0 * c_cgs * t_0.to(u.s).value * u.cm
M_0 = 1e-4 * u.Msun
M_0_g = M_0.to(u.g).value
E_0 = gamma_0 * M_0_g * c_cgs**2 * u.erg
p_0 = gamma_0 * beta_0 * M_0_g * c_cgs * (u.g * u.cm / u.s)

time = np.geomspace(1, 1e4, 300) * u.day

result = engine.compute_shock_properties(
    time,
    rho_ej=rho_ej,
    v_ej=v_ej,
    U_int_ej=U_int_ej,
    rho_csm=rho_csm,
    v_csm=v_csm,
    U_int_csm=U_int_csm,
    R_0=R_0,
    E_0=E_0,
    p_0=p_0,
    M_0=M_0,
    t_0=t_0,
)

R_free = beta_0 * c_cgs * time.to(u.s).value * u.cm

# %%
# Chevalier Interaction Plots
# ---------------------------

fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
t_days = time.to_value(u.day)

axes[0].loglog(t_days, result["radius"].to_value(u.cm), label="Relativistic thin-shell")
axes[0].loglog(t_days, R_free.to_value(u.cm), ls="--", label=r"Free expansion ($\beta_0 c$)")
axes[0].set_ylabel(r"$R_{\rm sh}$ (cm)")
axes[0].legend()

axes[1].semilogx(t_days, result["beta"].value)
axes[1].axhline(beta_0, ls="--", color="C1", label=rf"$\beta_0 = {beta_0}$")
axes[1].set_ylabel(r"$\beta_{\rm sh} = v_{\rm sh}/c$")
axes[1].legend()

axes[2].semilogx(t_days, result["lorentz_factor"].value)
axes[2].set_ylabel(r"$\Gamma_{\rm sh}$")

axes[3].loglog(t_days, result["mass"].to_value(u.Msun))
axes[3].axhline(M_ej.to_value(u.Msun), ls="--", color="C1", label=r"$M_{\rm ej}$")
axes[3].set_ylabel(r"$M_s\;(M_\odot)$")
axes[3].set_xlabel("Time (days)")
axes[3].legend()

for ax in axes:
    ax.grid(True, which="both", ls="--", alpha=0.4)

fig.tight_layout()
plt.show()
