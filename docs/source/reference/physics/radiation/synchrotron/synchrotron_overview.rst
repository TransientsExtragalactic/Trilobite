.. _synchrotron_overview:

==============================
Synchrotron Emission Overview
==============================

Synchrotron radiation, produced by relativistic electrons spiraling in a magnetic
field, is the dominant emission process in the radio and X-ray afterglows of many
astrophysical transients.  It governs the multi-wavelength light curves of
radio supernovae :footcite:p:`demarchiRadioAnalysisSN2004C2022`, gamma-ray burst
afterglows :footcite:p:`1998ApJ...497L..17S`, and the delayed radio flares of
tidal disruption events :footcite:p:`wuDelayedRadioEmission2025`.  Interpreting
these observations requires a modeling framework that can handle a wide range of
physical regimes, from non-relativistic shocks to mildly-relativistic ejecta,
while consistently tracking the microphysical processes that shape the spectrum.

Triceratops provides an end-to-end synchrotron modeling toolkit: from the fundamental
single-electron physics through population-averaged emission and radiative cooling,
up to the broadband one-zone spectral energy distributions (SEDs) used directly in
multi-wavelength fitting and Bayesian inference.  The infrastructure is intentionally
layered — each level can be used independently or composed into a full modeling
pipeline.

.. hint::

    For the underlying physics, see :ref:`synchrotron_theory`.  That document covers
    the derivation of the synchrotron kernel functions, the theory of power-law
    electron populations, equipartition arguments, and radiative cooling, with
    references to the primary literature.

.. contents::
    :local:
    :depth: 2

----

Module Overview
---------------

The synchrotron module is organized into four submodules, each responsible for a
distinct layer of the modeling stack.  Users working with high-level SED models
typically only interact directly with :mod:`~triceratops.radiation.synchrotron.SEDs`
and :mod:`~triceratops.radiation.synchrotron.microphysics`; the lower layers are
used implicitly.

.. list-table::
    :header-rows: 1
    :widths: 32 68

    * - Module
      - Responsibility
    * - :mod:`~triceratops.radiation.synchrotron.core`
      - Fundamental quantities: gyrofrequency, critical frequency, synchrotron kernel
        functions :math:`F(x)` and :math:`G(x)`, single-electron spectra, and
        integrals over arbitrary electron distributions.
        See :ref:`synchrotron_core`.
    * - :mod:`~triceratops.radiation.synchrotron.microphysics`
      - Electron distribution utilities (power-law and broken power-law moments,
        normalization conversion) and equipartition closure relations that map
        shock energy densities to microphysical parameters.
        See :ref:`synchrotron_microphysics`.
    * - :mod:`~triceratops.radiation.synchrotron.SEDs`
      - One-zone synchrotron SED models covering all combinations of self-absorption
        (SSA) and radiative cooling.  Each class provides a
        ``from_physics_to_params`` method to convert physical inputs to
        phenomenological SED parameters.
        See :ref:`synchrotron_seds`.
    * - :mod:`~triceratops.radiation.synchrotron.cooling`
      - Radiative cooling engines for synchrotron and inverse-Compton losses.
        Each engine exposes the cooling rate :math:`dE/dt`, cooling time
        :math:`t_{\rm cool}`, and the cooling Lorentz factor :math:`\gamma_c`.
        See :ref:`synchrotron_cooling`.

----

Core Synchrotron Quantities
---------------------------

.. hint::

    :ref:`synchrotron_core` contains the full API documentation and worked examples
    for all functions described in this section, including kernel interpolation and
    the integrated spectrum API.

At the core of synchrotron modeling are a number of fundamental quantities that describe various features
of the emission. This includes the the **gyrofrequency**
:math:`\nu_g = eB / (m_e c \gamma)` and the **critical synchrotron frequency**
:math:`\nu_c = 3 e B \gamma^2 / (4\pi m_e c)`.  Nearly all synchrotron power from
that electron is emitted within roughly an order of magnitude of :math:`\nu_c`.

Tools for computing these frequencies are provided in :mod:`~triceratops.radiation.synchrotron.core`, along with the
significant integrals over Bessel functions that arise in the single-electron spectrum and the population-averaged emissivity.
These integrals are encapsulated in the **synchrotron kernel functions** :math:`F(x)` and :math:`G(x)`, which are
implemented as fast, asymptotic-aware interpolators for use in inner-loop calculations.

The example below computes the critical frequency and the single-electron power
spectrum for three representative Lorentz factors in a :math:`B = 1` G field,
illustrating how the emission peak shifts to higher frequencies with increasing
:math:`\gamma`:

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u
   from triceratops.radiation.synchrotron.core import (
       compute_nu_critical,
       compute_single_electron_power,
       get_first_kernel_interpolator,
   )

   B = 1.0 * u.G
   nu = np.logspace(6, 18, 500) * u.Hz
   gammas = [1e2, 1e3, 1e4]
   colors = ['steelblue', 'darkorange', 'firebrick']

   # Build a fast asymptotic-aware interpolator for the synchrotron kernel F(x).
   # This is much faster than evaluating the Bessel function integral directly
   # and is the recommended approach for any inner-loop calculation.
   F_interp = get_first_kernel_interpolator(x_min=1e-5, x_max=1e3, num_points=1000)

   fig, ax = plt.subplots(figsize=(7, 4))

   for gamma, color in zip(gammas, colors):
       nu_c = compute_nu_critical(gamma=gamma, B=B)

       # compute_single_electron_power accepts a kernel_function keyword so that
       # users can supply the interpolated kernel rather than the direct evaluator.
       P = compute_single_electron_power(
           nu=nu, gamma=gamma, B=B,
           kernel_function=F_interp,
       )

       ax.loglog(
           nu.value, P.value,
           color=color, lw=1.8,
           label=rf'$\gamma = 10^{{{int(np.log10(gamma))}}}$'
                 rf', $\nu_c = {nu_c.to(u.GHz).value:.1e}$ GHz',
       )
       ax.axvline(nu_c.value, ls='--', color=color, lw=0.8, alpha=0.7)

   ax.set_xlabel(r'Frequency [Hz]', fontsize=12)
   ax.set_ylabel(r'$P_{\rm single}(\nu)$ [erg s$^{-1}$ Hz$^{-1}$]', fontsize=12)
   ax.set_title(r'Single-electron synchrotron spectra, $B = 1$ G', fontsize=11)
   ax.legend(fontsize=9)
   ax.set_ylim([1e-26,1e-20])
   ax.grid(True, which='both', ls='--', alpha=0.4)
   plt.tight_layout()

----

Synchrotron Microphysics
-------------------------

.. hint::

    :ref:`synchrotron_microphysics` covers the full API for electron distribution
    utilities and equipartition closure.  :ref:`synchrotron_cooling` covers the
    cooling-engine interface.  For the theory behind these, see the corresponding
    sections in :ref:`synchrotron_theory`.

The two central microphysical ingredients in synchrotron modeling are the
**electron energy distribution** and the **magnetic field strength**.  Since
neither can be measured directly, they must be connected to macroscopic shock
parameters through a closure relation.

Triceratops adopts the standard **equipartition closure**: a fixed fraction
:math:`\epsilon_e` of the post-shock thermal energy density goes into accelerated
electrons, and a fraction :math:`\epsilon_B` goes into the magnetic field.
The :mod:`~triceratops.radiation.synchrotron.microphysics` module provides the
utilities needed to evaluate these fractions, compute distribution moments, and
obtain emissivities directly from shock conditions.

The example below shows how a single call to
:func:`~triceratops.radiation.synchrotron.microphysics.compute_equipartition_magnetic_field`
and
:func:`~triceratops.radiation.synchrotron.microphysics.compute_bol_emissivity_from_thermal_energy_density`
converts a post-shock thermal energy density into the observable magnetic field and
bolometric synchrotron emissivity:

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u
   from triceratops.radiation.synchrotron.microphysics import (
       compute_equipartition_magnetic_field,
       compute_bol_emissivity_from_thermal_energy_density,
   )

   # Sweep over post-shock thermal energy densities typical of radio transients
   # (from mildly to strongly shocked ejecta).
   u_therm = np.logspace(-4, 2, 100) * u.erg / u.cm**3

   # Fixed microphysical parameters
   epsilon_e = 0.1   # 10% of shock energy into electrons
   epsilon_B = 0.01  # 1% of shock energy into magnetic field
   p         = 3.0   # electron power-law index
   gamma_min = 100.0
   gamma_max = 1e8

   # ── Equipartition magnetic field ──────────────────────────────────────────
   # B = sqrt(8 pi epsilon_B u_therm)
   B = compute_equipartition_magnetic_field(
       u_therm=u_therm,
       epsilon_B=epsilon_B,
   )

   # ── Bolometric synchrotron emissivity ─────────────────────────────────────
   # j_bol = (4/3) sigma_T c (B^2/8pi) epsilon_e/epsilon_B * (moment of distribution)
   j_bol = compute_bol_emissivity_from_thermal_energy_density(
       u_therm=u_therm,
       epsilon_E=epsilon_e,
       epsilon_B=epsilon_B,
       p=p,
       gamma_min=gamma_min,
       gamma_max=gamma_max,
   )

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

   ax1.loglog(u_therm.value, B.to(u.G).value, color='steelblue', lw=2)
   ax1.set_xlabel(r'$u_{\rm th}$ [erg cm$^{-3}$]', fontsize=12)
   ax1.set_ylabel(r'$B$ [G]', fontsize=12)
   ax1.set_title(
       r'Equipartition $B$  ($\epsilon_B = 0.01$)',
       fontsize=11,
   )
   ax1.grid(True, which='both', ls='--', alpha=0.4)

   ax2.loglog(u_therm.value, j_bol.to(u.erg / u.s / u.cm**3).value,
              color='darkorange', lw=2)
   ax2.set_xlabel(r'$u_{\rm th}$ [erg cm$^{-3}$]', fontsize=12)
   ax2.set_ylabel(r'$j_{\rm bol}$ [erg s$^{-1}$ cm$^{-3}$]', fontsize=12)
   ax2.set_title(
       r'Bolometric emissivity  ($\epsilon_e = 0.1$, $p = 3$)',
       fontsize=11,
   )
   ax2.grid(True, which='both', ls='--', alpha=0.4)

   plt.tight_layout()

Radiative cooling modifies the electron spectrum — and therefore the observable
SED — on timescales set by the synchrotron cooling rate and, in some regimes, by
inverse-Compton losses.  Triceratops implements this through a set of pluggable
**cooling engines** (one per physical process) exposed in
:mod:`~triceratops.radiation.synchrotron.cooling`.  Each engine returns the cooling
Lorentz factor :math:`\gamma_c` and the associated break frequency :math:`\nu_c`
for use in the SED models described in the next section.

----

Synchrotron SEDs
-----------------

.. hint::

    :ref:`synchrotron_seds` contains the full SED API, a guide to choosing the right
    model for your source, and comprehensive worked examples including spectrum
    inversion.  :ref:`synch_sed_theory` derives the individual spectral regimes and
    the normalization conventions used throughout the module.

The most practical entry point to synchrotron modeling in Triceratops is the family
of **one-zone SED models** in :mod:`~triceratops.radiation.synchrotron.SEDs`.
Each model assembles a broadband spectrum by smoothly joining power-law segments
across the characteristic break frequencies — the injection frequency
:math:`\nu_m`, the self-absorption frequency :math:`\nu_a`, the cooling frequency
:math:`\nu_c`, and the maximum electron frequency :math:`\nu_{\max}`.

Four physically-motivated classes are provided:

.. list-table::
    :header-rows: 1
    :widths: 42 10 10 38

    * - Class
      - Cooling
      - SSA
      - Use when...
    * - :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_SynchrotronSED`
      - No
      - No
      - Simple optically-thin power-law; no breaks beyond :math:`\nu_m`
    * - :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_Cooling_SynchrotronSED`
      - Yes
      - No
      - Optically thin emission with a fast- or slow-cooling break
    * - :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_SSA_SynchrotronSED`
      - No
      - Yes
      - Dense or compact sources with an SSA turnover and negligible cooling
    * - :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_Cooling_SSA_SynchrotronSED`
      - Yes
      - Yes
      - Full broadband modeling -- the most general option

One-Zone SED: A Worked Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example below models a radio transient (SN-like source) using
:class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_SSA_SynchrotronSED`:
an optically-thick synchrotron spectrum with an SSA turnover at GHz frequencies and
no appreciable radiative cooling.  The physical parameters — magnetic field, source
radius, electron spectrum, and microphysical fractions — are converted to
phenomenological SED parameters through ``from_physics_to_params``, which implements
the equipartition closure :footcite:p:`Chevalier1998SynchrotronSelfAbsorption`.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import units as u
   from triceratops.radiation.synchrotron import PowerLaw_SSA_SynchrotronSED

   # ── Instantiate the SED model ─────────────────────────────────────────────
   # SED objects are stateless: physical parameters are passed at call time,
   # not stored on the object.  This makes them safe to reuse in parameter
   # sweeps and inference loops without hidden state.
   sed = PowerLaw_SSA_SynchrotronSED()

   # ── Physical source parameters ────────────────────────────────────────────
   # These are the quantities a user would typically know from the source model
   # or shock dynamics:
   B         = 0.5 * u.G           # post-shock magnetic field strength
   R         = 1e16 * u.cm         # characteristic emitting-region radius
   gamma_min = 100.0               # minimum electron Lorentz factor
   p         = 3.0                 # power-law index of the electron distribution
   eps_E     = 0.1                 # fraction of shock energy in electrons
   eps_B     = 0.1                 # fraction of shock energy in magnetic field
   D_L       = 100 * u.Mpc         # luminosity distance to the source

   # ── Convert physical parameters to phenomenological SED parameters ────────
   # from_physics_to_params applies the equipartition closure: given B and R
   # it computes the injection frequency nu_m, the SSA frequency nu_a, and the
   # normalization F_norm, together with a solid-angle factor omega.
   params = sed.from_physics_to_params(
       B=B, R=R,
       gamma_min=gamma_min,
       p=p,
       epsilon_E=eps_E,
       epsilon_B=eps_B,
       luminosity_distance=D_L,
       pitch_average=True,   # average over electron pitch angles
   )

   # ── Frequency grid: radio through soft X-ray ─────────────────────────────
   nu = np.logspace(8, 15, 500) * u.Hz

   # ── Evaluate the SED ──────────────────────────────────────────────────────
   # The sed() method assembles the piecewise power-law spectrum.  Note that
   # nu_a (the SSA turnover) is determined internally from nu_m, omega, p, and
   # gamma_m -- it does not need to be supplied explicitly.
   Fnu = sed.sed(
       nu,
       nu_m=params['nu_m'],       # injection break frequency
       F_norm=params['F_norm'],   # flux normalization at nu_m
       nu_max=params['nu_max'],   # high-frequency cutoff
       omega=params['omega'],     # effective solid angle (encodes R and D_L)
       gamma_m=gamma_min,         # minimum Lorentz factor (sets nu_a position)
       p=p,
   )

   # ── Plot ──────────────────────────────────────────────────────────────────
   fig, ax = plt.subplots(figsize=(7, 4))

   ax.loglog(
       nu.to(u.GHz).value,
       Fnu.to(u.mJy).value,
       color='steelblue', lw=2,
   )

   # Mark the characteristic frequencies returned by from_physics_to_params
   ax.axvline(
       params['nu_a'].to(u.GHz).value,
       ls='--', color='firebrick', lw=1.2, label=r'$\nu_a$ (SSA turnover)',
   )
   ax.axvline(
       params['nu_m'].to(u.GHz).value,
       ls='--', color='darkorange', lw=1.2, label=r'$\nu_m$ (injection break)',
   )

   ax.set_xlabel('Frequency [GHz]', fontsize=12)
   ax.set_ylabel(r'$F_\nu$ [mJy]', fontsize=12)
   ax.set_title(
       r'One-zone SSA synchrotron SED (no cooling), $D_L = 100$ Mpc',
       fontsize=11,
   )
   ax.legend(fontsize=10)
   ax.grid(True, which='both', ls='--', alpha=0.4)
   plt.tight_layout()

Spectrum Inversion
^^^^^^^^^^^^^^^^^^^

In many observational scenarios the situation is reversed: you have a measured peak
frequency :math:`\nu_{\rm pk}` and peak flux :math:`F_{\rm pk}` and want to
recover the underlying physical conditions (magnetic field, source radius, electron
energy, etc.).  This **equipartition inversion** is provided by the
``from_params_to_physics`` method on the SSA SED classes.  For a detailed walkthrough
— including the assumptions, the derivation, and how to handle the degeneracy between
regimes — see :ref:`synchrotron_seds`.

----

Further Reading
---------------

The table below maps common modeling tasks to the relevant documentation:

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - If you want to...
      - See...
    * - Understand the physics behind the module
      - :ref:`synchrotron_theory`
    * - Work with fundamental quantities (kernels, single-electron spectra)
      - :ref:`synchrotron_core`
    * - Compute population-averaged emissivities and equipartition quantities
      - :ref:`synchrotron_microphysics`
    * - Build or fit a broadband SED model
      - :ref:`synchrotron_seds`
    * - Understand the SED spectral-regime derivations
      - :ref:`synch_sed_theory`
    * - Model radiative cooling and the cooling break
      - :ref:`synchrotron_cooling`
    * - Learn about stratified synchrotron self-absorption
      - :ref:`stratified_absorption`

.. footbibliography::
