.. _thin_disk:

=====================================================
Steady-State Thin Disk
=====================================================

The steady-state thin-disk models describe the **steady-state** radial structure of
a geometrically thin, optically thick accretion disk in which viscosity is
parameterised by the Shakura-Sunyaev :math:`\alpha` prescription
:footcite:p:`1973A&A....24..337S`.
Unlike the :ref:`one-zone models <one_zone_disk>`, which integrate the disk evolution
in time, these models evaluate the **instantaneous** disk structure at an arbitrary
radius given fixed global parameters :math:`(M_{\rm BH},\,\dot{M},\,R_{\rm in})`.

They are appropriate when:

- the disk has reached (or can be approximated as) a **steady state**,
- you need to compute the **multi-colour blackbody SED** of an optically thick disk,
- or you want to check structural scalings (surface density, scale height, etc.)
  against analytic expectations.

.. contents::
    :local:
    :depth: 2

----

Quick Start
-----------

The most common task is constructing the spectral energy distribution (SED) of a
thin disk with given physical parameters.  The example below sets up a
:math:`10\,M_\odot` black hole accreting at :math:`\dot{M} = 10^{16}\,\mathrm{g\,s^{-1}}`,
places it at a distance of 10 kpc, and plots the multi-colour blackbody SED from
the far-UV through soft X-rays.

.. plot::
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from astropy import constants as const
   from astropy import units as u
   from triceratops.dynamics.accretion import AlphaDisk

   # ── Physical parameters ───────────────────────────────────────────────────
   M_BH  = 10 * const.M_sun           # black hole mass
   mdot  = 1e16 * u.g / u.s          # mass accretion rate (Eddington-ish for 10 M_sun)
   alpha = 0.1                        # Shakura-Sunyaev viscosity parameter

   # Inner truncation radius: 6 R_g (Schwarzschild ISCO for a non-spinning BH)
   R_g   = (const.G * M_BH / const.c**2).to(u.cm)
   R_in  = 6.0 * R_g

   # Outer radius: where the disk effectively truncates
   R_out = 1e3 * R_in

   # Luminosity distance — needed to convert L_nu to F_nu (observed flux density)
   D_L   = 10 * u.kpc

   # ── Instantiate the disk model ────────────────────────────────────────────
   # AlphaDisk implements the SS73 / FKR analytical scalings.
   # The alpha parameter is the only model-level constant; physical parameters
   # (M_BH, mdot, R_in) are passed at evaluation time.
   disk = AlphaDisk(alpha=alpha)

   # ── Frequency grid — far-UV through soft X-ray ───────────────────────────
   # 500 log-spaced points from 1e13 Hz (~12000 Angstrom, NIR) to 1e18 Hz (~4 keV)
   nu = np.geomspace(1e14, 1e18, 500) * u.Hz

   # ── Compute the multi-colour blackbody SED ────────────────────────────────
   # compute_sed integrates the Planck function B_nu(T_eff(r)) over all annuli.
   # Setting D_L causes the returned dict to include both L_nu and F_nu.
   sed = disk.compute_sed(
       nu,
       M_BH,
       mdot,
       R_in,
       R_out=R_out,
       D_L=D_L,
       cos_theta=1.0,   # face-on inclination (maximises observed flux)
       N_r=500,         # number of radial quadrature points
   )

   # sed["L_nu"] -- spectral luminosity [erg/s/Hz], always returned
   # sed["F_nu"] -- flux density [erg/s/Hz/cm^2], returned when D_L is set
   nu_F_nu = (nu * sed["F_nu"]).to(u.erg / u.s / u.cm**2)

   # ── Plot: nu * F_nu vs nu (standard SED presentation) ────────────────────
   fig, ax = plt.subplots(figsize=(7, 4))

   ax.loglog(nu.value, nu_F_nu.value, color="steelblue", lw=2)

   ax.set_xlabel(r"Frequency  $\nu$  [Hz]", fontsize=12)
   ax.set_ylabel(r"$\nu F_\nu$  [erg s$^{-1}$ cm$^{-2}$]", fontsize=12)
   ax.set_title(
       r"Multi-colour blackbody SED — "
       r"$10\,M_\odot$ BH, $\dot{M}=10^{16}$ g s$^{-1}$, $D_L=10$ kpc",
       fontsize=11,
   )
   ax.grid(True, which="both", ls="--", alpha=0.4)
   plt.tight_layout()

----

The Thin Disk Solution
-----------------------

The analytical solution implemented here was derived by
:footcite:t:`1973A&A....24..337S` in the seminal paper that introduced the
:math:`\alpha`-viscosity prescription, and is presented in the textbook form used
throughout this module by :footcite:t:`frank2002accretion` (hereafter FKR).
The solution follows from the steady-state mass and angular momentum conservation
equations combined with the :math:`\alpha`-viscosity relation
:math:`\nu = \alpha c_s H`, a Keplerian rotation profile, and a zero-torque
inner boundary condition at :math:`R_{\rm in}`.

The scalings implemented in :class:`~triceratops.dynamics.accretion.AlphaDisk`
correspond to the **gas-pressure-dominated, Kramers-opacity** (zone C) regime
of :footcite:t:`frank2002accretion` Eq. 5.49.  In this regime the midplane
temperature is low enough that radiation pressure is negligible and the dominant
opacity source is bound-free/free-free (Kramers) absorption rather than electron
scattering.  The resulting analytical scalings are:

.. math::

    \begin{aligned}
    \Sigma &= 5.2\;
    \alpha^{-4/5}\,
    \dot{M}_{16}^{7/10}\,
    M_1^{1/4}\,
    R_{10}^{-3/4}\,
    f^{14/5}
    \quad [\mathrm{g\,cm^{-2}}],
    \\[4pt]
    T_c &= 1.4\times10^{4}\;
    \alpha^{-1/5}\,
    \dot{M}_{16}^{3/10}\,
    M_1^{1/4}\,
    R_{10}^{-3/4}\,
    f^{6/5}
    \quad [\mathrm{K}],
    \\[4pt]
    H &= 1.7\times10^{8}\;
    \alpha^{-1/10}\,
    \dot{M}_{16}^{3/20}\,
    M_1^{-3/8}\,
    R_{10}^{9/8}\,
    f^{3/5}
    \quad [\mathrm{cm}],
    \\[4pt]
    \rho &= 3.1\times10^{-8}\;
    \alpha^{-7/10}\,
    \dot{M}_{16}^{11/20}\,
    M_1^{5/8}\,
    R_{10}^{-15/8}\,
    f^{11/5}
    \quad [\mathrm{g\,cm^{-3}}],
    \\[4pt]
    \tau &= 190\;
    \alpha^{-4/5}\,
    \dot{M}_{16}^{1/5}\,
    f^{4/5},
    \\[4pt]
    \nu &= 1.8\times10^{14}\;
    \alpha^{4/5}\,
    \dot{M}_{16}^{3/10}\,
    M_1^{-1/4}\,
    R_{10}^{3/4}\,
    f^{6/5}
    \quad [\mathrm{cm^2\,s^{-1}}],
    \\[4pt]
    v_R &= 2.7\times10^{4}\;
    \alpha^{4/5}\,
    \dot{M}_{16}^{3/10}\,
    M_1^{-1/4}\,
    R_{10}^{-1/4}\,
    f^{-14/5}
    \quad [\mathrm{cm\,s^{-1}}],
    \end{aligned}

where :math:`\dot{M}_{16} \equiv \dot{M}/10^{16}\ \mathrm{g\,s^{-1}}`,
:math:`M_1 \equiv M_{\rm BH}/M_\odot`,
:math:`R_{10} \equiv R/10^{10}\ \mathrm{cm}`, and the inner-boundary
correction factor satisfies

.. math::

    f^4 \;\equiv\; 1 - \sqrt{\frac{R_{\rm in}}{R}}.

The factor :math:`f \to 0` at :math:`R \to R_{\rm in}` (enforcing the zero-torque
inner boundary condition) and :math:`f \to 1` at :math:`R \gg R_{\rm in}`,
recovering the asymptotic power-law scalings.

----

Usage Guide
-----------

This section walks through the full workflow for working with
:class:`~triceratops.dynamics.accretion.AlphaDisk` — from construction through
unit handling to SED computation.  All public methods accept both plain floats
(assumed CGS) and :class:`~astropy.units.Quantity` objects, so you can mix-and-match
as is most convenient.

Instantiating the Model
^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~triceratops.dynamics.accretion.AlphaDisk` takes a single model-level
parameter: the Shakura-Sunyaev viscosity :math:`\alpha`.  Physical disk parameters
(:math:`M_{\rm BH}`, :math:`\dot{M}`, :math:`R_{\rm in}`) are *not* stored on the
object; they are passed at evaluation time.  This makes the same ``disk`` instance
reusable across parameter surveys without re-instantiation.

.. code-block:: python

    from triceratops.dynamics.accretion import AlphaDisk

    disk = AlphaDisk(alpha=0.1)   # 0 < alpha <= 1

Computing the Disk Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Call :meth:`~triceratops.dynamics.accretion.AlphaDisk.compute` (or equivalently
``disk(radius, M_BH, mdot, R_in)``) to evaluate the full set of structural
variables at one or more radii:

.. code-block:: python

    import numpy as np
    from astropy import constants as const
    from astropy import units as u

    M_BH = 10 * const.M_sun
    mdot = 1e16 * u.g / u.s
    R_g  = (const.G * M_BH / const.c**2).to(u.cm)
    R_in = 6.0 * R_g

    # Evaluate on a log-spaced radial grid from just outside R_in to 10^4 R_in
    R = np.geomspace(1.01, 1e4, 300) * R_in

    result = disk.compute(R, M_BH, mdot, R_in)

The return value is a plain Python ``dict`` whose values are
:class:`~astropy.units.Quantity` arrays with appropriate CGS units attached.

.. list-table::
    :header-rows: 1
    :widths: 15 45 25

    * - Key
      - Description
      - Units
    * - ``"Sigma"``
      - Surface density :math:`\Sigma`
      - :math:`\mathrm{g\,cm^{-2}}`
    * - ``"T_c"``
      - Midplane temperature :math:`T_c`
      - K
    * - ``"H"``
      - Pressure scale height :math:`H`
      - cm
    * - ``"rho"``
      - Midplane mass density :math:`\rho = \Sigma / (2H)`
      - :math:`\mathrm{g\,cm^{-3}}`
    * - ``"tau"``
      - Vertical optical depth :math:`\tau`
      - dimensionless
    * - ``"nu"``
      - Kinematic viscosity :math:`\nu = \alpha c_s H`
      - :math:`\mathrm{cm^2\,s^{-1}}`
    * - ``"v_R"``
      - Radial drift velocity :math:`v_R`
      - :math:`\mathrm{cm\,s^{-1}}`

Handling Units
^^^^^^^^^^^^^^^

All public methods use :func:`~triceratops.utils.misc_utils.ensure_in_units`
internally to convert inputs to CGS before computation.  This means you can pass
parameters in any consistent unit system and the conversion happens transparently:

.. code-block:: python

    # All three of these are equivalent
    result1 = disk.compute(R, 10 * const.M_sun, 1e16 * u.g / u.s, R_in)
    result2 = disk.compute(R, 10 * const.M_sun, 1e19 * u.kg / u.s, R_in)
    result3 = disk.compute(R.to(u.km), 10 * const.M_sun, 1e16 * u.g / u.s, R_in)

Outputs are always returned as :class:`~astropy.units.Quantity` objects, so you
can convert to preferred units with standard astropy machinery:

.. code-block:: python

    print(result["Sigma"].to(u.kg / u.m**2))   # surface density in SI
    print(result["T_c"].to(u.eV, equivalencies=u.temperature_energy()))

If you pass raw floats (no units attached), they are assumed to be in CGS:

.. code-block:: python

    # Radius in cm, mass in g, mdot in g/s, R_in in cm -- all CGS
    result_cgs = disk.compute(1e10, 2e34, 1e16, 3e6)

The Effective Temperature Profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~triceratops.dynamics.accretion.AlphaDisk.compute_effective_temperature`
returns :math:`T_{\rm eff}(r)` from the first-principles viscous dissipation
formula — independently of the zone-C structural scalings.  This is the temperature
that governs the emitted spectrum from each annulus and the one that enters the SED
computation:

.. math::

    T_{\rm eff}(r) =
    \left(\frac{3\,G\,M_{\rm BH}\,\dot{M}}
               {8\pi\,\sigma_{\rm SB}\,r^3}
         \left[1-\sqrt{\frac{R_{\rm in}}{r}}\right]
    \right)^{1/4}.

Note that :math:`T_c \neq T_{\rm eff}`.  The midplane temperature :math:`T_c`
returned by :meth:`~triceratops.dynamics.accretion.AlphaDisk.compute` is hotter
because photons must diffuse vertically through the disk; the two are related by
:math:`T_c = \bigl(\tfrac{3\tau}{4}\bigr)^{1/4} T_{\rm eff}`.

.. code-block:: python

    T_eff = disk.compute_effective_temperature(R, M_BH, mdot, R_in)
    print(T_eff)   # Quantity in K, same shape as R

Computing the SED
^^^^^^^^^^^^^^^^^^

:meth:`~triceratops.dynamics.accretion.AlphaDisk.compute_sed` integrates the
Planck function over all annuli to produce the multi-colour blackbody spectrum.
The radial integral is evaluated on a log-spaced grid using the trapezoidal rule.
At minimum you must supply a frequency grid, the three physical disk parameters,
and an outer radius:

.. code-block:: python

    nu = np.geomspace(1e13, 1e18, 500) * u.Hz

    sed = disk.compute_sed(
        nu, M_BH, mdot, R_in,
        R_out=1e4 * R_in,   # outer truncation radius
    )
    L_nu = sed["L_nu"]      # spectral luminosity [erg/s/Hz], always present

To obtain the observed flux density at a known source distance, pass the luminosity
distance ``D_L``.  You may also specify the inclination via ``cos_theta``
(default 1.0 = face-on):

.. code-block:: python

    sed = disk.compute_sed(
        nu, M_BH, mdot, R_in,
        R_out=1e4 * R_in,
        D_L=10 * u.kpc,     # luminosity distance -- activates F_nu output
        cos_theta=0.5,      # 60 degree inclination from the disk normal
        N_r=500,            # radial quadrature points (default)
    )
    L_nu = sed["L_nu"]      # [erg/s/Hz]
    F_nu = sed["F_nu"]      # [erg/s/Hz/cm^2], only present when D_L is given

Bolometric Luminosity
^^^^^^^^^^^^^^^^^^^^^^

For a quick analytic estimate of the total radiated power — without evaluating the
full SED integral — use
:meth:`~triceratops.dynamics.accretion.AlphaDisk.compute_bolometric_luminosity`.
This returns the exact result of integrating the viscous dissipation profile over
both disk faces to :math:`R_{\rm out} \to \infty`:

.. math::

    L_{\rm bol} = \frac{G\,M_{\rm BH}\,\dot{M}}{2\,R_{\rm in}}.

.. code-block:: python

    L_bol = disk.compute_bolometric_luminosity(M_BH, mdot, R_in)
    print(L_bol.to(u.erg / u.s))

----

Relation to One-Zone Models
----------------------------

The thin-disk models and the :ref:`one-zone models <one_zone_disk>` are complementary:

.. list-table::
    :header-rows: 1
    :widths: 30 35 35

    * -
      - Thin disk (:class:`~triceratops.dynamics.accretion.AlphaDisk`)
      - One-zone (:class:`~triceratops.dynamics.accretion.GasPressureDisk`)
    * - Time dependence
      - Steady state (snapshot at fixed :math:`\dot{M}`)
      - Time-evolved ODE (tracks :math:`M_D(t)`, :math:`J_D(t)`)
    * - Radial resolution
      - Full radial profile :math:`f(R)`
      - Single zone at characteristic radius :math:`R_D`
    * - SED
      - Multi-colour blackbody over all annuli
      - Single-temperature blackbody at :math:`T_{\rm eff}(R_D)`
    * - Typical use
      - Spectral fitting, structural checks, SS73 theory validation
      - Light-curve modeling, Bayesian inference, parameter sweeps

----

API Reference
-------------

.. currentmodule:: triceratops.dynamics.accretion

.. autosummary::
    :toctree: generated/

    ThinDiskBase
    AlphaDisk

.. footbibliography::
