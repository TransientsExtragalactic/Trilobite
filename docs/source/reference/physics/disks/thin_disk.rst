.. _thin_disk:

=====================================================
Steady-State Thin Disk (Shakura-Sunyaev Alpha Disk)
=====================================================

The steady-state thin-disk models describe the time-**averaged** radial structure of
a geometrically thin, optically thick accretion disk in which viscosity is
parameterised by the Shakura-Sunyaev :math:`\alpha` prescription.  Unlike the
:ref:`one-zone models <one_zone_disk>`, which integrate the disk evolution in time,
these models evaluate the **instantaneous** disk structure at an arbitrary radius
given fixed global parameters :math:`(M_{\rm BH},\,\dot{M},\,R_{\rm in})`.

They are appropriate when:

- the disk has reached (or can be approximated as) a **steady-state**,
- you need to compute the **multi-colour blackbody SED** of an optically thick disk,
- or you want to check structural scalings (surface density, scale height, etc.)
  against analytic expectations.

.. contents::
    :local:
    :depth: 2

----

Quick Start
-----------

.. code-block:: python

    import numpy as np
    from astropy import constants as const
    from astropy import units as u
    from triceratops.dynamics.accretion import AlphaDisk

    # 1. Instantiate the model with the viscosity parameter.
    disk = AlphaDisk(alpha=0.1)

    # 2. Evaluate the disk structure at a grid of radii.
    R_g = (const.G * 10 * const.M_sun / const.c**2).to(u.cm)
    R_in = 6 * R_g  # Schwarzschild ISCO for a 10 M_sun black hole
    R = np.geomspace(1.01, 1e4, 300) * R_in

    result = disk.compute(R, 10 * const.M_sun, 1e16 * u.g / u.s, R_in)

    print(result["Sigma"])   # surface density [g cm^-2]
    print(result["T_c"])     # midplane temperature [K]
    print(result["H"])       # scale height [cm]

    # 3. Compute the bolometric luminosity.
    L_bol = disk.compute_bolometric_luminosity(10 * const.M_sun, 1e16 * u.g / u.s, R_in)
    print(L_bol.to(u.erg / u.s))

    # 4. Compute the multi-colour blackbody SED.
    nu = np.geomspace(1e13, 1e19, 500) * u.Hz
    sed = disk.compute_sed(
        nu, 10 * const.M_sun, 1e16 * u.g / u.s, R_in,
        R_out=1e4 * R_in, D_L=10 * u.kpc,
    )
    print(sed["L_nu"])   # spectral luminosity [erg/s/Hz]
    print(sed["F_nu"])   # flux density [erg/s/Hz/cm^2]

----

The Shakura-Sunyaev Scalings
-----------------------------

The :class:`~triceratops.dynamics.accretion.AlphaDisk` evaluates the canonical
SS73 (zone-B, gas-pressure-dominated, electron-scattering opacity) scalings at
radius :math:`R`:

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

The factor :math:`f \to 0` at :math:`R \to R_{\rm in}` (zero-torque inner
boundary condition) and :math:`f \to 1` at :math:`R \gg R_{\rm in}`, recovering
the asymptotic power-law scalings.

.. note::

    These are the **zone-B** (gas-pressure-dominated, electron-scattering opacity)
    scalings from Shakura & Sunyaev (1973).  They are valid for moderately-accreting
    stellar-mass and supermassive black holes where gas pressure dominates over
    radiation pressure.

----

Output Fields
-------------

:meth:`~triceratops.dynamics.accretion.AlphaDisk.compute` returns a dict with
the following keys:

.. list-table::
    :header-rows: 1
    :widths: 15 45 20

    * - Key
      - Description
      - Units
    * - ``"Sigma"``
      - Surface density :math:`\Sigma`
      - g cm⁻²
    * - ``"T_c"``
      - Midplane temperature :math:`T_c`
      - K
    * - ``"H"``
      - Pressure scale height :math:`H`
      - cm
    * - ``"rho"``
      - Midplane mass density :math:`\rho = \Sigma / (2H)`
      - g cm⁻³
    * - ``"tau"``
      - Vertical optical depth :math:`\tau`
      - dimensionless
    * - ``"nu"``
      - Kinematic viscosity :math:`\nu = \alpha c_s H`
      - cm² s⁻¹
    * - ``"v_R"``
      - Radial drift velocity :math:`v_R`
      - cm s⁻¹

----

Effective Temperature and SED
------------------------------

The **effective (surface) temperature** — which enters the blackbody radiation — is
computed from first principles via the viscous dissipation formula, independently of
the SS73 scaling prescription:

.. math::

    T_{\rm eff}(r) =
    \left(\frac{3\,G\,M_{\rm BH}\,\dot{M}}
               {8\pi\,\sigma_{\rm SB}\,r^3}
         \left[1 - \sqrt{\frac{R_{\rm in}}{r}}\right]
    \right)^{1/4}.

Note that :math:`T_c \neq T_{\rm eff}`; the two are related by the optical
depth through the disk:
:math:`T_c = \bigl(\tfrac{3\tau}{4}\bigr)^{1/4} T_{\rm eff}`.

The **multi-colour blackbody SED** integrates the Planck function weighted by
:math:`T_{\rm eff}(r)` over all annuli:

.. math::

    L_\nu = 4\pi^2 \int_{R_{\rm in}}^{R_{\rm out}} B_\nu\!\left[T_{\rm eff}(r)\right] r\,\mathrm{d}r.

If a luminosity distance :math:`D_L` is supplied, the observed flux density

.. math::

    F_\nu = \frac{4\pi\cos\theta}{D_L^2}
            \int_{R_{\rm in}}^{R_{\rm out}} B_\nu\!\left[T_{\rm eff}(r)\right] r\,\mathrm{d}r

is also returned.

.. code-block:: python

    nu = np.geomspace(1e13, 1e19, 500) * u.Hz
    sed = disk.compute_sed(
        nu, M_BH, mdot, R_in, R_out=1e4 * R_in,
        D_L=10 * u.kpc,   # omit to get only L_nu
        cos_theta=1.0,     # face-on inclination
        N_r=500,           # radial quadrature points
    )
    L_nu = sed["L_nu"]   # always present
    F_nu = sed["F_nu"]   # present only when D_L is given

The **bolometric luminosity** from the analytic result (integrating the
dissipation profile over both disk faces to :math:`R_{\rm out} \to \infty`):

.. math::

    L_{\rm bol} = \frac{G\,M_{\rm BH}\,\dot{M}}{2\,R_{\rm in}}.

.. code-block:: python

    L_bol = disk.compute_bolometric_luminosity(M_BH, mdot, R_in)

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
