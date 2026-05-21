.. _free_free_theory:

==========================
Free-Free Radiation Theory
==========================

Free-free radiation—also called **bremsstrahlung** ("braking radiation" in German)—is the continuum
emission produced when a free electron is accelerated by the Coulomb field of an ion without being
captured.  It is one of the most ubiquitous thermal radiation mechanisms in astrophysics and is
particularly important for modeling the radio and centimeter-wave emission from ionized circumstellar
media (CSM) around supernovae, HII regions, stellar coronae, and the hot intra-cluster medium.

In the context of Triceratops, free-free emission is primarily relevant as a **foreground opacity**
source in front of synchrotron-emitting shock regions and as a direct contributor to the total
radio flux at early times when the CSM is dense and ionized. It is also relevant to shock-driven
X-ray emission in many transients.

In this documentation we summarize the key formulae and conventions used throughout the Triceratops
free-free module.

.. hint::

    A comprehensive treatment can be found in :footcite:t:`RybickiLightman`,
    Chapters 5.2–5.3. Another accessible treatment of the details is described in
    :footcite:t:`lu_2026_18603474`.

.. contents::
    :local:
    :depth: 2

Overview
--------

When a charged particle is accelerated in the **Coulomb Field** of another particle, it is accelerated,
producing **bremsstrahlung** or **free-free emission**. This is a common source of very high energy photons
in ionized plasmas and is the focus of this section. In general, free-free emission occurs between
electrons and ions in a plasma; in some cases, the plasma is thermalized allowing for
so-called **thermal bremsstrahlung** to occur.

It can be shown that the power emitted by a single electron of charge :math:`e` and mass :math:`m_e`
accelerated by a Coulomb field of charge :math:`Ze` is given by the following formula:

.. math::

    \frac{dW}{d\omega} = \begin{cases}
    \frac{8Z^2e^6}{3\pi c^3 m_e^2 v^2 b^2}, & \text{if } \omega < \frac{v}{b} \\
    0, & \text{if } \omega > \frac{v}{b}
    \end{cases}

where :math:`v` is the velocity of the electron and :math:`b` is the impact parameter of the collision.
This formula indicates that the power emitted by a single electron is inversely proportional to the square of
its velocity and the square of the impact parameter, and it is directly proportional to the square of the
charge of the ion.

The modeling of free-free emission therefore comes down to calculating the total power emitted by a population of
electrons and ions in a plasma, which involves integrating over the distribution of electron velocities and impact
parameters.

Thermal Free-Free Emission and Absorption
------------------------------------------

A detailed calculation (see :footcite:t:`RybickiLightman`, Chapter 5.2) of the total power emitted by a
thermal population of electrons reveals that the emissivity of the plasma can be expressed as
:footcite:p:`lu_2026_18603474, RybickiLightman`:

.. math::

    j_\nu = \frac{16 \sqrt{\pi} e^6}{3\sqrt{3} m_e^2 c^3} \left(\frac{m_e}{2k_B T_e}\right)^{1/2} Z^2
    n_e n_i e^{-h\nu / k_B T_e} g_{\rm ff}( T_e, \nu),

where :math:`n_e` and :math:`n_i` are the number densities of electrons and ions, respectively, :math:`T_e` is the
electron temperature, and :math:`g_{\rm ff}` is the free-free Gaunt factor,
a dimensionless correction factor that accounts for quantum mechanical effects (see sections below).

In CGS units, this becomes

.. math::

    \boxed{
    j_\nu = 6.8 \times 10^{-38} Z^2 n_e n_i T_e^{-1/2}
    e^{-h\nu / k_B T_e} g_{\rm ff}( T_e, \nu) \quad \text{erg s}^{-1} \text{cm}^{-3} \text{Hz}^{-1} \text{sr}^{-1}.
    }

In local thermodynamic equilibrium (LTE), the absorption coefficient can be derived from Kirchhoff's law, which
states that the emissivity and absorption coefficient are related by the Planck function:

.. math::

    \alpha_\nu = \frac{j_\nu}{B_\nu(T_e)},

implying that :footcite:p:`lu_2026_18603474, RybickiLightman`:

.. math::

    \boxed{
    \alpha_\nu^{\rm ff}
    =
    \frac{4\sqrt{\pi} e^6}{3\sqrt{3} m_e^2 h c}
    \left(\frac{2 m_e}{k_B T_e}\right)^{1/2}
    \frac{n_e}{\nu^3}
    \left(1 - e^{-h\nu/k_B T_e}\right)
    \sum_i g_{{\rm ff},i} Z_i^2 n_i.
    }

which again can be expressed in CGS units as

.. math::

    \boxed{
    \alpha_\nu^{\rm ff} = 3.7 \times 10^8 Z^2 n_e n_i T_e^{-1/2} \nu^{-3}
    \left(1 - e^{-h\nu/k_B T_e}\right) g_{\rm ff}( T_e, \nu) \quad \text{cm}^{-1}.
    }

In the Rayleigh–Jeans limit (where :math:`h\nu \ll k_B T_e`), the absorption coefficient simplifies to

.. math::

    \alpha_\nu^{\rm ff} \approx 0.018 Z^2 n_e n_i T_e^{-3/2} \nu^{-2} g_{\rm ff}( T_e, \nu).

.. hint::

    These are the two core equations that govern free-free emission and absorption in a thermal plasma.
    Triceratops implements these formulae in :mod:`~triceratops.radiation.free_free.core`.

Composition
^^^^^^^^^^^

In the discussion above, we have implicitly assumed a single ion species with charge
:math:`Z` and number density :math:`n_i`. In realistic astrophysical plasmas, however,
the gas is composed of multiple ionic species, each contributing independently to the
free–free emission and absorption.

The emissivity is therefore obtained by summing over all ion species:

.. math::

    j_\nu =
    \frac{16 \sqrt{\pi} e^6}{3\sqrt{3} m_e^2 c^3}
    \left(\frac{m_e}{2k_B T_e}\right)^{1/2}
    n_e
    \sum_i n_i Z_i^2
    e^{-h\nu / k_B T_e}
    g_{{\rm ff},i}( T_e, \nu),

which differs from the single-species expression only through the replacement

.. math::

    Z^2 n_i \;\longrightarrow\; \sum_i Z_i^2 n_i.

Similarly, the absorption coefficient becomes

.. math::

    \alpha_\nu^{\rm ff}
    =
    \frac{4\sqrt{\pi} e^6}{3\sqrt{3} m_e^2 h c}
    \left(\frac{2 m_e}{k_B T_e}\right)^{1/2}
    \frac{n_e}{\nu^3}
    \left(1 - e^{-h\nu/k_B T_e}\right)
    \sum_i Z_i^2 n_i\, g_{{\rm ff},i}( T_e, \nu),

which matches the form already written above. :contentReference[oaicite:0]{index=0}

It is often convenient to express these sums in terms of number fractions
:math:`x_i = n_i / n_{\rm ion}`, where :math:`n_{\rm ion} = \sum_i n_i`. In this case,

.. math::

    \sum_i Z_i^2 n_i = n_{\rm ion} \sum_i Z_i^2 x_i,

and one may define an effective Gaunt factor

.. math::

    g_{\rm ff,eff}(T_e, \nu)
    =
    \sum_i Z_i^2 x_i\, g_{{\rm ff},i}(T_e, \nu),

such that the emissivity takes the same form as the single-species expression:

.. math::

    j_\nu \propto n_e n_{\rm ion} g_{\rm ff,eff}.

Finally, we note that the electron density itself depends on the composition through

.. math::

    n_e = \sum_i Z_i n_i,

so that both the normalization and the effective Gaunt factor are composition-dependent.

The Gaunt-Factor
-----------------

The Gaunt factor :math:`g_{\rm ff}( T_e, \nu)` is a dimensionless correction factor that accounts for
various quantum mechanical effects in the calculation of free-free emission and absorption.

The Coulomb Logarithm
^^^^^^^^^^^^^^^^^^^^^

We consider first
a scenario in which a single electron is accelerated by the Coulomb field of many ions with density :math:`n_i`
and charge :math:`Ze`. The emitting spectrum is calculated as :footcite:p:`lu_2026_18603474, RybickiLightman`:

.. math::

    P_\omega(v) = \int db\;2\pi b n_i v \frac{dW}{d\omega} = \frac{16 Z^2 e^6 n_i}{3m_e^2 c^3 v} \ln \Lambda,

where :math:`\ln \Lambda` is the Coulomb logarithm, which accounts for the range of impact parameters
that contribute to the emission:

.. math::

    \ln \Lambda \simeq \ln \left( \frac{b_{\rm max}}{b_{\rm min}} \right),

where :math:`b_{\rm max}` and :math:`b_{\rm min}` are the maximum and minimum impact parameters, respectively.

The clear question becomes the following: *What are the maximum and minimum impact parameters?*.

If the electron is affected by an ion at **too close an impact parameter**, then the impulse approximation on which
the above formulae are based breaks down, and the electron's trajectory is significantly altered. This sets a minimum
impact parameter of

.. math::

    b_{\rm min} \simeq \frac{2|q_1q_2|}{m_1 v^2} = \frac{2 Z e^2}{m_e v^2}.

This is also not the only limit on the minimum impact parameter, however. The velocity of the electron implies
a quantum mechanical uncertainty in its position

.. math::

    b_{\rm min} \simeq \frac{2\hbar}{m_e v},

at which scales the scattering becomes fundamentally quantum mechanical and the classical treatment breaks down. \

If the electron is affected by an ion at **too large an impact parameter**, several effects can come into play. The
most important is that of **Debye shielding**: the Coulomb field of the ion is screened by the surrounding plasma,
which sets a maximum impact parameter of

.. math::

    b_{\rm max} \simeq \lambda_D = \sqrt{\frac{k_B T_e}{4\pi n_e e^2}}.

In atomic plasmas, the "orbital timescale" of the electron around the ion can also set a maximum impact parameter,
but this is typically less relevant in astrophysical contexts.

The details of the Coulomb logarithm have been explored in **great detail** in the literature.

The Gaunt Factor
^^^^^^^^^^^^^^^^

Subject to some conventions and approximation, the Gaunt factor is essentially the Coulomb logarithm in the
context of thermal bremsstrahlung. In the low frequency limit, the Gaunt factor is given by

.. math::

    g_{\rm ff}(\nu, T_e) \simeq \frac{\sqrt{3}}{\pi} \log \left[\frac{1}{Z}
    \frac{(k_B T_e)^{3/2}}{h\nu \sqrt{\rm Ry}}\min\left(1,Z\sqrt{\frac{\rm Ry}{k_B T}}\right)\right].

In Triceratops, we provide a number of options for actually performing the Gaunt factor calculations,

The Gaunt Factor in Practice
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In practice, the Gaunt factor is **not evaluated directly from the Coulomb logarithm**, but instead
computed using a combination of **analytic approximations** and **tabulated data** that incorporate
the full quantum-mechanical treatment of electron–ion scattering. In Triceratops, we provide a unified
interface for evaluating :math:`g_{\rm ff}(Z, T, \nu)` across a wide range of physical regimes.

Analytic Approximations
~~~~~~~~~~~~~~~~~~~~~~~

For performance-critical applications (e.g. MCMC inference), we provide fast analytic
approximations that capture the correct asymptotic behavior.

The default implementation is based on the formulation of :footcite:t:`lu_2026_18603474`:

.. math::

    g_{\rm ff} \approx \ln\!\left(e + \exp(Q)\right),

with

.. math::

    Q = 6 - \frac{\sqrt{3}}{\pi} \ln A,

and

.. math::

    A = \frac{\nu_9}{T_4}
        \max\!\left(0.25,\;\frac{Z}{T_4^{1/2}}\right)^{-1}.

Here :math:`\nu_9 = \nu / 10^9\,{\rm Hz}` and :math:`T_4 = T / 10^4\,{\rm K}`.

A few important features:

- The form :math:`\ln(e + \exp Q)` ensures **smooth interpolation** between regimes
- The Gaunt factor remains **positive and well-behaved**
- Numerical stability is enforced using a log-sum-exp formulation

This approximation is accurate to within a few percent over a wide range of
non-relativistic conditions and is therefore used as the **default**.

A simpler classical approximation due to :footcite:t:`draine2011physics` is also provided:

.. math::

    g_{\rm ff} \approx \ln\!\left(e + \exp(Q)\right),

.. math::

    Q = 5.960 - \frac{\sqrt{3}}{\pi}
        \ln\!\left(\nu_9 T_4^{-3/2} Z\right).

This expression captures the correct logarithmic scaling but is somewhat less accurate,
particularly outside the classical regime.

Tabulated Gaunt Factors
~~~~~~~~~~~~~~~~~~~~~~~

For higher-precision work, Triceratops includes interpolators based on the
tabulations of :footcite:t:`2014MNRAS.444..420V`.

These are implemented as multidimensional interpolators over the variables
:math:`(\log_{10} u, \log_{10}\gamma^2)`:

- **Non-relativistic tables (2D):**

  .. math::

      g_{\rm ff} = g_{\rm ff}(\log_{10} u, \log_{10}\gamma^2)

  In this case, the ion charge :math:`Z` enters only through the coordinate transform.

- **Relativistic tables (3D):**

  .. math::

      g_{\rm ff} = g_{\rm ff}(Z, \log_{10} u, \log_{10}\gamma^2)

  Here, :math:`Z` appears explicitly as an interpolation axis, capturing residual
  charge dependence at high temperatures.

These interpolators use bilinear or trilinear interpolation and are **lazy-loaded**
from HDF5 tables on first use.

Optical Depth
-------------

A very common scenario in transient modeling is free-free absorption of synchrotron emission by ionized CSM.  In this
case, the key quantity is the **free-free optical depth** :math:`\tau_{\rm ff}` along the line of sight through
the absorbing material, which determines the observed spectral energy distribution (SED) and its evolution with time.

The **free-free optical depth** along a ray of length :math:`L` through a uniform medium is

.. math::

    \tau_{\rm ff} = \int_0^L \alpha_\nu(l)\,dl \approx \alpha_\nu\,L.

A source with :math:`\tau_{\rm ff} \gg 1` is **optically thick** to free-free absorption: the
observed spectrum approaches the Rayleigh–Jeans tail of a blackbody,
:math:`S_\nu \propto \nu^2 T`, rather than the underlying synchrotron power law.  At the
**turnover frequency** :math:`\nu_{\rm ff}` where :math:`\tau_{\rm ff} \approx 1`, the SED
transitions between these two behaviors.

In the Rayleigh–Jeans approximation, the turnover frequency scales approximately as

.. math::

    \nu_{\rm ff} \propto \bigl(n_e^2\, L\, T^{-3/2}\bigr)^{1/2},

where :math:`n_e^2 L` is the **emission measure** (EM).  Measuring :math:`\nu_{\rm ff}` from
multi-frequency radio observations therefore provides a diagnostic of the CSM density and
temperature.

CSM Density Profiles
^^^^^^^^^^^^^^^^^^^^

For a stellar wind or ejecta-interaction scenario, the absorbing material is not uniform.
Triceratops supports free-free optical depth calculations for several idealized
circumstellar medium (CSM) density profiles, described fully in the
:ref:`free_free_emission` documentation:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Profile
     - Density law
     - Typical use case
   * - Uniform shell
     - :math:`\rho = {\rm const}`
     - Thin swept-up shells, HII regions
   * - Stellar wind
     - :math:`\rho \propto r^{-2}`
     - Steady-state mass-loss winds
   * - Power law
     - :math:`\rho \propto r^{-p}` (arbitrary :math:`p`)
     - Generalized CSM profiles
   * - Arbitrary quadrature
     - Callable :math:`\alpha_\nu(r)`, :math:`n_e(r)`, :math:`n_i(r)`, :math:`T(r)`
     - Arbitrary CSM structure, hydrodynamic outputs
   * - Array-based
     - Pre-tabulated :math:`r`, :math:`\alpha_\nu` grids
     - Numerical models, hydrodynamic snapshots

Astrophysical Context
---------------------

Free-free absorption is observationally most significant in systems with dense, ionized CSM.  In
radio supernovae, the CSM expelled by the progenitor star before the explosion can be sufficiently
optically thick to free-free absorption to suppress the early-time radio emission entirely,
producing a characteristic spectral turnover that moves to lower frequencies as the shock sweeps
out the CSM
:footcite:p:`ChevalierFranssonHandbook`.

The temporal evolution of the optical depth follows directly from the time evolution of the
shock radius :math:`R(t)` and the CSM density profile.  For a steady wind
(:math:`\rho_{\rm CSM} \propto r^{-2}`), the free-free optical depth along the line of sight
through the unshocked wind decreases as

.. math::

    \tau_{\rm ff}(\nu, t) \propto \dot{M}^2\, v_w^{-2}\, R(t)^{-3}\, \nu^{-2}\, T^{-3/2},

where :math:`\dot{M}` and :math:`v_w` are the progenitor's mass-loss rate and wind velocity.
Fitting this evolution to observed multi-frequency radio light curves constrains the progenitor's
mass-loss history.

----

.. footbibliography::
