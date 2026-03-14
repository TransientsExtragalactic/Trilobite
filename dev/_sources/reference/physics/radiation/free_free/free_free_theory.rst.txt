.. _free_free_theory:

=========================
Free-Free Radiation Theory
=========================

Free-free radiation—also called **bremsstrahlung** ("braking radiation" in German)—is the continuum
emission produced when a free electron is accelerated by the Coulomb field of an ion without being
captured.  It is one of the most ubiquitous thermal radiation mechanisms in astrophysics and is
particularly important for modeling the radio and centimeter-wave emission from ionized circumstellar
media (CSM) around supernovae, HII regions, stellar coronae, and the hot intra-cluster medium.

In the context of Triceratops, free-free emission is primarily relevant as a **foreground opacity**
source in front of synchrotron-emitting shock regions and as a direct contributor to the total
radio flux at early times when the CSM is dense and ionized.

In this documentation we summarize the key formulae and conventions used throughout the Triceratops
free-free module.  A comprehensive treatment can be found in :footcite:t:`RybickiLightman`,
Chapters 5.2–5.3.

.. contents::
    :local:
    :depth: 2

The Emission Process
--------------------

Free-free radiation is produced by the *free–free* transition in which an electron scatters off an
ion without being bound.  The electron is deflected—accelerated—by the Coulomb field of a particle
of charge :math:`Ze`, and the resulting acceleration drives the emission of a photon.  Because no
energy levels are involved, the process produces a smooth (continuum) spectrum.

The dominant contribution in an astrophysical plasma comes from electron–proton (or more generally
electron–ion) encounters.  For a fully ionized thermal plasma with electron number density
:math:`n_e`, ion number density :math:`n_i`, mean ionic charge :math:`Z`, and electron
temperature :math:`T`, the spectral emissivity is

.. math::
    :label: eq_ff_emissivity

    \boxed{
    j_\nu =
    C_{\rm ff}\,
    Z^2\, n_e\, n_i\,
    T^{-1/2}\,
    e^{-h\nu / k_B T}\,
    g_{\rm ff}(Z, T, \nu),
    }

where :math:`C_{\rm ff} = 6.8 \times 10^{-38}` erg s\ :sup:`-1` cm\ :sup:`3` Hz\ :sup:`-1`
sr\ :sup:`-1` K\ :sup:`1/2` is a numerical constant (see :footcite:t:`RybickiLightman`,
Eq. 5.14a), and :math:`g_{\rm ff}` is the dimensionless **free-free Gaunt factor** described
in detail below.

Several features of this expression are worth noting immediately:

* **Exponential high-frequency cutoff.** The factor :math:`e^{-h\nu/k_BT}` reflects the Boltzmann
  probability of finding an electron with enough kinetic energy to produce a photon of frequency
  :math:`\nu`.  At radio and millimeter wavelengths (:math:`h\nu \ll k_BT`) this factor is
  essentially unity, so the free-free spectrum is nearly flat.
* **Temperature dependence.** Even ignoring the exponential cutoff, the emissivity decreases as
  :math:`T^{-1/2}` — hotter plasma radiates less efficiently at a given frequency because the
  electrons move faster and spend less time near each ion.
* **Density squared scaling.** Because each photon involves one electron and one ion,
  :math:`j_\nu \propto n_e n_i \approx n^2` for a fully ionized pure hydrogen plasma. This makes
  free-free emission sensitive to density clumping.

The Absorption Coefficient
--------------------------

By Kirchhoff's law, any medium that can emit thermal radiation must also absorb it in proportion.
The free-free **absorption coefficient** is

.. math::
    :label: eq_ff_absorption

    \boxed{
    \alpha_\nu =
    C_\alpha\,
    Z^2\, n_e\, n_i\,
    T^{-1/2}\,
    \nu^{-3}\,
    \bigl(1 - e^{-h\nu/k_B T}\bigr)\,
    g_{\rm ff}(Z, T, \nu),
    }

where :math:`C_\alpha = 3.7 \times 10^{8}` CGS (:footcite:t:`RybickiLightman`, Eq. 5.19).  The
factor :math:`(1 - e^{-h\nu/k_BT})` encodes **stimulated emission** (the negative absorption
contribution from stimulated transitions) and is sometimes written as a correction to the naive
absorption.

A key observational consequence is the steep **frequency scaling** of the absorption:
:math:`\alpha_\nu \propto \nu^{-3}` (in the limit :math:`h\nu \ll k_BT`), making free-free
opacity very large at low radio frequencies and negligible at X-ray energies.

.. note::

    Equations :eq:`eq_ff_emissivity` and :eq:`eq_ff_absorption` are related by the relation
    :math:`j_\nu = \alpha_\nu B_\nu(T)`, where :math:`B_\nu(T)` is the Planck function.
    Triceratops computes both quantities independently but they share the same Gaunt factor
    and physical constants for consistency.

The Free-Free Gaunt Factor
--------------------------

The **Gaunt factor** :math:`g_{\rm ff}(Z, T, \nu)` is a dimensionless quantum-mechanical
correction of order unity to the classical bremsstrahlung cross section.  In the purely classical
treatment, the emission would be proportional to the square of the electron's acceleration during
each collision, and the resulting spectrum would underestimate the true emission (or absorption) by
a frequency- and temperature-dependent amount.  The Gaunt factor accounts for this discrepancy.

The Gaunt factor is a function of two dimensionless parameters that characterize the collision:

* :math:`u \equiv h\nu / k_B T` — the ratio of the photon energy to the thermal energy,
  often written as :math:`\log_{10} u`.
* :math:`\gamma^2 \equiv Z^2 R_y / k_B T` — a measure of the importance of the ionic potential
  relative to the thermal energy, often written as :math:`\log_{10}\gamma^2`, where
  :math:`R_y = 13.6\,{\rm eV}` is the Rydberg energy.

Values of :math:`g_{\rm ff}` are typically in the range 1–20 and vary logarithmically across the
relevant astrophysical parameter space.

Approximations and Tables
^^^^^^^^^^^^^^^^^^^^^^^^^

Triceratops provides two approaches to evaluating the Gaunt factor.

**Analytic approximation.** A convenient closed-form approximation derived in
:footcite:t:`Draine2011ISM` (Eq. 10.9) is

.. math::

    g_{\rm ff} \approx \frac{\sqrt{3}}{\pi}\,
    \ln\!\left[
        \exp\!\left(5.960 - \frac{\sqrt{3}}{\pi}\ln Z\right) + e \cdot \frac{k_B T^{3/2}}{h \nu}
    \right].

This approximation is fast and adequate for quick estimates, particularly in the radio through
infrared regime, but is less accurate than a tabulated interpolator.

**Tabulated interpolation.** For production-quality calculations Triceratops provides two
interpolators built from the van Hoof et al. (2014) tables:footcite:p:`2014MNRAS.444..420V`,
which are among the most accurate publicly available Gaunt factor grids:

* :class:`~triceratops.radiation.free_free.gaunt_factor.NonRelativisticGauntFactorInterpolator` —
  a two-dimensional interpolator over :math:`\log_{10} u` and :math:`\log_{10} \gamma^2`.  Z enters
  only through the coordinate transform.  This interpolator is appropriate for plasma temperatures
  and frequencies where relativistic corrections to the electron trajectories are negligible.

* :class:`~triceratops.radiation.free_free.gaunt_factor.RelativisticGauntFactorInterpolator` —
  a three-dimensional interpolator that includes an explicit Z axis to capture the residual
  Z-dependence of relativistic corrections.  This is the appropriate choice for high-temperature
  (:math:`T \gtrsim 10^7` K) or high-Z plasma.

Both interpolators use :func:`scipy.interpolate.RegularGridInterpolator` internally and are
described in full detail on the :ref:`free_free_emission` page.

Limiting Regimes
----------------

Two important limiting forms of the free-free formulae are used throughout the Triceratops
implementation.

The Rayleigh–Jeans Limit
^^^^^^^^^^^^^^^^^^^^^^^^^

When :math:`h\nu \ll k_BT` — the **Rayleigh–Jeans (RJ) regime** — the exponential factors
simplify:

.. math::

    e^{-h\nu/k_BT} \approx 1, \qquad
    1 - e^{-h\nu/k_BT} \approx \frac{h\nu}{k_BT}.

The emissivity then becomes **frequency-independent**:

.. math::

    j_\nu^{\rm RJ} = C_{\rm ff}\, Z^2\, n_e\, n_i\, T^{-1/2}\, g_{\rm ff},

while the absorption coefficient scales as

.. math::

    \alpha_\nu^{\rm RJ} \propto
    Z^2\, n_e\, n_i\, T^{-3/2}\, \nu^{-2}\, g_{\rm ff}.

The :math:`\nu^{-2}` scaling of :math:`\alpha_\nu^{\rm RJ}` is the standard **radio bremsstrahlung
opacity** commonly quoted for HII regions and circumstellar shells.  The steeper temperature
dependence (:math:`T^{-3/2}` vs. :math:`T^{-1/2}`) reflects the substitution
:math:`(1-e^{-h\nu/k_BT}) \to h\nu/k_BT`.

The RJ condition is satisfied when

.. math::

    \nu \ll \frac{k_B T}{h} \approx 2.08 \times 10^{10}\,\frac{T}{1\,{\rm K}}\;{\rm Hz.}

For :math:`T = 10^4` K this corresponds to :math:`\nu \lesssim 2 \times 10^{14}` Hz, comfortably
covering the entire radio through near-infrared regime.

The Wien Limit
^^^^^^^^^^^^^^

When :math:`h\nu \gg k_BT` — the **Wien regime** — the stimulated-emission correction approaches
unity:

.. math::

    1 - e^{-h\nu/k_BT} \approx 1,

and the absorption coefficient reduces to

.. math::

    \alpha_\nu^{\rm Wien} \propto
    Z^2\, n_e\, n_i\, T^{-1/2}\, \nu^{-3}\, g_{\rm ff}.

This regime applies in the hard X-ray or gamma-ray band for typical plasma temperatures.
Note that in this limit the emissivity formula is **mathematically identical** to the full
expression :eq:`eq_ff_emissivity`—the Wien limit only simplifies the absorption.

.. note::

    Triceratops exposes explicit Wien-limit backends (``_log_ff_Wien_emissivity`` and
    ``_log_ff_Wien_absorption``) primarily for clarity and testing.  For most practical
    calculations the full expressions should be preferred.

Optical Depth
-------------

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
