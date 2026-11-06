.. _synch_numerical_sed_theory:

==============================================
Methods: Quadrature Based SEDs
==============================================

.. seealso::

    - :ref:`synch_sed_theory` for the methods used to compute analytical synchrotron SEDs.
    - :ref:`synch_numerical_sed` for the user guide for the numerical synchrotron code.

This document describes the numerical methods implemented in Trilobite for **computing synchrotron spectral energy
distributions (SEDs) from an arbitrary electron distribution**. The analytical library described in :ref:`synch_sed_theory`
covers a wide range of important cases, but there are many physically interesting scenarios for which a closed-form
expression for the SED is either unavailable or impractical to derive. These include:

- **Arbitrary electron distributions**, such as thermal (Maxwell-Jüttner), broken power-laws with non-trivial cooling,
  or numerically evolved particle spectra. The analytical library is restricted to simple power-law forms.
- **Multi-component sources**, where several distinct emission regions contribute to the observed SED and each must be
  evaluated independently before being combined.
- **Heterogeneous magnetic field structures**, where the field varies across the source and a simple single-zone
  treatment is insufficient.

In these cases, a robust numerical method is needed. The approach implemented in Trilobite is based on direct
quadrature of the fundamental synchrotron integrals, performed in log-space for numerical stability. The central
objective of this note is to document the mathematics and algorithmic choices behind that implementation, which lives
in :mod:`trilobite.radiation.synchrotron.SEDs.numerical`.

.. hint::

    The algorithm described in this section is sufficiently quick for use in MCMC hot loop models, making it
    an excellent workhorse for fitting complex synchrotron emission scenarios.

.. contents::
    :local:
    :depth: 2

Overview
--------

Even in the **analytical theory** of synchrotron emission, the fundamental building blocks are the emissivity and
absorption coefficients, which can be represented as **integrals over fundamental kernels** (see :ref:`synchrotron_theory`).

To be as specific as possible in our construction, we will first consider the case of emission from a **single region**
subject to the following geometric / physical assumptions:

1. The source is a **homogeneous slab** of thickness :math:`\ell` along the line of sight with a uniform magnetic field
   strength :math:`B`.
2. The electron distribution is **homogeneous** throughout the source and is described by a distribution function
   :math:`N(\gamma)`, where :math:`\gamma` is the electron Lorentz factor.
3. The source moves with a **bulk Lorentz** factor :math:`\Gamma_{\rm bulk}` at an angle :math:`\theta_{\rm obs}` to the
   line of sight, resulting in a Doppler factor :math:`\delta = [\Gamma (1 - \beta \cos \theta_{\rm obs})]^{-1}`.
4. The source is located at a **luminosity distance** :math:`d_L` from the observer and is observed at a corresponding
   redshift :math:`z`. This also implies an angular diameter distance :math:`d_A = d_L / (1 + z)^2` and **corresponding
   angular size** :math:`d\Omega = A / d_A^2`, where :math:`A` is the physical area of the source on the plane of the sky.

Under these assumptions, the observed SED can be computed by integrating the radiative transfer equation along
the line of sight through the source, which reduces to a simple expression in terms of the emissivity :math:`j_\nu`
and absorption coefficient :math:`\alpha_\nu`. This will be the starting point for our model.


Absorption and Emission Coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    Unless otherwise stated, we will work in the comoving frame of the source, where the bulk motion
    is zero. Relativistic transformations to the observer frame will be applied at the end of the calculation.

For a population of relativistic electrons with (rest-frame) number density :math:`dN/d\gamma = N(\gamma)` spiraling in a magnetic
field :math:`B` at a pitch angle :math:`\alpha` to the field, the emissivity is given by (see e.g. :footcite:t:`RybickiLightman`,
Chapter 6; :footcite:t:`lu_2026_18603474`, Chapter 8;
:footcite:t:`ghisellini2013radiative`, Chapter 4):

.. math::

    j_\nu = \frac{\sqrt{3}\,e^3 B \sin\alpha}{4\pi m_e c^2}
            \int_{\gamma_{\min}}^{\gamma_{\max}} N(\gamma)\,F\!\left(\frac{\nu}{\nu_c}\right) d\gamma,

where :math:`\nu_c = c_{1\gamma} B \sin\alpha\,\gamma^2` is the **critical synchrotron frequency** for an electron
with Lorentz factor :math:`\gamma`, and :math:`c_{1\gamma} = 3e/(4\pi m_e c)`. The function :math:`F(x)` is the
**first synchrotron kernel**, defined by

.. math::

    F(x) = x \int_x^\infty K_{5/3}(z)\, dz,

where :math:`K_{5/3}(z)` is the modified Bessel function of the second kind of order :math:`5/3`. The kernel
is a smooth, unimodal function that peaks at :math:`x \approx 0.29` and describes the spectral shape of the
radiation emitted by a single electron.

The synchrotron self-absorption coefficient is, likewise, given by\ :footcite:p:`lu_2026_18603474, RybickiLightman, ghisellini2013radiative`

.. math::

    \alpha_\nu = -\frac{\sqrt{3}\,e^3 B \sin\alpha}{8\pi m_e^2 c^2 \nu^2}
                 \int_{\gamma_{\min}}^{\gamma_{\max}} F\!\left(\frac{\nu}{\nu_c}\right)\,\gamma^2\,
                 \frac{d}{d\gamma}\!\left(\frac{N(\gamma)}{\gamma^2}\right) d\gamma.

Both :math:`j_\nu` and :math:`\alpha_\nu` require a numerical integration over the electron distribution that
involves evaluating the kernel :math:`F(x)` at each point in a two-dimensional grid of :math:`(\nu, \gamma)`
values. The kernel itself is expensive to evaluate directly (it requires numerical integration of a Bessel function),
so efficient computation demands that it be pre-tabulated and interpolated.

The following sections describe the methods by which Trilobite achieves these calculations with a high
degree of fidelity and numerical stability, including the handling of asymptotic regimes, the use of log-space quadrature,
and the application of integration by parts to simplify the absorption coefficient integral.

----

Synchrotron Kernels
-------------------

At the core of our numerical synchrotron implementation is a nuanced handling of the so-called
**synchrotron kernels**, which are the special functions that appear in the integrals for the emissivity and absorption coefficients.

The First Kernel
^^^^^^^^^^^^^^^^^

The central special function in synchrotron theory is the **synchrotron kernel of the first kind**, defined by\ :footcite:p:`RybickiLightman`

.. math::

    F(x) = x \int_x^\infty K_{5/3}(z)\, dz.

From a numerical standpoint, it is not only infeasible to evaluate :math:`F(x)` frequently to construct an SED, but also
quite pathological to simply evaluate it at all for certain values of :math:`x`. Particularly in the limit :math:`x\gg 1`,
the Bessel function :math:`K_{5/3}(z)` is exponentially suppressed, and numerical quadrature begins to fail.

Fortunately, asymptotic forms of :math:`F(x)` have been
derived in both the small-:math:`x` and large-:math:`x` limits\ :footcite:p:`RybickiLightman, lu_2026_18603474`:

.. math::

    F(x) \;\approx\; \frac{4\pi}{2^{1/3} 3^{1/2} \Gamma(1/3)}\,x^{1/3} \quad\text{for } x \ll 1,
    \qquad\qquad
    F(x) \;\approx\; \sqrt{\frac{\pi x}{2}}\,e^{-x} \quad\text{for } x \gg 1.

It may also be well approximated by\ :footcite:p:`lu_2026_18603474`

.. math::

    F(x) \approx \frac{\left[(\pi x/2)^{s_1/2} + 1\right]^{1/s_1}}{[(2.150x^{1/3})^{-s_2} + 1]^{1/s_2}} e^{-x},

where :math:`s_1 = 1.430` and :math:`s_2 = 2.627`. To properly capture the behavior of :math:`F(x)` across its entire
domain, Trilobite uses the exact Bessel integral for intermediate :math:`x` values and switches to
the asymptotic forms outside a user-configurable range. The exact method is preferred for production
calculations, while the approximation can be used for rapid prototyping.

In addition to the kernel itself, we are also interested in its derivative

.. math::

    \frac{dF}{dx} = \frac{\partial}{\partial x}\!\left[x \int_x^\infty K_{5/3}(z)\, dz\right],

which, by the Leibniz rule, can be expressed as

.. math::

    \frac{dF}{dx} = \frac{F(x)}{x} - x K_{5/3}(x).

Which has the asymptotic behavior

.. math::

    \frac{dF}{dx} \;\approx\; \frac{4\pi}{3\sqrt{3}} \frac{1}{\Gamma(1/3)2^{1/3}} x^{-2/3} \quad\text{for } x \ll 1,
    \qquad\qquad
    \frac{dF}{dx} \;\approx\; \sqrt{\frac{\pi x}{2}} e^{-x} \left(\frac{1}{2x} - 1\right) \quad\text{for } x \gg 1.

The Pitch-Angle Averaged Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the electron pitch angles are distributed **isotropically**, as is appropriate for most astrophysical
environments where magnetic field fluctuations randomize the electron trajectories on timescales shorter than a
dynamical time, one should use the pitch-angle averaged emissivity. In this case, no single pitch angle
:math:`\alpha` characterizes the source, and the emission must be averaged over the isotropic distribution. The
relevant kernel is the **pitch-angle averaged first kernel**\ :footcite:p:`lu_2026_18603474`, defined by

.. math::

    \bar{F}(x) = \int_0^{1} F\!\left(\frac{x}{\xi}\right) \frac{\xi^2}{\sqrt{1-\xi^2}}\, d\xi.

It can be shown that :math:`\bar{F}(x)` has a closed form in terms of the various Bessel
functions\ :footcite:p:`1986A&A...164L..16C, 1988ApJ...334L...5G`

.. math::

    \bar{F}(x)
    =
    \frac{x^2}{2}
    \left[
        K_{4/3}(x/2)\,K_{1/3}(x/2)
        -
        \frac{3x}{10}
        \left(
            K_{4/3}^2(x/2) - K_{1/3}^2(x/2)
        \right)
    \right],

where :math:`K_\nu` is the modified Bessel function of the second kind.

The asymptotic behavior of this averaged kernel is

.. math::

    \bar{F}(x)
    \;\approx\;
    \frac{2^{1/3}}{5}\,\Gamma^2\!\left(\frac{1}{3}\right)\,x^{1/3}
    \quad\text{for } x \ll 1,
    \qquad\qquad
    \bar{F}(x)
    \;\approx\;
    \frac{\pi}{2}
    \left[
        1 - \frac{99}{162x}
    \right]
    e^{-x}
    \quad\text{for } x \gg 1.

An analytic approximation valid over the full domain :math:`x \in (0,\infty)`
was provided by :footcite:p:`2007A&A...465..695Z` and is given by

.. math::

    \bar{F}(x)
    \;\approx\;
    \frac{
        \left[
            \left(\frac{\pi x}{2}\right)^{s_1/2} + 1
        \right]^{1/s_1}
    }{
        \left[
            (0.893\,x^{1/3})^{-s_2} + 1
        \right]^{1/s_2}
    }
    e^{-x},

where :math:`s_1 = 1.430` and :math:`s_2 = 2.627`. This approximation
reproduces the exact kernel to better than one percent over the full
range of :math:`x`.

In Trilobite, the exact Bessel-function form is used in the intermediate
regime, while the asymptotic limits are applied at small and large
:math:`x` to ensure numerical stability and efficiency.



Computing Radiative Quantities
--------------------------------

To avoid repeated evaluation of the synchrotron kernels, Trilobite precomputes a table of these values on
a configurable grid of :math:`x` values. Then for any given frequency :math:`\nu`, magnetic field
strength :math:`B`, and pitch angle :math:`\alpha`,
the kernel can be interpolated rapidly to determine the correct value of :math:`F(x)` for each electron
Lorentz factor :math:`\gamma` in the distribution.
From this, we can rapidly evaluate various radiative quantities of interest.

.. _sec_emissivity:

Emissivity
^^^^^^^^^^^

With the kernel interpolator in hand, the emissivity integral

.. math::

    j_\nu = \chi\,B\sin\alpha
            \int_{\gamma_{\min}}^{\gamma_{\max}} F\!\left(\frac{\nu}{\nu_c}\right)\,N(\gamma)\, d\gamma,

where :math:`\chi = \sqrt{3} e^3 / (4\pi m_e c^2)`. We evaluate this integral by constructing a grid of
:math:`\gamma` values for each :math:`\nu`, computing :math:`x(\nu, \gamma)`, and then interpolating the kernel
at each point to build the integrand. The integral
is evaluated using a
Riemann sum in :math:`\ln\gamma`. On a grid :math:`\{\gamma_i\}` the change of variables
:math:`d\gamma = \gamma\, d\!\ln\gamma` gives

.. math::

    j_\nu \approx \chi\,B\sin\alpha \sum_{i} F\!\left(x_i\right)\,N(\gamma_i)\,\gamma_i\,\Delta\!\ln\gamma_i,

where :math:`x_i = \nu / (c_{1\gamma}\,B\sin\alpha\,\gamma_i^2)` and :math:`\Delta\!\ln\gamma_i` is the grid
spacing at point :math:`i` (computed via central differences). The product :math:`w_i \equiv \gamma_i\,\Delta\!\ln\gamma_i`
serves as the **quadrature weight** and is precomputed once for a given grid.

.. _sec_absorption:

Absorption
^^^^^^^^^^

The standard expression for the self-absorption coefficient,

.. math::

    \alpha_\nu = -\frac{\sqrt{3}\,e^3 B\sin\alpha}{8\pi m_e^2 c^2 \nu^2}
                 \int_{\gamma_{\min}}^{\gamma_{\max}} F\!\left(\frac{\nu}{\nu_c}\right)\,\gamma^2\,
                 \frac{d}{d\gamma}\!\left(\frac{N(\gamma)}{\gamma^2}\right) d\gamma,

involves the derivative of the quantity :math:`N(\gamma)/\gamma^2`. For a numerical distribution :math:`N(\gamma)`
given only on a discrete grid, computing this derivative reliably is awkward and introduces additional numerical
error. Trilobite avoids this difficulty entirely by applying an **integration by parts** (IBP) to transform the
integral into one that involves only :math:`N(\gamma)` itself and the log-derivative of the kernel.

Setting :math:`u = F(x)` and :math:`dv = \gamma^2\,\frac{d}{d\gamma}(N/\gamma^2) d\gamma`, and noting that the
boundary terms vanish for any physical distribution (since :math:`N \to 0` at both :math:`\gamma_{\min}` and
:math:`\gamma_{\max}`), the IBP yields

.. math::

    \int F(x)\,\gamma^2\,\frac{d}{d\gamma}\!\left(\frac{N}{\gamma^2}\right) d\gamma
    = -\int \frac{N(\gamma)}{\gamma^2}\,\frac{d}{d\gamma}\!\left[\gamma^2 F(x)\right] d\gamma.

The derivative :math:`d/d\gamma[\gamma^2 F(x)]` is evaluated by noting that
:math:`x = \nu / (c_{1\gamma} B\sin\alpha\,\gamma^2)`, so :math:`dx/d\gamma = -2x/\gamma`, giving

.. math::

    \frac{d}{d\gamma}\!\left[\gamma^2 F(x)\right]
    = 2\gamma F(x) + \gamma^2 F'(x)\,\frac{dx}{d\gamma}
    = 2\gamma F(x) - 2\gamma\, x F'(x)
    = 2\gamma\bigl(F(x) - x F'(x)\bigr).

Substituting back and converting to :math:`d\!\ln\gamma`, the integral becomes

.. math::

    \int F(x)\,\gamma^2\,\frac{d}{d\gamma}\!\left(\frac{N}{\gamma^2}\right) d\gamma
    = -2\int_{\gamma_{\min}}^{\gamma_{\max}} \bigl(F(x) - x F'(x)\bigr)\,N(\gamma)\, d\!\ln\gamma.

Inserting this into the expression for :math:`\alpha_\nu`, the factor of :math:`-2` cancels the overall minus sign
and converts the :math:`8\pi` denominator into :math:`4\pi`, so

.. math::

    |\alpha_\nu| = \frac{\chi}{m_e}\,\frac{B\sin\alpha}{\nu^2}
                   \int_{\gamma_{\min}}^{\gamma_{\max}} \bigl(F(x) - x F'(x)\bigr)\,N(\gamma)\, d\!\ln\gamma.

This form has a crucial property: the integrand is **guaranteed to be non-negative** for any physical distribution
:math:`N(\gamma) \geq 0`. To see this, write

.. math::

    F(x) - x F'(x) = F(x)\left(1 - \frac{d\ln F}{d\ln x}\right).

Since :math:`F(x) > 0` everywhere, it suffices to show that :math:`d\ln F/d\ln x < 1`. This holds because
:math:`F(x)` grows no faster than :math:`x^{1/3}` (log-slope :math:`1/3`) at small :math:`x` and falls
exponentially (log-slope :math:`\approx -x \ll 0`) at large :math:`x`, so the log-derivative is bounded above by
:math:`1/3 < 1` across the entire domain. A positive definite integrand means the sum can be evaluated directly in
log-space via :func:`~scipy.special.logsumexp` without any sign-handling — the same strategy used for the emissivity.

The practical evaluation is identical in structure to the emissivity: one builds the :math:`(n_\nu \times n_\gamma)`
matrix of :math:`x_{ij}` values, looks up :math:`\ln F(x_{ij})` and the derivative :math:`d\ln F/d\ln x\big|_{x_{ij}}`
from their respective splines, computes the IBP kernel

.. math::

    \ln\bigl(F(x_{ij}) - x_{ij} F'(x_{ij})\bigr)
    = \ln F(x_{ij}) + \ln\!\left(1 - \frac{d\ln F}{d\ln x}\bigg|_{x_{ij}}\right),

and then sums over :math:`\gamma` with :func:`~scipy.special.logsumexp`. The final result is

.. math::

    \ln|\alpha_\nu| = \ln\!\left(\frac{\chi}{m_e}\right) + \ln B + \ln\sin\alpha - 2\ln\nu
               + \mathrm{logsumexp}_i\!\left[\ln(F - xF')_{\nu i} + n_i + \tilde{w}_i - \ln\gamma_i\right],

where the additional :math:`-\ln\gamma_i` converts the quadrature weights :math:`w_i = \gamma_i\,\Delta\!\ln\gamma_i`
into pure :math:`d\!\ln\gamma` spacings :math:`\Delta\!\ln\gamma_i`, as required by the IBP form of the integral.

The Radiative Transfer Solution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the emissivity and absorption coefficient have been evaluated at each frequency, the specific intensity
emerging from a homogeneous, spherical source is given by the solution to the one-dimensional radiative transfer
equation:

.. math::

    I_\nu = \frac{j_\nu}{\alpha_\nu}\,\bigl(1 - e^{-\tau_\nu}\bigr) = S_\nu\,\bigl(1 - e^{-\tau_\nu}\bigr),

where :math:`S_\nu = j_\nu / \alpha_\nu` is the **source function** and :math:`\tau_\nu = |\alpha_\nu|\,R` is the
optical depth through a source of line-of-sight depth :math:`R`. This expression interpolates naturally between
the optically thin limit (:math:`\tau_\nu \ll 1`, where :math:`I_\nu \approx j_\nu R`) and the optically thick
Rayleigh-Jeans-like limit (:math:`\tau_\nu \gg 1`, where :math:`I_\nu \approx S_\nu`).

In log-space the radiative transfer solution reads

.. math::

    \ln I_\nu = \ln j_\nu - \ln|\alpha_\nu| + \ln\!\bigl(1 - e^{-\tau_\nu}\bigr),

where :math:`\tau_\nu = \exp(\ln|\alpha_\nu| + \ln R)`. The term :math:`\ln(1 - e^{-\tau})` is evaluated using
``numpy.expm1`` as :math:`\ln(-\mathrm{expm1}(-\tau))`, which avoids catastrophic cancellation for small
:math:`\tau` (where :math:`1 - e^{-\tau} \approx \tau` and a naive subtraction loses all significant digits). For
very large optical depths, :math:`\tau` is clipped before exponentiation to prevent overflow. The result is a
numerically stable expression for :math:`\ln I_\nu` that is accurate from the deeply optically thin to the deeply
optically thick regimes.

Relativistic Corrections
-------------------------

The formulas above are all written in the **rest frame of the emitting region**. For astrophysical transients,
the emitting plasma is frequently moving at relativistic bulk velocities, and the source may also be at a
cosmological redshift. Both effects must be accounted for before the computed specific intensity can be compared
with observed data.

Let :math:`\beta` be the bulk velocity of the emitting region as a fraction of :math:`c`, let :math:`\theta` be
the angle between the bulk velocity vector and the observer line-of-sight, and let :math:`z` be the cosmological
redshift of the source. The relativistic Doppler factor is

.. math::

    \mathcal{D} = \frac{1}{\Gamma_{\rm bulk}\,\bigl(1 - \beta\cos\theta\bigr)},

where :math:`\Gamma_{\rm bulk} = (1 - \beta^2)^{-1/2}` is the bulk Lorentz factor. The combined
Doppler-plus-redshift correction factor is

.. math::

    \mathcal{C} = \frac{\mathcal{D}}{1 + z} = \frac{1}{\Gamma_{\rm bulk}\,(1-\beta\cos\theta)\,(1+z)}.

The relationship between the **observed** frequency :math:`\nu_{\rm obs}` and the **rest-frame** frequency
:math:`\nu_{\rm rf}` is

.. math::

    \nu_{\rm rf} = \frac{\nu_{\rm obs}}{\mathcal{C}}.

The rest-frame specific intensity :math:`I_{\nu_{\rm rf}}` is computed at this transformed frequency using the
radiative transfer solution described above. The **observed** specific intensity then follows from the standard
Lorentz invariance of :math:`I_\nu / \nu^3`:

.. math::

    I_{\nu_{\rm obs}} = \mathcal{C}^3\,I_{\nu_{\rm rf}}\!\left(\nu_{\rm rf}\right).

In log-space the correction is simply an additive shift:

.. math::

    \ln I_{\nu_{\rm obs}} = \ln I_{\nu_{\rm rf}} + 3\ln\mathcal{C}.

.. note::

    There is **no** Equal Arrival Time Surface (EATS) correction in the current implementation. The source is
    treated as a homogeneous sphere at a single instant in the observer frame; the finite light-travel time across
    the source is not accounted for. For moderately relativistic sources (:math:`\Gamma \lesssim \mathrm{few}`)
    this is typically not the dominant source of error, but users modeling highly relativistic jets should
    be aware of this limitation.

Once the observed specific intensity :math:`I_{\nu_{\rm obs}}` has been computed, the spectral flux density at
the observer follows from integrating over the solid angle subtended by the source:

.. math::

    F_\nu = \frac{A_{\rm eff}}{D_A^2}\,I_{\nu_{\rm obs}},

where :math:`A_{\rm eff}` is the effective projected area of the source on the sky (defaulting to :math:`\pi R^2`
for a sphere) and :math:`D_A` is the angular diameter distance to the source. This is the quantity returned by
:meth:`~trilobite.radiation.synchrotron.SEDs.numerical.NumericalSynchrotronEngine.compute_flux_density`, which
handles distance resolution (including conversion from luminosity distance, proper distance, or redshift) via
:func:`~trilobite.physics_utils.cosmology.resolve_cosmological_distances`.

.. important::

    In the case of **ultra-relativistic outflows** (e.g. gamma-ray burst jets), extra care must be taken to account
    for differences in the light-travel time across the source, meaning that the observed emission at a given time may '
    not correspond to a single rest-frame snapshot of the source. In this model, we assume that the depth of the
    slab is **sufficiently small** that the light-travel time across it is negligible compared to the timescale of
    evolution of the
    source and therefore ignore EATS effects.

----

Classical Models
-----------------

The *classical* synchrotron SED models (see :ref:`synchrotron_seds`) are widely used throughout the literature as
a convenient, analytically tractable approach for modeling synchrotron emission from astrophysical sources. While these
models are a staple of modern analysis and are implemented in Trilobite, it is occasionally useful to produce analogous
models using the more accurate and flexible numerical methods described here. For this reason, a number of
these "classical models" are here described again in the context of our numerical engine.

In all of these models, we are concerned with emission from a homogeneous spherical region with radius :math:`R`,
magnetic field :math:`B`, and an electron distribution :math:`N(\gamma)`, which is usually chosen to be a power-law.
Additionally, the surface of the source may be expanding with bulk Lorentz-factor :math:`\Gamma_{\rm bulk}` and
the source may be at a cosmological redshift :math:`z`.

Tradiationally, (see :ref:`synch_sed_theory`) the SEDs in this scenario are computed analytically by means of
a fairly complex case-by-case analysis of the relative ordering of the characteristic frequencies
:math:`\nu_a`, :math:`\nu_m`, and :math:`\nu_c`. In contrast, the numerical engine described here computes
the SEDs by directly evaluating the emissivity and absorption coefficient at each frequency and then applying
the radiative transfer solution, without any assumptions about the relative ordering of the characteristic frequencies.
This means that the numerical SEDs are more accurate and flexible than the classical models, but they also
require more computational resources to evaluate.

The Non-Relativistic Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the non-relativistic limit, we consider a spherical region which appears on the sky as a disk with solid angle

.. math::

    \Omega = \frac{\pi R^2 f_A}{D_A^2},

where :math:`D_A` is the angular diameter distance to the source, and :math:`f_A` is a geometric factor which accounts for
the fact that the effective area of the source may be smaller than :math:`\pi R^2` (e.g. if the source is a
thin shell rather than a filled sphere).

As is done in the analytic case, we ignore limb-darkening effects and assume that every sightline through the
source has the same depth :math:`\ell`. To compute the depth :math:`\ell`, we assume that the source is a sphere of
radius :math:`R` with a volume filling factor :math:`f_V` (i.e. the emitting plasma fills a fraction :math:`f_V` of
the volume of the sphere). The depth is then given by

.. math::

    \ell = f_V^{1/3} R.

From this, the specific intensity at the surface of the source may be computed and used to determine
the flux density as described above. In the non-relativistic limit, we ignore bulk motion of the source.


The Ultra-Relativistic Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the ultra-relativistic limit the emitting shell expands with a bulk Lorentz factor :math:`\Gamma \gg 1`.
Relativistic beaming confines the observable emission to a forward cone of half-opening angle
:math:`\sim 1/\Gamma` around the line of sight. The model therefore replaces the full spherical surface
with the **beamed cap** — the small patch of the shell whose emission is directed toward the on-axis observer
— and neglects equal-arrival-time-surface (EATS) corrections, treating the entire cap as a single
homogeneous slab evaluated at one instant in the observer frame.

Under the beamed-cap approximation the effective projected area and comoving slab depth are

.. math::

    A_\mathrm{eff} = \frac{\pi R^2 f_A}{\Gamma^2}, \qquad
    \ell' = \frac{f_V R}{\Gamma},

where :math:`R` is the lab-frame blast-wave radius, :math:`f_A` and :math:`f_V` are the area and volume
filling factors respectively, and :math:`\Gamma` enters both expressions because the beaming solid angle
scales as :math:`1/\Gamma^2` and the comoving depth is Lorentz-contracted by one factor of :math:`\Gamma`.
The bulk velocity follows self-consistently from the Lorentz factor,

.. math::

    \beta = \sqrt{1 - \Gamma^{-2}},

and the observer is fixed on-axis (:math:`\theta = 0`). With :math:`\theta = 0` the relativistic Doppler
factor reduces to

.. math::

    \mathcal{D} = \frac{1}{\Gamma(1-\beta)} \approx 2\Gamma \quad (\Gamma \gg 1),

giving the combined correction factor

.. math::

    \mathcal{C} = \frac{\mathcal{D}}{1+z} \approx \frac{2\Gamma}{1+z}.

The comoving emissivity and absorption coefficient are computed from the effective depth :math:`\ell'`
using the same slab radiative transfer described above. The observed specific intensity then follows from
the standard relativistic transformation

.. math::

    I_{\nu_\mathrm{obs}} = \mathcal{C}^3\,I_{\nu_\mathrm{rf}}\!\left(\frac{\nu_\mathrm{obs}}{\mathcal{C}}\right),

and the flux density is obtained by integrating over the effective solid angle

.. math::

    F_\nu = \frac{A_\mathrm{eff}}{D_A^2}\,I_{\nu_\mathrm{obs}}
          = \frac{\pi R^2 f_A}{\Gamma^2 D_A^2}\,I_{\nu_\mathrm{obs}}.

.. important::

    The beamed-cap approximation is valid only while the ejecta are highly relativistic
    (:math:`\Gamma \gg 1`). As :math:`\Gamma` decreases toward unity the cap widens and the
    EATS corrections grow; for :math:`\Gamma \lesssim \mathrm{few}` the non-relativistic
    engine is the more appropriate choice. Additionally, the model ignores the finite
    light-travel time across the cap itself; see the note in the `Relativistic Corrections`_
    section for further discussion.

Composite Models
------------------

Having now described the physics of the one-zone slab model in detail, we may now consider how to use this formalism
to build more complex models. These models are referred to as **composite models** in the codebase, and they are built
by summing together multiple instances of the single-zone slab model described above.

The Inhomogeneous Cylinder Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One very simple instance of a composite model is that of the so-called **inhomogeneous cylinder**. In this model,
we consider an emitting region composed of **edge-on, nested cylinders** of varying radii, each with its own
electron distribution, depth, and magnetic field strength.

.. note::

    For clarity, we restrict the bulk-velocity in this case to be perpendicular to the line of sight, corresponding
    to expansion of the different shells. This will, of course, mean that for ultra-relativistic outflows, the observer
    will see a vanishingly small flux.

As before, for a specific radius :math:`r`, one can easily compute the specific intensity :math:`I_\nu(r)` as described
above. The corresponding flux density contribution from this radius is just

.. math::

    dF_\nu = \frac{2\pi r\,dr}{D_A^2}\,I_\nu(r),

and the total flux density is then given by integrating over the radius:

.. math::

    F_\nu = \int_0^R \frac{2\pi r}{D_A^2}\,I_\nu(r)\,dr.

By discretizing the radius into a set of :math:`n_r` annuli, this integral can be evaluated as a sum:

.. math::

    F_\nu \approx \sum_{j=1}^{n_r} \frac{2\pi r_j\,\Delta r_j}{D_A^2}\,I_\nu(r_j).

.. dropdown:: Algorithmic Details

    Algorithmically, we implement this model using a single grid of :math:`\gamma_i` values shared among all radii.
    The user may then supply a grid of radii, :math:`r_k`, along with values for the magnetic field :math:`B_k`,
    the electron distribution :math:`N_{ik} = N(\gamma_i, r_k)`, and the depth :math:`\ell_k` at each radius.

    We then compute the specific intensity :math:`I_\nu(r_k)` at each radius using the single-zone slab model
    described above, and sum over radii to get the total flux density.

The Aspherical Shell Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another simple case in which the composite model formalism is useful is that of an aspherical shell. Consider
an emitting surface such that, for any angular position relative to the observer :math:`(\theta, \phi)`, the depth
of the emitting region, its magnetic field strength, and its electron distribution are all
functions of :math:`\theta` and :math:`\phi`.

Assuming the observer is located at :math:`\theta = 0`, a region at :math:`(\theta, \phi)` will contribute a
surface element of area :math:`dA = R^2 \sin\theta\,d\theta\,d\phi` to the observed flux density, and the specific
intensity from this region can be computed using the single-zone slab model as described above. The total flux
density is then given by integrating over the surface of the shell:

.. math::

    F_\nu = \int_0^{2\pi} \int_0^\pi \frac{R^2 \sin\theta}{D_A^2}\,I_\nu(\theta, \phi)\,d\theta\,d\phi.

To be as efficient as possible, we recognize that this integral may be written as

.. math::

    F_\nu = \int_{-1}^{1} d\mu \int_0^{2\pi} \frac{R^2}{D_A^2}\,I_\nu(\mu, \phi)\,d\phi,

and we may therefore introduce Gauss-Legendre quadrature in :math:`\mu = \cos\theta` to evaluate the integral
over :math:`\theta` with high accuracy using a small number of points. The integral over :math:`\phi` is
then evaluated using a simple trapezoidal rule.

In cases where there is an axis of symmetry, the memory load of the algorithm may be reduced; however,
one still generally needs to compute the full 2D grid of :math:`I_\nu(\theta, \phi)` values to perform the integration,
so the computational cost is not significantly reduced.

The Axisymmetric Shell Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A minor modification of the above model is that of an **axisymmetric shell**, where the observer is on-axis with
the symmetry axis of the shell. In this case, the specific intensity is independent of :math:`\phi`, and the flux
density is given by

.. math::

    F_\nu = \int_0^\pi \frac{2\pi R^2 \sin\theta}{D_A^2}\,I_\nu(\theta)\,d\theta.

This integral can again be evaluated using Gauss-Legendre quadrature in :math:`\mu = \cos\theta` as described above.

.. _synch_off_axis_model:

The Off-Axis Axisymmetric Shell Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The on-axis restriction :math:`\theta_\mathrm{obs} = 0` is lifted in the **off-axis axisymmetric shell model**,
implemented by :class:`~trilobite.radiation.synchrotron.SEDs.numerical.aspherical.OffAxisAsymmetricSynchrotronEngine`.
The outflow retains axial symmetry — physical properties depend only on the zone polar angle
:math:`\theta_\mathrm{jet}` — but the observer may be located at an arbitrary polar angle
:math:`\theta_\mathrm{obs}` relative to the symmetry axis.

**Geometry.** For a zone at colatitude :math:`\theta_\mathrm{jet}` and azimuth :math:`\phi`, the cosine of the
angle between the zone velocity and the observer line of sight is

.. math::

    \mu_\mathrm{obs}(\mu, \phi)
    =
    \cos\theta_\mathrm{obs}\,\mu
    +
    \sin\theta_\mathrm{obs}\,\sqrt{1-\mu^2}\,\cos\phi,

where :math:`\mu = \cos\theta_\mathrm{jet}`. The relativistic Doppler factor for this zone is

.. math::

    \mathcal{D}(\mu, \phi)
    =
    \frac{1}{\Gamma\bigl(1 - \beta(\mu)\,\mu_\mathrm{obs}(\mu, \phi)\bigr)},

where :math:`\Gamma = (1-\beta^2)^{-1/2}` and :math:`\beta(\mu)` is the bulk speed at zone :math:`\mu`.

**Flux formula.** Each surface element contributes a projected area
:math:`R^2 \max(\mu_\mathrm{obs}, 0)\,d\mu\,d\phi` toward the observer. Back-facing zones
(:math:`\mu_\mathrm{obs} \le 0`) contribute no observed flux in the thin-shell approximation. The total
observed spectral flux density is

.. math::

    F_\nu
    =
    \frac{2}{D_A^2}
    \int_{-1}^{1}
    \int_0^{\pi}
    f_A(\mu)\,R(\mu)^2\,
    \max\!\bigl(\mu_\mathrm{obs},\,0\bigr)
    \left(\frac{\mathcal{D}(\mu,\phi)}{1+z}\right)^{\!3}
    I'_{\nu'(\mu,\phi)}(\mu)\,
    d\mu\,d\phi,

where the factor of two exploits the azimuthal symmetry of an axisymmetric outflow
(:math:`\phi \to 2\pi - \phi` leaves the integrand unchanged), reducing the azimuthal domain to
:math:`[0, \pi]`.

.. note::

    The full-sphere integration convention (:math:`\mu \in [-1,1]`) is used here. Back-facing material
    (:math:`\mu_\mathrm{obs} \le 0`) is suppressed by the :math:`\max(\mu_\mathrm{obs}, 0)` factor rather
    than being excluded from the domain. This is equivalent to excluding it, but avoids any dependence of
    the integration domain on :math:`\theta_\mathrm{obs}`.

**LOS path length.** For a thin shell of perpendicular comoving thickness :math:`\Delta r'(\mu)`, the
line-of-sight path length at zone :math:`(\mu, \phi)` is

.. math::

    \ell'(\mu, \phi)
    =
    \frac{\Delta r'(\mu)}{\max\!\bigl(\mu_\mathrm{obs}(\mu, \phi),\,\mu_\mathrm{min}\bigr)},

where :math:`\mu_\mathrm{min} = 10^{-6}` prevents division by zero at grazing incidence. At forward
angles the enhanced path length is exactly compensated by the vanishing projected-area factor, so
grazing zones contribute negligible flux.

.. admonition:: Convention: ``shell_thickness`` vs ``slab_depth``

    The parameter name differs between the on-axis and off-axis engines to reflect this geometric
    distinction. :class:`~trilobite.radiation.synchrotron.SEDs.numerical.aspherical.OnAxisAsymmetricSynchrotronEngine`
    accepts ``slab_depth``, the direct LOS depth per zone (the caller is responsible for any angular
    correction). :class:`~trilobite.radiation.synchrotron.SEDs.numerical.aspherical.OffAxisAsymmetricSynchrotronEngine`
    accepts ``shell_thickness``, the perpendicular depth :math:`\Delta r'`, and applies the
    :math:`1/\max(\mu_\mathrm{obs}, \mu_\mathrm{min})` correction internally.

**On-axis limit.** Setting :math:`\theta_\mathrm{obs} = 0` gives :math:`\mu_\mathrm{obs} = \mu`, so
back-facing zones (:math:`\mu < 0`) vanish and the :math:`\phi`-integral evaluates to :math:`\pi`,
recovering

.. math::

    F_\nu\big|_{\theta_\mathrm{obs}=0}
    =
    \frac{2\pi}{D_A^2}
    \int_0^1
    f_A(\mu)\,R(\mu)^2\,\mu\,
    \left(\frac{\mathcal{D}(\mu,0)}{1+z}\right)^3
    I'_{\nu'}(\mu)\,d\mu,

which matches the on-axis formula with the LOS-corrected slab depth :math:`\ell' = \Delta r' / \mu`.

**Quadrature.** Both the :math:`\mu` and :math:`\phi` integrals are evaluated by
Gauss--Legendre quadrature with :math:`n_\theta` and :math:`n_\phi` nodes respectively. The
:math:`\phi` loop is executed in Python over :math:`n_\phi` nodes; at each node the inherited
on-axis inner radiative-transfer kernel evaluates all :math:`n_\theta` zones simultaneously,
so memory is bounded to the on-axis footprint regardless of :math:`n_\phi`.


.. rubric:: References

.. footbibliography::
