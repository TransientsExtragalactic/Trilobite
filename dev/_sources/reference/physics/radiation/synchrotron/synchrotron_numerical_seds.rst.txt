.. _synch_numerical_sed_theory:
==============================================
Methods: Quadrature Based SEDs
==============================================

.. seealso::

    - :ref:`synch_sed_theory` for the methods used to compute analytical synchrotron SEDs.
    - :ref:`synch_numerical_sed` for the user guide for the numerical synchrotron code.

This document describes the numerical methods implemented in Triceratops for **computing synchrotron spectral energy
distributions (SEDs) from an arbitrary electron distribution**. The analytical library described in :ref:`synch_sed_theory`
covers a wide range of important cases, but there are many physically interesting scenarios for which a closed-form
expression for the SED is either unavailable or impractical to derive. These include:

- **Arbitrary electron distributions**, such as thermal (Maxwell-Jüttner), broken power-laws with non-trivial cooling,
  or numerically evolved particle spectra. The analytical library is restricted to simple power-law forms.
- **Multi-component sources**, where several distinct emission regions contribute to the observed SED and each must be
  evaluated independently before being combined.
- **Heterogeneous magnetic field structures**, where the field varies across the source and a simple single-zone
  treatment is insufficient.

In these cases, a robust numerical method is needed. The approach implemented in Triceratops is based on direct
quadrature of the fundamental synchrotron integrals, performed in log-space for numerical stability. The central
objective of this note is to document the mathematics and algorithmic choices behind that implementation, which lives
in :mod:`triceratops.radiation.synchrotron.SEDs.numerical`.

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
For a population of relativistic electrons with number density :math:`dN/d\gamma = N(\gamma)` spiraling in a magnetic
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

The following sections describe the methods by which Triceratops achieves these calculations with a high
degree of fidelity and numerical stability, including the handling of asymptotic regimes, the use of log-space quadrature,
and the application of integration by parts to simplify the absorption coefficient integral.

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
domain, Triceratops uses the exact Bessel integral for intermediate :math:`x` values and switches to
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

In Triceratops, the exact Bessel-function form is used in the intermediate
regime, while the asymptotic limits are applied at small and large
:math:`x` to ensure numerical stability and efficiency.

.. _sec_emissivity:

Computing the Emissivity
-------------------------

To avoid repeated evaluation of the synchrotron kernels, Triceratops precomputes a table of these values on
a configurable grid of :math:`x` values. Then for any given frequency :math:`\nu`, magnetic field strength :math:`B`, and pitch angle :math:`\alpha`,
the kernel can be interpolated rapidly to determine the correct value of :math:`F(x)` for each electron Lorentz factor :math:`\gamma` in the distribution.

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

Computing the Absorption Coefficient
-------------------------------------

The standard expression for the self-absorption coefficient,

.. math::

    \alpha_\nu = -\frac{\sqrt{3}\,e^3 B\sin\alpha}{8\pi m_e^2 c^2 \nu^2}
                 \int_{\gamma_{\min}}^{\gamma_{\max}} F\!\left(\frac{\nu}{\nu_c}\right)\,\gamma^2\,
                 \frac{d}{d\gamma}\!\left(\frac{N(\gamma)}{\gamma^2}\right) d\gamma,

involves the derivative of the quantity :math:`N(\gamma)/\gamma^2`. For a numerical distribution :math:`N(\gamma)`
given only on a discrete grid, computing this derivative reliably is awkward and introduces additional numerical
error. Triceratops avoids this difficulty entirely by applying an **integration by parts** (IBP) to transform the
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
--------------------------------

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
:func:`numpy.expm1` as :math:`\ln(-\mathrm{expm1}(-\tau))`, which avoids catastrophic cancellation for small
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

    \mathcal{D} = \frac{1}{\Gamma_{\rm bulk}\,\bigl(1 + \beta\cos\theta\bigr)},

where :math:`\Gamma_{\rm bulk} = (1 - \beta^2)^{-1/2}` is the bulk Lorentz factor. The combined
Doppler-plus-redshift correction factor is

.. math::

    \mathcal{C} = \frac{\mathcal{D}}{1 + z} = \frac{1}{\Gamma_{\rm bulk}\,(1+\beta\cos\theta)\,(1+z)}.

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
:meth:`~triceratops.radiation.synchrotron.SEDs.numerical.NumericalSynchrotronEngine.compute_flux_density`, which
handles distance resolution (including conversion from luminosity distance, proper distance, or redshift) via
:func:`~triceratops.physics_utils.resolve_cosmological_distances`.

.. rubric:: References

.. footbibliography::
