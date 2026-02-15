.. _synch_sed_theory:
===========================================
Theory of Synchrotron SEDs
===========================================

Having established the foundations of synchrotron radiation theory in :ref:`synchrotron_theory` this document is
intended to develop the theory behind Triceratops' implementation of **synchrotron SEDs**. This is a critical task
on the basis that the literature, spanning some 40 years at this point, is highly fractured in its methodology concerning
these SEDs and their construction (see e.g. :footcite:t:`Chevalier1998SynchrotronSelfAbsorption`, :footcite:t:`ChevalierFranssonHandbook`,
:footcite:t:`GranotSari2002SpectralBreaks`, :footcite:t:`GaoSynchrotronReview2013`, :footcite:t:`2025ApJ...992L..18S`, and
references therein). Our goal in this presentation is to (a) provide a background to users who are not familiar with
the details of this theory and, more importantly, (b) to establish our methodology in as robust a manner as possible ensuring
that Triceratops remains extensible, reproducible, and accurate.

.. contents::

Overview
--------

.. note::

    For readers unfamiliar with elementary theory of synchrotron radiation, it is worthwhile to
    first read :ref:`synchrotron_theory` before proceeding.

In general, the SEDs produced by synchrotron emission from transients are well described by **broken power-law** profiles.
More precisely, **smoothed broken power-laws** have been determined to be a well suited option for interpolating between
the standard power-law regimes of each SED.

For any given scenario, the SED is characterized by a set of **break frequencies** :math:`(\nu_1,\nu_2,\ldots)` between
which the SED follows standard asymptotic behaviors characterized by a set of spectral slopes :math:`(\alpha_{1,2},
\alpha_{2,3},\ldots)`. As described in :ref:`synchrotron_sed_methods`, we consider a total of 4 break frequencies in
Triceratops:

- The **minimum injection frequency** :math:`\nu_m` determined by the characteristic synchrotron frequency of the lowest energy
  electrons injected by the shock.
- The **maximum injection frequency** :math:`\nu_{\rm max}` determined by the synchrotron frequency of the most energetic
  electrons injected by the shock.
- The **cooling frequency** :math:`\nu_c` corresponding to the frequency at which cooling is efficient enough to have
  lead to significant cooling within the dynamical time.
- The **self absorption frequency** :math:`\nu_a` determined by the frequency at which the optical depth to self-absorption
  is unity.

.. hint::

    It's not always the case that one wants to use SEDs which consider **all** of these breaks. We therefore implement
    various combinations in Triceratops as well (i.e. SSA but no cooling or cooling but no SSA). Likewise, because the
    maximum injection frequency is often irrelevant, SEDs without it are available.

In order to determine the SED for a particular scenario, one needs to, self-consistently,

1. Determine the **ordering of the relevant frequencies**,
2. **Identify the correct SED** based on the ordering,
3. **Normalize the SED** based on the emission geometry,
4. Calculate the SED.

As mentioned above, there are a great many implementations of this general scheme with significant variety in the
exact methodology for elements like the calculation of the break frequencies, normalization, etc. In most cases,
these methods are consistent with one another up to an order of unity.


.. _synchrotron_sed_methods:
Methodology
-----------

We now begin our discussion of the methodology used in our construction of the synchrotron SEDs. In the subsections
below, we will describe in detail each element of SED construction; however, for those familiar with the literature, it
should be noted that we follow the formulation of :footcite:t:`GranotSari2002SpectralBreaks` to construct our SEDs and
to describe the various power-law components. Because we include an additional break frequency (:math:`\nu_{\rm max}`) a
few additional scenarios are considered here which are not therein mentioned.

Because :footcite:t:`GranotSari2002SpectralBreaks` use a methodology for normalization which is not fully generalizable without
numerical quadrature, we choose instead to follow the approximations described in :footcite:t:`sari1999jets` and used throughout
the literature (see e.g. :footcite:t:`2025ApJ...992L..18S`).

Parameters, Hyper-Parameters, and Physical Quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This document contains a great deal of mathematical notation describing the various SEDs and their construction. It is
therefore worthwhile to provide a **bird's-eye view** of the various types of quantities used herein.

At it's core, an **SED** is a function :math:`F(\nu, F_{\rm norm}, \boldsymbol{\nu}_{\rm brk}, \boldsymbol{\Theta})` which
provides the **flux density** :math:`F_\nu` at a given frequency :math:`\nu` given a set of **break frequencies**
:math:`\boldsymbol{\nu}_{\rm brk} = (\nu_1, \nu_2, \ldots)`, a normalization, and a set of
**hyper-parameters** :math:`\boldsymbol{\Theta} = (\theta_1, \theta_2, \ldots)`.

- The **break frequencies** are the critical frequencies which define the transitions between different
  power-law segments in the SED. These are, in general, functions of the underlying **physical parameters**
  of the system such as magnetic field strength, electron distribution properties, etc.

  .. important::

        Throughout this document, we refer to the observer-frame quantities without a prime and to
        the rest-frame quantities with a prime.

- The **hyper-parameters** are parameters which define the shape of the SED but are not directly
  related to the physical parameters of the system. Examples include the electron power-law index
  :math:`p` and the smoothness parameters :math:`s_{(i,j)}` which define the sharpness of transitions
  between power-law segments. This also includes parameters such as the pitch angle :math:`\alpha`,
  the filling factor :math:`f`, etc. These may **also** be functions of the underlying physical parameters.

In addition to the **break frequencies** and **hyper-parameters**, there may be **internal parameters** which are
self-consistently determined from the other parameters. An example of this is the self-absorption frequency
:math:`\nu_a` which is, in general, a function of the other break frequencies and hyper-parameters.

In the tables below, we summarize the various types of quantities used in this document:

.. rubric:: SED Parameters

.. list-table::
    :widths: 20 50
    :header-rows: 1
    :name: sed_parameter_table

    * - Parameter
      - Description
    * - :math:`F_{\rm norm}`
      - The normalization of the SED. This is taken to be the expected flux of the dominant power-law
        segment at the dominant break frequency. See :ref:`sed_normalization` for details.
    * - :math:`\nu_m`
      - The minimum injection frequency. See :ref:`synchrotron_sed_injection_frequencies` for details.
    * - :math:`\nu_{\rm max}`
      - The maximum injection frequency. See :ref:`synchrotron_sed_injection_frequencies` for details.
    * - :math:`\nu_c`
      - The cooling frequency. See :ref:`synchrotron_sed_cooling_frequency` for details.

.. rubric:: SED Hyper-Parameters

.. list-table::
    :widths: 20 50
    :header-rows: 1
    :name: sed_hyperparameter_table

    * - Hyper-Parameter
      - Description
    * - :math:`p`
      - The electron power-law index.
    * - :math:`s_{(i,j)}`
      - The smoothness parameter between power-law segments i and j. In practice, we use a single smoothness
        parameter for all breaks in a given SED.
    * - :math:`\alpha`
      - The pitch angle between the electron velocity and the magnetic field. This is only used when fixed pitch
        angle synchrotron emission is desired.
    * - :math:`\Omega`
      - The effective angular radiating area of the source. This is only used when SSA is included in the SED.
        See :ref:`synchrotron_abs_frequency` for details.
    * - :math:`\gamma_{\min}`
      - The minimum electron Lorentz factor in the power-law distribution.
    * - :math:`\Gamma_{\rm bulk}`
      - The bulk Lorentz factor of the outflow.
    * - :math:`z`
      - The redshift of the source.

.. rubric:: SED Internal Parameters

.. list-table::
    :widths: 50 50
    :header-rows: 1
    :name: sed_internalparameter_table

    * - Internal Parameter
      - Description
    * - :math:`\nu_a`
      - The self-absorption frequency. See :ref:`synchrotron_abs_frequency` for details.
    * - :math:`F_{\rm pk}`
      - The peak flux density of the SED. This is determined by the normalization and the
        break frequencies. See :ref:`sed_normalization` for details.

.. _sed_surgery:
The Shape of SEDs
^^^^^^^^^^^^^^^^^

*Note*: Unless specifically noted, we describe quantities in the comoving frame of the emitting source.

To begin, we introduce the formal notation used to describe each of the SEDs which is to be constructed in
this document. We present (and implement in the code) two versions of any given SED:

- The **Smoothed** SED, which uses smoothed broken power laws,
- The **Discrete** SED, which uses piecewise defined power laws.

This is done to allow for easy comparison with the literature since both are used. In either case, the normalizations
of the SEDs and the positions of the breaks are the same.

To be precise, and to avoid confusion in our derivations below, we adopt a few standard notations. First,
the flux density between **any two adjacent regions** (i.e., power-law segments [PLSs]) will be connected
using a **smoothly broken power-law** (SBPL) of the form:

.. math::

    F_{\nu}^{(1,2)} = F_{\nu,0}^{(1,2)} 2^{-s_{(1,2)}} \left[
        \left(\frac{\nu}{\nu_{brk}}\right)^{\alpha_1/s_{(1,2)}} +
        \left(\frac{\nu}{\nu_{brk}}\right)^{\alpha_2/s_{(1,2)}}
    \right]^{s_{(1,2)}},

where :math:`\alpha_1` and :math:`\alpha_2` are the spectral indices in the two regions, :math:`\nu_{brk}` is the break
frequency between them, :math:`F_{\nu,0}^{(1,2)}` is the normalization constant for the broken
power-law, and :math:`s_{(1,2)}` is the smoothness parameter that controls the sharpness of the transition.

We likewise define the **scale-free** SBPL between two adjacent regions as:

.. math::

    \tilde{F}_{\nu}^{(1,2)} = 2^{-s_{(1,2)}}\left[
        1 +
        \left(\frac{\nu}{\nu_{brk}}\right)^{(\alpha_2-\alpha_1)/s_{(1,2)}}
    \right]^{s_{(1,2)}}.

Once each segment has been identified, we can then construct the resulting smoothed SED by multiplying a single
scaled SED segment with a number of other *scaled* SED segments in a process we call **SED surgery**.
We may be precise about our notion of **SED surgery** by recognizing that a spectrum composed of
multiple power-law segments may be constructed by multiplying together the scale-free SBPLs
between each adjacent pair of regions and then normalizing the entire SED with a single flux scale. Thus, for
a break :math:`\nu_0` with known flux normalization :math:`F_{\nu,0}`, and additional breaks
at :math:`\nu_1, \nu_2, \ldots, \nu_n`, the full SED may be written as:

.. math::
    :label: full_sed_surgery

    F_\nu = F_{\nu,0} \prod_{i=0}^{n-1} \tilde{F}_{\nu}^{(i,i+1)}.

.. important::

    In the :mod:`~radiation.synchrotron.SEDs` module, each of these SED surgery products is implemented
    as a single function in the low-level API. In that case, we universally require that the curve be unity
    at its maximum (corresponding to one of the break frequencies) in the limit that :math:`|s_{(i,j)}| \to 0`
    (i.e., the discrete limit). This allows for a single normalization to be applied to the entire SED.

    In the high-level Object-Oriented API, we instead use the :math:`F_{\rm norm}` convention described in
    :ref:`sed_normalization` to normalize the SEDs. This is because self-consistent solutions for the internal
    parameters are not guaranteed with phenomenological normalization.

Break Frequencies
^^^^^^^^^^^^^^^^^

Before proceeding to discuss the construction of various SEDs, it is necessary to describe the precise methodology
with which we compute the various break frequencies used in the SEDs. In each of the following sections, we describe
this methodology in detail.

.. hint::

    In :ref:`synchrotron_seds`, we describe the actual code implementation of all of the SEDs. It is worth noting that,
    in general, we provide implementations for SEDs in terms of the relevant frequencies so that, should one wish to
    do so, any prescription for computing the break frequencies may be used.

.. note::

    In this section, as is our convention throughout, the rest-frame quantities are labeled with a prime.

.. _synchrotron_sed_injection_frequencies:
The Injection Frequencies
~~~~~~~~~~~~~~~~~~~~~~~~~

**PARAMETER TYPE:** Free Parameter

The injection frequencies are a function of the detailed microphysics of shock acceleration and must, in general,
be prescribed by some downstream model of that physics. In the context of SED construction, the critical one should,
in general, use whichever physical prescription is most appropriate to compute :math:`\gamma_{\rm min}` and
:math:`\gamma_{\rm max}`.

Once these Lorentz factors are know, the **injection frequencies** are fixed up to a factor for the treatment of
pitch angle :math:`\alpha` between the electron velocity and the magnetic field. As we have done elsewhere, we support
both the isotropic pitch angle averaged case and the fixed pitch angle case. In each case, the (rest-frame) injection frequencies
are as follows:

.. tab-set::

    .. tab-item:: Pitch-Averaged

        In the case of an isotropic pitch angle distribution, our convention for the pitch averaged break frequency is
        to use the average synchrotron frequency given by :eq:`eq_synch_frequency` from :ref:`synchrotron_theory`. Thus,

        .. math::
            :label: injection_frequency_min_iso

            \nu_m' = \left<\nu'_{\rm synch}(\gamma')\right>_{\sin \alpha} = \frac{3 q B'}{2 \pi^2 m c}\left[\gamma'_{\min}\right]^2,

        In the same manner, the maximum injection frequency is

        .. math::

            \nu_{\rm max}' = \left<\nu'_{\rm synch}(\gamma')\right>_{\sin \alpha} = \frac{3 q B'}{2 \pi^2 m c}\left[\gamma'_{\max}\right]^2.

    .. tab-item:: Fixed Pitch Angle

        In the case of a fixed pitch angle :math:`\alpha`, the injection frequencies are given by the standard
        characteristic synchrotron frequency from :eq:`eq_synch_frequency` from :ref:`synchrotron_theory`:

        .. math::
            :label: injection_frequency_min

            \nu_m' = \nu'_{\rm char}(\gamma'_{\min}) = \frac{3 q B' \sin \alpha}{4 \pi m c}\left[\gamma'_{\min}\right]^2,

        and

        .. math::

            \nu_{\rm max}' = \nu_{\rm char}'(\gamma'_{\max}) = \frac{3 q B' \sin \alpha}{4 \pi m c}\left[\gamma'_{\max}\right]^2.

To convert the rest-frame quantity to the observed quantity, one must apply the corresponding Doppler correction

.. math::

    \delta = \Gamma_{\rm bulk}(1+\beta_{\rm bulk}),

so that the observed injection frequencies are

.. math::

    \nu_{m} = \frac{\delta}{1+z} \nu_m', \quad \nu_{\rm max} =  \frac{\delta}{1+z} \nu'_{\rm max}.

In practice, the minimum injection frequency is typically treated as a free parameter in a model and fit for, while
the maximum injection frequency is often neglected entirely due to its location at very high frequencies.

.. hint::

    It is somewhat confusing that :math:`\nu_m` is a **free parameter**, but :math:`\gamma_{\min}` is a
    **hyper-parameter**. The difference between these two quantities is a factor of :math:`B`, the magnetic field
    strength. Generally, :math:`\gamma_{\rm min}` (or :math:`\gamma_{\rm max}`) is modeled *a priori* from shock
    acceleration theory, while :math:`B` is determined from the overall energetics of the system and is therefore
    treated as a physical parameter. Because :math:`\nu_m` depends on both, it is treated as a free parameter.

.. admonition:: Convention Note

    This particular quantity is subject to a wide range of conventions in the literature (see e.g.
    :footcite:t:`GranotSari2002SpectralBreaks`, :footcite:t:`GaoSynchrotronReview2013`, :footcite:t:`2025ApJ...992L..18S`
    :footcite:t:`demarchiRadioAnalysisSN2004C2022`). We have here taken the route of defining the injection frequencies in
    a manner most in keeping with the general theory of synchrotron radiation as described in :ref:`synchrotron_theory`.

.. _synchrotron_sed_cooling_frequency:
The Cooling Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~

**PARAMETER TYPE:** Free Parameter

Consider a population of electrons subject to a cooling process with a cooling rate :math:`\Lambda'(\gamma')`. Electrons
with energy :math:`E' = m_e c^2 \gamma'` will then cool on a timescale (in the comoving frame)

.. math::

    t'_{\rm cool}(\gamma) = \frac{E'}{\Lambda'(\gamma')} = \frac{m_e c^2 \gamma'}{\Lambda'(\gamma')}.

If, in order to cool significantly, the dynamical time must exceed the cooling time for a particular energy, we can
define the **cooling Lorentz factor** :math:`\gamma_c'` as the Lorentz factor for which the cooling timescale
equals the dynamical timescale :math:`t'_{\rm dyn}` of the system:

.. math::

    \boxed{
    t'_{\rm cool}(\gamma_c')
    =
    \frac{m_e c^2 \gamma_c'}{\Lambda'(\gamma_c')}
    =
    t'_{\rm dyn}.
    }

This then implies

.. math::
    :label: cooling_lorentz_factor

    \boxed{
    \gamma_c'
    =
    \frac{m_e c^2}{\Lambda'(\gamma_c') t'_{\rm dyn}}.
    }

The corresponding **cooling frequency** is then given by :eq:`eq_synch_frequency` from :ref:`synchrotron_theory` as

.. math::
    :label: cooling_frequency

    \boxed{
    \nu_c'
    =
    \frac{3 q B' \sin \alpha}{4 \pi m c} \gamma_c'^2
    =
    \frac{3 q m_e c^3}{4\pi}
    \left(
    \frac{B' \sin \alpha}
    {\left[\Lambda'(\gamma_c')\right]^2\, t_{\rm dyn}'^{\,2}}
    \right).
    }

In the observer frame, one factor of :math:`\delta/(1+z)` is applied to covert the frequency into the observer frame;
however, time dilation of the dynamical time modifies the frequency as well. Given a dynamical time in the observer
frame, the corresponding rest-frame dynamical time is

.. math::

    t'_{\rm dyn} = t_{\rm dyn} \frac{\delta}{1+z},

meaning that the cooling frequency is defined in the observer frame as

.. math::
    :label: cooling_frequency_observer

    \boxed{
    \nu_c = \left(\frac{1+z}{\delta}\right) \frac{3 q m_e c^3}{4 \pi}
    \left[\frac{B'\sin\alpha}{\left[\Lambda'(\gamma_c')\right]^2 t_{\rm dyn}^2}\right]
    }

The precise value of the cooling frequency should be determined from the dominant cooling process affecting the electron
population. In most astrophysical synchrotron sources, the most important cooling processes are synchrotron cooling
and inverse Compton cooling.

One can likewise consider the case where multiple cooling processes are acting simultaneously. In such a case, the total
cooling rate is the sum of the individual cooling rates and the formalism above applies similarly. :eq:`cooling_frequency`
is the homogeneous pitch-angle averaged cooling frequency; however, one may also use the pitch-averaged cooling
frequency as

.. math::

    \boxed{
    \nu_c = \left(\frac{1+z}{\delta}\right) \frac{3 q m_e c^3}{2 \pi^2}
    \left[\frac{B'\sin\alpha}{\left[\Lambda'(\gamma_c')\right]^2 t_{\rm dyn}^2}\right]
    }

.. hint::

    In general, one selects a cooling mechanism (see :ref:`synchrotron_theory` and :ref:`synchrotron_microphysics`)
    and computes :math:`\gamma_c(t)` as a function of time and provides that to the SED. Cooling is implemented in
    :mod:`~radiation.synchrotron.cooling`.

.. _synchrotron_abs_frequency:
The Absorption Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~

**PARAMETER TYPE:** Internal Parameter

The self-absorption frequency is a less trivial quantity to compute, as it depends on the radiative transfer
properties of the source and is therefore dependent on the SED one is using. This creates a circular dependency
which must be resolved by considering every possible SED configuration given a known :math:`\nu'_m` and :math:`\nu'_c`,
computing the value of :math:`\nu'_a` in each case and then checking for self-consistency with the assumed SED and
its assumptions.

In the most rigorous sense, the absorption frequency :math:`\nu'_a` is determined by the condition that the
optical depth to self-absorption equals unity:

.. math::

    \tau_{\nu'_{a}} = \alpha_{\nu'_a} L = 1,

The form of :math:`\alpha_\nu` depends explicitly on the structure of the absorbing electron population (see
:ref:`synchrotron_theory` for details). One could, in principle, perform these computations in full detail; however,
an alternative approach has been developed in the literature :footcite:p:`duran2013radius` which allows for
approximate expressions for :math:`\nu'_a`.

We assume, as was done in the development of the normalization approach, that the absorption at a particular frequency
is dominated by a mono-energetic population of electrons. In such a case, the optically thick emission from the source
should be well approximated by a blackbody with brightness temperature :math:`kT_{b} = \gamma'_\nu m_e c^2`, where
:math:`\gamma'_\nu` is the Lorentz factor corresponding to the dominant absorbing population of electrons:

.. math::

    \gamma'_\nu = {\rm max}\left(\gamma'_a, {\rm min}\left(\gamma'_c,\gamma'_m\right)\right).

This corresponds to a source function :math:`S'_\nu = 2\left[\nu_a'\right]^2 m_e \gamma'_\nu`. The corresponding flux
:math:`F'_\nu` should then be

.. math::

    F'_\nu = 2\left[\nu'_a\right]^2 m_e \gamma'_\nu \frac{A}{D_A^2},

where :math:`A` is the effective radiating area of the source. Equating this to the **optically thin** flux
from the normalized SED at :math:`\nu'_a` then allows one to solve for :math:`\nu'_a`. In practice, we instead parameterize
this flux density in terms of the **effective angular radiating area** :math:`\Omega = A/D_A^2` so that the distance
does not need to be provided as a hyper-parameter.

.. important::

    For code users, it is important to note that while the inclusion of :math:`\nu'_a` would seem to be similar
    in complexity to the other breaks, using an SSA enabled SED **REQUIRES** the user to provide some elements of
    the underlying geometry of the source (i.e., the effective area and volume) in order to compute :math:`\nu'_a`
    in each dependent scenario.

    Thus, an SSA SED function :math:`F'_{\rm \nu}(\nu'; \nu'_m,\nu'_c,\nu'_a,\ldots)` should be thought of more properly
    as

    .. math::

        F'_{\rm \nu}(\nu'; \nu'_m,\nu'_c,A,V,\ldots),

    where now :math:`\nu'_a = f(\nu'_m,\nu'_c,A,V,\ldots)` is computed internally.

Computing The Self-Absorption Frequency
#######################################

In general, we assume that, at :math:`\nu_a` (an unknown break), the flux density is given by the blackbody
approximation described above:

.. math::

    F'_\nu = 2\left[\nu'_a\right]^2 m_e \gamma'_\nu \Omega = F_{\nu', \rm thin}(\nu_a),

where :math:`F_{\nu', \rm thin}(\nu'_a)` is the flux density at :math:`\nu'_a` computed from the optically thin SED,
and :math:`\gamma_{\nu'}` is the Lorentz factor of the **dominant absorbing electrons**:

.. math::

    \gamma'_\nu = {\rm max}\left(\gamma'_a, {\rm min}\left(\gamma'_c,\gamma'_m\right)\right).

This defines an implicit equation for :math:`\nu'_a` which may be solved algebraically (or numerically if necessary).
Thus, for any set of break frequencies and hyper-parameters, one may compute :math:`\nu'_a` by solving the equation
and then construct the SED using the computed frequency.

----

The Single Electron SED
-----------------------

.. hint::

    The single electron SED can be found in the :mod:`radiation.synchrotron.core` module.

.. note::

    In this case, it is effectively non-sensical to describe a volume emitting flux or intensity. We therefore
    simply describe the **spectral power density** (i.e., power per unit frequency) :math:`P(\nu)` of a single electron.

Let us now describe the simplest synchrotron SED: that of a single electron. As described above, the
emission from a single electron is characterized by the synchrotron kernel function
:math:`F(x)`, where :math:`x = \nu/\nu_{\rm char}`. The resulting SED takes the form:

.. math::

    \boxed{
    P(\nu) = \frac{\sqrt{3}q^3 B \sin\alpha}{m c^2} F\left(\frac{\nu}{\nu_{\rm char}}\right),
    }

Importantly, there are two asymptotic regimes of the single-electron SED:

The Low Frequency Regime
^^^^^^^^^^^^^^^^^^^^^^^^

In the low-frequency regime, the synchrotron kernel takes the form

.. math::

    F(x) \approx \frac{4\pi}{\sqrt{3}\,\Gamma\left(\frac{1}{3}\right)} \left(\frac{x}{2}\right)^{1/3} \propto x^{1/3}.

As such, the corresponding SED takes the form

.. math::

    P(\nu) \approx \frac{4\pi q^3}{m_e c^2} \left(B\sin\alpha\right)
    \Gamma\left(\frac{1}{3}\right)^{-1} \left(\frac{\nu}{2\nu_{\rm char}}\right)^{1/3} \propto \nu^{1/3}.

The High Frequency Regime
^^^^^^^^^^^^^^^^^^^^^^^^^

In the high-frequency regime, the synchrotron kernel takes the form

.. math::

    F(x) \approx \sqrt{\frac{\pi x}{2}} e^{-x} \propto x^{1/2} e^{-x}.

As such, the corresponding SED takes the form

.. math::

    P(\nu) \approx \frac{\sqrt{3}\pi^{1/2} q^3}{\sqrt{2} m_e c^2} \left(B\sin\alpha\right)
    \left(\frac{\nu}{\nu_{\rm char}}\right)^{1/2} e^{-\nu/\nu_{\rm char}} \propto \nu^{1/2} e^{-\nu/\nu_{\rm char}}.

At high frequencies, the SED exhibits an exponential cutoff beyond the characteristic frequency
:math:`\nu_{\rm char}`.


Power Law Synchrotron SEDs
---------------------------
Having now established all of the relevant theory and mathematical tools, we are
finally in a position to derive the classic synchrotron SEDs used in astrophysical
modeling. In the sections that follow, we will derive each of the standard synchrotron SED regimes
and describe the normalization in each case.

Asymptotic Regimes
^^^^^^^^^^^^^^^^^^

Because various broadband SEDs have segments with the same SPL slopes, we begin by listing the different
possible SPL segments that can arise in synchrotron SEDs from power-law electron distributions. This follows
the naming convention of :footcite:t:`GranotSari2002SpectralBreaks`.

.. tab-set::

    .. tab-item:: SPL A (:math:`F_\nu \propto \nu^{5/2}`)

        SPL A occurs in the optically thick regime below the self-absorption frequency :math:`\nu_a`, but above
        the minimum electron frequency :math:`\nu_m`. In this regime, the SED is dominated by self-absorbed synchrotron
        emission from the full power-law distribution of electrons. The resulting SED is derived in most references
        in synchrotron emission, (e.g. :footcite:t:`RybickiLightman`, Chapter 6; :footcite:t:`1970ranp.book.....P`).

    .. tab-item:: SPL B (:math:`F_\nu \propto \nu^{2}`)

        SPL B occurs in the optically thick regime below both the self-absorption frequency :math:`\nu_a` and
        the minimum electron frequency :math:`\nu_m`. In this regime, the SED is dominated by self-absorbed synchrotron
        emission from the low-energy tail of the single-electron SED. This results in a characteristic spectral slope
        of :math:`F_\nu \propto \nu^{2}`.

        More precisely, the absorption coefficient to synchrotron self-abortion is
        (:eq:`eq:ssa_exact` from :ref:`synchrotron_theory`):

        .. math::

            \alpha_\nu
            =
            -\frac{1}{8\pi m_e \nu^2}
            \int_{\gamma_{\min}}^{\gamma_{\max}}
            P(\nu,\gamma)\,
            \gamma^2
            \frac{\partial}{\partial\gamma}
            \left[
                \frac{1}{\gamma^2}
                \frac{dN}{d\gamma}
            \right]
            d\gamma.

        In the low-frequency limit, the integral is dominated by the lowest energy electrons in the population and
        the frequency is in the low-frequency tail of those single-electron SEDs. In that case, the entire integral term
        scales as :math:`\nu^{1/3}` and the absorption coefficient scales as :math:`\alpha_\nu \propto \nu^{-5/3}`. The
        emissivity in this case scales as :math:`j_\nu \propto \nu^{1/3}` as well. The source function is then
        :math:`S_\nu = j_\nu/\alpha_\nu \propto \nu^{2}`, leading to the characteristic SPL B slope.

    .. tab-item:: SPL C (:math:`F_\nu \propto \nu^{11/8}`)

        SPL C is a special case first identified in :footcite:t:`2000ApJ...534L.163G` in the context of GRB afterglows.
        It arises in scenarios with fast cooling such that the post-shock electrons are able to cool very rapidly
        compared to the dynamical timescale. This leads to a stratified structure in the source, with a thin layer
        of uncooled electrons near the shock front and a larger volume of cooled electrons behind them. In this case,
        the self-absorption is dominated by the cooler, downstream electrons, leading to a modified self-absorption
        regime with a characteristic slope of
        :math:`F_\nu \propto \nu^{11/8}`.

        To see a derivation of this result, see the theory note: :ref:`stratified_absorption`.

    .. tab-item:: SPL D (:math:`F_\nu \propto \nu^{1/3}`)

        SPL D occurs in the optically thin regime below the minimum electron frequency :math:`\nu_m`. In this
        regime, the SED is dominated by synchrotron emission from the low-energy tail of the single-electron SED.
        This results in a characteristic spectral slope of :math:`F_\nu \propto \nu^{1/3}`.

    .. tab-item:: SPL E (:math:`F_\nu \propto \nu^{1/3}`)

        SPL E has the same slope as SPL D, but occurs in the optically thin regime between the cooling frequency
        :math:`\nu_c` and the SSA frequency :math:`\nu_a`. In this regime, the SED is dominated by synchrotron emission
        from the low-energy tail of the single-electron SED, similar to SPL D.

    .. tab-item:: SPL F (:math:`F_\nu \propto \nu^{-1/2}`)

        SPL E occurs in the optically thin regime above the cooling frequency :math:`\nu_c` but
        below the minimum electron frequency :math:`\nu_m`. In this regime, the SED is dominated by
        synchrotron emission from the cooled portion of the electron distribution. Because this population has
        a characteristic electron index of :math:`p=2`, the emissivity scales as :math:`j_\nu \propto \nu^{-1/2}`.

    .. tab-item:: SPL G (:math:`F_\nu \propto \nu^{-(p-1)/2}`)

        SPL E occurs in the optically thin regime above the minimum electron frequency :math:`\nu_m` but
        below the cooling frequency :math:`\nu_c` (if present). In this regime, the SED is dominated by
        synchrotron emission from the full power-law distribution of electrons that have not yet cooled.
        This results in a characteristic spectral slope of :math:`F_\nu \propto \nu^{-(p-1)/2}`.

    .. tab-item:: SPL H (:math:`F_\nu \propto \nu^{-p/2}`)

        SPL H occurs in the optically thin regime above the cooling frequency :math:`\nu_c`. In this
        regime, the SED is dominated by synchrotron emission from the cooled portion of the electron
        distribution. This results in a characteristic spectral slope of :math:`F_\nu \propto \nu^{-p/2}`.

    .. tab-item:: SPL I (:math:`F_\nu \propto \nu^{1/2} \exp(-\nu/\nu_{\rm char})`)

        SPL I occurs in the extreme high-frequency regime above the maximum electron frequency :math:`\nu_{\max}`.
        In this regime, the SED is dominated by the exponential cutoff in the single-electron SEDs of the highest
        energy electrons. This results in a characteristic spectral slope of
        :math:`F_\nu \propto \nu^{1/2} \exp(-\nu/\nu_{\rm char})`.

Spectral Breaks
^^^^^^^^^^^^^^^

As with the SPL segments, we can list the different possible spectral breaks combining two such segments and
a given break frequency. Again, we follow the naming convention of :footcite:t:`GranotSari2002SpectralBreaks`.

.. tab-set::

    .. tab-item:: 1

        *Slopes*: SPL B to SPL D (:math:`F_\nu \propto \nu^{2}` to :math:`F_\nu \propto \nu^{1/3}`)

        *Break Frequency*: :math:`\nu_a`

        This break occurs at the self-absorption frequency :math:`\nu_a`, transitioning from the optically thick
        SPL B regime to the optically thin SPL D regime. The corresponding SBPL is

        .. math::
            :label: break_1_SBPL

            F_{\nu}^{(B,D)} = F^{(B,D)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_a}\right)^{2/s_{(B,D)}}
                +
                \left(\frac{\nu}{\nu_a}\right)^{(1/3)/s_{(B,D)}}
            \right]^{s_{(B,D)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_1_scale_free_SBPL

            \tilde{F}_{\nu}^{(B,D)} = \left[
                1+
                \left(\frac{\nu}{\nu_a}\right)^{(-5/3)/s_{(B,D)}}
            \right]^{s_{(B,D)}}.


    .. tab-item:: 2

        *Slopes*: SPL D to SPL G (:math:`F_\nu \propto \nu^{1/3}` to :math:`F_\nu \propto \nu^{-(p-1)/2}`)

        *Break Frequency*: :math:`\nu_m`

        This break occurs at the minimum electron frequency :math:`\nu_m`, transitioning from the optically thin
        SPL D regime to the optically thin SPL F regime. The corresponding SBPL is

        .. math::
            :label: break_2_SBPL

            F_{\nu}^{(D,G)} = F^{(D,G)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_m}\right)^{(1/3)/s_{(D,G)}}
                +
                \left(\frac{\nu}{\nu_m}\right)^{-(p-1)/2s_{(D,G)}}
            \right]^{s_{(D,G)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_2_scale_free_SBPL

            \tilde{F}_{\nu}^{(D,G)} = \left[
                1 +
                \left(\frac{\nu}{\nu_m}\right)^{(1-3p)/6s_{(D,G)}}
            \right]^{s_{(D,G)}}.

    .. tab-item:: 3

        *Slopes*: SPL G to SPL H (:math:`F_\nu \propto \nu^{-(p-1)/2}` to :math:`F_\nu \propto \nu^{-p/2}`)

        *Break Frequency*: :math:`\nu_c`

        This break occurs at the cooling frequency :math:`\nu_c`, transitioning from the optically thin
        SPL G regime to the optically thin SPL H regime. The corresponding SBPL is

        .. math::
            :label: break_3_SBPL

            F_{\nu}^{(G,H)} = F^{(G,H)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_c}\right)^{-(p-1)/2s_{(G,H)}}
                +
                \left(\frac{\nu}{\nu_c}\right)^{-p/2s_{(G,H)}}
            \right]^{s_{(G,H)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_3_scale_free_SBPL

            \tilde{F}_{\nu}^{(G,H)} = \left[
                1 +
                \left(\frac{\nu}{\nu_c}\right)^{-1/2s_{(G,H)}}
            \right]^{s_{(G,H)}}.

    .. tab-item:: 4

        *Slopes*: SPL B to SPL A (:math:`F_\nu \propto \nu^{2}` to :math:`F_\nu \propto \nu^{5/2}`)

        *Break Frequency*: :math:`\nu_m`

        This break occurs at the self-absorption frequency :math:`\nu_a`, transitioning from the optically thick
        SPL B regime to the optically thin SPL D regime. The corresponding SBPL is

        .. math::
            :label: break_4_SBPL

            F_{\nu}^{(B,A)} = F^{(B,A)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_a}\right)^{2/s_{(B,A)}}
                +
                \left(\frac{\nu}{\nu_a}\right)^{5/2s_{(B,A)}}
            \right]^{s_{(B,A)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_4_scale_free_SBPL

            \tilde{F}_{\nu}^{(B,A)} = \left[
                1 +
                \left(\frac{\nu}{\nu_a}\right)^{1/2s_{(B,A)}}
            \right]^{s_{(B,A)}}.

    .. tab-item:: 5

        *Slopes*: SPL A to SPL G (:math:`F_\nu \propto \nu^{5/2}` to :math:`F_\nu \propto \nu^{-(p-1)/2}`)

        *Break Frequency*: :math:`\nu_a`

        This break occurs at the self-absorption frequency :math:`\nu_a`, transitioning from the optically thick
        SPL B regime to the optically thin SPL D regime. The corresponding SBPL is

        .. math::
            :label: break_5_SBPL

            F_{\nu}^{(A,G)} = F^{(A,G)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_a}\right)^{5/2s_{(A,G)}}
                +
                \left(\frac{\nu}{\nu_a}\right)^{-(p-1)/2s_{(A,G)}}
            \right]^{s_{(A,G)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_5_scale_free_SBPL

            \tilde{F}_{\nu}^{(A,G)} = \left[
                1 +
                \left(\frac{\nu}{\nu_a}\right)^{-(2+4)/2s_{(A,G)}}
            \right]^{s_{(A,G)}}.

    .. tab-item:: 6

        *Slopes*: SPL A to SPL H (:math:`F_\nu \propto \nu^{5/2}` to :math:`F_\nu \propto \nu^{-p/2}`)

        *Break Frequency*: :math:`\nu_a`

        This break occurs at the self-absorption frequency :math:`\nu_a`, transitioning from the optically thick
        SPL B regime to the optically thin SPL D regime. The corresponding SBPL is

        .. math::
            :label: break_6_SBPL

            F_{\nu}^{(A,H)} = F^{(A,H)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_a}\right)^{5/2s_{(A,H)}}
                +
                \left(\frac{\nu}{\nu_a}\right)^{-p/2s_{(A,H)}}
            \right]^{s_{(A,H)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_6_scale_free_SBPL

            \tilde{F}_{\nu}^{(A,H)} = \left[
                1 +
                \left(\frac{\nu}{\nu_a}\right)^{-(p+5)/2s_{(A,H)}}
            \right]^{s_{(A,H)}}.

    .. tab-item:: 7

        *Slopes*: SPL B to SPL C (:math:`F_\nu \propto \nu^{2}` to :math:`F_\nu \propto \nu^{11/8}`)

        *Break Frequency*: :math:`\nu_{ac}`

        This break occurs at the stratified self-absorption frequency :math:`\nu_{ac}`, transitioning from the optically thick
        SPL B regime to the stratified self-absorption SPL C regime. The corresponding SBPL is

        .. math::
            :label: break_7_SBPL

            F_{\nu}^{(B,C)} = F^{(B,C)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_{ac}}\right)^{2/s_{(B,C)}}
                +
                \left(\frac{\nu}{\nu_{ac}}\right)^{(11/8)/s_{(B,C)}}
            \right]^{s_{(B,C)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_7_scale_free_SBPL

            \tilde{F}_{\nu}^{(B,C)} = \left[
                1+
                \left(\frac{\nu}{\nu_{ac}}\right)^{(-5/8)/s_{(B,C)}}
            \right]^{s_{(B,C)}}.

    .. tab-item:: 8

        *Slopes*: SPL C to SPL F (:math:`F_\nu \propto \nu^{11/8}` to :math:`F_\nu \propto \nu^{-1/2}`)

        *Break Frequency*: :math:`\nu_a`

        This break occurs at the self-absorption frequency :math:`\nu_a`, transitioning from the optically thick
        SPL B regime to the optically thin SPL D regime. The corresponding SBPL is

        .. math::
            :label: break_8_SBPL

            F_{\nu}^{(C,F)} = F^{(C,F)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_a}\right)^{(11/8)/s_{(C,F)}}
                +
                \left(\frac{\nu}{\nu_a}\right)^{-1/2s_{(C,F)}}
            \right]^{s_{(C,F)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_8_scale_free_SBPL

            \tilde{F}_{\nu}^{(C,F)} = \left[
                1 +
                \left(\frac{\nu}{\nu_a}\right)^{(-15/8)/s_{(C,F)}}
            \right]^{s_{(C,F)}}.

    .. tab-item:: 9

        *Slopes*: SPL F to SPL H (:math:`F_\nu \propto \nu^{-1/2}` to :math:`F_\nu \propto \nu^{-p/2}`)

        *Break Frequency*: :math:`\nu_m`

        This break occurs at the minimum electron frequency :math:`\nu_m`, transitioning from the optically thin
        SPL D regime to the optically thin SPL F regime. The corresponding SBPL is

        .. math::
            :label: break_9_SBPL

            F_\nu}^{(F,H)} = F^{(F,H)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_m}\right)^{-1/2s_{(F,H)}}
                +
                \left(\frac{\nu}{\nu_m}\right)^{-p/2s_{(F,H)}}
            \right]^{s_{(F,H)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_9_scale_free_SBPL

            \tilde{F}_{\nu}^{(F,H)} = \left[
                1 +
                \left(\frac{\nu}{\nu_m}\right)^{-(p-1)/2s_{(F,H)}}
            \right]^{s_{(F,H)}}.

    .. tab-item:: 10

        *Slopes*: SPL C to SPL E (:math:`F_\nu \propto \nu^{11/8}` to :math:`F_\nu \propto \nu^{1/3}`)

        *Break Frequency*: :math:`\nu_a`

        This break occurs at the self-absorption frequency :math:`\nu_a`, transitioning from the optically thick
        SPL B regime to the optically thin SPL D regime. The corresponding SBPL is

        .. math::
            :label: break_10_SBPL

            F_{\nu}^{(C,E)} = F^{(C,E)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_a}\right)^{(11/8)/s_{(C,E)}}
                +
                \left(\frac{\nu}{\nu_a}\right)^{(1/3)/s_{(C,E)}}
            \right]^{s_{(C,E)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_10_scale_free_SBPL

            \tilde{F}_{\nu}^{(C,E)} = \left[
                \left(\frac{\nu}{\nu_a}\right)^{(-25/24)/s_{(C,E)}}
                +
                1
            \right]^{s_{(C,E)}}.

    .. tab-item:: 11

        *Slopes*: SPL E to SPL F (:math:`F_\nu \propto \nu^{1/3}` to :math:`F_\nu \propto \nu^{-1/2}`)

        *Break Frequency*: :math:`\nu_c`

        This break occurs at the cooling frequency :math:`\nu_c`, transitioning from the optically thin
        SPL G regime to the optically thin SPL H regime. The corresponding SBPL is

        .. math::
            :label: break_11_SBPL

            F_{\nu}^{(E,F)} = F^{(E,F)}_{\nu,0} \left[
                \left(\frac{\nu}{\nu_c}\right)^{(1/3)/s_{(E,F)}}
                +
                \left(\frac{\nu}{\nu_c}\right)^{-1/2s_{(E,F)}}
            \right]^{s_{(E,F)}}.

        and the scale-free SBPL is

        .. math::
            :label: break_11_scale_free_SBPL

            \tilde{F}_{\nu}^{(E,F)} = \left[
                \left(\frac{\nu}{\nu_c}\right)^{-5/6s_{(E,F)}}
                +
                1
            \right]^{s_{(E,F)}}.

Our final set of spectral breaks occur as one ventures into the asymptotic high-frequency regime beyond the
maximum electron frequency :math:`\nu_{\max}`. In this regime, the SED transitions from any of the optically thin
segments (SPL F, SPL G, or SPL H) to the exponential cutoff segment (SPL I). Because the
exponential cutoff is not a power law, we do not provide SBPL representations for these breaks, but instead
provide **exponential cutoff functions**. For discrete (non-smooth) representations of SEDs, we use the function
:math:`\Phi(\nu,\nu_{\rm max})` to denote the cutoff:

.. math::

    \Phi(\nu,\nu_{\rm max}) = \left(\frac{\nu}{\nu_{\max}}\right)^{1/2}
    \exp\left(1 -\frac{\nu}{\nu_{\max}}\right).

In the smoothed case described above, we instead need a **scale-free exponential cutoff function** which does
not interfere with the normalization of the SED at lower frequencies. We therefore define:

.. math::

    \tilde{\Phi}(\nu,\nu_{\max}) = \begin{cases}
        1, & \nu < \nu_{\max} \\
        \left(\frac{\nu}{\nu_{\max}}\right)^{1/2}
        \exp\left(1 -\frac{\nu}{\nu_{\max}}\right), & \nu \geq \nu_{\max}.
    \end{cases}


Broadband SEDs
^^^^^^^^^^^^^^

For each of the broadband SEDs described below, we provide two different formulations of the SED: one utilizing
the smoothed broken power-law (SBPL) construction described in :ref:`sed_surgery`, and one providing the full piecewise
definition of the SED. The normalization procedure is described in :ref:`sed_normalization`.

The Power Law SED
~~~~~~~~~~~~~~~~~

.. note::

    The SED referred to here is known as *the* "power-law synchrotron SED" throughout Triceratops' literature
    and documentation. Modifiers such as "with cooling" or "with SSA" are used to indicate the presence of
    additional physical processes.

.. rubric:: Parameters

.. list-table::
    :widths: 35 85
    :header-rows: 1

    * - Parameter Class
      - Parameters
    * - **Free Parameters**
      - :math:`\nu_m`, :math:`\nu_{\max}`, :math:`F_{\nu,\rm pk}`.
    * - **Derived Parameters**
      - None.
    * - **Break Frequencies**
      - :math:`\nu_m`, :math:`\nu_{\max}`.
    * - **Hyper-Parameters**
      - :math:`p`, :math:`s`.

.. rubric:: Description

We start with the simplest power-law SED: that of a power-law distribution of electrons with no cooling and
no absorption. In this case, the only break frequencies are the minimum and maximum electron frequencies, leading
to segments of SPL H, SPL F, and SPL D. The smoothed SED may be constructed as:

.. math::

    F_\nu = F^{(D,G)}_\nu \tilde{\Phi}(\nu,\nu_{\max}),

The discrete SED segments are:

.. math::

    F_\nu = F_{\nu,\rm pk} \begin{cases}
        \left(\frac{\nu}{\nu_m}\right)^{1/3}, & \nu < \nu_m \quad \text{(SPL D)}\\
        \left(\frac{\nu}{\nu_m}\right)^{-(p-1)/2}, & \nu_m \leq \nu < \nu_{\max} \quad \text{(SPL G)}\\
        \left(\frac{\nu_{\max}}{\nu_m}\right)^{-(p-1)/2}
        \Phi(\nu,\nu_{\rm max}), & \nu \geq \nu_{\max} \quad \text{(SPL I)}
    \end{cases}

The SSA Power Law SED
~~~~~~~~~~~~~~~~~~~~~~

.. rubric:: Parameters

.. list-table::
    :widths: 35 85
    :header-rows: 1

    * - Parameter Class
      - Parameters
    * - **Free Parameters**
      - :math:`\nu_m`, :math:`\nu_{\max}`, :math:`F_{\nu,\rm pk}`.
    * - **Derived Parameters**
      - :math:`\nu_a`.
    * - **Break Frequencies**
      - :math:`\nu_m`, :math:`\nu_{\max}`, :math:`\nu_a`.
    * - **Hyper-Parameters**
      - :math:`p`, :math:`s`, :math:`\Omega`, :math:`\gamma_{\rm min}`

.. rubric:: Description

We now progress to the case with SSA but no cooling. In this case, there are 2(3) orderings of the break frequencies:

1. :math:`\nu_a < \nu_m < \nu_{\max}`: In this case, the SED segments are SPL B, SPL D, SPL F, and SPL H.
2. :math:`\nu_m < \nu_a < \nu_{\max}`: In this case, the SED segments are SPL B, SPL A, SPL F, and SPL H.
3. :math:`\nu_m < \nu_{\max} < \nu_a`: This scenario is non-physical as self-absorption requires electrons with energies
   at or near the characteristic energy of the absorbed frequency. Since there are no electrons above the maximum
   cutoff, there is no way to self-absorb at those frequencies.

.. tab-set::

    .. tab-item:: Spectrum 1

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. list-table::
            :widths: 15 15 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_a`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_a \leq \nu < \nu_m`
              - SPL D
              - :math:`1/3`
            * - 3
              - :math:`\nu_m \leq \nu < \nu_{\max}`
              - SPL G
              - :math:`-(p-1)/2`
            * - 4
              - :math:`\nu \geq \nu_{\max}`
              - SPL I
              - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = \tilde{F}^{(B,D)}_\nu F^{(D,G)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we have selected to normalize at the (D,G) break at :math:`\nu_m`. The discrete SED segments are:

        .. math::

            F_\nu = F_{\nu,0} \begin{cases}
                \left(\frac{\nu}{\nu_a}\right)^{2}\left(\frac{\nu_a}{\nu_m}\right)^{1/3}, & \nu < \nu_a \quad \text{(SPL B)}\\
                \left(\frac{\nu}{\nu_m}\right)^{1/3},& \nu_a < \nu < \nu_m \quad \text{(SPL D)}\\
                \left(\frac{\nu}{\nu_m}\right)^{-(p-1)/2}, & \nu_m \leq \nu < \nu_{\max} \quad \text{(SPL G)}\\
                \left(\frac{\nu_{\max}}{\nu_m}\right)^{-(p-1)/2}
                \Phi(\nu,\nu_{\rm max}), & \nu \geq \nu_{\max} \quad \text{(SPL I)}\\
            \end{cases}

        .. rubric:: The Absorption Frequency

        In this case, the absorption frequency :math:`\nu_a` does not correspond to the **peak-frequency** of the SED. We
        therefore need to follow the power-law segments to find the peak. Thus, the
        flux from the **optically thin** portion of the SED at :math:`\nu_a` is given by

        .. math::

            F_{\nu}(\nu_a) = F_{\nu,\rm pk} \left(\frac{\nu_a}{\nu_m}\right)^{1/3}.

        The optically thick side must be

        .. math::

            F_{\nu}(\nu_a) = 2\nu_a^2 \gamma_a m_e \Omega = 2\nu_a^2 m_e \Omega
            \left(\frac{\nu_a}{\nu_m}\right)^{1/2} \gamma_m,

        where we make use of :math:`\gamma_m` as a hyper-parameter to relate :math:`\gamma_a` and :math:`\nu_a`. Thus,

        .. math::

            \boxed{
            \nu_a = \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{6/13} \nu_m^{1/13}.
            }

    .. tab-item:: Spectrum 2

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. list-table::
            :widths: 15 15 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_m`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_m \leq \nu < \nu_a`
              - SPL A
              - :math:`5/2`
            * - 3
              - :math:`\nu_a \leq \nu < \nu_{\max}`
              - SPL G
              - :math:`-(p-1)/2`
            * - 4
              - :math:`\nu \geq \nu_{\max}`
              - SPL I
              - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = F^{(B,A)}_\nu \tilde{F}^{(A,G)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we have selected to normalize at the (B,A) break at :math:`\nu_m`. The discrete SED segments are:

        .. math::

            F_\nu = F_{\nu,0} \begin{cases}
                \left(\frac{\nu}{\nu_m}\right)^{2}, & \nu < \nu_m \quad \text{(SPL B)}\\
                \left(\frac{\nu}{\nu_m}\right)^{5/2}, & \nu_m < \nu < \nu_a \quad \text{(SPL A)}\\
                \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
                \left(\frac{\nu}{\nu_a}\right)^{-(p-1)/2}, & \nu_a \leq \nu < \nu_{\max} \quad \text{(SPL G)}\\
                \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
                \left(\frac{\nu_{\max}}{\nu_m}\right)^{-(p-1)/2}
                \Phi(\nu,\nu_{\rm max}), & \nu \geq \nu_{\max} \quad \text{(SPL I)}\\
            \end{cases}

        .. rubric:: The Absorption Frequency

        In this case, the absorption frequency :math:`\nu_a` corresponds to the **peak-frequency** of the SED. The
        flux from the **optically thin** portion of the SED at :math:`\nu_a` is given by :math:`F_{\nu,\rm pk}`. The
        optically thick side must be

        .. math::

            F_{nu,\rm pk} = 2\nu_a^2 \gamma_a m_e \Omega = 2\nu_a^2 m_e \Omega \left(\frac{\nu_a}{\nu_m}\right)^{1/2} \gamma_m,

        where we make use of :math:`\gamma_m` as a hyper-parameter to relate :math:`\gamma_a` and :math:`\nu_a`. Thus,

        .. math::

            \boxed{
            \nu_a = \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{2/5} \nu_m^{1/5}.
            }

Cooling Power Law SEDs
~~~~~~~~~~~~~~~~~~~~~~

The other simple scenario worth considering is the SED from a synchrotron source with non-negligible cooling
and no SSA. In this case, the three relevant break frequencies are :math:`\nu_m`, :math:`\nu_c`, and
:math:`\nu_{\rm max}`. There are 3 possible configurations

1. :math:`\nu_c < \nu_m < \nu_{\rm max}`: The **fast-cooling** regime. The SED here is composed of segments SPL E,
   SPL F, SPL H, and SPL I with slopes :math:`1/3, 1/2, -p/2, {\rm exp}`. The maximum in this case occurs at
   :math:`\nu_c`, and so we use that point to normalize.
2. :math:`\nu_m < \nu_c < \nu_{\rm max}`: The **slow-cooling** regime. The SED here is composed of segments SPL D,
   SPL G, SPL H, and SPL I with slopes :math:`1/3, -(p-1)/2, -p/2, {\rm exp}`.
   The maximum in this case occurs at :math:`\nu_m`, and so we use that point to normalize.
3. :math:`\nu_m < \nu_{\rm max} < \nu_c`: The **uncooled regime**. The SED here is identical to the standard
   power-law SED and is therefore not described in any further detail.

.. tab-set::

    .. tab-item:: Spectrum 1

        In this spectrum, there are 4 SPL segments connected by 3 breaks. Because the population is rapidly cooled,
        the bulk of electrons are effectively reduced to :math:`\gamma_c` and the corresponding peak in the spectrum
        occurs at :math:`\nu_c`.

        .. list-table::
            :widths: 15 15 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_c`
              - SPL E
              - :math:`1/3`
            * - 2
              - :math:`\nu_c < \nu < \nu_m`
              - SPL F
              - :math:`-1/2`
            * - 3
              - :math:`\nu_m \leq \nu < \nu_{\max}`
              - SPL H
              - :math:`-p/2`
            * - 4
              - :math:`\nu \geq \nu_{\max}`
              - SPL I
              - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = F^{(E,F)}_\nu \tilde{F}^{(F,H)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we will normalize at :math:`\nu_c` using the cooled population and corresponding electron
        distribution function. The discrete SED segments are:

        .. math::

            F_\nu = F_{\nu,0} \begin{cases}
                \left(\frac{\nu}{\nu_c}\right)^{1/3}, & \nu < \nu_c \quad \text{(SPL E)}\\
                \left(\frac{\nu}{\nu_c}\right)^{-1/2}, & \nu_c < \nu < \nu_m \quad \text{(SPL F)}\\
                \left(\frac{\nu_m}{\nu_c}\right)^{-1/2}
                \left(\frac{\nu}{\nu_m}\right)^{-p/2}, & \nu_m < \nu < \nu_{\rm max} \quad \text{(SPL H)}\\
                \left(\frac{\nu_m}{\nu_c}\right)^{-1/2}
                \left(\frac{\nu_{\rm max}}{\nu_m}\right)^{-p/2}
                \left(\frac{\nu}{\nu_{\rm max}}\right)^{-1/2}
                \exp\left(1-\frac{\nu}{\nu_{\rm max}}\right), & \nu > \nu_{\rm max} \quad \text{(SPL I)}.
            \end{cases}

        .. rubric:: Normalization

        In this case, the peak of the frequency occurs at :math:`\nu_c` and we therefore normalize there. The
        normalization takes the form of :eq:`fast_cooling_norm` and :eq:`fast_cooling_norm_iso` (*we show only the
        fixed pitch angle case for brevity*):

        .. math::

            F_{c,0} \approx \chi (B\sin\alpha)^{1/2}
            K_0 \left(\frac{\nu_m}{\nu_c}\right)\,\gamma_{c}
            \frac{V}{D_L^2},


    .. tab-item:: Spectrum 2

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. list-table::
            :widths: 15 15 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_c`
              - SPL D
              - :math:`1/3`
            * - 2
              - :math:`\nu_c < \nu < \nu_m`
              - SPL G
              - :math:`-(p-1)/2`
            * - 3
              - :math:`\nu_m \leq \nu < \nu_{\max}`
              - SPL H
              - :math:`-p/2`
            * - 4
              - :math:`\nu \geq \nu_{\max}`
              - SPL I
              - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = F^{(D,G)}_\nu \tilde{F}^{(G,H)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we have selected to normalize at the (D,G) break at :math:`\nu_m`. The discrete SED segments are:

        .. math::

            F_\nu = F_{\nu,0} \begin{cases}
                \left(\frac{\nu}{\nu_m}\right)^{1/3}, & \nu < \nu_m \quad \text{(SPL D)}\\
                \left(\frac{\nu}{\nu_m}\right)^{-(p-1)/2}, & \nu_m < \nu < \nu_c \quad \text{(SPL G)}\\
                \left(\frac{\nu}{\nu_c}\right)^{-p/2}
                \left(\frac{\nu_c}{\nu_m}\right)^{-(p-1)/2}, & \nu_c < \nu < \nu_{\rm max} \quad \text{(SPL H)}\\
                \left(\frac{\nu_c}{\nu_m}\right)^{-(p-1)/2}
                \left(\frac{\nu_{\rm max}}{\nu_c}\right)^{-p/2}
                \left(\frac{\nu}{\nu_{\rm max}}\right)^{1/2}
                \exp\left(1-\frac{\nu}{\nu_{\rm max}}\right), & \nu > \nu_{\rm max} \quad \text{(SPL I)}.
            \end{cases}

        The dominant electron population at the spectrum peak is the population of **uncooled electrons**. As such,
        the normalization takes the form of :eq:`slow_cooling_norm`:

        .. math::

            F_{\nu,0} = F_{\nu_m,0}

Cooling+SSA Power Law SEDs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We are now prepared to introduce the complete set of synchrotron SEDs relevant to the most generic scenarios in which
both SSA and cooling are relevant. We therefore have the break frequencies :math:`\nu_m`, :math:`\nu_a`, :math:`\nu_c`,
and :math:`\nu_{\rm max}`. In addition, for absorption dominated regimes with fast cooling, we have the additional
break frequency :math:`\nu_{\rm ac}` from stratified SSA (see the theory note: :ref:`stratified_absorption`). This leads
to 8 regimes characterized by the cooling state and the radiation transfer state at maximum:

- A spectrum is either **fast cooling** (:math:`\nu_c < \nu_m`), **slow cooling** (:math:`\nu_m <\nu_c < \nu_{\rm max}`)
  or **no cooling** (:math:`\nu_c > \nu_{\rm max}`).
- A spectrum is optically **thick** at maximum if :math:`\nu_a > \rm{min}(\nu_a,\nu_c)` and is optically **thin** at
  peak if :math:`\nu_a < \rm{min}(\nu_a,\nu_c)`.

The resulting spectra are

1. :math:`(\nu_a < \nu_m < \nu_{\rm max} < \nu_c)`: This is the **thin, no cooling** spectrum.
2. :math:`(\nu_m < \nu_a < \nu_{\rm max} < \nu_c)`: This it the **thick, no cooling** spectrum.
3. :math:`(\nu_a < \nu_m < \nu_c < \nu_{\rm max})`: This is the **thin, slow cooling** spectrum.
4. :math:`(\nu_m < \nu_a < \nu_c < \nu_{\rm max})`: This is the **thick, slow cooling** spectrum.
5. :math:`(\nu_a < \nu_c < \nu_m < \nu_{\rm max})`: This is the **thin, fast cooling** spectrum.
6. :math:`(\nu_c < \nu_a < \nu_m < \nu_{\rm max})`: This is the **thick, fast cooling** spectrum.
7. :math:`(\nu_c, \nu_m < \nu_a < \nu_{\rm max})`: This is the **extremely thick, fast cooling** spectrum.

In the tab set below, we'll go through each of these and discuss the normalization and the corresponding SEDs for the
various cases.

.. tab-set::

    .. tab-item:: Spectrum 1 :math:`(\nu_a < \nu_m < \nu_{\rm max} < \nu_c)`

        This is the **SSA-only** spectrum in which cooling is irrelevant over the
        emitting band because :math:`\nu_c` lies above the high-energy cutoff
        :math:`\nu_{\max}`. It is therefore equivalent to spectrum 1 from our discussion above
        regarding non-cooling synchrotron SEDs.

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. list-table::
            :widths: 15 15 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_a`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_a \leq \nu < \nu_m`
              - SPL D
              - :math:`1/3`
            * - 3
              - :math:`\nu_m \leq \nu < \nu_{\max}`
              - SPL G
              - :math:`-(p-1)/2`
            * - 4
              - :math:`\nu \geq \nu_{\max}`
              - SPL I
              - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = \tilde{F}^{(B,D)}_\nu F^{(D,G)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we have selected to normalize at the (D,G) break at :math:`\nu_m`. The discrete SED segments are:

        .. math::

            F_\nu = F_{\nu,0} \begin{cases}
                \left(\frac{\nu}{\nu_a}\right)^{2}\left(\frac{\nu_a}{\nu_m}\right)^{1/3}, & \nu < \nu_a \quad \text{(SPL B)}\\
                \left(\frac{\nu}{\nu_m}\right)^{1/3},& \nu_a < \nu < \nu_m \quad \text{(SPL D)}\\
                \left(\frac{\nu}{\nu_m}\right)^{-(p-1)/2}, & \nu_m \leq \nu < \nu_{\max} \quad \text{(SPL G)}\\
                \left(\frac{\nu_{\max}}{\nu_m}\right)^{-(p-1)/2}
                \Phi(\nu,\nu_{\rm max}), & \nu \geq \nu_{\max} \quad \text{(SPL I)}\\
            \end{cases}

        .. rubric:: The Absorption Frequency

        In this case, the absorption frequency :math:`\nu_a` does not correspond to the **peak-frequency** of the SED. We
        therefore need to follow the power-law segments to find the peak. Thus, the
        flux from the **optically thin** portion of the SED at :math:`\nu_a` is given by

        .. math::

            F_{\nu}(\nu_a) = F_{\nu,\rm pk} \left(\frac{\nu_a}{\nu_m}\right)^{1/3}.

        The optically thick side must be

        .. math::

            F_{\nu}(\nu_a) = 2\nu_a^2 \gamma_a m_e \Omega = 2\nu_a^2 m_e \Omega
            \left(\frac{\nu_a}{\nu_m}\right)^{1/2} \gamma_m,

        where we make use of :math:`\gamma_m` as a hyper-parameter to relate :math:`\gamma_a` and :math:`\nu_a`. Thus,

        .. math::

            \boxed{
            \nu_a = \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{6/13} \nu_m^{1/13}.
            }


    .. tab-item:: Spectrum 2 :math:`(\nu_m < \nu_a < \nu_{\rm max} < \nu_c)`

        This is the **SSA-only** spectrum in which cooling is irrelevant over the
        emitting band because :math:`\nu_c` lies above the high-energy cutoff
        :math:`\nu_{\max}`. It is therefore equivalent to spectrum 2 from our discussion above
        regarding non-cooling synchrotron SEDs.

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. list-table::
            :widths: 15 15 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_m`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_m \leq \nu < \nu_a`
              - SPL A
              - :math:`5/2`
            * - 3
              - :math:`\nu_a \leq \nu < \nu_{\max}`
              - SPL G
              - :math:`-(p-1)/2`
            * - 4
              - :math:`\nu \geq \nu_{\max}`
              - SPL I
              - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = F^{(B,A)}_\nu \tilde{F}^{(A,G)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we have selected to normalize at the (B,A) break at :math:`\nu_m`. The discrete SED segments are:

        .. math::

            F_\nu = F_{\nu,0} \begin{cases}
                \left(\frac{\nu}{\nu_m}\right)^{2}, & \nu < \nu_m \quad \text{(SPL B)}\\
                \left(\frac{\nu}{\nu_m}\right)^{5/2}, & \nu_m < \nu < \nu_a \quad \text{(SPL A)}\\
                \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
                \left(\frac{\nu}{\nu_a}\right)^{-(p-1)/2}, & \nu_a \leq \nu < \nu_{\max} \quad \text{(SPL G)}\\
                \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
                \left(\frac{\nu_{\max}}{\nu_m}\right)^{-(p-1)/2}
                \Phi(\nu,\nu_{\rm max}), & \nu \geq \nu_{\max} \quad \text{(SPL I)}\\
            \end{cases}

        .. rubric:: The Absorption Frequency

        In this case, the absorption frequency :math:`\nu_a` corresponds to the **peak-frequency** of the SED. The
        flux from the **optically thin** portion of the SED at :math:`\nu_a` is given by :math:`F_{\nu,\rm pk}`. The
        optically thick side must be

        .. math::

            F_{nu,\rm pk} = 2\nu_a^2 \gamma_a m_e \Omega = 2\nu_a^2 m_e \Omega \left(\frac{\nu_a}{\nu_m}\right)^{1/2} \gamma_m,

        where we make use of :math:`\gamma_m` as a hyper-parameter to relate :math:`\gamma_a` and :math:`\nu_a`. Thus,

        .. math::

            \boxed{
            \nu_a = \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{2/5} \nu_m^{1/5}.
            }

    .. tab-item:: Spectrum 3 :math:`(\nu_a < \nu_m < \nu_c < \nu_{\rm max})`

        This is the standard **slow-cooling + SSA** spectrum with all three breaks
        present in-band.

        .. list-table::
            :widths: 15 22 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_a`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_a < \nu < \nu_m`
              - SPL D
              - :math:`1/3`
            * - 3
              - :math:`\nu_m \le \nu < \nu_c`
              - SPL G
              - :math:`-(p-1)/2`
            * - 4
              - :math:`\nu_c \le \nu < \nu_{\max}`
              - SPL H
              - :math:`-p/2`
            * - 5
              - :math:`\nu \ge \nu_{\max}`
              - SPL I
              - cutoff

        The SBPL SED may be constructed as:

        .. math::

            F_\nu
            =
            F_{\nu_m,0}
            \,
            \tilde{F}_\nu^{(B,D)}(\nu;\nu_a)
            \,
            F_\nu^{(D,G)}(\nu;\nu_m)
            \,
            \tilde{F}_\nu^{(G,H)}(\nu;\nu_c)
            \,
            \tilde{\Phi}(\nu;\nu_{\max})

        The corresponding discrete SED takes the form

        .. math::

            F_\nu = F_{\nu,0}\begin{cases}
            \left(\dfrac{\nu_a}{\nu_m}\right)^{1/3}\left(\dfrac{\nu}{\nu_a}\right)^2,
            & \nu < \nu_a \quad \text{(SPL B)},\\[6pt]
            \left(\dfrac{\nu}{\nu_m}\right)^{1/3},
            & \nu_a \le \nu < \nu_m \quad \text{(SPL D)},\\[6pt]
            \left(\dfrac{\nu}{\nu_m}\right)^{-(p-1)/2},
            & \nu_m \le \nu < \nu_c \quad \text{(SPL G)},\\[6pt]
            \left(\dfrac{\nu_c}{\nu_m}\right)^{-(p-1)/2}
            \left(\dfrac{\nu}{\nu_c}\right)^{-p/2},
            & \nu_c \le \nu < \nu_{\max} \quad \text{(SPL H)},\\[6pt]
            \left(\dfrac{\nu_c}{\nu_m}\right)^{-(p-1)/2}
            \left(\dfrac{\nu_{\max}}{\nu_c}\right)^{-p/2}
            \Phi(\nu,\nu_{\rm max}),& \nu > \nu_{\rm max}  \quad \text{(SPL I)}
            \end{cases}

        .. rubric:: The Absorption Frequency

        In this case, the absorption frequency :math:`\nu_a` does not correspond to the **peak-frequency** of the SED. We
        therefore need to follow the power-law segments to find the peak. Thus, the
        flux from the **optically thin** portion of the SED at :math:`\nu_a` is given by

        .. math::

            F_{\nu}(\nu_a) = F_{\nu,\rm pk} \left(\frac{\nu_a}{\nu_m}\right)^{1/3}.

        The optically thick side must be

        .. math::

            F_{\nu}(\nu_a) = 2\nu_a^2 \gamma_a m_e \Omega = 2\nu_a^2 m_e \Omega
            \left(\frac{\nu_a}{\nu_m}\right)^{1/2} \gamma_m,

        where we make use of :math:`\gamma_m` as a hyper-parameter to relate :math:`\gamma_a` and :math:`\nu_a`. Thus,

        .. math::

            \boxed{
            \nu_a = \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{6/13} \nu_m^{1/13}.
            }


    .. tab-item:: Spectrum 4 :math:`(\nu_m < \nu_a < \nu_c < \nu_{\rm max})`

        This is the **slow-cooling + SSA** spectrum with :math:`\nu_a` above
        :math:`\nu_m`, producing an optically thick :math:`\nu^{5/2}` segment
        between :math:`\nu_m` and :math:`\nu_a`.

        .. list-table::
            :widths: 15 22 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_m`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_m < \nu < \nu_a`
              - SPL A
              - :math:`5/2`
            * - 3
              - :math:`\nu_a \le \nu < \nu_c`
              - SPL G
              - :math:`-(p-1)/2`
            * - 4
              - :math:`\nu_c \le \nu < \nu_{\max}`
              - SPL H
              - :math:`-p/2`
            * - 5
              - :math:`\nu \ge \nu_{\max}`
              - SPL I
              - cutoff

        We anchor the SED at :math:`\nu_m` so that the SED takes the form

        .. math::

            F_\nu
            =
            F_{\nu_m,0}
            \,
            F_\nu^{(B,A)}(\nu;\nu_m)
            \,
            \tilde{F}_\nu^{(A,G)}(\nu;\nu_a)
            \,
            \tilde{F}_\nu^{(G,H)}(\nu;\nu_c)
            \,
            \tilde{\Phi}(\nu;\nu_{\max})

        The corresponding discrete SED takes the form

        .. math::

            F_\nu = F_{\nu,0}\begin{cases}
            \left(\frac{\nu}{\nu_m}\right)^2,&\nu<\nu_m \quad \text{(SPL B)}\\
            \left(\frac{\nu}{\nu_m}\right)^{5/2},&\nu_m<\nu<\nu_a \quad \text{(SPL A)}\\
            \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
            \left(\frac{\nu}{\nu_a}\right)^{-(p-1)/2},& \nu_a < \nu < \nu_c \quad \text{(SPL G)}\\
            \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
            \left(\frac{\nu_c}{\nu_a}\right)^{-(p-1)/2}
            \left(\frac{\nu}{\nu_c}\right)^{-p/2},& \nu_c < \nu < \nu_{\rm max} \quad \text{(SPL H)}\\
            \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
            \left(\frac{\nu_c}{\nu_a}\right)^{-(p-1)/2}
            \left(\frac{\nu}{\nu_c}\right)^{-p/2}
            \Phi(\nu,\nu_{\rm max}),& \nu > \nu_{\rm max} \quad \text{(SPL I)}\\
            \end{cases}

        .. rubric:: The Absorption Frequency

        In this case, the absorption frequency :math:`\nu_a` corresponds to the **peak-frequency** of the SED. The
        flux from the **optically thin** portion of the SED at :math:`\nu_a` is given by :math:`F_{\nu,\rm pk}`. The
        optically thick side must be

        .. math::

            F_{nu,\rm pk} = 2\nu_a^2 \gamma_a m_e \Omega = 2\nu_a^2 m_e \Omega \left(\frac{\nu_a}{\nu_m}\right)^{1/2} \gamma_m,

        where we make use of :math:`\gamma_m` as a hyper-parameter to relate :math:`\gamma_a` and :math:`\nu_a`. Thus,

        .. math::

            \boxed{
            \nu_a = \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{2/5} \nu_m^{1/5}.
            }


    .. tab-item:: Spectrum 5 :math:`(\nu_a < \nu_c < \nu_m < \nu_{\rm max})`

        Spectrum 5 is the first of the two spectra in this formalism which is subject to the
        effects of **stratified SSA**, which introduces an additional SSA break at a frequency
        :math:`\nu_{\rm ac}`. The segments of the spectrum are

        .. list-table::
            :widths: 15 22 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_{ac}`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_{ac} < \nu < \nu_a`
              - SPL C
              - :math:`11/8`
            * - 3
              - :math:`\nu_a < \nu < \nu_c`
              - SPL E
              - :math:`1/3`
            * - 4
              - :math:`\nu_c \le \nu < \nu_m`
              - SPL F
              - :math:`-1/2`
            * - 5
              - :math:`\nu_m \le \nu < \nu_{\max}`
              - SPL H
              - :math:`-p/2`
            * - 6
              - :math:`\nu \ge \nu_{\max}`
              - SPL I
              - cutoff

        The SBPL SED may be constructed as:

        .. math::

            F_\nu
            =
            \,
            \tilde{F}_\nu^{(B,C)}(\nu;\nu_{\rm ac})
            \,
            \tilde{F}_\nu^{(C,E)}(\nu;\nu_a)
            \,
            F_\nu^{(E,F)}(\nu;\nu_c)
            \,
            \tilde{F}_\nu^{(F,H)}(\nu;\nu_m)
            \,
            \tilde{\Phi}(\nu;\nu_{\max})

        Piecewise spectrum (normalized at :math:`\nu_c`):

        .. math::

            F_\nu = F_{\nu,0}\begin{cases}
            \left(\frac{\nu_a}{\nu_c}\right)^{1/3}
            \left(\frac{\nu_{\rm ac}}{\nu_a}\right)^{11/8}
            \left(\frac{\nu}{\nu_{\rm ac}}\right)^2,& \nu < \nu_{\rm ac},\quad \text{(SPL B)}\\[6pt]
            \left(\frac{\nu_a}{\nu_c}\right)^{1/3}
            \left(\frac{\nu}{\nu_a}\right)^{11/8},& \nu_{\rm ac} < \nu < \nu_a,\quad \text{(SPL C)}\\[6pt]
            \left(\dfrac{\nu}{\nu_c}\right)^{1/3},& \nu_a \le \nu < \nu_c,\quad \text{(SPL E)}\\[6pt]
            \left(\dfrac{\nu}{\nu_c}\right)^{-1/2},& \nu_c \le \nu < \nu_m,\quad \text{(SPL F)}\\[6pt]
            \left(\dfrac{\nu_m}{\nu_c}\right)^{-1/2}
            \left(\dfrac{\nu}{\nu_m}\right)^{-p/2},& \nu_m \le \nu < \nu_{\max},\quad \text{(SPL H)}\\[6pt]
            \left(\dfrac{\nu_m}{\nu_c}\right)^{-1/2}
            \left(\dfrac{\nu_{\max}}{\nu_m}\right)^{-p/2}
            \Phi_{\rm cut}(\nu;\nu_{\max}),& \nu \ge \nu_{\max} \quad \text{(SPL I)}.
            \end{cases}


        .. rubric:: The Absorption Frequency

        In this case, the absorption frequency :math:`\nu_a` does **not** correspond to the peak of the SED. Instead, the
        spectrum peaks at the cooling break :math:`\nu_c`, with the absorption break occurring at lower frequency,
        :math:`\nu_a < \nu_c < \nu_m`. As a result, the flux density at :math:`\nu_a` must be obtained by propagating
        *downward* from the peak using the appropriate optically thin power-law segment.

        Between :math:`\nu_a` and :math:`\nu_c`, the spectrum follows a :math:`\nu^{1/3}` scaling. The flux density on the
        optically thin side at the absorption frequency is therefore

        .. math::

            F_\nu(\nu_a)
            =
            F_{\nu,\rm pk}
            \left(\frac{\nu_a}{\nu_c}\right)^{1/3},

        where :math:`F_{\nu,\rm pk}` denotes the peak flux density at :math:`\nu_c`.

        On the optically thick side, the emission at :math:`\nu_a` is well approximated by a blackbody with effective
        temperature :math:`kT_{\rm eff} = \gamma_a m_e c^2`, where the Lorentz factor of the emitting electrons satisfies

        .. math::

            \gamma_a
            =
            \gamma_m
            \left(\frac{\nu_a}{\nu_m}\right)^{1/2}.

        The corresponding optically thick flux density is therefore

        .. math::

            F_\nu(\nu_a)
            =
            2\nu_a^2 m_e \gamma_a \Omega
            =
            2 m_e \Omega \gamma_m
            \left(\frac{\nu_a}{\nu_m}\right)^{1/2}
            \nu_a^2.

        Equating the optically thin and optically thick expressions at :math:`\nu_a` yields

        .. math::

            F_{\nu,\rm pk}
            \left(\frac{\nu_a}{\nu_c}\right)^{1/3}
            =
            2 m_e \Omega \gamma_m
            \nu_m^{-1/2}
            \nu_a^{5/2}.

        Solving for the absorption frequency, we obtain

        .. math::

            \boxed{
            \nu_a
            =
            \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{6/13}
            \nu_m^{3/13}
            \nu_c^{-2/13}.
            }

        This expression makes explicit that, in this spectral ordering, the absorption frequency depends not only on the
        peak flux density and angular size of the source, but also on the location of the cooling break, reflecting the
        fact that the SED peak occurs at :math:`\nu_c` rather than at the absorption frequency itself.



    .. tab-item:: Spectrum 6 :math:`(\nu_c < \nu_a < \nu_m < \nu_{\rm max})`

        Spectrum 6 is the second case in which the SSA break due to stratified SSA appears at
        :math:`\nu_{\rm ac}`. Additionally, because :math:`\nu_c` is obscured by SSA, we also have to
        perform power-law propagation to correct the normalization, making this one of the trickier of the
        SED cases.

        .. list-table::
            :widths: 15 22 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - PLS Type
              - Slope
            * - 1
              - :math:`\nu < \nu_{\rm ac}`
              - SPL B
              - :math:`2`
            * - 2
              - :math:`\nu_{\rm ac} < \nu < \nu_a`
              - SPL C
              - :math:`11/8`
            * - 3
              - :math:`\nu_a \le \nu < \nu_m`
              - SPL F
              - :math:`-1/2`
            * - 4
              - :math:`\nu_m \le \nu < \nu_{\max}`
              - SPL H
              - :math:`-p/2`
            * - 5
              - :math:`\nu \ge \nu_{\max}`
              - SPL I
              - cutoff

        The SBPL SED may be constructed as:

        .. math::

            F_\nu
            =
            \,
            \tilde{F}_\nu^{(B,C)}(\nu;\nu_{\rm ac})
            \,
            F_\nu^{(C,F)}(\nu;\nu_a)
            \,
            \tilde{F}_\nu^{(F,H)}(\nu;\nu_m)
            \,
            \tilde{\Phi}(\nu;\nu_{\max})

        The corresponding discrete SED is

        .. math::

            F_\nu = F_{\nu,0}\begin{cases}
            \left(\frac{\nu_{\rm ac}}{\nu_{\rm a}}\right)^{11/8}
            \left(\frac{\nu}{\nu_{\rm ac}}\right)^2,& \nu < \nu_{\rm ac},\quad \text{(SPL B)}\\[6pt]
            \left(\frac{\nu}{\nu_{\rm a}}\right)^{11/8},& \nu_{\rm ac} < \nu < \nu_a,\quad \text{(SPL C)}\\[6pt]
            \left(\dfrac{\nu}{\nu_a}\right)^{-1/2},& \nu_a \le \nu < \nu_m,\quad \text{(SPL F)}\\[6pt]
            \left(\dfrac{\nu_m}{\nu_a}\right)^{-1/2}
            \left(\dfrac{\nu}{\nu_m}\right)^{-p/2},& \nu_m \le \nu < \nu_{\max},\quad \text{(SPL H)}\\[6pt]
            \left(\dfrac{\nu_m}{\nu_a}\right)^{-1/2}
            \left(\dfrac{\nu_{\max}}{\nu_m}\right)^{-p/2}
            \Phi_{\rm cut}(\nu;\nu_{\max}),& \nu \ge \nu_{\max} \quad \text{(SPL I)}.
            \end{cases}

        .. rubric:: The Absorption Frequency

        In this case, the absorption frequency :math:`\nu_a` corresponds to the **peak-frequency** of the SED. The
        flux from the **optically thin** portion of the SED at :math:`\nu_a` is given by :math:`F_{\nu,\rm pk}`. The
        optically thick side must be

        .. math::

            F_{nu,\rm pk} = 2\nu_a^2 \gamma_a m_e \Omega = 2\nu_a^2 m_e \Omega \left(\frac{\nu_a}{\nu_m}\right)^{1/2} \gamma_m,

        where we make use of :math:`\gamma_m` as a hyper-parameter to relate :math:`\gamma_a` and :math:`\nu_a`. Thus,

        .. math::

            \boxed{
            \nu_a = \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{2/5} \nu_m^{1/5}.
            }


    .. tab-item:: Spectrum 7 :math:`(\nu_c, \nu_m < \nu_a < \nu_{\rm max})`

        This spectrum corresponds to scenarios where SSA is dominant over both cooling and
        the minimum injection break. In this regime, the relative ordering of :math:`\nu_m` and
        :math:`\nu_c` is irrelevant because the post-shock material becomes optically thick to
        SSA immediately and so cooled material does not have the ability to contribute to the
        spectrum. We therefore see the traditional low-energy tail :math:`\nu^2` up to the
        minimum injection energy :math:`\nu_m`, beyond which we obtain the standard
        :math:`\nu^{5/2}` scaling. Finally, beyond the absorption break, we have optically
        thin emission from the steady state cooled population of electrons deeper in the
        post-shock material producing the typical :math:`\nu^{-p/2}`.

        In this spectrum, the regimes are as follows

        .. list-table::
            :widths: 15 22 15 15
            :header-rows: 1

            * - Segment
              - Frequency Range
              - SPL Type
              - Slope
            * - 1
              - :math:`\nu < \nu_m`
              - SPL B
              - :math:`2`
            * - 3
              - :math:`\nu_m \le \nu < \nu_a`
              - SPL A
              - :math:`5/2`
            * - 4
              - :math:`\nu_a \le \nu < \nu_{\rm max}`
              - SPL H
              - :math:`-p/2`
            * - 5
              - :math:`\nu \ge \nu_{\rm max}`
              - SPL I
              - cutoff

        The SBPL SED may be constructed as:

        .. math::

            F_\nu
            =
            F_{\nu_m,0}
            \,
            F_\nu^{(B,A)}(\nu;\nu_m)
            \,
            \tilde{F}_\nu^{(A,H)}(\nu;\nu_a)
            \,
            \tilde{\Phi}(\nu;\nu_{\max})

        The discrete SED is therefore

        .. math::

            F_\nu
            =
            F_{\nu,0}
            \begin{cases}
                \left(\frac{\nu}{\nu_m}\right)^{2},
                & \nu < \nu_m
                \quad \text{(SPL B)}
                \\[6pt]
                \left(\frac{\nu}{\nu_m}\right)^{5/2},
                & \nu_m < \nu < \nu_a
                \quad \text{(SPL A)}
                \\[6pt]
                \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
                \left(\frac{\nu}{\nu_a}\right)^{-p/2},
                & \nu_a < \nu < \nu_{\rm max}
                \quad \text{(SPL H)}
                \\[6pt]
                \left(\frac{\nu_a}{\nu_m}\right)^{5/2}
                \left(\frac{\nu_{\rm max}}{\nu_a}\right)^{-p/2}
                \Phi(\nu,\nu_{\rm max})
                & \nu > \nu_{\rm max}
                \quad \text{(SPL I)}.
            \end{cases}

        .. rubric:: The Absorption Frequency

        In this case, the absorption frequency :math:`\nu_a` corresponds to the **peak-frequency** of the SED. The
        flux from the **optically thin** portion of the SED at :math:`\nu_a` is given by :math:`F_{\nu,\rm pk}`. The
        optically thick side must be

        .. math::

            F_{nu,\rm pk} = 2\nu_a^2 \gamma_a m_e \Omega = 2\nu_a^2 m_e \Omega \left(\frac{\nu_a}{\nu_m}\right)^{1/2} \gamma_m,

        where we make use of :math:`\gamma_m` as a hyper-parameter to relate :math:`\gamma_a` and :math:`\nu_a`. Thus,

        .. math::

            \boxed{
            \nu_a = \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{2/5} \nu_m^{1/5}.
            }

----

.. _sed_normalization:
SED Normalization
------------------

We have, at this stage, fully developed the framework through which synchrotron SEDs are constructed it is worth
discussing how one connects these SEDs to physical parameters. This is generically **not possible without some
concession of additional information** because the SEDs themselves are scale-free. In other words, given a set of break
frequencies and a normalization flux density, there are an infinite number of physical configurations which can produce the same
observed SED. Therefore, to connect the SEDs to physical parameters, we must **adopt a normalization / closure**.

The most common such closure is that of **equipartition** between the energy in relativistic electrons and magnetic fields.
This approach is widely used in the literature (see, e.g., :footcite:t:`demarchiRadioAnalysisSN2004C2022`,
:footcite:t:`Margutti2019COW`, :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`, :footcite:t:`ChevalierXRayRadioEmission1982`,
among others) and is often justified on the grounds that it minimizes the total energy budget of the system. However, it is
important to note that this is an **assumption** rather than a physical requirement. In other words, there is no guarantee that
the system will actually be in equipartition, and indeed there is evidence from some systems that it is not.

In Triceratops, we implement the SEDs in a **scale-free** manner such that the user may select **any normalization scheme** they
wish. This is achieved by separating the calculation of the break frequencies and spectral shapes from the normalization
itself. The user may then implement any desired closure scheme to connect the SEDs to physical parameters. For example, one may
choose to implement equipartition, or one may choose to implement a different closure based on, e.g., observations or theoretical
considerations.

For convenience, we do provide a normalization scheme implemented directly in the SED classes of :mod:`radiation.synchrotron.SEDs`
which is based on Equipartition and is described here.

.. admonition:: Convention Note

    We here adopt a normalization scheme based on the approach of :footcite:t:`1998ApJ...497L..17S` and used in
    :footcite:t:`2025ApJ...992L..18S`, :footcite:t:`sari1999jets`, :footcite:t:`duran2013radius`,
    :footcite:t:`2020MNRAS.493.3521B` among others. This is different from the approach used in
    :footcite:t:`GranotSari2002SpectralBreaks` which was an insufficiently general scheme to suite a modular codebase
    such as Triceratops.

While seemingly a simple undertaking, the normalization of the SEDs is a complex element of the theory. This is, in
part, because various approaches have been used in the literature ranging from exact calculations using numerical quadrature
to a variety of approximate schemes. In our case, we select a scheme based off of that described in
:footcite:t:`1998ApJ...497L..17S`, which is an approximate scheme based off the notion that, at an (optically thin) break frequency
:math:`\nu_{\rm brk}`, the synchrotron emission is dominated a single set of electrons with lorentz factors :math:`\gamma_{\rm brk}`
and number density :math:`n_e(\gamma_{\rm brk})`. One may, therefore, use this approximation to estimate the normalization
to within a factor of order unity.

To be more precise, consider a population of electrons with a distribution of Lorentz factors :math:`dN/d\gamma(\gamma)`. The
emissivity of synchrotron emission for such a population (see :ref:`synchrotron_theory`) is

.. math::

    j_\nu = \frac{\sqrt{3} q^3 B \sin \alpha}{4\pi mc^2} \int_{\gamma_{\rm min}}^{\gamma_{\rm max}} \frac{dN}{d\gamma} F\left(
        \frac{\nu}{\nu_c(\gamma)} \right) d\gamma,

where :math:`q` is the electron charge, :math:`B` is the magnetic field strength, :math:`\gamma_{\rm min}` and :math:`\gamma_{\rm max}`
are the bounding Lorentz factors for the population, and :math:`F` is the synchrotron spectrum. :math:`\nu_c(\gamma)` is
the critical frequency function

.. math::

    \nu_c(\gamma) = \frac{3q}{4\pi m c} B \sin \alpha \gamma^2 = c_{1,\gamma} (B \sin \alpha) \gamma^2,

where :math:`c_{1,\gamma}` is as defined in :ref:`synchrotron_theory`.

Letting :math:`x(\gamma)\equiv \nu/\nu_c(\gamma)`, we have :math:`dx/d\gamma = -2x/\gamma` and thus
:math:`d\gamma = -(\gamma/2x)\,dx`. Since :math:`F(x)` is sharply peaked near :math:`x\sim\mathcal{O}(1)`,
the integral is dominated by :math:`\gamma=\gamma_\nu` satisfying :math:`\nu\sim\nu_c(\gamma_\nu)`, and the remaining
kernel integral contributes only an order-unity constant. Therefore,

.. math::

    j_\nu \approx \frac{\sqrt{3} q^3 B \sin \alpha}{4\pi m_e c^2}\,\left.\frac{dN}{d\gamma}\right|_{\gamma=\gamma_\nu}\,\gamma_\nu.

.. note::

    We here, as in :footcite:t:`1998ApJ...497L..17S`, suppress an order-unity factor stemming from integration
    of the kernel. Furthermore, in that paper, the value :math:`\frac{dN}{d\gamma}(\gamma_\nu) \gamma_\nu` is
    approximated as an **effective number density** of electrons emitting at frequency :math:`\nu`. While this
    is a useful interpretation, we choose to retain the explicit dependence on the electron distribution function
    for clarity.

Assuming an **effective emitting volume** V at a luminosity distance :math:`D_L`,

.. math::

    F_\nu \approx \frac{V}{D_L^2} j_\nu.

.. hint::

    A common approach in the literature which can be used, but is not *required* is to assume a spherical emitting
    region of radius :math:`R` and a *filling factor* :math:`f`, such that :math:`V = f (4\pi R^3/3)`.

This is therefore the basis of our normalization scheme with a number of caveats:

1. The approximation made above is only really sensible when used to describe the emissivity of the **maximal population
   of electrons**. We therefore always anchor our normalization to the **peak emission frequency**; however,
2. The presence of absorption may **obscure the optically thin peak**. To correct for this, we use the known shape
   of the corresponding SED and the relevant break frequencies to "extrapolate" the optically thin peak down to the
   absorption break and then back along the correct PLS to correct the normalization for absorption effects.

To be more precise about which population of electrons we use to normalize the SED, we note that the determination of the
correct population of electrons is determined by the cooling regime, with fast-cooling and slow-cooling scenarios corresponding
to populations peaking at :math:`\nu_c` and :math:`\nu_m`, respectively. In the tabs below, we describe the normalization
for each case:

.. important::

    For clarity of language, we emphasize a **DIFFERENCE** between the **normalization frequency** (i.e., the frequency
    at which we anchor the normalization of the SED) and the **peak emission frequency** (i.e., the frequency at which
    the SED peaks).

    In :mod:`~radiation.synchrotron.SEDs`, we **NEVER** refer directly to the **normalization frequency**. This is an
    internal conceptualization used to tie microphysics and dynamics to SED normalization. When parameterizing each
    SED, we normalize using the **peak emission frequency** which is *calculated* from the normalization frequency and
    power-law slopes of the SED.

.. tab-set::

    .. tab-item:: Fast Cooling

        .. admonition:: TODO

            Are there concerns here about the electron population actually cooling like this? We need to touch base
            with Raf.

        In the case of a **fast-cooling** electron population, the flux of electrons below the injection
        frequency :math:`\nu_m` produces a steady-state electron distribution which, by our conventions,
        (see :ref:`synchrotron_cooling_theory`) is

        .. math::

            \frac{dN}{d\gamma} = K_0 \begin{cases}
                  0, & \gamma < \gamma_c \\
                  \left(\frac{\gamma}{\gamma_{\rm m}}\right)^{-2}, & \gamma_c \le \gamma < \gamma_{\rm m} \\
                  \left(\frac{\gamma}{\gamma_{\rm m}}\right)^{-(p+1)}, & \gamma_{\rm max} \ge \gamma \ge \gamma_{\rm m} \\
                    0, & \gamma > \gamma_{\max}
            \end{cases},

        where :math:`K_0` is the normalization constant of the electron distribution. One can obtain a normalization
        through many means, most commonly equipartition arguments or shock acceleration theory.

        The corresponding **normalizing** flux and frequency are therefore :math:`F_{c,0}` at
        :math:`\nu_c = \nu(\gamma_c)`. The population follows

        .. math::

            \left.\frac{dN}{d\gamma}\right|_{\gamma=\gamma_c} = K_0 \left(\frac{\gamma_c}{\gamma_{\rm m}}\right)^{-2}.

        Given that :math:`\gamma(\nu) \propto \nu^{1/2}`, we therefore have

        .. math::

            \left.\frac{dN}{d\gamma}\right|_{\gamma=\gamma_\nu} = K_0 \left(\frac{\nu_m}{\nu_c}\right).

        The corresponding emissivity at :math:`\nu_c` is therefore

        .. math::

            j_\nu \approx
            \frac{\sqrt{3} q^3 B \sin \alpha}{4\pi m_e c^2}
            K_0 \left(\frac{\nu_m}{\nu_c}\right)\,\gamma_{c}.

        Collecting the constants, we have

        .. math::
            :label: eq_chi

            \chi = \frac{\sqrt{3} q^3}{4\pi m_e c^2},

        and

        .. math::
            :label: eq_chi_iso

            \chi_{\rm iso} = \frac{\sqrt{3} q^3}{2\pi^2 m_e c^2},

        Normalization takes the form

        .. math::
            :label: fast_cooling_norm

            F_{c,0} \approx \chi (B\sin\alpha) K_0 \left(\frac{\nu_m}{\nu_c}\right)\,\gamma_{c} \frac{V}{D_L^2},

        or, equivalently, for the pitch-angle averaged case,

        .. math::
            :label: fast_cooling_norm_iso

            F_{c0,{\rm iso}} \approx \chi_{\rm iso} B K_0 \left(\frac{\nu_m}{\nu_c}\right)\,\gamma_{c} \frac{V}{D_L^2}.

    .. tab-item:: Slow Cooling

        In the case of a **slow-cooling** electron population, the distribution of electrons follows

        .. math::

            \frac{dN}{d\gamma} = N_0 \begin{cases}
                  0, & \gamma < \gamma_{\rm m} \\
                  \gamma^{-p}, & \gamma_{\max} \ge \gamma \ge \gamma_{\rm m} \\
                    0, & \gamma > \gamma_{\max}
            \end{cases},

        where :math:`N_0` is the normalization constant of the electron distribution. As such

        .. math::

            \left.\frac{dN}{d\gamma}\right|_{\gamma=\gamma_\nu} = N_0 \gamma_\nu^{-p}.

        so, at the normalizing frequency :math:`\nu_m = \nu(\gamma_{\min})`, we have

        .. math::

            \left.\frac{dN}{d\gamma}\right|_{\gamma=\gamma_m} \gamma_{m} = N_0 \gamma_{\min}^{1-p}.

        The emissivity is then

        .. math::

            j_\nu \approx
            \frac{\sqrt{3} q^3 B \sin \alpha}{4\pi m_e c^2}
            N_0 \gamma_{\min}^{1-p}.

        Collecting the constants, we have

        .. math::

            \chi = \frac{\sqrt{3} q^3}{4\pi m_e c^2},

        and

        .. math::

            \chi_{\rm iso} = \frac{\sqrt{3} q^3}{2\pi^2 m_e c^2},

        the normalizing flux at :math:`\nu_m` is

        .. math::
            :label: slow_cooling_norm

            F_{m,0} \approx \chi (B\sin\alpha) N_0 \gamma_{\rm m}^{1-p} \frac{V}{D_L^2},

        or, equivalently, for the pitch-angle averaged case,

        .. math::
            :label: slow_cooling_norm_iso

            F_{m0,{\rm iso}} \approx \chi_{\rm iso} B N_0 \gamma_{\rm m}^{1-p}  \frac{V}{D_L^2}.

Given a selected normalizing population (:math:`F_{c,0}` at :math:`\nu_c` for fast-cooling or :math:`F_{m,0}` at :math:`\nu_m` for slow-cooling),
one may then use equipartition to compute the relevant coefficients in the flux normalization expressions above. In the sections
below, we'll go through each of the SED cases and describe how to connect the normalization frequency to the peak emission frequency
and thus the full SED normalization.

.. important::

    In the **observer frame**, the above expressions should be modified by a factor
    :math:`\delta^3/(1+z)`, where :math:`\delta` is the Doppler factor of the emitting material
    and :math:`z` is the redshift of the source.

The Power-Law SED Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We begin with the **canonical synchrotron spectral energy distribution (SED)**: a power-law SED in the absence
of radiative cooling or synchrotron self-absorption. In this regime, the emitting plasma is optically thin
at all frequencies of interest, and the electron distribution retains its injected power-law form.

Under these assumptions, the only spectral breaks present in the SED occur at the characteristic synchrotron
frequencies associated with the minimum and maximum electron Lorentz factors,
:math:`\nu_m` and :math:`\nu_{\max}`, respectively. The peak of the broadband SED occurs at the
**injection frequency** :math:`\nu_m`, corresponding to synchrotron emission from electrons with
Lorentz factor :math:`\gamma_{\rm min}`.

.. rubric:: Parameters

The normalization of the canonical SED is fixed by the following physical parameters:

.. list-table::
    :widths: 10 35 85
    :header-rows: 1

    * - Parameter
      - Name
      - Notes
    * - :math:`B`
      - Magnetic Field Strength
      - Magnetic field strength in the emission region. Determines the characteristic synchrotron
        frequencies and enters the SED normalization.
    * - :math:`V_{\rm eff}`
      - Effective Emitting Volume
      - Volume over which relativistic electrons radiate efficiently.
    * - :math:`D_L`
      - Luminosity Distance
      - Luminosity distance to the source.
    * - :math:`p`
      - Electron Power-Law Index
      - Power-law index of the injected electron distribution.
    * - :math:`\gamma_{\rm min}`
      - Minimum Lorentz Factor
      - Lower cutoff of the electron Lorentz factor distribution.
    * - :math:`\gamma_{\rm max}`
      - Maximum Lorentz Factor
      - Upper cutoff of the electron Lorentz factor distribution.
    * - :math:`\epsilon_e`
      - Electron Energy Fraction
      - Fraction of post-shock internal energy carried by relativistic electrons.
    * - :math:`\epsilon_B`
      - Magnetic Energy Fraction
      - Fraction of post-shock internal energy stored in magnetic fields.
    * - :math:`\sin\alpha`
      - Pitch Angle Factor
      - Pitch angle dependence of synchrotron emission. Triceratops supports either fixed pitch
        angles or isotropic pitch-angle averaging.

.. rubric:: Method

The normalization of the power-law SED is most naturally anchored at the injection frequency
:math:`\nu_m`, defined as the characteristic synchrotron frequency of electrons with
Lorentz factor :math:`\gamma_{\rm min}`:

.. math::

    \nu_m \equiv \nu_{\rm synch}(\gamma_{\rm min}).

For a given choice of pitch-angle treatment, the magnetic field strength :math:`B` therefore
uniquely determines the location of the spectral peak.

The flux density at this frequency sets the overall normalization of the SED. For the case of a
fixed pitch angle, the flux normalization at :math:`\nu_m` is given by
:eq:`slow_cooling_norm` as

.. math::

    F_{m,0} \approx
    \chi \, (B \sin\alpha)\,
    N_0 \,
    \gamma_{\rm min}^{\,1-p}
    \frac{V_{\rm eff}}{D_L^2},

where :math:`N_0` is the normalization of the electron distribution and :math:`\chi` is a numerical
coefficient arising from the synchrotron kernel and fundamental constants.

For isotropically distributed pitch angles, this expression becomes

.. math::

    F_{m,0}^{\rm (iso)} \approx
    \chi_{\rm iso}\,
    B\,
    N_0 \,
    \gamma_{\rm min}^{\,1-p}
    \frac{V_{\rm eff}}{D_L^2},

where :math:`\chi_{\rm iso}` incorporates the pitch-angle averaging.

The electron normalization :math:`N_0` is not a free parameter. Instead, it is determined by
equipartition arguments through the choice of :math:`\epsilon_e` and :math:`\epsilon_B`, which
relate the electron and magnetic energy densities to the post-shock internal energy density.
Once :math:`N_0` is fixed in this manner, the flux at :math:`\nu_m` — and hence the normalization
of the entire canonical synchrotron SED — is fully specified.

All remaining portions of the SED then follow from simple power-law propagation away from
:math:`\nu_m` until the high-energy cutoff at :math:`\nu_{\max}`.


The SSA Power Law SED
^^^^^^^^^^^^^^^^^^^^^^
Normalization is made trickier in the SSA case because the peak of the SED may occur at either :math:`\nu_m` or
:math:`\nu_a`, depending on the relative ordering of these frequencies. Additionally, the exact value of the absorption
frequency :math:`\nu_a` depends on the normalization itself, requiring a somewhat more careful treatment.

.. rubric:: Parameters

The normalization of the canonical SED is fixed by the following physical parameters:

.. list-table::
    :widths: 10 35 85
    :header-rows: 1

    * - Parameter
      - Name
      - Notes
    * - :math:`B`
      - Magnetic Field Strength
      - Magnetic field strength in the emission region. Determines the characteristic synchrotron
        frequencies and enters the SED normalization.
    * - :math:`V_{\rm eff}`
      - Effective Emitting Volume
      - Volume over which relativistic electrons radiate efficiently.
    * - :math:`D_L`
      - Luminosity Distance
      - Luminosity distance to the source.
    * - :math:`p`
      - Electron Power-Law Index
      - Power-law index of the injected electron distribution.
    * - :math:`\gamma_{\rm min}`
      - Minimum Lorentz Factor
      - Lower cutoff of the electron Lorentz factor distribution.
    * - :math:`\gamma_{\rm max}`
      - Maximum Lorentz Factor
      - Upper cutoff of the electron Lorentz factor distribution.
    * - :math:`\epsilon_e`
      - Electron Energy Fraction
      - Fraction of post-shock internal energy carried by relativistic electrons.
    * - :math:`\epsilon_B`
      - Magnetic Energy Fraction
      - Fraction of post-shock internal energy stored in magnetic fields.
    * - :math:`\sin\alpha`
      - Pitch Angle Factor
      - Pitch angle dependence of synchrotron emission. Triceratops supports either fixed pitch
        angles or isotropic pitch-angle averaging.
    * - :math:`\Omega`
      - Angular Size
      - Angular size of the emitting region. Required to compute the SSA frequency.

.. rubric:: Method

We begin by computing the characteristic synchrotron frequencies
:math:`\nu_m` and :math:`\nu_{\rm max}` from the magnetic field strength
and electron distribution parameters. We then compute the *equivalent
optically thin* flux normalization at :math:`\nu_m`, denoted
:math:`F_{m,0}`, using :eq:`slow_cooling_norm` or
:eq:`slow_cooling_norm_iso`. This quantity represents the flux at
:math:`\nu_m` implied by the assumed optically thin power-law segments
for a given trial SED ordering.

If :math:`\nu_a < \nu_m`, then :math:`\nu_m` corresponds to the true peak
of the emission and the normalization is complete. If instead
:math:`\nu_a > \nu_m`, the true peak of the SED occurs at
:math:`\nu_a`, and the normalization must be propagated from
:math:`\nu_m` up to :math:`\nu_a` along the appropriate optically thin
power-law segment in order to determine the true peak flux
:math:`F_{\nu,\rm pk}`.

This procedure is complicated by the fact that the absorption frequency
:math:`\nu_a` itself depends on the peak flux. To resolve this
self-consistently, we compute :math:`\nu_a` assuming each possible
ordering of :math:`\nu_a` relative to :math:`\nu_m`, and then determine
which ordering satisfies its own consistency conditions.

The relationship between the normalization flux :math:`F_{m,0}` at
:math:`\nu_m` and the true peak flux :math:`F_{\nu,\rm pk}` therefore
depends on the ordering of :math:`\nu_a` relative to :math:`\nu_m`. We
treat each case in turn below.

.. tab-set::

    .. tab-item:: Spectrum 1 (:math:`\nu_a < \nu_m < \nu_{\rm max}`)

        In this case, the absorption frequency lies below the injection
        frequency, so the true peak of the SED occurs at
        :math:`\nu_m`. The presumed optically thin normalization
        :math:`F_{m,0}` therefore directly corresponds to the true peak
        flux,

        .. math::

            F_{\nu,\rm pk} = F_{m,0}.

        The absorption frequency is precisely that derived previously since the
        peak and the normalization are coincident:

        .. math::

            \nu_a
            =
            \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{2/5}
            \nu_m^{1/5}.


    .. tab-item:: Spectrum 2 (:math:`\nu_m < \nu_a < \nu_{\rm max}`)

        In this case, the absorption frequency lies above the injection
        frequency, so the true peak of the SED occurs at
        :math:`\nu_a`. The normalization must therefore be propagated
        from :math:`\nu_m` up to :math:`\nu_a` along the optically thin
        segment :math:`F_\nu \propto \nu^{-(p-1)/2}`.

        The relationship between the presumed optically thin
        normalization at :math:`\nu_m` and the true peak flux at
        :math:`\nu_a` is

        .. math::

            F_{\nu,\rm pk}
            =
            F_{m,0}
            \left(\frac{\nu_a}{\nu_m}\right)^{-(p-1)/2}.

        The absorption frequency in this regime is given by the SSA
        matching condition derived above,

        .. math::

            \nu_a
            =
            \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{2/5}
            \nu_m^{1/5}.

        Combining these expressions and solving self-consistently for
        :math:`\nu_a` yields

        .. math::

            \boxed{
            \nu_a
            =
            \left(\frac{F_{m,0}}{2 m_e \Omega \gamma_m}\right)^{2/(p+4)}
            \nu_m^{(p+2)/(p+4)}.
            }

        One may compute this candidate value :math:`\nu_a^{(2)}` and
        verify that :math:`\nu_m < \nu_a^{(2)} < \nu_{\rm max}` to confirm
        that this regime applies.

Cooling Power Law SEDs
^^^^^^^^^^^^^^^^^^^^^^

In the case of cooling without SSA, normalization becomes (once again) a relatively simple task of anchoring
the SED at the appropriate peak frequency. In the case of **fast-cooling** electrons, the peak of the SED occurs
at the **cooling frequency** :math:`\nu_c`, while in the case of **slow-cooling** electrons, the peak occurs
at the **injection frequency** :math:`\nu_m`. In both cases, the SED remains optically thin at all
frequencies of interest, and so no complications arise from absorption effects.

Unlike the uncooled case in which we always anchored the SED normalization at :math:`\nu_m`, here we
must distinguish between the two cooling regimes to determine the appropriate normalization frequency. In fast
cooling case, we use **equipartition** to fix the normalization of the steady-state electron distribution
(see :ref:`synchrotron_cooling_theory`).

.. rubric:: Parameters

The normalization of the canonical SED is fixed by the following physical parameters:

.. list-table::
    :widths: 10 35 85
    :header-rows: 1

    * - Parameter
      - Name
      - Notes
    * - :math:`B`
      - Magnetic Field Strength
      - Magnetic field strength in the emission region. Determines the characteristic synchrotron
        frequencies and enters the SED normalization.
    * - :math:`V_{\rm eff}`
      - Effective Emitting Volume
      - Volume over which relativistic electrons radiate efficiently.
    * - :math:`D_L`
      - Luminosity Distance
      - Luminosity distance to the source.
    * - :math:`p`
      - Electron Power-Law Index
      - Power-law index of the injected electron distribution.
    * - :math:`\gamma_{\rm min}`
      - Minimum Lorentz Factor
      - Lower cutoff of the electron Lorentz factor distribution.
    * - :math:`\gamma_{\rm max}`
      - Maximum Lorentz Factor
      - Upper cutoff of the electron Lorentz factor distribution.
    * - :math:`\gamma_c`
      - Cooling Lorentz Factor
      - Lorentz factor above which electrons cool efficiently within the dynamical time.
    * - :math:`\epsilon_e`
      - Electron Energy Fraction
      - Fraction of post-shock internal energy carried by relativistic electrons.
    * - :math:`\epsilon_B`
      - Magnetic Energy Fraction
      - Fraction of post-shock internal energy stored in magnetic fields.
    * - :math:`\sin\alpha`
      - Pitch Angle Factor
      - Pitch angle dependence of synchrotron emission. Triceratops supports either fixed pitch
        angles or isotropic pitch-angle averaging.

.. rubric:: Method

We begin by computing the characteristic Lorentz factors :math:`\gamma_{\rm min}` and
:math:`\gamma_c`, along with their associated synchrotron frequencies
:math:`\nu_m` and :math:`\nu_c`. The cooling regime is then determined by their ordering:

- **Slow cooling**: :math:`\gamma_c > \gamma_{\rm min}` (equivalently :math:`\nu_c > \nu_m`)
- **Fast cooling**: :math:`\gamma_c < \gamma_{\rm min}` (equivalently :math:`\nu_c < \nu_m`)

In the slow-cooling regime, the dominant population of emitting electrons is that at
:math:`\gamma_{\rm min}`, and the SED is normalized at :math:`\nu_m` using the optically thin
normalization described in :ref:`sed_normalization`.

In the fast-cooling regime, the steady-state electron distribution is modified by radiative
losses above :math:`\gamma_c`. In this case, equipartition arguments are used to normalize the
cooled electron distribution (see :ref:`synchrotron_cooling_theory`), and the SED is anchored
at the cooling frequency :math:`\nu_c`.

Once the appropriate normalization frequency is identified and the corresponding flux
normalization is computed, the remainder of the SED follows directly by propagating the
normalization across the relevant optically thin power-law segments.

Cooling+SSA Power Law SEDs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In combining the two cases of cooling and SSA, normalization becomes a mixture of both cases. As in the
SSA case above, if the peak is optically thick, one must correctly propagate the normalization to the
absorption break before continuing. However, the location of the peak itself is determined by the cooling
regime as in the cooling-only case. In other words, in the fast-cooling case, the peak occurs at
:math:`\nu_c`, while in the slow-cooling case, the peak occurs at :math:`\nu_m`. Nonetheless, the infrastructure
remains the same as in the SSA case above, with the added complexity of determining the cooling regime
first.


.. rubric:: Parameters

The normalization of the canonical SED is fixed by the following physical parameters:

.. list-table::
    :widths: 10 35 85
    :header-rows: 1

    * - Parameter
      - Name
      - Notes
    * - :math:`B`
      - Magnetic Field Strength
      - Magnetic field strength in the emission region. Determines the characteristic synchrotron
        frequencies and enters the SED normalization.
    * - :math:`V_{\rm eff}`
      - Effective Emitting Volume
      - Volume over which relativistic electrons radiate efficiently.
    * - :math:`D_L`
      - Luminosity Distance
      - Luminosity distance to the source.
    * - :math:`p`
      - Electron Power-Law Index
      - Power-law index of the injected electron distribution.
    * - :math:`\gamma_{\rm min}`
      - Minimum Lorentz Factor
      - Lower cutoff of the electron Lorentz factor distribution.
    * - :math:`\gamma_{\rm max}`
      - Maximum Lorentz Factor
      - Upper cutoff of the electron Lorentz factor distribution.
    * - :math:`\gamma_c`
      - Cooling Lorentz Factor
      - Lorentz factor above which electrons cool efficiently within the dynamical time.
    * - :math:`\epsilon_e`
      - Electron Energy Fraction
      - Fraction of post-shock internal energy carried by relativistic electrons.
    * - :math:`\epsilon_B`
      - Magnetic Energy Fraction
      - Fraction of post-shock internal energy stored in magnetic fields.
    * - :math:`\sin\alpha`
      - Pitch Angle Factor
      - Pitch angle dependence of synchrotron emission. Triceratops supports either fixed pitch
        angles or isotropic pitch-angle averaging.
    * - :math:`\Omega`
      - Angular Size
      - Angular size of the emitting region. Required to compute the SSA frequency.

.. rubric:: Method

.. tab-set::

    .. tab-item:: Spectrum 1 :math:`(\nu_a <\nu_m <\nu_{\rm max} <\nu_c)`

        In this case, the absorption frequency lies below the injection
        frequency, so the true peak of the SED occurs at
        :math:`\nu_m`. The presumed optically thin normalization
        :math:`F_{m,0}` therefore directly corresponds to the true peak
        flux,

        .. math::

            F_{\nu,\rm pk} = F_{m,0}.

        The absorption frequency is precisely that derived previously since the
        peak and the normalization are coincident:

        .. math::

            \boxed{
            \nu_a = \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{6/13} \nu_m^{1/13}.
            }

    .. tab-item:: Spectrum 2 :math:`(\nu_m < \nu_a < \nu_{\rm max} < \nu_c)`

        In this case, the absorption frequency lies above the injection
        frequency, so the true peak of the SED occurs at
        :math:`\nu_a`. The normalization must therefore be propagated
        from :math:`\nu_m` up to :math:`\nu_a` along the optically thin
        segment :math:`F_\nu \propto \nu^{-(p-1)/2}`.

        The relationship between the presumed optically thin
        normalization at :math:`\nu_m` and the true peak flux at
        :math:`\nu_a` is

        .. math::

            F_{\nu,\rm pk}
            =
            F_{m,0}
            \left(\frac{\nu_a}{\nu_m}\right)^{-(p-1)/2}.

        The absorption frequency in this regime is given by the SSA
        matching condition derived above,

        .. math::

            \nu_a
            =
            \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{2/5}
            \nu_m^{1/5}.

        Combining these expressions and solving self-consistently for
        :math:`\nu_a` yields

        .. math::

            \boxed{
            \nu_a
            =
            \left(\frac{F_{m,0}}{2 m_e \Omega \gamma_m}\right)^{2/(p+4)}
            \nu_m^{(p+2)/(p+4)}.
            }

        One may compute this candidate value :math:`\nu_a^{(2)}` and
        verify that :math:`\nu_m < \nu_a^{(2)} < \nu_{\rm max}` to confirm
        that this regime applies.

    .. tab-item:: Spectrum 3 :math:`(\nu_a < \nu_m < \nu_c < \nu_{\rm max})`

        In this case, the absorption frequency lies below the injection
        frequency, so the true peak of the SED occurs at
        :math:`\nu_m`. The presumed optically thin normalization
        :math:`F_{m,0}` therefore directly corresponds to the true peak
        flux,

        .. math::

            F_{\nu,\rm pk} = F_{m,0}.

        The absorption frequency is precisely that derived previously since the
        peak and the normalization are coincident:

        .. math::

            \boxed{
            \nu_a = \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{6/13} \nu_m^{1/13}.
            }

    .. tab-item:: Spectrum 4 :math:`(\nu_m < \nu_a < \nu_c < \nu_{\rm max})`

        In this case, the absorption frequency lies above the injection
        frequency, so the true peak of the SED occurs at
        :math:`\nu_a`. The normalization must therefore be propagated
        from :math:`\nu_m` up to :math:`\nu_a` along the optically thin
        segment :math:`F_\nu \propto \nu^{-(p-1)/2}`.

        The relationship between the presumed optically thin
        normalization at :math:`\nu_m` and the true peak flux at
        :math:`\nu_a` is

        .. math::

            F_{\nu,\rm pk}
            =
            F_{m,0}
            \left(\frac{\nu_a}{\nu_m}\right)^{-(p-1)/2}.

        The absorption frequency in this regime is given by the SSA
        matching condition derived above,

        .. math::

            \nu_a
            =
            \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{2/5}
            \nu_m^{1/5}.

        Combining these expressions and solving self-consistently for
        :math:`\nu_a` yields

        .. math::

            \boxed{
            \nu_a
            =
            \left(\frac{F_{m,0}}{2 m_e \Omega \gamma_m}\right)^{2/(p+4)}
            \nu_m^{(p+2)/(p+4)}.
            }

        One may compute this candidate value :math:`\nu_a^{(2)}` and
        verify that :math:`\nu_m < \nu_a^{(2)} < \nu_{\rm max}` to confirm
        that this regime applies.

    .. tab-item:: Spectrum 5 :math:`(\nu_a < \nu_c < \nu_m < \nu_{\rm max})`

        In this case, the absorption frequency lies below the cooling
        frequency, so the true peak of the SED occurs at
        :math:`\nu_c`. The presumed optically thin normalization
        :math:`F_{c,0}` therefore directly corresponds to the true peak
        flux,

        .. math::

            F_{\nu,\rm pk} = F_{c,0}.

        The absorption frequency is precisely that derived previously since the
        peak and the normalization are coincident:

        .. math::

            \boxed{
            \nu_a
            =
            \left(\frac{F_{c,0}}{2 m_e \Omega \gamma_m}\right)^{6/13}
            \nu_m^{3/13}
            \nu_c^{-2/13}.
            }

    .. tab-item:: Spectrum 6 :math:`(\nu_c < \nu_a < \nu_m < \nu_{\rm max})`

        In this case, the absorption frequency lies above the cooling frequency,
        :math:`\nu_a`. The normalization must therefore be propagated
        from :math:`\nu_c` up to :math:`\nu_a` along the optically thin
        segment :math:`F_\nu \propto \nu^{-p/2}`.

        The relationship between the presumed optically thin
        normalization at :math:`\nu_c` and the true peak flux at
        :math:`\nu_a` is

        .. math::

            F_{\nu,\rm pk}
            =
            F_{c,0}
            \left(\frac{\nu_a}{\nu_c}\right)^{-1/2}.

        The absorption frequency in this regime is given by the SSA
        matching condition derived above,

        .. math::

            \nu_a
            =
            \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{2/5}
            \nu_m^{1/5}.

        Combining these expressions and solving self-consistently for
        :math:`\nu_a` yields

        .. math::

            \boxed{
            \nu_a
            =
            \left(\frac{F_{m,0}}{2 m_e \Omega \gamma_m}\right)^{1/3}
            \nu_m^{1/6} \nu_c^{1/6}.
            }

    .. tab-item:: Spectrum 7 :math:`(\nu_c, \nu_m < \nu_a < \nu_{\rm max})`

        In this case, the absorption frequency lies above both the cooling and
        the injection frequencies. Therefore, the true peak of the SED occurs at
        :math:`\nu_a` and the normalization must be propagated. It turns out that,
        by coincidence, the propagation is identical whether we are fast cooling or
        slow cooling; however, the normalization of the underlying electron population
        is different in each case (as is the resulting normalization flux).

        Let :math:`F_{0}` be the appropriate normalization at either :math:`\nu_c` (fast cooling)
        or :math:`\nu_m` (slow cooling). In either case, the true peak of the SED occurs at
        :math:`\nu_a` and the normalization must be propagated from
        :math:`\nu_c` or :math:`\nu_m` up to :math:`\nu_a`. In the fast-cooling case,

        .. math::

            F_{\rm pk} = F_{c,0} \left(\frac{nu_m}{nu_c}\right)^{-1/2}
                                         \left(\frac{\nu_a}{\nu_m}\right)^{-p/2}
                               = F_{c,0} \nu_m^{(p-1)/2} \nu_c^{1/2} \nu_a^{-p/2}.

        In the slow-cooling case,

        .. math::

            F_{\rm pk} = F_{m,0} \left(\frac{\nu_c}{\nu_m}\right)^{-(p-1)/2}
                                         \left(\frac{\nu_a}{\nu_c}\right)^{-p/2}
                               = F_{m,0} \nu_m^{(p-1)/2} \nu_c^{1/2} \nu_a^{-p/2}.

        Since the propagation is identical in both cases, we may simply write

        .. math::

            F_{\rm pk} = F_{0} \nu_m^{(p-1)/2} \nu_c^{1/2} \nu_a^{-p/2}.

        The absorption frequency in this regime is given by the SSA
        matching condition derived above,

        .. math::

            \nu_a
            =
            \left(\frac{F_{\nu,\rm pk}}{2 m_e \Omega \gamma_m}\right)^{2/5}
            \nu_m^{1/5}.

        With these expressions, we may solve self-consistently for
        :math:`\nu_a`:

        .. math::

            \nu_a = \left(\frac{F_0}{2 m_e \Omega \gamma_m}\right)^{4/(10+p)}
                     \nu_m^{2p/(10+p)} \nu_c^{1/(10+p)}.

References
----------
.. footbibliography::
