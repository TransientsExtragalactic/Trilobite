.. _synch_sed_theory:
===========================================
Methods: Single-Zone Synchrotron SEDs
===========================================

.. seealso::

    - :ref:`synchrotron_theory` for a broad discussion of the relevant synchrotron theory used in
      this theory module.
    - :ref:`synchrotron_cooling_theory` for a discussion of synchrotron cooling and populations. Some of this
      theory is relevant for the discussion in this note.

This document is intended to develop the theory behind Triceratops' implementation of **single-zone synchrotron SEDs**. This is a critical task
on the basis that the literature, spanning some 40 years at this point, is highly fractured in its methodology concerning
these SEDs and their construction (see e.g. :footcite:t:`Chevalier1998SynchrotronSelfAbsorption`, :footcite:t:`ChevalierFranssonHandbook`,
:footcite:t:`GranotSari2002SpectralBreaks`, :footcite:t:`GaoSynchrotronReview2013`, :footcite:t:`2025ApJ...992L..18S`, and
references therein). Our goal in this presentation is to (a) provide a background to users who are not familiar with
the details of this theory and, more importantly, (b) to establish our methodology in as robust a manner as possible ensuring
that Triceratops remains extensible, reproducible, and accurate.

.. contents::
    :local:
    :depth: 2

Overview
--------

The theory of single-zone synchrotron SEDs as described in this theory note is highly non-trivial and must be done
with exceeding care.  In the subsections
below, we will describe in detail each element of SED construction; however, for those familiar with the literature, it
should be noted that we follow the formulation of :footcite:t:`GranotSari2002SpectralBreaks` to construct our SEDs and
to describe the various power-law components. Because we include an additional break frequency (:math:`\nu_{\rm max}`) a
few additional scenarios are considered here which are not therein mentioned.

Because :footcite:t:`GranotSari2002SpectralBreaks` use a methodology for normalization which is not fully generalizable without
numerical quadrature, we choose instead to follow the approximations described in :footcite:t:`sari1999jets` and used throughout
the literature (see e.g. :footcite:t:`2025ApJ...992L..18S`).

.. important::

    Throughout this note, for the sake of clarity, we restrict ourself to the **comoving frame of the emitting region**.
    The implementations of these SEDs in the code use the appropriate Lorentz transformations to ensure that the SEDs are
    correctly computed in the observer frame. However, the theory is more straightforwardly described in the comoving
    frame and so we adopt this convention here.

Foundational Assumptions
^^^^^^^^^^^^^^^^^^^^^^^^^

In developing the theory here described, a number of assumptions are made. Of these, a few are foundational to
the model and should not, under any circumstances, be violated. These include:

- The emission region is a **single zone**, meaning that electrons emitting synchrotron are all well described by the
  same physical parameters (magnetic field strength, electron distribution, etc.). This need not imply that the
  source is a point-source; only that its combined emission may be described as a single, effective radiation zone.
- The **injected electron distribution** is a power law in Lorentz factor, with a minimum Lorentz factor
  :math:`\gamma_m`, a power-law index :math:`p`, and (optionally) a maximum Lorentz factor :math:`\gamma_M`.

.. todo:: Are there any other foundational assumptions to be made.

Parameters, Hyperparameters, and Physical Quantities
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

.. dropdown:: SED Parameters

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

.. dropdown:: SED Hyper-Parameters

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

.. dropdown:: SED Internal Parameters

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


---

.. _single_zone_sed_structure:
Single-Zone SED Structure
-------------------------

As described above, the single-zone SED is a **broken power law** with a number of break frequencies and a normalization.
The exact structure of the SED depends on the relative ordering of the break frequencies and the value of the hyper-parameters.
In keeping with the formalism of :footcite:t:`GranotSari2002SpectralBreaks`, we adopt a mathematical formalism in which
the SED is described as **the product of a normalized power-law segment** and a **series of multiplicative factors** which
describe the **transitions between power-law segments**.

SED Surgery
^^^^^^^^^^^

Each SED is **anchored** by a **single power-law segment** (SPLS)
determined by the dominant break frequency and the relevant
hyper-parameters. This takes the form

.. math::

    F_\nu = F_{\rm pk}
    \left(\frac{\nu}{\nu_{\rm brk}}\right)^{\alpha_0},

where :math:`\nu_{\rm brk}` is the dominant break frequency and
:math:`\alpha_0` is the spectral slope of the anchored segment.

Additional breaks are introduced through **SED Surgery**, in which
multiplicative, scale-free modification factors are applied at the
relevant break frequencies.

For a break at :math:`\nu_i`, we introduce the kernel

.. math::

    \tilde f(x)
    =
    \left[
        1 + x^{(a_{\rm right} - a_{\rm left})/s}
    \right]^s,
    \qquad
    x = \frac{\nu}{\nu_i},

where:

- :math:`a_{\rm left}` is the spectral slope for :math:`\nu < \nu_i`,
- :math:`a_{\rm right}` is the spectral slope for :math:`\nu > \nu_i`,
- :math:`s` is the smoothness parameter.

This kernel enforces a transition from
:math:`a_{\rm left}` on the left to
:math:`a_{\rm right}` on the right,
with total slope change

.. math::

    \Delta\alpha = a_{\rm right} - a_{\rm left}.

The magnitude :math:`|s|` controls the sharpness of the break,
with smaller values approaching the sharp broken power-law limit.

The sign of :math:`s` must be determined based on the curvature of the transition:

- :math:`s > 0` produces a concave-down transition.
- :math:`s < 0` produces a concave-up transition.

The full SED is therefore

.. math::

    F_\nu
    =
    F_{\rm pk}
    \left(\frac{\nu}{\nu_{\rm brk}}\right)^{\alpha_0}
    \prod_i \tilde f_i\!\left(\frac{\nu}{\nu_i}\right),

and in the limit :math:`|s|\to 0` this reduces exactly to the
canonical broken power-law spectrum anchored at :math:`F_{\rm pk}`.


Break Frequencies
^^^^^^^^^^^^^^^^^

Before proceeding to discuss the construction of various SEDs, it is necessary to describe the precise methodology
with which we compute the various break frequencies used in the SEDs. In each of the following sections, we describe
this methodology in detail.

.. _single_zone_injection_frequencies:
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

            \nu_m = \left<\nu_{\rm synch}(\gamma)\right>_{\sin \alpha} = \frac{3 q B}{2 \pi^2 m c}\left[\gamma_{\min}\right]^2,

        In the same manner, the maximum injection frequency is

        .. math::

            \nu_{\rm max} = \left<\nu_{\rm synch}(\gamma)\right>_{\sin \alpha} = \frac{3 q B}{2 \pi^2 m c}\left[\gamma_{\max}\right]^2.

    .. tab-item:: Fixed Pitch Angle

        In the case of a fixed pitch angle :math:`\alpha`, the injection frequencies are given by the standard
        characteristic synchrotron frequency from :eq:`eq_synch_frequency` from :ref:`synchrotron_theory`:

        .. math::
            :label: injection_frequency_min

            \nu_m = \nu_{\rm char}(\gamma_{\min}) = \frac{3 q B \sin \alpha}{4 \pi m c}\left[\gamma_{\min}\right]^2,

        and

        .. math::

            \nu_{\rm max} = \nu_{\rm char}(\gamma_{\max}) = \frac{3 q B \sin \alpha}{4 \pi m c}\left[\gamma_{\max}\right]^2.


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

.. _single_zone_cooling_frequency:
The Cooling Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~

**PARAMETER TYPE:** Free Parameter

Consider a population of electrons subject to a cooling process with a cooling rate :math:`\Lambda(\gamma)`. Electrons
with energy :math:`E = m_e c^2 \gamma` will then cool on a timescale (in the comoving frame)

.. math::

    t_{\rm cool}(\gamma) = \frac{E}{\Lambda(\gamma)} = \frac{m_e c^2 \gamma}{\Lambda(\gamma)}.

If, in order to cool significantly, the dynamical time must exceed the cooling time for a particular energy, we can
define the **cooling Lorentz factor** :math:`\gamma_c` as the Lorentz factor for which the cooling timescale
equals the dynamical timescale :math:`t_{\rm dyn}` of the system:

.. math::

    \boxed{
    t_{\rm cool}(\gamma_c)
    =
    \frac{m_e c^2 \gamma_c}{\Lambda(\gamma_c)}
    =
    t_{\rm dyn}.
    }

This then implies

.. math::
    :label: cooling_lorentz_factor

    \boxed{
    \gamma_c
    =
    \frac{m_e c^2}{\Lambda(\gamma_c) t_{\rm dyn}}.
    }

The corresponding **cooling frequency** is then given by :eq:`eq_synch_frequency` from :ref:`synchrotron_theory` as

.. math::
    :label: cooling_frequency

    \boxed{
    \nu_c
    =
    \frac{3 q B \sin \alpha}{4 \pi m c} \gamma_c^2
    =
    \frac{3 q m_e c^3}{4\pi}
    \left(
    \frac{B \sin \alpha}
    {\left[\Lambda(\gamma_c)\right]^2\, t_{\rm dyn}^{\,2}}
    \right).
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
    \left[\frac{B\sin\alpha}{\left[\Lambda(\gamma_c)\right]^2 t_{\rm dyn}^2}\right]
    }

.. hint::

    In general, one selects a cooling mechanism (see :ref:`synchrotron_theory` and :ref:`synchrotron_microphysics`)
    and computes :math:`\gamma_c(t)` as a function of time and provides that to the SED. Cooling is implemented in
    :mod:`~radiation.synchrotron.cooling`.

.. _single_zone_ssa_frequency:
The Absorption Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~

**PARAMETER TYPE:** Internal Parameter

The self-absorption frequency is a less trivial quantity to compute, as it depends on the radiative transfer
properties of the source and is therefore dependent on the SED one is using. This creates a circular dependency
which must be resolved by considering every possible SED configuration given a known :math:`\nu_m` and :math:`\nu_c`,
computing the value of :math:`\nu_a` in each case and then checking for self-consistency with the assumed SED and
its assumptions.

In the most rigorous sense, the absorption frequency :math:`\nu_a` is determined by the condition that the
optical depth to self-absorption equals unity:

.. math::

    \tau_{\nu_{a}} = \alpha_{\nu_a} L = 1,

The form of :math:`\alpha_\nu` depends explicitly on the structure of the absorbing electron population (see
:ref:`synchrotron_theory` for details). One could, in principle, perform these computations in full detail; however,
an alternative approach has been developed in the literature :footcite:p:`duran2013radius` which allows for
approximate expressions for :math:`\nu_a`.

We assume, as was done in the development of the normalization approach, that the absorption at a particular frequency
is dominated by a mono-energetic population of electrons. In such a case, the optically thick emission from the source
should be well approximated by a blackbody with brightness temperature :math:`kT_{b} = \gamma_\nu m_e c^2`, where
:math:`\gamma_\nu` is the Lorentz factor corresponding to the dominant absorbing population of electrons:

.. math::

    \gamma_\nu = {\rm max}\left(\gamma_a, {\rm min}\left(\gamma_c,\gamma_m\right)\right).

This corresponds to a source function :math:`S_\nu = 2\left[\nu_a\right]^2 m_e \gamma_\nu`. The corresponding flux
:math:`F_\nu` should then be

.. math::

    F_\nu = 2\left[\nu_a\right]^2 m_e \gamma_\nu \frac{A}{D_A^2},

where :math:`A` is the effective radiating area of the source. Equating this to the **optically thin** flux
from the normalized SED at :math:`\nu_a` then allows one to solve for :math:`\nu_a`. In practice, we instead parameterize
this flux density in terms of the **effective angular radiating area** :math:`\Omega = A/D_A^2` so that the distance
does not need to be provided as a hyper-parameter.

.. important::

    For code users, it is important to note that while the inclusion of :math:`\nu_a` would seem to be similar
    in complexity to the other breaks, using an SSA enabled SED **REQUIRES** the user to provide some elements of
    the underlying geometry of the source (i.e., the effective area and volume) in order to compute :math:`\nu_a`
    in each dependent scenario.

    Thus, an SSA SED function :math:`F_{\rm \nu}(\nu; \nu_m,\nu_c,\nu_a,\ldots)` should be thought of more properly
    as

    .. math::

        F_{\rm \nu}(\nu; \nu_m,\nu_c,A,V,\ldots),

    where now :math:`\nu_a = f(\nu_m,\nu_c,A,V,\ldots)` is computed internally.

Computing The Self-Absorption Frequency
#######################################

In general, we assume that, at :math:`\nu_a` (an unknown break), the flux density is given by the blackbody
approximation described above:

.. math::

    F_\nu = 2\left[\nu_a\right]^2 m_e \gamma_\nu \Omega = F_{\nu, \rm thin}(\nu_a),

where :math:`F_{\nu, \rm thin}(\nu_a)` is the flux density at :math:`\nu_a` computed from the optically thin SED,
and :math:`\gamma_{\nu}` is the Lorentz factor of the **dominant absorbing electrons**:

.. math::

    \gamma_\nu = {\rm max}\left(\gamma_a, {\rm min}\left(\gamma_c,\gamma_m\right)\right).

This defines an implicit equation for :math:`\nu_a` which may be solved algebraically (or numerically if necessary).
Thus, for any set of break frequencies and hyper-parameters, one may compute :math:`\nu_a` by solving the equation
and then construct the SED using the computed frequency.

The Single Power-Law Segment (SPLS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

In addition to each of the SPLs described above and the corresponding scale-free transition kernels, SEDs which have
a high frequency cutoff :math:`\nu_{\rm max}` also have an additional break. In this regime, the SED transitions from any of the optically thin
segments (SPL F, SPL G, or SPL H) to the exponential cutoff segment (SPL I). Because the
exponential cutoff is not a power law, we do not provide SBPL representations for these breaks, but instead
provide **exponential cutoff functions**. In the smoothed case described above, we instead need a **scale-free exponential cutoff function** which does
not interfere with the normalization of the SED at lower frequencies. We therefore define:

.. math::

    \tilde{\Phi}(\nu,\nu_{\max}) = \begin{cases}
        1, & \nu < \nu_{\max} \\
        \left(\frac{\nu}{\nu_{\max}}\right)^{1/2}
        \exp\left(1 -\frac{\nu}{\nu_{\max}}\right), & \nu \geq \nu_{\max}.
    \end{cases}

.. _single_zone_sed_normalization:
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

The Normalizing Flux and Frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Power-Law Propagation of the Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The flux normalizations derived above (:math:`F_{m,0}` at :math:`\nu_m` in slow cooling and
:math:`F_{c,0}` at :math:`\nu_c` in fast cooling) set the overall amplitude of the **optically thin**
spectrum.  However, in general these *normalization frequencies* do not coincide with the
**peak of the observed SED**:

- In the absence of self-absorption, the observed peak coincides with the optically thin peak.
- When synchrotron self-absorption is important, the observed peak occurs at :math:`\nu_a` and the
  optically thin peak may be obscured.

To connect the theoretical normalization (:math:`F_{m,0}` or :math:`F_{c,0}`) to the parameter
:math:`F_{\rm pk}` used to anchor the SED, we propagate the normalization across the relevant
optically thin power-law segments.

In its most general form, if two frequencies :math:`\nu_1` and :math:`\nu_2` lie within the same
optically thin power-law segment with slope :math:`\alpha` (i.e. :math:`F_\nu \propto \nu^\alpha`),
then

.. math::

    F_\nu(\nu_2)
    =
    F_\nu(\nu_1)\left(\frac{\nu_2}{\nu_1}\right)^{\alpha}.

When multiple breaks lie between :math:`\nu_1` and :math:`\nu_2`, propagation is performed
piecewise across each segment:

.. math::

    F_\nu(\nu_2)
    =
    F_\nu(\nu_1)\prod_{j}
    \left(\frac{\nu_{j+1}}{\nu_{j}}\right)^{\alpha_j},

where :math:`\{\nu_j\}` is the ordered list of intervening break frequencies (including endpoints)
and :math:`\alpha_j` is the spectral slope on the segment :math:`(\nu_j,\nu_{j+1})`.

In practice, we use this propagation in two distinct ways:

1. **Optically thin peak (no SSA-dominated peak)**

   If the spectrum is optically thin at its maximum, then the SED peak occurs at a known break
   frequency :math:`\nu_{\rm pk}` (typically :math:`\nu_m` or :math:`\nu_c` depending on the cooling
   regime), and the peak flux is simply the corresponding normalization:

   .. math::

      (\nu_{\rm pk},F_{\rm pk}) =
      \begin{cases}
      (\nu_m, F_{m,0}), & \text{slow cooling} \\
      (\nu_c, F_{c,0}), & \text{fast cooling}.
      \end{cases}

   In this case no propagation is required.

2. **SSA-dominated peak**

   If self-absorption sets the observed peak, then :math:`\nu_{\rm pk}=\nu_a` and the observed peak
   flux :math:`F_{\rm pk}=F_\nu(\nu_a)` must be obtained by propagating the optically thin
   normalization up to :math:`\nu_a` along the appropriate optically thin slopes, i.e.

   .. math::

      F_{\rm pk}
      =
      F_{\rm norm}
      \times
      \left[\text{optically thin propagation from }\nu_{\rm norm}\text{ to }\nu_a\right],

   where :math:`(\nu_{\rm norm},F_{\rm norm})` is :math:`(\nu_m,F_{m,0})` in slow cooling or
   :math:`(\nu_c,F_{c,0})` in fast cooling.

.. _single_zone_sed_inversion:
Inversion
---------

The final theoretical topic which must be discussed in the context of our single-zone SED implementation is that of
inversion. By inversion, we mean the process of taking an observed SED and inferring the underlying physical parameters which
produced it. This is a non-trivial process which is often complicated by the fact that knowledge of the
peak flux :math:`F_{\rm pk}` and the peak frequency :math:`\nu_{\rm pk}` alone is insufficient to break the degeneracies
between the various physical parameters. In general, one must have some additional information (e.g., from dynamics or other
observables) to break these degeneracies and infer the physical parameters.

It is therefore standard practice to invoke a **closure relation** to connect the SED parameters to physical parameters.
As is the case for normalization, where one must invoke a **closure relation** to connect the SED to physical parameters,
one must also invoke a closure relation to perform inversion.

Triceratops does not provide an exhaustive set of closure relations as, following our development philosophy, we prefer
to allow for modeling flexibility rather than prescribing a particular set of assumptions. However, we do provide a few commonly
used closure relations in the documentation, and we encourage users to implement their own closure relations as needed
for their particular applications.

.. hint::

    In many contexts, closure is not necessary: the user may construct a forward model which includes the relevant
    physical assumptions in the construction of the dynamical parameters used in normalization. The process of inference will
    then reveal the underlying physical parameters without the need for an explicit inversion of the SED. However, in
    some cases, it may be desirable to perform an explicit inversion of the SED, and in these cases, closure relations are required.


Closure Assumptions
^^^^^^^^^^^^^^^^^^^

In what follows, we assume that the following observables are available for inversion:

- The **rest-frame** break frequency :math:`\nu_{\rm brk}` and its physical interpretation
  (e.g. :math:`\nu_m` or :math:`\nu_c`).
- The **rest-frame** flux density at that break frequency,
  :math:`F_{\rm brk}`.

To close the system and permit inversion to physical parameters, we adopt the
following physical assumptions:

The synchrotron emission arises from a population of relativistic electrons
in either the fast- or slow-cooling regime (see :ref:`synchrotron_cooling_theory`).
In both cases, the electron distribution may be written in separable form:

.. math::

    N(\gamma) = N_0 f(\gamma),

where :math:`f(\gamma)` encodes the shape of the distribution and possesses a known
first moment

.. math::

    M_\gamma^{(1)} \equiv \int \gamma f(\gamma)\, d\gamma.

We assume that fixed fractions of the post-shock internal energy density
are deposited into relativistic electrons and magnetic fields:

.. math::

    u_e = \epsilon_e u_{\rm int},
    \qquad
    u_B = \epsilon_B u_{\rm int}
    = \frac{B^2}{8\pi}.

Combining these relations yields the normalization closure

.. math::

    N_0
    =
    \frac{\epsilon_e B^2}
         {8\pi \epsilon_B m_e c^2 M_\gamma^{(1)}}
    =
    \tilde{N_0} B^2,

where :math:`\tilde{N_0}` is introduced to simplify the algebra.

We further assume that the emitting region is spherical with radius :math:`R`
and effective volume

.. math::

    V_{\rm eff} = \frac{4\pi}{3} R^3 f_V,

and angular size

.. math::

    \Omega = \frac{\pi R^2}{D_A^2} f_A.

For notational convenience, we introduce the constants

.. tab-set::

    .. tab-item:: Isotropic Pitch Angle

        .. math::

            \begin{aligned}
            \nu_{\rm synchrotron}(\gamma) &= c_{1,\rm ISO} B \gamma^2, \\
            F_{m,0} &= Q_{m,\rm ISO} B^3 R^3 \tilde{N_0} , \\
            F_{c,0} &= Q_{c,\rm ISO} B^3 R^3 \tilde{N_0} ,
            \end{aligned}

        where

        .. math::

            \begin{aligned}
            c_{1,\rm ISO} &= \frac{3 q_e}{2\pi^2 m_e c}, \\
            Q_{m,\rm ISO} &= \frac{4}{3}\pi f_V \chi_{\rm ISO} \gamma_m^{1-p} D_L^{-2}, \\
            Q_{c,\rm ISO} &= \frac{4}{3}\pi f_V \chi_{\rm ISO}  \gamma_m^2 \gamma_c^{-1} D_L^{-2}.
            \end{aligned}

    .. tab-item:: Fixed Pitch Angle

        .. math::

            \begin{aligned}
            \nu_{\rm synchrotron}(\gamma) &= c_{1} B \gamma^2 \sin\alpha, \\
            F_{m,0} &= Q_{m,0} B^3 R^3 \tilde{N_0} \sin\alpha, \\
            F_{c,0} &= Q_{c,0} B^3 R^3 \tilde{N_0} \sin\alpha,
            \end{aligned}

        where

        .. math::

            \begin{aligned}
            c_{1} &= \frac{3 q_e}{4\pi m_e c}, \\
            Q_{m,0} &= \frac{4}{3}\pi f_V \chi \gamma_m^{1-p} D_L^{-2}, \\
            Q_{c,0} &= \frac{4}{3}\pi f_V \chi \gamma_m^2 \gamma_c^{-1} D_L^{-2}.
            \end{aligned}

Under these assumptions, each of the SED forms described above may be explicitly inverted.

.. note::

    Going forward, we present the **fixed pitch angle** case, but present critical results in both
    scenarios.

Inverting from Optically Thin Peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest case to describe is the inversion based on the SEDs **with optically thin peaks** at either
:math:`\nu_m` or :math:`\nu_c`. In these cases, the break frequency is directly related to the characteristic
synchrotron frequency of electrons at either :math:`\gamma_m` or :math:`\gamma_c`, and the flux normalization
at the break frequency is **directly related to the optically thin normalization** at that frequency.
This allows for a straightforward inversion to physical parameters.

To begin the process, we recognize that the break frequency is related to the Lorentz factor of the
electrons responsible for the emission at that frequency through the characteristic synchrotron frequency:


.. math::
    :label: inversion_nu_m_fixed

    \nu_m = \frac{3 q B \sin \alpha}{4 \pi m c}\left[\gamma_{\min}\right]^2 = c_{1} B \sin \alpha \gamma_{\min}^2.

This immediately allows us to invert for the magnetic field strength as

.. tab-set::

    .. tab-item:: Isotropic Pitch Angle

        .. math::
            :label: inversion_B_iso_thin

            \boxed{
            B = \frac{\nu_m}{c_{1,\rm ISO} \gamma_{\min}^2},
            }

    .. tab-item:: Fixed Pitch Angle

        .. math::
            :label: inversion_B_fixed_thin

            \boxed{
            B \sin \alpha = \frac{\nu_m}{c_{1} \gamma_{\min}^2}.
            }

The flux normalization at the break frequency may then be used to determine the radius:

.. tab-set::

    .. tab-item:: Isotropic Pitch Angle

        .. math::
            :label: inversion_R_iso_thin

            \boxed{
            R = \left(\frac{F_{m,0}}{Q_{m,\rm ISO} B^3 \tilde{N_0}}\right)^{1/3},
            }

    .. tab-item:: Fixed Pitch Angle

        .. math::
            :label: inversion_R_fixed_thin

            \boxed{
            R = \left(\frac{F_{0}}{Q_{0} B^3 \tilde{N_0} \sin \alpha}\right)^{1/3},
            }

where :math:`F_0` and :math:`Q_0` are the appropriate normalization flux and constant for the cooling regime in question
(i.e., :math:`F_{m,0}` and :math:`Q_{m,0}` for slow cooling and :math:`F_{c,0}` and :math:`Q_{c,0}` for fast cooling).

A similar technique can be employed when the break appears at the cooling frequency instead of the injection
frequency with the exception that one must replace :math:`\gamma_m` with :math:`\gamma_c` when computing the
magnetic field.

Inverting for Optically Thick Peaks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the SSA frequency :math:`\nu_a` sets the observed peak, the inversion process is more complex. In this case,
the observed peak frequency and flux are not directly related to the characteristic synchrotron frequency
of a particular electron population, and the flux normalization at the peak frequency is not directly related to the
optically thin normalization. Instead, one must use the known shape of the SED and the relevant break frequencies
to "extrapolate" from the optically thin normalization to the observed peak frequency and flux.

Regardless of the spectral regime, the peak flux will correspond to the blackbody flux at :math:`\nu_a`:

.. math::
    :label: inversion_F_a_fixed

    F_\nu(\nu_a)
    =
    2\nu_a^2 m_e \gamma_a \Omega
    =
    2 m_e \Omega \gamma_m
    \left(\frac{\nu_a}{\nu_m}\right)^{1/2}
    \nu_a^2.

Defining :math:`P_0` such that

.. tab-set::

    .. tab-item:: Isotropic Pitch Angle

        .. math::

            P_{0,\rm ISO} = 2\pi m_e f_A D_A^{-2} c_{1,\rm ISO}^{-1/2},

    .. tab-item:: Fixed Pitch Angle

        .. math::

            P_0 = 2\pi m_e f_A D_A^{-2} c_{1}^{-1/2} \sin^{-1/2} \alpha,

the result is a general condition for the SSA frequency:

.. math::

    F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_0 R^2 B^{-1/2}

A second condition may be obtained by requiring that the peak flux at the peak frequency be consistent with
the normalization as described in the previous section and used in the optically thin case. I general, this
results in a set of equations

.. math::

    F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_0 R^2 B^{-1/2},
    F_{\rm brk}^\alpha \nu_{\rm brk}^\beta = C R^\gamma B^\delta,

where :math:`C` is a constant derived from the normalization and :math:`\alpha`, :math:`\beta`, :math:`\gamma`, and :math:`\delta`
are constants which depend on the cooling
regime and the spectral regime. Solving this system of equations then allows for the inversion to physical parameters.
The case is solved explicitly for each of the relevant regimes in their corresponding documentation pages.

Single-Zone SEDs
-----------------

The Single-Zone Power-Law SED
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

    F_\nu = F^{(D,G)}_\nu \tilde{\Phi}(\nu,\nu_{\max}).

.. rubric:: Normalization

In this SED the only spectral breaks present occur at the characteristic synchrotron
frequencies associated with the minimum and maximum electron Lorentz factors,
:math:`\nu_m` and :math:`\nu_{\max}`, respectively. The peak of the broadband SED occurs at the
**injection frequency** :math:`\nu_m`, corresponding to synchrotron emission from electrons with
Lorentz factor :math:`\gamma_{\rm min}`.

The normalization of the canonical SED is fixed by the following physical parameters:

.. dropdown:: Normalization Parameters

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

The normalization of the power-law SED is most naturally anchored at the injection frequency
:math:`\nu_m`. For a given choice of pitch-angle treatment, the magnetic field strength :math:`B` therefore
uniquely determines the location of the spectral peak.

The flux density at this frequency sets the overall normalization of the SED.

.. tab-set::

    .. tab-item:: Fixed Pitch Angle

        For the case of a
        fixed pitch angle, the flux normalization at :math:`\nu_m` is given by
        :eq:`slow_cooling_norm` as

        .. math::

            F_{m,0} \approx
            \chi \, (B \sin\alpha)\,
            N_0 \,
            \gamma_{\rm min}^{\,1-p}
            \frac{V_{\rm eff}}{D_L^2},

    .. tab-item:: Isotropic Pitch Angle

        For the case of isotropic pitch-angle averaging, the flux normalization at :math:`\nu_m` is given by
        :eq:`slow_cooling_norm_iso` as

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

.. rubric:: Inversion

The inversion is the simple case of an optically thin peak at :math:`\nu_m`.

.. dropdown:: Inversion Parameters

    .. list-table::
        :widths: 10 35 85
        :header-rows: 1

        * - Parameter
          - Name
          - Notes
        * - :math:`F_{\rm brk}`
          - Break Flux
          - Flux density at the break frequency used for inversion.
        * - :math:`\nu_{\rm brk}`
          - Break Frequency
          - Frequency of the break used for inversion.
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

Using :eq:`inversion_nu_m_fixed`, we can invert for the magnetic field strength. Using :eq:`inversion_R_fixed_thin`, we can then
invert for the radius of the emitting region. The remaining parameters are either fixed by assumptions (e.g., :math:`p`)
or are degenerate with the normalization (e.g., :math:`\epsilon_e` and :math:`\epsilon_B`).

.. tab-set::

    .. tab-item:: Isotropic Pitch Angle

        The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

        .. math::

            \boxed{
            B = \frac{\nu_{\rm brk}}{c_{1,\rm ISO} \gamma_{\min}^2},
            \qquad
            R = \left(\frac{F_{\rm brk}}{Q_{m,\rm ISO} B^3 \tilde{N_0}}\right)^{1/3}.
            }

    .. tab-item:: Fixed Pitch Angle

        The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

        .. math::

            \boxed{
            B = \frac{\nu_{\rm brk}}{c_{1} \gamma_{\min}^2 \sin\alpha},
            \qquad
            R = \left(\frac{F_{\rm brk}}{Q_{m,0} B^3 \tilde{N_0} \sin \alpha}\right)^{1/3}.
            }

Cooling SEDs
^^^^^^^^^^^^

Another simple scenario worth considering is the SED from a synchrotron source with non-negligible cooling
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

.. rubric:: Parameters

.. list-table::
    :widths: 35 85
    :header-rows: 1

    * - Parameter Class
      - Parameters
    * - **Free Parameters**
      - :math:`\nu_m`, :math:`\nu_{\max}`, :math:`F_{\nu,\rm pk}`, :math:`\nu_c`.
    * - **Derived Parameters**
      - None.
    * - **Break Frequencies**
      - :math:`\nu_m`, :math:`\nu_{\max}`, :math:`\nu_c`.
    * - **Hyper-Parameters**
      - :math:`p`, :math:`s`.

.. rubric:: Spectral Regimes

.. tab-set::

    .. tab-item:: Fast Cooling (:math:`\nu_c < \nu_m`)

        In this spectrum, there are 4 SPL segments connected by 3 breaks. Because the population is rapidly cooled,
        the bulk of electrons are effectively reduced to :math:`\gamma_c` and the corresponding peak in the spectrum
        occurs at :math:`\nu_c`.

        .. dropdown:: Spectral Segments

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
        distribution function.

        .. rubric:: Normalization

        In this SED the only spectral breaks present occur at the characteristic synchrotron
        frequencies associated with the minimum (injection) frequency, the cooling frequency, and the maximum electron frequency,
        :math:`\nu_m`, :math:`\nu_c`, and :math:`\nu_{\max}`, respectively. The peak of the broadband SED occurs at the
        **cooling frequency** :math:`\nu_c`, corresponding to synchrotron emission from electrons with
        Lorentz factor :math:`\gamma_{\rm c}`.

        The normalization of the canonical SED is fixed by the following physical parameters:

        .. dropdown:: Normalization Parameters

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
                  - Lorentz factor above which electrons cool efficiently. In the no-cooling limit, :math:`\gamma_c \to \infty`.
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

        The normalization of the power-law SED is most naturally anchored at the injection frequency
        :math:`\nu_c`. For a given choice of pitch-angle treatment, the magnetic field strength :math:`B` therefore
        uniquely determines the location of the spectral peak.

        The flux density at this frequency sets the overall normalization of the SED.

        .. tab-set::

            .. tab-item:: Fixed Pitch Angle

                For the case of a
                fixed pitch angle, the flux normalization at :math:`\nu_c` is given by
                :eq:`fast_cooling_norm` as

                .. math::

                    F_{c,0} \approx
                    \chi \, (B \sin\alpha)\,
                    N_0 \,
                    \gamma_{\rm min}^2 \gamma_c^{-1}
                    \frac{V_{\rm eff}}{D_L^2},

            .. tab-item:: Isotropic Pitch Angle

                For the case of isotropic pitch-angle averaging, the flux normalization at :math:`\nu_m` is given by
                :eq:`fast_cooling_norm_iso` as

                .. math::

                    F_{c,0}^{\rm (iso)} \approx
                    \chi_{\rm iso}\,
                    B\,
                    N_0 \,
                    \gamma_{\rm min}^2 \gamma_c^{-1}
                    \frac{V_{\rm eff}}{D_L^2},

                where :math:`\chi_{\rm iso}` incorporates the pitch-angle averaging.

        The electron normalization :math:`N_0` is not a free parameter. Instead, it is determined by
        equipartition arguments through the choice of :math:`\epsilon_e` and :math:`\epsilon_B`, which
        relate the electron and magnetic energy densities to the post-shock internal energy density.
        Once :math:`N_0` is fixed in this manner, the flux at :math:`\nu_c` — and hence the normalization
        of the entire canonical synchrotron SED — is fully specified.


        .. rubric:: Inversion

        The inversion is the simple case of an optically thin peak at :math:`\nu_c`.

        .. dropdown:: Inversion Parameters

            .. list-table::
                :widths: 10 35 85
                :header-rows: 1

                * - Parameter
                  - Name
                  - Notes
                * - :math:`F_{\rm brk}`
                  - Break Flux
                  - Flux density at the break frequency used for inversion.
                * - :math:`\nu_{\rm brk}`
                  - Break Frequency
                  - Frequency of the break used for inversion.
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
                  - Lorentz factor above which electrons cool efficiently. In the no-cooling limit, :math:`\gamma_c \to \infty`.
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

        Using :eq:`inversion_nu_m_fixed` but instead using :math:`\nu_c`, we can invert for the magnetic field strength.
        Using :eq:`inversion_R_fixed_thin`, we can then
        invert for the radius of the emitting region. The remaining parameters are either fixed by assumptions (e.g., :math:`p`)
        or are degenerate with the normalization (e.g., :math:`\epsilon_e` and :math:`\epsilon_B`).

        .. tab-set::

            .. tab-item:: Isotropic Pitch Angle

                The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1,\rm ISO} \gamma_{c}^2},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{c,\rm ISO} B^3 \tilde{N_0}}\right)^{1/3}.
                    }

            .. tab-item:: Fixed Pitch Angle

                The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1} \gamma_{\min}^2 \sin\alpha},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{c,0} B^3 \tilde{N_0} \sin \alpha}\right)^{1/3}.
                    }

    .. tab-item:: Slow Cooling (:math:`\nu_m < \nu_c`)

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. dropdown:: Spectral Segments

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

        where we have selected to normalize at the (D,G) break at :math:`\nu_m`.

        .. rubric:: Normalization

        In this SED the only spectral breaks present occur at the characteristic synchrotron
        frequencies associated with the minimum (injection) frequency, the cooling frequency, and the maximum electron frequency,
        :math:`\nu_m`, :math:`\nu_c`, and :math:`\nu_{\max}`, respectively. The peak of the broadband SED occurs at the
        **injection frequency** :math:`\nu_c`, corresponding to synchrotron emission from electrons with
        Lorentz factor :math:`\gamma_{\rm m}`.

        The normalization of the canonical SED is fixed by the following physical parameters:

        .. dropdown:: Normalization Parameters

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
                  - Lorentz factor above which electrons cool efficiently. In the no-cooling limit, :math:`\gamma_c \to \infty`.
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

        The normalization of the power-law SED is most naturally anchored at the injection frequency
        :math:`\nu_m`. For a given choice of pitch-angle treatment, the magnetic field strength :math:`B` therefore
        uniquely determines the location of the spectral peak.

        The flux density at this frequency sets the overall normalization of the SED.

        .. tab-set::

            .. tab-item:: Fixed Pitch Angle

                For the case of a
                fixed pitch angle, the flux normalization at :math:`\nu_c` is given by
                :eq:`slow_cooling_norm` as

                .. math::

                    F_{m,0} \approx
                    \chi \, (B \sin\alpha)\,
                    N_0 \,
                    \gamma_{\rm min}^{\,1-p}
                    \frac{V_{\rm eff}}{D_L^2},

            .. tab-item:: Isotropic Pitch Angle

                For the case of isotropic pitch-angle averaging, the flux normalization at :math:`\nu_m` is given by
                :eq:`slow_cooling_norm_iso` as

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

        .. rubric:: Inversion

        The inversion is the simple case of an optically thin peak at :math:`\nu_m`.

        .. dropdown:: Inversion Parameters

            .. list-table::
                :widths: 10 35 85
                :header-rows: 1

                * - Parameter
                  - Name
                  - Notes
                * - :math:`F_{\rm brk}`
                  - Break Flux
                  - Flux density at the break frequency used for inversion.
                * - :math:`\nu_{\rm brk}`
                  - Break Frequency
                  - Frequency of the break used for inversion.
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
                  - Lorentz factor above which electrons cool efficiently. In the no-cooling limit, :math:`\gamma_c \to \infty`.
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

        Using :eq:`inversion_nu_m_fixed`, we can invert for the magnetic field strength. Using :eq:`inversion_R_fixed_thin`, we can then
        invert for the radius of the emitting region. The remaining parameters are either fixed by assumptions (e.g., :math:`p`)
        or are degenerate with the normalization (e.g., :math:`\epsilon_e` and :math:`\epsilon_B`).

        .. tab-set::

            .. tab-item:: Isotropic Pitch Angle

                The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1,\rm ISO} \gamma_{\min}^2},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{m,\rm ISO} B^3 \tilde{N_0}}\right)^{1/3}.
                    }

            .. tab-item:: Fixed Pitch Angle

                The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1} \gamma_{\min}^2 \sin\alpha},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{m,0} B^3 \tilde{N_0} \sin \alpha}\right)^{1/3}.
                    }

        .. note::

            In the slow-cooling regime, there is still dependence on :math:`\gamma_c` in the inversion because the
            moment of the electron distribution appearing in :math:`\tilde{N_0}` depends on :math:`\gamma_c` through
            the shape of the electron distribution.

    .. tab-item:: No Cooling (:math:`\nu_c \to \infty`)

        In this spectrum, there are 3 SPL segments connected by 2 breaks:

        .. dropdown:: Spectral Segments

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
                  - :math:`\nu \geq \nu_{\max}`
                  - SPL I
                  - N/A

        In this case, the smoothed SED may be constructed as:

        .. math::

            F_\nu = F^{(D,G)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we have selected to normalize at the (D,G) break at :math:`\nu_m`.

        .. rubric:: Normalization

        In this SED the only spectral breaks present occur at the characteristic synchrotron
        frequencies associated with the minimum (injection) frequency, the cooling frequency, and the maximum electron frequency,
        :math:`\nu_m`, :math:`\nu_c`, and :math:`\nu_{\max}`, respectively. The peak of the broadband SED occurs at the
        **injection frequency** :math:`\nu_c`, corresponding to synchrotron emission from electrons with
        Lorentz factor :math:`\gamma_{\rm m}`.

        The normalization of the canonical SED is fixed by the following physical parameters:

        .. dropdown:: Normalization Parameters

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
                  - Lorentz factor above which electrons cool efficiently. In the no-cooling limit, :math:`\gamma_c \to \infty`.
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

        The normalization of the power-law SED is most naturally anchored at the injection frequency
        :math:`\nu_m`. For a given choice of pitch-angle treatment, the magnetic field strength :math:`B` therefore
        uniquely determines the location of the spectral peak.

        The flux density at this frequency sets the overall normalization of the SED.

        .. tab-set::

            .. tab-item:: Fixed Pitch Angle

                For the case of a
                fixed pitch angle, the flux normalization at :math:`\nu_c` is given by
                :eq:`slow_cooling_norm` as

                .. math::

                    F_{m,0} \approx
                    \chi \, (B \sin\alpha)\,
                    N_0 \,
                    \gamma_{\rm min}^{\,1-p}
                    \frac{V_{\rm eff}}{D_L^2},

            .. tab-item:: Isotropic Pitch Angle

                For the case of isotropic pitch-angle averaging, the flux normalization at :math:`\nu_m` is given by
                :eq:`slow_cooling_norm_iso` as

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

        .. rubric:: Inversion

        The inversion is the simple case of an optically thin peak at :math:`\nu_m`.

        .. dropdown:: Inversion Parameters

            .. list-table::
                :widths: 10 35 85
                :header-rows: 1

                * - Parameter
                  - Name
                  - Notes
                * - :math:`F_{\rm brk}`
                  - Break Flux
                  - Flux density at the break frequency used for inversion.
                * - :math:`\nu_{\rm brk}`
                  - Break Frequency
                  - Frequency of the break used for inversion.
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
                  - Lorentz factor above which electrons cool efficiently. In the no-cooling limit, :math:`\gamma_c \to \infty`.
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

        Using :eq:`inversion_nu_m_fixed`, we can invert for the magnetic field strength. Using :eq:`inversion_R_fixed_thin`, we can then
        invert for the radius of the emitting region. The remaining parameters are either fixed by assumptions (e.g., :math:`p`)
        or are degenerate with the normalization (e.g., :math:`\epsilon_e` and :math:`\epsilon_B`).

        .. tab-set::

            .. tab-item:: Isotropic Pitch Angle

                The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1,\rm ISO} \gamma_{\min}^2},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{m,\rm ISO} B^3 \tilde{N_0}}\right)^{1/3}.
                    }

            .. tab-item:: Fixed Pitch Angle

                The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1} \gamma_{\min}^2 \sin\alpha},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{m,0} B^3 \tilde{N_0} \sin \alpha}\right)^{1/3}.
                    }

Absorption SEDs
----------------

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

    .. tab-item:: Optically Thin Peak (:math:`\nu_a < \nu_{m}`)

        .. rubric:: SED Shape

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. dropdown:: Spectral Segments

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

        where we have selected to normalize at the (D,G) break at :math:`\nu_m`.


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


        .. rubric:: Normalization

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


        .. rubric:: Inversion

        The inversion is the simple case of an optically thin peak at :math:`\nu_m`.

        .. dropdown:: Inversion Parameters

            .. list-table::
                :widths: 10 35 85
                :header-rows: 1

                * - Parameter
                  - Name
                  - Notes
                * - :math:`F_{\rm brk}`
                  - Break Flux
                  - Flux density at the break frequency used for inversion.
                * - :math:`\nu_{\rm brk}`
                  - Break Frequency
                  - Frequency of the break used for inversion.
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

        Using :eq:`inversion_nu_m_fixed`, we can invert for the magnetic field strength. Using :eq:`inversion_R_fixed_thin`, we can then
        invert for the radius of the emitting region. The remaining parameters are either fixed by assumptions (e.g., :math:`p`)
        or are degenerate with the normalization (e.g., :math:`\epsilon_e` and :math:`\epsilon_B`).

        .. tab-set::

            .. tab-item:: Isotropic Pitch Angle

                The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1,\rm ISO} \gamma_{\min}^2},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{m,\rm ISO} B^3 \tilde{N_0}}\right)^{1/3}.
                    }

            .. tab-item:: Fixed Pitch Angle

                The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1} \gamma_{\min}^2 \sin\alpha},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{m,0} B^3 \tilde{N_0} \sin \alpha}\right)^{1/3}.
                    }


    .. tab-item:: Optically Thick Peak (:math:`\nu_m < \nu_{a}`)

        .. rubric:: SED Shape

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. dropdown:: Spectral Segments

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

        .. todo:: This could be clarified.

        .. math::

            F_\nu = F^{(B,A)}_\nu \tilde{F}^{(A,G)}_\nu
                    \tilde{\Phi}(\nu,\nu_{\max}),

        where we have selected to normalize at the (B,A) break at :math:`\nu_m`.

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

        .. rubric:: Normalization

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


        .. rubric:: Inversion

        In this case, we have a more complicated instance of the optically thick peak inversion.

        .. dropdown:: Inversion Parameters

            .. list-table::
                :widths: 10 35 85
                :header-rows: 1

                * - Parameter
                  - Name
                  - Notes
                * - :math:`F_{\rm brk}`
                  - Break Flux
                  - Flux density at the break frequency used for inversion.
                * - :math:`\nu_{\rm brk}`
                  - Break Frequency
                  - Frequency of the break used for inversion.
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

        Because the SSA frequency corresponds to the peak, we have the standard equation

        .. tab-set::

            .. tab-item:: Isotropic Pitch Angle

                The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

                .. math::

                    F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0,\rm ISO} R^2 B^{-1/2}

            .. tab-item:: Fixed Pitch Angle

                The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

                .. math::

                    F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0} R^2 B^{-1/2},

                where :math:`P_0` absorbs the pitch-angle dependence.

        The second closure in this case appears because the peak frequency determined from the normalization must
        match that observed in the data. Thus,

        .. math::

           \nu_{\rm brk}^{(p-1)/2} F_{\rm pk} =\left[c_1 \gamma_m^2 \sin \alpha\right]^{(p-1)/2} Q_{m,0} R^3 B^{(p+5)/2} \tilde{N_0}

        Letting

        .. math::

            A = \left[c_1 \gamma_m^2 \sin \alpha\right]^{(p-1)/2} Q_{m,0} \tilde{N_0},

        we have the coupled equations

        .. math::

            \boxed{
            F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0} R^2 B^{-1/2},
            \qquad
            \nu_{\rm brk}^{(p-1)/2} F_{\rm pk} = A R^3 B^{(p+5)/2}.
            }

        The solution to which is

        .. tab-set::

            .. tab-item:: Fixed Pitch Angle

                .. math::

                    \boxed{
                    \begin{aligned}
                    R &= A^{-1/(2p+13)} P_0^{-(p+5)/(2p+13)} F_{\rm brk}^{(p+6)/(2p+13)} \nu_{\rm brk}^{-1}\\
                    B &= A^{-4/(2p+13)} P_0^{6/(2p+13)} F_{\rm brk}^{-2/(2p+13)} \nu_{\rm brk},
                    \end{aligned}
                    }

                where

                .. math::

                    A = \left[c_1 \gamma_m^2\right]^{(p-1)/2} \left(\sin \alpha\right)^{(p+1)/2} Q_{m,0} \tilde{N_0},

            .. tab-item:: Isotropic Pitch Angle

                .. math::

                    \boxed{
                    \begin{aligned}
                    R &= A^{-1/(2p+13)} P_{0,\rm ISO}^{-(p+5)/(2p+13)} F_{\rm brk}^{(p+6)/(2p+13)} \nu_{\rm brk}^{-1}\\
                    B &= A^{-4/(2p+13)} P_{0,\rm ISO}^{6/(2p+13)} F_{\rm brk}^{-2/(2p+13)} \nu_{\rm brk},
                    \end{aligned}
                    }

                where

                .. math::

                    A = \left[c_{1,\rm ISO} \gamma_m^2 \right]^{(p-1)/2} Q_{m,\rm ISO} \tilde{N_0}.

Cooling + Absorption SEDs
-------------------------

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

        .. rubric:: SED Shape

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. dropdown:: Spectral Segments

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

        where we have selected to normalize at the (D,G) break at :math:`\nu_m`.

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

        .. rubric:: Normalization

        .. dropdown:: Normalization Parameters

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

        .. rubric:: Inversion

        The inversion is the simple case of an optically thin peak at :math:`\nu_m`.

        .. dropdown:: Inversion Parameters

            .. list-table::
                :widths: 10 35 85
                :header-rows: 1

                * - Parameter
                  - Name
                  - Notes
                * - :math:`F_{\rm brk}`
                  - Break Flux
                  - Flux density at the break frequency used for inversion.
                * - :math:`\nu_{\rm brk}`
                  - Break Frequency
                  - Frequency of the break used for inversion.
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

        Using :eq:`inversion_nu_m_fixed`, we can invert for the magnetic field strength. Using :eq:`inversion_R_fixed_thin`, we can then
        invert for the radius of the emitting region. The remaining parameters are either fixed by assumptions (e.g., :math:`p`)
        or are degenerate with the normalization (e.g., :math:`\epsilon_e` and :math:`\epsilon_B`).

        .. tab-set::

            .. tab-item:: Isotropic Pitch Angle

                The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1,\rm ISO} \gamma_{\min}^2},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{m,\rm ISO} B^3 \tilde{N_0}}\right)^{1/3}.
                    }

            .. tab-item:: Fixed Pitch Angle

                The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1} \gamma_{\min}^2 \sin\alpha},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{m,0} B^3 \tilde{N_0} \sin \alpha}\right)^{1/3}.
                    }


    .. tab-item:: Spectrum 2 :math:`(\nu_m < \nu_a < \nu_{\rm max} < \nu_c)`

        This is the **SSA-only** spectrum in which cooling is irrelevant over the
        emitting band because :math:`\nu_c` lies above the high-energy cutoff
        :math:`\nu_{\max}`. It is therefore equivalent to spectrum 2 from our discussion above
        regarding non-cooling synchrotron SEDs.

        .. rubric:: SED Shape

        In this spectrum, there are 4 SPL segments connected by 3 breaks:

        .. dropdown:: Spectral Segments

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

        where we have selected to normalize at the (B,A) break at :math:`\nu_m`.

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
        .. rubric:: Normalization

        .. dropdown:: Normalization Parameters

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

        .. rubric:: Inversion

        In this case, we have a more complicated instance of the optically thick peak inversion.

        .. dropdown:: Inversion Parameters

            .. list-table::
                :widths: 10 35 85
                :header-rows: 1

                * - Parameter
                  - Name
                  - Notes
                * - :math:`F_{\rm brk}`
                  - Break Flux
                  - Flux density at the break frequency used for inversion.
                * - :math:`\nu_{\rm brk}`
                  - Break Frequency
                  - Frequency of the break used for inversion.
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

        Because the SSA frequency corresponds to the peak, we have the standard equation

        .. tab-set::

            .. tab-item:: Isotropic Pitch Angle

                The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

                .. math::

                    F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0,\rm ISO} R^2 B^{-1/2}

            .. tab-item:: Fixed Pitch Angle

                The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

                .. math::

                    F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0} R^2 B^{-1/2},

                where :math:`P_0` absorbs the pitch-angle dependence.

        The second closure in this case appears because the peak frequency determined from the normalization must
        match that observed in the data. Thus,

        .. math::

           \nu_{\rm brk}^{(p-1)/2} F_{\rm pk} =\left[c_1 \gamma_m^2 \sin \alpha\right]^{(p-1)/2} Q_{m,0} R^3 B^{(p+5)/2} \tilde{N_0}

        Letting

        .. math::

            A = \left[c_1 \gamma_m^2 \sin \alpha\right]^{(p-1)/2} Q_{m,0} \tilde{N_0},

        we have the coupled equations

        .. math::

            \boxed{
            F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0} R^2 B^{-1/2},
            \qquad
            \nu_{\rm brk}^{(p-1)/2} F_{\rm pk} = A R^3 B^{(p+5)/2}.
            }

        The solution to which is

        .. tab-set::

            .. tab-item:: Fixed Pitch Angle

                .. math::

                    \boxed{
                    \begin{aligned}
                    R &= A^{-1/(2p+13)} P_0^{-(p+5)/(2p+13)} F_{\rm brk}^{(p+6)/(2p+13)} \nu_{\rm brk}^{-1}\\
                    B &= A^{-4/(2p+13)} P_0^{6/(2p+13)} F_{\rm brk}^{-2/(2p+13)} \nu_{\rm brk},
                    \end{aligned}
                    }

                where

                .. math::

                    A = \left[c_1 \gamma_m^2\right]^{(p-1)/2} \left(\sin \alpha\right)^{(p+1)/2} Q_{m,0} \tilde{N_0},

            .. tab-item:: Isotropic Pitch Angle

                .. math::

                    \boxed{
                    \begin{aligned}
                    R &= A^{-1/(2p+13)} P_{0,\rm ISO}^{-(p+5)/(2p+13)} F_{\rm brk}^{(p+6)/(2p+13)} \nu_{\rm brk}^{-1}\\
                    B &= A^{-4/(2p+13)} P_{0,\rm ISO}^{6/(2p+13)} F_{\rm brk}^{-2/(2p+13)} \nu_{\rm brk},
                    \end{aligned}
                    }

                where

                .. math::

                    A = \left[c_{1,\rm ISO} \gamma_m^2 \right]^{(p-1)/2} Q_{m,\rm ISO} \tilde{N_0}.

    .. tab-item:: Spectrum 3 :math:`(\nu_a < \nu_m < \nu_c < \nu_{\rm max})`

        This is the standard **slow-cooling + SSA** spectrum with all three breaks
        present in-band.

        .. rubric:: SED Shape

        .. dropdown:: Spectral Segments

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

        .. rubric:: Normalization

        .. dropdown:: Normalization Parameters

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

        .. rubric:: Inversion

        The inversion is the simple case of an optically thin peak at :math:`\nu_m`.

        .. dropdown:: Inversion Parameters

            .. list-table::
                :widths: 10 35 85
                :header-rows: 1

                * - Parameter
                  - Name
                  - Notes
                * - :math:`F_{\rm brk}`
                  - Break Flux
                  - Flux density at the break frequency used for inversion.
                * - :math:`\nu_{\rm brk}`
                  - Break Frequency
                  - Frequency of the break used for inversion.
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

        Using :eq:`inversion_nu_m_fixed`, we can invert for the magnetic field strength. Using :eq:`inversion_R_fixed_thin`, we can then
        invert for the radius of the emitting region. The remaining parameters are either fixed by assumptions (e.g., :math:`p`)
        or are degenerate with the normalization (e.g., :math:`\epsilon_e` and :math:`\epsilon_B`).

        .. tab-set::

            .. tab-item:: Isotropic Pitch Angle

                The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1,\rm ISO} \gamma_{\min}^2},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{m,\rm ISO} B^3 \tilde{N_0}}\right)^{1/3}.
                    }

            .. tab-item:: Fixed Pitch Angle

                The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1} \gamma_{\min}^2 \sin\alpha},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{m,0} B^3 \tilde{N_0} \sin \alpha}\right)^{1/3}.
                    }

    .. tab-item:: Spectrum 4 :math:`(\nu_m < \nu_a < \nu_c < \nu_{\rm max})`

        This is the **slow-cooling + SSA** spectrum with :math:`\nu_a` above
        :math:`\nu_m`, producing an optically thick :math:`\nu^{5/2}` segment
        between :math:`\nu_m` and :math:`\nu_a`.

        .. rubric:: SED Shape

        .. dropdown:: Spectral Segments

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

        .. rubric:: Normalization

        .. dropdown:: Normalization Parameters

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

        .. rubric:: Inversion

        In this case, we have a more complicated instance of the optically thick peak inversion.

        .. dropdown:: Inversion Parameters

            .. list-table::
                :widths: 10 35 85
                :header-rows: 1

                * - Parameter
                  - Name
                  - Notes
                * - :math:`F_{\rm brk}`
                  - Break Flux
                  - Flux density at the break frequency used for inversion.
                * - :math:`\nu_{\rm brk}`
                  - Break Frequency
                  - Frequency of the break used for inversion.
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

        Because the SSA frequency corresponds to the peak, we have the standard equation

        .. tab-set::

            .. tab-item:: Isotropic Pitch Angle

                The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

                .. math::

                    F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0,\rm ISO} R^2 B^{-1/2}

            .. tab-item:: Fixed Pitch Angle

                The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

                .. math::

                    F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0} R^2 B^{-1/2},

                where :math:`P_0` absorbs the pitch-angle dependence.

        The second closure in this case appears because the peak frequency determined from the normalization must
        match that observed in the data. Thus,

        .. math::

           \nu_{\rm brk}^{(p-1)/2} F_{\rm pk} =\left[c_1 \gamma_m^2 \sin \alpha\right]^{(p-1)/2} Q_{m,0} R^3 B^{(p+5)/2} \tilde{N_0}

        Letting

        .. math::

            A = \left[c_1 \gamma_m^2 \sin \alpha\right]^{(p-1)/2} Q_{m,0} \tilde{N_0},

        we have the coupled equations

        .. math::

            \boxed{
            F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0} R^2 B^{-1/2},
            \qquad
            \nu_{\rm brk}^{(p-1)/2} F_{\rm pk} = A R^3 B^{(p+5)/2}.
            }

        The solution to which is

        .. tab-set::

            .. tab-item:: Fixed Pitch Angle

                .. math::

                    \boxed{
                    \begin{aligned}
                    R &= A^{-1/(2p+13)} P_0^{-(p+5)/(2p+13)} F_{\rm brk}^{(p+6)/(2p+13)} \nu_{\rm brk}^{-1}\\
                    B &= A^{-4/(2p+13)} P_0^{6/(2p+13)} F_{\rm brk}^{-2/(2p+13)} \nu_{\rm brk},
                    \end{aligned}
                    }

                where

                .. math::

                    A = \left[c_1 \gamma_m^2 \sin \alpha\right]^{(p-1)/2} \left(\sin^{(p+1)/2} \alpha\right) Q_{m,0} \tilde{N_0},

            .. tab-item:: Isotropic Pitch Angle

                .. math::

                    \boxed{
                    \begin{aligned}
                    R &= A^{-1/(2p+13)} P_{0,\rm ISO}^{-(p+5)/(2p+13)} F_{\rm brk}^{(p+6)/(2p+13)} \nu_{\rm brk}^{-1}\\
                    B &= A^{-4/(2p+13)} P_{0,\rm ISO}^{6/(2p+13)} F_{\rm brk}^{-2/(2p+13)} \nu_{\rm brk},
                    \end{aligned}
                    }

                where

                .. math::

                    A = \left[c_{1,\rm ISO} \gamma_m^2 \right]^{(p-1)/2} Q_{m,\rm ISO} \tilde{N_0}.

    .. tab-item:: Spectrum 5 :math:`(\nu_a < \nu_c < \nu_m < \nu_{\rm max})`

        Spectrum 5 is the first of the two spectra in this formalism which is subject to the
        effects of **stratified SSA**, which introduces an additional SSA break at a frequency
        :math:`\nu_{\rm ac}`. The segments of the spectrum are

        .. rubric:: SED Shape

        .. dropdown:: Spectral Segments

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


        .. rubric:: Normalization

        .. dropdown:: Normalization Parameters

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


        .. rubric:: Inversion

        The inversion is the simple case of an optically thin peak at :math:`\nu_c`.

        .. dropdown:: Inversion Parameters

            .. list-table::
                :widths: 10 35 85
                :header-rows: 1

                * - Parameter
                  - Name
                  - Notes
                * - :math:`F_{\rm brk}`
                  - Break Flux
                  - Flux density at the break frequency used for inversion.
                * - :math:`\nu_{\rm brk}`
                  - Break Frequency
                  - Frequency of the break used for inversion.
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
                  - Lorentz factor above which electrons cool efficiently. In the no-cooling limit, :math:`\gamma_c \to \infty`.
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

        Using :eq:`inversion_nu_m_fixed` but instead using :math:`\nu_c`, we can invert for the magnetic field strength.
        Using :eq:`inversion_R_fixed_thin`, we can then
        invert for the radius of the emitting region. The remaining parameters are either fixed by assumptions (e.g., :math:`p`)
        or are degenerate with the normalization (e.g., :math:`\epsilon_e` and :math:`\epsilon_B`).

        .. tab-set::

            .. tab-item:: Isotropic Pitch Angle

                The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1,\rm ISO} \gamma_{\min}^2},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{c,\rm ISO} B^3 \tilde{N_0}}\right)^{1/3}.
                    }

            .. tab-item:: Fixed Pitch Angle

                The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

                .. math::

                    \boxed{
                    B = \frac{\nu_{\rm brk}}{c_{1} \gamma_{\min}^2 \sin\alpha},
                    \qquad
                    R = \left(\frac{F_{\rm brk}}{Q_{c,0} B^3 \tilde{N_0} \sin \alpha}\right)^{1/3}.
                    }

    .. tab-item:: Spectrum 6 :math:`(\nu_c < \nu_a < \nu_m < \nu_{\rm max})`

        Spectrum 6 is the second case in which the SSA break due to stratified SSA appears at
        :math:`\nu_{\rm ac}`. Additionally, because :math:`\nu_c` is obscured by SSA, we also have to
        perform power-law propagation to correct the normalization, making this one of the trickier of the
        SED cases.

        .. rubric:: SED Shape

        .. dropdown:: Spectral Segments

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

        .. rubric:: Normalization

        .. dropdown:: Normalization Parameters

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

        .. rubric:: Inversion

        In this case, we have a more complicated instance of the optically thick peak inversion.

        .. dropdown:: Inversion Parameters

            .. list-table::
                :widths: 10 35 85
                :header-rows: 1

                * - Parameter
                  - Name
                  - Notes
                * - :math:`F_{\rm brk}`
                  - Break Flux
                  - Flux density at the break frequency used for inversion.
                * - :math:`\nu_{\rm brk}`
                  - Break Frequency
                  - Frequency of the break used for inversion.
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

        Because the SSA frequency corresponds to the peak, we have the standard equation

        .. tab-set::

            .. tab-item:: Isotropic Pitch Angle

                The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

                .. math::

                    F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0,\rm ISO} R^2 B^{-1/2}

            .. tab-item:: Fixed Pitch Angle

                The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

                .. math::

                    F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0} R^2 B^{-1/2},

                where :math:`P_0` absorbs the pitch-angle dependence.

        The second closure in this case appears because the peak frequency determined from the normalization must
        match that observed in the data. Thus,

        .. math::

           \nu_{\rm brk}^{1/2} F_{\rm pk} =\left[c_1 \gamma_c^2 \sin \alpha\right]^{1/2} Q_{c,0} R^3 B^{7/2} \tilde{N_0}

        Letting

        .. math::

            A = \left[c_1 \gamma_c^2 \sin \alpha\right]^{1/2} Q_{c,0} \tilde{N_0},

        we have the coupled equations

        .. math::

            \boxed{
            F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0} R^2 B^{-1/2},
            \qquad
            \nu_{\rm brk}^{1/2} F_{\rm pk} = A R^3 B^{7/2}.
            }

        The solution to which is

        .. tab-set::

            .. tab-item:: Fixed Pitch Angle

                .. math::

                    \boxed{
                    \begin{aligned}
                    R &= A^{-1/17} P_0^{-7/17} F_{\rm brk}^{8/17} \nu_{\rm brk}^{-1}\\
                    B &= A^{-4/17} P_0^{6/17} F_{\rm brk}^{-2/17} \nu_{\rm brk},
                    \end{aligned}
                    }

                where

                .. math::

                    A = \left[c_1 \gamma_c^2 \right]^{1/2} \sin^{3/2}\alpha Q_{c,0} \tilde{N_0},

            .. tab-item:: Isotropic Pitch Angle

                .. math::

                    \boxed{
                    \begin{aligned}
                    R &= A^{-1/17} P_{0,\rm ISO}^{-7/17} F_{\rm brk}^{8/17} \nu_{\rm brk}^{-1}\\
                    B &= A^{-4/17} P_{0,\rm ISO}^{6/17} F_{\rm brk}^{-2/17} \nu_{\rm brk},
                    \end{aligned}
                    }

                where

                .. math::

                    A = \left[c_{1,\rm ISO} \gamma_c^2 \right]^{1/2} Q_{c,\rm ISO} \tilde{N_0}.

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

        .. rubric:: SED Shape

        In this spectrum, the regimes are as follows

        .. dropdown:: Spectral Segments

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

        .. rubric:: Normalization

        .. dropdown:: Normalization Parameters

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

        .. rubric:: Inversion

        In this case, we have a more complicated instance of the optically thick peak inversion.

        .. dropdown:: Inversion Parameters

            .. list-table::
                :widths: 10 35 85
                :header-rows: 1

                * - Parameter
                  - Name
                  - Notes
                * - :math:`F_{\rm brk}`
                  - Break Flux
                  - Flux density at the break frequency used for inversion.
                * - :math:`\nu_{\rm brk}`
                  - Break Frequency
                  - Frequency of the break used for inversion.
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

        Because the SSA frequency corresponds to the peak, we have the standard equation

        .. tab-set::

            .. tab-item:: Isotropic Pitch Angle

                The inversion for the magnetic field strength and radius in the isotropic pitch-angle case is given by

                .. math::

                    F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0,\rm ISO} R^2 B^{-1/2}

            .. tab-item:: Fixed Pitch Angle

                The inversion for the magnetic field strength and radius in the fixed pitch-angle case is given by

                .. math::

                    F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0} R^2 B^{-1/2},

                where :math:`P_0` absorbs the pitch-angle dependence.

        The second closure in this case appears because the peak frequency determined from the normalization must
        match that observed in the data. From above, we have the relationship that

        .. math::

            F_{\rm pk} = F_{0} \nu_m^{(p-1)/2} \nu_c^{1/2} \nu_a^{-p/2},

        where :math:`F_0` is the appropriate normalization at either :math:`\nu_c` (fast cooling)
        or :math:`\nu_m` (slow cooling). As such, we have the condition that

        .. math::

           \nu_{\rm brk}^{p/2} F_{\rm pk} = c_1^{p/2} \gamma_m^{p-1} \gamma_c \sin^{p+2/2} \alpha
           Q_0 \tilde{N}_0 R^3 B^{(p+6)/2}.

        Letting

        .. math::

            A = c_1^{p/2} \gamma_m^{p-1} \gamma_c \sin^{p+2/2} \alpha Q_0 \tilde{N}_0,

        we have the coupled equations

        .. math::

            \boxed{
            F_{\rm brk} \nu_{\rm brk}^{-5/2} = P_{0} R^2 B^{-1/2},
            \qquad
            \nu_{\rm brk}^{p/2} F_{\rm pk} = A R^3 B^{(p+6)/2}
            }

        The solution to which is

        .. tab-set::

            .. tab-item:: Fixed Pitch Angle

                .. math::

                    \boxed{
                    \begin{aligned}
                    R &= A^{-1/(2p+15)} P_0^{-(p+6)/(2p+15)} F_{\rm brk}^{(p+7)/(2p+15)} \nu_{\rm brk}^{-1}\\
                    B &= A^{-4/(2p+15)} P_0^{6/(2p+15)} F_{\rm brk}^{-2/(2p+15)} \nu_{\rm brk},
                    \end{aligned}
                    }

                where

                .. math::

                    A = c_1^{p/2} \gamma_m^{p-1} \gamma_c \sin^{p+2/2} \alpha Q_0 \tilde{N}_0,

                and :math:`Q_0` is either :math:`Q_{c,0}` (fast cooling) or :math:`Q_{m,0}` (slow cooling).

            .. tab-item:: Isotropic Pitch Angle

                .. math::

                    \boxed{
                    \begin{aligned}
                    R &= A^{-1/(2p+15)} P_{0,\rm ISO}^{-(p+6)/(2p+15)} F_{\rm brk}^{(p+7)/(2p+15)} \nu_{\rm brk}^{-1}\\
                    B &= A^{-4/(2p+15)} P_{0,\rm ISO}^{6/(2p+15)} F_{\rm brk}^{-2/(2p+15)} \nu_{\rm brk},
                    \end{aligned}
                    }

                where

                .. math::

                    A = c_{1,\rm ISO}^{p/2} \gamma_m^{p-1} \gamma_c \alpha Q_{0,\rm ISO} \tilde{N}_0,

                and :math:`Q_{0,\rm ISO}` is either :math:`Q_{c,\rm ISO}`
               (fast cooling) or :math:`Q_{m,\rm ISO}` (slow cooling).

References
----------
.. footbibliography::
