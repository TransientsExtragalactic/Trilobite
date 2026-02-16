.. _stratified_absorption:
==============================================
Theory Note: Stratified Synchrotron Absorption
==============================================

In certain scenarios, extreme cooling of the post-shock material results in a modification to the
canonical synchrotron self-absorption (SSA) spectrum. This note outlines the theoretical basis for
this effect, referred to as **stratified synchrotron self-absorption**, and derives the modified spectral
scaling that arises in this regime. The classical paper on this effect is :footcite:t:`2000ApJ...534L.163G`, which
first introduced the concept in the context of gamma-ray burst afterglows. This effect is, generally, only relevant
in relativistic shocks where post-shock fluid velocity allows for large spatial stratification of electron energies.

.. contents::

Standard Synchrotron Self-Absorption
------------------------------------

To introduce the notion of **stratified synchrotron self-absorption (SSA)**, it is useful to first
recount the standard argument for SSA in a *homogeneous* synchrotron-emitting plasma and to make
explicit the assumptions underlying the familiar low-frequency behavior of the spectrum.

For an **isotropic distribution** of relativistic electrons with differential number density
:math:`N(\gamma) \equiv d N / d\gamma`, the synchrotron self-absorption coefficient may be written
(see, e.g., :footcite:t:`RybickiLightman`) as

.. math::

    \alpha_\nu
    \propto
    \frac{1}{\nu^2}
    \int d\gamma\;
    P(\nu,\gamma)\,
    \gamma^2
    \frac{d}{d\gamma}
    \!\left[
        \frac{N(\gamma)}{\gamma^2}
    \right],

where :math:`P(\nu,\gamma)` is the synchrotron power per unit frequency emitted by a single electron
of Lorentz factor :math:`\gamma`.

At a **fixed observing frequency** :math:`\nu`, the structure of the synchrotron kernel implies that
electrons with characteristic synchrotron frequency
:math:`\nu_{\rm synch}(\gamma) \ll \nu` lie on the exponentially suppressed high-frequency tail of
:math:`P(\nu,\gamma)` and therefore **contribute negligibly to the absorption coefficient**. As a result,
the effective support of the integral is restricted to Lorentz factors

.. math::

    \gamma \gtrsim \gamma_{\rm synch}(\nu),
    \qquad
    \nu_{\rm synch}\!\left(\gamma_{\rm synch}(\nu)\right) \sim \nu,

and the absorption coefficient may be written schematically as

.. math::

    \alpha_\nu
    \propto
    \frac{1}{\nu^2}
    \int_{\gamma_{\rm synch}(\nu)}^{\infty}
    d\gamma\;
    P(\nu,\gamma)\,
    \gamma^2
    \frac{d}{d\gamma}
    \!\left[
        \frac{N(\gamma)}{\gamma^2}
    \right].

.. note::

    The big idea here being that *at each observing frequency*, only electrons above a certain
    Lorentz factor contribute to absorption, because lower-energy electrons simply cannot absorb photons
    at that frequency (the synchrotron kernel is exponentially suppressed there). Likewise, it is the lowest-energy
    electrons above the cutoff that dominate absorption at low frequencies.

Now consider the regime in which the characteristic Lorentz factor :math:`\gamma_{\rm synch}(\nu)` lies **below the low-energy cutoff**
:math:`\gamma_m` of the electron distribution, i.e.

.. math::

    \nu \ll \nu_m \equiv \nu_{\rm synch}(\gamma_m).

In this case, the **only contribution** comes from the low-energy tail of the synchrotron kernel, for which

.. math::

    P(\nu,\gamma) \propto \left(\frac{\nu}{\nu_{\rm synch}(\gamma)}\right)^{1/3}
    \propto \nu^{1/3}\,\gamma^{-2/3}.

Substituting this expression into the absorption coefficient yields the scaling

.. math::

    \alpha_\nu
    \propto
    \nu^{-5/3}

where the remaining integral is independent of :math:`\nu`. Thus, in a homogeneous source at
frequencies well below :math:`\nu_m`, the absorption coefficient scales as
:math:`\alpha_\nu \propto \nu^{-5/3}`.

In the same frequency regime, the synchrotron emissivity scales as
:math:`j_\nu \propto \nu^{1/3}`, since both emission and absorption are controlled by electrons at
the fixed Lorentz factor :math:`\gamma \simeq \gamma_m`. The resulting SSA source function,

.. math::

    S_\nu \equiv \frac{j_\nu}{\alpha_\nu},

therefore scales as

.. math::

    S_\nu \propto \nu^{2}.

In the optically thick limit (:math:`\alpha_\nu L \gg 1`, where :math:`L` is the characteristic size
of the emitting region), radiative transfer implies that the **emergent specific intensity approaches
the source function**. For a source of fixed angular size, this yields the familiar low-frequency
synchrotron self-absorbed spectrum,

.. math::

    F_\nu \propto \nu^{2},
    \qquad
    \nu \ll \nu_m.

This conclusion relies critically on the assumption that the identity of the electrons controlling
both emission and absorption is *independent of frequency* at sufficiently low :math:`\nu`.

In realistic astrophysical shocks, however—particularly in the **fast-cooling regime**—the
post-shock plasma is not homogeneous. Electrons are continuously accelerated at the shock front and
cool as they are advected downstream, producing a strong spatial stratification in the characteristic
electron energy. As a result, the lowest-energy electrons capable of absorbing a given frequency
depend on position within the flow. This stratification qualitatively modifies the optically thick
synchrotron spectrum and leads to deviations from the standard :math:`F_\nu \propto \nu^2` scaling,
as discussed below.

.. figure:: ../../../../images/theory/stratified_ssa.png
    :width: 100%
    :align: center
    :alt: Schematic illustration of stratified synchrotron self-absorption.

    **Schematic illustration of stratified synchrotron self-absorption in a fast-cooling shock.**
    Electrons are injected at the shock front (bottom) with minimum Lorentz factor
    :math:`\gamma_m` and subsequently cool as they advect downstream, producing a thin
    **uncooled layer** (:math:`\gamma \simeq \gamma_m`) followed by a much thicker **cooled layer**,
    in which the characteristic electron Lorentz factor decreases with distance behind the shock,
    :math:`\gamma_c(x) \propto x^{-1}`. For a given observing frequency :math:`\nu`, synchrotron
    self-absorption becomes effective only once the radiation reaches layers where
    :math:`\nu \lesssim \nu_{\rm sync}(\gamma_c(x))`; deeper layers then dominate the optical-depth
    accumulation. The shaded regions indicate the locations of the frequency-dependent SSA
    photospheres (:math:`\tau_\nu \simeq 1`) for two representative frequencies. At low frequencies
    (:math:`\nu < \nu_{\rm ac}`), the photosphere lies within the uncooled layer, yielding the standard
    optically thick scaling :math:`S_\nu \propto \nu^2`. At higher frequencies
    (:math:`\nu_{\rm ac} < \nu < \nu_a`), the photosphere recedes into the cooled layer, where the
    decrease of the effective electron temperature with depth modifies the emergent spectrum,
    producing the stratified self-absorption slope :math:`S_\nu \propto \nu^{11/8}`. The transition
    frequency :math:`\nu_{\rm ac}` marks the frequency at which the SSA photosphere coincides with
    the boundary between the uncooled and cooled layers.


Stratified SSA
---------------

The primary difference between standard SSA and stratified SSA lies in the spatial variation of
the electron energy distribution behind the shock front. In the stratified case, the electrons
responsible for absorption at a given frequency vary with depth into the post-shock region,
leading to a frequency-dependent SSA photosphere and a modified emergent spectrum. In this section, we'll
derive the relevant results and implications for the spectrum.

Post-Shock Stratification
^^^^^^^^^^^^^^^^^^^^^^^^^^

We begin by recognizing that, if cooling is sufficiently fast, then a significant degree of stratification
develops immediately behind the shock front due to radiative losses. In each subsequent layer downstream,
the characteristic electron energy decreases as electrons cool. This stratification affects both the emissivity
and absorption coefficient as a function of depth behind the shock and significantly complicates the resulting emission.

Given a cooling process (e.g., synchrotron and/or inverse Compton) with rate function :math:`\Lambda`, the
cooling timescale for an electron of Lorentz factor :math:`\gamma` is

.. math::

    t_{\rm cool}(\gamma) = \frac{\gamma m_e c^2}{\Lambda(\gamma)}.

For electrons injected at the shock with a minimum Lorentz factor :math:`\gamma_m`, the cooling timescale
:math:`t_{\rm cool}(\gamma_m)` sets the thickness of the uncooled layer behind the shock:

.. math::

    \ell_{\rm uncooled} \sim v_{\rm ps}\,t_{\rm cool}(\gamma_m).

Likewise, for a given dynamical timescale :math:`t_{\rm dyn}`, one may define a cooling Lorentz factor

.. math::

    \gamma_c : t_{\rm cool}(\gamma_c) = t_{\rm dyn}.

and a corresponding cooling layer thickness

.. math::

    \ell_{\rm cool} \sim v_{\rm ps}\,t_{\rm dyn}.

The relative scales of these two lengths determine the degree of stratification in the post-shock region:

.. math::

    \boxed{
    \frac{\ell_{\rm uncooled}}{\ell_{\rm cool}} = \frac{t_{\rm cool}(\gamma_m)}{t_{\rm dyn}} = \frac{\gamma_c}{\gamma_m}.
    }

In the **fast-cooling regime** (:math:`\gamma_c \ll \gamma_m`), the uncooled layer is very thin
(:math:`\ell_{\rm uncooled} \ll \ell_{\rm cool}`), and most of the post-shock region is occupied by cooled electrons
whose characteristic energy decreases with distance behind the shock. This leads to our big idea:

.. admonition:: Big Idea

    In the fast-cooling regime, the coldest electrons are located farthest from the shock front, meaning
    that the surface from which SSA emission escapes **depends on frequency**. For extremely **low** frequencies,
    even the uncooled layer is optically thick, leading to typical behavior, but at higher frequencies, the
    SSA photosphere recedes into progressively colder layers, modifying the emergent spectrum.

Radiative Transfer Through a Stratified Slab
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now consider synchrotron radiation of a fixed observing frequency :math:`\nu` emitted at some
depth behind the shock and propagating outward toward the observer. Because electrons are injected
*only at the shock front* and subsequently cool as they advect downstream, the post-shock region
may be regarded as a stack of layers with monotonically increasing characteristic electron energy
toward the shock.

It is therefore useful to describe the post-shock region in terms of two distinct slabs:

1. an **uncooled slab**, extending from the shock front to depth
   :math:`\ell_{\rm uncooled} \sim v_{\rm ps} t_{\rm cool}(\gamma_m)`, in which electrons retain
   their injection energy :math:`\gamma \simeq \gamma_m`, and

2. a **cooled slab**, extending from :math:`\ell_{\rm uncooled}` to
   :math:`\ell_{\rm cool} \sim v_{\rm ps} t_{\rm dyn}`, in which the characteristic electron
   Lorentz factor decreases with depth due to radiative losses.

We adopt a coordinate :math:`x` measured *from the shock front downstream*, such that
:math:`x = 0` corresponds to freshly shocked material.

Frequency-Dependent Absorption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider radiation of frequency :math:`\nu` propagating outward through successive layers at
increasingly smaller :math:`x`. In a given layer at depth :math:`x`, the maximum electron Lorentz
factor is set by cooling, :math:`\gamma_{\rm cool}(x)`, with an associated synchrotron frequency

.. math::

    \nu_{\rm cool}(x) \equiv \nu_{\rm synch}(\gamma_{\rm cool}(x)).

If :math:`\nu > \nu_{\rm cool}(x)`, then *all* electrons in that layer lie on the exponentially
suppressed high-frequency tail of the synchrotron kernel. In this case, the local absorption
coefficient is negligible:

.. math::

    \alpha_\nu(x) \simeq 0.

Such layers are effectively transparent at frequency :math:`\nu`.

Conversely, once the radiation reaches a layer for which

.. math::

    \nu \lesssim \nu_{\rm cool}(x),

electrons are present that can absorb efficiently at frequency :math:`\nu`. From this point
toward the shock, the SSA optical depth begins to accumulate. We define :math:`x_k(\nu)` as the
*first* layer encountered (moving outward) for which this condition is satisfied.

The SSA photosphere at frequency :math:`\nu` is then defined implicitly by

.. math::

    \tau_\nu(x_{\rm LS,\nu})
    =
    \int_{x_k(\nu)}^{x_{\rm LS,\nu}} \alpha_\nu(x')\,dx'
    \simeq 1,

where :math:`x_{\rm LS,\nu}` denotes the last-scattering surface for photons of frequency
:math:`\nu`.

Absorption Coefficient in the Cooled Slab
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the cooled slab, electrons have cooled down to the local Lorentz factor
:math:`\gamma_{\rm cool}(x)`, and no electrons with :math:`\gamma > \gamma_{\rm cool}(x)` are
present. Absorption is therefore dominated by electrons near this cutoff.

Because absorption occurs only for electrons satisfying
:math:`\nu \ll \nu_{\rm synch}(\gamma)`, the relevant part of the synchrotron kernel is its
low-frequency asymptotic form,

.. math::

    P(\nu,\gamma) \propto \nu^{1/3}\,\gamma^{-2/3}.

Substituting this into the SSA coefficient yields the local scaling

.. math::

    \alpha_\nu(x)
    \propto
    \nu^{-5/3}\,\gamma_{\rm cool}(x)^{-5/3}.

In the fast-cooling regime, radiative losses imply

.. math::

    \gamma_{\rm cool}(x) \propto x^{-1},

so that

.. math::

    \alpha_\nu(x) \propto \nu^{-5/3}\,x^{5/3}.

Source Functions in the Stratified Medium
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A key simplification of the stratified SSA problem is that *any layer which defines the SSA
photosphere must lie on the Rayleigh–Jeans tail of the local synchrotron spectrum*. Otherwise,
the absorption coefficient would be exponentially suppressed and optical depth could not reach
unity.

As a result, the local SSA source function always takes the Rayleigh–Jeans form,

.. math::

    S_\nu(x) = \frac{j_\nu(x)}{\alpha_\nu(x)} \propto \nu^2\,\gamma(x).

The distinction between the two slabs enters only through the depth-dependence of
:math:`\gamma(x)`.

(1) Uncooled Slab
~~~~~~~~~~~~~~~~~

In the uncooled slab (:math:`x \lesssim \ell_{\rm uncooled}`), electrons retain their injection
energy,

.. math::

    \gamma(x) \simeq \gamma_m = \text{constant}.

The local source function is therefore independent of depth:

.. math::

    S_\nu^{\rm (uncooled)} \propto \nu^2\,\gamma_m.

If the SSA photosphere lies within this slab, the emergent spectrum recovers the standard
optically thick synchrotron scaling,

.. math::

    F_\nu \propto \nu^2.

(2) Cooled Slab
~~~~~~~~~~~~~~~

In the cooled slab (:math:`x \gtrsim \ell_{\rm uncooled}`), the characteristic electron energy
decreases with depth,

.. math::

    \gamma(x) \propto x^{-1}.

As a result, the local SSA source function varies with depth as

.. math::

    S_\nu^{\rm (cooled)}(x) \propto \nu^2\,\gamma(x) \propto \nu^2\,x^{-1}.

To determine the emergent spectrum, we must therefore establish how the depth of the SSA
photosphere depends on observing frequency.

As shown above, absorption at frequency :math:`\nu` becomes effective only once the radiation
reaches layers for which :math:`\nu \lesssim \nu_{\rm cool}(x)`. From that point inward toward the
shock, the SSA optical depth accumulates according to

.. math::

    \tau_\nu(x)
    =
    \int_{x_k(\nu)}^{x} \alpha_\nu(x')\,dx',

where :math:`x_k(\nu)` is the first layer capable of absorbing at frequency :math:`\nu`.

In the cooled slab, the local absorption coefficient scales as

.. math::

    \alpha_\nu(x)
    \propto
    \nu^{-5/3}\,\gamma_{\rm cool}(x)^{-5/3}
    \propto
    \nu^{-5/3}\,x^{5/3}.

Integrating this expression yields

.. math::

    \tau_\nu(x)
    \propto
    \nu^{-5/3}
    \int x'^{5/3}\,dx'
    \propto
    \nu^{-5/3}\,x^{8/3}.

The SSA photosphere :math:`x_{\rm LS,\nu}` is defined by the condition
:math:`\tau_\nu(x_{\rm LS,\nu}) \simeq 1`, implying

.. math::

    x_{\rm LS,\nu} \propto \nu^{-5/8}.

The observed intensity at frequency :math:`\nu` is determined by the local source function
evaluated at the SSA photosphere,

.. math::

    I_\nu \simeq S_\nu^{\rm (cooled)}\!\left(x_{\rm LS,\nu}\right).

Substituting the scalings derived above, we find

.. math::

    I_\nu
    \propto
    \nu^2\,x_{\rm LS,\nu}^{-1}
    \propto
    \nu^2\,\nu^{5/8}
    =
    \nu^{11/8}.

For a source of fixed angular size, the flux density therefore scales as

.. math::

    \boxed{
    F_\nu \propto \nu^{11/8},
    \qquad
    \nu_{\rm ac} < \nu < \nu_a.
    }

This modified optically thick slope arises *entirely* from the frequency dependence of the SSA
photosphere depth in the cooling-stratified post-shock region. The local synchrotron source
function remains Rayleigh–Jeans at all relevant depths; the deviation from the standard
:math:`F_\nu \propto \nu^2` behavior is a geometric consequence of stratification rather than a
change in microphysics.

Spectral Breaks and Spectral Implications
-----------------------------------------

As we have shown previously, there are **two distinct spectral regimes** for stratified SSA determined by
the location of the SSA photosphere relative to the uncooled and cooled slabs:

- If the SSA photosphere lies within the **uncooled slab** (:math:`\nu < \nu_{\rm ac}`), the emergent spectrum
  recovers the standard optically thick synchrotron scaling,

  .. math::

      F_\nu \propto \nu^{2}.

- If the SSA photosphere lies within the **cooled slab** (:math:`\nu_{\rm ac} < \nu < \nu_a`), the emergent spectrum
  exhibits the modified scaling derived above,

  .. math::

      F_\nu \propto \nu^{11/8}.

.. important::

    TODO: Complete the discussion of spectral breaks, including expressions for :math:`\nu_{\rm ac}` and :math:`\nu_a`,
    and implications for astrophysical sources.
