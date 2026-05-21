.. _barnoil_duran:
===============================================
Theory Note: The Barniol-Duran Formalism
===============================================

A common motif in Newtonian treatments of synchrotron radiation from a generic emission region is the approach
of **formal equipartition**, in which one computes the *minimum* energy of the system given its emission properties.
More concretely, the synchrotron emission arises from a population of electrons with kinetic energy :math:`E_e` and
from a magnetic field with energy :math:`E_B`. The total energy of the system is then :math:`E = E_e + E_B`, and the
equipartition energy is the minimum of this function subject to the constraint that the system be capable
of producing the observed emission.

The approach of :footcite:t:`duran2013radius` extends this formalism to relativistic outflows and is
thus widely applicable for synchrotron inversion of relativistic sources. Likewise, there exist modifications
which extend the formalism to account for Newtonian correction in cases where the outflow is not highly relativistic.

In this theory note, we describe the basic principles of the Barniol-Duran formalism.

Problem Overview
----------------

In the Barniol-Duran problem, we consider an emitting source with the following properties:

- The source is at **redshift** :math:`z`, and **luminosity distance** :math:`d_L`.
- The source is characterized by a peak flux density :math:`F_{\nu,p}` at a peak frequency :math:`\nu_p`.
- The source has an emitting region of size :math:`R` and a magnetic field strength :math:`B`.
- The source is expanding with a **bulk Lorentz factor** :math:`\Gamma`
- The emission at :math:`\nu_p` is dominated by electrons with Lorentz factor :math:`\gamma_e`

The goal of the formalism is to take our knowledge of :math:`(\nu, F_{\nu})` and infer the physical parameters of
the system, such as :math:`R`, :math:`B`, :math:`\Gamma`, and :math:`\gamma_e`.

Source Geometry
----------------

For a relativistic outflow which is *effectively spherical* (i.e., the outflow is either truly spherical
or is a jet with an opening angle :math:`\theta_j \gtrsim 1/\Gamma`), the observed emission arises from a region of
size

.. math::

    A \approx \frac{\pi R^2}{\Gamma^2},

and volume

.. math::

    V \approx \frac{\pi R^3}{\Gamma^4}.

To allow for deviations from these cases, or for cases where not the entire emitting region is active, we introduce
a **filling factor** :math:`f_A` and :math:`f_V` for the area and volume, respectively. Thus, the effective area
and volume of the emitting region are

.. math::

    A = f_A \frac{\pi R^2}{\Gamma^2}, \qquad V = f_V \frac{\pi R^3}{\Gamma^4}.

Results
-------

As described in :footcite:t:`duran2013radius`, the equipartition radius and energy are given by

.. math::

    \begin{aligned}
    R_{\rm eq} &\approx \left( 1.7 \times 10^{17} \, {\rm cm} \right) \left[ F_{p,{\rm mJy}}^{\frac{8}{17}} d_{L,28}^{
    \frac{16}{17}} \nu_{p,10}^{-\frac{1}{17}} \eta^{\frac{35}{51}} (1+z)^{-\frac{25}{17}} \right]
    \frac{\Gamma^{\frac{10}{17}}}{f_A^{\frac{7}{17}} f_V^{\frac{1}{17}}}, \\
    E_{\rm eq} &\approx \left( 2.5 \times 10^{49} \, {\rm erg} \right) \left[ F_{p,{\rm mJy}}^{\frac{20}{17}}
    d_{L,28}^{\frac{40}{17}} \nu_{p,10}^{-1} \eta^{\frac{15}{17}} (1+z)^{-\frac{37}{17}} \right]
    \frac{f_V^{\frac{6}{17}}}{f_A^{\frac{9}{17}} \Gamma^{\frac{26}{17}}},
    \end{aligned}

where

.. math::

    \eta = \begin{cases}
    \frac{\nu_p}{\nu_a} & \nu_a < \nu_p, \\
    1 & \nu_a > \nu_p.
    \end{cases}

Assuming that the outflow is in its **coasting phase** (i.e., the outflow is not decelerating), then the bulk Lorentz
factor is related to the radius by

.. math::

    t \approx \frac{R(1+z)}{2c\Gamma^2},

and we have instead that

.. math::

    \begin{aligned}
    R_{\rm eq} &\approx \left( 7.5 \times 10^{17} \, {\rm cm} \right) \left[F_{p,{\rm mJy}}^{2/3} d_{L,28}^{4/3}
    \nu_{p,10}^{-17/12} \eta^{35/36} (1+z)^{-5/3} t_d^{-5/12}\right] f_A^{-7/12} f_V^{-1/12} \\
    \Gamma &\approx 12 \left[F_{p,{\rm mJy}}^{1/3} d_{L,28}^{2/3} \nu_{p,10}^{-17/24} \eta^{35/72}
    (1+z)^{-1/3} t_d^{-17/24}\right] f_A^{-7/24} f_V^{-1/24}, \\
    E_{\rm eq} &\approx \left( 5.7\times10^{47} \, {\rm erg} \right) \left[F_{p,{\rm mJy}}^{2/3} d_{L,28}^{4/3}
    \nu_{p,10}^{1/12} \eta^{5/36}
    (1+z)^{-5/3} t_d^{13/12}\right] f_A^{-1/12} f_V^{5/12}.
    \end{aligned}

Furthermore, if equipartition is assumed to be exact,

.. math::

    \begin{aligned}
    \gamma_e &\approx 525 \left[F_{p,{\rm mJy}} d_{L,28}^2 \nu_{p,10}^{-2} \eta^{5/3} (1+z)^{-3} \right]
    \Gamma f_A^{-1} R_{17}^2, \\
    N_e = 1\times 10^{54} \left[F_{p,{\rm mJy}}^{3} d_{L,28}^{6} \nu_{p,10}^{-5} \eta^{10/3} (1+z)^{-8} \right]
    \frac{1}{f_A^2 R_{17}^4}. \\
    B = (1.3\times 10^{-2}\;{\rm G}) \left[F_{p,{\rm mJy}}^{-2} d_{L,28}^{-4} \nu_{p,10}^{5} \eta^{-\frac{10}{3}}
    (1+z)^{7} \right]
    \frac{f_A^2 R_{17}^4}{\Gamma^3}.
    \end{aligned}

.. warning::

    It is a common mistake to use Barniol-Duran equipartition formulae to infer the physical parameters of a
    system without first checking that the
    system is in the appropriate regime for the application of the formalism. In particular, the formulae above are
    only valid for sources which are in the **coasting phase** of their evolution, and thus are not decelerating. If
    the source is decelerating, then the formulae above will yield incorrect results. In such cases, one should instead
    use the more general formulae which do not assume coasting, or use a different formalism altogether.

     Additionally, Barniol-Duran is **specifically** a relativistic prescription unless additional Newtonian corrections
     are included. Thus, if the source is not relativistic, then the formulae above will also yield incorrect
     results. In such cases, one should instead use a Newtonian formalism, or use the more
     general formulae which include Newtonian corrections.

Implementation
--------------

The coasting-phase inversion described above is implemented in Triceratops as
:func:`~triceratops.radiation.synchrotron.SEDs.one_zone_closure.invert_barniol_duran_coasting`.
Given an observed peak flux density, peak frequency, and observer time, it returns the
equipartition radius :math:`R`, bulk Lorentz factor :math:`\Gamma`, total energy :math:`E`,
and the derived microphysical quantities :math:`\gamma_e`, :math:`N_e`, and :math:`B`.
See :ref:`sed_standalone_closures` for an overview of all available standalone inversion
functions.

.. footbibliography::
