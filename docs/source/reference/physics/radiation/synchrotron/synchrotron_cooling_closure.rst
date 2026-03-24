.. _synchrotron_cooling_closure:

=============================================
Synchrotron Cooling Closure
=============================================

In the default implementation of the synchrotron SED inversion routines,
the cooling Lorentz factor :math:`\gamma_c` is treated as a free parameter.
While this provides a flexible parameterization of the spectral energy
distribution (SED), it is often physically inappropriate: in many systems
the cooling Lorentz factor is determined by a specific cooling mechanism
and therefore should not be independently free.

In principle, this issue can be avoided by performing inference with a
fully self-consistent model that includes both the dynamical evolution
and the radiative cooling processes. However, such models are frequently
unnecessarily complex for rapid exploratory analysis of synchrotron data.

To address this, the Triceratops synchrotron SED framework provides a
built-in **cooling closure relation** corresponding to synchrotron
radiative cooling. This closure allows the cooling Lorentz factor to be
computed self-consistently from the magnetic field strength and the
source age, thereby removing :math:`\gamma_c` as an independent parameter.

.. important::

    We follow standard notation introduced in :ref:`synch_sed_theory` for all quantities. Please refer to the
    subsection :ref:`single_zone_sed_inversion` for details.

Overview
--------

Under synchrotron cooling, the cooling Lorentz factor evolves according
to

.. math::
   :label: synch_cooling_gamma_c

   \gamma_c
   =
   \frac{6\pi m_e c}{\sigma_T B^2 t}
   =
   \Theta_c B^{-2} t^{-1},

where

.. math::

   \Theta_c \equiv \frac{6\pi m_e c}{\sigma_T}.

If the system age :math:`t` is treated as a known model parameter,
Equation :eq:`synch_cooling_gamma_c` provides a closure relation
linking :math:`\gamma_c` and the magnetic field strength :math:`B`.

.. note::

    It is common practice to treat :math:`t` as the **dynamical time** or time since
    the explosion as measured in the comoving frame of the emitting plasma. This should, formally, be
    treated as the **effective cooling time** for the emission region.

When combined with the standard synchrotron SED normalization relations,
this closure enables a direct inversion from the observed peak frequency
and flux

.. math::

   (\nu_{\rm brk}, F_{\rm brk})

to the physical parameters

.. math::

   (B, R, \gamma_c).

Assumptions and Caveats
-----------------------

Several assumptions underlie the validity of this closure relation.

First, the closure is applicable **only in the slow-cooling regime**.
If the system is in the fast-cooling regime, the assumptions required
to maintain a single power-law electron distribution are not
self-consistent. Conversely, if cooling is negligible, the cooling
Lorentz factor does not influence the SED and the closure becomes
irrelevant.

Second, the electron energy distribution is assumed to be described
everywhere by a **single power law**. This approximation is valid when

.. math::

   \gamma_c \gg \gamma_m,

but it breaks down when the cooling break approaches the injection
break (:math:`\gamma_c \sim \gamma_m`). In that case the electron
distribution must be treated as a broken power law, and the inversion
generally requires numerical closure rather than analytic expressions.

Finally, the closure assumes that **synchrotron cooling dominates**.
If additional cooling mechanisms (e.g. inverse Compton losses) are
important, the cooling Lorentz factor is modified and the present
closure relation does not apply.

Throughout this section we adopt the standard synchrotron normalization
conventions defined in :ref:`synch_sed_theory`.

Definitions
-----------

The cooling Lorentz factor is defined by Equation
:eq:`synch_cooling_gamma_c`.

The normalization of the electron distribution is

.. math::
   :label: electron_norm_definition

   N_0
   =
   \frac{\epsilon_e B^2}
        {8\pi \epsilon_B m_e c^2 M_\gamma^{(1)}},

where :math:`M_\gamma^{(1)}` is the standard closure moment defined
in the synchrotron normalization formalism.

Inversion Strategy
------------------

In the absence of cooling closure, the SED inversion maps
:math:`(\nu_{\rm brk}, F_{\rm brk})` to physical parameters assuming
that :math:`\gamma_c` is already known.

When synchrotron cooling dominates, however,
:math:`\gamma_c` becomes a derived quantity determined by
Equation :eq:`synch_cooling_gamma_c`.
The inversion problem then becomes a system of three equations
for the three unknowns

.. math::

   B, \quad R, \quad \gamma_c.

These equations consist of

1. the SSA peak condition,
2. the optically thin normalization condition,
3. the synchrotron cooling closure relation.

Solving this system yields the desired physical parameters.

Optically Thin Inversions
^^^^^^^^^^^^^^^^^^^^^^^^^^

When the SED peak is **optically thin**, it occurs at either
:math:`\nu_m` or :math:`\nu_c`.


If the break occurs at the injection frequency :math:`\nu_m`,
the synchrotron relation gives

.. math::
   :label: nu_m_definition

   \nu_m
   =
   \frac{3 q B \sin\alpha}{4\pi m_e c}\gamma_{\min}^2
   =
   c_1 B \sin\alpha \gamma_{\min}^2.

The magnetic field strength is therefore

.. math::
   :label: B_from_nu_m

   B
   =
   \frac{\nu_{\rm brk}}
        {c_1 \gamma_{\min}^2 \sin\alpha}.

Optically Thick Inversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the observed peak corresponds to the synchrotron
self-absorption (SSA) frequency :math:`\nu_a`,
the inversion becomes more involved.

In this case the peak flux corresponds to the blackbody limit
of the synchrotron emission,

.. math::
   :label: inversion_F_a_fixed

   F_\nu(\nu_a)
   =
   2\nu_a^2 m_e \gamma_a \Omega
   =
   2 m_e \Omega \gamma_m
   \left(\frac{\nu_a}{\nu_m}\right)^{1/2}
   \nu_a^2.

Defining the constant

.. tab-set::

   .. tab-item:: Isotropic Pitch Angle

      .. math::

         P_{0,\rm ISO}
         =
         2\pi m_e f_A D_A^{-2} c_{1,\rm ISO}^{-1/2}

   .. tab-item:: Fixed Pitch Angle

      .. math::

         P_0
         =
         2\pi m_e f_A D_A^{-2}
         c_1^{-1/2}
         \sin^{-1/2}\alpha

the SSA peak condition becomes

.. math::
   :label: ssa_peak_equation

   F_{\rm brk} \nu_{\rm brk}^{-5/2}
   =
   P_0 R^2 B^{-1/2}.

A second constraint arises from the requirement that the spectral
normalization reproduce the observed peak frequency and flux.
In general this yields relations of the form

.. math::

   F_{\rm brk}^\alpha \nu_{\rm brk}^\beta
   =
   C R^\gamma B^\delta \gamma_c^\epsilon,

where the constants depend on the spectral regime.

Together with the cooling closure relation, these equations
determine :math:`B`, :math:`R`, and :math:`\gamma_c`.

----

Specific SED Regimes
--------------------

We now present the explicit inversion formulae for the various SED regimes
in the slow-cooling case.


Slow-Cooling SED
^^^^^^^^^^^^^^^^

In the canonical slow-cooling regime the peak occurs at
:math:`\nu_m`.

.. tab-set::

   .. tab-item:: Isotropic Pitch Angle

      .. math::

         \boxed{
         B
         =
         \frac{\nu_m}{c_{1,\rm ISO}\gamma_{\min}^2}
         }

   .. tab-item:: Fixed Pitch Angle

      .. math::

         \boxed{
         B\sin\alpha
         =
         \frac{\nu_m}{c_1\gamma_{\min}^2}
         }

Once the magnetic field is known, the cooling Lorentz factor follows
from Equation :eq:`synch_cooling_gamma_c`, and the radius is obtained
from the optically thin normalization.

.. tab-set::

   .. tab-item:: Isotropic Pitch Angle

      .. math::
         :label: inversion_R_iso_thin

         \boxed{
         R
         =
         \left(
         \frac{F_{m,0}}
              {Q_{m,\rm ISO} B^3 \tilde N_0}
         \right)^{1/3}
         }

   .. tab-item:: Fixed Pitch Angle

      .. math::
         :label: inversion_R_fixed_thin

         \boxed{
         R
         =
         \left(
         \frac{F_0}
              {Q_0 B^3 \tilde N_0 \sin\alpha}
         \right)^{1/3}
         }

----

Spectrum 4
^^^^^^^^^^

Spectrum 4 corresponds to the common ordering

.. math::

   \nu_m < \nu_a < \nu_c < \nu_{\rm brk}.

The SSA frequency determines the peak. The governing equations are

.. math::

   F_{\rm brk} \nu_{\rm brk}^{-5/2}
   =
   P_0 R^2 B^{-1/2}

.. math::

   \nu_{\rm brk}^{(p-1)/2} F_{\rm pk}
   =
   A R^3 B^{(p+5)/2}

and


.. math::


   \gamma_c
   =
   \Theta_c B^{-2} t^{-1}.


Solving this system yields

.. tab-set::

    .. tab-item:: Isotropic Pitch Angle

        .. math::

            \boxed{
            \begin{aligned}
            R &=
            \Theta_c^{-1/(2p+5)}
            A^{-1/(2p+5)}
            P_{0,\rm ISO}^{-(p+1)/(2p+5)}
            F_{\rm brk}^{(p+2)/(2p+5)}
            \nu_{\rm brk}^{-(2p+3)/(2p+5)}
            t^{1/(2p+5)} \\
            B &=
            \Theta_c^{-4/(2p+5)}
            A^{-4/(2p+5)}
            P_{0,\rm ISO}^{6/(2p+5)}
            F_{\rm brk}^{-2/(2p+5)}
            \nu_{\rm brk}^{(2p+13)/(2p+5)}
            t^{4/(2p+5)}\\
            \gamma_c &= \Theta_c^{(2p+13)/(2p+5)}
            A^{8/(2p+5)}
            P_{0,\rm ISO}^{-12/(2p+5)}
            F_{\rm brk}^{4/(2p+5)}
            \nu_{\rm brk}^{-2(2p+13)/(2p+5)}
            t^{-(2p+13)/(2p+5)}
            \end{aligned}
            }

        where

        .. math::

            A = \left[c_{1,\rm ISO} \gamma_m^2 \right]^{(p-1)/2} (\sin^{(p+1)/2} \alpha) Q_{m,\rm ISO} \tilde{N}_0,

        and :math:`\tilde{N}_0` is computed using our assumptions about the electron distribution from above.

    .. tab-item:: Fixed Pitch Angle

        .. math::

            \boxed{
            \begin{aligned}
            R &=
            \Theta_c^{-1/(2p+5)}
            A^{-1/(2p+5)}
            P_0^{-(p+1)/(2p+5)}
            F_{\rm brk}^{(p+2)/(2p+5)}
            \nu_{\rm brk}^{-(2p+3)/(2p+5)}
            t^{1/(2p+5)} \\
            B &=
            \Theta_c^{-4/(2p+5)}
            A^{-4/(2p+5)}
            P_0^{6/(2p+5)}
            F_{\rm brk}^{-2/(2p+5)}
            \nu_{\rm brk}^{(2p+13)/(2p+5)}
            t^{4/(2p+5)}\\
            \gamma_c &= \Theta_c^{(2p+13)/(2p+5)}
            A^{8/(2p+5)}
            P_0^{-12/(2p+5)}
            F_{\rm brk}^{4/(2p+5)}
            \nu_{\rm brk}^{-2(2p+13)/(2p+5)}
            t^{-(2p+13)/(2p+5)}
            \end{aligned}
            }

        where

        .. math::

            A = \left[c_1 \gamma_m^2 \sin \alpha\right]^{(p-1)/2} (\sin^{(p+1)/2} \alpha) Q_{m,0} \tilde{N}_0,

        and :math:`\tilde{N}_0` is computed using our assumptions about the electron distribution from above.

----

Spectrum 7
^^^^^^^^^^

In Spectrum 7 the cooling break couples directly to the inversion.

The governing equations are

.. math::

   F_{\rm brk}\nu_{\rm brk}^{-5/2}
   =
   P_0 R^2 B^{-1/2}

.. math::

   \nu_{\rm brk}^{p/2}F_{\rm pk}
   =
   A\gamma_c R^3 B^{(p+6)/2}


.. math::


   \gamma_c
   =
   \Theta_c B^{-2} t^{-1}

Solving this system yields

.. tab-set::

    .. tab-item:: Isotropic Pitch Angle

        .. math::

            \boxed{
            \begin{aligned}
            R &=
            \Theta_c^{-1/(2p+7)}
            A^{-1/(2p+7)}
            P_{0,\rm ISO}^{-(p+6)/(2p+7)}
            F_{\rm brk}^{(p+7)/(2p+7)}
            \nu_{\rm brk}^{-(2p+5)/(2p+7)}
            t^{1/(2p+7)} \\
            B &=
            \Theta_c^{-4/(2p+7)}
            A^{-4/(2p+7)}
            P_{0,\rm ISO}^{6/(2p+7)}
            F_{\rm brk}^{-2/(2p+7)}
            \nu_{\rm brk}^{(2p+15)/(2p+7)}
            t^{4/(2p+7)}\\
            \gamma_c &= \Theta_c^{(2p+15)/(2p+7)}
            A^{8/(2p+7)}
            P_{0,\rm ISO}^{-12/(2p+7)}
            F_{\rm brk}^{4/(2p+7)}
            \nu_{\rm brk}^{-2(2p+15)/(2p+7)}
            t^{-(2p+15)/(2p+7)}
            \end{aligned}
            }

        where

        .. math::

            A = c_{1,\rm ISO}^{p/2} \gamma_m^{p-1} \alpha Q_{m,\rm ISO} \tilde{N}_0,

        and :math:`\tilde{N}_0` is computed using our assumptions about the electron distribution from above.

    .. tab-item:: Fixed Pitch Angle

        .. math::

            \boxed{
            \begin{aligned}
            R &=
            \Theta_c^{-1/(2p+7)}
            A^{-1/(2p+7)}
            P_0^{-(p+6)/(2p+7)}
            F_{\rm brk}^{(p+7)/(2p+7)}
            \nu_{\rm brk}^{-(2p+5)/(2p+7)}
            t^{1/(2p+7)} \\
            B &=
            \Theta_c^{-4/(2p+7)}
            A^{-4/(2p+7)}
            P_0^{6/(2p+7)}
            F_{\rm brk}^{-2/(2p+7)}
            \nu_{\rm brk}^{(2p+15)/(2p+7)}
            t^{4/(2p+7)}\\
            \gamma_c &= \Theta_c^{(2p+15)/(2p+7)}
            A^{8/(2p+7)}
            P_0^{-12/(2p+7)}
            F_{\rm brk}^{4/(2p+7)}
            \nu_{\rm brk}^{-2(2p+15)/(2p+7)}
            t^{-(2p+15)/(2p+7)}
            \end{aligned}
            }

        where

        .. math::

            A = c_1^{p/2} \gamma_m^{p-1} \gamma_c \sin^{p+2/2} \alpha Q_{m,0} \tilde{N}_0,

        and :math:`\tilde{N}_0` is computed using our assumptions about the electron distribution from above.
