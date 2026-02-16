.. _synchrotron_theory:
===================================
Synchrotron Radiation Theory
===================================

Synchrotron radiation is one of the most fundamental radiative processes in the world of transient astrophysics. It is
responsible for a wide variety of observed phenomena, from the radio afterglows of gamma-ray bursts (GRBs) to the emission from
active galactic nuclei (AGN) jets. Because Triceratops is so heavily reliant on synchrotron radiation for modeling
transient sources, it is important to have a solid understanding of the theory; particularly the conventions and
terminology used throughout the Triceratops documentation.

In this documentation, we seek to provide a concise overview of the topics one needs to understand to begin using
Triceratops for synchrotron modeling. We cannot hope to provide a comprehensive review of synchrotron radiation theory;
however, we provide references throughout to more detailed treatments of the subject and the relevant literature
for various results.

.. contents::
    :local:
    :depth: 2

Synchrotron From a Single Electron
----------------------------------

To describe, from first principles, the relevant synchrotron theory used in Triceratops, we first need to develop
the emission of a single relativistic electron spiraling in a magnetic field. With that in hand, it will be a
relatively straightforward process to extend to a population of electrons with some distribution function.

Synchrotron emission is the relativistic generalization of cyclotron emission, which is the radiation emitted
by a charged particle spiraling in a magnetic field. The key difference is that in the relativistic case, the
emission is beamed in the direction of motion of the particle, leading to a characteristic spectrum and angular distribution.

From the Fourier uncertainty principle, the shorter a pulse of radiation lasts in time, the broader its frequency
composition must be. Thus, for the relativistic electrons in this case, the emission is concentrated in short pulses
as the electron's velocity vector sweeps past the observer. This leads to a characteristic synchrotron spectrum which
can extend to very high frequencies.

The Cyclotron Frequency
^^^^^^^^^^^^^^^^^^^^^^^^^

In the classical case, we consider a magnetic field :math:`{\bf B} = B_0 \hat{\bf z}` and a particle of
charge :math:`q` bound to the :math:`x-y`plane. The **Lorentz Force** dictates that

.. math::

    {\bf F} = m {\bf a} = \frac{q}{c}{\bf v} \times {\bf B}.

This will cause gyroscopic motion about the field lines. We imagine constructing our coordinate system
so that :math:`{\bf v}_0 = v_0 \hat{\bf x}.` Then we achieve a force always directed inward of magnitude

.. math::

    F = \frac{qB_0v_0}{c} = m\omega^2 r \implies \omega = \frac{qB_0}{mc}.

This is the **cyclotron frequency** of the particle:

.. math::

    \boxed{
    \omega_{\rm cyclotron} = \frac{qB_0}{mc},\;\text{(gaussian units)}.
    }

In the relativistic case, we must account for time dilation. The relativistic gyrofrequency is therefore

.. math::

    \boxed{
    \omega_B = \frac{qB_0}{\gamma mc},\;\text{(gaussian units)}.
    }

Synchrotron Power
^^^^^^^^^^^^^^^^^^^^^

Already, because we know the acceleration of the electron, we can compute the total power emitted from synchrotron
radiation using the Larmor formula. The Larmor formula states that the power emitted by an accelerating charge is

.. math::

    P = \frac{2}{3} \frac{q^2 a^2}{c^3}.

Given that the acceleration from the Lorentz force must maintain circular motion, we have

.. math::

    a = \omega_B^2 r = \omega_B^2 v_{\perp} = \frac{qB_0 v_\perp}{\gamma m c}

so,

.. math::

    P = \frac{2}{3} \frac{q^4 \gamma^2 B^2}{c^5 m^2} v_{\perp}^2 = \frac{4}{3} \sigma_T c \beta^2 \gamma^2 U_B.

In the relativistic limit (:math:`\beta \approx 1`), this becomes

.. math::

    \boxed{
    P_{\rm synch} = \frac{4}{3} \sigma_T c \gamma^2 U_B,
    }

where :math:`U_B = B^2/8\pi` is the magnetic energy density.

The Characteristic Synchrotron Frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The critical physical insight that distinguishes **synchrotron radiation** from its
non-relativistic cyclotron counterpart is the emergence of **strong relativistic beaming**.
Although an accelerated charged particle emits radiation continuously in its instantaneous
rest frame, relativistic aberration causes that emission to be concentrated into a narrow
forward cone of angular width :math:`\Delta\theta \sim 1/\gamma` in the lab frame.

As a relativistic particle spirals along a magnetic field line, this forward emission cone
does not remain pointed toward a distant observer. Instead, the observer receives radiation
only during the brief interval when the cone sweeps across the line of sight. The observed
signal therefore takes the form of a **temporally localized pulse**, rather than a continuous
sinusoidal wave.

From Fourier theory, a signal confined to a short interval in time necessarily corresponds to
a **broad frequency spectrum**. Thus, even though the underlying motion of the particle is
periodic, relativistic beaming transforms what would otherwise be narrow-band cyclotron
emission into the broadband synchrotron spectrum characteristic of relativistic sources.

The particle emits
radiation into a cone of angular width :math:`\Delta\theta \sim 2/\gamma`. As the particle
moves along its curved trajectory of radius :math:`a`, the arc length over which the observer
remains inside the beam is approximately

.. math::

    \Delta s \simeq \frac{2a}{\gamma}.

The radius of the particle’s helical trajectory is

.. math::

    r = \frac{v}{\omega_B \sin\varphi},

where :math:`\varphi` is the pitch angle between the particle velocity and the magnetic field,
and :math:`\omega_B = qB/(\gamma m_e c)` is the relativistic gyrofrequency.

The pulse begins when the edge of the emission cone first intersects the observer’s line of
sight and ends when it exits the cone. The corresponding emission time interval is therefore

.. math::

    \Delta t_{\rm emit}
    \simeq
    \frac{2}{\gamma \omega_B \sin\varphi}.

Because photons emitted at the beginning and end of the pulse originate from different
locations along the trajectory, there is an additional light-travel-time compression factor.
Accounting for this effect yields an observed pulse duration

.. math::

    \Delta t_{\rm obs}
    \simeq
    \frac{2}{\gamma \omega_B \sin\varphi}(1-\beta).

In the ultra-relativistic limit, :math:`1-\beta \simeq 1/(2\gamma^2)`, so the pulse duration
scales as

.. math::

    \Delta t_{\rm obs}
    \sim
    \frac{1}{\gamma^3 \omega_B \sin\varphi}.

This timescale defines a characteristic upper frequency in the synchrotron spectrum,
corresponding to the inverse pulse duration. We therefore define the **characteristic synchrotron
frequency** as

.. math::

    \omega_c
    =
    \frac{3}{2}\,\gamma^3 \omega_B \sin\varphi.

Expressed as a linear frequency, this becomes

.. math::
    :label: eq_synch_frequency

    \boxed{
    \nu_c
    =
    \frac{3}{4\pi}\,\gamma^3 \omega_B \sin\varphi
    =
    \frac{3}{4\pi}\,\gamma^2\,\frac{qB}{m_e c}\,\sin\varphi.
    }

It is common to introduce the constant :math:`c_1`:footcite:p:`1970ranp.book.....P`, which is defined such that

.. math::

    \nu_c = c_1 B E^2 \sin \varphi,\;\;c_1 = \frac{3e}{4\pi m_e^3 c^5} = 6.27\times 10^{18}\;{\rm cgs}.

.. note::

    For clarity: we refer to this frequency as it is defined in :footcite:t:`RybickiLightman` as the
    **characteristic synchrotron frequency**. In the context of *populations* of electrons, conventions diverge;
    however, we prefer to contain those differences of convention to that context. See the section on
    electron populations for more details.

To provide a thorough and consistent theoretical foundation, it is here useful to introduce equivalent notation
for :math:`c_1` in terms of the :math:`\gamma` instead of :math:`E`. We therefore define

.. math::

    c_{1,\gamma} = \frac{3}{4\pi} \frac{e}{m_e c} = 8.59 \times 10^6\;{\rm cm^{1/2}\;g^{-1/2}}.

As a useful reference, for :math:`\gamma = 1` is a 1 Gauss field, the characteristic synchrotron frequency is therefore

.. math::

    \nu_c(\gamma=1, B=1\,{\rm G}, \sin\varphi=1) = c_{1,\gamma} = 8.59 \times 10^6\;{\rm Hz}.

For a :math:`\gamma = 10^2`,

.. math::

    \nu_c(\gamma=10^2, B=1\,{\rm G}, \sin\varphi=1) = 8.59 \times 10^{10}\;{\rm Hz}.

Thus, one expects that, in many scenarios, synchrotron emission from relativistic electrons will peak in the radio
band.

.. dropdown:: Pitch-Angle Averaging

    Throughout Triceratops, we provide two standard options for handing the pitch angle dependence of synchrotron
    emission: (a) assume a fixed pitch angle, or (b) perform a pitch-angle average assuming an isotropic
    distribution of pitch angles.

    To perform the pitch angle averaged calculations, we perform this averaging when computing the *emissivity*,
    *absorption*, and related representations of the ensemble synchrotron emission. We therefore **DO NOT** define
    a quantity :math:`\nu_{\rm m,iso}` or similar, instead opting to perform the averaging directly on the relevant
    quantities.

The Single Electron Synchrotron Spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section, we will derive the synchrotron spectrum of a single electron spiraling in a magnetic field. From
elementary electrodynamics (see :footcite:t:`RybickiLightman` for a detailed treatment), the power radiated per unit
frequency per unit solid angle is given by

.. math::

    \frac{dE}{dt\,d\omega d\Omega} = \frac{q^2 \omega^2}{4\pi^2 c}\left|
    \int_{-\infty}^{\infty} \frac{\hat{n} \times (\hat{n} \times \beta)}{1-\hat{n} \cdot \beta}
    e^{i\omega(t' - \hat{n} \cdot r(t')/c)} dt' \right|^2.

Consider an electron such that, at a retarded time :math:`t' = 0`, it is a distance :math:`a` from the origin and has
velocity along the x-axis. Let
the observer be oriented along :math:`\hat{n}` and let the vector from the electron's position to the origin be
:math:`\boldsymbol{\epsilon}_\perp`. Let :math:`\boldsymbol{\epsilon}_\parallel = \hat{n} \times
\boldsymbol{\epsilon}_\perp` be the vector perpendicular to both :math:`\hat{n}` and :math:`\boldsymbol{\epsilon}_\perp`.
Finally, let :math:`\theta` be the angle between :math:`\hat{\bf n}` and the x-axis.

In this geometry, the phase angle of the electron in its orbit is just

.. math::

    \phi = \frac{vt'}{a},

and the velocity vector is

.. math::

    \boldsymbol{\beta} = \beta \left(\cos \phi \hat{\bf x} + \sin \phi \boldsymbol{\epsilon}_\perp \right).

Thus, the cross product in the integrand becomes

.. math::

    \hat{n} \times (\hat{n} \times \beta) = \beta\left(\cos \phi \sin \theta \boldsymbol{\epsilon}_\parallel -
    \sin \phi \boldsymbol{\epsilon}_\perp \right).

The phase term in the exponential is

.. math::

    t - \frac{\hat{n} \cdot r(t')}{c} = t' - \frac{a}{c} \cos \theta \sin \phi.

After some simplifications (see the drop down), we arrive at the final result for the power radiated per unit frequency
per unit solid angle:

.. dropdown:: Small-Angle Expansion and Evaluation of the Master Integral

    We now evaluate the master expression for the radiated energy in the relativistic limit.
    For ultra-relativistic particles we take :math:`\beta \simeq 1`, and because synchrotron
    emission is concentrated into short pulses and strongly beamed, we may expand all angular
    quantities to lowest nontrivial order.

    In particular, for small angles :math:`\theta` and :math:`\phi` we use

    .. math::

        \cos\theta \simeq 1 - \frac{\theta^2}{2},
        \qquad
        \sin\phi \simeq \phi - \frac{\phi^3}{6},

    which implies

    .. math::

        \cos\theta\,\sin\phi
        \simeq
        \phi - \frac{\phi\theta^2}{2} - \frac{\phi^3}{6}.

    Using the orbital phase relation :math:`\phi = vt'/a`, we obtain

    .. math::

        \cos\theta\,\sin\phi
        \simeq
        \frac{vt'}{a}\left(1 - \frac{\theta^2}{2}\right)
        - \frac{v^3 t'^3}{6a^3}.

    Substituting this into the phase factor of the exponential yields

    .. math::

        t - \frac{\hat{n}\cdot r(t')}{c}
        =
        (1 - \beta)t'
        + \frac{\beta\theta^2}{2}t'
        + \frac{\beta^3 c^2 t'^3}{6a^2}.

    Factoring out :math:`\Gamma^{-2} \simeq 2(1-\beta)`, this may be written as

    .. math::

        t
        =
        \frac{1}{2\Gamma^2}
        \left[
            \left(1 + \Gamma^2\theta^2\right)t'
            + \frac{c^2\Gamma^2 t'^3}{3a^2}
        \right].

    The master formula may now be decomposed into its perpendicular and parallel polarization
    components. After simplification, we find

    .. math::

        \begin{aligned}
        \frac{dE_\perp}{dt\,d\omega\,d\Omega}
        &=
        \frac{q^2\omega^2}{4\pi^2 c}
        \left|
        \int_{-\infty}^{\infty}
        \frac{ct'}{a}
        \exp\!\left[
            \frac{i\omega}{2\Gamma^2}
            \left(
                \Theta_\gamma^2 t'
                + \frac{c^2\Gamma^2 t'^3}{3a^2}
            \right)
        \right]
        dt'
        \right|^2, \\[6pt]
        \frac{dE_\parallel}{dt\,d\omega\,d\Omega}
        &=
        \frac{q^2\omega^2\theta^2}{4\pi^2 c}
        \left|
        \int_{-\infty}^{\infty}
        \exp\!\left[
            \frac{i\omega}{2\Gamma^2}
            \left(
                \Theta_\gamma^2 t'
                + \frac{c^2\Gamma^2 t'^3}{3a^2}
            \right)
        \right]
        dt'
        \right|^2,
        \end{aligned}

    where we have defined

    .. math::

        \Theta_\gamma^2 \equiv 1 + \Gamma^2\theta^2.

    Although these integrals appear formidable, they may be reduced to standard forms by
    introducing the dimensionless variables

    .. math::

        y = \Gamma \frac{ct'}{a\Theta_\gamma},
        \qquad
        \eta = \frac{\omega a \Theta_\gamma^3}{3c\Gamma^3}.

    In terms of these variables, the expressions simplify to

    .. math::

        \boxed{
        \begin{aligned}
        \frac{dE_\perp}{dt\,d\omega\,d\Omega}
        &=
        \frac{q^2\omega^2}{4\pi^2 c}
        \left(\frac{a\Theta_\gamma^2}{\Gamma^2 c}\right)^2
        \left|
        \int_{-\infty}^{\infty}
        y
        \exp\!\left[
            \frac{3i\eta}{2}
            \left(
                y + \frac{y^3}{3}
            \right)
        \right]
        dy
        \right|^2, \\[6pt]
        \frac{dE_\parallel}{dt\,d\omega\,d\Omega}
        &=
        \frac{q^2\omega^2\theta^2}{4\pi^2 c}
        \left(\frac{a\Theta_\gamma}{\Gamma c}\right)^2
        \left|
        \int_{-\infty}^{\infty}
        y
        \exp\!\left[
            \frac{3i\eta}{2}
            \left(
                y + \frac{y^3}{3}
            \right)
        \right]
        dy
        \right|^2.
        \end{aligned}
        }

    These integrals are well known and may be expressed in terms of modified Bessel functions.
    Evaluating them yields the standard synchrotron expressions

    .. math::

        \boxed{
        \begin{aligned}
        \frac{dE_\perp}{dt\,d\omega\,d\Omega}
        &=
        \frac{q^2\omega^2}{3\pi c}
        \left(\frac{a\Theta_\gamma^2}{\Gamma^2 c}\right)^2
        K_{2/3}^2(\eta), \\[6pt]
        \frac{dE_\parallel}{dt\,d\omega\,d\Omega}
        &=
        \frac{q^2\omega^2\theta^2}{3\pi^2 c}
        \left(\frac{a\Theta_\gamma}{\Gamma c}\right)^2
        K_{1/3}^2(\eta),
        \end{aligned}
        }

    where :math:`K_\nu` denotes the modified Bessel function of the second kind. Because the
    emission is strongly beamed, the dominant contribution arises near :math:`\theta \simeq 0`,
    in which case

    .. math::

        \eta \simeq \frac{\omega}{2\omega_c},

    with :math:`\omega_c` the characteristic synchrotron frequency.

.. math::

    \boxed{
    \begin{aligned}
    \frac{dE_\perp}{dt\,d\omega\,d\Omega}
    &=
    \frac{q^2\omega^2}{3\pi c}
    \left(\frac{a\Theta_\gamma^2}{\Gamma^2 c}\right)^2
    K_{2/3}^2(\eta), \\[6pt]
    \frac{dE_\parallel}{dt\,d\omega\,d\Omega}
    &=
    \frac{q^2\omega^2\theta^2}{3\pi^2 c}
    \left(\frac{a\Theta_\gamma}{\Gamma c}\right)^2
    K_{1/3}^2(\eta),
    \end{aligned}
    }

where :math:`K_\nu` denotes the modified Bessel function of the second kind. Because the
emission is strongly beamed, the dominant contribution arises near :math:`\theta \simeq 0`,
in which case

.. math::

    \eta \simeq \frac{\omega}{2\omega_c},

with :math:`\omega_c` the characteristic synchrotron frequency.

Because the radiation is strongly beamed, it will only emit into a thin band of solid angles with width approximately
:math:`1/\gamma` centered around the cone :math:`1/\alpha` (see :footcite:t:`RybickiLightman` for details). Integrating
over solid angle, we find the total power radiated per unit frequency is

.. dropdown:: Details

.. math::

    \begin{aligned}
        \frac{dE_\perp}{dt\;d\omega} &= \frac{\sqrt{3} q^2 \Gamma \sin \alpha}{2c}\left(F(x)+G(x)\right)\\
        \frac{dE_\parallel}{dt\;d\omega} &= \frac{\sqrt{3}q^2 \Gamma \sin \alpha}{2c} \left(F(x)-G(x)\right)
    \end{aligned}

where

.. math::

    F(x) = x \int_x^\infty K_{5/3}(\xi)\;d\xi,\;\;G(x) = xK_{2/3}(x),

and :math:`x = \frac{\omega}{\omega_c}`.

.. note::

    This is *not* a typo: :math:`\eta = \frac{\omega}{2\omega_c}` in the previous section, but here we have
    :math::`x = \frac{\omega}{\omega_c}`.

Summing these two contributions, the total power radiated per unit frequency is

.. math::

    \boxed{
    P(\omega) = \frac{\sqrt{3}q^3 B \sin\alpha}{2\pi m c^2} F\left(\frac{\omega}{\omega_c}\right),
    }

In terms of frequency,

.. math::

    \boxed{
    P(\nu) = \frac{\sqrt{3}q^3 B \sin\alpha}{m c^2} F\left(\frac{\nu}{\nu_c}\right),
    }

where :math:`\nu_c` is the characteristic synchrotron frequency defined previously.

.. important::

    In Triceratops, :math:`F(x)` is referred to as the **first synchrotron kernel** and :math:`G(x)` as
    the **second synchrotron kernel**.


.. _synch_theory_populations:
Synchrotron From A Population of Electrons
------------------------------------------

.. important::

    At this point, convention in the literature diverges in certain respects. As we describe the relevant theory,
    we seek to be (a) explicit regarding the conventions and (b) true to a **single convention**: that used in
    :footcite:t:`1970ranp.book.....P`, which has become the general standard in the astrophysical literature
    (e.g. :footcite:t:`demarchiRadioAnalysisSN2004C2022`, :footcite:t:`Margutti2019COW`, etc.).


With the all-important single-electron synchrotron spectrum in hand, we can now extend our arguments to the
emission from a population of electrons. There are generally two ways to define such a population: either by
**energy** or by **Lorentz Factor**.

.. note::

    In Triceratops, we offer implementations and support for the use of *both approaches*; however, the
    Lorentz factor approach is the **standard convention** unless the documentation clearly indicates otherwise.

    In this documentation, we will always write the Lorentz factor distribution as :math:`\frac{dN}{d\gamma}`,
    and the energy distribution as :math:`\frac{dN}{dE}`.

Let the distribution function be :math:`\frac{dN}{d\gamma}`. Then,

.. math::

    P(\nu) = \int_{\gamma_{\rm min}}^{\gamma_{\rm max}} \frac{dN}{d\gamma} P(\nu, \gamma) d\gamma =
    \frac{\sqrt{3} q^3 B \sin \alpha}{mc^2} \int_{\gamma_{\rm min}}^{\gamma_{\rm max}} \frac{dN}{d\gamma} F\left(
        \frac{\nu}{\nu_c(\gamma)} \right) d\gamma,

where :math:`P(\nu, \gamma)` is the single-electron synchrotron power derived previously.

Now, in general, this integration is not analytically tractable for arbitrary distribution functions; however, there
are some scenarios which are of particular relevance to astrophysical sources where analytic results can be obtained.

.. note::

    Support for performing these integrations numerically for arbitrary distribution functions is a planned
    feature of Triceratops, but not currently implemented.

The Spectrum of a Power-Law Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most important case in which the above formalism has a known closure is that of a power-law distribution
of electrons. This is the most common assumption in astrophysical synchrotron modeling, as it is both
theoretically motivated by diffusive shock acceleration :footcite:t:`caprioliParticleAccelerationShocks2023` and
empirically successful in explaining non-thermal emission across a wide range of astrophysical environments.

In the literature, there are two typical ways to parameterize a power-law distribution of electrons: either in
terms of :math:`\gamma` or in terms of energy :math:`E`:

.. tab-set::

    .. tab-item:: Lorentz Factor Distribution

        .. math::

            \frac{dN}{d\gamma} = N_0 \gamma^{-p},\;\;\gamma_{\min} \le \gamma \le \gamma_{\max}.

    .. tab-item:: Energy Distribution

        .. math::

            \frac{dN}{dE} = N_{0,E} E^{-p},\;\;E_{\min} \le E \le E_{\max}.

Given the normalization in one parameterization, the other normalization can be computed via

.. math::

    \frac{dN}{dE} = \frac{dN}{d\gamma} \frac{d\gamma}{dE} \implies N_{0,E} = N_0 (m_e c^2)^{p-1}.


.. important::

    In Triceratops, we strive to fully support both such parameterizations; however, the Lorentz factor
    parameterization is the **standard convention** unless the documentation clearly indicates otherwise.

Now, given a power-law distribution in Lorentz factor:

.. math::

    \frac{dN}{d\gamma} = N_0 \gamma^{-p},\;\;\gamma_{\min} \le \gamma \le \gamma_{\max},

we can perform the integral over the electron distribution to find the total synchrotron power. The
*power per unit frequency per unit volume* is given by :footcite:p:`RybickiLightman`:

.. math::

    P(\nu) = \frac{\sqrt{3} e^3}{4m_e c^2} N_0 \left(B \sin \alpha\right)^{(p+1)/2} \left(\frac{p+(7/3)}{p+1}\right)
    \Gamma\left(\frac{3p-1}{12}\right) \Gamma\left(\frac{3p+7}{12}\right) \left(\frac{2\pi m_e c \nu}{3e}\right)^{-(p-1)/2}.

.. hint::

    The exact expression in :footcite:t:`RybickiLightman` is identical but uses identities of the
    Gamma function to rewrite the result in a slightly different form.

While this result is not immediately appealing, it can, in large part, be reduced to some critical scalings and
relevant constants. We can write the frequency term as

.. math::

    \left(\frac{2\pi m_e c\nu}{3e}\right)^{-(p-1)/2} = \left(m_e c^2\right)^{(p-1)}\left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

We can then also introduce another useful piece of notation adopted from :footcite:t:`1970ranp.book.....P`:

.. math::

    c_5(p) = \frac{\sqrt{3}}{16 \pi} \left(\frac{e^3}{m_ec^2}\right) \frac{p+7/3}{p+1} \Gamma\left(\frac{3p-1}{12}\right)
    \Gamma\left(\frac{3p + 7}{12}\right).

We therefore have the critical result for the total power in either of the two parameterizations:

.. tab-set::

    .. tab-item:: Lorentz Factor Distribution

        .. math::

            P(\nu) = 4\pi c_5(p) N_0 \left(m_e c^2\right)^{(p-1)} \left(B \sin \alpha\right)^{(p+1)/2}
            \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

    .. tab-item:: Energy Distribution

        .. math::

            P(\nu) = 4\pi c_5(p) N_{0,E} \left(B \sin \alpha\right)^{(p+1)/2}
            \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

Emissivity
~~~~~~~~~~

Another useful quantity in synchrotron radiation theory is the **emissivity**,
:math:`j_\nu`, defined as the power emitted per unit frequency per unit volume per
unit solid angle.

Because synchrotron emission depends explicitly on the angle between the electron
velocity and the magnetic field, it is useful to first distinguish between:

1. the emissivity for electrons radiating at a **fixed pitch angle**, and
2. the **angle-averaged emissivity** appropriate for an isotropic electron population.

We treat these cases in turn.

**Single Pitch-Angle Emissivity**:

Consider a population of electrons radiating at a single pitch angle
:math:`\alpha` relative to the magnetic field.
For a power-law distribution of electrons, the synchrotron emissivity at that
pitch angle is

.. tab-set::

    .. tab-item:: Lorentz Factor Distribution

        .. math::

            j_\nu(\alpha)
            =
            c_5(p)\,N_0\,(m_e c^2)^{p-1}
            \left(B\sin\alpha\right)^{(p+1)/2}
            \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

    .. tab-item:: Energy Distribution

        .. math::

            j_\nu(\alpha)
            =
            c_5(p)\,N_{0,E}
            \left(B\sin\alpha\right)^{(p+1)/2}
            \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

These expressions represent the emissivity **per unit solid angle**, evaluated for
electrons whose velocities make a fixed angle :math:`\alpha` with the magnetic
field.

They are therefore *not* yet the physically relevant emissivity unless the electron
population truly occupies a single pitch angle.

**Angle-Averaged Emissivity**:

Given that one knows the emissivity at a fixed pitch angle and the distribution of
pitch angles, one can compute the angle-averaged emissivity by integrating over
the pitch-angle distribution:

.. math::

    j_\nu = \int_0^\pi j_\nu(\alpha)\,f(\alpha)\,d\alpha.

In most astrophysical applications, electrons are assumed to have an **isotropic
distribution of pitch angles** (however, some studies have considered the effects of anisotropy
:footcite:p:`YangPitchAngles`).
In this case, the observed emissivity must be averaged over the pitch-angle
distribution.

For an isotropic velocity distribution, the probability density for the pitch
angle is

.. math::

    f(\alpha)\,d\alpha
    =
    \frac{1}{2}\,\sin\alpha\,d\alpha,
    \qquad
    0 \le \alpha \le \pi.

The angle-averaged emissivity is therefore

.. math::

    j_\nu
    =
    \int_0^\pi j_\nu(\alpha)\,f(\alpha)\,d\alpha
    =
    \frac{1}{2}
    \int_0^\pi
    j_\nu(\alpha)\,\sin\alpha\,d\alpha.

Substituting the fixed-angle emissivity gives

.. math::

    j_\nu
    \propto
    B^{(p+1)/2}
    \int_0^\pi
    \sin^{(p+1)/2}\!\alpha\;\sin\alpha\;d\alpha
    =
    B^{(p+1)/2}
    \int_0^\pi
    \sin^{(p+3)/2}\!\alpha\;d\alpha.

The remaining integral is a standard one :footcite:p:`CRCTables`:

.. math::

    \int_0^\pi \sin^k\alpha\;d\alpha
    =
    \sqrt{\pi}\,
    \frac{\Gamma\!\left(\frac{k+1}{2}\right)}
         {\Gamma\!\left(\frac{k+2}{2}\right)}.

Letting

.. math::

    c_{5,\rm ISO}(p) = c_5(p) \Gamma\left(\frac{p+5}{4}\right) \Gamma^{-1}\left(\frac{p+7}{4}\right),

the angle-averaged emissivity for an isotropic distribution of pitch angles is

.. tab-set::

    .. tab-item:: Lorentz Factor Distribution

        .. math::

            j_\nu(\alpha)
            =
            c_{\rm 5, ISO}(p)\,N_0\,(m_e c^2)^{p-1}
            \left(B\right)^{(p+1)/2}
            \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

    .. tab-item:: Energy Distribution

        .. math::

            j_\nu(\alpha)
            =
            c_{\rm 5, ISO}(p)\,N_{0,E}
            \left(B\right)^{(p+1)/2}
            \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.


Conventions in the Literature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two common conventions appear in synchrotron modeling:

- Some authors (e.g. :footcite:t:`Chevalier1998SynchrotronSelfAbsorption`) simply
  approximate :math:`\sin\alpha \simeq 1`, effectively assuming that all electrons
  radiate perpendicular to the magnetic field.

- Others (e.g. :footcite:t:`ChevalierFranssonHandbook`, :footcite:t:`Pacholczyk1970`)
  explicitly perform the isotropic pitch-angle average described above.

Both conventions lead to the same **spectral scalings**, differing only by factors
of order unity.

.. note::

    Triceratops supports **both** approaches. The low-level API exposes the
    pitch-angle–dependent emissivity :math:`j_\nu(\alpha)`, while the high-level
    routines return the angle-averaged emissivity using the isotropic assumption.

.. _synch_equipartition_theory:
Microphysical Closures and Equipartition
----------------------------------------

A critical element of most synchrotron modeling pipelines is a treatment of the coupling between the dynamics
of the radiating material and the resulting synchrotron emission. A detailed treatment of this coupling is generally
beyond the scope of most modeling efforts, and so we instead rely on a set of microphysical parameters that
parameterize our ignorance of the detailed physics at play.

Equipartition
^^^^^^^^^^^^^^

The most common approach to microphysical closure is to assume some form of equipartition between the energy
in the magnetic fields and relativistic electrons and the thermal energy of the shocked plasma. This approach
is motivated by the idea that shocks are efficient at converting kinetic energy into thermal energy, and that
some fraction of this thermal energy is then partitioned into magnetic fields and relativistic particles.

A common choice throughout the modern literature (e.g. :footcite:t:`Margutti2019COW`,
:footcite:t:`demarchiRadioAnalysisSN2004C2022`, :footcite:t:`wuDelayedRadioEmission2025`, etc.) is to introduce
the parameters :math:`\epsilon_e` and :math:`\epsilon_B`, which represent the fraction of the thermal energy density
that goes into relativistic electrons and magnetic fields, respectively.

Given a thermal energy density :math:`U_{\rm thermal}`, we can then write the energy densities in relativistic electrons
and magnetic fields as:

.. math::

    U_e = \epsilon_e U_{\rm thermal},

and

.. math::

    U_B = \epsilon_B U_{\rm thermal} = \frac{B^2}{8 \pi},

where :math:`\epsilon_e` and :math:`\epsilon_B` are dimensionless parameters typically in the range
:math:`0 < \epsilon_e, \epsilon_B < 1`, and :math:`U_B` is the magnetic energy density given by

.. math::

    U_B = \frac{B^2}{8 \pi},

and :math:`B` is the magnetic field strength.

.. note::

    The resulting implications for the emission of synchrotron is dictated by the way that :math:`U_e` is distributed
    into the population of relativistic electrons. In the most common scenario, where we have a power-law distribution of
    relativistic electrons, equipartition can uniquely determine the normalization of the distribution given
    :math:`\epsilon_e`, :math:`\epsilon_B`, and :math:`U_{\rm thermal}`.


Power-Law Electron Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A central assumption in most synchrotron emission models is that the population of relativistic
electrons accelerated by a shock follows a power-law distribution in Lorentz factor (and energy). This assumption
is motivated both by theoretical considerations of diffusive shock
acceleration:footcite:p:`caprioliParticleAccelerationShocks2023` and by its empirical success in explaining non-thermal
emission across a wide range of astrophysical environments.

We write the electron distribution function as

.. math::

    \frac{dN}{d\gamma} = N_0 \gamma^{-p},
    \qquad
    \gamma_{\min} \le \gamma \le \gamma_{\max},

where :math:`p` is the power-law index, :math:`N` is the number (density), :math:`\gamma_{\min}`
and :math:`\gamma_{\max}` define the support of the distribution, and :math:`N_0` is a normalization constant.
Outside this range, the distribution is assumed to vanish.

.. note::

    In some instances, authors refer instead to the energy distribution of electrons:

    .. math::

        \frac{dN}{dE} = N_{E,0} E^{-p}.

    Triceratops adopts the **Lorentz factor** formulation as the **canonical standard**; however, both are
    implemented in the relevant API (see :mod:`radiation.synchrotron.microphysics`).

.. note::

    :math:`N` may be either the total number of electrons or the number density of electrons, depending on context.
    Triceratops generally works with number densities to facilitate coupling with hydrodynamical quantities. The
    conversion between the two is just a division by the relevant volume.

Equipartition for Power-Law Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

    The relevant API in Triceratops is in the :mod:`radiation.synchrotron.microphysics` module. See
    :ref:`synchrotron_microphysics` for details on use.

Assuming a power-law distribution of electrons, we can derive normalization of the distribution
from our closure relationship. Here, we describe the procedure when assuming equipartition with some
choice of :math:`\epsilon_e` and :math:`\epsilon_B`.

Given a thermal energy density :math:`U_{\rm thermal}`, the energy density in relativistic electrons is (by
equipartition)

.. math::

    U_e = \epsilon_e U_{\rm thermal}.

The total energy density of the electron population is

.. tab-set::

    .. tab-item:: In Terms of Lorentz Factor

        .. math::

            U_e = m_e c^2 \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma \frac{dN}{d\gamma} d\gamma
            = N_0 m_e c^2 \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^{1 - p} d\gamma.

    .. tab-item:: In Terms of Energy

        .. math::

            U_e = \int_{E_{\min}}^{E_{\max}} E \frac{dN}{dE} dE = N_{E,0} \int_{E_{\min}}^{E_{\max}} E^{1 - p} dE.

Letting

.. math::

    M^{(\ell)}_{\gamma} = N_0 \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^{\ell} \gamma^{-p} d\gamma,

be the :math:`\ell`-th moment of the Lorentz factor distribution, we can write the energy density as

.. math::

    \boxed{
    U_e = m_e c^2 N_0 M^{(1)}_{\gamma}.
    }

Solving for :math:`N_0`, we find

.. math::

    \boxed{
    N_0 = \frac{\epsilon_e U_{\rm thermal}}{m_e c^2 M^{(1)}_{\gamma}}.
    }

In terms of the magnetic field, we also have

.. math::

    B = \sqrt{8 \pi \epsilon_B U_{\rm thermal}} \implies N_0 = \frac{\epsilon_e B^2}{8 \pi \epsilon_B m_e c^2 M^{(1)}_{\gamma}}.

.. important::

    This is the canonical way to convert dynamics into synchrotron emission in Triceratops when assuming
    equipartition and a power-law distribution of electrons. See :mod:`radiation.synchrotron.microphysics`
    for the relevant API.

Another useful computation which is made possible with equipartition is the **total emitted power** from synchrotron.
From the Larmor formula, we previously derived that the total power of a **single electron** is

.. math::

    P_{\rm synch} = \frac{4}{3} \sigma_T c \gamma^2 \beta^2 U_B.

Thus, the total power from a **population** of electrons in the relativistic limit (:math:`\beta \approx 1`) is

.. math::

    P_{\rm total} = \frac{4}{3} \sigma_T c U_B \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^2 \frac{dN}{d\gamma} d\gamma
    = \frac{4}{3} \sigma_T c U_B N_0 M^{(2)}_{\gamma}.

From equipartition, we know :math:`N_0` in terms of :math:`U_{\rm thermal}` and :math:`\epsilon_e`, :math:`\epsilon_B`, so we can write

.. math::

    \boxed{
    P_{\rm total} = \frac{4}{3} \sigma_T c \frac{\epsilon_e \epsilon_B u_{\rm therm}^2}{m_e c^2} \left(
        \frac{M^{(2)}_{\gamma}}{M^{(1)}_{\gamma}}
    \right).
    }

.. note::

    One frequently uses the fact that :math:`N_{\rm eff} = N_0 M^{(2)}_{\gamma}` is the effective number of
    radiating electrons in the population (see :ref:`synchrotron_seds` for details). This allows us to write

    .. math::

        P_{\rm total} = \frac{4}{3} \sigma_T c U_B N_{\rm eff}.

.. _electron_cooling:
Cooling of Electrons
^^^^^^^^^^^^^^^^^^^^

.. hint::

    For a much more detailed discussion of electron cooling, including all of the relevant theory which
    is applied in Triceratops, see :ref:`synchrotron_cooling_theory`.

As a population, relativistic electrons will cool and thereby evolve away from their injected energy distribution.
This evolution can imprint additional spectral breaks and modify the normalization of the synchrotron emission.

In many applications, only electrons above a critical Lorentz factor :math:`\gamma_c` cool efficiently within the
relevant timescale (often the dynamical time). A common phenomenological approximation is:

- for :math:`\gamma>\gamma_c`, electrons are in the cooled steady-state regime, :math:`P\propto \gamma^{-(p+k-1)}`;
- for :math:`\gamma<\gamma_c`, cooling is negligible and the distribution retains the injection slope, :math:`P\propto\gamma^{-p}`.

Imposing continuity at :math:`\gamma=\gamma_c` yields the broken power-law form

.. math::

    \boxed{
    P_{\rm SS}(\gamma)
    =
    \begin{cases}
        \dfrac{Q_0}{\Lambda\,(p-1)}\,\gamma^{-(p+k-1)}, & \gamma > \gamma_c, \\[10pt]
        \dfrac{Q_0}{\Lambda\,(p-1)}\,\gamma_c^{-(k-1)}\,\gamma^{-p}, & \gamma < \gamma_c.
    \end{cases}
    }

Synchrotron Cooling
~~~~~~~~~~~~~~~~~~~

The most obvious source of cooling from relativistic electrons in a synchrotron-emitting plasma is the synchrotron
radiation itself. The power radiated by a single electron due to synchrotron emission is

.. math::

    P_{\rm synch} = \frac{4}{3} \sigma_T c \gamma^2 \beta^2 U_B,

where :math:`\sigma_T` is the Thomson cross section, :math:`c` is the speed of light, :math:`\gamma` is the
Lorentz factor of the electron, :math:`\beta` is the dimensionless velocity (:math:`\beta = v/c`),
and :math:`U_B` is the magnetic energy density. In the relativistic limit (:math:`\beta \approx 1`), this simplifies to

.. math::

    P_{\rm synch} \approx \frac{4}{3} \sigma_T c \gamma^2 U_B.

.. hint::

    Since :math:`P \propto \gamma^2`, the resulting steady state distribution steepens by one power in the
    synchrotron-cooled regime (see previous section).

In this case, the cooling timescale is

.. math::

    \tau_{\rm synch, cool} = \frac{\gamma m_e c^2}{P_{\rm synch}} = \frac{3 m_e c}{4 \sigma_T U_B \gamma}.

If we then define a dynamical timescale :math:`t_{\rm dyn}`, we can solve for the cooling Lorentz factor:

.. math::

    \boxed{
    \gamma_c = \frac{3 m_e c}{4 \sigma_T U_B t_{\rm dyn}} = \frac{6 \pi m_e c}{\sigma_T B^2 t_{\rm dyn}}.
    }

The corresponding synchrotron frequency is

.. math::

    \nu_c = \frac{18 \pi m_e c e}{\sigma_T^2 B^3 t_{\rm dyn}^2}.

.. hint::

    This is implemented in the :mod:`radiation.synchrotron.frequencies` module!


IC Cooling
~~~~~~~~~~

In addition to synchrotron radiation, relativistic electrons may also lose energy through
**inverse Compton (IC) scattering**, in which electrons transfer energy to ambient photon fields.
This process is particularly important in environments with strong radiation backgrounds,
such as dense star-forming regions, AGN environments, or compact transients.

In the **Thomson regime**, where the photon energy in the electron rest frame satisfies
:math:`\gamma h\nu \ll m_e c^2`, the power radiated by a single electron due to inverse Compton
scattering is

.. math::

    P_{\rm IC}
    =
    \frac{4}{3}\,\sigma_T c\,\gamma^2 \beta^2 U_{\rm rad},

where :math:`U_{\rm rad}` is the energy density of the target photon field. In the relativistic
limit (:math:`\beta \approx 1`), this simplifies to

.. math::

    P_{\rm IC}
    \approx
    \frac{4}{3}\,\sigma_T c\,\gamma^2 U_{\rm rad}.

This expression is formally identical to the synchrotron power, with the magnetic energy
density :math:`U_B` replaced by the radiation energy density :math:`U_{\rm rad}`. As a result,
inverse Compton cooling produces the same qualitative effect on the electron distribution as
synchrotron cooling.

The corresponding inverse Compton cooling timescale is

.. math::

    \tau_{\rm IC, cool}
    =
    \frac{\gamma m_e c^2}{P_{\rm IC}}
    =
    \frac{3 m_e c}{4 \sigma_T U_{\rm rad}\,\gamma}.

Defining a dynamical timescale :math:`t_{\rm dyn}`, we may introduce the **inverse Compton cooling
Lorentz factor**

.. math::

    \boxed{
    \gamma_{c,{\rm IC}}
    =
    \frac{3 m_e c}{4 \sigma_T U_{\rm rad}\,t_{\rm dyn}}.
    }

Electrons with :math:`\gamma > \gamma_{c,{\rm IC}}` cool efficiently via inverse Compton scattering
within a dynamical time, while lower-energy electrons remain effectively uncooled.

.. note::

    In many astrophysical environments, electrons cool through **both** synchrotron and inverse
    Compton losses. In this case, the total cooling rate is additive:

    .. math::

        P_{\rm tot}
        =
        P_{\rm synch} + P_{\rm IC}
        =
        \frac{4}{3}\,\sigma_T c\,\gamma^2\left(U_B + U_{\rm rad}\right).

    The effective cooling Lorentz factor is therefore

    .. math::

        \gamma_c
        =
        \frac{3 m_e c}{4 \sigma_T \left(U_B + U_{\rm rad}\right) t_{\rm dyn}}.

    This regime is often parameterized by the **Compton \(Y\)-parameter**,
    :math:`Y \equiv U_{\rm rad}/U_B`, which quantifies the relative importance of IC cooling
    compared to synchrotron cooling.

.. warning::

    At sufficiently high electron energies, inverse Compton scattering enters the
    **Klein–Nishina regime**, where the scattering cross section is reduced and the
    :math:`\gamma^2` scaling of the cooling rate breaks down. Triceratops currently assumes
    Thomson-regime IC cooling; Klein–Nishina corrections are not yet implemented.

.. hint::

    This is implemented in the :mod:`radiation.synchrotron.frequencies` module! See
    :func:`~radiation.synchrotron.frequencies.compute_IC_cooling_gamma`,
    :func:`~radiation.synchrotron.frequencies.compute_IC_cooling_frequency`, and the
    associated low-level API.

Absorption Processes in Synchrotron Radiation
---------------------------------------------

It is not always the case that radiation observed by an observer is simply the emitted synchrotron radiation; it may
be modified by absorption processes in the emitting plasma or along the line of sight. Two important absorption
processes that can affect synchrotron radiation are synchrotron self-absorption (SSA) and free-free absorption.

In this section, we'll discuss both processes and derive the critical results.

.. _ssa:
Synchrotron Self-Absorption
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    The presentation given here ignores the effects of extreme electron cooling on SSA. A theory note
    on this particular phenomenon is presented in :ref:`stratified_absorption`.

At sufficiently low frequencies, synchrotron photons emitted by a population of relativistic
electrons may be re-absorbed by the *same* electrons that produced them. This process is known as
**synchrotron self-absorption (SSA)** and leads to a suppression of the emergent radiation below a
characteristic frequency :math:`\nu_a`.

Unlike free–free absorption, SSA is an intrinsically **non-thermal** process: the absorbing
particles are the relativistic electrons themselves, and the absorption is governed by the same
microphysics that produces the synchrotron emission.

For a population of electrons with differential distribution
:math:`N(\gamma) \equiv dN/d\gamma`, the synchrotron emissivity (power emitted per unit volume per
unit frequency per unit solid angle) is

.. math::

    j_\nu
    =
    \frac{1}{4\pi}
    \int_{\gamma_{\min}}^{\gamma_{\max}}
    N(\gamma)\,P(\nu,\gamma)\,d\gamma,

where :math:`P(\nu,\gamma)` is the single-electron synchrotron power derived previously.

The same population also absorbs synchrotron photons. Defining
:math:`B(\gamma,\nu)` as the absorption probability per unit time for an electron of Lorentz
factor :math:`\gamma` interacting with radiation of frequency :math:`\nu`, the absorption
coefficient is

.. math::

    \alpha_\nu
    =
    \int_{\gamma_{\min}}^{\gamma_{\max}}
    N(\gamma)\,B(\gamma,\nu)\,d\gamma.

At first glance, :math:`B(\gamma,\nu)` appears to be an independent and complicated quantity.
However, it may be related directly to the emissivity using `**detailed balance** <https://en.wikipedia.org/wiki/Detailed_balance>`__.
For synchrotron radiation, electrons occupy a continuum of energies rather than discrete levels.
Nevertheless, the logic of Einstein coefficients still applies if we treat each infinitesimal
energy interval as an effective “state.”

An electron absorbing a photon of frequency :math:`\nu` undergoes a transition

.. math::

    \gamma
    \;\longrightarrow\;
    \gamma + \frac{h\nu}{m_e c^2}.

Detailed balance requires that, in thermodynamic equilibrium, the rates of absorption and
stimulated emission balance the spontaneous emission. This condition allows the absorption
coefficient to be written *entirely* in terms of the synchrotron emission power
(see :footcite:t:`RybickiLightman`, Chapter 6):

.. math::

    \boxed{
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
    d\gamma
    }.

This expression is exact and holds for *any* electron distribution :math:`N(\gamma)`. It is the
fundamental result underlying synchrotron self-absorption.

.. important::

    The derivative term reflects the fact that absorption depends on the **gradient of the
    electron distribution in energy space**, not simply on the number of electrons present.
    Physically, absorption is efficient only when lower-energy states are more populated than
    higher-energy states, ensuring a net upward transition rate.

The SSA Spectrum
^^^^^^^^^^^^^^^^

From radiative transfer theory, the specific intensity emerging from a homogeneous synchrotron
source of path length :math:`L` is

.. math::

    I_\nu(\tau) = I_\nu(0) e^{-\tau_\nu} + \int_0^\tau S_\nu(\xi) e^{-\tau-\xi} \;d\xi.

Since it is absorption and subsequent re-emission by the same electrons that produce the SSA emission, we can
take the initial intensity to be zero and, assuming a constant source function :math:`S_\nu = j_\nu/\alpha_\nu`, we find

.. math::

    I_\nu = S_\nu \left(1-e^{-\tau_\nu}\right).

Thus, if one can successfully compute the absorption coefficient :math:`\alpha_\nu`, one can compute the emergent intensity
from a synchrotron source including SSA effects.

It is often useful to define the **SSA frequency** :math:`\nu_a` as the frequency at which the optical depth
:math:`\tau_\nu = \alpha_\nu \ell` equals unity:

.. math::

    \tau_{\nu_a} = 1 \implies \alpha_{\nu_a} \ell = 1.

In doing so, the optical depth to SSA then becomes

.. math::

    \tau_\nu = \frac{\alpha_\nu}{\alpha_{\nu_a}}.

SSA In Power-Law Electron Distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's now specialize to the case of a power-law distribution of electrons. Assume that

.. math::

    \frac{dN}{d\gamma} = N_0 \gamma^{-p}, \qquad \gamma_{\min} \le \gamma \le \gamma_{\max}.

We know, therefore, that

.. math::

    \gamma^2 \frac{\partial}{\partial\gamma}\left[\frac{1}{\gamma^2} \frac{dN}{d\gamma}\right]
    =
    - (p+2) N_0 \gamma^{-(p+1)},

so the absorption coefficient becomes

.. math::

    \alpha_\nu = \frac{(p+2) N_0}{8\pi m_e \nu^2} \int_{\gamma_{\min}}^{\gamma_{\max}} P(\nu,\gamma) \gamma^{-(p+1)} d\gamma.

We have already computed integrals of this form before. Recall that

.. math::

    \int_{\gamma_{\min}}^{\gamma_{\max}} P(\nu,\gamma) N_0 \gamma^{-s} d\gamma = 4\pi c_5(s) N_0 \left(m_e c^2\right)^{s-1}
    (B\sin\alpha)^{(s+1)/2} \left(\frac{\nu}{2 c_1}\right)^{-(s-1)/2}.

We therefore introduce another constant, :math:`c_6(p)` to capture the relevant factors:

.. math::

    c_6(p) = \sqrt{3} \frac{\pi}{72} em_e^5 c^{10} \left(p+\frac{10}{3}\right) \Gamma\left(\frac{3p+2}{12}\right) \Gamma\left(\frac{3p+10}{12}\right).

With this, the absorption coefficient for a power-law distribution of electrons is

.. tab-set::

    .. tab-item:: Lorentz Factor Distribution

        .. math::

            \alpha_\nu
            =
            c_6(p)\,N_0\,(m_e c^2)^{p-1}
            (B\sin\alpha)^{(p+2)/2}
            \left(\frac{\nu}{2 c_1}\right)^{-(p+4)/2},

    .. tab-item:: Energy Distribution

        .. math::

            \alpha_\nu
            =
            c_6(p)\,N_{0,E}
            (B\sin\alpha)^{(p+2)/2}
            \left(\frac{\nu}{2 c_1}\right)^{-(p+4)/2}.

.. dropdown:: Aside: What electrons contribute to SSA?

    A common source of confusion in synchrotron self-absorption theory is the question of *which electrons*
    actually contribute to the absorption coefficient at a given frequency. It is important to distinguish
    between (i) the **frequency scaling** of :math:`\alpha_\nu` and (ii) the **support of the integral**
    defining :math:`\alpha_\nu` in Lorentz factor.

    Starting from the exact expression,

    .. math::
        :label: eq:ssa_exact

        \alpha_\nu
        =
        -\frac{1}{8\pi m_e \nu^2}
        \int d\gamma \;
        P(\nu,\gamma)\,
        \gamma^2
        \frac{\partial}{\partial\gamma}
        \left[
            \frac{1}{\gamma^2}
            \frac{dN}{d\gamma}
        \right],

    the contribution of electrons at Lorentz factor :math:`\gamma` is controlled entirely by the
    single–electron synchrotron kernel :math:`P(\nu,\gamma)`.

    For a fixed observing frequency :math:`\nu`, define :math:`\gamma_\nu` implicitly by

    .. math::

        \nu = \nu_c(\gamma_\nu).

    The synchrotron kernel has two qualitatively distinct regimes:

    1. **High-frequency regime** (:math:`\nu \gg \nu_c(\gamma)`):

       In this case the kernel is exponentially suppressed,

       .. math::

           P(\nu,\gamma) \propto
           \exp\!\left[-\frac{\nu}{\nu_c(\gamma)}\right],

       and electrons with :math:`\gamma < \gamma_\nu` contribute negligibly to the integral,
       regardless of their number density or the shape of :math:`dN/d\gamma`.
       No physically reasonable electron distribution can overcome this exponential cutoff.

    2. **Low-frequency regime** (:math:`\nu \ll \nu_c(\gamma)`):

       In this regime the kernel follows a power law,

       .. math::

           P(\nu,\gamma) \propto \nu^{1/3}\,\gamma^{-2/3},

       so electrons with :math:`\gamma > \gamma_\nu` contribute efficiently to absorption.

    As a result, the *support* of the absorption integral is confined to

    .. math::

        \gamma \gtrsim \gamma_\nu,

    while electrons with :math:`\gamma \ll \gamma_\nu` are exponentially ineffective absorbers at
    frequency :math:`\nu`, even if they dominate the total number density.

    Within the contributing region :math:`\gamma > \gamma_\nu`, both the synchrotron kernel
    :math:`P(\nu,\gamma)` and physically relevant electron distributions :math:`dN/d\gamma`
    decrease with increasing :math:`\gamma`. Consequently, the dominant contribution to
    :math:`\alpha_\nu` arises from electrons near the lower boundary of this region,

    .. math::

        \gamma \sim \gamma_\nu,
        \qquad
        \nu \sim \nu_m(\gamma).

    This is the precise sense in which synchrotron self-absorption at frequency :math:`\nu`
    is said to be “dominated” by electrons satisfying :math:`\nu \sim \nu_m(\gamma)`. The statement
    refers to the *support of the absorption integral*, not to the emission spectrum of those electrons.

Recalling that the emissivity for a power-law distribution of electrons is

.. math::

        j_\nu = c_5(p)\,N_{0,E}
        \left(B\sin\alpha\right)^{(p+1)/2}
        \left(\frac{\nu}{2c_1}\right)^{-(p-1)/2}.

We have the all-important source function

.. math::

    S_\nu = \frac{j_\nu}{\alpha_\nu} =
    \frac{c_5(p)}{c_6(p)} (B\sin\alpha)^{-1/2} \left(\frac{\nu}{2 c_1}\right)^{5/2}.

From our discussion in the previous section, we know that

.. math::

    \tau_\nu = \frac{\alpha_\nu}{\alpha_{\nu_a}} = \left(\frac{\nu}{\nu_a}\right)^{-(p+4)/2}.

And therefore, the specific intensity including SSA effects is (e.g. :footcite:t:`demarchiRadioAnalysisSN2004C2022`)

.. math::

    I_\nu = S_\nu(\nu_a) J_\nu(y, p),

where

.. math::

    J_\nu(y,p) = y^{5/2} \left(1 - e^{-y^{-(p+4)/2}}\right),
    \qquad
    y = \frac{\nu}{\nu_a}.

Assuming some known path length :math:`\ell`, one can compute :math:`\nu_a` from the condition :math:`\alpha_{\nu_a} \ell = 1`.
This implies that

.. tab-set::

    .. tab-item:: Lorentz Factor Distribution

        .. math::

            \nu_a = 2c_1 \left(c_6 \ell\right)^{2/(p+4)} \left(N_0 (m_e c^2)^{p-1}\right)^{2/(p+4)} (B\sin\alpha)^{(p+2)/(p+4)}.

    .. tab-item:: Energy Distribution

        .. math::

            \nu_a = 2c_1 \left(c_6 \ell\right)^{2/(p+4)} \left(N_{0,E}\right)^{2/(p+4)} (B\sin\alpha)^{(p+2)/(p+4)}.

The details of how :math:`\ell` should be determined are a matter of dynamics and not of radiation.

.. note::

    It is worth emphasizing that the SSA frequency depends on the properties of the electron population. Thus, for
    any set of assumptions about the behavior of the underlying electrons, the corresponding SSA frequency may be quite
    different.

Approximate Methods
~~~~~~~~~~~~~~~~~~~

While the above treatment of the SSA frequency is precise, it is complicated by the fact that one must specify
the detailed electron distribution (including any stratification due to radiative cooling) in order to evaluate
:math:`\tau_\nu = \int \alpha_\nu\,d\ell`.  A widely used alternative is therefore to determine the self-absorption
frequency :math:`\nu_a` by matching the asymptotic optically thin and optically thick fluxes across the turnover
(e.g. CITATIONS).

The key observation is that at frequencies where the source is optically thick (:math:`\tau_\nu \gg 1`), the
emergent intensity approaches the local synchrotron source function,

.. math::

    I_\nu \approx S_\nu \equiv \frac{j_\nu}{\alpha_\nu},

evaluated at the depth where :math:`\tau_\nu \sim 1`.  At a fixed observing frequency :math:`\nu`, synchrotron
emission and absorption are dominated by electrons in a narrow range of Lorentz factors.  Motivated by this,
we first consider a mono-energetic electron population with Lorentz factor :math:`\gamma_0` and number density
:math:`n_e`, i.e.

.. math::

    N(\gamma) = n_e\,\delta(\gamma-\gamma_0).

For an isotropic distribution, the synchrotron emissivity and absorption coefficient may be written (RL / CITATION)

.. math::

    j_\nu = \frac{1}{4\pi}\int d\gamma\,N(\gamma)\,P_\nu(\gamma),

.. math::

    \alpha_\nu = -\frac{1}{8\pi m_e \nu^2}\int d\gamma\,P_\nu(\gamma)\,\gamma^2
    \frac{\partial}{\partial\gamma}\left[\frac{N(\gamma)}{\gamma^2}\right],

where :math:`P_\nu(\gamma)` is the single-electron synchrotron power per unit frequency.  Substituting the
mono-energetic form yields

.. math::

    j_\nu = \frac{n_e}{4\pi}\,P_\nu(\gamma_0),

and

.. math::

    \alpha_\nu
    =
    \frac{n_e}{8\pi m_e \nu^2}
    \left[
      \frac{\partial P_\nu}{\partial\gamma}
      +\frac{2}{\gamma}P_\nu
    \right]_{\gamma=\gamma_0}.

The corresponding source function is therefore

.. math::

    S_\nu \equiv \frac{j_\nu}{\alpha_\nu}
    =
    2 m_e \nu^2\,
    \frac{P_\nu(\gamma_0)}
    {\left[\frac{\partial P_\nu}{\partial\gamma}+\frac{2}{\gamma}P_\nu\right]_{\gamma=\gamma_0}}.

For a smooth synchrotron kernel :math:`P_\nu(\gamma)`, the combination in the denominator varies on a fractional
Lorentz-factor scale :math:`\Delta\gamma/\gamma \sim \mathcal{O}(1)`, implying
:math:`\partial P_\nu/\partial\gamma \sim P_\nu/\gamma` up to factors of order unity.  Hence the ratio above is
:math:`\sim \gamma_0` (again up to factors of order unity), and we may write the optically thick source function as

.. math::

    S_\nu(\gamma_0) \sim \frac{2C_s}{c^2} \nu^2 \gamma_0 m_e c^2,

where :math:`C_S = \mathcal{O}(1)` absorbs kernel- and pitch-angle-dependent constants.  This is simply the
Rayleigh--Jeans form with an effective (brightness) temperature :math:`kT_{\rm eff}\sim \gamma_0 m_e c^2`.

If the emitting region subtends projected area :math:`A` on the sky, the optically thick flux at :math:`\nu`
is then

.. math::

    F_\nu^{\rm (thick)}(\nu)
    \approx
    \frac{A}{D^2}\,S_\nu
    \sim
    \frac{2\nu^2}{c^2}\gamma_0 m_e c^2 \frac{A}{D^2},

with :math:`C_{\rm thick}=\mathcal{O}(1)`.

To apply this to synchrotron self-absorption, we identify :math:`\gamma_0` with the characteristic electron
Lorentz factor dominating emission and absorption at :math:`\nu=\nu_a` (denoted :math:`\gamma_a`), so that
:math:`F_\nu^{\rm (thick)}(\nu_a)` is determined by :math:`S_\nu(\gamma_a)`.  The self-absorption frequency
may then be obtained by matching the optically thick and optically thin asymptotic fluxes at the turnover,

.. math::

    F_\nu^{\rm (thick)}(\nu_a) \approx F_\nu^{\rm (thin)}(\nu_a),

where :math:`F_\nu^{\rm (thin)}` is computed from the appropriate optically thin spectral segment for the
problem at hand.  This yields an algebraic equation for :math:`\nu_a`.

.. note::

    Given that this approach is only approximate, it is often customary to absorb the various order-unity
    constants into a single fudge factor as described above. This **includes** pitch averaging effects.


References
-----------
.. footbibliography::
