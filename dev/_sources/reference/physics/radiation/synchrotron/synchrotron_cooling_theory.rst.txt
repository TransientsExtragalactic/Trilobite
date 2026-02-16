.. _synchrotron_cooling_theory:
==========================================
Theory of Synchrotron Electron Populations
==========================================

Synchrotron emission from astrophysical sources is determined not only by the
radiative properties of individual electrons, but also by the **energy
distribution of the electron population as a whole**. In realistic environments,
this distribution evolves over time due to injection, radiative losses, and other
processes.

In this section, we develop the general theory governing the evolution of
relativistic electron populations and derive the steady-state distributions most
commonly used in synchrotron modeling.

.. contents::
   :local:
   :depth: 2

Overview
--------

Synchrotron emission observed from astrophysical sources almost always arises
from **ensembles of relativistic electrons**, rather than from isolated
particles. The collective radiative output therefore depends not only on the
microphysics of synchrotron emission, but also on the **statistical properties
and evolution of the electron population**.

In realistic environments, electron populations are **dynamical**. Electrons
are continuously injected by physical processes such as diffusive shock
acceleration, magnetic reconnection, or turbulent heating. Once injected, they
lose energy through radiative processes—most notably synchrotron and inverse
Compton cooling—as well as through adiabatic expansion or escape from the
emitting region. These competing processes reshape the electron distribution
over time, imprinting characteristic breaks and slopes that directly determine
the observed synchrotron spectrum.

A self-consistent description of synchrotron emission therefore requires
tracking the **time evolution of the electron energy distribution**, accounting
for injection, cooling, and transport in energy space. In many astrophysical
situations, this evolution leads to a steady-state distribution whose form is
largely independent of initial conditions and depends primarily on the balance
between injection and energy losses.

In this section, we develop the general framework used to describe evolving
electron populations and derive the steady-state distributions most commonly
assumed in synchrotron modeling. These results provide the foundation for
understanding the spectral regimes, break frequencies, and absorption features
discussed in subsequent sections.

Formalism of Electron Population Evolution
------------------------------------------

Before deriving any specific results, we first establish the fundamental
quantities and assumptions that govern the evolution of relativistic electron
populations.

The central object of interest is the **electron distribution function**
:math:`N(E,t)`, defined such that :math:`N(E,t)\,dE` is the number (or number
density) of electrons with energies in the interval :math:`(E, E+dE)` at time
:math:`t`. Given knowledge of this distribution at all times, the synchrotron
emission from the population may be computed by integrating the single-electron
emissivity over energy:

.. math::

    P(\nu,t)
    =
    \int N(E,t)\,P_\nu(E)\,dE,

where :math:`P_\nu(E)` is the synchrotron power per unit frequency emitted by a
single electron of energy :math:`E`. A detailed discussion of the
single-electron emissivity is provided in :ref:`synchrotron_theory`.

.. note::

    In many astrophysical applications, it is more convenient to work in terms of
    the electron Lorentz factor :math:`\gamma` rather than the energy :math:`E`.
    The two are related by :math:`E = \gamma m_e c^2`, where :math:`m_e` is the
    electron mass and :math:`c` is the speed of light. All results presented here
    can be straightforwardly translated to :math:`\gamma`-space, and we will do so. In
    general, Triceratops defaults to the use of the :math:`\gamma`-space distribution; however,
    for this discussion of theory, we prefer to work in terms of energy for clarity.

In realistic astrophysical environments, the electron distribution evolves
continuously due to a variety of physical processes. These include the injection
of new relativistic electrons, radiative energy losses (e.g. synchrotron and
inverse Compton cooling), adiabatic expansion of the emitting plasma, and escape
from the emission region. To describe this evolution in a unified manner, we
employ a **continuity equation in energy space**.

To construct this equation, consider an infinitesimal interval
:math:`(E, E+dE)` in energy space. At time :math:`t`, the number of electrons
contained in this interval is :math:`N(E,t)\,dE`. The number of electrons within
this domain may change in two distinct ways:

1. **Direct addition or removal of particles**, through physical injection or
   loss processes; and
2. **Transport through energy space**, as electrons gain or lose energy and
   flow across the boundaries of the interval.

The first effect is described by an **injection term**
:math:`Q(E,t)`, defined such that :math:`Q(E,t)\,dE\,dt` is the net number of
electrons injected into (or removed from) the energy interval
:math:`(E,E+dE)` during the time interval :math:`(t,t+dt)`. Injection may arise
from processes such as diffusive shock acceleration, magnetic reconnection, or
pair creation, while negative values of :math:`Q` may represent escape or
catastrophic losses.

The second effect is described by the **energy loss rate**
:math:`\dot{E}(E) \equiv dE/dt`, which specifies how individual electrons move
through energy space. For cooling processes, :math:`\dot{E}(E) < 0`, so electrons
systematically drift toward lower energies, generating a flux of particles
across energy boundaries.

Combining these effects, conservation of particle number in energy space leads
to the continuity equation

.. math::

    \frac{\partial N(E,t)}{\partial t}
    +
    \frac{\partial}{\partial E}
    \left[\dot{E}(E)\,N(E,t)\right]
    =
    Q(E,t).

This equation provides the fundamental framework for modeling the evolution of
relativistic electron populations. In the sections that follow, we will explore
its solutions under various assumptions regarding injection, cooling, and
dynamical timescales, and connect these solutions directly to the resulting
synchrotron spectral energy distributions.

Solutions to the Continuity Equation
------------------------------------

The continuity equation governing the evolution of the electron distribution is
a **one-dimensional advection equation in energy space**, with a source term
representing particle injection. In full generality, it may be solved using the
`method of characteristics <https://en.wikipedia.org/wiki/Method_of_characteristics>`__.

Expanding the energy derivative, the continuity equation may be written as

.. math::

    \frac{\partial N(E,t)}{\partial t}
    +
    \dot{E}(E)\,
    \frac{\partial N(E,t)}{\partial E}
    +
    N(E,t)\,
    \frac{d\dot{E}}{dE}
    =
    Q(E,t).

To eliminate the partial derivatives, we introduce **characteristic curves**
:math:`E(\tau)` in energy–time space, defined such that

.. math::

    \frac{dE}{d\tau} = \dot{E}(E),

where :math:`\tau` is a parameter labeling position along the characteristic.
Physically, these curves correspond to the energy trajectories followed by
individual electrons as they cool or gain energy.

Along a characteristic, the total derivative of the distribution function is

.. math::

    \frac{dN}{d\tau}
    =
    \frac{\partial N}{\partial t}
    +
    \dot{E}(E)\,
    \frac{\partial N}{\partial E}.

Substituting this into the expanded continuity equation yields an ordinary
differential equation along each characteristic:

.. math::

    \frac{dN}{d\tau}
    +
    N(E,\tau)\,
    \frac{d\dot{E}}{dE}
    =
    Q(E,\tau).

This is a first-order linear ODE and may be solved using an integrating factor.
The resulting **general solution** for the electron distribution is

.. math::

    N(E,t)
    =
    \frac{\dot E(E_0)}{\dot E(E)}\,N_0(E_0)
    +
    \frac{1}{\dot E(E)}
    \int_{0}^{t}
    \dot E\!\big(E(t')\big)\,
    Q\!\big(E(t'),t'\big)\,dt',

or, equivalently,

.. math::

    N(E,t)
    =
    \frac{\dot E(E_0)}{\dot E(E)}\,N_0(E_0)
    +
    \frac{1}{\dot E(E)}
    \int_{E}^{E_0}
    Q(E',t')\,dE',

where :math:`t'` is the time at which an electron that has energy :math:`E` at time
:math:`t` possessed energy :math:`E'`, as determined by the characteristic
equation :math:`dE/dt = \dot{E}(E)`.

.. hint::

    While the general solution is formally exact, it is impractical for most
    applications because it requires knowledge of the full time-dependent injection
    function :math:`Q(E,t)` and the initial distribution :math:`N_0`. One then also needs
    to integrate :math:`\dot{E}(E)` to obtain the characteristics :math:`E(t)`.

Steady State Solutions
^^^^^^^^^^^^^^^^^^^^^^

In many cases, we are most interested in the **steady-state solutions** of the cooling equation described above. In
this limit, we have :math:`E_0(E,t) \rightarrow \infty` as :math:`t \rightarrow \infty`, and the general solution's
memory of the initial conditions is lost. The steady-state solutions depend only on the injection function
:math:`Q(E)` and the cooling function :math:`\dot{E}(E)`:

.. math::

    N_{\rm SS}(E) = \frac{1}{\dot{E}} \int_0^t \dot{E}(E',t') Q(E',t') dt',

or, in terms of energy,

.. math::

    N_{\rm SS}(E) = \frac{1}{\dot{E}} \int_E^{\infty} Q(E') dE'.

.. note::

    It is worth recognizing that, when :math:`\dot{E} = 0`, there is no steady-state
    solution. In this case, electrons simply accumulate at each energy, and the distribution
    grows linearly with time.

Steady-State Below the Injection Minimum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One particularly important regime arises when electron are injected only above energy
:math:`E_{\min}`, so that :math:`Q(E) = 0` for :math:`E < E_{\min}`. In this case, we have

.. math::

    N_{\rm SS}(E) = \frac{1}{\dot{E}(E)} \int_{E_{\min}}^{\infty} Q(E') dE' \propto \frac{1}{|\dot{E}(E)|}.

As such, the steady-state distribution below the injection minimum is determined solely by the cooling function. This
is a universal result that holds regardless of the specific form of the injection spectrum.

Power-Law Steady States
~~~~~~~~~~~~~~~~~~~~~~~

A common-place scenario is that of an injection function that is a power law in energy:

.. math::

    Q(E) = Q_0 E^{-p}, \quad E \geq E_{\min}.

Likewise, it is often the case that the cooling function is also a power law:

.. math::

    \dot{E}(E) = -A E^{q}.

The resulting steady-state distribution is then

.. math::

    N_{\rm SS}(E)
    =
    \frac{Q_0}{A (p - q - 1)}\,E^{-(p + 1 - q)},
    \quad
    E \geq E_{\min}.

This shows that power-law injection and cooling functions lead to power-law steady-state distributions, with the
slope determined by both the injection index :math:`p` and the cooling index :math:`q`. As discussed in depth below,
the typical case of synchrotron cooling corresponds to :math:`q = 2`, leading to the well-known result that
the steady-state distribution steepens by one power of energy compared to the injection spectrum.

.. math::

    N_{\rm SS}(E)
    =
    \frac{Q_0}{A (p - 3)}\,E^{-(p - 1)},
    \quad
    E \geq E_{\min}.

Cooling Mechanisms
------------------

The evolution of relativistic electron populations is governed not only by
injection processes, but also by the mechanisms through which electrons lose
energy. These **cooling processes** determine both the shape of the electron
distribution and the locations of spectral breaks in the resulting synchrotron
emission.

In most astrophysical environments relevant to synchrotron radiation, electron
cooling is dominated by **radiative losses**, particularly synchrotron emission
and inverse Compton scattering. In expanding systems, **adiabatic cooling** may
also play an important role. Each of these processes may be described by an
energy loss rate :math:`\dot{E}(E)`, which enters directly into the continuity
equation governing electron population evolution.

Below, we summarize the most important cooling mechanisms and their consequences
for synchrotron-emitting electron populations.

Synchrotron Cooling
^^^^^^^^^^^^^^^^^^^

The most ubiquitous radiative loss mechanism for relativistic electrons in
magnetized plasmas is **synchrotron cooling**. As electrons spiral in magnetic
fields, they continuously radiate energy, causing their Lorentz factors to
decrease over time.

The total synchrotron power radiated by a single electron is

.. math::

    P_{\rm synch}
    =
    \frac{4}{3}\,\sigma_T c\,\gamma^2 \beta^2 U_B,

where :math:`\sigma_T` is the Thomson cross section,
:math:`\gamma` is the electron Lorentz factor,
:math:`\beta = v/c`, and
:math:`U_B = B^2/(8\pi)` is the magnetic energy density.
In the ultra-relativistic limit (:math:`\beta \simeq 1`), this reduces to

.. math::

    P_{\rm synch}
    \approx
    \frac{4}{3}\,\sigma_T c\,\gamma^2 U_B.

Because the emitted power scales as :math:`\gamma^2`, higher-energy electrons
cool more rapidly than lower-energy ones. This energy dependence is responsible
for the appearance of **cooling breaks** in both the electron distribution and
the synchrotron spectrum.

The corresponding synchrotron cooling timescale is

.. math::

    \tau_{\rm synch}
    =
    \frac{\gamma m_e c^2}{P_{\rm synch}}
    =
    \frac{3 m_e c}{4 \sigma_T U_B \gamma}.

Equating this timescale with a characteristic dynamical time
:math:`t_{\rm dyn}` defines the **synchrotron cooling Lorentz factor**

.. math::

    \boxed{
    \gamma_c
    =
    \frac{3 m_e c}{4 \sigma_T U_B t_{\rm dyn}}
    =
    \frac{6\pi m_e c}{\sigma_T B^2 t_{\rm dyn}}.
    }

Electrons with :math:`\gamma > \gamma_c` cool efficiently within a dynamical
time, while those with :math:`\gamma < \gamma_c` remain effectively uncooled.
This distinction naturally leads to broken power-law electron distributions.

The synchrotron frequency associated with electrons at the cooling Lorentz
factor is

.. math::

    \boxed{
    \nu_c
    =
    \frac{3 e B}{4\pi m_e c}\,\gamma_c^2.
    }

.. note::

    These quantities are implemented in
    :mod:`radiation.synchrotron.frequencies`.

Inverse Compton Cooling
^^^^^^^^^^^^^^^^^^^^^^^

Relativistic electrons may also cool via **inverse Compton (IC) scattering** off
ambient photon fields. This process is particularly important in environments
with intense radiation backgrounds, such as star-forming regions, AGN
environments, or compact transients.

In the **Thomson regime**, where the photon energy in the electron rest frame
satisfies :math:`\gamma h\nu \ll m_e c^2`, the inverse Compton power radiated by a
single electron is

.. math::

    P_{\rm IC}
    =
    \frac{4}{3}\,\sigma_T c\,\gamma^2 U_{\rm rad},

where :math:`U_{\rm rad}` is the energy density of the target photon field.
This expression is formally identical to the synchrotron power, with the magnetic
energy density :math:`U_B` replaced by :math:`U_{\rm rad}`.

The corresponding inverse Compton cooling timescale is

.. math::

    \tau_{\rm IC}
    =
    \frac{3 m_e c}{4 \sigma_T U_{\rm rad}\,\gamma}.

When both synchrotron and inverse Compton losses are present, the total radiative
cooling rate is additive:

.. math::

    P_{\rm tot}
    =
    \frac{4}{3}\,\sigma_T c\,\gamma^2
    \left(U_B + U_{\rm rad}\right).

It is common to define the **Compton \(Y\)-parameter**

.. math::

    Y \equiv \frac{U_{\rm rad}}{U_B},

which measures the relative importance of inverse Compton cooling compared to
synchrotron cooling.

.. warning::

    At sufficiently high electron energies, inverse Compton scattering enters
    the **Klein--Nishina regime**, where the scattering cross section is reduced
    and the :math:`\gamma^2` scaling of the cooling rate breaks down.
    Triceratops currently assumes Thomson-regime IC cooling.

Adiabatic Cooling
^^^^^^^^^^^^^^^^^

In expanding systems—such as supernova remnants, blast waves, or relativistic
outflows—electrons may also lose energy through **adiabatic expansion** of the
emitting plasma. Unlike radiative losses, adiabatic cooling does not produce
radiation; instead, electron energies decrease because the plasma performs
macroscopic work as it expands.

In a fully ionized plasma, electrons are **not free-streaming** on macroscopic
scales. Charge neutrality and electromagnetic forces couple electrons to the
bulk flow (typically dominated dynamically by the ions and fields), so that the
electron population is advected with an expanding fluid element. Adiabatic
cooling of the electrons is therefore a kinematic consequence of expansion: as
the comoving volume grows, the electron momenta decrease, and hence so do their
energies.

For an isotropic ultra-relativistic particle distribution, the
pressure is related to the internal energy density :math:`u` by

.. math::

    P = \frac{1}{3}\,u,

so that the adiabatic index is :math:`\gamma_{\rm ad}=4/3`. For an adiabatic
process, the first law gives

.. math::

    dU = -P\,dV,

with :math:`U=uV`. Differentiating,

.. math::

    d(uV) = -\frac{1}{3}u\,dV
    \quad\Rightarrow\quad
    V\,du = -\frac{4}{3}u\,dV,

so

.. math::

    \frac{du}{u} = -\frac{4}{3}\frac{dV}{V}
    \quad\Rightarrow\quad
    u \propto V^{-4/3}.

For a fixed number of particles :math:`N` in the comoving element, the mean
particle energy scales as

.. math::

    \langle E\rangle \sim \frac{uV}{N} \propto V^{-1/3},

which implies

.. math::

    \dot E_{\rm ad} = -\frac{1}{3}\frac{\dot V}{V}\,E.

This contribution can be added to the total energy loss rate in the continuity
equation.

.. math::

    \dot E_{\rm ad}(t) = -\frac{1}{3}\frac{\dot V}{V}\,E
    \;=\;
    -\frac{1}{3}(\nabla\cdot \mathbf{u})\,E.

.. important::

    Because :math:`\dot E_{\rm ad}\propto -E`, adiabatic cooling preserves the slope
    of power-law electron distributions, in contrast to radiative losses such as
    synchrotron and inverse Compton cooling for which :math:`\dot E\propto -E^2`.

In systems where both effects are important, a commonly used effective loss law
is

.. math::

    \dot E(E,t) = -A(t)\,E^2 - \frac{1}{3}\,(\nabla\cdot \mathbf{u})\,E,

which captures radiative steepening at high energies and adiabatic evolution of
the overall normalization.

.. important::

    Although the bulk expansion is typically driven dynamically by the ions and
    magnetic fields, adiabatic cooling applies to the relativistic electrons
    because they are advected with the expanding flow and their momenta redshift
    accordingly. The electron population need not dominate the pressure for
    adiabatic losses to operate.


Summary of Cooling Mechanisms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The dominant cooling mechanisms encountered in synchrotron-emitting systems are:

- **Synchrotron cooling**: :math:`\dot{E} \propto -E^2`, steepens electron
  distributions above the cooling break.
- **Inverse Compton cooling**: also :math:`\dot{E} \propto -E^2` in the Thomson
  regime, enhancing radiative losses by a factor :math:`1+Y`.
- **Adiabatic cooling**: :math:`\dot{E} \propto -E`, reduces electron energies
  without altering spectral slopes.

The competition between these processes determines the structure of the electron
population and, ultimately, the observed synchrotron spectral energy
distribution.

.. _equipart_cooling_theory:
Equipartition for Cooled Electron Populations
---------------------------------------------

As described in :ref:`synchrotron_theory`, it is generally not possible to get a direct constraint on
the structure of the electron population from synchrotron observations alone. Instead, one typically
relies on **equipartition assumptions** to relate the energy density in relativistic electrons to
that in the magnetic field. This approach is also used when modeling the synchrotron emission from cooled
electron populations.

.. hint::

    For background on the equipartition formalism, see the relevant section in :ref:`synchrotron_theory`.

The equipartition analysis of cooled electron populations roughly follows that of the uncooled
case, with some important modifications to account for the altered electron distribution.

We begin by considering a **fast-cooling** electron population, where the cooling Lorentz factor
:math:`\gamma_c` lies below the injection minimum :math:`\gamma_m`. In this regime, all injected electrons cool
efficiently. For notational convenience, we define the **normalization parameter** :math:`K_{\rm 0}` such that

.. math::

    N(\gamma)
    =
    K_{\rm 0}
    \begin{cases}
    \left(\frac{\gamma}{\gamma_m}\right)^{-2}, & \gamma_c \leq \gamma < \gamma_m, \\
    \left(\frac{\gamma}{\gamma_m}\right)^{-(p+1)}, & \gamma_m \leq \gamma < \gamma_{\max},\\
    0 & \text{elsewhere}.
    \end{cases}

.. hint::

    Note that this convention **differs** from that used for the uncooled electron distribution in which we used
    :math:`N = N_0 \gamma^{-p}` above :math:`\gamma_m`. Here, to allow for easier manipulation of the broken-power-law,
    we normalize to :math:`\gamma_m` and include the appropriate scaling in each segment.

As in the case of uncooled electrons, we can also write this distribution in terms of the **energy**:

.. math::

    N(E)
    =
    K_{\rm E,0}
    \begin{cases}
    \left(\frac{E}{E_m}\right)^{-2}, & E_c \leq E < E_m, \\
    \left(\frac{E}{E_m}\right)^{-(p+1)}, & E_m \leq E < E_{\max},\\
    0 & \text{elsewhere}.
    \end{cases}

For a **slow-cooling** population, where :math:`\gamma_m < \gamma_c`, only the highest-energy electrons cool
efficiently. The distribution is then

.. math::

    N(\gamma)
    =
    H_{\rm 0}
    \begin{cases}
    \left(\frac{\gamma}{\gamma_c}\right)^{-p}, & \gamma_m \leq \gamma < \gamma_c, \\
    \left(\frac{\gamma}{\gamma_c}\right)^{-(p+1)}, & \gamma_c \leq \gamma < \gamma_{\max},\\
    0 & \text{elsewhere}.
    \end{cases}

or, in terms of energy,

.. math::

    N(E)
    =
    H_{\rm E,0}
    \begin{cases}
    \left(\frac{E}{E_c}\right)^{-p}, & E_m \leq E < E_c, \\
    \left(\frac{E}{E_c}\right)^{-(p+1)}, & E_c \leq E < E_{\max},\\
    0 & \text{elsewhere}.
    \end{cases}


Following our approach in :ref:`synchrotron_theory`, we can compute moments of the distribution as

.. math::

    M_\gamma^{(k)} = K_{\rm 0} \int_{\gamma_c}^{\gamma_{\max}} \gamma^k N(\gamma) d\gamma,

or, in terms of energy,

.. math::

    M_E^{(k)} = K_{\rm E,0} \int_{E_c}^{E_{\max}} E^k N(E) dE.

The equipartition analysis then proceeds in exactly the same manner as described in :ref:`synchrotron_theory`, with
the modified electron distribution used to compute the the relevant moments and corresponding energy densities.



Computational Implementation
-----------------------------

.. important::

    Not Yet Implemented.

References
----------
.. footbibliography::
