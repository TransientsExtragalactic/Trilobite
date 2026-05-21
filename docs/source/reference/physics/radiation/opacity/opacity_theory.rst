.. _opacity_theory:

=================
Opacity Theory
=================

Opacity quantifies how strongly a medium attenuates radiation.  In Trilobite it appears in a
variety of physical contexts: as the Rosseland mean opacity driving radiative diffusion in
optically thick accretion disks, as a photon-absorption cross-section in spectral-transfer
calculations, as the column-averaged opacity entering free-free optical-depth estimates for
supernova CSM, and as a grey effective opacity in single-zone stellar-wind models.  The
:mod:`trilobite.radiation.opacity` module is deliberately **context-agnostic**: it provides a
uniform evaluation interface so that the same opacity object can be dropped into any solver that
needs :math:`\kappa(\rho, T)`, its logarithmic derivatives, or both.

This document reviews the physical basis of the opacity laws currently implemented in Trilobite
and derives the power-law scalings and partial derivatives that enter iterative thermal solvers.

For usage instructions see :ref:`opacity_user_guide`.  For a guide to implementing new opacity
laws see :ref:`opacity_dev_guide`.

.. contents::
    :local:
    :depth: 2

----

Definitions and Context
-----------------------

The opacity :math:`\kappa` (:math:`\mathrm{cm^2\,g^{-1}}`) is defined as the attenuation
cross-section *per unit mass* of the medium.  For a plasma of density :math:`\rho`, the
monochromatic absorption coefficient is

.. math::

    \alpha_\nu = \kappa_\nu\,\rho,

so the optical depth across a column of surface density :math:`\Sigma` is
:math:`\tau_\nu = \kappa_\nu\,\Sigma`.

Depending on the physical context, different frequency-averaged opacities are appropriate:

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Mean
      - When to use
    * - **Rosseland mean** :math:`\kappa_R`
      - Radiative diffusion in optically thick media (accretion disk interiors,
        stellar envelopes).  Harmonic average over :math:`\partial B_\nu/\partial T`.
    * - **Planck mean** :math:`\kappa_P`
      - Emission/absorption in optically thin media.  Arithmetic average over :math:`B_\nu`.
    * - **Grey approximation** :math:`\kappa`
      - Single effective opacity independent of frequency; used when a simple effective
        description is sufficient or when the frequency dependence is not modelled.

The **Rosseland mean** is defined as

.. math::
    :label: eq:rosseland_mean

    \frac{1}{\kappa_R} =
    \frac{\displaystyle\int_0^\infty \frac{1}{\kappa_\nu}
          \frac{\partial B_\nu}{\partial T}\,d\nu}
         {\displaystyle\int_0^\infty \frac{\partial B_\nu}{\partial T}\,d\nu},

which weights each frequency by the gradient of the Planck function.  Because
:math:`\partial B_\nu / \partial T` peaks near :math:`h\nu \approx 3.83\,k_BT`, the
Rosseland mean is dominated by the *most transparent* frequency channels (those with
low :math:`\kappa_\nu`) — a harmonic average that correctly captures photon diffusion in
optically thick material.

The **Planck mean** is the arithmetic average weighted by the emission spectrum,

.. math::
    :label: eq:planck_mean

    \kappa_P =
    \frac{\displaystyle\int_0^\infty \kappa_\nu\,B_\nu\,d\nu}
         {\displaystyle\int_0^\infty B_\nu\,d\nu},

where :math:`B_\nu(T) = (2h\nu^3/c^2)/(\exp(h\nu/k_BT)-1)` is the Planck function.
Unlike :math:`\kappa_R`, the Planck mean is dominated by the *highest*-opacity channels, making
it appropriate for optically thin emission and radiative cooling calculations.  By Jensen's
inequality :math:`\kappa_P \geq \kappa_R` everywhere.

Opacity Laws in Trilobite
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All currently implemented opacity laws are Rosseland mean and live in
:mod:`trilobite.radiation.opacity.grey_opacity.rosseland`.  The module structure is:

.. code-block:: text

    trilobite.radiation.opacity
    └── grey_opacity/          frequency-averaged (grey) opacities
        ├── rosseland/         Rosseland mean — all current implementations
        └── planck/            Planck mean — stub, ready for future work

Each class carries a :attr:`~trilobite.radiation.opacity.base.OpacityLaw.mean_type`
attribute (``"rosseland"`` for all current laws) so that solvers can distinguish opacity
types at runtime without inspecting class identity.  Planck mean implementations would
set ``mean_type = "planck"`` and live in
:mod:`trilobite.radiation.opacity.grey_opacity.planck`.

The top-level :func:`~trilobite.radiation.opacity.utils.get_opacity` resolver exposes
all registered laws by string key regardless of their location in the subpackage hierarchy.

----

Electron Scattering
-------------------

For a fully ionized plasma, electrons scatter photons via the **Thomson process**.  The
cross-section per electron is the Thomson cross-section :math:`\sigma_T = 6.652 \times 10^{-25}`
cm², which gives a mass opacity

.. math::
    :label: eq:electron_scattering

    \boxed{
    \kappa_{\rm es} = \frac{\sigma_T}{m_p}\,\frac{1 + X}{2}
    \approx 0.34\,\mathrm{cm^2\,g^{-1}}
    }

for solar hydrogen mass fraction :math:`X = 0.70`.  The factor :math:`(1+X)/2` arises because
there are :math:`(1+X)/2` free electrons per proton mass in a fully ionized H/He mixture.

Electron scattering is a **grey** and **constant** opacity: it does not depend on density,
temperature, or frequency at the photon energies relevant to accretion disks.  It therefore
dominates the opacity whenever the Kramers contribution (see below) is small, which occurs in
low-density, high-temperature environments.

The partial derivatives are identically zero:

.. math::

    \frac{\partial \ln \kappa_{\rm es}}{\partial \ln \rho} = 0,
    \qquad
    \frac{\partial \ln \kappa_{\rm es}}{\partial \ln T} = 0.

----

Kramers Opacity Laws
--------------------

For a fully ionized plasma at temperatures :math:`T \lesssim 10^{7\text{–}8}` K, photon--matter
interactions are dominated by **free–free** (bremsstrahlung) absorption and **bound–free**
(photoionization) absorption.  Both processes share the same power-law dependence on density and
temperature — a result known as **Kramers' law** — and can therefore be combined into a single
effective opacity.

The Kramers Power Laws
~~~~~~~~~~~~~~~~~~~~~~

The **free–free** Kramers opacity arises from inverse bremsstrahlung (stimulated emission
of photons by free electrons in the Coulomb field of ions).  Integrating the monochromatic
free-free absorption coefficient over a Planck-weighted frequency average and expressing the
result as a mass opacity gives :footcite:t:`RybickiLightman`

.. math::
    :label: eq:kramers_ff

    \kappa_{\rm ff}(\rho, T)
    = \kappa_{\rm ff,0}\,\rho\,T^{-3.5},
    \qquad
    \kappa_{\rm ff,0} \approx 3.68 \times 10^{22}\;\mathrm{cm^5\,g^{-2}\,K^{3.5}},

for solar composition without a Gaunt-factor correction.

The **bound–free** Kramers opacity arises from photoionization of partially or highly ionized
atoms.  The same Planck-weighted average yields the same power-law form but with a larger
normalization :footcite:t:`frank2002accretion`:

.. math::
    :label: eq:kramers_bf

    \kappa_{\rm bf}(\rho, T)
    = \kappa_{\rm bf,0}\,\rho\,T^{-3.5},
    \qquad
    \kappa_{\rm bf,0} \approx 4.34 \times 10^{25}\;\mathrm{cm^5\,g^{-2}\,K^{3.5}}.

Because both processes share the exponents :math:`(\rho^1, T^{-3.5})`, the **combined Kramers
opacity** is

.. math::
    :label: eq:kramers_combined

    \kappa_{\rm ff+bf}(\rho, T)
    = (\kappa_{\rm ff,0} + \kappa_{\rm bf,0})\,\rho\,T^{-3.5}
    \equiv \kappa_{\rm Kr,0}\,\rho\,T^{-3.5},

where :math:`\kappa_{\rm Kr,0} = \kappa_{\rm ff,0} + \kappa_{\rm bf,0} \approx 4.34 \times 10^{25}`
cm⁵ g⁻² K³·⁵ (dominated by the bound-free term).

.. admonition:: Physical Picture

    The :math:`\rho\,T^{-3.5}` scaling can be understood as follows.  The absorption cross-section
    per electron–ion pair grows as :math:`\nu^{-3}` (free–free) and the Boltzmann factor penalizes
    photons with :math:`h\nu \gg k_B T`.  Integrating over a thermal photon distribution weighted
    by :math:`\partial B_\nu / \partial T` selects photons near the peak of the Planck function
    at :math:`\nu_{\rm peak} \propto T`, yielding an effective cross-section
    :math:`\propto T^{-3.5}`.  The density factor arises because the absorption rate is
    proportional to the density of both the absorber (electron) and target (ion).

Partial Derivatives for Pure Kramers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Working in logarithmic space — as required by the Cython solvers — the partial derivatives are
constant and exact:

.. math::
    :label: eq:kramers_partials

    \frac{\partial \ln \kappa_{\rm Kr}}{\partial \ln \rho} = 1,
    \qquad
    \frac{\partial \ln \kappa_{\rm Kr}}{\partial \ln T} = -3.5.

These values are independent of the normalization constant and hold for
:class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersFFOpacity`,
:class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersBFOpacity`, and
:class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersOpacity`.

----

Combined Kramers and Electron Scattering
-----------------------------------------

The most physically complete grey opacity for a hot, fully ionized plasma includes both
Thomson scattering and Kramers absorption.  Because these are distinct processes that
contribute additively to the total opacity,

.. math::
    :label: eq:kramers_es

    \kappa(\rho, T)
    = \kappa_{\rm es}
      + \kappa_{\rm Kr,0}\,\rho\,T^{-3.5}.

This form interpolates smoothly between two limiting regimes:

.. list-table::
    :header-rows: 1

    * - Regime
      - Condition
      - :math:`\kappa \approx`
      - :math:`\partial\ln\kappa/\partial\ln\rho`
      - :math:`\partial\ln\kappa/\partial\ln T`
    * - ES-dominated
      - :math:`\kappa_{\rm Kr}\,\rho\,T^{-3.5} \ll \kappa_{\rm es}`
      - :math:`\kappa_{\rm es}`
      - :math:`\to 0`
      - :math:`\to 0`
    * - Kramers-dominated
      - :math:`\kappa_{\rm Kr}\,\rho\,T^{-3.5} \gg \kappa_{\rm es}`
      - :math:`\kappa_{\rm Kr}\,\rho\,T^{-3.5}`
      - :math:`\to 1`
      - :math:`\to -3.5`

Partial Derivatives for the Combined Form
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\kappa_{\rm Kr}(\rho,T) \equiv \kappa_{\rm Kr,0}\,\rho\,T^{-3.5}` denote the
Kramers contribution.  Then

.. math::
    :label: eq:kramers_es_partials

    \frac{\partial \ln \kappa}{\partial \ln \rho}
    = \frac{\kappa_{\rm Kr}(\rho,T)}{\kappa(\rho,T)},
    \qquad
    \frac{\partial \ln \kappa}{\partial \ln T}
    = -3.5\,\frac{\kappa_{\rm Kr}(\rho,T)}{\kappa(\rho,T)}.

These expressions hold for
:class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersFFESOpacity`,
:class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersBFESOpacity`, and
:class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersESOpacity`.
Note that the two partial derivatives are always related by

.. math::

    \frac{\partial \ln \kappa}{\partial \ln T}
    = -3.5\,\frac{\partial \ln \kappa}{\partial \ln \rho},

which provides a useful consistency check when implementing new combined-form laws.

----

Log-Space Representation
-------------------------

The Cython-level disk solvers work entirely in **natural-logarithm CGS space** for numerical
stability.  For a function :math:`\kappa(\rho, T)`, the key quantities passed between Python
and Cython are

.. math::

    \ln\kappa,\quad
    \frac{\partial\ln\kappa}{\partial\ln\rho},\quad
    \frac{\partial\ln\kappa}{\partial\ln T},

evaluated at :math:`(\ln T,\,\ln\rho)` rather than :math:`(T,\,\rho)`.  This choice ensures
that the solver can represent opacity values spanning many decades (from
:math:`{\sim}10^{-7}` to :math:`{\sim}10^2` cm² g⁻¹) without overflow or cancellation
errors.

The correspondence between physical opacity and log-space representation is

.. list-table::
    :header-rows: 1

    * - Public API
      - Log-space private method
      - Returns
    * - :meth:`~trilobite.radiation.opacity.base.OpacityLaw.opacity`
      - :meth:`~trilobite.radiation.opacity.base.OpacityLaw._log_opacity`
      - :math:`\ln\kappa`
    * - :meth:`~trilobite.radiation.opacity.base.OpacityLaw.dlogkappa_dlogrho`
      - :meth:`~trilobite.radiation.opacity.base.OpacityLaw._dlogkappa_dlogrho`
      - :math:`\partial\ln\kappa/\partial\ln\rho`
    * - :meth:`~trilobite.radiation.opacity.base.OpacityLaw.dlogkappa_dlogT`
      - :meth:`~trilobite.radiation.opacity.base.OpacityLaw._dlogkappa_dlogT`
      - :math:`\partial\ln\kappa/\partial\ln T`

----

Summary of Implemented Laws
----------------------------

.. list-table::
    :header-rows: 1

    * - Class
      - Formula
      - :math:`\partial\ln\kappa/\partial\ln\rho`
      - :math:`\partial\ln\kappa/\partial\ln T`
    * - :class:`~trilobite.radiation.opacity.grey_opacity.base.ConstantGreyOpacity`
      - :math:`\kappa_0`
      - 0
      - 0
    * - :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.ElectronScatteringOpacity`
      - :math:`\kappa_{\rm es}`
      - 0
      - 0
    * - :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersFFOpacity`
      - :math:`\kappa_{\rm ff,0}\,\rho\,T^{-3.5}`
      - 1
      - −3.5
    * - :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersBFOpacity`
      - :math:`\kappa_{\rm bf,0}\,\rho\,T^{-3.5}`
      - 1
      - −3.5
    * - :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersOpacity`
      - :math:`\kappa_{\rm Kr,0}\,\rho\,T^{-3.5}`
      - 1
      - −3.5
    * - :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersFFESOpacity`
      - :math:`\kappa_{\rm es} + \kappa_{\rm ff,0}\,\rho\,T^{-3.5}`
      - :math:`(0,\,1)`
      - :math:`(-3.5,\,0)`
    * - :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersBFESOpacity`
      - :math:`\kappa_{\rm es} + \kappa_{\rm bf,0}\,\rho\,T^{-3.5}`
      - :math:`(0,\,1)`
      - :math:`(-3.5,\,0)`
    * - :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersESOpacity`
      - :math:`\kappa_{\rm es} + \kappa_{\rm Kr,0}\,\rho\,T^{-3.5}`
      - :math:`(0,\,1)`
      - :math:`(-3.5,\,0)`

Values in the derivative columns denote the asymptotic limits (ES-dominated, Kramers-dominated)
for the combined-form laws.

    * - :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.OPALOpacity`
      - :math:`\kappa_R(T,\rho)` from bilinear table interpolation
      - :math:`\partial\ln\kappa/\partial\ln\rho` (table-derived)
      - :math:`\partial\ln\kappa/\partial\ln T` (table-derived)

----

.. _opacity-opal-theory:

Table-based Rosseland Mean Opacity (OPAL)
-----------------------------------------

The OPAL project :footcite:t:`Badnell2005` provides pre-computed Rosseland mean opacities
for a wide range of stellar compositions.  Trilobite ships a bundled table of 126
compositions from the Asplund & Grevesse (2005) solar mixture, covering

.. math::

    3.75 \le \log_{10}(T\,[\mathrm{K}]) \le 8.70,
    \qquad
    -8 \le \log_{10}(R) \le 1,

where :math:`R \equiv \rho / T_6^3` and :math:`T_6 = 10^{-6}\,T`.  Cells outside the
physically valid region are marked ``NaN`` by the OPAL authors and are handled gracefully
by the out-of-bounds strategy chosen at construction time.

Coordinate System
~~~~~~~~~~~~~~~~~

OPAL tables are stored on a :math:`(\log_{10}T,\,\log_{10}R)` grid.  The compressed variable
:math:`R` effectively normalises density by the temperature-dependent degeneracy threshold,
making the valid table region approximately rectangular in :math:`\log_{10}T`–:math:`\log_{10}R`
space even though the valid :math:`\log_{10}T`–:math:`\log_{10}\rho` region is highly irregular.

The coordinate transformation (applied inside the Cython kernel) is

.. math::

    \log_{10}R = \log_{10}\rho - 3\,(\log_{10}T - 6).

Trilobite also supports tables stored on a direct :math:`(\log_{10}T,\,\log_{10}\rho)` grid
via the ``'T_rho'`` coordinate system, in which no transformation is applied.

Bilinear Interpolation
~~~~~~~~~~~~~~~~~~~~~~

Within each rectangular cell of the :math:`(\log_{10}T,\,\log_{10}R)` grid, the Cython kernel
:class:`~trilobite.radiation.opacity.grey_opacity.rosseland._opal_table.C_OPALTableOpacity` applies standard
bilinear interpolation of :math:`\log_{10}\kappa`:

.. math::

    \log_{10}\kappa \approx
      (1-t_x)(1-t_y)\,k_{00}
    + t_x(1-t_y)\,k_{10}
    + (1-t_x)t_y\,k_{01}
    + t_x t_y\,k_{11},

where :math:`t_x = (\log_{10}T - g_1^{(i)}) / (g_1^{(i+1)} - g_1^{(i)})` and similarly
for :math:`t_y`.  If any corner :math:`k_{ij}` is ``NaN``, the cell is treated as invalid
and the out-of-bounds handler is invoked.

Log-space Derivatives
~~~~~~~~~~~~~~~~~~~~~

Because the interpolation is piecewise bilinear, the log-space derivatives are piecewise
constant within each cell and can be computed analytically from the cell corners:

.. math::

    \frac{\partial\ln\kappa}{\partial\ln\rho}
    = \frac{\partial\log_{10}\kappa}{\partial\log_{10}g_2}

    \frac{\partial\ln\kappa}{\partial\ln T}\bigg|_\rho
    = \begin{cases}
        \dfrac{\partial\log_{10}\kappa}{\partial\log_{10}T}\bigg|_R
        - 3\,\dfrac{\partial\log_{10}\kappa}{\partial\log_{10}R}\bigg|_T
        & \text{(T\_R system)} \\[6pt]
        \dfrac{\partial\log_{10}\kappa}{\partial\log_{10}T}\bigg|_\rho
        & \text{(T\_rho system)}
      \end{cases}

The :math:`-3` cross-term in the T\_R system arises from the chain rule
:math:`\partial\log_{10}R/\partial\log_{10}T\big|_\rho = -3`.

Out-of-bounds Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

Queries outside the table domain (or landing in NaN cells) are handled by one of three
strategies selected at construction time:

.. list-table::
    :header-rows: 1
    :widths: 15 85

    * - Mode
      - Behaviour
    * - ``'raise'``
      - Raises :exc:`ValueError` with the out-of-range coordinates (default; safest for
        debugging).
    * - ``'clamp'``
      - Returns the opacity at the nearest valid boundary point silently.  Useful when
        the query point is expected to be near but occasionally outside the domain.
    * - ``'nan'``
      - Returns ``NaN`` silently.  Useful for plotting or masking workflows where gaps
        are preferable to exceptions.

When to Use OPAL vs. Analytic Laws
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 30 35 35

    * - Scenario
      - Recommended law
      - Reason
    * - High-:math:`T`, low-:math:`\rho` (accretion disk corona)
      - :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.ElectronScatteringOpacity`
      - Constant ES dominates; fast; exact.
    * - Intermediate :math:`T` (disk midplane, :math:`10^5`–:math:`10^7` K)
      - :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersESOpacity`
        or :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.OPALOpacity`
      - KramersES is fast and analytic; OPAL captures the iron-group peak and
        non-power-law behaviour.
    * - Stellar interior / full :math:`(T,\rho)` sweep
      - :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.OPALOpacity`
      - Accurate Rosseland mean including composition dependence.
    * - Parameter-grid studies or MCMC
      - :class:`~trilobite.radiation.opacity.grey_opacity.rosseland.models.KramersESOpacity`
      - Analytic evaluation is O(1); OPAL bilinear interpolation is slightly slower.

----

.. footbibliography::
