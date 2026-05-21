.. _free_free_emission:

================================
Free-Free Emission Tools
================================

.. hint::

    See :ref:`free_free_theory` for a detailed overview of the underlying physics of
    free-free (bremsstrahlung) radiation.

The :mod:`triceratops.radiation.free_free` module provides a complete set of tools for
computing thermal free-free emission and absorption in ionized astrophysical plasmas.  The
module is organized into three sub-modules:

* :mod:`~triceratops.radiation.free_free.core` — monochromatic emissivity and absorption
  coefficients.
* :mod:`~triceratops.radiation.free_free.gaunt_factor` — Gaunt factor evaluation (analytic
  approximation and tabulated interpolators).
* :mod:`~triceratops.radiation.free_free.absorption` — free-free optical depth integration
  over a variety of CSM density profiles.

Each sub-module follows Triceratops's standard **two-level API pattern**: a private, log-space
backend optimized for numerical performance and a public, unit-aware interface that accepts and
returns :class:`astropy.units.Quantity` objects.

.. contents::
    :local:
    :depth: 2

----

Emissivity and Absorption Coefficients
---------------------------------------

The :mod:`~triceratops.radiation.free_free.core` module implements the monochromatic thermal
free-free emissivity :math:`j_\nu` and absorption coefficient :math:`\alpha_\nu` together
with their Rayleigh–Jeans and Wien limiting forms.

The Two-Level API
~~~~~~~~~~~~~~~~~

All computations are performed in two layers:

* **Private backends** (``_log_ff_*``) operate entirely in natural-logarithm CGS space.
  All six inputs (``log_nu``, ``log_n_e``, ``log_n_i``, ``log_Z``, ``log_T``, ``g_ff``) are
  expected in natural log CGS form, and the functions return the natural log of the result.
  Working in log-space provides numerical stability over the many decades of dynamic range
  that arise in astrophysical plasmas and eliminates unit overhead in performance-critical
  loops.

* **Public wrappers** (``compute_ff_*``) perform unit coercion via
  :func:`~triceratops.utils.misc_utils.ensure_in_units`, delegate to the appropriate backend,
  and return :class:`~astropy.units.Quantity` objects with appropriate CGS units.

.. important::

    The private backends accept plain floats assumed to be in natural-log CGS.  Passing
    un-logged values or quantities in non-CGS units to the private functions will produce
    silently incorrect results.  Unless you are building a performance-critical pipeline,
    use the public API.

Emissivity
~~~~~~~~~~

The thermal free-free spectral emissivity follows
:footcite:t:`RybickiLightman`, Eq. 5.14:

.. math::

    j_\nu =
    C_{\rm ff}\,
    Z^2\, n_e\, n_i\,
    T^{-1/2}\,
    e^{-h\nu / k_B T}\,
    g_{\rm ff},

where :math:`C_{\rm ff} = 6.8 \times 10^{-38}` CGS.

Triceratops implements this via :func:`~triceratops.radiation.free_free.core.compute_ff_emissivity`:

.. tab-set::

    .. tab-item:: High-Level API

        The public API accepts :class:`astropy.units.Quantity` inputs and returns the
        emissivity as a :class:`~astropy.units.Quantity` in
        erg s\ :sup:`-1` cm\ :sup:`-3` Hz\ :sup:`-1` sr\ :sup:`-1`:

        .. code-block:: python

            import numpy as np
            from astropy import units as u
            from triceratops.radiation.free_free.core import compute_ff_emissivity

            j = compute_ff_emissivity(
                nu=1e9 * u.Hz,
                n_e=1e3 * u.cm**-3,
                n_i=1e3 * u.cm**-3,
                Z=1,
                T=1e4 * u.K,
            )
            print(j.to(u.erg / u.s / u.cm**3 / u.Hz / u.sr))

        Plain floats are also accepted and interpreted as Hz, cm\ :sup:`-3`, and K
        respectively—the Gaunt factor ``g_ff`` is always dimensionless:

        .. code-block:: python

            j = compute_ff_emissivity(nu=1e9, n_e=1e3, n_i=1e3, Z=1, T=1e4)

    .. tab-item:: Low-Level API

        The private backend :func:`~triceratops.radiation.free_free.core._log_ff_emissivity`
        accepts natural logarithms of all arguments:

        .. code-block:: python

            import numpy as np
            from triceratops.radiation.free_free.core import _log_ff_emissivity

            log_j = _log_ff_emissivity(
                log_nu=np.log(1e9),   # ln(ν / Hz)
                log_n_e=np.log(1e3),  # ln(n_e / cm⁻³)
                log_n_i=np.log(1e3),
                log_Z=np.log(1.0),
                log_T=np.log(1e4),    # ln(T / K)
                g_ff=5.0,
            )
            j = np.exp(log_j)  # erg s⁻¹ cm⁻³ Hz⁻¹ sr⁻¹

The Rayleigh–Jeans emissivity—valid when :math:`h\nu \ll k_BT`—drops the exponential factor
and is **frequency-independent**, scaling as :math:`T^{-1/2}`.  It is implemented via
:func:`~triceratops.radiation.free_free.core.compute_ff_RJ_emissivity` (high-level) and
:func:`~triceratops.radiation.free_free.core._log_ff_RJ_emissivity` (log-space backend).

.. note::

    The ``log_nu`` argument is accepted by ``_log_ff_RJ_emissivity`` for interface
    consistency but does not contribute to the result.  The output is broadcast to the
    shape of ``log_nu`` so that the function behaves identically to the full emissivity
    when used in vectorized expressions.

Absorption Coefficient
~~~~~~~~~~~~~~~~~~~~~~

The thermal free-free absorption coefficient is (:footcite:t:`RybickiLightman`, Eq. 5.19):

.. math::

    \alpha_\nu =
    C_\alpha\,
    Z^2\, n_e\, n_i\,
    T^{-1/2}\,
    \nu^{-3}\,
    \bigl(1 - e^{-h\nu / k_B T}\bigr)\,
    g_{\rm ff},

where :math:`C_\alpha = 3.7 \times 10^{8}` CGS.

This is implemented via :func:`~triceratops.radiation.free_free.core.compute_ff_absorption`:

.. tab-set::

    .. tab-item:: High-Level API

        .. code-block:: python

            import numpy as np
            from astropy import units as u
            from triceratops.radiation.free_free.core import compute_ff_absorption

            alpha = compute_ff_absorption(
                nu=1e9 * u.Hz,
                n_e=1e4 * u.cm**-3,
                n_i=1e4 * u.cm**-3,
                Z=1,
                T=1e4 * u.K,
            )
            print(alpha.to(u.cm**-1))

        The result can be integrated along a line of sight to obtain the optical depth.
        For a uniform slab of thickness :math:`L`:

        .. code-block:: python

            nu_arr = np.geomspace(1e8, 1e12, 300) * u.Hz
            alpha_arr = compute_ff_absorption(
                nu=nu_arr, n_e=1e5 * u.cm**-3, n_i=1e5 * u.cm**-3, Z=1, T=1e4 * u.K
            )
            tau = (alpha_arr * 1e17 * u.cm).decompose()

    .. tab-item:: Low-Level API

        The log-space backend :func:`~triceratops.radiation.free_free.core._log_ff_absorption`
        evaluates :math:`\log(1 - e^{-h\nu/k_BT})` via ``numpy.log1p`` to avoid
        catastrophic cancellation at low frequencies where :math:`h\nu/k_BT \ll 1`:

        .. code-block:: python

            import numpy as np
            from triceratops.radiation.free_free.core import _log_ff_absorption

            log_alpha = _log_ff_absorption(
                log_nu=np.log(1e9),
                log_n_e=np.log(1e4),
                log_n_i=np.log(1e4),
                log_Z=np.log(1.0),
                log_T=np.log(1e4),
                g_ff=5.0,
            )

The Rayleigh–Jeans approximation—valid when :math:`h\nu \ll k_BT`—replaces
:math:`(1-e^{-h\nu/k_BT}) \to h\nu/k_BT`, yielding :math:`\alpha_\nu^{\rm RJ} \propto \nu^{-2}
T^{-3/2}`.  This is the standard radio bremsstrahlung opacity scaling.  It is implemented via
:func:`~triceratops.radiation.free_free.core.compute_ff_RJ_absorption` and its log-space backend.

----

Gaunt Factor Evaluation
-----------------------

The :mod:`~triceratops.radiation.free_free.gaunt_factor` module provides two approaches to
evaluating the free-free Gaunt factor :math:`g_{\rm ff}(Z, T, \nu)`.

The Gaunt factor depends on two dimensionless collision parameters:

.. math::

    u \equiv \frac{h\nu}{k_B T},
    \qquad
    \gamma^2 \equiv \frac{Z^2 R_y}{k_B T},

where :math:`R_y = 13.6\,{\rm eV}` is the Rydberg energy.  Triceratops uses these parameters
internally to map physical inputs :math:`(Z, T, \nu)` onto the table axes.

Analytic Approximation
~~~~~~~~~~~~~~~~~~~~~~

A fast closed-form approximation due to :footcite:t:`Draine2011ISM` (Eq. 10.9) is provided by
:func:`~triceratops.radiation.free_free.gaunt_factor.gaunt_ff_draine`.  It accepts
unit-aware or plain-float inputs and is suitable for quick estimates in the radio through
infrared regime:

.. code-block:: python

    from triceratops.radiation.free_free.gaunt_factor import gaunt_ff_draine

    # Scalar
    gff = gaunt_ff_draine(Z=1, T=1e4, nu=1e10)

    # With astropy units
    from astropy import units as u
    gff = gaunt_ff_draine(Z=1, T=1e6 * u.K, nu=5.0 * u.GHz)

.. note::

    The Draine approximation is accurate to a few percent in the RJ regime but can deviate
    more significantly at high frequencies or extreme temperatures.  For production-quality
    calculations, the tabulated interpolators described below are preferred.

Tabulated Interpolators
~~~~~~~~~~~~~~~~~~~~~~~

Triceratops ships two pre-computed Gaunt factor tables from
:footcite:t:`2014MNRAS.444..420V` and provides a class hierarchy for convenient, accurate
interpolation.

.. list-table::
   :widths: 35 20 45
   :header-rows: 1

   * - Class
     - Table dimensions
     - When to use
   * - :class:`~triceratops.radiation.free_free.gaunt_factor.NonRelativisticGauntFactorInterpolator`
     - 2D (:math:`\log_{10} u`,  :math:`\log_{10}\gamma^2`)
     - Default; appropriate for most radio and thermal-plasma applications
   * - :class:`~triceratops.radiation.free_free.gaunt_factor.RelativisticGauntFactorInterpolator`
     - 3D (:math:`Z`, :math:`\log_{10} u`, :math:`\log_{10}\gamma^2`)
     - High-temperature (:math:`T \gtrsim 10^7` K) or high-:math:`Z` plasma

Both classes share the same interface: they are callable with :math:`(Z, T, \nu)` and support
the Python ``in`` operator to test whether a point lies within the table boundaries.

.. tab-set::

    .. tab-item:: Non-Relativistic Interpolator

        The :class:`~triceratops.radiation.free_free.gaunt_factor.NonRelativisticGauntFactorInterpolator`
        wraps a 2D :func:`scipy.interpolate.RegularGridInterpolator` over the non-relativistic
        van Hoof (2014) table.  The ionic charge :math:`Z` enters only through the coordinate
        transform to :math:`\log_{10}\gamma^2` and is not a table axis.

        The default instance can be obtained via the factory function
        :func:`~triceratops.radiation.free_free.gaunt_factor.get_default_gaunt_interpolator`:

        .. code-block:: python

            import numpy as np
            from triceratops.radiation.free_free.gaunt_factor import (
                get_default_gaunt_interpolator,
            )

            nr = get_default_gaunt_interpolator()

            # Scalar evaluation
            gff = nr(Z=1, T=1e4, nu=1e10)
            print(gff)           # ≈ 4.46

            # Array evaluation
            T_arr = np.geomspace(1e4, 1e8, 50)
            gff_arr = nr(Z=1, T=T_arr, nu=1e10)

            # Domain check
            print((1, 1e4, 1e10) in nr)  # True

    .. tab-item:: Relativistic Interpolator

        The :class:`~triceratops.radiation.free_free.gaunt_factor.RelativisticGauntFactorInterpolator`
        wraps a 3D :func:`scipy.interpolate.RegularGridInterpolator`.  Z is a first-class table
        axis, allowing the interpolator to capture the explicit Z-dependence of relativistic
        corrections beyond what is encoded in :math:`\log_{10}\gamma^2` alone.

        The default relativistic instance can be obtained via
        :func:`~triceratops.radiation.free_free.gaunt_factor.get_default_relativistic_gaunt_interpolator`:

        .. code-block:: python

            import numpy as np
            from triceratops.radiation.free_free.gaunt_factor import (
                get_default_relativistic_gaunt_interpolator,
            )

            r = get_default_relativistic_gaunt_interpolator()

            # Scalar evaluation
            gff = r(Z=1, T=1e4, nu=1e10)
            print(gff)           # ≈ 4.71

            # Domain check
            print((1, 1e4, 1e10) in r)  # True

Class Hierarchy
^^^^^^^^^^^^^^^

Both interpolators share a common abstract base class,
:class:`~triceratops.radiation.free_free.gaunt_factor.GauntFactorInterpolatorBase`, which
defines the public interface and provides the HDF5 loading helper.  Users implementing custom
Gaunt factor tables can subclass this base class to integrate them into the Triceratops API.

.. code-block:: python

    from triceratops.radiation.free_free.gaunt_factor import (
        GauntFactorInterpolatorBase,
        NonRelativisticGauntFactorInterpolator,
        RelativisticGauntFactorInterpolator,
        get_default_gaunt_interpolator,
        get_default_relativistic_gaunt_interpolator,
    )

    nr = get_default_gaunt_interpolator()
    r  = get_default_relativistic_gaunt_interpolator()

    assert isinstance(nr, GauntFactorInterpolatorBase)
    assert isinstance(r,  GauntFactorInterpolatorBase)

    print(nr.description)  # Non-relativistic free-free Gaunt factors (...)
    print(r.description)   # Relativistic free-free Gaunt factors (...)

----

Free-Free Optical Depth
------------------------

The :mod:`~triceratops.radiation.free_free.absorption` module provides routines to integrate
:math:`\alpha_\nu(r)` along a line of sight through an ionized CSM for five physically motivated
density profiles.  For each profile type there is both an **exact** variant (using the full
:math:`(1 - e^{-h\nu/k_BT})` factor) and a **Rayleigh–Jeans** variant (using the approximation
:math:`\propto \nu^{-2} T^{-3/2}`).  All public functions accept and return unit-bearing
quantities; the quadrature-based variants accept CGS-only callables for maximum flexibility.

.. important::

    For the quadrature-based profiles (``_from_quadrature`` variants), the callable arguments
    must return values in **CGS units** without any astropy unit overhead.  This is the only
    place in the public API where plain CGS callables are required; all other parameters
    remain unit-aware.

Profile Overview
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Function pair
     - Description
   * - :func:`~triceratops.radiation.free_free.absorption.compute_ff_optical_depth_from_quadrature` /
       :func:`~triceratops.radiation.free_free.absorption.compute_ff_RJ_optical_depth_from_quadrature`
     - Adaptive quadrature over arbitrary user-supplied CGS callables for
       :math:`n_e(r)`, :math:`n_i(r)`, :math:`T(r)`.  Maximum flexibility for non-analytic
       or tabulated CSM models.
   * - :func:`~triceratops.radiation.free_free.absorption.compute_ff_optical_depth_from_arrays` /
       :func:`~triceratops.radiation.free_free.absorption.compute_ff_RJ_optical_depth_from_arrays`
     - Trapezoidal integration over pre-computed (:math:`r`, :math:`\alpha_\nu`) or
       (:math:`r`, :math:`n_e`, :math:`n_i`, :math:`T`) grids.  Ideal for
       post-processing hydrodynamic simulation outputs.
   * - :func:`~triceratops.radiation.free_free.absorption.compute_ff_optical_depth_wind` /
       :func:`~triceratops.radiation.free_free.absorption.compute_ff_RJ_optical_depth_wind`
     - Analytic integral for a steady stellar wind with
       :math:`\rho \propto r^{-2}` (constant mass-loss rate :math:`\dot{M}`, wind velocity
       :math:`v_w`).  Gives :math:`\tau_{\rm ff} \propto r^{-3}`.
   * - :func:`~triceratops.radiation.free_free.absorption.compute_ff_optical_depth_shell` /
       :func:`~triceratops.radiation.free_free.absorption.compute_ff_RJ_optical_depth_shell`
     - Analytic integral for a uniform-density shell between radii :math:`r` and
       :math:`r_{\rm max}`.  Returns :math:`\tau_{\rm ff} = \alpha_\nu (r_{\rm max} - r)`.
   * - :func:`~triceratops.radiation.free_free.absorption.compute_ff_optical_depth_powerlaw` /
       :func:`~triceratops.radiation.free_free.absorption.compute_ff_RJ_optical_depth_powerlaw`
     - Analytic integral for a generalized power-law density profile
       :math:`\rho \propto r^{-p}` between :math:`r` and :math:`r_{\rm max}`.  Encompasses
       the wind (:math:`p = 2`) and uniform (:math:`p = 0`) cases.

Usage Examples
~~~~~~~~~~~~~~

.. tab-set::

    .. tab-item:: Quadrature

        Compute the free-free optical depth through a CSM described by arbitrary number
        density and temperature profiles, using adaptive quadrature from :math:`r` to
        :math:`r_{\rm max}`:

        .. code-block:: python

            import numpy as np
            from astropy import units as u
            from triceratops.radiation.free_free.absorption import (
                compute_ff_optical_depth_from_quadrature,
            )

            # Define CGS callables (no astropy units inside)
            def n_e(r):
                return 1e3 * (r / 1e15) ** -2  # cm⁻³, wind profile

            def n_i(r):
                return n_e(r)

            def T(r):
                return 1e4  # Kelvin, uniform temperature

            tau = compute_ff_optical_depth_from_quadrature(
                nu=1e9 * u.Hz,
                Z=1.0,
                n_e_func=n_e,
                n_i_func=n_i,
                T_func=T,
                r=1e15 * u.cm,
                r_max=2e15 * u.cm,
                g_ff=5.0,
            )
            print(tau)  # dimensionless optical depth

    .. tab-item:: Wind Profile

        For a steady stellar wind with mass-loss rate :math:`\dot{M}` and wind velocity
        :math:`v_w`, use the analytic wind integral:

        .. code-block:: python

            from astropy import units as u
            from triceratops.radiation.free_free.absorption import (
                compute_ff_optical_depth_wind,
                compute_ff_RJ_optical_depth_wind,
            )

            # Wind parameters
            mdot = 1e-5 * u.Msun / u.yr
            v_w  = 10   * u.km / u.s

            tau_exact = compute_ff_optical_depth_wind(
                nu=1e9 * u.Hz,
                Z=1.0,
                mdot=mdot,
                v_wind=v_w,
                T=1e4 * u.K,
                r=1e15 * u.cm,
                g_ff=5.0,
            )

            tau_rj = compute_ff_RJ_optical_depth_wind(
                nu=1e9 * u.Hz,
                Z=1.0,
                mdot=mdot,
                v_wind=v_w,
                T=1e4 * u.K,
                r=1e15 * u.cm,
                g_ff=5.0,
            )

    .. tab-item:: Shell Profile

        For a uniform-density shell, the integral reduces to
        :math:`\tau = \alpha_\nu (r_{\rm max} - r)`:

        .. code-block:: python

            from astropy import units as u
            from triceratops.radiation.free_free.absorption import (
                compute_ff_optical_depth_shell,
            )

            tau = compute_ff_optical_depth_shell(
                nu=1e9 * u.Hz,
                Z=1.0,
                n_e=1e3 * u.cm**-3,
                n_i=1e3 * u.cm**-3,
                T=1e4 * u.K,
                r=1e15 * u.cm,
                r_max=2e15 * u.cm,
                g_ff=5.0,
            )

    .. tab-item:: Power-Law Profile

        For a generalized :math:`\rho \propto r^{-p}` profile, the integral is computed
        analytically.  The power-law exponent is :math:`p = 0` (uniform) or :math:`p = 2`
        (wind) as special cases:

        .. code-block:: python

            from astropy import units as u
            from triceratops.radiation.free_free.absorption import (
                compute_ff_optical_depth_powerlaw,
            )

            # General p = 1.5 profile
            tau = compute_ff_optical_depth_powerlaw(
                nu=1e9 * u.Hz,
                Z=1.0,
                n_e=1e3 * u.cm**-3,
                n_i=1e3 * u.cm**-3,
                T=1e4 * u.K,
                r=1e15 * u.cm,
                r_max=2e15 * u.cm,
                p=1.5,
                g_ff=5.0,
            )

.. warning::

    For power-law profiles with :math:`p > 0.5` (i.e. :math:`k \equiv 1-2p < 0`), the
    optical depth integral converges as :math:`r_{\rm max} \to \infty`.  Setting
    ``r_max=np.inf`` is valid and will return the asymptotic total optical depth in this
    case.  For :math:`p < 0.5` the integral diverges at :math:`r_{\rm max} \to \infty`
    and a finite upper limit must be provided.

----

.. footbibliography::
