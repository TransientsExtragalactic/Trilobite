.. _blackbody_user_guide:

====================
Blackbody Radiation
====================

The :mod:`trilobite.radiation.blackbody` module provides the Planck function and
related thermal-radiation utilities.  It follows the standard two-level API pattern
used throughout Trilobite: private ``_*_cgs`` kernels operate on plain NumPy arrays
with no unit overhead (suitable for MCMC hot loops), and public wrappers accept and
return :class:`~astropy.units.Quantity` objects for interactive use.

.. contents::
    :local:
    :depth: 2

----

Quick Start
-----------

The example below plots :math:`B_\nu` for three representative temperatures and marks
the Wien peak of each.

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    from trilobite.radiation.blackbody import planck_B, wien_peak_frequency

    nu = np.geomspace(1e11, 1e17, 500) * u.Hz
    temperatures = [1e4, 1e5, 1e6] * u.K
    labels = [r"$10^4\,\mathrm{K}$", r"$10^5\,\mathrm{K}$", r"$10^6\,\mathrm{K}$"]

    fig, ax = plt.subplots(figsize=(7, 4))

    for T, label in zip(temperatures, labels):
        B = planck_B(T, nu=nu)
        line, = ax.loglog(nu.value, B.value, label=label)
        nu_peak = wien_peak_frequency(T)
        B_peak = planck_B(T, nu=nu_peak)
        ax.axvline(nu_peak.value, color=line.get_color(), linestyle="--", alpha=0.5)

    ax.set_xlabel(r"Frequency $\nu$ [Hz]")
    ax.set_ylabel(r"$B_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$]")
    ax.set_title("Planck Function")
    ax.legend()
    fig.tight_layout()

----

.. _blackbody_theory:

Theory
------

Planck Function
~~~~~~~~~~~~~~~

The Planck function gives the specific intensity (spectral radiance) of a blackbody
in thermal equilibrium at temperature :math:`T`:

.. math::

    B_\nu(\nu, T) = \frac{2 h \nu^3}{c^2}
                    \frac{1}{e^{h\nu / k_{\rm B} T} - 1}

in units of :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.  The equivalent
expression per unit wavelength is:

.. math::

    B_\lambda(\lambda, T) = \frac{2 h c^2}{\lambda^5}
                            \frac{1}{e^{h c / \lambda k_{\rm B} T} - 1}

in units of :math:`\mathrm{erg\,s^{-1}\,cm^{-3}\,sr^{-1}}`.  Both are related by
:math:`B_\lambda = B_\nu \cdot c / \lambda^2`.

**Numerical stability.** The denominator :math:`e^x - 1` is evaluated using
``numpy.expm1`` to avoid cancellation errors in the Rayleigh-Jeans limit
(:math:`x \ll 1`).

Wien Displacement Law
~~~~~~~~~~~~~~~~~~~~~

Differentiating :math:`B_\nu` with respect to :math:`\nu` and setting the result to
zero yields the transcendental equation :math:`x_{\rm peak} = 3(1 - e^{-x_{\rm peak}})`,
whose solution is :math:`x_{\rm peak} \approx 2.8214`.  The peak frequency is therefore:

.. math::

    \nu_{\rm peak} = \frac{x_{\rm peak}\,k_{\rm B}}{h}\,T
                   \approx 5.879 \times 10^{10}\,T\ \mathrm{Hz\,K^{-1}}

Similarly, the peak wavelength of :math:`B_\lambda` satisfies:

.. math::

    \lambda_{\rm peak} = \frac{b}{T},
    \qquad b \approx 0.2898\;\mathrm{cm\,K}

Note that :math:`\nu_{\rm peak}` and :math:`\lambda_{\rm peak}` are not related by
:math:`c = \lambda \nu` because they are maxima of different representations of the
spectrum.

Stefan-Boltzmann Law
~~~~~~~~~~~~~~~~~~~~

Integrating the Planck function over all frequencies and the outgoing hemisphere gives
the bolometric radiative flux from a blackbody surface:

.. math::

    F = \sigma_{\rm SB}\,T^4,
    \qquad \sigma_{\rm SB} \approx 5.670 \times 10^{-5}
    \;\mathrm{erg\,s^{-1}\,cm^{-2}\,K^{-4}}

----

Usage Guide
-----------

Per-Frequency Specific Intensity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`~trilobite.radiation.blackbody.planck_B` to evaluate :math:`B_\nu`
at an array of frequencies:

.. code-block:: python

    import numpy as np
    import astropy.units as u
    from trilobite.radiation.blackbody import planck_B

    nu = np.geomspace(1e13, 1e17, 200) * u.Hz
    T  = 1e5 * u.K
    B  = planck_B(T, nu=nu)          # shape (200,), unit erg / (s cm**2 Hz sr)

Per-Wavelength Specific Intensity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :func:`~trilobite.radiation.blackbody.planck_B` with the ``lam`` keyword to evaluate :math:`B_\lambda`:

.. code-block:: python

    import numpy as np
    import astropy.units as u
    from trilobite.radiation.blackbody import planck_B

    lam = np.geomspace(1e-6, 1e-3, 200) * u.cm   # UV through IR
    T   = 5778.0 * u.K
    B   = planck_B(T, lam=lam)      # unit erg / (s cm**3 sr)

Wien Peak
~~~~~~~~~

:func:`~trilobite.radiation.blackbody.wien_peak_frequency` and
:func:`~trilobite.radiation.blackbody.wien_peak_wavelength` return the peak
of :math:`B_\nu` and :math:`B_\lambda`, respectively:

.. code-block:: python

    from astropy import units as u
    from trilobite.radiation.blackbody import wien_peak_frequency, wien_peak_wavelength

    T = 5778.0 * u.K
    print(wien_peak_frequency(T).to(u.THz))    # ~339 THz for the Sun
    print(wien_peak_wavelength(T).to(u.nm))    # ~501 nm for the Sun

Bolometric Flux
~~~~~~~~~~~~~~~

:func:`~trilobite.radiation.blackbody.stefan_boltzmann_flux` returns
:math:`F = \sigma_{\rm SB} T^4`:

.. code-block:: python

    from astropy import units as u
    from trilobite.radiation.blackbody import stefan_boltzmann_flux

    F = stefan_boltzmann_flux(5778.0 * u.K)
    print(F.to(u.L_sun / u.R_sun**2))    # should be ~1 L_sun R_sun^-2

Using CGS Kernels Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For performance-critical code (e.g. inside an MCMC loop), import the private CGS
functions directly and pass plain NumPy arrays:

.. code-block:: python

    import numpy as np
    from trilobite.radiation.blackbody import _log_planck_fnu_cgs

    log_nu = np.log(np.geomspace(1e13, 1e17, 200))   # natural log, Hz
    log_T  = np.log(1e5)                               # natural log, K
    log_B  = _log_planck_fnu_cgs(log_nu, log_T)       # plain ndarray, log erg / (s cm**2 Hz sr)
    B      = np.exp(log_B)

.. important::

    The CGS kernels accept only plain NumPy arrays and Python floats.  Passing an
    :class:`~astropy.units.Quantity` will raise an error.  No unit checking is
    performed — it is the caller's responsibility to pass values in CGS.

----

API Reference
-------------

.. autofunction:: trilobite.radiation.blackbody.planck_B

.. autofunction:: trilobite.radiation.blackbody.planck_surface_flux

.. autofunction:: trilobite.radiation.blackbody.planck_flux

.. autofunction:: trilobite.radiation.blackbody.wien_peak_frequency

.. autofunction:: trilobite.radiation.blackbody.wien_peak_wavelength

.. autofunction:: trilobite.radiation.blackbody.stefan_boltzmann_flux
