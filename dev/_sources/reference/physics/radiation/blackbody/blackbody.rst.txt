.. _blackbody_user_guide:

====================
Blackbody Radiation
====================

The :mod:`trilobite.radiation.blackbody` module provides the Planck function and
related thermal-radiation utilities.  It follows the standard two-level API pattern
used throughout Trilobite: private ``_*_cgs`` kernels operate on plain NumPy arrays
with no unit overhead (suitable for MCMC hot loops), and public wrappers accept and
return :class:`~astropy.units.Quantity` objects for interactive use.

The module also provides a **UV-suppressed (modified) blackbody**, in which the flux below a cutoff wavelength is
suppressed. This is the standard description of the spectra of superluminous supernovae, and is documented below in
:ref:`blackbody_cutoff`.

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


.. _blackbody_cutoff:

UV-Suppressed (Modified) Blackbody
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optical and near-infrared colors of many supernovae are well described by a blackbody, but their ultraviolet
spectra are fainter than a blackbody would be, because of blended absorption lines and metal-line blanketing. For
superluminous supernovae the deficit is large and well characterized. A simple description, used by
:footcite:t:`Nicholl_2017` in the MOSFiT code, multiplies the blackbody by a transmission that falls linearly below a
cutoff wavelength :math:`\lambda_{\rm cut}` (about 3000 Angstrom for superluminous supernovae):

.. math::

    F_\lambda \propto B_\lambda(T)\,\min\left(1, \frac{\lambda}{\lambda_{\rm cut}}\right).

The transmission is 100% at the cutoff and zero at :math:`\lambda = 0`. Here :math:`\lambda` is the **rest-frame**
wavelength. The suppressed flux is not lost: the spectrum is renormalised so that it carries the same bolometric flux as
the blackbody, :math:`L = 4\pi R^2 \sigma_{\rm SB} T^4`, which means that the flux at longer wavelengths is
raised. The renormalisation is :math:`1/f(T)`, where

.. math::

    f(T) = \frac{\int B_\lambda \min(1, \lambda/\lambda_{\rm cut})\,d\lambda}{\int B_\lambda\,d\lambda}
    = 1 - \frac{15}{\pi^4}\sum_{n=1}^{\infty} e^{-n x_c}
    \left(\frac{x_c^2}{n^2} + \frac{4 x_c}{n^3} + \frac{6}{n^4}\right),
    \qquad x_c = \frac{h c}{\lambda_{\rm cut} k_{\rm B} T},

is the fraction of the blackbody flux that survives. The series is exact, converges geometrically, and is evaluated
with 256 terms, which is accurate to :math:`10^{-14}` for :math:`T \lesssim 3\times10^5` K at a cutoff of 3000 Angstrom. The
spectrum of the modified blackbody is therefore

.. math::

    F_\lambda = B_\lambda(T)\,\frac{\min(1, \lambda/\lambda_{\rm cut})}{f(T)}.

Three properties are useful to know:

- **The effect grows with temperature.** At 8000 K the surviving fraction is :math:`f = 0.97`, and the modified
  blackbody differs little from a blackbody; at 15000 K it is 0.82, and at 30000 K it is 0.52.
- **Only the far UV is dimmer.** Because of the renormalisation, the spectrum is *brighter* than the blackbody wherever
  :math:`\lambda/\lambda_{\rm cut} > f(T)`, which includes the whole range around and above the cutoff, by the factor
  :math:`1/f` at and above the cutoff wavelength.
- **The photosphere is unchanged.** The temperature and radius are still those of the Stefan-Boltzmann law. Only the
  spectrum emitted at them differs.

With a redshift, the suppression is applied at the rest-frame wavelength, :math:`\lambda' = \lambda_{\rm obs}/(1+z)`, and the
observed flux density is that of the blackbody, with the same factors, times :math:`(1+z)` in frequency or
:math:`1/(1+z)` in wavelength, which conserves the bolometric flux, :math:`L/4\pi D_L^2`.

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

.. _blackbody_cutoff_usage:

Observed Flux and Luminosity of a Spherical Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~trilobite.radiation.blackbody.planck_flux` gives the flux density observed from a spherical blackbody of
radius :math:`R`, and :func:`~trilobite.radiation.blackbody.planck_luminosity` its specific luminosity. Both take
the keyword ``lambda_cut`` to use the modified blackbody instead. It is the rest-frame cutoff wavelength; when it is
omitted, the spectrum is a plain blackbody.

.. code-block:: python

    import numpy as np
    import astropy.units as u
    from trilobite.radiation.blackbody import planck_flux, planck_luminosity

    T, R = 15000 * u.K, 5e15 * u.cm
    nu = np.geomspace(3e14, 3e15, 100) * u.Hz       # observer frame

    # Observed flux density, with and without the UV suppression
    plain = planck_flux(T, R, nu=nu, redshift=0.1)
    modified = planck_flux(T, R, nu=nu, redshift=0.1, lambda_cut=3000 * u.AA)

    # The specific luminosity of the source, in the rest frame
    lam = np.geomspace(1000, 10000, 100) * u.AA
    L_lam = planck_luminosity(T, R, lam=lam, lambda_cut=3000 * u.AA)

Both functions conserve the bolometric flux, so that integrating either spectrum over frequency gives the same
:math:`L/4\pi D_L^2` (and :math:`L = 4\pi R^2 \sigma_{\rm SB} T^4` for the luminosity).

The figure compares the two spectra of a 15000 K source.

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    from trilobite.radiation.blackbody import planck_luminosity, planck_cutoff_fraction

    T, R, cut = 15000 * u.K, 5e15 * u.cm, 3000 * u.AA
    lam = np.geomspace(800, 12000, 400) * u.AA

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lam.value, planck_luminosity(T, R, lam=lam).value, label="blackbody")
    ax.plot(lam.value, planck_luminosity(T, R, lam=lam, lambda_cut=cut).value, label="UV-suppressed")
    ax.axvline(cut.value, color="k", ls="--", alpha=0.4)
    ax.set_xlabel(r"Rest-frame wavelength [$\AA$]")
    ax.set_ylabel(r"$L_\lambda$ [erg s$^{-1}$ cm$^{-1}$]")
    ax.set_title(f"Surviving fraction f = {planck_cutoff_fraction(T, cut):.2f}")
    ax.legend()
    fig.tight_layout()

Surviving Fraction
~~~~~~~~~~~~~~~~~~

:func:`~trilobite.radiation.blackbody.planck_cutoff_fraction` returns the fraction :math:`f(T)` of the blackbody flux that
survives the suppression, which tells you how different the modified spectrum is from a blackbody:

.. code-block:: python

    import astropy.units as u
    from trilobite.radiation.blackbody import planck_cutoff_fraction

    planck_cutoff_fraction([8000, 15000, 30000] * u.K, 3000 * u.AA)
    # array([0.9747, 0.8162, 0.5223])

CGS Kernels
~~~~~~~~~~~

The CGS kernels take the cutoff as the keyword ``lambda_cut``, in **cm**, with 0 meaning no suppression:
:func:`~trilobite.radiation.blackbody._log_specific_flux_fnu_cgs`,
:func:`~trilobite.radiation.blackbody._log_specific_flux_flambda_cgs`,
:func:`~trilobite.radiation.blackbody._log_specific_luminosity_fnu_cgs` and
:func:`~trilobite.radiation.blackbody._log_specific_luminosity_flambda_cgs`. This is what the optical supernova models of
:ref:`arnett` call.

.. code-block:: python

    import numpy as np
    from trilobite.radiation.blackbody import _log_specific_flux_fnu_cgs

    log_nu = np.log(np.geomspace(3e14, 3e15, 100))     # observer frame, Hz
    log_fnu = _log_specific_flux_fnu_cgs(
        log_nu, np.log(1.5e4), np.log(5e15), np.log(1e27), z=0.1, lambda_cut=3.0e-5
    )

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
    :no-index:

.. autofunction:: trilobite.radiation.blackbody.planck_surface_flux
    :no-index:

.. autofunction:: trilobite.radiation.blackbody.planck_flux
    :no-index:

.. autofunction:: trilobite.radiation.blackbody.planck_luminosity
    :no-index:

.. autofunction:: trilobite.radiation.blackbody.planck_cutoff_fraction
    :no-index:

.. autofunction:: trilobite.radiation.blackbody.wien_peak_frequency
    :no-index:

.. autofunction:: trilobite.radiation.blackbody.wien_peak_wavelength
    :no-index:

.. autofunction:: trilobite.radiation.blackbody.stefan_boltzmann_flux
    :no-index:

----

.. footbibliography::
