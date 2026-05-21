.. _synchrotron_core:

=====================================
Core Synchrotron Tools
=====================================

.. hint::

    See :ref:`synchrotron_theory` for a detailed overview of the theory of synchrotron radiation.

The :mod:`trilobite.radiation.synchrotron.core` module provides the foundational building
blocks for synchrotron radiation calculations in Trilobite. Unlike higher-level
modules, this module does **not** assume any particular electron distribution,
microphysical closure, or spectral regime. Instead, it exposes the minimal
theoretical ingredients needed to construct synchrotron emission models from
first principles.

The tools in this module fall into two categories:

1. **Fundamental synchrotron frequencies**: gyrofrequency, critical frequency,
   characteristic emission frequency, and their inverses.
2. **Synchrotron kernel functions**: :func:`~trilobite.radiation.synchrotron.core.compute_first_synchrotron_kernel`
   and :func:`~trilobite.radiation.synchrotron.core.compute_averaged_first_synchrotron_kernel` for direct evaluation
   of :math:`F(x)` and :math:`\bar{F}(x)`, backed by private log-space
   implementations consumed by
   :class:`~trilobite.radiation.synchrotron.SEDs.numerical.NumericalSynchrotronEngine`.

.. note::

    Population-averaged emissivities, absorption coefficients, and spectral flux
    densities are not implemented here. Use
    :class:`~trilobite.radiation.synchrotron.SEDs.numerical.NumericalSynchrotronEngine`
    for those quantities.

.. currentmodule:: trilobite.radiation.synchrotron.core

----

Frequencies
-----------

Synchrotron emission is governed by a small number of characteristic frequencies
associated with the motion of relativistic electrons in a magnetic field.

The Gyrofrequency
~~~~~~~~~~~~~~~~~

The **gyrofrequency** describes the frequency at which a charged particle orbits
magnetic field lines. For a relativistic electron of Lorentz factor :math:`\gamma`
in a magnetic field of strength :math:`B`:

.. math::

    \nu_g = \frac{e B}{m_e c \gamma}.

.. tab-set::

    .. tab-item:: High-Level API

        .. code-block:: python

            from trilobite.radiation.synchrotron.core import compute_gyrofrequency
            import astropy.units as u

            nu_g = compute_gyrofrequency(gamma=100, B=1*u.G)

    .. tab-item:: Low-Level API

        .. code-block:: python

            from trilobite.radiation.synchrotron.core import _optimized_compute_nu_gyro

            nu_g = _optimized_compute_nu_gyro(gamma=100, B=1.0)  # B in Gauss, returns Hz


The Critical Synchrotron Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **critical frequency** characterizes the peak of single-electron synchrotron
emission :footcite:p:`RybickiLightman`:

.. math::

    \nu_c = \frac{3 e B \sin\alpha\,\gamma^2}{4 \pi m_e c}.

Nearly all synchrotron power from a single electron is emitted within roughly an
order of magnitude of :math:`\nu_c`.

.. tab-set::

    .. tab-item:: High-Level API

        The pitch angle ``alpha`` is a plain float in radians (default :math:`\pi/2`):

        .. code-block:: python

            from trilobite.radiation.synchrotron.core import compute_nu_critical
            import astropy.units as u

            nu_c = compute_nu_critical(gamma=100, B=1*u.G)

    .. tab-item:: Low-Level API

        Pass ``sin_alpha`` directly rather than the angle:

        .. code-block:: python

            from trilobite.radiation.synchrotron.core import _optimized_compute_nu_critical

            nu_c = _optimized_compute_nu_critical(gamma=100, B=1.0, sin_alpha=1.0)


The Characteristic Emission Frequency and Its Inverse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`compute_synchrotron_frequency` adds optional **pitch-angle averaging**.
Setting ``pitch_average=True`` (the default) replaces :math:`\sin\alpha` with
:math:`\langle\sin\alpha\rangle = 2/\pi`, appropriate for an isotropic population:

.. tab-set::

    .. tab-item:: compute_synchrotron_frequency

        .. code-block:: python

            import numpy as np
            from trilobite.radiation.synchrotron.core import compute_synchrotron_frequency
            import astropy.units as u

            # Pitch-angle averaged (default)
            nu_m = compute_synchrotron_frequency(gamma=1e3, B=0.5*u.G)

            # Fixed pitch angle
            nu_m = compute_synchrotron_frequency(
                gamma=1e3, B=0.5*u.G,
                alpha=np.pi/4, pitch_average=False,
            )

    .. tab-item:: compute_synchrotron_gamma

        The inverse mapping is :func:`compute_synchrotron_gamma`:

        .. code-block:: python

            from trilobite.radiation.synchrotron.core import compute_synchrotron_gamma
            import astropy.units as u

            gamma = compute_synchrotron_gamma(nu=1e9*u.Hz, B=0.5*u.G)

----

Synchrotron Kernels
--------------------

The spectral shape of synchrotron emission from a single electron is encoded in
the **first synchrotron kernel** :footcite:p:`RybickiLightman`

.. math::

    F(x) = x \int_x^\infty K_{5/3}(z)\,dz,

where :math:`K_{5/3}` is the modified Bessel function of the second kind and
:math:`x = \nu / \nu_c`. When electron pitch angles are isotropically distributed,
the relevant quantity is the **pitch-angle-averaged kernel** :math:`\bar{F}(x)`
(see :ref:`synch_numerical_sed_theory` for the exact form).

Both are available as public functions:

- :func:`compute_first_synchrotron_kernel` for :math:`F(x)`
- :func:`compute_averaged_first_synchrotron_kernel` for :math:`\bar{F}(x)`

Both accept a ``method`` argument: ``"exact"`` (default) uses Bessel-function
quadrature in the interior stitched to asymptotic forms at the domain edges;
``"lu"`` uses a fast closed-form approximation accurate to within a few percent.

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    from trilobite.radiation.synchrotron.core import (
        compute_first_synchrotron_kernel,
        compute_averaged_first_synchrotron_kernel,
    )

    x = np.logspace(-3, 1, 300)
    F     = compute_first_synchrotron_kernel(x)
    F_avg = compute_averaged_first_synchrotron_kernel(x)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(x, F,     lw=2, label=r'$F(x)$')
    ax.loglog(x, F_avg, lw=2, label=r'$\bar{F}(x)$', ls='--')
    ax.set_xlabel(r'$x = \nu / \nu_c$', fontsize=12)
    ax.set_ylabel(r'Kernel value', fontsize=12)
    ax.set_title('Synchrotron kernel functions', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()

.. note::

    For performance-critical applications such as inference loops, prefer
    :class:`~trilobite.radiation.synchrotron.SEDs.numerical.NumericalSynchrotronEngine`,
    which pre-tabulates the kernel on a spline grid with
    :meth:`~trilobite.radiation.synchrotron.SEDs.numerical.NumericalSynchrotronEngine.load_first_kernel`
    and avoids repeated Bessel-function quadrature.

.. rubric:: References

.. footbibliography::
