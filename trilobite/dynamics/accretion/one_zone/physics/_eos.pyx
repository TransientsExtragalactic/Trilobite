#cython: language_level=3, boundscheck=False
r"""
Equation-of-state (EOS) primitives for one-zone accretion disk models (Cython).

Provides the low-level thermodynamic functions consumed by disk closure
functions.  Each function maps a temperature (and, where needed, disk geometry)
to the isothermal sound speed :math:`c_s`.  No closure pipeline logic lives
here — these are pure thermodynamic primitives.

Functions
---------
compute_ideal_gas_cs
    Analytic ideal-gas sound speed.  Takes only :math:`T` and :math:`\mu`.
compute_gas_rad_cs
    Gas + radiation pressure sound speed.  Requires :math:`\Sigma` and
    :math:`\Omega` because radiation pressure depends on density.

Adding a new EOS
----------------
Add a new ``cdef inline double compute_<regime>_cs(...)`` function here and
declare it in ``_eos.pxd``.  The closure functions in ``models/`` that use it
can then ``cimport`` the new primitive.

See Also
--------
:mod:`trilobite.dynamics.accretion.one_zone.closure` :
    Physical constants and struct definitions.
"""
from libc.math cimport sqrt

from ..closure cimport K_B_CGS, M_P_CGS, RAD_A_CGS


DEF SQRT_2PI = 2.506628274631000


cdef inline double compute_ideal_gas_cs(double T, double mu) nogil:
    r"""Isothermal sound speed for a pure ideal gas.

    .. math::

        c_s = \sqrt{\frac{k_B\,T}{\mu\,m_p}}

    Parameters
    ----------
    T : double
        Midplane temperature (K).
    mu : double
        Mean molecular weight (dimensionless).

    Returns
    -------
    double
        Sound speed (cm s⁻¹).
    """
    return sqrt(K_B_CGS * T / (mu * M_P_CGS))


cdef inline double compute_gas_rad_cs(
    double T, double mu, double Sigma, double Omega
) nogil:
    r"""Isothermal sound speed for a gas + radiation pressure EOS.

    Because the disk scale height :math:`H = c_s / \Omega` and
    :math:`\rho = \Sigma \Omega / (\sqrt{2\pi}\,c_s)`, the combined EOS

    .. math::

        c_s^2 = \frac{k_B T}{\mu m_p} + \frac{a T^4}{3 \rho}

    gives a quadratic in :math:`c_s`:

    .. math::

        c_s^2 - A\,c_s - B = 0,
        \qquad
        A = \frac{a T^4 \sqrt{2\pi}}{3\,\Sigma\,\Omega},
        \quad
        B = \frac{k_B T}{\mu m_p}.

    The positive physical root is returned using a numerically stable form.

    Parameters
    ----------
    T : double
        Midplane temperature (K).
    mu : double
        Mean molecular weight (dimensionless).
    Sigma : double
        Surface density (g cm⁻²).
    Omega : double
        Keplerian angular velocity (s⁻¹).

    Returns
    -------
    double
        Sound speed (cm s⁻¹).
    """
    cdef double A, B, D

    B = K_B_CGS * T / (mu * M_P_CGS)
    A = RAD_A_CGS * T * T * T * T * SQRT_2PI / (3.0 * Sigma * Omega)

    # Positive root of c_s^2 - A c_s - B = 0:  c_s = (A + sqrt(A^2 + 4B)) / 2.
    # A >= 0, B > 0 → no catastrophic cancellation.
    D = sqrt(A * A + 4.0 * B)
    return 0.5 * (A + D)
