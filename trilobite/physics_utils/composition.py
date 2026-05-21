"""Simple utilities for physics calculations."""


# ============================================================= #
# Mean Molecular Weight Calculations                            #
# ============================================================= #
# These functions compute the mean molecular weight and its variants
# based on the mass fractions of hydrogen (X), helium (Y), and metals (Z).
# The calculations assume full ionization and are commonly used in
# astrophysical contexts.
def compute_mean_molecular_weight(
    X: float,
    Y: float,
    Z: float = 0.0,
):
    r"""
    Compute the mean molecular weight per particle assuming full ionization.

    Parameters
    ----------
    X : float
        Hydrogen mass fraction.
    Y : float
        Helium mass fraction.
    Z : float, optional
        Metal mass fraction. Default is 0.

    Returns
    -------
    float
        Mean molecular weight per particle :math:`\mu`.

    Notes
    -----
    The mean molecular weight is defined as the mass per free particle
    (ions + electrons) in units of the proton mass:

    .. math::

        \mu = \frac{\rho}{(n_i + n_e) m_p}.

    Assuming full ionization and a composition of hydrogen, helium,
    and metals, we have:

    - Hydrogen: 1 proton + 1 electron
    - Helium: 1 nucleus (mass 4) + 2 electrons
    - Metals: approximated as fully ionized with
      :math:`Z/A \approx 1/2`

    The resulting expression is

    .. math::

        \mu = \frac{1}{2X + \tfrac{3}{4}Y + \tfrac{1}{2}Z}.

    This reduces to the primordial expression when :math:`Z = 0`.

    Examples
    --------
    Solar composition (:math:`X=0.70, Y=0.28, Z=0.02`) gives
    :math:`\mu \approx 0.62`.
    """
    return 1.0 / (2.0 * X + 0.75 * Y + 0.5 * Z)


def compute_mean_molecular_weight_per_electron(
    X: float,
    Y: float,
    Z: float = 0.0,
):
    r"""
    Compute the mean molecular weight per free electron assuming full ionization.

    Parameters
    ----------
    X : float
        Hydrogen mass fraction.
    Y : float
        Helium mass fraction.
    Z : float, optional
        Metal mass fraction. Default is 0.

    Returns
    -------
    float
        Mean molecular weight per free electron :math:`\mu_e`.

    Notes
    -----
    The mean molecular weight per electron is defined as

    .. math::

        \mu_e = \frac{\rho}{n_e m_p}.

    For a fully ionized plasma consisting of hydrogen, helium,
    and metals:

    - Hydrogen contributes 1 electron per proton
    - Helium contributes 2 electrons per 4 baryons
    - Metals are approximated as contributing
      :math:`Z/A \approx 1/2`

    This yields

    .. math::

        \mu_e = \frac{1}{X + \tfrac{1}{2}Y + \tfrac{1}{2}Z}.

    This reduces to the primordial expression when :math:`Z = 0`.

    Examples
    --------
    Solar composition (:math:`X=0.70, Y=0.28, Z=0.02`) gives
    :math:`\mu_e \approx 1.17`.
    """
    return 1.0 / (X + 0.5 * Y + 0.5 * Z)


def compute_mean_molecular_weight_per_ion(
    X: float,
    Y: float,
    Z: float = 0.0,
):
    r"""
    Compute the mean molecular weight per ion assuming full ionization.

    Parameters
    ----------
    X : float
        Hydrogen mass fraction.
    Y : float
        Helium mass fraction.
    Z : float, optional
        Metal mass fraction. Default is 0.

    Returns
    -------
    float
        Mean molecular weight per ion :math:`\mu_i`.

    Notes
    -----
    The mean molecular weight per ion is defined as

    .. math::

        \mu_i = \frac{\rho}{n_i m_p}.

    For a fully ionized plasma:

    - Hydrogen contributes one ion per baryon
    - Helium contributes one ion per four baryons
    - Metals are approximated as one ion per
      :math:`A \approx 2Z`

    This gives

    .. math::

        \mu_i = \frac{1}{X + \tfrac{1}{4}Y + \tfrac{1}{2}Z}.

    Examples
    --------
    Solar composition (:math:`X=0.70, Y=0.28, Z=0.02`) gives
    :math:`\mu_i \approx 1.29`.
    """
    return 1.0 / (X + 0.25 * Y + 0.5 * Z)


def compute_mean_molecular_weight_primordial(hydrogen_fraction: float):
    r"""Compute the mean molecular weight assuming all non-hydrogen species are Helium.

    Parameters
    ----------
    hydrogen_fraction: float
        The relevant hydrogen fraction (:math:`\chi_H<1`).

    Returns
    -------
    float
        The mean molecular weight.

    Notes
    -----
    The mean molecular weight is the mass per particle in a fluid. Thus,

    .. math::

        \mu = \frac{\sum_k n_k m_k + (n_{k,e^-} m_{e})}{\sum_k n_k + n_{k,e^-}} \approx
         \frac{\sum_k n_k m_k}{\sum_k n_k + n_{k,e^-}},

    where :math:`k` denotes each species. In the most typical case where hydrogen and helium dominate the
    calculation, we have
    1 proton and 1 electron from the hydrogen and 1 He nucleus and 2 electrons for the helium. Thus, we have

    .. math::

        \mu = \frac{n_{\rm H} + 4n_{\rm He}}{2n_{\rm H} + 3n_{\rm He}}.

    If all non-hydrogen species are Helium, then :math:`n_{\rm He} = N(1-\chi_H)/4`, so

    .. math::

        \mu = \frac{1}{2\chi_H + (3/4)(1-\chi_H)}.

    """
    return 1.0 / (2.0 * hydrogen_fraction + 0.75 * (1.0 - hydrogen_fraction))


def compute_mean_molecular_weight_per_electron_primordial(hydrogen_fraction: float):
    r"""Compute the mean molecular weight (:math:`\mu`) per free electron for a given primordial Hydrogen fraction.

    Parameters
    ----------
    hydrogen_fraction: float
        The relevant hydrogen fraction (:math:`\chi_H<1`).

    Returns
    -------
    float
        The mean molecular weight.

    Notes
    -----
    The mean molecular weight per electron is

    .. math::

        \mu = \frac{\sum_k n_k m_k + (n_{k,e^-} m_{e})}{n_{k,e^-}} \approx \frac{\sum_k n_k m_k}{n_{k,e^-}},

    where :math:`k` denotes each species. In the most typical case where hydrogen and helium dominate the
    calculation, we have
    1 proton and 1 electron from the hydrogen and 1 He nucleus and 2 electrons for the helium. Thus, we have

    .. math::

        \mu = \frac{n_{\rm H} + 4n_{\rm He}}{1n_{\rm H} + 2n_{\rm He}}.

    If all non-hydrogen species are Helium, then :math:`n_{\rm He} = N(1-\chi_H)/4`, so

    .. math::

        \mu = \frac{1}{1\chi_H + (1/2)(1-\chi_H)}.

    """
    return 1 / (hydrogen_fraction + 0.5 * (1 - hydrogen_fraction))
