"""Standard opacity law implementations for Triceratops."""

import numpy as np

from triceratops.radiation.opacity.base import GreyOpacityLaw

# ------------------------------------------------------------------ #
#  Standard Kramers normalisation constants                           #
# ------------------------------------------------------------------ #

#: Free-free Kramers :math:`\kappa_0` (solar composition, no Gaunt-factor correction).
KAPPA_FF_0: float = 3.68e22
#: Bound-free Kramers :math:`\kappa_0` (solar composition, no Gaunt-factor correction).
KAPPA_BF_0: float = 4.34e25
#: Combined (FF + BF) Kramers :math:`\kappa_0`.  Valid because both terms share the
#: same :math:`\rho` and :math:`T` power-law indices, so they add as a single
#: effective normalisation.
KAPPA_KR_0: float = KAPPA_FF_0 + KAPPA_BF_0


# ================================================================ #
#  Opacities                                                       #
# ================================================================ #
class ConstantOpacity(GreyOpacityLaw):
    r"""Opacity law with a fixed, user-specified :math:`\kappa`.

    Pure-Python (no Cython extension).

    Parameters
    ----------
    kappa : float
        Opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
    """

    def __init__(self, kappa: float):
        # Set _LOG_KAPPA as an instance attribute, shadowing the class default.
        # GreyOpacityLaw._log_opacity returns self._LOG_KAPPA, so this is all
        # that is needed to fix the constant.
        self._LOG_KAPPA = np.log(float(kappa))
        super().__init__()

    @property
    def kappa(self) -> float:
        r"""Opacity value in :math:`\mathrm{cm^2\,g^{-1}}`."""
        return float(np.exp(self._LOG_KAPPA))


class ElectronScatteringOpacity(GreyOpacityLaw):
    r"""Constant electron-scattering (Thomson) opacity.

    .. math::

        \kappa_{\rm es} = \frac{\sigma_T}{m_p}\,\frac{1+X}{2}
                        \approx 0.34\,\mathrm{cm^2\,g^{-1}}

    for fully ionised solar composition.

    Parameters
    ----------
    kappa_es : float, optional
        Electron-scattering opacity in :math:`\mathrm{cm^2\,g^{-1}}`.  Default 0.34.
    """

    IS_C_BACKED = True

    #: Exposed so callers can read back the value without going through the C object.
    kappa_es: float

    def __init__(self, kappa_es: float = 0.34):
        self.kappa_es = kappa_es
        # Keep _LOG_KAPPA in sync for introspection (not reached via _log_opacity
        # when IS_C_BACKED=True, but makes the object inspectable without Cython).
        self._LOG_KAPPA = np.log(kappa_es)
        super().__init__(kappa_es=kappa_es)

    def _initialize_C_object(self, kappa_es=0.34):
        from triceratops.radiation.opacity.models._electron_scattering import (
            C_ElectronScatteringOpacity,
        )

        # noinspection PyArgumentList
        return C_ElectronScatteringOpacity(kappa_es=kappa_es)


class KramersFFOpacity(GreyOpacityLaw):
    r"""Free-free (bremsstrahlung) Kramers opacity.

    .. math::

        \kappa_{\rm ff}(\rho, T) = \kappa_0\,\rho\,T^{-3.5}

    with the standard solar-composition default:

    .. math::

        \kappa_{\rm ff,0} \approx 3.68\times10^{22}\ {\rm cm^5\,g^{-2}\,K^{3.5}}

    Parameters
    ----------
    kappa0 : float, optional
        Normalisation constant in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.  Defaults to
        :data:`~triceratops.radiation.opacity.models.core.KAPPA_FF_0`.  Override with a composition- or
        Gaunt-factor-corrected value when available.
    """

    IS_C_BACKED = True

    def __init__(self, kappa0: float = KAPPA_FF_0):
        self.kappa0 = kappa0
        super().__init__(kappa0=kappa0)

    def _initialize_C_object(self, kappa0):
        from triceratops.radiation.opacity.models._kramers import C_KramersOpacity

        # noinspection PyArgumentList
        return C_KramersOpacity(kappa0=kappa0)


class KramersBFOpacity(GreyOpacityLaw):
    r"""Bound-free (photoionisation) Kramers opacity.

    .. math::

        \kappa_{\rm bf}(\rho, T) = \kappa_0\,\rho\,T^{-3.5}

    with the standard solar-composition default:

    .. math::

        \kappa_{\rm bf,0} \approx 4.34\times10^{25}\ {\rm cm^5\,g^{-2}\,K^{3.5}}

    Parameters
    ----------
    kappa0 : float, optional
        Normalisation constant in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.  Defaults to
        :data:`~triceratops.radiation.opacity.models.core.KAPPA_BF_0`.
    """

    IS_C_BACKED = True

    def __init__(self, kappa0: float = KAPPA_BF_0):
        self.kappa0 = kappa0
        super().__init__(kappa0=kappa0)

    def _initialize_C_object(self, kappa0):
        from triceratops.radiation.opacity.models._kramers import C_KramersOpacity

        # noinspection PyArgumentList
        return C_KramersOpacity(kappa0=kappa0)


class KramersOpacity(GreyOpacityLaw):
    r"""Combined (free-free + bound-free) Kramers opacity.

    Because both terms share the same power-law dependence on :math:`\rho` and
    :math:`T`, they combine into a single effective normalisation:

    .. math::

        \kappa_{\rm ff+bf}(\rho, T) = (\kappa_{\rm ff,0} + \kappa_{\rm bf,0})\,\rho\,T^{-3.5}

    with the standard solar-composition default:

    .. math::

        \kappa_{\rm ff+bf,0} \approx 4.34\times10^{25}\ {\rm cm^5\,g^{-2}\,K^{3.5}}

    (dominated by the bound-free term).

    Parameters
    ----------
    kappa0 : float, optional
        Combined normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.  Defaults to
        :data:`~triceratops.radiation.opacity.models.core.KAPPA_KR_0` = ``KAPPA_FF_0 + KAPPA_BF_0``.
    """

    IS_C_BACKED = True

    def __init__(self, kappa0: float = KAPPA_KR_0):
        self.kappa0 = kappa0
        super().__init__(kappa0=kappa0)

    def _initialize_C_object(self, kappa0):
        from triceratops.radiation.opacity.models._kramers import C_KramersOpacity

        # noinspection PyArgumentList
        return C_KramersOpacity(kappa0=kappa0)


# ================================================================ #
#  Kramers + electron-scattering (combined) opacities             #
# ================================================================ #


class KramersFFESOpacity(GreyOpacityLaw):
    r"""Free-free Kramers opacity plus electron scattering.

    .. math::

        \kappa(\rho, T) = \kappa_{\rm es} + \kappa_{\rm ff,0}\,\rho\,T^{-3.5}

    Parameters
    ----------
    kappa0 : float, optional
        Free-free normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
        Defaults to :data:`~triceratops.radiation.opacity.models.core.KAPPA_FF_0`.
    kappa_es : float, optional
        Electron-scattering opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
        Default 0.34.
    """

    IS_C_BACKED = True

    def __init__(self, kappa0: float = KAPPA_FF_0, kappa_es: float = 0.34):
        self.kappa0 = kappa0
        self.kappa_es = kappa_es
        super().__init__(kappa0=kappa0, kappa_es=kappa_es)

    def _initialize_C_object(self, kappa0, kappa_es=0.34):
        from triceratops.radiation.opacity.models._kramers_es import C_KramersESOpacity

        return C_KramersESOpacity(kappa0=kappa0, kappa_es=kappa_es)


class KramersBFESOpacity(GreyOpacityLaw):
    r"""Bound-free Kramers opacity plus electron scattering.

    .. math::

        \kappa(\rho, T) = \kappa_{\rm es} + \kappa_{\rm bf,0}\,\rho\,T^{-3.5}

    Parameters
    ----------
    kappa0 : float, optional
        Bound-free normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
        Defaults to :data:`~triceratops.radiation.opacity.models.core.KAPPA_BF_0`.
    kappa_es : float, optional
        Electron-scattering opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
        Default 0.34.
    """

    IS_C_BACKED = True

    def __init__(self, kappa0: float = KAPPA_BF_0, kappa_es: float = 0.34):
        self.kappa0 = kappa0
        self.kappa_es = kappa_es
        super().__init__(kappa0=kappa0, kappa_es=kappa_es)

    def _initialize_C_object(self, kappa0, kappa_es=0.34):
        from triceratops.radiation.opacity.models._kramers_es import C_KramersESOpacity

        return C_KramersESOpacity(kappa0=kappa0, kappa_es=kappa_es)


class KramersESOpacity(GreyOpacityLaw):
    r"""Combined (free-free + bound-free) Kramers opacity plus electron scattering.

    .. math::

        \kappa(\rho, T) = \kappa_{\rm es} + (\kappa_{\rm ff,0} + \kappa_{\rm bf,0})\,\rho\,T^{-3.5}

    The most physically complete grey opacity for a hot, fully ionised plasma
    where both Thomson scattering and thermal bremsstrahlung/photoionisation
    contribute.

    Parameters
    ----------
    kappa0 : float, optional
        Combined Kramers normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
        Defaults to :data:`~triceratops.radiation.opacity.models.core.KAPPA_KR_0` = ``KAPPA_FF_0 + KAPPA_BF_0``.
    kappa_es : float, optional
        Electron-scattering opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
        Default 0.34.
    """

    IS_C_BACKED = True

    def __init__(self, kappa0: float = KAPPA_KR_0, kappa_es: float = 0.34):
        self.kappa0 = kappa0
        self.kappa_es = kappa_es
        super().__init__(kappa0=kappa0, kappa_es=kappa_es)

    def _initialize_C_object(self, kappa0, kappa_es=0.34):
        from triceratops.radiation.opacity.models._kramers_es import C_KramersESOpacity

        return C_KramersESOpacity(kappa0=kappa0, kappa_es=kappa_es)
