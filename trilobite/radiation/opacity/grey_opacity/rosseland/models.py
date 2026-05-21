r"""Grey (frequency-averaged) opacity laws for use in radiative diffusion problems.

The Rosseland mean opacity is defined as

.. math::

    \frac{1}{\kappa_R} =
        \frac{\int (1/\kappa_\nu)\,(\partial B_\nu/\partial T)\,d\nu}
             {\int (\partial B_\nu/\partial T)\,d\nu}.

It governs radiative transport in optically thick media:

.. math::

    F_{\rm rad} =
        -\frac{4acT^3}{3\kappa_R\rho}\,\nabla T.

Analytic Opacity Laws
---------------------

Closed-form power-law fits suited for diffusive, fully ionised plasmas.
All analytic opacities are Cython-backed for performance.

+-----------------------------------------------+--------------------------------------------------------------+
| Class                                         | Description                                                  |
+===============================================+==============================================================+
| :class:`~trilobite.radiation.opacity.grey_  | Constant grey opacity:                                       |
| opacity.base.ConstantGreyOpacity`             | :math:`\kappa(\rho,T) = \kappa_0`.                           |
+-----------------------------------------------+--------------------------------------------------------------+
| :class:`ElectronScatteringOpacity`            | Thomson scattering floor:                                    |
|                                               | :math:`\kappa_{\rm es} \approx 0.34\,\mathrm{cm^2\,g^{-1}}`  |
|                                               | (solar composition).                                         |
+-----------------------------------------------+--------------------------------------------------------------+
| :class:`KramersFFOpacity`                     | Free-free opacity:                                           |
|                                               | :math:`\kappa \propto \rho\,T^{-3.5}`.                       |
+-----------------------------------------------+--------------------------------------------------------------+
| :class:`KramersBFOpacity`                     | Bound-free opacity:                                          |
|                                               | :math:`\kappa \propto \rho\,T^{-3.5}`.                       |
+-----------------------------------------------+--------------------------------------------------------------+
| :class:`KramersOpacity`                       | Combined free-free + bound-free opacity.                     |
+-----------------------------------------------+--------------------------------------------------------------+
| :class:`KramersFFESOpacity`                   | Free-free Kramers opacity with electron-scattering floor.    |
+-----------------------------------------------+--------------------------------------------------------------+
| :class:`KramersBFESOpacity`                   | Bound-free Kramers opacity with electron-scattering floor.   |
+-----------------------------------------------+--------------------------------------------------------------+
| :class:`KramersESOpacity`                     | Combined Kramers opacity with electron-scattering floor.     |
+-----------------------------------------------+--------------------------------------------------------------+

Tabulated Opacity Laws
----------------------

Numerical Rosseland mean tables for arbitrary compositions, including
metal-line blanketing that is inaccessible to simple power laws.

+--------------------------+-----------------------------------------------------------------+
| Class                    | Description                                                     |
+==========================+=================================================================+
| :class:`OPALOpacity`     | Bilinear interpolation on a 2-D OPAL table                      |
|                          | (:footcite:t:`2005ASPC..336...25A`).                            |
|                          |                                                                 |
|                          | Use :meth:`OPALOpacity.load_default` or                         |
|                          | :func:`~trilobite.radiation.opacity.utils.load_opal_opacity`  |
|                          | to construct instances from the bundled tables.                 |
+--------------------------+-----------------------------------------------------------------+

"""

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from trilobite.radiation.opacity.grey_opacity.base import GreyOpacityLaw

if TYPE_CHECKING:
    from types import SimpleNamespace

# ================================================================ #
#  Kramers normalisation constants                                 #
# ================================================================ #

#: Free-free Kramers coefficient for solar composition (no Gaunt-factor
#: correction), in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
KAPPA_FF_0: float = 3.68e22

#: Bound-free Kramers coefficient for solar composition (no Gaunt-factor
#: correction), in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
KAPPA_BF_0: float = 4.34e25

#: Combined (FF + BF) Kramers coefficient, equal to ``KAPPA_FF_0 + KAPPA_BF_0``.
#: Both terms share the same :math:`\rho\,T^{-3.5}` dependence, so they add
#: as a single effective normalisation.
KAPPA_KR_0: float = KAPPA_FF_0 + KAPPA_BF_0


# ================================================================ #
#  Analytic Rosseland mean opacity laws                            #
# ================================================================ #
class ElectronScatteringOpacity(GreyOpacityLaw):
    r"""Thomson electron-scattering opacity.

    Electron scattering is *grey* (frequency-independent) and provides a
    constant opacity floor throughout a fully ionised plasma.  The value
    follows from the Thomson cross-section :math:`\sigma_T` and the
    mean particle mass per free electron:

    .. math::

        \kappa_{\rm es}
            = \frac{\sigma_T}{m_{\rm H}}\,\frac{1+X}{2}
            \approx 0.34\ \mathrm{cm^2\,g^{-1}} \quad (X = 0.70,\ {\rm solar}),

    where :math:`X` is the hydrogen mass fraction.  This term dominates at
    high temperatures (:math:`T \gtrsim 10^7\,\mathrm{K}`) and low densities
    where bound–free and free–free cross-sections become negligible, for
    example in stellar coronae, hot accretion-disc surfaces, and the
    hydrogen-ionised outer layers of SNe~Ia.

    For pure helium (:math:`X=0`) the value drops to
    :math:`\approx 0.20\,\mathrm{cm^2\,g^{-1}}`; for hydrogen-rich
    ejecta (:math:`X \approx 0.70`) use the solar default.

    Parameters
    ----------
    kappa_es : float, optional
        Thomson opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
        Default ``0.34`` (solar composition, :math:`X = 0.70`).

    See Also
    --------
    ~trilobite.radiation.opacity.grey_opacity.base.ConstantGreyOpacity : Generic constant opacity without
     the electron-scattering context.
    KramersESOpacity : Adds a :math:`\rho T^{-3.5}` Kramers term on top of this floor.
    OPALOpacity : Tabulated opacity that includes Thomson scattering automatically.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from trilobite.radiation.opacity.grey_opacity.rosseland import (
            ElectronScatteringOpacity,
        )

        op_solar = (
            ElectronScatteringOpacity()
        )  # κ = 0.34 cm² g⁻¹, X = 0.70
        op_he = ElectronScatteringOpacity(
            0.20
        )  # helium-dominated ejecta
        kap = op_solar.opacity(
            1e-5 * u.g / u.cm**3, 1e7 * u.K
        )
        # → 0.34 cm² g⁻¹ (independent of ρ and T)
    """

    IS_C_BACKED = True
    mean_type = "rosseland"

    #: Exposed so callers can read back the value without accessing the C companion.
    kappa_es: float

    def __init__(self, kappa_es: float = 0.34):
        self.kappa_es = kappa_es
        super().__init__(kappa_es=kappa_es)

    def _initialize_C_object(self, kappa_es=0.34):
        from trilobite.radiation.opacity.grey_opacity.rosseland._electron_scattering import (
            C_ElectronScatteringOpacity,
        )

        # noinspection PyArgumentList
        return C_ElectronScatteringOpacity(kappa_es=kappa_es)


class KramersFFOpacity(GreyOpacityLaw):
    r"""Free-free (bremsstrahlung) Kramers opacity.

    The free-free Rosseland mean opacity follows the classical Kramers scaling

    .. math::

        \kappa_{\rm ff}(\rho, T)
            = \kappa_{\rm ff,0}\,\rho\,T^{-3.5},

    where composition factors :math:`(1+X)` are absorbed into the normalisation
    :attr:`kappa0`.  This is the dominant grey opacity source at
    :math:`T \lesssim 10^7\,\mathrm{K}` in hot, diffuse, fully ionised
    gas where there are few bound electrons for photoionisation (e.g. hot
    ISM, stellar coronae, accretion disc corona).  Below
    :math:`\sim\!10^4\,\mathrm{K}` recombination invalidates the fully
    ionised assumption and bound-free opacity grows rapidly.

    The steep :math:`T^{-3.5}` dependence means this term falls sharply at
    high temperatures and is usually sub-dominant to electron scattering
    above :math:`\sim\!10^8\,\mathrm{K}`.

    Parameters
    ----------
    kappa0 : float, optional
        Free-free normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
        Default ``3.68e22`` (solar composition, :math:`X = 0.70`, no
        Gaunt-factor correction; :footcite:t:`Rybicki1986`).

    See Also
    --------
    KramersBFOpacity : Bound-free counterpart; larger by ~3 orders of magnitude.
    KramersOpacity : Combined FF + BF (use when both mechanisms are active).
    KramersFFESOpacity : Adds an electron-scattering floor for high-:math:`T` accuracy.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from trilobite.radiation.opacity.grey_opacity.rosseland import (
            KramersFFOpacity,
        )

        op = (
            KramersFFOpacity()
        )  # κ₀ = 3.68 × 10²² cm⁵ g⁻² K³·⁵
        op_custom = KramersFFOpacity(
            kappa0=1.0e22
        )  # non-solar composition
        kap = op.opacity(1e-5 * u.g / u.cm**3, 1e6 * u.K)
    """

    IS_C_BACKED = True
    mean_type = "rosseland"

    def __init__(self, kappa0: float = KAPPA_FF_0):
        self.kappa0 = kappa0
        super().__init__(kappa0=kappa0)

    def _initialize_C_object(self, kappa0):
        from trilobite.radiation.opacity.grey_opacity.rosseland._kramers import C_KramersOpacity

        # noinspection PyArgumentList
        return C_KramersOpacity(kappa0=kappa0)


class KramersBFOpacity(GreyOpacityLaw):
    r"""Bound-free (photoionisation) Kramers opacity.

    The bound-free Rosseland mean follows the same :math:`\rho\,T^{-3.5}`
    power law as the free-free term but with a significantly larger coefficient:

    .. math::

        \kappa_{\rm bf}(\rho, T)
            = \kappa_{\rm bf,0}\,\rho\,T^{-3.5},

    where composition factors :math:`Z\,(1+X)` are absorbed into :attr:`kappa0`.
    For solar composition :math:`\kappa_{\rm bf,0} \approx 4.34 \times 10^{25}`, roughly
    three orders of magnitude above the free-free coefficient, so bound-free
    dominates the Kramers opacity in most stellar-envelope conditions
    (:math:`T \sim 10^4\text{–}10^6\,\mathrm{K}`).

    Use this class when photoionisation is the primary opacity mechanism and
    electron scattering is negligible (low temperature, high density).

    Parameters
    ----------
    kappa0 : float, optional
        Bound-free normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
        Default ``4.34e25`` (solar composition, :math:`X = 0.70`, :math:`Z = 0.02`,
        no Gaunt-factor correction; :footcite:t:`Rybicki1986`).

    See Also
    --------
    KramersFFOpacity : Free-free counterpart; smaller by ~3 orders of magnitude.
    KramersOpacity : Combined FF + BF (preferred unless isolation is needed).
    KramersBFESOpacity : Adds an electron-scattering floor for high-:math:`T` accuracy.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from trilobite.radiation.opacity.grey_opacity.rosseland import (
            KramersBFOpacity,
        )

        op = (
            KramersBFOpacity()
        )  # κ₀ = 4.34 × 10²⁵ cm⁵ g⁻² K³·⁵
        op_lowZ = KramersBFOpacity(
            kappa0=1.0e24
        )  # metal-poor composition
        kap = op.opacity(1e-5 * u.g / u.cm**3, 1e6 * u.K)
    """

    IS_C_BACKED = True
    mean_type = "rosseland"

    def __init__(self, kappa0: float = KAPPA_BF_0):
        self.kappa0 = kappa0
        super().__init__(kappa0=kappa0)

    def _initialize_C_object(self, kappa0):
        from trilobite.radiation.opacity.grey_opacity.rosseland._kramers import C_KramersOpacity

        # noinspection PyArgumentList
        return C_KramersOpacity(kappa0=kappa0)


class KramersOpacity(GreyOpacityLaw):
    r"""Combined free-free and bound-free Kramers opacity.

    Both the free-free and bound-free Rosseland terms share the same
    :math:`\rho\,T^{-3.5}` scaling, so they add to a single effective
    normalisation:

    .. math::

        \kappa_{\rm Kramers}(\rho, T)
            = (\kappa_{\rm ff,0} + \kappa_{\rm bf,0})\,\rho\,T^{-3.5}
            \approx 4.38 \times 10^{25}\,\rho\,T^{-3.5}
            \quad (\mathrm{solar\ composition}),

    where the bound-free coefficient (:math:`4.34 \times 10^{25}`) dominates
    over the free-free coefficient (:math:`3.68 \times 10^{22}`) by roughly
    three orders of magnitude.  This is the standard grey Kramers opacity
    for fully ionised stellar envelopes where electron scattering is not yet
    important (:math:`T \lesssim 10^7\,\mathrm{K}`,
    :math:`\rho \gtrsim 10^{-6}\,\mathrm{g\,cm^{-3}}`).

    Parameters
    ----------
    kappa0 : float, optional
        Combined normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
        Default ``4.38e25`` (:math:`\kappa_{\rm ff,0} + \kappa_{\rm bf,0}`,
        solar composition; :footcite:t:`Rybicki1986`).

    See Also
    --------
    KramersFFOpacity : Free-free contribution in isolation.
    KramersBFOpacity : Bound-free contribution in isolation.
    KramersESOpacity : Adds an electron-scattering floor — preferred for :math:`T \gtrsim 10^6\,\mathrm{K}`.
    OPALOpacity : Full numerical Rosseland mean including line blanketing.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from trilobite.radiation.opacity.grey_opacity.rosseland import (
            KramersOpacity,
        )

        op = (
            KramersOpacity()
        )  # κ₀ = 4.38 × 10²⁵ cm⁵ g⁻² K³·⁵
        op_custom = KramersOpacity(
            kappa0=2.0e25
        )  # non-solar composition
        kap = op.opacity(1e-5 * u.g / u.cm**3, 1e6 * u.K)
    """

    IS_C_BACKED = True
    mean_type = "rosseland"

    def __init__(self, kappa0: float = KAPPA_KR_0):
        self.kappa0 = kappa0
        super().__init__(kappa0=kappa0)

    def _initialize_C_object(self, kappa0):
        from trilobite.radiation.opacity.grey_opacity.rosseland._kramers import C_KramersOpacity

        # noinspection PyArgumentList
        return C_KramersOpacity(kappa0=kappa0)


class KramersFFESOpacity(GreyOpacityLaw):
    r"""Free-free Kramers opacity with an electron-scattering floor.

    Combines a frequency-independent Thomson floor with the free-free Kramers
    term:

    .. math::

        \kappa(\rho, T)
            = \kappa_{\rm es}
            + \kappa_{\rm ff,0}\,\rho\,T^{-3.5}.

    This is physically appropriate in hot, diffuse, fully ionised gas where
    electron scattering and bremsstrahlung both contribute to the Rosseland
    mean (e.g. Wolf-Rayet winds, diffuse accretion disc coronae,
    :math:`T \sim 10^7\text{–}10^9\,\mathrm{K}`).  At high temperatures
    the constant :math:`\kappa_{\rm es}` term dominates; at lower temperatures
    and higher densities the :math:`\rho T^{-3.5}` term takes over.

    Prefer :class:`KramersESOpacity` in environments where bound-free
    opacity is also significant (most stellar-structure contexts).

    Parameters
    ----------
    kappa0 : float, optional
        Free-free normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
        Default ``3.68e22`` (solar composition, :math:`X = 0.70`).
    kappa_es : float, optional
        Electron-scattering opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
        Default ``0.34`` (solar composition, fully ionised).

    See Also
    --------
    KramersBFESOpacity : Bound-free variant with the same floor structure.
    KramersESOpacity : Preferred combined FF + BF form including electron scattering.
    ElectronScatteringOpacity : Thomson floor alone, without the Kramers term.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from trilobite.radiation.opacity.grey_opacity.rosseland import (
            KramersFFESOpacity,
        )

        op = (
            KramersFFESOpacity()
        )  # κ_es = 0.34, κ₀ = 3.68 × 10²² cm⁵ g⁻² K³·⁵
        op_he = KramersFFESOpacity(
            kappa_es=0.20
        )  # helium-dominated composition
        kap = op.opacity(1e-5 * u.g / u.cm**3, 1e7 * u.K)
    """

    IS_C_BACKED = True
    mean_type = "rosseland"

    def __init__(self, kappa0: float = KAPPA_FF_0, kappa_es: float = 0.34):
        self.kappa0 = kappa0
        self.kappa_es = kappa_es
        super().__init__(kappa0=kappa0, kappa_es=kappa_es)

    def _initialize_C_object(self, kappa0, kappa_es=0.34):
        from trilobite.radiation.opacity.grey_opacity.rosseland._kramers_es import C_KramersESOpacity

        return C_KramersESOpacity(kappa0=kappa0, kappa_es=kappa_es)


class KramersBFESOpacity(GreyOpacityLaw):
    r"""Bound-free Kramers opacity with an electron-scattering floor.

    Combines a Thomson scattering floor with the bound-free Kramers term:

    .. math::

        \kappa(\rho, T)
            = \kappa_{\rm es}
            + \kappa_{\rm bf,0}\,\rho\,T^{-3.5}.

    This is appropriate for stellar envelopes spanning a wide temperature
    range (:math:`T \sim 10^4\text{–}10^7\,\mathrm{K}`) where
    photoionisation is the dominant Kramers contribution but the electron
    scattering floor becomes non-negligible in the hotter, more diffuse
    upper layers.  The bound-free coefficient (``4.34e25``) is large enough
    that the free-free term (:class:`KramersFFOpacity`, ``3.68e22``) is
    essentially irrelevant, making this a good single-mechanism approximation
    to the full Kramers + electron-scattering model.

    Prefer :class:`KramersESOpacity` when you want both FF and BF included.

    Parameters
    ----------
    kappa0 : float, optional
        Bound-free normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
        Default ``4.34e25`` (solar composition, :math:`X = 0.70`,
        :math:`Z = 0.02`).
    kappa_es : float, optional
        Electron-scattering opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
        Default ``0.34`` (solar composition, fully ionised).

    See Also
    --------
    KramersFFESOpacity : Free-free variant of the same floor structure.
    KramersESOpacity : Combined FF + BF form — preferred in most contexts.
    OPALOpacity : Full numerical opacity including metal-line blanketing.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from trilobite.radiation.opacity.grey_opacity.rosseland import (
            KramersBFESOpacity,
        )

        op = (
            KramersBFESOpacity()
        )  # κ_es = 0.34, κ₀ = 4.34 × 10²⁵ cm⁵ g⁻² K³·⁵
        op_lowZ = KramersBFESOpacity(
            kappa0=1.0e24, kappa_es=0.20
        )  # metal-poor, helium-rich
        kap = op.opacity(1e-5 * u.g / u.cm**3, 1e6 * u.K)
    """

    IS_C_BACKED = True
    mean_type = "rosseland"

    def __init__(self, kappa0: float = KAPPA_BF_0, kappa_es: float = 0.34):
        self.kappa0 = kappa0
        self.kappa_es = kappa_es
        super().__init__(kappa0=kappa0, kappa_es=kappa_es)

    def _initialize_C_object(self, kappa0, kappa_es=0.34):
        from trilobite.radiation.opacity.grey_opacity.rosseland._kramers_es import C_KramersESOpacity

        return C_KramersESOpacity(kappa0=kappa0, kappa_es=kappa_es)


class KramersESOpacity(GreyOpacityLaw):
    r"""Combined Kramers opacity with an electron-scattering floor.

    The most complete grey analytic opacity for a hot, fully ionised plasma.
    It sums a Thomson scattering floor with both free-free and bound-free
    Kramers contributions:

    .. math::

        \kappa(\rho, T)
            = \kappa_{\rm es}
            + (\kappa_{\rm ff,0} + \kappa_{\rm bf,0})\,\rho\,T^{-3.5}
            \approx 0.34 + 4.38 \times 10^{25}\,\rho\,T^{-3.5}
            \quad (\mathrm{solar\ composition}).

    This is the standard choice for a self-consistent grey opacity across the
    full range of stellar interior conditions:

    - **High** :math:`T`, **low** :math:`\rho`: electron scattering dominates
      (:math:`\kappa \to 0.34\,\mathrm{cm^2\,g^{-1}}`).
    - **Low** :math:`T`, **high** :math:`\rho`: Kramers term dominates.

    The transition between regimes occurs near
    :math:`\rho\,T^{-3.5} \sim 7.8 \times 10^{-27}\,\mathrm{g\,cm^{-3}\,K^{3.5}}`.

    Parameters
    ----------
    kappa0 : float, optional
        Combined Kramers normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
        Default ``4.38e25`` (:math:`\kappa_{\rm ff,0} + \kappa_{\rm bf,0}`,
        solar composition).
    kappa_es : float, optional
        Electron-scattering opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
        Default ``0.34`` (solar composition, :math:`X = 0.70`, fully ionised).

    See Also
    --------
    KramersFFESOpacity : Free-free only — use when BF is negligible.
    KramersBFESOpacity : Bound-free only — nearly equivalent to this class.
    OPALOpacity : Full numerical Rosseland mean with line blanketing; preferred
                  when accuracy matters near the iron opacity bump.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from trilobite.radiation.opacity.grey_opacity.rosseland import (
            KramersESOpacity,
        )

        op = KramersESOpacity()  # solar defaults
        op_he = KramersESOpacity(
            kappa_es=0.20, kappa0=2.0e25
        )  # helium-rich ejecta
        kap = op.opacity(1e-5 * u.g / u.cm**3, 1e7 * u.K)
    """

    IS_C_BACKED = True
    mean_type = "rosseland"

    def __init__(self, kappa0: float = KAPPA_KR_0, kappa_es: float = 0.34):
        self.kappa0 = kappa0
        self.kappa_es = kappa_es
        super().__init__(kappa0=kappa0, kappa_es=kappa_es)

    def _initialize_C_object(self, kappa0, kappa_es=0.34):
        from trilobite.radiation.opacity.grey_opacity.rosseland._kramers_es import C_KramersESOpacity

        return C_KramersESOpacity(kappa0=kappa0, kappa_es=kappa_es)


# ================================================================ #
#  Table-based Rosseland mean opacity                              #
# ================================================================ #

#: Out-of-bounds mode codes passed to C_OPALTableOpacity.
_OOB_MODES = {"raise": 0, "clamp": 1, "nan": 2}


class OPALOpacity(GreyOpacityLaw):
    r"""Rosseland mean opacity from bilinear interpolation on a 2-D OPAL table.

    The OPAL tables (:footcite:t:`Iglesias1996`) provide numerically computed
    Rosseland mean opacities on a grid of :math:`\log_{10}T` and
    :math:`\log_{10}R`, where

    .. math::

        R = \frac{\rho}{\,T_6^3\,},
        \quad T_6 = 10^{-6}\,T\,[\mathrm{K}],

    covering temperatures :math:`3.75 \le \log_{10}T \le 8.7` and a wide
    density range.  They include contributions from electron scattering,
    free-free, bound-free, and bound-bound (line) transitions, making them
    substantially more accurate than analytic Kramers fits at intermediate
    temperatures where iron-group line blanketing is important.

    This class delegates all arithmetic to a Cython extension that performs
    bilinear interpolation in :math:`(\log_{10}T, \log_{10}R)` space.

    The bundled OPAL table set (Asplund & Grevesse 2005, 126 compositions) is
    loaded lazily on first access via :meth:`get_tables` and cached for the
    lifetime of the process.  Use :meth:`find` or :meth:`where` to identify
    composition indices, and :meth:`load_default` to construct an instance.

    Parameters
    ----------
    grid_T : array_like, shape (n1,)
        :math:`\log_{10}(T\,[\mathrm{K}])` grid values (strictly increasing).
    grid_R : array_like, shape (n2,)
        :math:`\log_{10}(R)` grid values (strictly increasing).
    kappa : array_like, shape (n1, n2)
        :math:`\log_{10}(\kappa\,[\mathrm{cm^2\,g^{-1}}])`.
        ``NaN`` marks out-of-range cells.
    out_of_bounds : {'raise', 'clamp', 'nan'}, optional
        Behaviour when a query :math:`(T, \rho)` falls outside the table
        grid or lands on a NaN-flagged cell.  Default ``'raise'``.

    See Also
    --------
    load_opal_opacity : Convenience loader for the bundled table.
    KramersESOpacity : Analytic alternative for rough estimates.

    Examples
    --------
    .. code-block:: python

        from trilobite.radiation.opacity import (
            get_opacity,
            load_opal_opacity,
        )
        from trilobite.radiation.opacity.grey_opacity.rosseland import (
            OPALOpacity,
        )
        import astropy.units as u

        # Via the registry (solar default)
        op = get_opacity("opal")

        # By composition
        op = load_opal_opacity(X=0.70, Z=0.02)

        # Inspect available compositions
        idx = OPALOpacity.find(X=0.70, Z=0.02)
        hits = OPALOpacity.where(Z=lambda z: z > 0.05)

        kap = op.opacity(1e-5 * u.g / u.cm**3, 1e7 * u.K)
        dkdT = op.dlogkappa_dlogT(
            1e-5 * u.g / u.cm**3, 1e7 * u.K
        )
    """

    IS_C_BACKED = True
    mean_type = "rosseland"

    #: Path to the bundled OPAL table HDF5 file.
    _PATH: ClassVar[Path] = Path(__file__).parent.parent.parent / "tables" / "asplund_grevesse_05.h5"
    #: Lazily populated namespace; ``None`` until first :meth:`get_tables` call.
    _TABLES: ClassVar = None

    # ------------------------------------------------------------------ #
    # Class-level table access                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def _read_h5(cls, path) -> "SimpleNamespace":
        """Read the OPAL HDF5 file at *path* into a SimpleNamespace of arrays."""
        from types import SimpleNamespace

        import h5py

        with h5py.File(path, "r") as f:
            ns = SimpleNamespace(
                grid_T=f["grid/grid_1"][:],
                grid_R=f["grid/grid_2"][:],
                opacity=f["opacity"][:],
                X=f["metadata/X"][:],
                Y=f["metadata/Y"][:],
                Z=f["metadata/Z"][:],
            )
        for arr in vars(ns).values():
            arr.flags.writeable = False
        return ns

    @classmethod
    def get_tables(cls, path=None) -> "SimpleNamespace":
        r"""Return the full OPAL table set, loading from disk on first call.

        The result is cached on the class after the first load (for the
        bundled file).  Passing a custom *path* bypasses the cache and reads
        fresh every time.

        Parameters
        ----------
        path : str or Path, optional
            Path to an HDF5 file with the same schema as the bundled table.
            If ``None``, the bundled Asplund & Grevesse (2005) file is used.

        Returns
        -------
        SimpleNamespace
            Read-only namespace with arrays:

            * ``grid_T``  — :math:`\log_{10}(T\,[\mathrm{K}])`, shape ``(70,)``
            * ``grid_R``  — :math:`\log_{10}(R)`, shape ``(19,)``
            * ``opacity`` — :math:`\log_{10}(\kappa)`, shape ``(126, 70, 19)``; ``NaN`` = invalid
            * ``X``, ``Y``, ``Z`` — composition fractions, shape ``(126,)``
        """
        if path is not None:
            return cls._read_h5(path)
        if cls._TABLES is None:
            cls._TABLES = cls._read_h5(cls._PATH)
        return cls._TABLES

    @classmethod
    def find(cls, *, X: float, Z: float, tol: float = 1e-6, table_path=None) -> int:
        r"""Return the unique composition index matching *(X, Z)*.

        Parameters
        ----------
        X : float
            Hydrogen mass fraction.
        Z : float
            Metal mass fraction.
        tol : float, optional
            Absolute tolerance for composition matching.  Default ``1e-6``.
        table_path : str or Path, optional
            Custom table file; defaults to the bundled table.

        Returns
        -------
        int
            Zero-based index into the table.

        Raises
        ------
        ValueError
            If no table matches or more than one table matches.

        Examples
        --------
        .. code-block:: python

            from trilobite.radiation.opacity.grey_opacity.rosseland import (
                OPALOpacity,
            )

            idx = OPALOpacity.find(X=0.70, Z=0.02)  # → 72
        """
        t = cls.get_tables(table_path)
        hits = np.where((np.abs(t.X - X) < tol) & (np.abs(t.Z - Z) < tol))[0]
        if len(hits) == 0:
            raise ValueError(f"No OPAL table found for X={X}, Z={Z}.")
        if len(hits) > 1:
            raise ValueError(f"Multiple OPAL tables match X={X}, Z={Z}: indices {hits.tolist()}.")
        return int(hits[0])

    @classmethod
    def where(cls, *, table_path=None, **conditions) -> list:
        r"""Return indices of tables satisfying all composition conditions.

        Parameters
        ----------
        table_path : str or Path, optional
            Custom table file; defaults to the bundled table.
        **conditions
            Each key must be one of ``"X"``, ``"Y"``, ``"Z"``.  The value
            may be:

            * a **scalar float** — matched within absolute tolerance ``1e-6``;
            * a **callable** ``f(arr) -> bool array`` — applied element-wise.

        Returns
        -------
        list of int
            Zero-based indices of matching tables.  Empty list if none match.

        Examples
        --------
        .. code-block:: python

            from trilobite.radiation.opacity.grey_opacity.rosseland import (
                OPALOpacity,
            )

            # All tables with Z = 0.02
            OPALOpacity.where(Z=0.02)

            # All metal-enriched tables
            OPALOpacity.where(Z=lambda z: z > 0.05)

            # Intersection of two conditions
            OPALOpacity.where(X=0.70, Z=0.02)
        """
        t = cls.get_tables(table_path)
        arrays = {"X": t.X, "Y": t.Y, "Z": t.Z}
        mask = np.ones(len(t.X), dtype=bool)
        for key, val in conditions.items():
            if key not in arrays:
                raise KeyError(f"Unknown composition key {key!r}. Valid keys: X, Y, Z.")
            arr = arrays[key]
            if callable(val):
                mask &= np.asarray(val(arr), dtype=bool)
            else:
                mask &= np.abs(arr - float(val)) <= 1e-6
        return list(np.where(mask)[0])

    # ------------------------------------------------------------------ #
    # Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        grid_T,
        grid_R,
        kappa,
        *,
        out_of_bounds: str = "raise",
    ) -> None:
        if out_of_bounds not in _OOB_MODES:
            raise ValueError(f"out_of_bounds must be one of {list(_OOB_MODES)}, got {out_of_bounds!r}.")
        self.out_of_bounds = out_of_bounds
        super().__init__(grid_T=grid_T, grid_R=grid_R, kappa=kappa, out_of_bounds=out_of_bounds)

    def _initialize_C_object(self, *, grid_T, grid_R, kappa, out_of_bounds):
        from trilobite.radiation.opacity.grey_opacity.rosseland._opal_table import C_OPALTableOpacity

        return C_OPALTableOpacity(
            np.array(grid_T, dtype=np.float64),
            np.array(grid_R, dtype=np.float64),
            np.array(kappa, dtype=np.float64, order="C"),
            _OOB_MODES[out_of_bounds],
        )

    @classmethod
    def load_default(
        cls,
        *,
        index: int = None,
        X: float = None,
        Z: float = None,
        table_path=None,
        out_of_bounds: str = "raise",
    ) -> "OPALOpacity":
        r"""Load an :class:`OPALOpacity` from the bundled OPAL table.

        Parameters
        ----------
        index : int, optional
            Zero-based composition index into the table.  Mutually exclusive
            with *X* / *Z*.
        X : float, optional
            Hydrogen mass fraction.  Must be paired with *Z*.
        Z : float, optional
            Metal mass fraction.  Must be paired with *X*.
        table_path : str or Path, optional
            Override the default bundled table path.
        out_of_bounds : {'raise', 'clamp', 'nan'}, optional
            Out-of-bounds behaviour.  Default ``'raise'``.

        Returns
        -------
        OPALOpacity

        Notes
        -----
        With no arguments, solar composition (X = 0.70, Z = 0.02) is used.
        """
        if index is not None and (X is not None or Z is not None):
            raise ValueError("Supply either 'index' or ('X', 'Z'), not both.")
        if (X is None) != (Z is None):
            raise ValueError("Supply both 'X' and 'Z' together, or neither.")
        t = cls.get_tables(table_path)
        if index is None:
            if X is not None and Z is not None:
                index = cls.find(X=X, Z=Z, table_path=table_path)
            else:
                index = cls.find(X=0.70, Z=0.02, table_path=table_path)
        return cls(t.grid_T, t.grid_R, t.opacity[index], out_of_bounds=out_of_bounds)

    # ------------------------------------------------------------------ #
    # Log-space interface (dispatches to Cython)                         #
    # ------------------------------------------------------------------ #

    def _log_opacity(self, log_T, log_rho):
        r"""Return :math:`\ln\kappa` via bilinear interpolation in the table."""
        if np.ndim(log_T) > 0:
            return self._c_object.log_opacity_array(
                np.ascontiguousarray(log_T, dtype=np.float64),
                np.ascontiguousarray(log_rho, dtype=np.float64),
            )
        return self._c_object.log_opacity(log_T, log_rho)

    def _dlogkappa_dlogrho(self, log_T, log_rho):
        r"""Return :math:`\partial\ln\kappa/\partial\ln\rho` from the table gradient."""
        if np.ndim(log_T) > 0:
            return self._c_object.dlogkappa_dlogrho_array(
                np.ascontiguousarray(log_T, dtype=np.float64),
                np.ascontiguousarray(log_rho, dtype=np.float64),
            )
        return self._c_object.dlogkappa_dlogrho(log_T, log_rho)

    def _dlogkappa_dlogT(self, log_T, log_rho):
        r"""Return :math:`\partial\ln\kappa/\partial\ln T` from the table gradient."""
        if np.ndim(log_T) > 0:
            return self._c_object.dlogkappa_dlogT_array(
                np.ascontiguousarray(log_T, dtype=np.float64),
                np.ascontiguousarray(log_rho, dtype=np.float64),
            )
        return self._c_object.dlogkappa_dlogT(log_T, log_rho)

    # ------------------------------------------------------------------ #
    # Text-file conversion utility                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def read_from_opal_txt(cls, txt_path, h5_path) -> None:
        r"""Parse an OPAL fixed-width text file and write it to HDF5.

        This classmethod is a provenance utility for (re-)generating the
        bundled ``asplund_grevesse_05.h5`` from the original source text.
        It does **not** construct an :class:`OPALOpacity` instance.

        Parameters
        ----------
        txt_path : str or Path
            Path to the OPAL-format fixed-width text file.
        h5_path : str or Path
            Output HDF5 path (overwritten if it exists).
        """
        import re as _re

        import h5py

        txt_path = Path(txt_path)
        h5_path = Path(h5_path)

        _INVALID = 9.999
        lines = txt_path.read_text().splitlines()

        data_start = None
        for i, line in enumerate(lines):
            if _re.match(r"^\*+\s+Tables\s+\*+", line.strip()):
                data_start = i + 1
                break
        if data_start is None:
            raise ValueError(f"Could not find '*** Tables ***' separator in {txt_path}.")

        table_hdr = _re.compile(
            r"TABLE\s*#\s*\d+\s+(\S+)\s+X=([0-9.]+)\s+Y=([0-9.]+)\s+"
            r"Z=([0-9.]+)\s+dX1=([0-9.]+)\s+dX2=([0-9.]+)"
        )
        col_hdr = _re.compile(r"logT\s+(.*)")

        blocks = []
        log_r_values = log_t_values = None
        i = data_start
        n_lines = len(lines)

        while i < n_lines:
            m = table_hdr.search(lines[i])
            if m:
                tid, X, Y, Z, dX1, dX2 = (
                    m.group(1),
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                    float(m.group(5)),
                    float(m.group(6)),
                )
                i += 1
                col_m = None
                while i < n_lines:
                    col_m = col_hdr.match(lines[i].strip())
                    if col_m:
                        break
                    i += 1
                if col_m is None:
                    raise ValueError("Missing 'logT ...' column header after TABLE block.")
                if log_r_values is None:
                    log_r_values = np.array([float(x) for x in col_m.group(1).split()], dtype=np.float64)
                i += 1
                rows_T, rows_data = [], []
                while i < n_lines:
                    dl = lines[i].strip()
                    if not dl:
                        i += 1
                        continue
                    if table_hdr.search(dl) or _re.match(r"^\*+", dl):
                        break
                    parts = dl.split()
                    try:
                        rows_T.append(float(parts[0]))
                    except (ValueError, IndexError):
                        i += 1
                        continue
                    rows_data.append(np.array([float(x) for x in parts[1:]], dtype=np.float64))
                    i += 1
                if not rows_T:
                    continue
                n2 = len(log_r_values)
                arr = np.full((len(rows_T), n2), np.nan, dtype=np.float64)
                for ri, row in enumerate(rows_data):
                    nc = min(len(row), n2)
                    arr[ri, :nc] = row[:nc]
                arr[np.abs(arr - _INVALID) < 1e-3] = np.nan
                if log_t_values is None:
                    log_t_values = np.array(rows_T, dtype=np.float64)
                blocks.append((tid, X, Y, Z, dX1, dX2, arr))
            else:
                i += 1

        if not blocks:
            raise ValueError(f"No TABLE blocks found in {txt_path}.")

        n1, n2, n_tbl = len(log_t_values), len(log_r_values), len(blocks)
        opacity = np.full((n_tbl, n1, n2), np.nan, dtype=np.float64)
        tids, Xs, Ys, Zs, dX1s, dX2s = [], [], [], [], [], []
        for ti, (tid, X, Y, Z, dX1, dX2, data) in enumerate(blocks):
            r1, r2 = min(data.shape[0], n1), min(data.shape[1], n2)
            opacity[ti, :r1, :r2] = data[:r1, :r2]
            tids.append(tid)
            Xs.append(X)
            Ys.append(Y)
            Zs.append(Z)
            dX1s.append(dX1)
            dX2s.append(dX2)

        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as f:
            f.attrs["source"] = "OPAL"
            f.attrs["reference"] = "Iglesias & Rogers 1996"
            f.attrs["mean_type"] = "rosseland"
            grp = f.create_group("grid")
            grp.create_dataset("grid_1", data=log_t_values)
            grp.create_dataset("grid_2", data=log_r_values)
            f.create_dataset("opacity", data=opacity)
            mgrp = f.create_group("metadata")
            mgrp.create_dataset("X", data=np.array(Xs))
            mgrp.create_dataset("Y", data=np.array(Ys))
            mgrp.create_dataset("Z", data=np.array(Zs))
            mgrp.create_dataset("dX1", data=np.array(dX1s))
            mgrp.create_dataset("dX2", data=np.array(dX2s))
            mgrp.create_dataset("table_id", data=np.array(tids, dtype=object), dtype=h5py.string_dtype())

    def __repr__(self) -> str:
        return f"OPALOpacity(out_of_bounds={self.out_of_bounds!r})"
