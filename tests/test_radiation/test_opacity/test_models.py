"""Tests for :mod:`triceratops.radiation.opacity.grey_opacity.rosseland.models`.

Organisation
------------
TestNormalisationConstants      -- KAPPA_FF_0, KAPPA_BF_0, KAPPA_KR_0 consistency
TestConstantGreyOpacity         -- fixed kappa, zero derivatives, .kappa property, mean_type
TestElectronScatteringOpacity   -- constant Thomson kappa, IS_C_BACKED, zero derivatives
TestKramersFFOpacity            -- kappa proportional to rho T^{-3.5}, unit derivatives, default kappa0
TestKramersBFOpacity            -- same scaling as FF, different normalisation
TestKramersOpacity              -- combined FF+BF, kappa0 = KAPPA_FF_0 + KAPPA_BF_0
TestKramersFFESOpacity          -- kappa = kappa_es + kappa_ff*rho*T^{-3.5}, mixed derivatives
TestKramersBFESOpacity          -- kappa = kappa_es + kappa_bf*rho*T^{-3.5}
TestKramersESOpacity            -- kappa = kappa_es + (kappa_ff+kappa_bf)*rho*T^{-3.5}
TestKramersPurePowerLawScaling  -- parametrised rho- and T-scaling for all pure Kramers laws
TestKramersESDerivativeLimits   -- ES-dominated and Kramers-dominated asymptotic limits
"""

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from triceratops.radiation.opacity.grey_opacity.base import ConstantGreyOpacity, GreyOpacityLaw
from triceratops.radiation.opacity.grey_opacity.rosseland.models import (
    KAPPA_BF_0,
    KAPPA_FF_0,
    KAPPA_KR_0,
    ElectronScatteringOpacity,
    KramersBFESOpacity,
    KramersBFOpacity,
    KramersESOpacity,
    KramersFFESOpacity,
    KramersFFOpacity,
    KramersOpacity,
)

# ------------------------------------------------------------------ #
# Shared reference conditions                                         #
# ------------------------------------------------------------------ #
#: Reference density: typical disc midplane value [g g/cm^3].
_RHO_CGS = 1.0e-5
#: Reference temperature: moderately hot disc midplane [K].
_T_CGS = 1.0e7
_RHO = _RHO_CGS * u.g / u.cm**3
_T = _T_CGS * u.K


# ================================================================== #
# TestNormalisationConstants                                          #
# ================================================================== #


class TestNormalisationConstants:
    """Consistency checks on the module-level Kramers normalisation constants."""

    def test_kappa_ff_0_positive(self):
        """KAPPA_FF_0 is a positive float."""
        assert KAPPA_FF_0 > 0

    def test_kappa_bf_0_positive(self):
        """KAPPA_BF_0 is a positive float."""
        assert KAPPA_BF_0 > 0

    def test_kappa_kr_0_equals_ff_plus_bf(self):
        """KAPPA_KR_0 equals KAPPA_FF_0 + KAPPA_BF_0 exactly."""
        assert KAPPA_KR_0 == pytest.approx(KAPPA_FF_0 + KAPPA_BF_0, rel=1e-12)

    def test_kappa_bf_dominates_ff(self):
        """Bound-free opacity dominates free-free for solar composition."""
        assert KAPPA_BF_0 > KAPPA_FF_0

    def test_kappa_ff_0_value(self):
        """KAPPA_FF_0 matches the standard solar-composition free-free value."""
        assert KAPPA_FF_0 == pytest.approx(3.68e22, rel=1e-6)

    def test_kappa_bf_0_value(self):
        """KAPPA_BF_0 matches the standard solar-composition bound-free value."""
        assert KAPPA_BF_0 == pytest.approx(4.34e25, rel=1e-6)


# ================================================================== #
# TestConstantOpacity                                                 #
# ================================================================== #


class TestConstantGreyOpacity:
    """Tests for :class:`~triceratops.radiation.opacity.grey_opacity.base.ConstantGreyOpacity`.

    Verifies that the opacity value is fixed, both partial derivatives are
    identically zero, and mean_type is correctly propagated.
    """

    def test_is_grey_opacity_law(self):
        """ConstantGreyOpacity inherits from GreyOpacityLaw."""
        assert isinstance(ConstantGreyOpacity(0.5), GreyOpacityLaw)

    def test_kappa_property_matches_constructor(self):
        """The .kappa property returns the value passed to the constructor."""
        law = ConstantGreyOpacity(0.5)
        assert law.kappa.value == pytest.approx(0.5, rel=1e-12)

    def test_opacity_returns_correct_value(self):
        """opacity() returns the constructor kappa in cm^2 g^{-1}."""
        law = ConstantGreyOpacity(0.5)
        result = law.opacity(_RHO, _T)
        assert result.value == pytest.approx(0.5, rel=1e-12)

    def test_opacity_independent_of_rho(self):
        """opacity() is constant: changing rho by six orders of magnitude has no effect."""
        law = ConstantGreyOpacity(0.5)
        kap1 = law.opacity(1.0e-10 * u.g / u.cm**3, _T)
        kap2 = law.opacity(1.0e-4 * u.g / u.cm**3, _T)
        assert kap1.value == pytest.approx(kap2.value, rel=1e-12)

    def test_opacity_independent_of_T(self):
        """opacity() is constant: changing T by several orders of magnitude has no effect."""
        law = ConstantGreyOpacity(0.5)
        kap1 = law.opacity(_RHO, 1.0e4 * u.K)
        kap2 = law.opacity(_RHO, 1.0e10 * u.K)
        assert kap1.value == pytest.approx(kap2.value, rel=1e-12)

    def test_dlogkappa_dlogrho_zero(self):
        """d ln kappa / d ln rho = 0 for a constant opacity."""
        law = ConstantGreyOpacity(0.5)
        assert law.dlogkappa_dlogrho(_RHO, _T) == pytest.approx(0.0, abs=1e-15)

    def test_dlogkappa_dlogT_zero(self):
        """d ln kappa / d ln T = 0 for a constant opacity."""
        law = ConstantGreyOpacity(0.5)
        assert law.dlogkappa_dlogT(_RHO, _T) == pytest.approx(0.0, abs=1e-15)

    def test_log_kappa_stored_correctly(self):
        """Internal _log_kappa equals ln(kappa)."""
        law = ConstantGreyOpacity(0.5)
        assert law._log_kappa == pytest.approx(np.log(0.5), rel=1e-12)

    def test_is_not_c_backed(self):
        """ConstantGreyOpacity is a pure-Python implementation (IS_C_BACKED = False)."""
        assert ConstantGreyOpacity.IS_C_BACKED is False

    def test_mean_type_default_is_none(self):
        """mean_type defaults to None when not specified."""
        assert ConstantGreyOpacity(0.5).mean_type is None

    def test_mean_type_set_by_constructor(self):
        """mean_type is stored when passed as a keyword argument."""
        assert ConstantGreyOpacity(0.5, mean_type="rosseland").mean_type == "rosseland"

    @pytest.mark.parametrize("kappa", [0.1, 0.34, 1.0, 100.0])
    def test_kappa_roundtrip(self, kappa):
        """opacity() round-trips through log-space without precision loss."""
        law = ConstantGreyOpacity(kappa)
        result = law.opacity(_RHO, _T)
        assert result.value == pytest.approx(kappa, rel=1e-12)


# ================================================================== #
# TestElectronScatteringOpacity                                       #
# ================================================================== #


class TestElectronScatteringOpacity:
    """Tests for :class:`~triceratops.radiation.opacity.grey_opacity.rosseland.models.ElectronScatteringOpacity`.

    Thomson scattering produces a grey (constant) opacity; the value does
    not depend on rho or T, and both partial derivatives vanish.
    """

    def test_is_grey_opacity_law(self):
        """ElectronScatteringOpacity inherits from GreyOpacityLaw."""
        assert isinstance(ElectronScatteringOpacity(), GreyOpacityLaw)

    def test_is_c_backed(self):
        """ElectronScatteringOpacity delegates to a Cython extension (IS_C_BACKED = True)."""
        assert ElectronScatteringOpacity.IS_C_BACKED is True

    def test_c_object_initialised(self):
        """A non-None _c_object is created at construction."""
        law = ElectronScatteringOpacity()
        assert law._c_object is not None

    def test_default_kappa_es(self):
        """The default electron-scattering opacity is 0.34 cm^2 g^{-1}."""
        law = ElectronScatteringOpacity()
        assert law.kappa_es == pytest.approx(0.34, rel=1e-12)

    def test_custom_kappa_es(self):
        """A custom kappa_es is stored and returned correctly."""
        law = ElectronScatteringOpacity(kappa_es=0.20)
        assert law.kappa_es == pytest.approx(0.20, rel=1e-12)

    def test_opacity_matches_kappa_es(self):
        """opacity() returns the stored kappa_es value in cm^2 g^{-1}."""
        kap_es = 0.34
        law = ElectronScatteringOpacity(kappa_es=kap_es)
        result = law.opacity(_RHO, _T)
        assert result.value == pytest.approx(kap_es, rel=1e-10)

    def test_opacity_independent_of_rho(self):
        """Thomson opacity is constant -- rho has no effect."""
        law = ElectronScatteringOpacity()
        kap1 = law.opacity(1.0e-10 * u.g / u.cm**3, _T)
        kap2 = law.opacity(1.0e-2 * u.g / u.cm**3, _T)
        assert kap1.value == pytest.approx(kap2.value, rel=1e-10)

    def test_opacity_independent_of_T(self):
        """Thomson opacity is constant -- T has no effect."""
        law = ElectronScatteringOpacity()
        kap1 = law.opacity(_RHO, 1.0e4 * u.K)
        kap2 = law.opacity(_RHO, 1.0e10 * u.K)
        assert kap1.value == pytest.approx(kap2.value, rel=1e-10)

    def test_dlogkappa_dlogrho_zero(self):
        """d ln kappa / d ln rho = 0 for electron scattering."""
        law = ElectronScatteringOpacity()
        assert law.dlogkappa_dlogrho(_RHO, _T) == pytest.approx(0.0, abs=1e-12)

    def test_dlogkappa_dlogT_zero(self):
        """d ln kappa / d ln T = 0 for electron scattering."""
        law = ElectronScatteringOpacity()
        assert law.dlogkappa_dlogT(_RHO, _T) == pytest.approx(0.0, abs=1e-12)


# ================================================================== #
# TestKramersFFOpacity                                                #
# ================================================================== #


class TestKramersFFOpacity:
    """Tests for :class:`~triceratops.radiation.opacity.grey_opacity.rosseland.models.KramersFFOpacity`.

    The free-free Kramers law obeys kappa = kappa0 rho T^{-3.5}, giving
    d ln kappa / d ln rho = 1 and d ln kappa / d ln T = -3.5 everywhere.
    """

    def test_is_grey_opacity_law(self):
        """KramersFFOpacity inherits from GreyOpacityLaw."""
        assert isinstance(KramersFFOpacity(), GreyOpacityLaw)

    def test_is_c_backed(self):
        """KramersFFOpacity uses a Cython extension (IS_C_BACKED = True)."""
        assert KramersFFOpacity.IS_C_BACKED is True

    def test_default_kappa0(self):
        """Default kappa0 equals KAPPA_FF_0."""
        law = KramersFFOpacity()
        assert law.kappa0 == pytest.approx(KAPPA_FF_0, rel=1e-12)

    def test_opacity_formula(self):
        """opacity() matches kappa0 rho T^{-3.5} at the reference point."""
        law = KramersFFOpacity()
        expected = KAPPA_FF_0 * _RHO_CGS * _T_CGS ** (-3.5)
        result = law.opacity(_RHO, _T)
        assert result.value == pytest.approx(expected, rel=1e-6)

    def test_dlogkappa_dlogrho_unity(self):
        """d ln kappa / d ln rho = 1 (power-law index on density)."""
        law = KramersFFOpacity()
        assert law.dlogkappa_dlogrho(_RHO, _T) == pytest.approx(1.0, rel=1e-6)

    def test_dlogkappa_dlogT_minus_3_5(self):
        """d ln kappa / d ln T = -3.5 (power-law index on temperature)."""
        law = KramersFFOpacity()
        assert law.dlogkappa_dlogT(_RHO, _T) == pytest.approx(-3.5, rel=1e-6)

    def test_custom_kappa0_stored(self):
        """A custom kappa0 is stored on the instance."""
        law = KramersFFOpacity(kappa0=1.0e20)
        assert law.kappa0 == pytest.approx(1.0e20, rel=1e-12)

    def test_custom_kappa0_changes_opacity(self):
        """Halving kappa0 halves the opacity value."""
        law_default = KramersFFOpacity()
        law_half = KramersFFOpacity(kappa0=KAPPA_FF_0 / 2)
        kap_default = law_default.opacity(_RHO, _T)
        kap_half = law_half.opacity(_RHO, _T)
        assert kap_default.value == pytest.approx(2 * kap_half.value, rel=1e-6)


# ================================================================== #
# TestKramersBFOpacity                                                #
# ================================================================== #


class TestKramersBFOpacity:
    """Tests for :class:`~triceratops.radiation.opacity.grey_opacity.rosseland.models.KramersBFOpacity`.

    Bound-free Kramers: kappa = kappa_bf,0 rho T^{-3.5}.  Same power-law structure
    as the free-free law but with a larger normalisation constant.
    """

    def test_default_kappa0(self):
        """Default kappa0 equals KAPPA_BF_0."""
        law = KramersBFOpacity()
        assert law.kappa0 == pytest.approx(KAPPA_BF_0, rel=1e-12)

    def test_opacity_formula(self):
        """opacity() matches kappa_bf,0 rho T^{-3.5} at the reference point."""
        law = KramersBFOpacity()
        expected = KAPPA_BF_0 * _RHO_CGS * _T_CGS ** (-3.5)
        result = law.opacity(_RHO, _T)
        assert result.value == pytest.approx(expected, rel=1e-6)

    def test_dlogkappa_dlogrho_unity(self):
        """d ln kappa / d ln rho = 1."""
        law = KramersBFOpacity()
        assert law.dlogkappa_dlogrho(_RHO, _T) == pytest.approx(1.0, rel=1e-6)

    def test_dlogkappa_dlogT_minus_3_5(self):
        """d ln kappa / d ln T = -3.5."""
        law = KramersBFOpacity()
        assert law.dlogkappa_dlogT(_RHO, _T) == pytest.approx(-3.5, rel=1e-6)

    def test_bf_exceeds_ff_at_reference(self):
        """Bound-free opacity exceeds free-free opacity (KAPPA_BF_0 > KAPPA_FF_0)."""
        kap_ff = KramersFFOpacity().opacity(_RHO, _T).value
        kap_bf = KramersBFOpacity().opacity(_RHO, _T).value
        assert kap_bf > kap_ff


# ================================================================== #
# TestKramersOpacity                                                  #
# ================================================================== #


class TestKramersOpacity:
    """Tests for :class:`~triceratops.radiation.opacity.grey_opacity.rosseland.models.KramersOpacity`.

    Combined (FF + BF) Kramers law: kappa = (kappa_ff,0 + kappa_bf,0) rho T^{-3.5}.
    The shared power-law structure allows the two terms to be combined into
    a single effective normalisation constant KAPPA_KR_0.
    """

    def test_default_kappa0(self):
        """Default kappa0 equals KAPPA_KR_0 = KAPPA_FF_0 + KAPPA_BF_0."""
        law = KramersOpacity()
        assert law.kappa0 == pytest.approx(KAPPA_KR_0, rel=1e-12)

    def test_opacity_equals_ff_plus_bf(self):
        """KramersOpacity equals KramersFFOpacity + KramersBFOpacity at the reference point."""
        kap_combined = KramersOpacity().opacity(_RHO, _T).value
        kap_ff = KramersFFOpacity().opacity(_RHO, _T).value
        kap_bf = KramersBFOpacity().opacity(_RHO, _T).value
        assert kap_combined == pytest.approx(kap_ff + kap_bf, rel=1e-6)

    def test_dlogkappa_dlogrho_unity(self):
        """d ln kappa / d ln rho = 1."""
        law = KramersOpacity()
        assert law.dlogkappa_dlogrho(_RHO, _T) == pytest.approx(1.0, rel=1e-6)

    def test_dlogkappa_dlogT_minus_3_5(self):
        """d ln kappa / d ln T = -3.5."""
        law = KramersOpacity()
        assert law.dlogkappa_dlogT(_RHO, _T) == pytest.approx(-3.5, rel=1e-6)


# ================================================================== #
# TestKramersFFESOpacity                                              #
# ================================================================== #


def _kramers_es_opacity(kappa_kr, rho_cgs, T_cgs, kappa_es=0.34):
    """Reference formula: kappa = kappa_es + kappa_kr * rho * T^{-3.5}."""
    return kappa_es + kappa_kr * rho_cgs * T_cgs ** (-3.5)


def _kramers_es_dlogrho(kappa_kr, rho_cgs, T_cgs, kappa_es=0.34):
    """d ln kappa / d ln rho = kappa_kr*rho*T^{-3.5} / kappa_total."""
    kap_kr = kappa_kr * rho_cgs * T_cgs ** (-3.5)
    kap_tot = kappa_es + kap_kr
    return kap_kr / kap_tot


def _kramers_es_dlogT(kappa_kr, rho_cgs, T_cgs, kappa_es=0.34):
    """d ln kappa / d ln T = -3.5 * kappa_kr*rho*T^{-3.5} / kappa_total."""
    return -3.5 * _kramers_es_dlogrho(kappa_kr, rho_cgs, T_cgs, kappa_es)


class TestKramersFFESOpacity:
    """Tests for :class:`~triceratops.radiation.opacity.grey_opacity.rosseland.models.KramersFFESOpacity`.

    kappa = kappa_es + kappa_ff,0 * rho * T^{-3.5}.  Both attributes (kappa0, kappa_es)
    must be stored, and the mixed derivatives must interpolate between the
    pure electron-scattering (zero) and pure Kramers (+/-1, -/+3.5) limits.
    """

    def test_default_kappa0(self):
        """Default kappa0 equals KAPPA_FF_0."""
        law = KramersFFESOpacity()
        assert law.kappa0 == pytest.approx(KAPPA_FF_0, rel=1e-12)

    def test_default_kappa_es(self):
        """Default kappa_es equals 0.34 cm^2 g^{-1}."""
        law = KramersFFESOpacity()
        assert law.kappa_es == pytest.approx(0.34, rel=1e-12)

    def test_opacity_formula(self):
        """opacity() matches kappa_es + kappa_ff,0 * rho * T^{-3.5} at the reference point."""
        law = KramersFFESOpacity()
        expected = _kramers_es_opacity(KAPPA_FF_0, _RHO_CGS, _T_CGS)
        result = law.opacity(_RHO, _T)
        assert result.value == pytest.approx(expected, rel=1e-6)

    def test_dlogkappa_dlogrho_formula(self):
        """d ln kappa / d ln rho matches the analytical formula."""
        law = KramersFFESOpacity()
        expected = _kramers_es_dlogrho(KAPPA_FF_0, _RHO_CGS, _T_CGS)
        result = law.dlogkappa_dlogrho(_RHO, _T)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_dlogkappa_dlogT_formula(self):
        """d ln kappa / d ln T matches the analytical formula."""
        law = KramersFFESOpacity()
        expected = _kramers_es_dlogT(KAPPA_FF_0, _RHO_CGS, _T_CGS)
        result = law.dlogkappa_dlogT(_RHO, _T)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_dlogkappa_dlogrho_between_zero_and_one(self):
        """d ln kappa / d ln rho lies in (0, 1) at the reference point."""
        law = KramersFFESOpacity()
        val = law.dlogkappa_dlogrho(_RHO, _T)
        assert 0.0 < val < 1.0

    def test_dlogkappa_dlogT_between_minus_3_5_and_zero(self):
        """d ln kappa / d ln T lies in (-3.5, 0) at the reference point."""
        law = KramersFFESOpacity()
        val = law.dlogkappa_dlogT(_RHO, _T)
        assert -3.5 < val < 0.0

    def test_is_c_backed(self):
        """KramersFFESOpacity uses a Cython extension (IS_C_BACKED = True)."""
        assert KramersFFESOpacity.IS_C_BACKED is True


# ================================================================== #
# TestKramersBFESOpacity                                              #
# ================================================================== #


class TestKramersBFESOpacity:
    """Tests for :class:`~triceratops.radiation.opacity.grey_opacity.rosseland.models.KramersBFESOpacity`.

    kappa = kappa_es + kappa_bf,0 * rho * T^{-3.5}.
    """

    def test_default_kappa0(self):
        """Default kappa0 equals KAPPA_BF_0."""
        law = KramersBFESOpacity()
        assert law.kappa0 == pytest.approx(KAPPA_BF_0, rel=1e-12)

    def test_opacity_formula(self):
        """opacity() matches kappa_es + kappa_bf,0 * rho * T^{-3.5}."""
        law = KramersBFESOpacity()
        expected = _kramers_es_opacity(KAPPA_BF_0, _RHO_CGS, _T_CGS)
        result = law.opacity(_RHO, _T)
        assert result.value == pytest.approx(expected, rel=1e-6)

    def test_dlogkappa_dlogrho_formula(self):
        """d ln kappa / d ln rho matches the analytical formula."""
        law = KramersBFESOpacity()
        expected = _kramers_es_dlogrho(KAPPA_BF_0, _RHO_CGS, _T_CGS)
        result = law.dlogkappa_dlogrho(_RHO, _T)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_dlogkappa_dlogT_formula(self):
        """d ln kappa / d ln T matches the analytical formula."""
        law = KramersBFESOpacity()
        expected = _kramers_es_dlogT(KAPPA_BF_0, _RHO_CGS, _T_CGS)
        result = law.dlogkappa_dlogT(_RHO, _T)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_bf_es_exceeds_ff_es(self):
        """KramersBFESOpacity exceeds KramersFFESOpacity at the reference point."""
        kap_ff_es = KramersFFESOpacity().opacity(_RHO, _T).value
        kap_bf_es = KramersBFESOpacity().opacity(_RHO, _T).value
        assert kap_bf_es > kap_ff_es


# ================================================================== #
# TestKramersESOpacity                                                #
# ================================================================== #


class TestKramersESOpacity:
    """Tests for :class:`~triceratops.radiation.opacity.grey_opacity.rosseland.models.KramersESOpacity`.

    kappa = kappa_es + (kappa_ff,0 + kappa_bf,0) * rho * T^{-3.5}.  This is the most
    physically complete grey opacity for hot, fully ionised plasma.
    """

    def test_default_kappa0(self):
        """Default kappa0 equals KAPPA_KR_0."""
        law = KramersESOpacity()
        assert law.kappa0 == pytest.approx(KAPPA_KR_0, rel=1e-12)

    def test_opacity_formula(self):
        """opacity() matches kappa_es + KAPPA_KR_0 * rho * T^{-3.5}."""
        law = KramersESOpacity()
        expected = _kramers_es_opacity(KAPPA_KR_0, _RHO_CGS, _T_CGS)
        result = law.opacity(_RHO, _T)
        assert result.value == pytest.approx(expected, rel=1e-6)

    def test_opacity_equals_ff_es_plus_bf_contribution(self):
        """KramersESOpacity ~= ElectronScattering + KramersOpacity at reference point."""
        kap_es = ElectronScatteringOpacity().opacity(_RHO, _T).value
        kap_kr = KramersOpacity().opacity(_RHO, _T).value
        kap_combined = KramersESOpacity().opacity(_RHO, _T).value
        assert kap_combined == pytest.approx(kap_es + kap_kr, rel=1e-6)

    def test_dlogkappa_dlogrho_formula(self):
        """d ln kappa / d ln rho matches the analytical formula."""
        law = KramersESOpacity()
        expected = _kramers_es_dlogrho(KAPPA_KR_0, _RHO_CGS, _T_CGS)
        result = law.dlogkappa_dlogrho(_RHO, _T)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_dlogkappa_dlogT_formula(self):
        """d ln kappa / d ln T matches the analytical formula."""
        law = KramersESOpacity()
        expected = _kramers_es_dlogT(KAPPA_KR_0, _RHO_CGS, _T_CGS)
        result = law.dlogkappa_dlogT(_RHO, _T)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_is_c_backed(self):
        """KramersESOpacity uses a Cython extension (IS_C_BACKED = True)."""
        assert KramersESOpacity.IS_C_BACKED is True


# ================================================================== #
# TestKramersPurePowerLawScaling                                      #
# ================================================================== #


@pytest.mark.parametrize(
    "LawClass, kappa0_attr, kappa0_val",
    [
        (KramersFFOpacity, "kappa0", KAPPA_FF_0),
        (KramersBFOpacity, "kappa0", KAPPA_BF_0),
        (KramersOpacity, "kappa0", KAPPA_KR_0),
    ],
    ids=["KramersFF", "KramersBF", "Kramers"],
)
class TestKramersPurePowerLawScaling:
    """Parametrised power-law scaling tests for all pure (no ES) Kramers laws.

    For kappa = kappa0 rho T^{-3.5} we expect:
    - Doubling rho doubles kappa (linear in density).
    - Reducing T by factor 2 increases kappa by 2^{3.5} (negative temperature index).
    """

    def test_opacity_doubles_with_rho(self, LawClass, kappa0_attr, kappa0_val):
        """Doubling rho doubles the opacity (linear density dependence)."""
        law = LawClass()
        kap1 = law.opacity(_RHO, _T)
        kap2 = law.opacity(2.0 * _RHO, _T)
        assert kap2.value == pytest.approx(2.0 * kap1.value, rel=1e-6)

    def test_opacity_scales_with_T_power(self, LawClass, kappa0_attr, kappa0_val):
        """Halving T increases opacity by 2^{3.5} (T^{-3.5} dependence)."""
        law = LawClass()
        kap1 = law.opacity(_RHO, _T)
        kap2 = law.opacity(_RHO, _T / 2.0)
        assert kap2.value == pytest.approx(2**3.5 * kap1.value, rel=1e-6)

    def test_dlogkappa_dlogrho_unity_at_multiple_points(self, LawClass, kappa0_attr, kappa0_val):
        """d ln kappa / d ln rho = 1 at several (rho, T) pairs."""
        law = LawClass()
        for rho in [1e-8, 1e-5, 1e-2]:
            for T in [1e5, 1e7, 1e9]:
                val = law.dlogkappa_dlogrho(rho * u.g / u.cm**3, T * u.K)
                assert val == pytest.approx(1.0, rel=1e-5), f"rho={rho}, T={T}"

    def test_dlogkappa_dlogT_minus_3_5_at_multiple_points(self, LawClass, kappa0_attr, kappa0_val):
        """d ln kappa / d ln T = -3.5 at several (rho, T) pairs."""
        law = LawClass()
        for rho in [1e-8, 1e-5, 1e-2]:
            for T in [1e5, 1e7, 1e9]:
                val = law.dlogkappa_dlogT(rho * u.g / u.cm**3, T * u.K)
                assert val == pytest.approx(-3.5, rel=1e-5), f"rho={rho}, T={T}"


# ================================================================== #
# TestKramersESDerivativeLimits                                       #
# ================================================================== #


@pytest.mark.parametrize(
    "LawClass, kappa_kr",
    [
        (KramersFFESOpacity, KAPPA_FF_0),
        (KramersBFESOpacity, KAPPA_BF_0),
        (KramersESOpacity, KAPPA_KR_0),
    ],
    ids=["KramersFFES", "KramersBFES", "KramersES"],
)
class TestKramersESDerivativeLimits:
    """Asymptotic derivative limits for Kramers + electron-scattering laws.

    In the ES-dominated regime (low rho or high T), derivatives approach zero.
    In the Kramers-dominated regime (high rho or low T), derivatives approach
    the pure Kramers values (d/drho -> 1, d/dT -> -3.5).
    """

    def test_es_dominated_dlogrho_near_zero(self, LawClass, kappa_kr):
        """d ln kappa / d ln rho -> 0 when electron scattering dominates (very low rho)."""
        law = LawClass()
        # At very low rho, Kramers term << kappa_es -> derivative -> 0
        rho_low = 1.0e-15 * u.g / u.cm**3
        T_high = 1.0e10 * u.K
        val = law.dlogkappa_dlogrho(rho_low, T_high)
        assert abs(val) < 0.01, f"Expected near-zero, got {val}"

    def test_es_dominated_dlogT_near_zero(self, LawClass, kappa_kr):
        """d ln kappa / d ln T -> 0 when electron scattering dominates (very low rho)."""
        law = LawClass()
        rho_low = 1.0e-15 * u.g / u.cm**3
        T_high = 1.0e10 * u.K
        val = law.dlogkappa_dlogT(rho_low, T_high)
        assert abs(val) < 0.01, f"Expected near-zero, got {val}"

    def test_kramers_dominated_dlogrho_near_unity(self, LawClass, kappa_kr):
        """d ln kappa / d ln rho -> 1 when Kramers term dominates (very high rho)."""
        law = LawClass()
        # At very high rho, Kramers term >> kappa_es -> derivative -> 1
        rho_high = 1.0e5 * u.g / u.cm**3
        T_low = 1.0e4 * u.K
        val = law.dlogkappa_dlogrho(rho_high, T_low)
        assert val == pytest.approx(1.0, abs=0.001), f"Expected ~1, got {val}"

    def test_kramers_dominated_dlogT_near_minus_3_5(self, LawClass, kappa_kr):
        """d ln kappa / d ln T -> -3.5 when Kramers term dominates (very high rho)."""
        law = LawClass()
        rho_high = 1.0e5 * u.g / u.cm**3
        T_low = 1.0e4 * u.K
        val = law.dlogkappa_dlogT(rho_high, T_low)
        assert val == pytest.approx(-3.5, abs=0.01), f"Expected ~-3.5, got {val}"

    def test_dlogkappa_dlogrho_dlogT_consistent(self, LawClass, kappa_kr):
        """d ln kappa / d ln T = -3.5 * d ln kappa / d ln rho at all points (both derived from same Kramers fraction)."""
        law = LawClass()
        for rho in [1e-8, 1e-5, 1e-2]:
            for T in [1e5, 1e7, 1e9]:
                rho_q = rho * u.g / u.cm**3
                T_q = T * u.K
                d_rho = law.dlogkappa_dlogrho(rho_q, T_q)
                d_T = law.dlogkappa_dlogT(rho_q, T_q)
                assert d_T == pytest.approx(-3.5 * d_rho, rel=1e-5), (
                    f"d/dT = {d_T} != -3.5 * d/drho = {-3.5 * d_rho} at rho={rho}, T={T}"
                )
