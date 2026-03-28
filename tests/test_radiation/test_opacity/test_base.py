"""Tests for :mod:`triceratops.radiation.opacity.base`.

Organisation
------------
TestOpacityLawInterface     — abstract interface contract for OpacityLaw
TestGreyOpacityLawInterface — GreyOpacityLaw pure-Python defaults and unit wrappers
"""

import numpy as np
import pytest
from astropy import units as u

from triceratops.radiation.opacity.base import GreyOpacityLaw, OpacityLaw

# ------------------------------------------------------------------ #
# Shared reference conditions                                         #
# ------------------------------------------------------------------ #
#: Reference density: typical disc midplane value [g cm⁻³].
_RHO = 1.0e-5 * u.g / u.cm**3
#: Reference temperature: moderately hot disc midplane [K].
_T = 1.0e7 * u.K


# ================================================================== #
# Helpers                                                             #
# ================================================================== #


class _PureGreyLaw(GreyOpacityLaw):
    """Minimal pure-Python GreyOpacityLaw for interface testing."""

    _LOG_KAPPA = np.log(0.5)  # κ = 0.5 cm² g⁻¹


class _UnimplementedLaw(OpacityLaw):
    """Bare OpacityLaw subclass that intentionally omits all overrides."""

    IS_C_BACKED = False


# ================================================================== #
# TestOpacityLawInterface                                             #
# ================================================================== #


class TestOpacityLawInterface:
    """Contract tests for the :class:`~triceratops.radiation.opacity.base.OpacityLaw` ABC.

    These tests verify that the abstract interface raises
    :exc:`NotImplementedError` when the private log-space methods are not
    overridden by a concrete subclass.
    """

    def test_log_opacity_not_implemented_raises(self):
        """_log_opacity raises NotImplementedError when IS_C_BACKED is False and not overridden."""
        law = _UnimplementedLaw()
        with pytest.raises(NotImplementedError, match="_log_opacity"):
            law._log_opacity(np.log(1e4), np.log(1e-5))

    def test_dlogkappa_dlogrho_not_implemented_raises(self):
        """_dlogkappa_dlogrho raises NotImplementedError when IS_C_BACKED is False and not overridden."""
        law = _UnimplementedLaw()
        with pytest.raises(NotImplementedError, match="_dlogkappa_dlogrho"):
            law._dlogkappa_dlogrho(np.log(1e4), np.log(1e-5))

    def test_dlogkappa_dlogT_not_implemented_raises(self):
        """_dlogkappa_dlogT raises NotImplementedError when IS_C_BACKED is False and not overridden."""
        law = _UnimplementedLaw()
        with pytest.raises(NotImplementedError, match="_dlogkappa_dlogT"):
            law._dlogkappa_dlogT(np.log(1e4), np.log(1e-5))

    def test_c_backed_initialise_raises_if_not_overridden(self):
        """_initialize_C_object raises NotImplementedError if IS_C_BACKED=True without override."""

        class _BadCBacked(OpacityLaw):
            IS_C_BACKED = True

        with pytest.raises(NotImplementedError, match="_initialize_C_object"):
            _BadCBacked()

    def test_is_c_backed_default_false(self):
        """IS_C_BACKED defaults to False on the base class."""
        assert OpacityLaw.IS_C_BACKED is False


# ================================================================== #
# TestGreyOpacityLawInterface                                         #
# ================================================================== #


class TestGreyOpacityLawInterface:
    """Tests for the pure-Python :class:`~triceratops.radiation.opacity.base.GreyOpacityLaw` defaults.

    Uses :class:`_PureGreyLaw` (κ = 0.5 cm² g⁻¹) as a concrete stand-in.
    """

    def test_opacity_returns_quantity(self):
        """opacity() returns an astropy Quantity."""
        law = _PureGreyLaw()
        result = law.opacity(_RHO, _T)
        assert isinstance(result, u.Quantity)

    def test_opacity_unit_is_cm2_per_g(self):
        """opacity() result has units of cm² g⁻¹."""
        law = _PureGreyLaw()
        result = law.opacity(_RHO, _T)
        assert result.unit.is_equivalent(u.cm**2 / u.g)

    def test_opacity_value_matches_log_kappa(self):
        """opacity() value equals exp(_LOG_KAPPA) regardless of (rho, T)."""
        law = _PureGreyLaw()
        expected = np.exp(_PureGreyLaw._LOG_KAPPA)
        result = law.opacity(_RHO, _T)
        assert result.value == pytest.approx(expected, rel=1e-12)

    def test_opacity_independent_of_rho(self):
        """Grey opacity is constant — changing rho does not change the result."""
        law = _PureGreyLaw()
        kap1 = law.opacity(1.0e-8 * u.g / u.cm**3, _T)
        kap2 = law.opacity(1.0e-2 * u.g / u.cm**3, _T)
        assert kap1.value == pytest.approx(kap2.value, rel=1e-12)

    def test_opacity_independent_of_T(self):
        """Grey opacity is constant — changing T does not change the result."""
        law = _PureGreyLaw()
        kap1 = law.opacity(_RHO, 1.0e4 * u.K)
        kap2 = law.opacity(_RHO, 1.0e9 * u.K)
        assert kap1.value == pytest.approx(kap2.value, rel=1e-12)

    def test_dlogkappa_dlogrho_zero(self):
        """Pure-Python grey law: d ln κ / d ln ρ = 0."""
        law = _PureGreyLaw()
        assert law.dlogkappa_dlogrho(_RHO, _T) == pytest.approx(0.0, abs=1e-15)

    def test_dlogkappa_dlogT_zero(self):
        """Pure-Python grey law: d ln κ / d ln T = 0."""
        law = _PureGreyLaw()
        assert law.dlogkappa_dlogT(_RHO, _T) == pytest.approx(0.0, abs=1e-15)

    def test_opacity_accepts_array_rho(self):
        """opacity() returns a finite value when called with an array of densities.

        Pure-Python grey laws return the scalar constant; C-backed laws may
        broadcast.  The key guarantee is that no exception is raised and the
        result is finite.
        """
        law = _PureGreyLaw()
        rho_arr = np.geomspace(1e-8, 1e-2, 5) * u.g / u.cm**3
        result = law.opacity(rho_arr, _T)
        assert np.all(np.isfinite(np.atleast_1d(result.value)))
