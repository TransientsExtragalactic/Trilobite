"""Tests for :mod:`triceratops.radiation.opacity.utils`.

Organisation
------------
TestGetOpacityFloatInput    — float/int input → ConstantOpacity
TestGetOpacityStringInput   — each registered string → correct concrete type
TestGetOpacityObjectInput   — GreyOpacityLaw instance → returned unchanged
TestGetOpacityErrorCases    — unknown string → ValueError, wrong type → TypeError
"""

import pytest

from triceratops.radiation.opacity.base import GreyOpacityLaw
from triceratops.radiation.opacity.models.core import (
    ConstantOpacity,
    ElectronScatteringOpacity,
    KramersBFESOpacity,
    KramersBFOpacity,
    KramersESOpacity,
    KramersFFESOpacity,
    KramersFFOpacity,
    KramersOpacity,
)
from triceratops.radiation.opacity.utils import _OPACITY_REGISTRY, get_opacity


# ================================================================== #
# TestGetOpacityFloatInput                                            #
# ================================================================== #


class TestGetOpacityFloatInput:
    """get_opacity wraps float/int values in a ConstantOpacity instance."""

    def test_float_returns_constant_opacity(self):
        """A float input returns a ConstantOpacity instance."""
        result = get_opacity(0.5)
        assert isinstance(result, ConstantOpacity)

    def test_float_preserves_kappa_value(self):
        """The ConstantOpacity wrapping a float preserves the opacity value."""
        result = get_opacity(0.5)
        assert result.kappa == pytest.approx(0.5, rel=1e-12)

    def test_int_returns_constant_opacity(self):
        """An integer input is accepted and wraps in ConstantOpacity."""
        result = get_opacity(1)
        assert isinstance(result, ConstantOpacity)

    def test_int_preserves_kappa_value(self):
        """Integer input yields a ConstantOpacity with the correct float kappa."""
        result = get_opacity(1)
        assert result.kappa == pytest.approx(1.0, rel=1e-12)

    def test_zero_raises_no_error(self):
        """get_opacity(0.0) does not raise; produces a ConstantOpacity with kappa=0."""
        # Zero opacity is physically degenerate but should not raise at construction.
        result = get_opacity(0.0)
        assert isinstance(result, ConstantOpacity)

    @pytest.mark.parametrize("kappa", [0.1, 0.34, 1.0, 1e3])
    def test_various_float_values(self, kappa):
        """get_opacity correctly wraps several representative float values."""
        result = get_opacity(kappa)
        assert result.kappa == pytest.approx(kappa, rel=1e-12)


# ================================================================== #
# TestGetOpacityStringInput                                           #
# ================================================================== #


class TestGetOpacityStringInput:
    """get_opacity resolves every registered string to the correct concrete type."""

    @pytest.mark.parametrize(
        "name, expected_class",
        [
            ("electron_scattering", ElectronScatteringOpacity),
            ("kramers_ff", KramersFFOpacity),
            ("kramers_bf", KramersBFOpacity),
            ("kramers", KramersOpacity),
            ("kramers_ff_es", KramersFFESOpacity),
            ("kramers_bf_es", KramersBFESOpacity),
            ("kramers_es", KramersESOpacity),
        ],
    )
    def test_string_resolves_to_correct_type(self, name, expected_class):
        """Each registered string returns an instance of the expected class."""
        result = get_opacity(name)
        assert isinstance(result, expected_class), (
            f"get_opacity({name!r}) returned {type(result).__name__}, expected {expected_class.__name__}"
        )

    @pytest.mark.parametrize("name", list(_OPACITY_REGISTRY.keys()))
    def test_all_registered_names_return_opacity_law(self, name):
        """Every string in the registry returns an OpacityLaw instance."""
        from triceratops.radiation.opacity.base import OpacityLaw

        result = get_opacity(name)
        assert isinstance(result, OpacityLaw)

    def test_registry_is_complete(self):
        """The registry contains all expected opacity names (analytic + table-based)."""
        expected_names = {
            "electron_scattering",
            "kramers_ff",
            "kramers_bf",
            "kramers",
            "kramers_ff_es",
            "kramers_bf_es",
            "kramers_es",
            "opal",
        }
        assert set(_OPACITY_REGISTRY.keys()) == expected_names


# ================================================================== #
# TestGetOpacityObjectInput                                           #
# ================================================================== #


class TestGetOpacityObjectInput:
    """get_opacity returns an existing GreyOpacityLaw instance unchanged."""

    def test_constant_opacity_returned_unchanged(self):
        """A ConstantOpacity passed to get_opacity is returned as the same object."""
        law = ConstantOpacity(0.5)
        result = get_opacity(law)
        assert result is law

    def test_electron_scattering_returned_unchanged(self):
        """An ElectronScatteringOpacity instance is returned unchanged."""
        law = ElectronScatteringOpacity()
        result = get_opacity(law)
        assert result is law

    def test_kramers_ff_returned_unchanged(self):
        """A KramersFFOpacity instance is returned unchanged."""
        law = KramersFFOpacity()
        result = get_opacity(law)
        assert result is law

    def test_kramers_es_returned_unchanged(self):
        """A KramersESOpacity instance is returned unchanged."""
        law = KramersESOpacity()
        result = get_opacity(law)
        assert result is law


# ================================================================== #
# TestGetOpacityErrorCases                                            #
# ================================================================== #


class TestGetOpacityErrorCases:
    """get_opacity raises descriptive errors for invalid inputs."""

    def test_unknown_string_raises_value_error(self):
        """An unregistered string raises ValueError with a helpful message."""
        with pytest.raises(ValueError, match="Unknown opacity"):
            get_opacity("nonexistent_opacity")

    def test_unknown_string_error_mentions_available(self):
        """The ValueError message lists the available opacity names."""
        with pytest.raises(ValueError, match="Available"):
            get_opacity("typo_opacity")

    def test_none_raises_type_error(self):
        """None input raises TypeError."""
        with pytest.raises(TypeError):
            get_opacity(None)

    def test_list_raises_type_error(self):
        """A list input raises TypeError."""
        with pytest.raises(TypeError):
            get_opacity([0.34])

    def test_dict_raises_type_error(self):
        """A dict input raises TypeError."""
        with pytest.raises(TypeError):
            get_opacity({"kappa": 0.34})

    def test_type_error_message_mentions_type(self):
        """TypeError message names the offending input type."""
        with pytest.raises(TypeError, match="list"):
            get_opacity([0.34])

    @pytest.mark.parametrize(
        "bad_input",
        ["electron_Scattering", "KramersFF", "KRAMERS_FF", "  electron_scattering  "],
        ids=["wrong_case", "camel_case", "upper_case", "leading_spaces"],
    )
    def test_string_matching_is_exact(self, bad_input):
        """String lookup is exact (case-sensitive, no whitespace stripping)."""
        with pytest.raises(ValueError):
            get_opacity(bad_input)
