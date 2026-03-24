"""
Tests for gP_esDisk.

Inherits all common tests from :class:`BaseTestOneZoneDisk`; this module only
declares the physical configuration and model-specific assertions.
"""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

from triceratops.dynamics.accretion.one_zone import gP_esDisk

from test_one_zone_disk_base import BaseTestOneZoneDisk


# ================================================================== #
# Concrete test class                                                #
# ================================================================== #


class TestgP_esDisk(BaseTestOneZoneDisk):
    """
    Tests for the canonical gas-pressure / electron-scattering disk.

    Physical setup
    --------------
    - 3 M☉ black hole, 0.1 M☉ disk, α = 0.1
    - R_D_0 / R_in = 3e13 / 3e6 = 1e7 ≫ 1  →  f(R) well-defined everywhere
    - t_visc_0 is bootstrapped self-consistently from the thermodynamic closure
    """

    MODEL = gP_esDisk
    DISK_KWARGS = {"mu": 0.62}

    M_BH = 3.0 * const.M_sun
    M_D_0 = 0.1 * const.M_sun
    R_IN = 3.0e6 * u.cm
    R_D_0 = 3.0e13 * u.cm
    ALPHA = 0.1

    # ——— Model-specific assertions ———

    def test_eos_mu(self):
        """EOS mean molecular weight matches the constructor argument."""
        assert np.isclose(self._disk().eos.mu, 0.62)

    def test_geometry_constants_accessible(self):
        """Geometry constants A, B, F are class attributes with Metzger+08 values."""
        disk = self._disk()
        assert np.isclose(disk._A, 1.62)
        assert np.isclose(disk._B, 1.33)
        assert np.isclose(disk._F, 1.6)

    def test_from_spec_dict_preserves_mu(self):
        """Round-trip through spec dict preserves the EOS mean molecular weight."""
        disk = self._disk()
        spec = disk.to_spec_dict()
        loaded = gP_esDisk.from_spec_dict(spec)
        assert np.isclose(loaded.eos.mu, disk.eos.mu)

    def test_from_spec_dict_geometry_constants_unchanged(self):
        """Geometry constants A, B, F are class-level and unchanged after round-trip."""
        disk = self._disk()
        spec = disk.to_spec_dict()
        loaded = gP_esDisk.from_spec_dict(spec)
        assert np.isclose(loaded._A, 1.62)
        assert np.isclose(loaded._B, 1.33)
        assert np.isclose(loaded._F, 1.6)
