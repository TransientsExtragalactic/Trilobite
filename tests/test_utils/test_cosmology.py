import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import Planck18

from triceratops.utils.config import triceratops_config
from triceratops.utils.cosmology import get_cosmology, resolve_cosmological_distances


def test_get_default_cosmology():
    """
    Ensure the default cosmology is correctly fetched from config.
    """
    triceratops_config["physics.default_cosmology"] = Planck18

    cosmo = get_cosmology()

    assert cosmo is Planck18


def test_resolve_from_redshift():
    """
    Check that distances can be computed from a supplied redshift.
    """
    z = 0.5

    result = resolve_cosmological_distances(redshift=z, cosmology=Planck18)

    assert "luminosity_distance" in result
    assert "angular_diameter_distance" in result
    assert "proper_distance" in result

    # Basic sanity checks
    assert result["luminosity_distance"].unit.is_equivalent(u.Mpc)
    assert result["angular_diameter_distance"].unit.is_equivalent(u.Mpc)
    assert result["proper_distance"].unit.is_equivalent(u.Mpc)

    # Known cosmological relation
    DL = result["luminosity_distance"]
    DA = result["angular_diameter_distance"]

    assert np.isclose((DL / DA).value, (1 + z) ** 2, rtol=1e-3)


def test_resolve_from_luminosity_distance():
    """
    Ensure redshift can be recovered from luminosity distance.
    """
    z_true = 0.3
    DL = Planck18.luminosity_distance(z_true)

    result = resolve_cosmological_distances(luminosity_distance=DL, cosmology=Planck18)

    z_est = result["redshift"]

    assert np.isclose(z_est, z_true, rtol=1e-3)


def test_resolve_input_validation():
    """
    Ensure the function raises when multiple inputs are supplied.
    """
    with pytest.raises(ValueError):
        resolve_cosmological_distances(
            redshift=0.1,
            luminosity_distance=1 * u.Gpc,
            cosmology=Planck18,
        )

    with pytest.raises(ValueError):
        resolve_cosmological_distances(cosmology=Planck18)
