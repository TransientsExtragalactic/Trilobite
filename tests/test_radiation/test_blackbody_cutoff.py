"""Tests of the UV-suppressed (modified) blackbody and of the redshift of the blackbody flux."""

import astropy.units as u
import numpy as np
import pytest
from astropy.cosmology import Planck18
from scipy.integrate import quad, trapezoid

from trilobite.radiation.blackbody import (
    _cutoff_transmitted_fraction_cgs,
    _log_planck_flambda_cgs,
    _log_planck_fnu_cgs,
    _log_specific_flux_flambda_cgs,
    _log_specific_flux_fnu_cgs,
    planck_cutoff_fraction,
    planck_flux,
    planck_luminosity,
)
from trilobite.radiation.constants import c_cgs, sigma_sb_cgs

_LAMBDA_CUT = 3.0e-5  # 3000 Angstrom


class TestTransmittedFraction:
    @pytest.mark.parametrize("T", [3000.0, 6000.0, 1.0e4, 2.0e4, 3.0e4])
    def test_matches_direct_integration(self, T):
        def planck(lam):
            return np.exp(_log_planck_flambda_cgs(np.log(lam), np.log(T)))

        suppressed = (
            quad(lambda lam: planck(lam) * lam / _LAMBDA_CUT, 1e-7, _LAMBDA_CUT, limit=200)[0]
            + quad(planck, _LAMBDA_CUT, 1e-1, limit=200)[0]
        )
        total = quad(planck, 1e-7, 1e-1, limit=400)[0]
        assert _cutoff_transmitted_fraction_cgs(T, _LAMBDA_CUT) == pytest.approx(suppressed / total, rel=1e-7)

    def test_limits(self):
        assert _cutoff_transmitted_fraction_cgs(1.0e4, 0.0) == 1.0
        assert _cutoff_transmitted_fraction_cgs(1.0e4, -1.0) == 1.0
        assert _cutoff_transmitted_fraction_cgs(1.0e4, 1.0e-9) == pytest.approx(1.0)  # cutoff far in the X-ray
        assert _cutoff_transmitted_fraction_cgs(1.0e4, 1.0e-1) < 1e-3  # cutoff in the radio: all suppressed

    def test_decreases_with_temperature(self):
        f = _cutoff_transmitted_fraction_cgs(np.array([4000.0, 8000.0, 1.6e4, 3.2e4]), _LAMBDA_CUT)
        assert np.all(np.diff(f) < 0)
        assert np.all((f > 0) & (f <= 1))

    def test_broadcasts(self):
        T = np.array([[5000.0], [1.0e4]])
        assert _cutoff_transmitted_fraction_cgs(T, _LAMBDA_CUT).shape == (2, 1)


class TestCutoffFlux:
    R, D = 1.0e15, 3.0e26

    def _flux(self, nu, T, z, lambda_cut):
        return np.exp(_log_specific_flux_fnu_cgs(np.log(nu), np.log(T), np.log(self.R), np.log(self.D), z, lambda_cut))

    @pytest.mark.parametrize("z", [0.0, 0.5, 2.0])
    def test_no_cutoff_is_a_blackbody_with_the_correct_redshift(self, z):
        nu = np.logspace(13, 16, 20)
        T = 1.0e4
        expected = (
            np.pi * np.exp(_log_planck_fnu_cgs(np.log(nu * (1 + z)), np.log(T))) * (self.R / self.D) ** 2 * (1 + z)
        )
        np.testing.assert_allclose(self._flux(nu, T, z, 0.0), expected, rtol=1e-12)

    @pytest.mark.parametrize("z", [0.0, 0.5])
    @pytest.mark.parametrize("T", [8.0e3, 1.5e4, 3.0e4])
    @pytest.mark.parametrize("lambda_cut", [0.0, _LAMBDA_CUT])
    def test_bolometric_flux_is_conserved(self, T, z, lambda_cut):
        """Suppression moves flux to longer wavelengths without changing the total, L / 4 pi D_L^2."""
        nu = np.logspace(11.0, 17.5, 400_000)
        bolometric = trapezoid(self._flux(nu, T, z, lambda_cut), nu)
        assert bolometric == pytest.approx(self.R**2 * sigma_sb_cgs * T**4 / self.D**2, rel=1e-6)

    def test_shape_of_the_suppression(self):
        z, T = 0.3, 1.2e4
        wavelength = np.array([500.0, 1500.0, 2999.0, 3001.0, 4000.0, 8000.0]) * 1e-8  # rest frame
        nu_obs = c_cgs / wavelength / (1 + z)
        ratio = self._flux(nu_obs, T, z, _LAMBDA_CUT) / self._flux(nu_obs, T, z, 0.0)
        fraction = _cutoff_transmitted_fraction_cgs(T, _LAMBDA_CUT)
        expected = np.minimum(wavelength / _LAMBDA_CUT, 1.0) / fraction
        np.testing.assert_allclose(ratio, expected, rtol=1e-10)
        assert ratio[0] < 1.0 < ratio[-1]  # the UV is dimmer and the optical brighter than the blackbody


class TestCutoffFluxInWavelength:
    R, D = 1.0e15, 3.0e26

    @pytest.mark.parametrize("z", [0.0, 0.5, 2.0])
    @pytest.mark.parametrize("lambda_cut", [0.0, _LAMBDA_CUT])
    def test_bolometric_flux_is_conserved(self, z, lambda_cut):
        """The same as in frequency: the integral of f_lambda over the observer wavelength is L / 4 pi D_L^2."""
        T = 1.2e4
        wavelength = np.geomspace(1e-7, 3e-3, 600_000)
        f = np.exp(
            _log_specific_flux_flambda_cgs(np.log(wavelength), np.log(T), np.log(self.R), np.log(self.D), z, lambda_cut)
        )
        bolometric = trapezoid(f, wavelength)
        assert bolometric == pytest.approx(self.R**2 * sigma_sb_cgs * T**4 / self.D**2, rel=2e-4)

    def test_frequency_and_wavelength_agree(self):
        """f_lambda = f_nu c / lambda^2 for the observed spectrum, with and without the cutoff."""
        z, T = 0.7, 9.0e3
        wavelength = np.geomspace(2e-6, 2e-4, 25)
        for lambda_cut in (0.0, _LAMBDA_CUT):
            f_lam = np.exp(
                _log_specific_flux_flambda_cgs(
                    np.log(wavelength), np.log(T), np.log(self.R), np.log(self.D), z, lambda_cut
                )
            )
            f_nu = np.exp(
                _log_specific_flux_fnu_cgs(
                    np.log(c_cgs / wavelength), np.log(T), np.log(self.R), np.log(self.D), z, lambda_cut
                )
            )
            np.testing.assert_allclose(f_lam, f_nu * c_cgs / wavelength**2, rtol=1e-10)


class TestPublicAPI:
    def test_planck_flux_is_bolometrically_correct_at_redshift(self):
        T, R, z = 1.0e4 * u.K, 1.0e15 * u.cm, 0.4
        nu = np.geomspace(1e12, 3e17, 200_000) * u.Hz
        D_L = Planck18.luminosity_distance(z).to(u.cm)
        expected = (R**2 * 5.670374419e-5 * u.erg / u.s / u.cm**2 / u.K**4 * T**4 / D_L**2).to_value(
            u.erg / u.s / u.cm**2
        )
        for cut in (None, 3000 * u.AA):
            flux = planck_flux(T, R, nu=nu, redshift=z, cosmology=Planck18, lambda_cut=cut)
            assert trapezoid(flux.value, nu.value) == pytest.approx(expected, rel=1e-6)

    def test_lambda_cut_units_and_default(self):
        T, R, nu = 1.2e4 * u.K, 1.0e15 * u.cm, np.geomspace(1e14, 3e15, 8) * u.Hz
        plain = planck_flux(T, R, nu=nu, redshift=0.1)
        np.testing.assert_allclose(plain.value, planck_flux(T, R, nu=nu, redshift=0.1, lambda_cut=None).value)
        in_angstrom = planck_flux(T, R, nu=nu, redshift=0.1, lambda_cut=3000 * u.AA)
        in_cm = planck_flux(T, R, nu=nu, redshift=0.1, lambda_cut=3.0e-5)  # a bare number is cm
        np.testing.assert_allclose(in_angstrom.value, in_cm.value)
        assert in_angstrom.unit == plain.unit

    def test_the_uv_is_dimmer_and_the_optical_brighter(self):
        T, R = 2.0e4 * u.K, 1.0e15 * u.cm
        lam = [1500.0, 2500.0, 3500.0, 6000.0] * u.AA
        ratio = planck_luminosity(T, R, lam=lam, lambda_cut=3000 * u.AA) / planck_luminosity(T, R, lam=lam)
        # Linear below the cutoff, constant above it. The renormalisation puts the spectrum above the blackbody wherever
        # lambda / lambda_cut > f(T), so only the far UV is dimmer.
        ratio = ratio.to_value(u.dimensionless_unscaled)
        assert ratio[0] < 1.0 < ratio[1] < ratio[2]
        assert ratio[2] == pytest.approx(ratio[3])
        assert ratio[0] == pytest.approx((1500.0 / 3000.0) / planck_cutoff_fraction(T, 3000 * u.AA))
        # Above the cutoff the ratio is exactly 1 / f(T), the renormalisation
        np.testing.assert_allclose(ratio[2:], 1.0 / planck_cutoff_fraction(T, 3000 * u.AA))

    def test_luminosity_integrates_to_the_stefan_boltzmann_luminosity(self):
        T, R = 1.5e4 * u.K, 1.0e15 * u.cm
        nu = np.geomspace(1e12, 3e17, 300_000) * u.Hz
        expected = 4 * np.pi * R**2 * 5.670374419e-5 * u.erg / u.s / u.cm**2 / u.K**4 * T**4
        for cut in (None, 3000 * u.AA):
            L_nu = planck_luminosity(T, R, nu=nu, lambda_cut=cut)
            assert trapezoid(L_nu.value, nu.value) == pytest.approx(expected.to_value(u.erg / u.s), rel=1e-6)

    def test_cutoff_fraction(self):
        fraction = planck_cutoff_fraction([8000, 15000, 30000] * u.K, 3000 * u.AA)
        assert fraction.shape == (3,)
        np.testing.assert_allclose(fraction, [0.97469838, 0.81621552, 0.52231366], rtol=1e-7)
        assert planck_cutoff_fraction(1.0e4 * u.K, 0 * u.AA) == 1.0
