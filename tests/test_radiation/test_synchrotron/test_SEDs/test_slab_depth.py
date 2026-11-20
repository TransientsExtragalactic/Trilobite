"""Every non-relativistic one-zone numerical SED uses the slab depth f_V^(1/3) R."""

import numpy as np
import pytest

from trilobite.radiation.synchrotron.SEDs import (
    Numerical_PowerLaw_Cooling_SSA_SynchrotronSED,
    Numerical_PowerLaw_SSA_SynchrotronSED,
    Numerical_Thermal_PowerLaw_SSA_SynchrotronSED,
    Numerical_Thermal_SSA_SynchrotronSED,
)

R_CGS, B_CGS, D_A_CGS = 1e16, 1.0, 3e26
CASES = {
    "power_law": (Numerical_PowerLaw_SSA_SynchrotronSED, (), {"p": 3.0}),
    "cooling": (Numerical_PowerLaw_Cooling_SSA_SynchrotronSED, (1e6,), {"p": 3.0}),
    "thermal": (Numerical_Thermal_SSA_SynchrotronSED, (), {"Theta": 5.0}),
    "mixed": (Numerical_Thermal_PowerLaw_SSA_SynchrotronSED, (), {"Theta": 5.0, "p": 3.0}),
}


@pytest.mark.parametrize("f_V", [1.0, 0.5, 0.125, 0.01])
@pytest.mark.parametrize("name", list(CASES))
def test_slab_depth_is_the_cube_root_of_the_volume_filling_factor_times_the_radius(monkeypatch, name, f_V):
    """The effective depth handed to the radiative transfer is f_V^(1/3) R."""
    sed_class, extra_positional, kwargs = CASES[name]
    sed = sed_class()
    captured = {}

    def spy(**engine_kwargs):
        captured.update(engine_kwargs)
        return np.zeros_like(engine_kwargs["log_nu"])

    monkeypatch.setattr(sed._engine, "_compute_log_pa_flux_density", spy)
    sed._log_opt_sed(np.array([1e9]), *extra_positional, B_CGS, R_CGS, D_A_CGS, f_V=f_V, **kwargs)

    slab_depth = captured["log_slab_depth"]
    assert slab_depth == pytest.approx(np.log(f_V ** (1.0 / 3.0) * R_CGS), rel=1e-12)
