r"""
Tests for :mod:`trilobite.radiation.synchrotron.spectral_inversion`.

Each Diggins26 inversion is checked by a round trip through an *independent* forward model built from the primitive
definitions of :math:`\nu_a` and :math:`F_{\rm asym}` (no reduced constants), with the electron normalization taken
from :meth:`~trilobite.radiation.synchrotron.electron_distributions.PowerLaw.normalize_from_magnetic_field`.
"""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

from trilobite.radiation.synchrotron import PowerLaw
from trilobite.radiation.synchrotron.spectral_inversion import (
    invert_ssa_combined_cooled_Diggins26,
    invert_ssa_DeMarchi22,
    invert_ssa_Diggins26,
    invert_ssa_ic_cooled_Diggins26,
    invert_ssa_synchrotron_cooled_Diggins26,
    invert_ssa_with_cooling_Diggins26,
)
from trilobite.radiation.synchrotron.utils import (
    c_1_cgs,
    c_1_gamma_cgs,
    compute_c5_parameter,
    compute_c6_parameter,
)

MC2 = (const.m_e * const.c**2).cgs.value
ME = const.m_e.cgs.value
C = const.c.cgs.value
SIGMA_T = const.sigma_T.cgs.value
FNU = u.erg / u.s / u.cm**2 / u.Hz

MICRO = {"p": 3.0, "epsilon_E": 0.1, "epsilon_B": 0.01, "f_A": 0.5, "f_V": 0.5}
D_TEST = (60 * u.Mpc).to_value(u.cm)


def _pitch(pitch_average, alpha=np.pi / 2):
    """Return the ``(sin, A_s, kwargs)`` triple for a pitch-angle convention."""
    if pitch_average:
        return 1.0, 9 * np.pi * ME * C / SIGMA_T, {"pitch_average": True}
    return np.sin(alpha), 6 * np.pi * ME * C / (SIGMA_T * np.sin(alpha) ** 2), {"alpha": alpha}


def forward(B, R, t, L, *, mode, pitch_average=False, alpha=np.pi / 2, gamma_min=1.0, gamma_max=np.inf, **micro):
    """
    Asymptotic ``(nu_a, F_asym)`` from the primitive definitions.

    ``mode`` is ``"uncooled"`` or ``"cooled"``; in the cooled case the electron index is ``p + 1`` and the
    normalization of the cooled segment is ``N_0 * gamma_c``.
    """
    micro = {**MICRO, **micro}
    p = micro["p"]
    sin, A_s, _ = _pitch(pitch_average, alpha)
    N0 = PowerLaw.normalize_from_magnetic_field(
        B * u.G,
        epsilon_B=micro["epsilon_B"],
        epsilon_E=micro["epsilon_E"],
        p=p,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    ).to_value(u.cm**-3)
    A_IC = 3 * np.pi * MC2 / SIGMA_T
    gc = 1.0 / (t * (B**2 / A_s + L / (A_IC * R**2)))
    q, K = (p, N0) if mode == "uncooled" else (p + 1, N0 * gc)
    c5, c6 = compute_c5_parameter(q, pitch_average=pitch_average), compute_c6_parameter(q, pitch_average=pitch_average)
    Xv = (c6 * K * MC2 ** (q - 1) * (B * sin) ** ((q + 2) / 2) * micro["f_V"] ** (1 / 3) * R) ** (2 / (q + 4))
    nu_a = 2 * c_1_cgs * Xv
    F_a = (c5 / c6) * np.pi * micro["f_A"] * R**2 / D_TEST**2 * (B * sin) ** -0.5 * Xv**2.5
    return nu_a, F_a, gc


def _micro_kwargs(**over):
    return {**MICRO, **over}


# ----------------------------------------------------------------------------------------------------------------- #
# Uncooled
# ----------------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [2.5, 3.0, 3.5])
@pytest.mark.parametrize("pitch_average", [False, True])
@pytest.mark.parametrize("B, R", [(0.1, 1e16), (1.0, 5e15), (3.0, 1e17)])
def test_uncooled_round_trip(B, R, p, pitch_average):
    nu, F, _ = forward(B, R, 1.0, 0.0, mode="uncooled", pitch_average=pitch_average, p=p)
    out = invert_ssa_Diggins26(nu * u.Hz, F * FNU, D_TEST * u.cm, **_micro_kwargs(p=p), **_pitch(pitch_average)[2])
    assert out["B"].unit == u.G and out["R"].unit == u.cm
    np.testing.assert_allclose(out["B"].value, B, rtol=1e-9)
    np.testing.assert_allclose(out["R"].value, R, rtol=1e-9)


def test_uncooled_fixed_angle_round_trip():
    alpha = 0.7
    nu, F, _ = forward(0.5, 3e16, 1.0, 0.0, mode="uncooled", alpha=alpha)
    out = invert_ssa_Diggins26(nu, F, D_TEST, alpha=alpha, **MICRO)
    np.testing.assert_allclose(out["B"].value, 0.5, rtol=1e-9)
    np.testing.assert_allclose(out["R"].value, 3e16, rtol=1e-9)


def test_bare_floats_match_quantities():
    nu, F, _ = forward(0.5, 3e16, 1.0, 0.0, mode="uncooled")
    a = invert_ssa_Diggins26(nu, F, D_TEST, **MICRO)
    b = invert_ssa_Diggins26(nu * u.Hz, F * FNU, (D_TEST * u.cm).to(u.Mpc), **MICRO)
    np.testing.assert_allclose(a["B"].value, b["B"].value, rtol=1e-12)
    np.testing.assert_allclose(a["R"].value, b["R"].value, rtol=1e-12)


def test_broadcasting():
    nu, F, _ = forward(0.5, 3e16, 1.0, 0.0, mode="uncooled")
    out = invert_ssa_Diggins26(nu * np.ones(4), F * np.ones(4), D_TEST, **MICRO)
    assert out["B"].shape == (4,)


def test_pitch_average_warns_when_alpha_given():
    nu, F, _ = forward(0.5, 3e16, 1.0, 0.0, mode="uncooled", pitch_average=True)
    with pytest.warns(UserWarning, match="alpha"):
        invert_ssa_Diggins26(nu, F, D_TEST, alpha=0.3, pitch_average=True, **MICRO)


def test_pitch_average_differs_from_right_angle():
    nu, F, _ = forward(0.5, 3e16, 1.0, 0.0, mode="uncooled")
    a = invert_ssa_Diggins26(nu, F, D_TEST, **MICRO)
    b = invert_ssa_Diggins26(nu, F, D_TEST, pitch_average=True, **MICRO)
    assert not np.isclose(a["B"].value, b["B"].value, rtol=1e-3)


def test_divergent_moment_raises():
    with pytest.raises(ValueError, match="not finite"):
        invert_ssa_Diggins26(1e9, 1e-26, D_TEST, p=1.8, epsilon_E=0.1, epsilon_B=0.1)


def test_gamma_max_not_above_gamma_min_raises():
    with pytest.raises(ValueError, match="gamma_max"):
        invert_ssa_Diggins26(1e9, 1e-26, D_TEST, gamma_min=10.0, gamma_max=5.0)


@pytest.mark.parametrize("p", [1.7, 2.0, 2.5])
@pytest.mark.parametrize("gamma_min, gamma_max", [(1.0, 1e4), (5.0, 1e5), (30.0, 1e3)])
def test_round_trip_finite_gamma_bounds(p, gamma_min, gamma_max):
    """Covers p <= 2, which requires a finite gamma_max, and gamma_min != 1 (exercises the vectorized moment)."""
    kw = dict(p=p, gamma_min=gamma_min, gamma_max=gamma_max)
    nu, F, _ = forward(0.5, 3e16, 1.0, 0.0, mode="uncooled", **kw)
    out = invert_ssa_Diggins26(nu, F, D_TEST, **_micro_kwargs(**kw))
    np.testing.assert_allclose(out["B"].value, 0.5, rtol=1e-9)
    np.testing.assert_allclose(out["R"].value, 3e16, rtol=1e-9)

    nu, F, gc = forward(0.5, 3e16, 1e7, 0.0, mode="cooled", **kw)
    out = invert_ssa_synchrotron_cooled_Diggins26(nu, F, D_TEST, 1e7, **_micro_kwargs(**kw))
    np.testing.assert_allclose(out["B"].value, 0.5, rtol=1e-9)
    np.testing.assert_allclose(out["R"].value, 3e16, rtol=1e-9)
    np.testing.assert_allclose(out["gamma_c"], gc, rtol=1e-9)


def test_delta_c_matches_closed_form():
    p, gamma_min = 3.0, 4.0
    nu, F, gc = forward(0.5, 1e16, 1e7, 0.0, mode="cooled", p=p, gamma_min=gamma_min)
    out = invert_ssa_synchrotron_cooled_Diggins26(nu, F, D_TEST, 1e7, **_micro_kwargs(p=p, gamma_min=gamma_min))
    np.testing.assert_allclose(out["delta_c"], (gc / gamma_min) ** (2 - p) / (p - 1), rtol=1e-9)
    assert out["slow_cooling"] == (gc > gamma_min)
    # Undefined (NaN) for a finite gamma_max.
    out = invert_ssa_synchrotron_cooled_Diggins26(nu, F, D_TEST, 1e7, **_micro_kwargs(gamma_max=1e6))
    assert np.isnan(out["delta_c"])


# ----------------------------------------------------------------------------------------------------------------- #
# Broadcasting
# ----------------------------------------------------------------------------------------------------------------- #
def _call(name, nu, F, D, t, L, **kw):
    """Call the named Diggins26 inversion, passing only the arguments it takes."""
    kw = {**MICRO, **kw}
    return {
        "uncooled": lambda: invert_ssa_Diggins26(nu, F, D, **kw),
        "synchrotron": lambda: invert_ssa_synchrotron_cooled_Diggins26(nu, F, D, t, **kw),
        "ic": lambda: invert_ssa_ic_cooled_Diggins26(nu, F, D, t, L, **kw),
        "combined": lambda: invert_ssa_combined_cooled_Diggins26(nu, F, D, t, L, **kw),
        "with_cooling": lambda: invert_ssa_with_cooling_Diggins26(nu, F, D, t, L, **kw),
    }[name]()


def _assert_shapes(out, shape):
    for key, val in out.items():
        if isinstance(val, dict):
            _assert_shapes(val, shape)
        else:
            assert np.shape(val) == shape, f"{key}: {np.shape(val)} != {shape}"


ALL_INVERSIONS = ["uncooled", "synchrotron", "ic", "combined", "with_cooling"]


@pytest.mark.filterwarnings("ignore:.*nu_c >= nu_a.*")
@pytest.mark.parametrize("name", ALL_INVERSIONS)
def test_observables_broadcast_like_numpy(name):
    """A (3, 1, 1) x (2, 1) x (4,) set of observables gives a (3, 2, 4) grid equal to the loop of scalar calls."""
    nu = np.array([2e9, 5e9, 8e9]).reshape(3, 1, 1) * u.Hz
    F = np.array([5e-26, 2e-25]).reshape(2, 1) * FNU
    t = np.array([1e6, 3e6, 1e7, 3e7]) * u.s
    L = np.array([1e41, 1e42, 1e43, 1e44]) * u.erg / u.s
    out = _call(name, nu, F, D_TEST * u.cm, t, L)
    # The uncooled inversion does not take t or L, so it has no trailing axis.
    shape = (3, 2, 1) if name == "uncooled" else (3, 2, 4)
    _assert_shapes(out, shape)

    for i, j, k in np.ndindex(3, 2, 4):
        ref = _call(name, nu[i, 0, 0], F[j, 0], D_TEST * u.cm, t[k], L[k])
        kk = k if shape[2] > 1 else 0
        for key in ("B", "R"):
            np.testing.assert_allclose(out[key].value[i, j, kk], ref[key].value, rtol=1e-10)
        if "branch" in out:
            assert out["branch"][i, j, kk] == ref["branch"]


@pytest.mark.filterwarnings("ignore:.*nu_c >= nu_a.*")
@pytest.mark.parametrize("name", ALL_INVERSIONS)
def test_microphysics_broadcast_like_numpy(name):
    """Array-valued p, epsilon_B, alpha, f_A, f_V and gamma_max broadcast against each other and the observables."""
    nu, F, D, t, L = 5e9 * u.Hz, 1e-25 * FNU, D_TEST * u.cm, 5e6 * u.s, 1e42 * u.erg / u.s
    p = np.array([2.5, 3.0, 3.5])  # (3,)
    eps_B = np.array([0.01, 0.1]).reshape(2, 1)  # (2, 1)
    alpha = np.array([0.6, 1.0, np.pi / 2]).reshape(3, 1, 1)  # (3, 1, 1)
    f_A = np.array([0.3, 0.5]).reshape(2, 1)  # (2, 1)
    kw = dict(p=p, epsilon_B=eps_B, alpha=alpha, f_A=f_A, f_V=np.array([0.2, 0.9]).reshape(2, 1), gamma_max=1e6)
    out = _call(name, nu, F, D, t, L, **kw)
    shape = (3, 2, 3)
    _assert_shapes(out, shape)

    for i, j, k in np.ndindex(*shape):
        scalar = dict(
            p=p[k],
            epsilon_B=eps_B[j, 0],
            alpha=alpha[i, 0, 0],
            f_A=f_A[j, 0],
            f_V=kw["f_V"][j, 0],
            gamma_max=1e6,
        )
        ref = _call(name, nu, F, D, t, L, **scalar)
        for key in ("B", "R"):
            np.testing.assert_allclose(out[key].value[i, j, k], ref[key].value, rtol=1e-10)


@pytest.mark.filterwarnings("ignore:.*nu_c >= nu_a.*")
@pytest.mark.parametrize("name", ALL_INVERSIONS)
def test_scalar_inputs_give_scalar_outputs(name):
    out = _call(name, 5e9, 1e-25, D_TEST, 5e6, 1e42)
    _assert_shapes(out, ())


@pytest.mark.filterwarnings("ignore:.*nu_c >= nu_a.*")
@pytest.mark.parametrize("name", ALL_INVERSIONS)
def test_quantity_arrays_in_other_units(name):
    """Vector Quantities in non-CGS units give the same answer as the CGS values."""
    nu_g = np.array([2.0, 5.0]) * u.GHz
    F_mJy = np.array([5.0, 20.0]) * u.mJy
    t_d = np.array([20.0, 60.0]) * u.day
    L_lsun = np.array([1e8, 1e9]) * u.Lsun
    D_mpc = (D_TEST * u.cm).to(u.Mpc)

    a = _call(name, nu_g, F_mJy, D_mpc, t_d, L_lsun)
    b = _call(name, nu_g.to_value(u.Hz), F_mJy.to_value(FNU), D_TEST, t_d.to_value(u.s), L_lsun.to_value(u.erg / u.s))
    _assert_shapes(a, (2,))
    for key in ("B", "R"):
        np.testing.assert_allclose(a[key].value, b[key].value, rtol=1e-10)


def test_combined_solver_converges_per_element_on_mixed_grid():
    """Elements with very different Y converge independently inside one vectorized call."""
    nu, F, t = 150e9, (20 * u.mJy).to_value(FNU), 8 * 86400.0
    Ls = np.logspace(38, 48, 25).reshape(5, 5)
    out = invert_ssa_combined_cooled_Diggins26(nu, F, D_TEST, t, Ls, **MICRO)
    assert out["B"].shape == (5, 5) and out["n_iter"].shape == (5, 5)
    ref = np.array([invert_ssa_combined_cooled_Diggins26(nu, F, D_TEST, t, L, **MICRO)["B"].value for L in Ls.ravel()])
    np.testing.assert_allclose(out["B"].value.ravel(), ref, rtol=1e-10)


# ----------------------------------------------------------------------------------------------------------------- #
# Synchrotron cooled
# ----------------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [2.5, 3.0])
@pytest.mark.parametrize("pitch_average", [False, True])
@pytest.mark.parametrize("B, R, t", [(0.3, 1e16, 5e6), (1.0, 3e16, 2e7)])
def test_synchrotron_cooled_round_trip(B, R, t, p, pitch_average):
    nu, F, gc = forward(B, R, t, 0.0, mode="cooled", pitch_average=pitch_average, p=p)
    out = invert_ssa_synchrotron_cooled_Diggins26(nu, F, D_TEST, t, **_micro_kwargs(p=p), **_pitch(pitch_average)[2])
    np.testing.assert_allclose(out["B"].value, B, rtol=1e-9)
    np.testing.assert_allclose(out["R"].value, R, rtol=1e-9)
    np.testing.assert_allclose(out["gamma_c"], gc, rtol=1e-9)
    assert out["nu_c"].unit == u.Hz
    assert out["consistent"] == (out["nu_c_over_nu_a"] < 1)


def test_synchrotron_cooled_warns_when_regime_violated():
    # An uncooled source (nu_c >> nu_a) inverted under the cooled assumption.
    nu, F, _ = forward(1.0, 1e16, 1.0, 0.0, mode="uncooled")
    with pytest.warns(UserWarning, match="nu_c >= nu_a"):
        out = invert_ssa_synchrotron_cooled_Diggins26(nu, F, D_TEST, 1e3, **MICRO)
    assert not out["consistent"]


def test_isotropic_cooling_constant_is_not_sin_to_one():
    """Isotropic losses use 9 pi m_e c / sigma_T, not the sin(alpha) -> 1 limit 6 pi m_e c / sigma_T."""
    nu, F, _ = forward(0.5, 1e16, 1e7, 0.0, mode="cooled", pitch_average=True)
    iso = invert_ssa_synchrotron_cooled_Diggins26(nu, F, D_TEST, 1e7, pitch_average=True, **MICRO)
    np.testing.assert_allclose(iso["B"].value, 0.5, rtol=1e-9)
    # Reproducing the isotropic result with the right-angle cooling constant must fail.
    fixed = invert_ssa_synchrotron_cooled_Diggins26(nu, F, D_TEST, 1e7, **MICRO)
    assert not np.isclose(fixed["B"].value, 0.5, rtol=1e-2)


# ----------------------------------------------------------------------------------------------------------------- #
# IC cooled
# ----------------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [2.5, 3.0])
@pytest.mark.parametrize("pitch_average", [False, True])
@pytest.mark.parametrize("B, R, t, L", [(0.1, 1e16, 5e6, 1e44), (0.3, 3e16, 2e7, 1e45)])
def test_ic_cooled_round_trip(B, R, t, L, p, pitch_average):
    # Remove the synchrotron channel from the forward model by inflating its cooling constant is not possible, so
    # build gamma_c from IC only and check the IC-only inversion against it.
    micro = _micro_kwargs(p=p)
    N0 = PowerLaw.normalize_from_magnetic_field(
        B * u.G, epsilon_B=micro["epsilon_B"], epsilon_E=micro["epsilon_E"], p=p, gamma_min=1.0, gamma_max=np.inf
    ).to_value(u.cm**-3)
    gc = (3 * np.pi * MC2 / SIGMA_T) * R**2 / (L * t)
    sin, _, kw = _pitch(pitch_average)
    q = p + 1
    c5, c6 = compute_c5_parameter(q, pitch_average=pitch_average), compute_c6_parameter(q, pitch_average=pitch_average)
    Xv = (c6 * N0 * gc * MC2 ** (q - 1) * (B * sin) ** ((q + 2) / 2) * micro["f_V"] ** (1 / 3) * R) ** (2 / (q + 4))
    nu = 2 * c_1_cgs * Xv
    F = (c5 / c6) * np.pi * micro["f_A"] * R**2 / D_TEST**2 * (B * sin) ** -0.5 * Xv**2.5

    out = invert_ssa_ic_cooled_Diggins26(nu, F, D_TEST, t, L, **micro, **kw)
    np.testing.assert_allclose(out["B"].value, B, rtol=1e-9)
    np.testing.assert_allclose(out["R"].value, R, rtol=1e-9)
    np.testing.assert_allclose(out["gamma_c"], gc, rtol=1e-9)
    assert "Y" in out and "ic_dominated" in out


def test_ic_cooled_requires_positive_luminosity():
    with pytest.raises(ValueError, match="positive"):
        invert_ssa_ic_cooled_Diggins26(1e10, 1e-26, D_TEST, 1e6, 0.0)


# ----------------------------------------------------------------------------------------------------------------- #
# Combined
# ----------------------------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def draws():
    rng = np.random.default_rng(1)
    n = 5000
    return {
        "B": 10 ** rng.uniform(-1.5, 1.5, n),
        "R": 10 ** rng.uniform(14.5, 16.5, n),
        "t": 86400 * 10 ** rng.uniform(0.5, 2.5, n),
        "L": 10 ** rng.uniform(40, 45, n),
    }


@pytest.mark.parametrize("pitch_average", [False, True])
def test_combined_round_trip_over_Y(draws, pitch_average):
    d = draws
    nu, F, gc = forward(d["B"], d["R"], d["t"], d["L"], mode="cooled", pitch_average=pitch_average)
    # Keep draws where the cooled forward model is self-consistent (slow cooling and nu_c well below nu_a).
    sin = _pitch(pitch_average)[0]
    nu_c = c_1_gamma_cgs * d["B"] * sin * gc**2
    keep = (gc > 1) & (nu_c < nu / 3)
    assert keep.sum() > 100

    out = invert_ssa_combined_cooled_Diggins26(
        nu[keep],
        F[keep],
        D_TEST,
        d["t"][keep],
        d["L"][keep],
        **MICRO,
        **_pitch(pitch_average)[2],
    )
    np.testing.assert_allclose(out["B"].value, d["B"][keep], rtol=1e-9)
    np.testing.assert_allclose(out["R"].value, d["R"][keep], rtol=1e-9)
    assert out["n_iter"].max() < 50


def test_combined_limits_and_lower_bounds():
    """The combined solution matches each single-channel closed form in its limit and exceeds both everywhere."""
    nu, F, t = 150e9, (20 * u.mJy).to_value(FNU), 8 * 86400.0
    Ls = np.logspace(38, 48, 60)
    full = invert_ssa_combined_cooled_Diggins26(nu, F, D_TEST, t, Ls, **MICRO)
    sync = invert_ssa_synchrotron_cooled_Diggins26(nu, F, D_TEST, t, **MICRO)
    with pytest.warns(UserWarning):
        ic = invert_ssa_ic_cooled_Diggins26(nu, F, D_TEST, t, Ls, **MICRO)

    B = full["B"].value
    assert np.all(B >= sync["B"].value * (1 - 1e-12))
    assert np.all(B >= ic["B"].value * (1 - 1e-12))
    np.testing.assert_allclose(B[0], sync["B"].value, rtol=1e-3)  # Y << 1
    np.testing.assert_allclose(B[-1], ic["B"].value[-1], rtol=1e-3)  # Y >> 1


def test_combined_zero_luminosity_is_synchrotron_only():
    nu, F, t = 150e9, (20 * u.mJy).to_value(FNU), 8 * 86400.0
    out = invert_ssa_combined_cooled_Diggins26(nu, F, D_TEST, t, 0.0, **MICRO)
    sync = invert_ssa_synchrotron_cooled_Diggins26(nu, F, D_TEST, t, **MICRO)
    np.testing.assert_allclose(out["B"].value, sync["B"].value, rtol=1e-10)
    np.testing.assert_allclose(out["R"].value, sync["R"].value, rtol=1e-10)
    assert out["Y"] == 0.0


# ----------------------------------------------------------------------------------------------------------------- #
# Regime-checking wrapper
# ----------------------------------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("pitch_average", [False, True])
def test_with_cooling_selects_correct_branch(draws, pitch_average):
    d = draws
    sin = _pitch(pitch_average)[0]
    nu_u, F_u, gc = forward(d["B"], d["R"], d["t"], d["L"], mode="uncooled", pitch_average=pitch_average)
    nu_c, F_c, _ = forward(d["B"], d["R"], d["t"], d["L"], mode="cooled", pitch_average=pitch_average)
    nuc = c_1_gamma_cgs * d["B"] * sin * gc**2
    cooled = nuc < nu_c
    nu = np.where(cooled, nu_c, nu_u)
    F = np.where(cooled, F_c, F_u)
    keep = (np.abs(np.log10(nuc / nu)) > np.log10(3)) & (gc > 1)
    assert keep.sum() > 100

    out = invert_ssa_with_cooling_Diggins26(
        nu[keep], F[keep], D_TEST, d["t"][keep], d["L"][keep], **MICRO, **_pitch(pitch_average)[2]
    )
    expected = np.where(cooled[keep], "cooled", "uncooled")
    assert np.mean(out["branch"] == expected) > 0.999
    ok = out["branch"] == expected
    np.testing.assert_allclose(out["B"].value[ok], d["B"][keep][ok], rtol=1e-9)
    np.testing.assert_allclose(out["R"].value[ok], d["R"][keep][ok], rtol=1e-9)
    assert set(out["cooled"]) == set(out["uncooled"]) == {"B", "R", "gamma_c", "nu_c"}


def test_with_cooling_flags_fast_cooling_and_ambiguous():
    # Fast cooling: a very long cooling time pushes gamma_c below gamma_min in the cooled branch.
    out = invert_ssa_with_cooling_Diggins26(
        np.array([5e9, 5e9]), np.array([1e-26, 1e-26]), D_TEST, np.array([1e3, 1e12]), np.array([0.0, 0.0]), **MICRO
    )
    flagged = np.isin(out["branch"], ["fast_cooling", "ambiguous"])
    assert np.all(np.isnan(out["B"].value[flagged]))
    assert np.all(np.isnan(out["R"].value[flagged]))


# ----------------------------------------------------------------------------------------------------------------- #
# De Marchi (2022)
# ----------------------------------------------------------------------------------------------------------------- #
def test_demarchi_returns_dict_with_units():
    out = invert_ssa_DeMarchi22(5 * u.GHz, 10 * u.mJy, 10 * u.Mpc, p=3.0)
    assert out["B"].unit == u.G and out["R"].unit == u.cm
    assert np.isfinite(out["B"].value) and np.isfinite(out["R"].value)


def test_demarchi_bare_floats_are_ghz_jy_mpc():
    a = invert_ssa_DeMarchi22(5.0, 0.01, 10.0)
    b = invert_ssa_DeMarchi22(5 * u.GHz, 10 * u.mJy, 10 * u.Mpc)
    np.testing.assert_allclose(a["B"].value, b["B"].value, rtol=1e-12)


def test_demarchi_rejects_p_equal_2():
    with pytest.raises(ValueError, match="p = 2"):
        invert_ssa_DeMarchi22(5.0, 0.01, 10.0, p=2.0)


def test_demarchi_agrees_with_diggins26_at_fixed_angle():
    """Both inversions solve the same uncooled SSA problem; they differ only in microphysics bookkeeping."""
    nu, F = 5 * u.GHz, 10 * u.mJy
    dm = invert_ssa_DeMarchi22(nu, F, 10 * u.Mpc, p=3.0, f=0.5, epsilon_B=0.1, epsilon_E=0.1, gamma_min=1.0)
    dg = invert_ssa_Diggins26(
        nu, F, 10 * u.Mpc, p=3.0, f_A=0.5, f_V=0.5, epsilon_B=0.1, epsilon_E=0.1, gamma_min=1.0, gamma_max=1e6
    )
    # The same scalings hold, so the two should agree to the order of unity.
    assert 0.1 < dm["B"].value / dg["B"].value < 10
    assert 0.1 < dm["R"].value / dg["R"].value < 10


def test_demarchi_broadcasts_observables_and_microphysics():
    nu = np.array([2.0, 5.0, 8.0]).reshape(3, 1) * u.GHz
    F = np.array([5.0, 20.0]) * u.mJy
    p = np.array([2.5, 3.0, 3.5]).reshape(3, 1, 1)
    out = invert_ssa_DeMarchi22(nu, F, 10 * u.Mpc, p=p, gamma_max=np.array([1e5, 1e6]).reshape(2, 1, 1, 1))
    assert out["B"].shape == (2, 3, 3, 2)
    # Axes of the (2, 3, 3, 2) result are (gamma_max, p, nu, F); index (0, 1, 2, 0).
    ref = invert_ssa_DeMarchi22(nu[2, 0], F[0], 10 * u.Mpc, p=p[1, 0, 0], gamma_max=1e5)
    np.testing.assert_allclose(out["B"].value[0, 1, 2, 0], ref["B"].value, rtol=1e-10)
    np.testing.assert_allclose(out["R"].value[0, 1, 2, 0], ref["R"].value, rtol=1e-10)
