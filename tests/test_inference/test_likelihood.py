"""Tests for :class:`~trilobite.inference.likelihood.base.LogGaussianCensoredLikelihood`."""

import numpy as np
import pytest
from scipy.stats import norm

from trilobite.data.core import InferenceData, Observable
from trilobite.inference import (
    GaussianCensoredLikelihood,
    InferenceProblem,
    LogGaussianCensoredLikelihood,
)
from trilobite.models.generic.curves import LinearModel

LN10 = np.log(10.0)
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
TRUE = {"m": 2.0, "b": 1.0}  # y = 3, 5, 7, 9, 11


def _data(value, error, upper=None, lower=None):
    return InferenceData(
        x={"x": X},
        observables={
            "y": Observable(
                value=np.asarray(value, dtype=float),
                error=np.asarray(error, dtype=float),
                upper=None if upper is None else np.asarray(upper, dtype=float),
                lower=None if lower is None else np.asarray(lower, dtype=float),
            )
        },
    )


@pytest.fixture
def model():
    return LinearModel()


@pytest.fixture
def truth(model):
    return np.asarray(model.forward_model({"x": X}, TRUE).y, dtype=float)


# =================================================================== #
# Analytic agreement                                                  #
# =================================================================== #
def test_detections_match_analytic_expression(model, truth):
    y_obs = truth * np.array([1.1, 0.9, 1.05, 0.8, 1.2])
    err = 0.1 * y_obs
    sys_dex = 0.15
    like = LogGaussianCensoredLikelihood(model, _data(y_obs, err), sys_dex=sys_dex)

    sigma = np.sqrt((err / (y_obs * LN10)) ** 2 + sys_dex**2)
    resid = (np.log10(y_obs) - np.log10(truth)) / sigma
    expected = -0.5 * np.sum(resid**2 + np.log(2 * np.pi * sigma**2))

    assert like.log_likelihood(TRUE) == pytest.approx(expected, rel=1e-12)


def test_upper_limit_matches_analytic_expression(model, truth):
    value = truth.copy()
    err = 0.1 * truth
    upper = np.full_like(truth, np.nan)
    value[-1] = np.nan
    upper[-1] = 4.0  # model predicts 11, so this limit is violated
    sys_dex = 0.2
    like = LogGaussianCensoredLikelihood(model, _data(value, err, upper=upper), sys_dex=sys_dex)

    det_sigma = np.sqrt((err[:-1] / (truth[:-1] * LN10)) ** 2 + sys_dex**2)
    lim_sigma = np.sqrt((err[-1] / (4.0 * LN10)) ** 2 + sys_dex**2)
    expected = -0.5 * np.sum(np.log(2 * np.pi * det_sigma**2))  # detections sit exactly on the model
    expected += norm.logcdf(np.log10(4.0), loc=np.log10(truth[-1]), scale=lim_sigma)

    assert like.log_likelihood(TRUE) == pytest.approx(expected, rel=1e-12)


def test_lower_limit_matches_analytic_expression(model, truth):
    value = truth.copy()
    err = 0.1 * truth
    lower = np.full_like(truth, np.nan)
    value[0] = np.nan
    lower[0] = 1.0  # model predicts 3, consistent with the limit
    like = LogGaussianCensoredLikelihood(model, _data(value, err, lower=lower), sys_dex=0.1)

    det_sigma = np.sqrt((err[1:] / (truth[1:] * LN10)) ** 2 + 0.1**2)
    lim_sigma = np.sqrt((err[0] / (1.0 * LN10)) ** 2 + 0.1**2)
    expected = -0.5 * np.sum(np.log(2 * np.pi * det_sigma**2))
    expected += norm.logsf(np.log10(1.0), loc=np.log10(truth[0]), scale=lim_sigma)

    assert like.log_likelihood(TRUE) == pytest.approx(expected, rel=1e-12)


def test_small_error_limit_matches_linear_likelihood(model, truth):
    """For small fractional errors, log and linear chi^2 differences agree."""
    data = _data(truth, 0.01 * truth)
    log_like = LogGaussianCensoredLikelihood(model, data)
    lin_like = GaussianCensoredLikelihood(model, data)

    p1, p2 = TRUE, {"m": 2.02, "b": 1.0}
    d_log = log_like.log_likelihood(p2) - log_like.log_likelihood(p1)
    d_lin = lin_like.log_likelihood(p2) - lin_like.log_likelihood(p1)

    assert d_log < 0
    assert d_log == pytest.approx(d_lin, rel=0.05)


def test_symmetric_in_ratio(model):
    """Over- and under-predicting by the same factor costs the same in log space."""
    const = _data(np.full(5, 10.0), np.full(5, 1.0))
    like_c = LogGaussianCensoredLikelihood(model, const, sys_dex=0.2)
    up = like_c.log_likelihood({"m": 0.0, "b": 100.0})
    down = like_c.log_likelihood({"m": 0.0, "b": 1.0})
    assert up == pytest.approx(down, rel=1e-12)


# =================================================================== #
# Invalid model output                                                #
# =================================================================== #
@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_invalid_model_output_gives_neg_inf(model, truth, monkeypatch, bad):
    like = LogGaussianCensoredLikelihood(model, _data(truth, 0.1 * truth), sys_dex=0.2)
    out = np.array(truth)
    out[2] = bad
    monkeypatch.setattr(model, "_forward_model_tupled", lambda x, p: (out,))
    assert like.log_likelihood(TRUE) == -np.inf


def test_zero_model_output_is_finite_and_consistent_with_upper_limit(model, truth, monkeypatch):
    value = truth.copy()
    upper = np.full_like(truth, np.nan)
    value[-1] = np.nan
    upper[-1] = 5.0
    like = LogGaussianCensoredLikelihood(model, _data(value, 0.1 * truth, upper=upper), sys_dex=0.2)

    out = np.array(truth)
    out[-1] = 0.0
    monkeypatch.setattr(model, "_forward_model_tupled", lambda x, p: (out,))
    assert np.isfinite(like.log_likelihood(TRUE))

    # A zero model value at a *detection* is penalized very strongly, but not -inf.
    out2 = np.array(truth)
    out2[0] = 0.0
    monkeypatch.setattr(model, "_forward_model_tupled", lambda x, p: (out2,))
    assert like.log_likelihood(TRUE) < -1e5


# =================================================================== #
# Validation                                                          #
# =================================================================== #
@pytest.mark.parametrize("sys_dex", [-0.1, np.nan, np.inf])
def test_invalid_sys_dex_raises(model, truth, sys_dex):
    with pytest.raises(ValueError, match="sys_dex"):
        LogGaussianCensoredLikelihood(model, _data(truth, 0.1 * truth), sys_dex=sys_dex)


def test_nonpositive_data_raises(model, truth):
    bad = truth.copy()
    bad[1] = 0.0
    with pytest.raises(ValueError, match="positive"):
        LogGaussianCensoredLikelihood(model, _data(bad, 0.1 * truth))


def test_zero_width_requires_sys_dex(model, truth):
    data = _data(truth, np.zeros_like(truth))
    with pytest.raises(ValueError, match="nonzero uncertainty"):
        LogGaussianCensoredLikelihood(model, data)
    # A systematic floor makes zero reported errors acceptable.
    like = LogGaussianCensoredLikelihood(model, data, sys_dex=0.1)
    assert np.isfinite(like.log_likelihood(TRUE))


# =================================================================== #
# Integration                                                         #
# =================================================================== #
def test_inference_problem_maps_invalid_model_to_neg_inf(model, truth):
    like = LogGaussianCensoredLikelihood(model, _data(truth, 0.1 * truth), sys_dex=0.2)
    problem = InferenceProblem(likelihood=like)
    problem.set_prior("m", "uniform", lower=-5.0, upper=5.0)
    problem.set_prior("b", "uniform", lower=-5.0, upper=5.0)
    problem.parameters["m"].initial_value = 2.0
    problem.parameters["b"].initial_value = 1.0

    assert np.isfinite(problem._log_free_posterior(problem.initial_theta))
    # m = -5 drives the model flux negative at larger x: -inf, not an exception.
    theta = np.array([-5.0 if n == "m" else 1.0 for n in problem.parameter_order])
    assert problem._log_free_posterior(theta) == -np.inf
