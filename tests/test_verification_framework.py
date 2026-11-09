"""
Tests of the verification machinery itself.

These are fast and always on. They do not run any verification; they check the opt-in mechanism and the bookkeeping that
ties verification tests to the models they verify and to ``ModelStatus.VERIFIED``.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from verification import registry
from verification.compare import Verifier, deviation, parse_records

from trilobite.models.core.status import ModelStatus
from trilobite.models.generic.curves import LinearModel

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent

# A real verification case may not exist yet, so the opt-in mechanism is tested on a throwaway project in a temporary
# directory: the real tests/conftest.py (and so the real hooks) next to one ordinary test and one verification test.
_DUMMY_VERIFICATION_TEST = """
import json


def test_records_a_comparison(request):
    for passed in (True, False):
        entry = {"label": "demo", "metric": "abs", "value": 0.1, "tolerance": 0.2, "passed": passed,
                 "test": request.node.nodeid, "models": [{"name": "SomeModel", "status": "unverified"}]}
        request.node.user_properties.append(("verification", json.dumps(entry)))
"""


@pytest.fixture
def project(tmp_path):
    """A throwaway project: the real conftest, an ordinary test and a verification test."""
    (tmp_path / "tests" / "verification").mkdir(parents=True)
    (tmp_path / "tests" / "conftest.py").write_text((TESTS / "conftest.py").read_text())
    (tmp_path / "tests" / "test_ordinary.py").write_text("def test_ordinary():\n    pass\n")
    (tmp_path / "tests" / "verification" / "test_demo.py").write_text(_DUMMY_VERIFICATION_TEST)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = --strict-markers -p no:cacheprovider\ntestpaths = tests\nmarkers =\n    verification: x\n"
    )
    return tmp_path


def _pytest(project, *args):
    result = subprocess.run([sys.executable, "-m", "pytest", *args], capture_output=True, text=True, cwd=project)
    return result.stdout + result.stderr


# =========================================================================== #
# Opt-in mechanism                                                            #
# =========================================================================== #
def test_verification_marker_is_registered(pytestconfig):
    assert any(line.startswith("verification:") for line in pytestconfig.getini("markers"))


def test_verification_is_deselected_by_default(project):
    out = _pytest(project, "--collect-only", "-q")
    assert "test_ordinary" in out
    assert "test_demo" not in out
    assert "1 deselected" in out


def test_run_verification_flag_selects_verification_tests(project):
    out = _pytest(project, "--collect-only", "-q", "--run-verification")
    assert "test_ordinary" in out and "test_demo" in out


def test_marker_expression_selects_only_verification_tests(project):
    out = _pytest(project, "--collect-only", "-q", "-m", "verification")
    assert "test_demo" in out
    assert "test_ordinary" not in out


def test_verification_tests_are_marked_by_directory_alone(project):
    # The dummy test carries no decorator; living under tests/verification is enough.
    out = _pytest(project, "-q", "-m", "verification", "--collect-only")
    assert "test_demo" in out


def test_default_run_writes_no_report(project):
    _pytest(project)
    assert not (project / "verification_report.json").exists()


def test_verification_run_prints_a_summary_and_writes_a_report(project):
    out = _pytest(project, "--run-verification", "--verification-report", "out.json")
    assert "model verification" in out
    assert "PASS  SomeModel [unverified]: demo" in out and "FAIL  SomeModel [unverified]: demo" in out
    report = json.loads((project / "out.json").read_text())
    assert report["n_comparisons"] == 2 and report["n_failed"] == 1
    assert [entry["passed"] for entry in report["results"]] == [True, False]


# =========================================================================== #
# Registration                                                                #
# =========================================================================== #
def test_every_verification_test_declares_what_it_verifies():
    assert registry.unregistered_tests() == []


def test_registered_cases_name_real_models():
    registry.load_all()
    models = set(registry.all_models())
    for case in registry.all_cases():
        for model in case.models:
            assert model in models, f"{case.test} verifies {model!r}, which is not a concrete in-tree model"


def test_verifies_rejects_bad_arguments():
    with pytest.raises(TypeError):
        registry.verifies()
    with pytest.raises(TypeError):
        registry.verifies(int)


def test_cases_are_not_inherited_by_subclasses(monkeypatch):
    monkeypatch.setattr(registry, "_CASES", [])
    child = type("_Child", (LinearModel,), {})

    @registry.verifies(LinearModel)
    def test_something():
        pass

    assert len(registry.cases_for(LinearModel)) == 1
    assert registry.cases_for(child) == []


# =========================================================================== #
# Status and evidence                                                         #
# =========================================================================== #
def test_verified_models_have_a_verification_test():
    registry.load_all()
    assert registry.verification_problems(registry.all_models()) == []


def test_verified_model_without_a_test_is_reported(monkeypatch):
    monkeypatch.setattr(registry, "_CASES", [])
    probe = type("_Probe", (LinearModel,), {"STATUS": ModelStatus.VERIFIED})
    problems = registry.verification_problems([probe])
    assert len(problems) == 1 and "no verification test" in problems[0]


def test_verified_model_with_a_test_passes(monkeypatch):
    monkeypatch.setattr(registry, "_CASES", [])
    probe = type("_Probe", (LinearModel,), {"STATUS": ModelStatus.VERIFIED})

    @registry.verifies(probe)
    def test_something():
        pass

    assert registry.verification_problems([probe]) == []


@pytest.mark.parametrize("status", [ModelStatus.NOVEL, ModelStatus.UNVERIFIED, ModelStatus.EXPERIMENTAL])
def test_other_statuses_need_no_test(status):
    probe = type("_Probe", (LinearModel,), {"STATUS": status})
    assert registry.verification_problems([probe]) == []


# =========================================================================== #
# Comparison helpers                                                          #
# =========================================================================== #
def _node():
    return SimpleNamespace(nodeid="tests/verification/test_x.py::test_y", user_properties=[], function=None, cls=None)


@pytest.mark.parametrize(
    "metric, actual, expected, value",
    [
        ("rel", [1.1, 2.0], [1.0, 2.0], 0.1),
        ("abs", [1.5, 2.0], [1.0, 2.0], 0.5),
        ("dex", [10.0, 1.0], [1.0, 1.0], 1.0),
    ],
)
def test_deviation_metrics(metric, actual, expected, value):
    assert deviation(actual, expected, metric) == pytest.approx(value)


def test_deviation_is_infinite_for_non_finite_values():
    assert deviation([1.0, np.nan], [1.0, 1.0]) == np.inf
    assert deviation(1.0, 0.0, "rel") == np.inf


def test_deviation_rejects_unknown_metric():
    with pytest.raises(ValueError):
        deviation(1.0, 1.0, "vibes")


def test_passing_comparison_is_recorded():
    node = _node()
    verifier = Verifier(node)
    assert verifier.agrees(1.001, 1.0, tol=0.01, label="close enough") == pytest.approx(0.001)
    (entry,) = parse_records(node.user_properties)
    assert entry["passed"] is True and entry["label"] == "close enough" and entry["tolerance"] == 0.01


def test_failing_comparison_is_recorded_and_raises():
    node = _node()
    verifier = Verifier(node)
    with pytest.raises(AssertionError, match="exceeds the tolerance"):
        verifier.agrees(2.0, 1.0, tol=0.01, label="far off")
    (entry,) = parse_records(node.user_properties)
    assert entry["passed"] is False and entry["value"] == pytest.approx(1.0)


def test_non_finite_deviation_is_recorded_as_json_safe_text():
    node = _node()
    with pytest.raises(AssertionError):
        Verifier(node).agrees(np.nan, 1.0, tol=0.1)
    assert json.loads(node.user_properties[0][1])["value"] == "inf"


def test_within_accepts_and_rejects_values_and_records_the_interval():
    node = _node()
    verifier = Verifier(node)
    verifier.within(0.26, 0.20, 0.31, label="inside")
    with pytest.raises(AssertionError, match="outside"):
        verifier.within(0.4, 0.20, 0.31, label="outside")
    inside, outside = parse_records(node.user_properties)
    assert inside["passed"] is True and outside["passed"] is False and inside["tolerance"] == [0.2, 0.31]


def test_records_carry_the_verified_models_and_their_status(monkeypatch):
    monkeypatch.setattr(registry, "_CASES", [])

    @registry.verifies(LinearModel)
    def test_something():
        pass

    node = SimpleNamespace(nodeid="n", user_properties=[], function=test_something, cls=None)
    Verifier(node).agrees(1.0, 1.0, tol=0.1)
    (entry,) = parse_records(node.user_properties)
    assert entry["models"] == [{"name": "LinearModel", "status": "unverified"}]
