# Model verification

Verification tests compare a model with published results or another independent reference. They are slower than the
unit tests, so they are deselected by default.

```bash
pytest --run-verification     # unit tests + verification
pytest -m verification        # verification only
```

A run prints a summary of every comparison and writes `verification_report.json` (`--verification-report` sets the
path). The `Model Verification` workflow runs it weekly and on demand.

## Status tags

A model's `STATUS` is `VERIFIED`, `NOVEL` (new physics with nothing published to compare with), or `UNVERIFIED` (the
default). The one enforced rule: a model can be `VERIFIED` only if a test here names it. Code review decides whether the
test is suitable.

## Adding a test

```python
from verification.registry import verifies

@verifies(SomeModel)
def test_reproduces_the_published_fit(verify):
    result = ...  # run the model
    verify.within(result, low, high, label="M_Ni")  # or verify.agrees(actual, expected, tol=..., metric="rel")
```

Everything under `tests/verification/` is marked `verification` automatically. Reference data goes in `data/` as ECSV
tables (see `data/README.md`); citations go in `docs/source/docs_bib.bib`.
