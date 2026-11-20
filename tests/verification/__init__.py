"""
Model verification suite.

Verification tests compare Trilobite's models with published results or another independent reference. They are slower
than the unit tests, and a model may only be marked ``ModelStatus.VERIFIED`` if one exists for it, so they are
**deselected by default**. Run them with::

    pytest --run-verification          # the unit tests plus the verification tests
    pytest -m verification             # only the verification tests

See ``README.md`` in this directory for how to add a case.
"""
