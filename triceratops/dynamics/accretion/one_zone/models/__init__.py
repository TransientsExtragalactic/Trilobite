r"""
Assembled one-zone disk closures.

Each module here defines a concrete :class:`~..closure.OneZoneClosure` subclass
that combines an EOS, the standard alpha-disk viscous derivative, and a writer
into a self-contained closure ready for the integrator.

Closures
--------

_gP
    Defines :func:`~._gP.gP_closure_func` — the gas-pressure iterative
    temperature solve used when ``FullPressureClosure(gas_pressure_only=True)``.
_igP
    Defines :func:`~._igP.igP_closure_func` and
    :class:`~._igP.FullPressureClosure`.  ``FullPressureClosure`` is the
    single concrete class replacing the former ``gPClosure`` / ``igPClosure``
    pair; pass ``gas_pressure_only=True/False`` to select the EOS path.
_igP_adv
    Defines :class:`~._igP_adv.AdvectiveClosure` — full pressure plus an
    advective cooling term controlled by an entropy-gradient parameter
    :math:`\\xi` (stored in ``params.advection.xi``).
    Setting :math:`\\xi \\to 0` recovers the non-advective ``igP`` limit.

All closures default to electron-scattering opacity and accept any
:class:`~triceratops.radiation.opacity.base.GreyOpacityLaw` via the
``opacity`` property.  Call :meth:`~..closure.OneZoneClosure.bind_runtime_parameters`
before passing to the integrator; each can optionally install the power-law
fallback source term via ``with_fallback=True``.
"""
