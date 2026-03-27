r"""
Assembled one-zone disk closures.

Each module here defines a concrete :class:`~..closure.OneZoneClosure` subclass
that combines an EOS, the standard alpha-disk viscous derivative, and a writer
into a self-contained closure ready for the integrator.

Closures
--------

_gP
    Gas-pressure EOS (:func:`~..physics._eos.compute_ideal_gas_cs`).
    Temperature solved iteratively via bracket-expansion + Brent's method.
_igP
    Full (gas + radiation) pressure EOS
    (:func:`~..physics._eos.compute_gas_rad_cs`).
    Same iterative solve; radiation pressure dominates at high :math:`T`.
_igP_adv
    Full pressure plus an advective cooling term
    :math:`q_{\\rm adv} = B\\,c_s^2\\,q_{\\rm visc}`, controlled by an
    entropy-gradient parameter :math:`\\xi` (``params.extra[0]``).
    Setting :math:`\\xi \\to 0` recovers the non-advective ``igP`` limit.

All three closures default to electron-scattering opacity and accept any
:class:`~triceratops.radiation.opacity.base.GreyOpacityLaw` via the
``opacity`` property.  Each can optionally install the power-law fallback
source term via ``with_fallback=True``.
"""
