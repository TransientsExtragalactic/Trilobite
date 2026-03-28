#cython: language_level=3, boundscheck=False
r"""
[C-LEVEL] Abstract closure framework for one-zone accretion disk models.

Defines the shared C-level interface, structs, function-pointer typedefs,
physical constants, and :class:`OneZoneClosure`, that every disk model
implementation must satisfy.

See Also
--------
:mod:`triceratops.dynamics.accretion.one_zone.integrator` :
    Explicit-Euler hot loop that consumes a :class:`OneZoneClosure`.
:mod:`triceratops.dynamics.accretion.one_zone._sources` :
    Optional source-term functions (e.g. fallback supply).
"""
from libc.math cimport log, pi
from cpython.ref cimport PyObject
from triceratops.radiation.opacity.opacity_base cimport C_GreyOpacityBase


# ---------------------------------------- #
# Constants                                #
# ---------------------------------------- #
# Physical constants — defined here, declared in closure.pxd so any .pyx
# can cimport them.  Single source of truth; no DEF duplication across files.
#
# DEVELOPER NOTE: If you are writing a new closure, your constants should be declared
# here and imported, not hard coded elsewhere. This ensures a single source of truth.

cdef double DISK_A   = 1.62                # Metzger+08 disk correction factor (dimensionless)
cdef double DISK_B   = 1.33                # Metzger+08 disk correction factor (dimensionless)
cdef double DISK_F0  = 1.6                 # Metzger+08 disk correction factor (dimensionless)
cdef double DISK_XI  = DISK_B / DISK_A    # Metzger+08 disk correction factor (dimensionless)
cdef double G_CGS    = 6.67430e-8          # Gravitational constant (cm^3 g^-1 s^-2)
cdef double K_B_CGS  = 1.380649e-16        # Boltzmann constant (erg K^-1)
cdef double M_P_CGS  = 1.67262192369e-24   # Proton mass (g)
cdef double KAPPA_ES = 0.34                # Electron scattering opacity (cm^2 g^-1)
cdef double RAD_A_CGS    = 7.5657e-15      # Radiation constant (erg cm^-3 K^-4)
cdef double SIGMA_SB_CGS = 5.6703744e-5    # Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4)

# Precomputed logs — log() is not a compile-time expression.
cdef double LOG_DISK_XI      = log(DISK_XI)       # log of xi (dimensionless)
cdef double LOG_DISK_A       = log(DISK_A)        # log of A  (dimensionless)
cdef double LOG_DISK_F0      = log(DISK_F0)       # log of F0 (dimensionless)
cdef double LOG_G_CGS        = log(G_CGS)         # log of G  (cm^3 g^-1 s^-2)
cdef double LOG_PI           = log(pi)            # log of pi (dimensionless)
cdef double LOG_K_B_CGS      = log(K_B_CGS)       # log of k_B (erg K^-1)
cdef double LOG_M_P_CGS      = log(M_P_CGS)       # log of m_p (g)
cdef double LOG_KAPPA_ES     = log(KAPPA_ES)      # log of kappa_es (cm^2 g^-1)
cdef double LOG_RAD_A_CGS    = log(RAD_A_CGS)     # log of a_rad (erg cm^-3 K^-4)
cdef double LOG_SIGMA_SB_CGS = log(SIGMA_SB_CGS)  # log of sigma_SB (erg cm^-2 s^-1 K^-4)


# ---------------------------------------- #
# OneZoneClosure extension type            #
# ---------------------------------------- #

cdef class OneZoneClosure:
    """
    Abstract base for one-zone disk closures.

    Concrete closures are Cython extension types that subclass this class and
    install C function pointers in ``__cinit__``.  The integrator calls those
    pointers in a tight ``nogil`` loop with no Python overhead at runtime.

    This is the primary point of extension for new physics in accretion disk models.
    Developers should implement new closure functions and bundle them into a single
    closure object in order to generate a new disk model.

    Attributes
    ----------
    n_result_fields : int
        Number of output columns written per time step. The ``writer_func`` attached to the closure
        must write exactly this many fields to the result array; the integrator allocates the array
        with this width before the first step.  Set in ``__cinit__``.

        The python layer is given a set of output fields and indices referring to their respective
        location in the resulting array.

    opacity : ~radiation.opacity.base.GreyOpacityLaw or ~radiation.opacity.opacity_base.C_GreyOpacityBase
        An opacity object which provides access to C-level callables for computing opacities as
        needed inside of the thermodynamics layer of the integrator.

        If a Python-level ``GreyOpacityLaw`` is assigned, the closure extracts the underlying
        C object and retains a reference to the Python wrapper for GC safety.  If a
        C-level ``C_GreyOpacityBase`` is assigned, it is used directly and no Python
        reference is retained.

        The corresponding C-level object is passed directly to the integrator at runtime.

    Notes
    -----
    **Function pointers**:

    To implement a closure, there are 4 function pointers; three mandatory, one optional:

    * ``_closure_fn`` — solves for the thermodynamic state given the current
      disk state; fills ``ClosureResult`` (``T_c``, ``c_s``, ``tau``, ``nu``,
      ``q_visc``, ``t_visc``).
    * ``_derivative_fn`` — computes ``dM/dt``, ``dJ/dt``, and adaptive ``dt``
      from the closure result; fills ``DiskStep``.
    * ``_writer_fn`` — packs one timestep of output into the result array.
    * ``_source_fn`` — optional; modifies ``DiskStep`` to inject a mass or
      angular-momentum source (e.g. fallback supply). ``NULL`` = no source.

    All pointers are initialised to ``NULL`` by the base ``__cinit__``.
    Call :meth:`is_ready` to confirm mandatory pointers are set before passing
    the closure to the integrator.

    **Opacity** — set ``self.opacity`` in ``__cinit__``.  ``_c_opacity``
    (typed ``C_GreyOpacityBase``) keeps the object alive for GC safety; the
    integrator extracts a raw ``PyObject*`` into ``DiskParameters.opacity``
    for GIL-free vtable dispatch inside the hot loop.

    Examples
    --------
    Minimal closure skeleton

    .. code-block:: python

        # my_closure.pyx
        from triceratops.dynamics.accretion.one_zone.closure cimport (
            OneZoneClosure, ClosureResult, DiskState, DiskDerived, DiskParameters,
        )
        from triceratops.dynamics.accretion.one_zone._writer cimport (
            N_RESULT_FIELDS, standard_writer_func,
        )
        from triceratops.dynamics.accretion.one_zone.physics._viscous cimport (
            viscous_derivative_func,
        )

        cdef int my_closure_func(
            const DiskState* state, const DiskDerived* derived,
            const DiskParameters* params, const ClosureResult* prev,
            ClosureResult* out,
        ) nogil:
            # ... solve for T_c; fill out.log_T_c, out.log_cs, etc. ...
            return 0

        cdef class MyClosure(OneZoneClosure):
            def __cinit__(self):
                from triceratops.radiation.opacity.models.core import ElectronScatteringOpacity
                self._closure_fn     = my_closure_func
                self._derivative_fn  = viscous_derivative_func
                self._writer_fn      = standard_writer_func
                self.n_result_fields = N_RESULT_FIELDS
                self.opacity         = ElectronScatteringOpacity()

    To add a fallback source term, import
    :func:`~triceratops.dynamics.accretion.one_zone.physics._fallback.fallback_source_func`
    and assign it to ``self._source_fn``.
    """

    def __cinit__(self, int n_result_fields=0, *args, **kwargs):
        self._closure_fn     = NULL
        self._derivative_fn  = NULL
        self._writer_fn      = NULL
        self._source_fn      = NULL
        self._c_opacity      = None
        self._opacity_obj    = None
        self._opacity_ptr    = NULL
        self.n_result_fields = n_result_fields

    cpdef bint is_ready(self):
        """True if the three mandatory function pointers are installed."""
        return (self._closure_fn    != NULL and
                self._derivative_fn != NULL and
                self._writer_fn     != NULL)

    @property
    def opacity(self):
        """The installed opacity law (a :class:`~triceratops.radiation.opacity.base.GreyOpacityLaw` instance)."""
        return self._opacity_obj

    @opacity.setter
    def opacity(self, obj):
        from triceratops.radiation.opacity.base import GreyOpacityLaw

        # Check the typing and extract the relevant pieces.
        if isinstance(obj, GreyOpacityLaw):
            c_obj = obj._c_object
        elif isinstance(obj, C_GreyOpacityBase):
            c_obj = obj
        else:
            raise TypeError(
                f"opacity must be a GreyOpacityLaw or C_GreyOpacityBase instance, "
                f"got {type(obj).__name__}"
            )
        self._opacity_obj = obj    # retain Python reference for GC safety
        self._c_opacity   = c_obj  # typed C reference for integrator
        # Pre-extract a raw void* so _pack_params can set params.opacity without the GIL.
        # Safe as long as self._c_opacity is alive (guaranteed by the ref above).
        self._opacity_ptr = <void*>(<PyObject*>c_obj)

    cdef void _pack_params(self, DiskParameters* p) nogil:
        """Base no-op: concrete closures override this to fill DiskParameters."""
        pass
