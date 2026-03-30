#cython: language_level=3, boundscheck=False
r"""
Python-owned C-struct wrapper classes for one-zone disk physics components.

These ``cdef`` classes act as simple lifetime managers for C-level parameter
structs (:c:type:`FallbackParams`, :c:type:`AdvectionParams`).  The pattern:

1. Concrete closures hold a ``CFallback`` (or ``CAdvection``) instance as a
   ``cdef`` member, which pins the struct in memory for the lifetime of the
   closure.
2. A raw ``FallbackParams*`` (or ``AdvectionParams*``) obtained via
   :meth:`CFallback.ptr` is stored in the closure and written into
   ``DiskParameters`` by ``_pack_params`` without the GIL.

Because the wrapper objects are ``cdef`` members of the closure (not local
variables), there is no dangling-pointer risk: the struct lives as long as
the closure lives.

See Also
--------
:mod:`triceratops.dynamics.accretion.one_zone.closure` :
    ``FallbackParams`` and ``AdvectionParams`` struct definitions.
"""
from ..closure cimport FallbackParams, AdvectionParams


# ======================================================== #
#  CFallback                                               #
# ======================================================== #

cdef class CFallback:
    """
    Python-owned wrapper around a :c:type:`FallbackParams` struct.

    Parameters
    ----------
    M_fb_0 : double
        Fallback mass supply rate at the reference time (g s⁻¹).
    R_c : double
        Circularisation radius (cm).
    t_fb : double
        Reference (peak) time (s).
    beta_fb : double
        Power-law index (dimensionless; typically 5/3).
    """

    def __cinit__(self, double M_fb_0, double R_c, double t_fb, double beta_fb):
        self.params.M_fb_0  = M_fb_0
        self.params.R_c     = R_c
        self.params.t_fb    = t_fb
        self.params.beta_fb = beta_fb

    cdef FallbackParams* ptr(self) nogil:
        """Return a raw pointer to the owned :c:type:`FallbackParams` struct."""
        return &self.params


# ======================================================== #
#  CAdvection                                              #
# ======================================================== #

cdef class CAdvection:
    """
    Python-owned wrapper around an :c:type:`AdvectionParams` struct.

    Parameters
    ----------
    xi : double
        Entropy gradient parameter (dimensionless, > 0).
    """

    def __cinit__(self, double xi):
        self.params.xi = xi

    cdef AdvectionParams* ptr(self) nogil:
        """Return a raw pointer to the owned :c:type:`AdvectionParams` struct."""
        return &self.params
