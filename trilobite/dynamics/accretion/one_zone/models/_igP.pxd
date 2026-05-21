from ..closure cimport (
    OneZoneClosure,
    ClosureResult, DiskDerived, DiskParameters, DiskState, FallbackParams,
)
from ..physics._param_wrappers cimport CFallback


cdef int igP_closure_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* prev,
    ClosureResult* out,
) nogil


cdef class FullPressureClosure(OneZoneClosure):
    cdef double _mu
    cdef double _M_BH
    cdef double _R_in
    cdef double _alpha
    cdef bint _has_fallback
    cdef CFallback _fallback_obj       # owns the FallbackParams memory
    cdef FallbackParams* _fallback_ptr  # NULL when disabled; valid while _fallback_obj alive

    cdef void _pack_params(self, DiskParameters* p) nogil
