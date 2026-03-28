from ..closure cimport (
    ClosureResult, DiskDerived, DiskParameters, DiskState, DiskStep,
    FallbackParams, AdvectionParams,
    OneZoneClosure,
)
from ..physics._param_wrappers cimport CFallback, CAdvection

cdef int igP_adv_closure_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* prev,
    ClosureResult* out,
) nogil

cdef int igP_adv_writer_func(
    const int step_index,
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    const DiskStep* step,
    double* result_array,
    int n_steps,
) nogil

cdef int ADV_N_RESULT_FIELDS  # = 21; see _igP_adv.pyx

cdef class AdvectiveClosure(OneZoneClosure):
    cdef double _mu
    cdef double _xi
    cdef double _M_BH
    cdef double _R_in
    cdef double _alpha
    cdef bint _has_fallback
    cdef CFallback  _fallback_obj       # owns FallbackParams memory
    cdef CAdvection _advection_obj      # owns AdvectionParams memory
    cdef FallbackParams*  _fallback_ptr  # NULL when disabled
    cdef AdvectionParams* _advection_ptr # always valid after bind_runtime_parameters

    cdef void _pack_params(self, DiskParameters* p) nogil
