from ..closure cimport (
    ClosureResult, DiskDerived, DiskParameters, DiskState, DiskStep,
    OneZoneClosure,
)

cdef int igP_es_adv_closure_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* prev,
    ClosureResult* out,
) nogil

cdef int igP_es_adv_writer_func(
    const int step_index,
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    const DiskStep* step,
    double* result_array,
    int n_steps,
) nogil

cdef int ADV_N_RESULT_FIELDS  # = 21; see _igP_es_adv.pyx

cdef class igP_es_advClosure(OneZoneClosure):
    pass
