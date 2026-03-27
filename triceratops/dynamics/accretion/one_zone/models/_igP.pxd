from ..closure cimport (
    OneZoneClosure,
    ClosureResult, DiskDerived, DiskParameters, DiskState,
)


cdef int igP_closure_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* prev,
    ClosureResult* out,
) nogil


cdef class igPClosure(OneZoneClosure):
    pass
