from ..closure cimport (
    OneZoneClosure,
    ClosureResult, DiskDerived, DiskParameters, DiskState,
)


cdef int gP_closure_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* prev,
    ClosureResult* out,
) nogil
