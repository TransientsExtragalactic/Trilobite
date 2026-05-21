from ..closure cimport (
    ClosureResult, DiskDerived, DiskParameters, DiskState, DiskStep,
)


cdef int fallback_source_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    DiskStep* step,
) nogil
