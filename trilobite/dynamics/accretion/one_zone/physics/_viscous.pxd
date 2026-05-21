from ..closure cimport (
    ClosureResult, DiskDerived, DiskParameters, DiskState, DiskStep,
)


cdef int viscous_derivative_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    DiskStep* out,
) nogil
