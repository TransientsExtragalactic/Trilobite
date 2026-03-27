from .closure cimport (
    ClosureResult, DiskDerived, DiskParameters, DiskState, DiskStep,
)

# Number of output fields per time step.
cdef int N_RESULT_FIELDS  # = 20; see _writer.pyx

cdef int standard_writer_func(
    const int step_index,
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    const DiskStep* step,
    double* result_array,
    int n_steps,
) nogil
