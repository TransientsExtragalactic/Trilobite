from ._integrator cimport (
    OneZoneClosure, closure_func, derivative_func, writer_func,
    DiskState, DiskDerived, DiskParameters, ClosureResult, DiskStep
)


cdef class GasPressureElectronScatteringClosure(OneZoneClosure):
    pass


cdef class FullPressureElectronScatteringClosure(OneZoneClosure):
    pass


# Non-inline declarations — callable via cimport from _source_terms.pyx
cdef int gas_pressure_closure_func(
    const DiskState*, const DiskDerived*, const DiskParameters*,
    const ClosureResult*, ClosureResult*) nogil

cdef int gas_pressure_derivative_func(
    const DiskState*, const DiskDerived*, const DiskParameters*,
    const ClosureResult*, DiskStep*) nogil

cdef int gas_pressure_writer_func(
    const int, const DiskState*, const DiskDerived*, const DiskParameters*,
    const ClosureResult*, const DiskStep*, double*, int) nogil

cdef int full_pressure_closure_func(
    const DiskState*, const DiskDerived*, const DiskParameters*,
    const ClosureResult*, ClosureResult*) nogil
