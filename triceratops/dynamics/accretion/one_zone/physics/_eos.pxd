# Declarations use cdef (not inline) — inline cannot be used in .pxd declarations
# because Cython generates function pointers for cimported functions.
# The implementations in _eos.pyx are still cdef inline for the C compiler.
cdef double compute_ideal_gas_cs(double T, double mu) nogil

cdef double compute_gas_rad_cs(
    double T, double mu, double Sigma, double Omega) nogil
