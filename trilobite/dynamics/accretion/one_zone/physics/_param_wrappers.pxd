from ..closure cimport FallbackParams, AdvectionParams


cdef class CFallback:
    """Python-owned wrapper that keeps a ``FallbackParams`` struct alive on the heap."""
    cdef FallbackParams params
    cdef FallbackParams* ptr(self) nogil


cdef class CAdvection:
    """Python-owned wrapper that keeps an ``AdvectionParams`` struct alive on the heap."""
    cdef AdvectionParams params
    cdef AdvectionParams* ptr(self) nogil
