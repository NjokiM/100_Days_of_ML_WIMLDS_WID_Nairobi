# cython: language_level=3, boundscheck=False, wraparound=False
# TODO (Part A): write a typed dot product over two contiguous float64 arrays.
# Use typed memoryviews (double[::1]) and a cdef-typed loop index and accumulator.

def dot(double[::1] a, double[::1] b):
    # TODO: raise ValueError if lengths differ, then loop in C and return the sum.
    raise NotImplementedError("implement dot()")
