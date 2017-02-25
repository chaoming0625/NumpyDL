import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def pool_forward(np.ndarray[DTYPE_t, ndim=4] input,
                 int dim_pool,
                 np.ndarray[DTYPE_t, ndim=4] output):
    output.fill(0)
    cdef int n_batch = input.shape[0]
    cdef int depth = input.shape[1]
    cdef int dim_input = input.shape[2]
    cdef int x, y # indices for batch and depth
    cdef int i, j # indices for one input slice
    cdef int m, n # indices for one output slice
    cdef DTYPE_t [:, :, :, :] input_buffer, output_buffer
    input_buffer = input
    output_buffer = output
    cdef DTYPE_t [:, :] input_xy, output_xy
    cdef DTYPE_t input_xyij
    for x in range(n_batch):
        for y in range(depth):
            input_xy = input_buffer[x, y]
            output_xy = output_buffer[x, y]
            for i in range(dim_input):
                m = i // dim_pool
                for j in range(dim_input):
                    n = j // dim_pool
                    input_xyij = input_xy[i, j]
                    if input_xyij > output_xy[m, n]:
                        output_xy[m, n] = input_xyij

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def pool_backward(np.ndarray[DTYPE_t, ndim=4] dJ_dout,
                  np.ndarray[DTYPE_t, ndim=4] input,
                  np.ndarray[DTYPE_t, ndim=4] output,
                  int dim_pool,
                  np.ndarray[DTYPE_t, ndim=4] dJ_din):
    '''
    ALTERS dJ_dout as a byproduct!
    '''
    dJ_din.fill(0)
    cdef int n_batch = input.shape[0]
    cdef int depth = input.shape[1]
    cdef int dim_input = input.shape[2]
    cdef int dim_output = output.shape[2]
    cdef int x, y # indices for batch and depth
    cdef int i, j # indices for one input slice
    cdef int m, n # indices for one output slice
    cdef DTYPE_t [:, :, :, :] input_buffer, output_buffer, dJ_din_buffer, dJ_dout_buffer
    input_buffer = input
    output_buffer = output
    dJ_din_buffer = dJ_din
    dJ_dout_buffer = dJ_dout
    cdef DTYPE_t [:, :] input_xy, output_xy, dJ_din_xy, dJ_dout_xy
    cdef bool found
    for x in range(n_batch):
        for y in range(depth):
            input_xy = input_buffer[x, y]
            output_xy = output_buffer[x, y]
            dJ_din_xy = dJ_din_buffer[x, y]
            dJ_dout_xy = dJ_dout_buffer[x, y]
            for i in range(dim_input):
                m = i // dim_pool
                for j in range(dim_input):
                    n = j // dim_pool
                    if input_xy[i, j] == output_xy[m, n]:
                        dJ_din_xy[i, j] = dJ_dout_xy[m, n]
                        dJ_dout_xy[m, n] = 0
