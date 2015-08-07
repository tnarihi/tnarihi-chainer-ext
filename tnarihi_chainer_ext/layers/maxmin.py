import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check

from six.moves import range


class Maximum(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, x):
        return utils.force_array(numpy.maximum(x[0], x[1])),

    def backward_cpu(self, x, gy):
        gx0 = (x[0] >= x[1]) * gy[0]
        gx1 = (x[0] <= x[1]) * gy[0]
        return gx0, gx1

    def forward_gpu(self, x):
        return cuda.gpuarray.maximum(*x),

    def backward_gpu(self, x, gy):
        gx0 = cuda.empty_like(x[0])
        gx1 = cuda.empty_like(x[1])
        cuda.elementwise(
            '''
               float* gx0, float* gx1, const float* x0, const float* x1,
               const float* gy
            ''', '''
               gx0[i] = gy[i] * (x0[i] >= x1[i]);
               gx1[i] = gy[i] * (x0[i] <= x1[i]);
            ''', 'maximum_bwd')(gx0, gx1, x[0], x[1], gy[0])
        return gx0, gx1


def maximum(a, b):  # maximum(a, b)
    return Maximum()(a, b)


class Minimum(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, x):
        return utils.force_array(numpy.minimum(x[0], x[1])),

    def backward_cpu(self, x, gy):
        gx0 = (x[0] <= x[1]) * gy[0]
        gx1 = (x[0] >= x[1]) * gy[0]
        return gx0, gx1

    def forward_gpu(self, x):
        return cuda.gpuarray.minimum(*x),

    def backward_gpu(self, x, gy):
        gx0 = cuda.empty_like(x[0])
        gx1 = cuda.empty_like(x[1])
        cuda.elementwise(
            '''
               float* gx0, float* gx1, const float* x0, const float* x1,
               const float* gy
            ''', '''
               gx0[i] = gy[i] * (x0[i] <= x1[i]);
               gx1[i] = gy[i] * (x0[i] >= x1[i]);
            ''', 'minimum_bwd')(gx0, gx1, x[0], x[1], gy[0])
        return gx0, gx1


def minimum(a, b):  # maximum(a, b)
    return Minimum()(a, b)


# TODO: Max(axis, keepdims), Min(axis, keepdims)
