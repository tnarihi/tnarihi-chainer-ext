import unittest

import numpy
from six.moves import zip

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

from tnarihi_chainer_ext.functions import maximum, minimum

if cuda.available:
    cuda.init()


class TestMaxMin(unittest.TestCase):

    def setUp(self):
        b, c, h, w = 2, 3, 4, 3
        self.x = numpy.random.uniform(-1, 1,
                                      (b, c, h, w)).astype(numpy.float32),
        self.x += numpy.random.uniform(-1, 1,
                                       (b, c, h, w)).astype(numpy.float32),
        self.gy = numpy.random.uniform(-1, 1,
                                       (b, c, h, w)).astype(numpy.float32)
                                       

    def forward_consistency(self, func):
        x_cpu = [chainer.Variable(x) for x in self.x]
        y_cpu = func(*x_cpu)
        self.assertEqual(y_cpu.data.dtype, numpy.float32)

        x_gpu = [chainer.Variable(cuda.to_gpu(x)) for x in self.x]
        y_gpu = func(*x_gpu)
        self.assertEqual(y_gpu.data.dtype, numpy.float32)

        gradient_check.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.gpu
    @condition.retry(3)
    def test_maximum_forward_consistency(self):
        self.forward_consistency(maximum)

    @attr.gpu
    @condition.retry(3)
    def test_minimum_forward_consistency(self):
        self.forward_consistency(minimum)

    def check_backward(self, func1, xs, gy):
        y = func1(*xs)
        y.grad = gy
        y.backward()

        func = y.creator
        f = lambda: func.forward(tuple(x.data for x in xs))
        gxs = gradient_check.numerical_grad(
            f, tuple(x.data for x in xs), (y.grad,), eps=1e-2)

        for x, gx in zip(xs, gxs):
            gradient_check.assert_allclose(gx, x.grad)

    @condition.retry(3)
    def test_maximum_backward_cpu(self):
        self.check_backward(
            maximum, tuple(chainer.Variable(x) for x in self.x), self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_maximum_backward_gpu(self):
        self.check_backward(
            maximum, tuple(chainer.Variable(cuda.to_gpu(x)) for x in self.x),
            cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_minimum_backward_cpu(self):
        self.check_backward(
            maximum, tuple(chainer.Variable(x) for x in self.x), self.gy)

    @attr.gpu
    @condition.retry(3)
    def test_minimum_backward_gpu(self):
        self.check_backward(
            maximum, tuple(chainer.Variable(cuda.to_gpu(x)) for x in self.x),
            cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
