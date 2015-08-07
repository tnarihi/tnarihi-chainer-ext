import unittest

import numpy
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from tnarihi_chainer_ext.functions.deconvolution_2d import (
    Deconvolution2D, get_deconv_outsize)

if cuda.available:
    cuda.init()


class TestDeconvolution2D(unittest.TestCase):

    def setUp(self, use_cudnn=True):
        b, c, h, w = 2, 3, 4, 3
        k, s, p = 3, 2, 1
        n = 2
        self.func = Deconvolution2D(
            c, n, k, stride=s, pad=p, use_cudnn=use_cudnn)
        self.func.b = numpy.random.uniform(
            -1, 1, self.func.b.shape).astype(numpy.float32)
        self.func.gW.fill(0)
        self.func.gb.fill(0)
        out_h = get_deconv_outsize(h, k, s, p)
        out_w = get_deconv_outsize(w, k, s, p)
        self.x = numpy.random.uniform(-1, 1,
                                      (b, c, h, w)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1,
                                       (b, n, out_h, out_w)
                                       ).astype(numpy.float32)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_consistency(self):
        x_cpu = chainer.Variable(self.x)
        y_cpu = self.func(x_cpu)
        self.assertEqual(y_cpu.data.dtype, numpy.float32)

        self.func.to_gpu()
        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        y_gpu = self.func(x_gpu)
        self.assertEqual(y_gpu.data.dtype, numpy.float32)

        gradient_check.assert_allclose(y_cpu.data, y_gpu.data.get())

    @attr.gpu
    def test_forward_consistency_im2col(self):
        self.func.use_cudnn = False
        self.test_forward_consistency()

    def check_backward(self, x_data, y_grad):
        x = chainer.Variable(x_data)
        y = self.func(x)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, gW, gb = gradient_check.numerical_grad(
            f, (x.data, func.W, func.b), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad)
        gradient_check.assert_allclose(gW, func.gW)
        gradient_check.assert_allclose(gb, func.gb)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.func.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_im2col(self):
        self.func.use_cudnn = False
        self.test_backward_gpu()

    def check_pickling(self, x_data):
        x = chainer.Variable(x_data)
        y = self.func(x)
        y_data1 = y.data

        del x, y

        pickled = pickle.dumps(self.func, -1)
        del self.func
        self.func = pickle.loads(pickled)

        x = chainer.Variable(x_data)
        y = self.func(x)
        y_data2 = y.data

        gradient_check.assert_allclose(y_data1, y_data2, atol=0, rtol=0)

    def test_pickling_cpu(self):
        self.check_pickling(self.x)

    @attr.gpu
    def test_pickling_gpu(self):
        self.func.to_gpu()
        self.check_pickling(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
