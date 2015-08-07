from chainer.function_set import FunctionSet
import chainer.functions as F

from tnarihi_chainer_ext import WrappedFunctions


def set_attr_from_args(d):
    '''call__set_attr_from_args(locals()) at the top of your `__init__` method'''
    kls = d['self']
    del d['self']
    kls.__dict__.update(d)


class VggConvUnit(WrappedFunctions):

    def __init__(self, ichannels, ochannels, num=2, pooling_method='max', bn=False, nonlin='relu'):
        '''
        '''
        set_attr_from_args(locals())
        assert num >= 2, 'Numer of convolutions of {} must be >=2.'.format(
            self.__class__.__name__)
        assert pooling_method in ['max', 'sub']
        assert nonlin == 'relu', 'Support ReLU only so far.'
        f = FunctionSet()
        for i in range(num):
            ic = ichannels if i == 0 else ochannels
            stride = 2 if (pooling_method == 'sub' and i == num - 1) else 1
            setattr(f, 'conv{}'.format(i+1),
                    F.Convolution2D(ic, ochannels, 3, stride=stride, pad=1))
        if bn:
            setattr(f, 'bn', F.BatchNormalization(ochannels))
        self.f = f

    def __call__(self, x):
        h = x
        for i in range(self.num):
            nonlin = F.relu  # TODO: other activation
            if i == self.num - 1 and self.bn:
                nonlin = lambda xx: F.relu(self.f.bn(xx))
            h = nonlin(getattr(self.f, 'conv{}'.format(i+1))(h))
        # TODO: other pooling method
        if self.pooling_method != 'sub':
            h = F.max_pooling_2d(h, 2, 2)
        return h
