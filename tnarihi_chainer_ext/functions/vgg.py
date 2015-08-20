from chainer.function_set import FunctionSet
import chainer.functions as F

from tnarihi_chainer_ext import WrappedFunctions


def set_attr_from_args(d):
    '''call __set_attr_from_args(locals()) at the top of your `__init__`
    method'''
    kls = d['self']
    del d['self']
    kls.__dict__.update(d)


class VggConvUnit(WrappedFunctions):

    def __init__(self, ichannels, ochannels, num=2, pooling_method='max',
                 bn=False, nonlin='relu'):
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
            f.bn = F.BatchNormalization(ochannels)
        self.f = f

    def _create_last_nonlinearity(self, train, finetune, no_last_nonlin):
        """Creating last nonlinearity (lambda) function
        """
        last_nonlin = lambda xx: xx
        # creating last nonlinearity
        if self.bn:
            pre_bn = last_nonlin
            last_nonlin = lambda xx: self.f.bn(
                pre_bn(xx), not train, finetune)
        if self.pooling_method != "sub":
            pre_pooling = last_nonlin
            # TODO: other pooling methods
            last_nonlin = lambda xx: F.max_pooling_2d(pre_pooling(xx), 2, 2)
        if not no_last_nonlin:
            pre_nonlin = last_nonlin
            # TODO: other activation
            last_nonlin = lambda xx: F.relu(pre_nonlin(xx))
        return last_nonlin

    def __call__(self, x, train=True, finetune=False, no_last_nonlin=False,
                 keep_intermediates=False):
        """
        Args:
            x (Variable) : Input with a shape of (b, c, h, w)
            train (bool): Flag for training, valid for BN layers
            finetune (bool): Flag for BN finetune mode
            no_last_nonlin (bool): Flag for turning off the last nonlinearity
            keep_intermediates (bool): Flag for keeping intermediate results
        """
        h = x
        self.hs = []
        for i in range(self.num):
            nonlin = F.relu  # TODO: other activation
            if i == self.num - 1:
                nonlin = self._create_last_nonlinearity(
                    train, finetune, no_last_nonlin)
            h = nonlin(getattr(self.f, 'conv{}'.format(i+1))(h))
            if keep_intermediates:
                self.hs += [h]
        return h

    def __getstate__(self):
        odict = self.__dict__.copy()
        if hasattr(odict, 'hs'):
            del odict['hs']
        return odict


def load_conv_params(params, prefix, convunit):
    """Load VGG conv parameters.

    Loading from the object from a mat file created by
    `vgg_caffemodel_to_mat.py`.

    Args:
        params (dict-like): A dictionary which has the conv parameters like
            'convx_y_W', 'convx_y_b' etc. Usually you get it from a mat file
            that is created by `vgg_caffemodel_to_mat.py`.
        prefix (str): If you want to put the parameters from `conv1_y_{W|b}`,
            you will set it as 'conv1'.
        convunit (VggConvUnit): A instance of `VggConvUnit` class.
    Returns:
        None
    """
    i = 0
    while True:
        key = prefix + '_{}'.format(i + 1)
        try:
            c = getattr(convunit.f, 'conv{}'.format(i + 1))
        except:
            break
        cc_W = params[key + '_W']
        cc_b = params[key + '_b']
        c.W = cc_W.copy()
        cc_b = cc_b.copy()
        print('weights {} is loaded'.format(key))
        i += 1


def load_fc_params(params, key, fc):
    """
    Load VGG fc parameters.

    Args:
        params (dict-like): See `load_conv_params`.
        key (str): See `prefix` argument in `load_conv_params`.
        fc (F.Linear): Linear object
    """
    fc.W = params[key + '_W'].copy()
    fc.b = params[key + '_b'].copy()
    print('weights {} is loaded'.format(key))


class Vgg16(WrappedFunctions):

    """An example implementation of VGG 16 layer as a Function class
    """

    def __init__(self, bn=False, pool_method='max'):
        from chainer import FunctionSet
        import chainer.functions as F
        self.f = FunctionSet(
            conv1=VggConvUnit(3, 64, 2, bn=bn, pooling_method=pool_method),
            conv2=VggConvUnit(64, 128, 2, bn=bn, pooling_method=pool_method),
            conv3=VggConvUnit(128, 256, 3, bn=bn, pooling_method=pool_method),
            conv4=VggConvUnit(256, 512, 3, bn=bn, pooling_method=pool_method),
            conv5=VggConvUnit(512, 512, 3, bn=bn, pooling_method=pool_method),
            fc6=F.Linear(512 * 7 * 7, 4096),
            fc7=F.Linear(4096, 4096),
            fc8=F.Linear(4096, 1000),
        )

    def __call__(self, x, train=True, finetune=False,
                 keep_intermediates=False):
        model = self.f
        self.clean_intermediates()
        h = x
        for i in range(5):
            h = getattr(model, 'conv{}'.format(i + 1))(h, train, finetune)
            if keep_intermediates:
                self.hs += [h]
        h = F.dropout(F.relu(model.fc6(h)), train=train)
        if keep_intermediates:
            self.hs += [h]
        h = F.dropout(F.relu(model.fc7(h)), train=train)
        if keep_intermediates:
            self.hs += [h]
        h = model.fc8(h)
        return h

    def clean_intermediates(self):
        self.hs = []
        for i in range(5):
            getattr(self.f, 'conv{}'.format(i + 1)).hs = []

    def forward_loss(self, x, t, train=True):
        h = self(x, train=train)
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)

    def __getstate__(self):
        odict = self.__dict__.copy()
        if hasattr(odict, 'hs'):
            del odict['hs']
        return odict

    def load_params(self, params, bgr2rgb=True):
        model = self.f
        for i in range(8):
            if i < 5:
                load_conv_params(params, 'conv{}'.format(i + 1),
                                 getattr(model, 'conv{}'.format(i + 1)))
            else:
                load_fc_params(params, 'fc{}'.format(i + 1),
                               getattr(model, 'fc{}'.format(i + 1)))
        if bgr2rgb:
            model.conv1.f.conv1.W = \
                model.conv1.f.conv1.W[:, ::-1].copy(order='C')
        return self
