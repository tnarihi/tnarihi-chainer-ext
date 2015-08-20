from future.builtins import super

import numpy as np

from chainer import function
from chainer.cuda import to_cpu

try:
    import h5py
    h5py_enabled = True
except ImportError:
    h5py_enabled = False


def _force_to_cpu(x):
    if isinstance(x, np.ndarray):
        return x
    return to_cpu(x)


def _copy_param(src, dst):
    if dst.size != src.size:
        # use size rather than shape since h5py store shape with ndim > 1
        print(
            'Shapes mismatch: src={}, dst={}. Skip'.format(
                src.shape, dst.shape))
        return
    if isinstance(dst, np.ndarray):
        dst.flat[:] = src.flat
    else:
        dst.get(src.reshape(dst.shape))


def _parameter_generator(func):
    if hasattr(func, 'parameter_names'):
        for paramname in func.parameter_names:
            yield paramname, getattr(func, paramname)
    else:
        for pind, param in func.parameters:
            yield str(pind), param


class WrappedFunctions(function.Function):

    """A base class for defining a new `Function` class which contains
    `FunctionSet` as `self.f` (parameterized functions) inside.

    you have to set a attribute `self.f = FunctionSet(...)` in your `__init__`
    method. Then, you will override `__call__(self, x)` method such that input
    `x` (could be multiple arguments) pass though your functions those could be
    either parameterized ones in `self.f` or nonparameterized ones.
    See `VggConvUnit` for detailed usage.
    """

    def to_cpu(self):
        super().to_cpu()
        self.f.to_cpu()

    def to_gpu(self, device=None):
        super().to_gpu(device)
        self.f.to_gpu(device)

    @property
    def parameters(self):
        return self.f.parameters

    @parameters.setter
    def parameters(self, params):
        self.f.parameters = params

    @property
    def gradients(self):
        return self.f.gradients

    @gradients.setter
    def gradients(self, grads):
        self.f.gradients = grads

    def save_parameters_hdf5(self, filename=None, h5mode='w', h5d=None):
        """
        Args:
            filename (str): filename of HDF5 file to be created or overwritten
            h5mode (str): HDF5 file mode
            h5d (HDF5 file descriptor): Used only in recursively call
        """
        assert h5py_enabled, "Install h5py."
        assert filename is None or h5d is None, \
            'Both of filename and h5d should not be provided.'
        assert filename is not None or h5d is not None, \
            'Either of filename or h5d should be provided.'

        def _core(h5d):
            for funcname, func in self.f._get_sorted_funcs():
                h5d_child = h5d.create_group(funcname)
                if isinstance(func, WrappedFunctions):
                    func.save_parameters_hdf5(h5d=h5d_child)
                else:
                    for paramname, param in _parameter_generator(func):
                        print('Saving parameter: {}/{}'.format(
                            h5d_child.name, paramname))
                        h5d_child[paramname] = _force_to_cpu(param)
        if h5d is None:
            print ('Saving hdf5 file: {}'.format(filename))
            with h5py.File(filename, h5mode) as h5d:
                _core(h5d)
        else:
            print ('Saving function: {}'.format(h5d.name))
            _core(h5d)

    def load_parameters_hdf5(self, filename=None, h5d=None):
        """
        Args:
            filename (str): filename of HDF5 file
            h5d (HDF5 file descriptor): Used only in recursively call
        """
        assert h5py_enabled, "Install h5py."
        assert filename is None or h5d is None, \
            'Both of filename and h5d should not be provided.'
        assert filename is not None or h5d is not None, \
            'Either of filename or h5d should be provided.'

        def _core(h5d):
            for funcname, func in self.f._get_sorted_funcs():
                try:
                    h5d_child = h5d[funcname]
                except KeyError:
                    print("Parameters of {}/{} doesn't exist. Skip.".format(
                        h5d.name, funcname))
                    continue
                if isinstance(func, WrappedFunctions):
                    func.load_parameters_hdf5(h5d=h5d_child)
                else:
                    for paramname, dst in _parameter_generator(func):
                        print('Loading parameter: {}/{}'.format(
                            h5d_child.name, paramname))
                        try:
                            src = h5d_child[paramname].value
                        except KeyError:
                            print ("Parameter doesn't exist. Skip.")
                            continue
                        _copy_param(src, dst)
        if h5d is None:
            print ('Loading hdf5 file: {}'.format(filename))
            with h5py.File(filename, 'r') as h5d:
                _core(h5d)
        else:
            print ('Loading function: {}'.format(h5d.name))
            _core(h5d)
