from future.builtins import super

from chainer import function


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
