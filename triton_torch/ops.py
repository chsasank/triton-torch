from torch.autograd import Function, grad
from triton.language import softmax

from .kernels import _add, _mul, _softmax


class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return _add(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_y = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        if ctx.needs_input_grad[1]:
            grad_y = grad_output
        return grad_x, grad_y


add = Add.apply


class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return _mul(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_x = grad_y = None
        if ctx.needs_input_grad[0]:
            grad_x = _mul(grad_output, y)
        if ctx.needs_input_grad[1]:
            grad_y = _mul(grad_output, x)
        return grad_x, grad_y


mul = Mul.apply


class Softmax(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _softmax(x)


softmax = Softmax.apply
