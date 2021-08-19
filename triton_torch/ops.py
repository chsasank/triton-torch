import torch
import triton
from .kernels import add_kernel, mul_kernel, softmax_kernel


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def mul(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def softmax(x):
    def _next_power_of_2(n):
        """Return the smallest power of 2 greater than or equal to n"""
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n += 1
        return n

    n_rows, n_cols = x.shape
    BLOCK_SIZE = _next_power_of_2(n_cols)
    y = torch.empty_like(x)
    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
