import torch
import triton
from .kernels import add_kernel, mul_kernel



def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.shape[0]
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def mul(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.shape[0]
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
