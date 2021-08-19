import torch
from triton_torch.ops import add, mul

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')

def test_add():
    output_torch = x + y
    output_triton = add(x, y)
    assert torch.max(torch.abs(output_torch - output_triton)) < 1e-6


def test_mul():
    output_torch = x * y
    output_triton = mul(x, y)
    assert torch.max(torch.abs(output_torch - output_triton)) < 1e-6
