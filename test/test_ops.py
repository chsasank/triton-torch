import torch
from triton_torch.ops import add, mul, softmax

torch.manual_seed(0)


def test_add():
    size = 98432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output_torch = x + y
    output_triton = add(x, y)
    assert torch.allclose(output_torch, output_triton)
    assert output_triton.shape == output_torch.shape


def test_add_2d():
    x = torch.rand(123, 102, device="cuda")
    y = torch.rand(123, 102, device="cuda")
    output_torch = x + y
    output_triton = add(x, y)
    assert torch.allclose(output_torch, output_triton)
    assert output_triton.shape == output_torch.shape


def test_mul():
    size = 98432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")
    output_torch = x * y
    output_triton = mul(x, y)
    assert torch.allclose(output_torch, output_triton)
    assert output_triton.shape == output_torch.shape


def test_mul_2d():
    x = torch.rand(123, 102, device="cuda")
    y = torch.rand(123, 102, device="cuda")
    output_torch = x + y
    output_triton = add(x, y)
    assert torch.allclose(output_torch, output_triton)
    assert output_triton.shape == output_torch.shape


def test_softmax():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device="cuda")
    y_triton = softmax(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch)
    assert y_triton.shape == y_torch.shape
