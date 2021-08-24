import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    **meta,
):
    BLOCK_SIZE = meta["BLOCK_SIZE"]

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output)


def _add(x, y):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    assert x.is_contiguous() and y.is_contiguous() and output.is_contiguous()
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


@triton.jit
def mul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    **meta,
):
    BLOCK_SIZE = meta["BLOCK_SIZE"]

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output)


def _mul(x, y):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    assert x.is_contiguous() and y.is_contiguous() and output.is_contiguous()
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, **meta
):
    row_idx = tl.program_id(0)
    BLOCK_SIZE = meta["BLOCK_SIZE"]
    row_start_ptr = input_ptr + row_idx * input_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def _softmax(x):
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

    assert x.is_cuda and y.is_cuda
    assert x.is_contiguous() and y.is_contiguous()

    softmax_kernel[(n_rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
