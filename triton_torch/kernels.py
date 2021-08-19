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
    BLOCK_SIZE = meta['BLOCK_SIZE']
    
    pid = tl.program_id(axis=0) 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output)


@triton.jit
def mul_kernel(
    x_ptr, 
    y_ptr,
    output_ptr, 
    n_elements, 
    **meta,
):
    BLOCK_SIZE = meta['BLOCK_SIZE']
    
    pid = tl.program_id(axis=0) 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y
    tl.store(output_ptr + offsets, output)

