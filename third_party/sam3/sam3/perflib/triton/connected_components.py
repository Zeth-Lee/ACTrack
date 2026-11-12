# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import math

import torch
import triton
import triton.language as tl


@triton.jit
def _any_combine(a, b):
    return a | b


@triton.jit
def tl_any(a, dim=0):
    return tl.reduce(a, dim, _any_combine)


@triton.jit
def _init_labels_kernel(
    input_ptr, labels_ptr, numel: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    input_values = tl.load(input_ptr + offsets, mask=mask, other=0)

    indices = tl.where((input_values != 0), offsets, -1)
    tl.store(labels_ptr + offsets, indices, mask=mask)


@triton.jit
def find(labels_ptr, indices, mask):
    current_pids = indices


    is_done = ~mask


    while tl_any(~is_done):

        work_mask = ~is_done
        parents = tl.load(labels_ptr + current_pids, mask=work_mask, other=-1)


        is_root = parents == current_pids
        is_sentinel = parents == -1
        is_done |= is_root | is_sentinel


        current_pids = tl.where(is_done, current_pids, parents)


    return current_pids


@triton.jit
def union(labels_ptr, a, b, process_mask):


    current_a = a
    current_b = b

    final_root = a

    done_mask = ~process_mask

    while tl_any(~done_mask):

        work_mask = process_mask & ~done_mask


        root_a = find(labels_ptr, current_a, work_mask)
        tl.debug_barrier()
        root_b = find(labels_ptr, current_b, work_mask)


        are_equal = root_a == root_b
        final_root = tl.where(are_equal & work_mask & ~done_mask, root_a, final_root)
        done_mask |= are_equal & work_mask


        a_is_smaller = root_a < root_b


        merge_mask_a_smaller = work_mask & a_is_smaller & ~are_equal
        ptr_b = labels_ptr + root_b
        old_val_b = tl.atomic_min(ptr_b, root_a, mask=merge_mask_a_smaller)


        success_b = old_val_b == root_b
        final_root = tl.where(success_b & work_mask & ~done_mask, root_a, final_root)
        done_mask |= success_b & merge_mask_a_smaller


        current_b = tl.where(success_b | ~merge_mask_a_smaller, current_b, old_val_b)


        merge_mask_b_smaller = work_mask & ~a_is_smaller & ~are_equal
        ptr_a = labels_ptr + root_a
        old_val_a = tl.atomic_min(ptr_a, root_b, mask=merge_mask_b_smaller)

        success_a = old_val_a == root_a
        final_root = tl.where(success_a & work_mask & ~done_mask, root_b, final_root)
        done_mask |= success_a & merge_mask_b_smaller


        current_a = tl.where(success_a | ~merge_mask_b_smaller, current_a, old_val_a)

    return final_root


@triton.jit
def _merge_helper(
    input_ptr,
    labels_ptr,
    base_offset,
    offsets_h,
    offsets_w,
    mask_2d,
    valid_current,
    current_values,
    current_labels,
    H,
    W,
    dx: tl.constexpr,
    dy: tl.constexpr,
):


    neighbor_h = offsets_h + dy
    neighbor_w = offsets_w + dx

    mask_n = (
        mask_2d
        & (neighbor_h[:, None] >= 0)
        & (neighbor_h[:, None] < H)
        & (neighbor_w[None, :] >= 0)
        & (neighbor_w[None, :] < W)
    )

    offsets_neighbor = neighbor_h[:, None] * W + neighbor_w[None, :]
    neighbor_values = tl.load(
        input_ptr + base_offset + offsets_neighbor, mask=mask_n, other=-1
    )

    mask_n = tl.ravel(mask_n)
    neighbor_labels = tl.load(
        labels_ptr + tl.ravel(base_offset + offsets_neighbor), mask=mask_n, other=-1
    )

    to_merge = (
        mask_n & (neighbor_labels != -1) & tl.ravel(current_values == neighbor_values)
    )
    valid_write = valid_current & to_merge


    return tl.where(
        valid_write,
        union(labels_ptr, current_labels, neighbor_labels, valid_write),
        current_labels,
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_H": 4, "BLOCK_SIZE_W": 16}, num_stages=1, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 4, "BLOCK_SIZE_W": 32}, num_stages=2, num_warps=4
        ),
    ],
    key=["H", "W"],
    restore_value=["labels_ptr"],
)
@triton.jit
def _local_prop_kernel(
    labels_ptr,
    input_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):


    pid_b = tl.program_id(0)
    pid_hw = tl.program_id(1)


    offsets_h = (pid_hw // tl.cdiv(W, BLOCK_SIZE_W)) * BLOCK_SIZE_H + tl.arange(
        0, BLOCK_SIZE_H
    )
    offsets_w = (pid_hw % tl.cdiv(W, BLOCK_SIZE_W)) * BLOCK_SIZE_W + tl.arange(
        0, BLOCK_SIZE_W
    )

    base_offset = pid_b * H * W
    offsets_2d = offsets_h[:, None] * W + offsets_w[None, :]
    mask_2d = (offsets_h[:, None] < H) & (offsets_w[None, :] < W)
    mask_1d = tl.ravel(mask_2d)


    current_labels = tl.load(
        labels_ptr + tl.ravel(base_offset + offsets_2d), mask=mask_1d, other=-1
    )
    current_values = tl.load(
        input_ptr + base_offset + offsets_2d, mask=mask_2d, other=-1
    )
    valid_current = mask_1d & (current_labels != -1)


    current_labels = _merge_helper(
        input_ptr,
        labels_ptr,
        base_offset,
        offsets_h,
        offsets_w,
        mask_2d,
        valid_current,
        current_values,
        current_labels,
        H,
        W,
        -1,
        0,
    )

    current_labels = _merge_helper(
        input_ptr,
        labels_ptr,
        base_offset,
        offsets_h,
        offsets_w,
        mask_2d,
        valid_current,
        current_values,
        current_labels,
        H,
        W,
        0,
        -1,
    )

    current_labels = _merge_helper(
        input_ptr,
        labels_ptr,
        base_offset,
        offsets_h,
        offsets_w,
        mask_2d,
        valid_current,
        current_values,
        current_labels,
        H,
        W,
        -1,
        -1,
    )
    current_labels = _merge_helper(
        input_ptr,
        labels_ptr,
        base_offset,
        offsets_h,
        offsets_w,
        mask_2d,
        valid_current,
        current_values,
        current_labels,
        H,
        W,
        -1,
        1,
    )


    tl.atomic_min(
        labels_ptr + tl.ravel(base_offset + offsets_2d), current_labels, mask=mask_1d
    )


@triton.jit
def _pointer_jump_kernel(
    labels_in_ptr, labels_out_ptr, numel: tl.constexpr, BLOCK_SIZE: tl.constexpr
):


    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel


    current_labels = tl.load(labels_in_ptr + offsets, mask=mask, other=-1)
    valid_mask = mask & (current_labels != -1)


    done_mask = ~valid_mask
    while tl_any(~(done_mask | ~valid_mask)):
        parent_labels = tl.load(
            labels_in_ptr + current_labels, mask=valid_mask, other=-1
        )

        are_equal = current_labels == parent_labels
        done_mask |= are_equal & valid_mask

        current_labels = tl.where(
            ~done_mask, tl.minimum(current_labels, parent_labels), current_labels
        )


    tl.store(labels_out_ptr + offsets, current_labels, mask=mask)


@triton.jit
def _count_labels_kernel(labels_ptr, sizes_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel


    labels = tl.load(labels_ptr + offsets, mask=mask, other=-1)
    valid_mask = mask & (labels != -1)


    tl.atomic_add(sizes_ptr + labels, 1, mask=valid_mask)


@triton.jit
def _broadcast_sizes_kernel(
    labels_ptr, sizes_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel


    labels = tl.load(labels_ptr + offsets, mask=mask, other=-1)
    valid_mask = mask & (labels != -1)


    component_sizes = tl.load(sizes_ptr + labels, mask=valid_mask, other=0)


    tl.store(out_ptr + offsets, component_sizes, mask=mask)


def connected_components_triton(input_tensor: torch.Tensor):

    assert input_tensor.is_cuda and input_tensor.is_contiguous(), (
        "Input tensor must be a contiguous CUDA tensor."
    )
    out_shape = input_tensor.shape
    if input_tensor.dim() == 4 and input_tensor.shape[1] == 1:
        input_tensor = input_tensor.squeeze(1)
    else:
        assert input_tensor.dim() == 3, (
            "Input tensor must be (B, H, W) or (B, 1, H, W)."
        )

    B, H, W = input_tensor.shape
    numel = B * H * W
    device = input_tensor.device


    labels = torch.empty_like(input_tensor, dtype=torch.int32)
    output = torch.empty_like(input_tensor, dtype=torch.int32)


    BLOCK_SIZE = 256
    grid_init = (triton.cdiv(numel, BLOCK_SIZE),)
    _init_labels_kernel[grid_init](
        input_tensor,
        labels,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )


    grid_local_prop = lambda meta: (
        B,
        triton.cdiv(H, meta["BLOCK_SIZE_H"]) * triton.cdiv(W, meta["BLOCK_SIZE_W"]),
    )
    _local_prop_kernel[grid_local_prop](labels, input_tensor, H, W)


    BLOCK_SIZE = 256
    grid_jump = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _pointer_jump_kernel[grid_jump](labels, output, numel, BLOCK_SIZE=BLOCK_SIZE)


    component_sizes_out = torch.empty_like(input_tensor, dtype=torch.int32)


    sizes_histogram = torch.zeros(numel, dtype=torch.int32, device=device)


    grid_count = (triton.cdiv(numel, BLOCK_SIZE),)
    _count_labels_kernel[grid_count](
        output, sizes_histogram, numel, BLOCK_SIZE=BLOCK_SIZE
    )


    grid_broadcast = (triton.cdiv(numel, BLOCK_SIZE),)
    _broadcast_sizes_kernel[grid_broadcast](
        output, sizes_histogram, component_sizes_out, numel, BLOCK_SIZE=BLOCK_SIZE
    )
    return output.view(out_shape) + 1, component_sizes_out.view(out_shape)
