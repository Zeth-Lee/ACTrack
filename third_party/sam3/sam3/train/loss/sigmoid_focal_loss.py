# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice


@triton.jit
def _inner_focal_loss_fwd(inputs, targets, alpha, gamma):
    inv_targets = 1 - targets

    sig = tl.sigmoid(inputs)


    max_val = tl.clamp(inputs, min=0, max=1e9)
    bce_loss = max_val - inputs * targets + tl.log(1 + tl.exp(-tl.abs(inputs)))


    p_t = sig * targets + (1 - sig) * inv_targets
    mod_factor = libdevice.pow(1 - p_t, gamma)


    alpha_t = alpha * targets + (1 - alpha) * inv_targets


    return alpha_t * mod_factor * bce_loss


@triton.jit
def sigmoid_focal_loss_fwd_kernel(
    inputs_ptr,
    targets_ptr,
    loss_ptr,
    alpha: float,
    gamma: float,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements


    inputs = tl.load(inputs_ptr + offset, mask=mask).to(tl.float32)
    targets = tl.load(targets_ptr + offset, mask=mask)

    final_loss = _inner_focal_loss_fwd(inputs, targets, alpha, gamma)


    tl.store(loss_ptr + offset, final_loss, mask=mask)


@triton.jit
def sigmoid_focal_loss_fwd_kernel_reduce(
    inputs_ptr,
    targets_ptr,
    loss_ptr,
    alpha: float,
    gamma: float,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    reduce_loc = pid % REDUCE_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inputs = tl.load(inputs_ptr + offset, mask=mask).to(tl.float32)
    targets = tl.load(targets_ptr + offset, mask=mask)

    final_loss = _inner_focal_loss_fwd(inputs, targets, alpha, gamma) * mask

    fl = tl.sum(final_loss)


    tl.atomic_add(loss_ptr + reduce_loc, fl)


@triton.jit
def _inner_focal_loss_bwd(inputs, targets, alpha, gamma):
    inv_targets = 1 - targets


    max_val = tl.clamp(inputs, min=0, max=1e9)
    bce_loss = max_val - inputs * targets + tl.log(1 + tl.exp(-tl.abs(inputs)))


    sig = tl.sigmoid(inputs)
    inv_sig = 1 - sig


    p_t = sig * targets + inv_sig * inv_targets
    tmp = libdevice.pow(1 - p_t, gamma - 1)
    mod_factor = tmp * (1 - p_t)


    alpha_t = alpha * targets + (1 - alpha) * inv_targets


    d_pt = (2 * targets - 1) * sig * inv_sig
    d_mod_factor = -gamma * d_pt * tmp

    d_bce_loss = sig - targets

    return alpha_t * (d_bce_loss * mod_factor + d_mod_factor * bce_loss)


@triton.jit
def sigmoid_focal_loss_bwd_kernel(
    inputs_ptr,
    targets_ptr,
    grad_inputs_ptr,
    grad_out_ptr,
    alpha: float,
    gamma: float,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    input_ptrs = inputs_ptr + offset
    target_ptrs = targets_ptr + offset
    grad_input_ptrs = grad_inputs_ptr + offset
    grad_out_ptrs = grad_out_ptr + offset

    inputs = tl.load(input_ptrs, mask=mask).to(tl.float32)
    targets = tl.load(target_ptrs, mask=mask)
    grad_out = tl.load(grad_out_ptrs, mask=mask)
    d_loss = grad_out * _inner_focal_loss_bwd(inputs, targets, alpha, gamma)
    tl.store(grad_input_ptrs, d_loss, mask=mask)


@triton.jit
def sigmoid_focal_loss_bwd_kernel_reduce(
    inputs_ptr,
    targets_ptr,
    grad_inputs_ptr,
    grad_out_ptr,
    alpha: float,
    gamma: float,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    input_ptrs = inputs_ptr + offset
    target_ptrs = targets_ptr + offset
    grad_input_ptrs = grad_inputs_ptr + offset

    inputs = tl.load(input_ptrs, mask=mask).to(tl.float32)
    targets = tl.load(target_ptrs, mask=mask)
    grad_out = tl.load(grad_out_ptr)
    d_loss = grad_out * _inner_focal_loss_bwd(inputs, targets, alpha, gamma)
    tl.store(grad_input_ptrs, d_loss, mask=mask)


class SigmoidFocalLoss(torch.autograd.Function):
    BLOCK_SIZE = 256

    @staticmethod
    def forward(ctx, inputs, targets, alpha=0.25, gamma=2):
        n_elements = inputs.numel()
        assert targets.numel() == n_elements
        input_shape = inputs.shape
        inputs = inputs.view(-1).contiguous()
        targets = targets.view(-1).contiguous()
        loss = torch.empty(inputs.shape, dtype=torch.float32, device=inputs.device)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        sigmoid_focal_loss_fwd_kernel[grid](
            inputs, targets, loss, alpha, gamma, n_elements, SigmoidFocalLoss.BLOCK_SIZE
        )
        ctx.save_for_backward(inputs.view(input_shape), targets.view(input_shape))
        ctx.alpha = alpha
        ctx.gamma = gamma
        return loss.view(input_shape)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, targets = ctx.saved_tensors
        alpha = ctx.alpha
        gamma = ctx.gamma
        n_elements = inputs.numel()
        input_shape = inputs.shape
        grad_inputs = torch.empty(
            inputs.shape, dtype=grad_output.dtype, device=grad_output.device
        )
        inputs_ptr = inputs.view(-1).contiguous()
        targets_ptr = targets.view(-1).contiguous()
        grad_output_ptr = grad_output.view(-1).contiguous()
        grad_inputs_ptr = grad_inputs
        assert grad_output.numel() == n_elements
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        sigmoid_focal_loss_bwd_kernel[grid](
            inputs_ptr,
            targets_ptr,
            grad_inputs_ptr,
            grad_output_ptr,
            alpha,
            gamma,
            n_elements,
            SigmoidFocalLoss.BLOCK_SIZE,
        )
        return grad_inputs.view(input_shape), None, None, None


triton_sigmoid_focal_loss = SigmoidFocalLoss.apply


class SigmoidFocalLossReduced(torch.autograd.Function):
    BLOCK_SIZE = 256
    REDUCE_SIZE = 32

    @staticmethod
    def forward(ctx, inputs, targets, alpha=0.25, gamma=2):
        n_elements = inputs.numel()
        input_shape = inputs.shape
        inputs = inputs.view(-1).contiguous()
        targets = targets.view(-1).contiguous()
        loss = torch.zeros(
            SigmoidFocalLossReduced.REDUCE_SIZE,
            device=inputs.device,
            dtype=torch.float32,
        )
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        sigmoid_focal_loss_fwd_kernel_reduce[grid](
            inputs,
            targets,
            loss,
            alpha,
            gamma,
            n_elements,
            SigmoidFocalLossReduced.BLOCK_SIZE,
            SigmoidFocalLossReduced.REDUCE_SIZE,
        )
        ctx.save_for_backward(inputs.view(input_shape), targets.view(input_shape))
        ctx.alpha = alpha
        ctx.gamma = gamma
        return loss.sum()

    @staticmethod
    def backward(ctx, grad_output):
        inputs, targets = ctx.saved_tensors
        alpha = ctx.alpha
        gamma = ctx.gamma
        n_elements = inputs.numel()
        input_shape = inputs.shape
        grad_inputs = torch.empty(
            inputs.shape, dtype=grad_output.dtype, device=grad_output.device
        )
        inputs_ptr = inputs.view(-1).contiguous()
        targets_ptr = targets.reshape(-1).contiguous()
        assert grad_output.numel() == 1
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        sigmoid_focal_loss_bwd_kernel_reduce[grid](
            inputs_ptr,
            targets_ptr,
            grad_inputs,
            grad_output,
            alpha,
            gamma,
            n_elements,
            SigmoidFocalLossReduced.BLOCK_SIZE,
        )
        return grad_inputs.view(input_shape), None, None, None


triton_sigmoid_focal_loss_reduce = SigmoidFocalLossReduced.apply
