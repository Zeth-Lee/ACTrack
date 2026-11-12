# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"cxpr_block_size": 128}),
        triton.Config({"cxpr_block_size": 256}),
        triton.Config({"cxpr_block_size": 512}),
        triton.Config({"cxpr_block_size": 1024}),
        triton.Config({"cxpr_block_size": 2048}),
        triton.Config({"cxpr_block_size": 4096}),
        triton.Config({"cxpr_block_size": 8192}),
    ],
    key=["num_boxes"],
)
@triton.jit
def _nms_suppression_kernel(

    iou_mask_ptr: tl.tensor,
    keep_mask_ptr: tl.tensor,

    num_boxes: tl.int32,

    iou_mask_stride: tl.int32,

    cxpr_block_size: tl.constexpr,
) -> None:


    for current_box_idx in range(num_boxes - 1):

        is_kept = tl.load(keep_mask_ptr + current_box_idx)
        if is_kept:


            iou_row_offset = current_box_idx * iou_mask_stride


            next_box_idx = current_box_idx + 1
            remaining_boxes = num_boxes - next_box_idx


            for block_idx in range(tl.cdiv(remaining_boxes, cxpr_block_size)):

                block_start = next_box_idx + block_idx * cxpr_block_size
                target_box_offsets = block_start + tl.arange(0, cxpr_block_size)
                target_box_mask = target_box_offsets < num_boxes


                suppression_mask = tl.load(
                    iou_mask_ptr + iou_row_offset + target_box_offsets,
                    mask=target_box_mask,
                    other=False,
                )
                suppression_mask = tl.cast(suppression_mask, tl.int1)


                tl.store(
                    keep_mask_ptr + target_box_offsets, False, mask=suppression_mask
                )


            tl.debug_barrier()


def nms_triton(
    ious: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:

    assert scores.dim() == 1, "Scores must be 1D"
    iou_mask = ious > iou_threshold
    assert iou_mask.dim() == 2
    assert iou_mask.shape[0] == iou_mask.shape[1] == scores.shape[0]
    assert iou_mask.device == scores.device
    assert iou_mask.dtype == torch.bool

    num_boxes = scores.size(0)
    keep_mask = torch.ones(len(scores), device=scores.device, dtype=torch.bool)


    _, sorted_indices = torch.sort(scores, dim=0, stable=True, descending=True)
    iou_mask = iou_mask[sorted_indices][:, sorted_indices].contiguous()


    stage2_grid = (1,)
    _nms_suppression_kernel[stage2_grid](

        iou_mask_ptr=iou_mask,
        keep_mask_ptr=keep_mask,

        num_boxes=num_boxes,

        iou_mask_stride=iou_mask.stride(0),
    )

    return sorted_indices[keep_mask]
