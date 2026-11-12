# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import numpy as np
import pycocotools.mask as maskUtils
import torch
from pycocotools import mask as mask_util


def instance_masks_to_semantic_masks(
    instance_masks: torch.Tensor, num_instances: torch.Tensor
) -> torch.Tensor:


    masks_per_query = torch.split(instance_masks, num_instances.tolist())

    return torch.stack([torch.any(masks, dim=0) for masks in masks_per_query], dim=0)


def mask_intersection(masks1, masks2, block_size=16):


    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool

    result = torch.zeros(
        masks1.shape[0], masks2.shape[0], device=masks1.device, dtype=torch.long
    )
    for i in range(0, masks1.shape[0], block_size):
        for j in range(0, masks2.shape[0], block_size):
            intersection = (
                (masks1[i : i + block_size, None] * masks2[None, j : j + block_size])
                .flatten(-2)
                .sum(-1)
            )
            result[i : i + block_size, j : j + block_size] = intersection
    return result


def mask_iom(masks1, masks2):

    assert masks1.shape[1:] == masks2.shape[1:]
    assert masks1.dtype == torch.bool and masks2.dtype == torch.bool


    intersection = mask_intersection(masks1, masks2)
    area1 = masks1.flatten(-2).sum(-1)
    area2 = masks2.flatten(-2).sum(-1)
    min_area = torch.min(area1[:, None], area2[None, :])
    return intersection / (min_area + 1e-8)


def compute_boundary(seg):

    assert seg.ndim >= 2
    e = torch.zeros_like(seg)
    s = torch.zeros_like(seg)
    se = torch.zeros_like(seg)

    e[..., :, :-1] = seg[..., :, 1:]
    s[..., :-1, :] = seg[..., 1:, :]
    se[..., :-1, :-1] = seg[..., 1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[..., -1, :] = seg[..., -1, :] ^ e[..., -1, :]
    b[..., :, -1] = seg[..., :, -1] ^ s[..., :, -1]
    b[..., -1, -1] = 0
    return b


def dilation(mask, kernel_size):


    assert mask.ndim == 3
    kernel_size = int(kernel_size)
    assert kernel_size % 2 == 1, (
        f"Dilation expects a odd kernel size, got {kernel_size}"
    )

    if mask.is_cuda:
        m = mask.unsqueeze(1).to(torch.float16)
        k = torch.ones(1, 1, kernel_size, 1, dtype=m.dtype, device=m.device)

        result = torch.nn.functional.conv2d(m, k, padding="same")
        result = torch.nn.functional.conv2d(result, k.transpose(-1, -2), padding="same")
        return result.view_as(mask) > 0

    all_masks = mask.view(-1, mask.size(-2), mask.size(-1)).numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    import cv2

    processed = [torch.from_numpy(cv2.dilate(m, kernel)) for m in all_masks]
    return torch.stack(processed).view_as(mask).to(mask)


def compute_F_measure(
    gt_boundary_rle, gt_dilated_boundary_rle, dt_boundary_rle, dt_dilated_boundary_rle
):

    gt_match = maskUtils.merge([gt_boundary_rle, dt_dilated_boundary_rle], True)
    dt_match = maskUtils.merge([dt_boundary_rle, gt_dilated_boundary_rle], True)

    n_dt = maskUtils.area(dt_boundary_rle)
    n_gt = maskUtils.area(gt_boundary_rle)

    if n_dt == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_dt > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_dt == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = maskUtils.area(dt_match) / float(n_dt)
        recall = maskUtils.area(gt_match) / float(n_gt)


    if precision + recall == 0:
        f_val = 0
    else:
        f_val = 2 * precision * recall / (precision + recall)

    return f_val


@torch.no_grad()
def rle_encode(orig_mask, return_areas=False):

    assert orig_mask.ndim == 3, "Mask must be of shape (N, H, W)"
    assert orig_mask.dtype == torch.bool, "Mask must have dtype=torch.bool"

    if orig_mask.numel() == 0:
        return []


    mask = orig_mask.transpose(1, 2)


    flat_mask = mask.reshape(mask.shape[0], -1)
    if return_areas:
        mask_areas = flat_mask.sum(-1).tolist()

    differences = torch.ones(
        mask.shape[0], flat_mask.shape[1] + 1, device=mask.device, dtype=torch.bool
    )
    differences[:, 1:-1] = flat_mask[:, :-1] != flat_mask[:, 1:]
    differences[:, 0] = flat_mask[:, 0]
    _, change_indices = torch.where(differences)

    try:
        boundaries = torch.cumsum(differences.sum(-1), 0).cpu()
    except RuntimeError as _:
        boundaries = torch.cumsum(differences.cpu().sum(-1), 0)

    change_indices_clone = change_indices.clone()

    for i in range(mask.shape[0]):

        beg = 0 if i == 0 else boundaries[i - 1].item()
        end = boundaries[i].item()
        change_indices[beg + 1 : end] -= change_indices_clone[beg : end - 1]


    change_indices = change_indices.tolist()

    batch_rles = []

    for i in range(mask.shape[0]):
        beg = 0 if i == 0 else boundaries[i - 1].item()
        end = boundaries[i].item()
        run_lengths = change_indices[beg:end]

        uncompressed_rle = {"counts": run_lengths, "size": list(orig_mask.shape[1:])}
        h, w = uncompressed_rle["size"]
        rle = mask_util.frPyObjects(uncompressed_rle, h, w)
        rle["counts"] = rle["counts"].decode("utf-8")
        if return_areas:
            rle["area"] = mask_areas[i]
        batch_rles.append(rle)

    return batch_rles


def robust_rle_encode(masks):


    assert masks.ndim == 3, "Mask must be of shape (N, H, W)"
    assert masks.dtype == torch.bool, "Mask must have dtype=torch.bool"

    try:
        return rle_encode(masks)
    except RuntimeError as _:
        masks = masks.cpu().numpy()
        rles = [
            mask_util.encode(
                np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F")
            )[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")
        return rles


def ann_to_rle(segm, im_info):

    h, w = im_info["height"], im_info["width"]
    if isinstance(segm, list):


        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif isinstance(segm["counts"], list):

        rle = mask_util.frPyObjects(segm, h, w)
    else:

        rle = segm
    return rle
