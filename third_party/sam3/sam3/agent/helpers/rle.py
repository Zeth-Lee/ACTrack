# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import numpy as np
import torch
from pycocotools import mask as mask_util


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
