# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


from __future__ import absolute_import, division, print_function, unicode_literals

import math
from typing import List, Tuple

import torch


from .boxes import Boxes


def pairwise_iou_rotated(boxes1, boxes2):

    return torch.ops.detectron2.box_iou_rotated(boxes1, boxes2)


class RotatedBoxes(Boxes):


    def __init__(self, tensor: torch.Tensor):

        device = (
            tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        )
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:


            tensor = tensor.reshape((0, 5)).to(dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 5, tensor.size()

        self.tensor = tensor

    def clone(self) -> "RotatedBoxes":

        return RotatedBoxes(self.tensor.clone())

    def to(self, device: torch.device, non_blocking: bool = False):

        return RotatedBoxes(self.tensor.to(device=device, non_blocking=non_blocking))

    def area(self) -> torch.Tensor:

        box = self.tensor
        area = box[:, 2] * box[:, 3]
        return area


    def normalize_angles(self) -> None:

        angle_tensor = (self.tensor[:, 4] + 180.0) % 360.0 - 180.0
        self.tensor = torch.cat((self.tensor[:, :4], angle_tensor[:, None]), dim=1)

    def clip(
        self, box_size: Tuple[int, int], clip_angle_threshold: float = 1.0
    ) -> None:

        h, w = box_size


        self.normalize_angles()

        idx = torch.where(torch.abs(self.tensor[:, 4]) <= clip_angle_threshold)[0]


        x1 = self.tensor[idx, 0] - self.tensor[idx, 2] / 2.0
        y1 = self.tensor[idx, 1] - self.tensor[idx, 3] / 2.0
        x2 = self.tensor[idx, 0] + self.tensor[idx, 2] / 2.0
        y2 = self.tensor[idx, 1] + self.tensor[idx, 3] / 2.0


        x1.clamp_(min=0, max=w)
        y1.clamp_(min=0, max=h)
        x2.clamp_(min=0, max=w)
        y2.clamp_(min=0, max=h)


        self.tensor[idx, 0] = (x1 + x2) / 2.0
        self.tensor[idx, 1] = (y1 + y2) / 2.0

        self.tensor[idx, 2] = torch.min(self.tensor[idx, 2], x2 - x1)
        self.tensor[idx, 3] = torch.min(self.tensor[idx, 3], y2 - y1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:

        box = self.tensor
        widths = box[:, 2]
        heights = box[:, 3]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "RotatedBoxes":

        if isinstance(item, int):
            return RotatedBoxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, (
            "Indexing on RotatedBoxes with {} failed to return a matrix!".format(item)
        )
        return RotatedBoxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "RotatedBoxes(" + str(self.tensor) + ")"

    def inside_box(
        self, box_size: Tuple[int, int], boundary_threshold: int = 0
    ) -> torch.Tensor:

        height, width = box_size

        cnt_x = self.tensor[..., 0]
        cnt_y = self.tensor[..., 1]
        half_w = self.tensor[..., 2] / 2.0
        half_h = self.tensor[..., 3] / 2.0
        a = self.tensor[..., 4]
        c = torch.abs(torch.cos(a * math.pi / 180.0))
        s = torch.abs(torch.sin(a * math.pi / 180.0))

        max_rect_dx = c * half_w + s * half_h
        max_rect_dy = c * half_h + s * half_w

        inds_inside = (
            (cnt_x - max_rect_dx >= -boundary_threshold)
            & (cnt_y - max_rect_dy >= -boundary_threshold)
            & (cnt_x + max_rect_dx < width + boundary_threshold)
            & (cnt_y + max_rect_dy < height + boundary_threshold)
        )

        return inds_inside

    def get_centers(self) -> torch.Tensor:

        return self.tensor[:, :2]

    def scale(self, scale_x: float, scale_y: float) -> None:

        self.tensor[:, 0] *= scale_x
        self.tensor[:, 1] *= scale_y
        theta = self.tensor[:, 4] * math.pi / 180.0
        c = torch.cos(theta)
        s = torch.sin(theta)


        self.tensor[:, 2] *= torch.sqrt((scale_x * c) ** 2 + (scale_y * s) ** 2)


        self.tensor[:, 3] *= torch.sqrt((scale_x * s) ** 2 + (scale_y * c) ** 2)


        self.tensor[:, 4] = torch.atan2(scale_x * s, scale_y * c) * 180 / math.pi

    @classmethod
    def cat(cls, boxes_list: List["RotatedBoxes"]) -> "RotatedBoxes":

        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, RotatedBoxes) for box in boxes_list])


        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @torch.jit.unused
    def __iter__(self):

        yield from self.tensor


def pairwise_iou(boxes1: RotatedBoxes, boxes2: RotatedBoxes) -> None:


    return pairwise_iou_rotated(boxes1.tensor, boxes2.tensor)
