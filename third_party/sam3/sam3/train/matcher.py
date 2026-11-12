# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import numpy as np
import torch
from sam3.model.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from scipy.optimize import linear_sum_assignment
from torch import nn


def _do_matching(cost, repeats=1, return_tgt_indices=False, do_filtering=False):
    if repeats > 1:
        cost = np.tile(cost, (1, repeats))

    i, j = linear_sum_assignment(cost)
    if do_filtering:

        valid_thresh = 1e8
        valid_ijs = [(ii, jj) for ii, jj in zip(i, j) if cost[ii, jj] < valid_thresh]
        i, j = zip(*valid_ijs) if len(valid_ijs) > 0 else ([], [])
        i, j = np.array(i, dtype=np.int64), np.array(j, dtype=np.int64)
    if return_tgt_indices:
        return i, j
    order = np.argsort(j)
    return i[order]


class HungarianMatcher(nn.Module):


    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2,
    ):

        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid() if focal_loss else nn.Softmax(-1)
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, (
            "all costs cant be 0"
        )
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, batched_targets):

        bs, num_queries = outputs["pred_logits"].shape[:2]


        out_prob = self.norm(
            outputs["pred_logits"].flatten(0, 1)
        )
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  


        tgt_bbox = batched_targets["boxes"]

        if "positive_map" in batched_targets:

            positive_map = batched_targets["positive_map"]
            assert len(tgt_bbox) == len(positive_map)

            if self.focal_loss:
                positive_map = positive_map > 1e-4
                alpha = self.focal_alpha
                gamma = self.focal_gamma
                neg_cost_class = (
                    (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
                )
                pos_cost_class = (
                    alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                )
                cost_class = (
                    (pos_cost_class - neg_cost_class).unsqueeze(1)
                    * positive_map.unsqueeze(0)
                ).sum(-1)
            else:

                cost_class = -(out_prob.unsqueeze(1) * positive_map.unsqueeze(0)).sum(
                    -1
                )
        else:

            tgt_ids = batched_targets["labels"]
            assert len(tgt_bbox) == len(tgt_ids)

            if self.focal_loss:
                alpha = self.focal_alpha
                gamma = self.focal_gamma
                neg_cost_class = (
                    (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
                )
                pos_cost_class = (
                    alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                )
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            else:


                cost_class = -out_prob[:, tgt_ids]


        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        assert cost_class.shape == cost_bbox.shape


        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )


        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu().numpy()

        sizes = torch.cumsum(batched_targets["num_boxes"], -1)[:-1]
        costs = [c[i] for i, c in enumerate(np.split(C, sizes.cpu().numpy(), axis=-1))]
        indices = [_do_matching(c) for c in costs]
        batch_idx = torch.as_tensor(
            sum([[i] * len(src) for i, src in enumerate(indices)], []), dtype=torch.long
        )
        src_idx = torch.from_numpy(np.concatenate(indices)).long()
        return batch_idx, src_idx


class BinaryHungarianMatcher(nn.Module):


    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
    ):

        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid()
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, (
            "all costs cant be 0"
        )

    @torch.no_grad()
    def forward(self, outputs, batched_targets, repeats=0, repeat_batch=1):

        if repeat_batch != 1:
            raise NotImplementedError("please use BinaryHungarianMatcherV2 instead")

        bs, num_queries = outputs["pred_logits"].shape[:2]


        out_prob = self.norm(outputs["pred_logits"].flatten(0, 1)).squeeze(
            -1
        )
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  


        tgt_bbox = batched_targets["boxes"]


        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        cost_class = -out_prob.unsqueeze(-1).expand_as(cost_bbox)

        assert cost_class.shape == cost_bbox.shape


        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )


        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu().numpy()

        sizes = torch.cumsum(batched_targets["num_boxes"], -1)[:-1]
        costs = [c[i] for i, c in enumerate(np.split(C, sizes.cpu().numpy(), axis=-1))]
        return_tgt_indices = False
        for c in costs:
            n_targ = c.shape[1]
            if repeats > 1:
                n_targ *= repeats
            if c.shape[0] < n_targ:
                return_tgt_indices = True
                break
        if return_tgt_indices:
            indices, tgt_indices = zip(
                *(
                    _do_matching(
                        c, repeats=repeats, return_tgt_indices=return_tgt_indices
                    )
                    for c in costs
                )
            )
            tgt_indices = list(tgt_indices)
            for i in range(1, len(tgt_indices)):
                tgt_indices[i] += sizes[i - 1].item()
            tgt_idx = torch.from_numpy(np.concatenate(tgt_indices)).long()
        else:
            indices = [_do_matching(c, repeats=repeats) for c in costs]
            tgt_idx = None

        batch_idx = torch.as_tensor(
            sum([[i] * len(src) for i, src in enumerate(indices)], []), dtype=torch.long
        )
        src_idx = torch.from_numpy(np.concatenate(indices)).long()
        return batch_idx, src_idx, tgt_idx


class BinaryFocalHungarianMatcher(nn.Module):


    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        alpha: float = 0.25,
        gamma: float = 2.0,
        stable: bool = False,
    ):

        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid()
        self.alpha = alpha
        self.gamma = gamma
        self.stable = stable
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, (
            "all costs cant be 0"
        )

    @torch.no_grad()
    def forward(self, outputs, batched_targets, repeats=1, repeat_batch=1):

        if repeat_batch != 1:
            raise NotImplementedError("please use BinaryHungarianMatcherV2 instead")

        bs, num_queries = outputs["pred_logits"].shape[:2]


        out_score = outputs["pred_logits"].flatten(0, 1).squeeze(-1)
        out_prob = self.norm(out_score)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  


        tgt_bbox = batched_targets["boxes"]


        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)


        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )


        if self.stable:
            rescaled_giou = (-cost_giou + 1) / 2
            out_prob = out_prob.unsqueeze(-1).expand_as(cost_bbox) * rescaled_giou
            cost_class = -self.alpha * (1 - out_prob) ** self.gamma * torch.log(
                out_prob
            ) + (1 - self.alpha) * out_prob**self.gamma * torch.log(1 - out_prob)
        else:

            log_out_prob = torch.nn.functional.logsigmoid(out_score)
            log_one_minus_out_prob = torch.nn.functional.logsigmoid(-out_score)
            cost_class = (
                -self.alpha * (1 - out_prob) ** self.gamma * log_out_prob
                + (1 - self.alpha) * out_prob**self.gamma * log_one_minus_out_prob
            )
        if not self.stable:
            cost_class = cost_class.unsqueeze(-1).expand_as(cost_bbox)

        assert cost_class.shape == cost_bbox.shape


        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu().numpy()

        sizes = torch.cumsum(batched_targets["num_boxes"], -1)[:-1]
        costs = [c[i] for i, c in enumerate(np.split(C, sizes.cpu().numpy(), axis=-1))]
        return_tgt_indices = False
        for c in costs:
            n_targ = c.shape[1]
            if repeats > 1:
                n_targ *= repeats
            if c.shape[0] < n_targ:
                return_tgt_indices = True
                break
        if return_tgt_indices:
            indices, tgt_indices = zip(
                *(
                    _do_matching(
                        c, repeats=repeats, return_tgt_indices=return_tgt_indices
                    )
                    for c in costs
                )
            )
            tgt_indices = list(tgt_indices)
            for i in range(1, len(tgt_indices)):
                tgt_indices[i] += sizes[i - 1].item()
            tgt_idx = torch.from_numpy(np.concatenate(tgt_indices)).long()
        else:
            indices = [_do_matching(c, repeats=repeats) for c in costs]
            tgt_idx = None

        batch_idx = torch.as_tensor(
            sum([[i] * len(src) for i, src in enumerate(indices)], []), dtype=torch.long
        )
        src_idx = torch.from_numpy(np.concatenate(indices)).long()
        return batch_idx, src_idx, tgt_idx


class BinaryHungarianMatcherV2(nn.Module):


    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal: bool = False,
        alpha: float = 0.25,
        gamma: float = 2.0,
        stable: bool = False,
        remove_samples_with_0_gt: bool = True,
    ):

        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Sigmoid()
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, (
            "all costs cant be 0"
        )
        self.focal = focal
        if focal:
            self.alpha = alpha
            self.gamma = gamma
            self.stable = stable
        self.remove_samples_with_0_gt = remove_samples_with_0_gt

    @torch.no_grad()
    def forward(
        self,
        outputs,
        batched_targets,
        repeats=1,
        repeat_batch=1,
        out_is_valid=None,
        target_is_valid_padded=None,
    ):

        _, num_queries = outputs["pred_logits"].shape[:2]

        out_score = outputs["pred_logits"].squeeze(-1)  
        out_bbox = outputs["pred_boxes"]  

        device = out_score.device

        num_boxes = batched_targets["num_boxes"].cpu()


        tgt_bbox = batched_targets["boxes_padded"]
        if self.remove_samples_with_0_gt:

            batch_keep = num_boxes > 0
            num_boxes = num_boxes[batch_keep]
            tgt_bbox = tgt_bbox[batch_keep]
            if target_is_valid_padded is not None:
                target_is_valid_padded = target_is_valid_padded[batch_keep]

        if repeat_batch > 1:


            num_boxes = num_boxes.repeat(repeat_batch)
            tgt_bbox = tgt_bbox.repeat(repeat_batch, 1, 1)
            if target_is_valid_padded is not None:
                target_is_valid_padded = target_is_valid_padded.repeat(repeat_batch, 1)


        if self.remove_samples_with_0_gt:
            if repeat_batch > 1:
                batch_keep = batch_keep.repeat(repeat_batch)
            out_score = out_score[batch_keep]
            out_bbox = out_bbox[batch_keep]
            if out_is_valid is not None:
                out_is_valid = out_is_valid[batch_keep]
        assert out_bbox.shape[0] == tgt_bbox.shape[0]
        assert out_bbox.shape[0] == num_boxes.shape[0]


        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)


        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        out_prob = self.norm(out_score)
        if not self.focal:
            cost_class = -out_prob.unsqueeze(-1).expand_as(cost_bbox)
        else:
            if self.stable:
                rescaled_giou = (-cost_giou + 1) / 2
                out_prob = out_prob.unsqueeze(-1).expand_as(cost_bbox) * rescaled_giou
                cost_class = -self.alpha * (1 - out_prob) ** self.gamma * torch.log(
                    out_prob
                ) + (1 - self.alpha) * out_prob**self.gamma * torch.log(1 - out_prob)
            else:

                log_out_prob = torch.nn.functional.logsigmoid(out_score)
                log_one_minus_out_prob = torch.nn.functional.logsigmoid(-out_score)
                cost_class = (
                    -self.alpha * (1 - out_prob) ** self.gamma * log_out_prob
                    + (1 - self.alpha) * out_prob**self.gamma * log_one_minus_out_prob
                )
            if not self.stable:
                cost_class = cost_class.unsqueeze(-1).expand_as(cost_bbox)

        assert cost_class.shape == cost_bbox.shape


        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )


        do_filtering = out_is_valid is not None or target_is_valid_padded is not None
        if out_is_valid is not None:
            C = torch.where(out_is_valid[:, :, None], C, 1e9)
        if target_is_valid_padded is not None:
            C = torch.where(target_is_valid_padded[:, None, :], C, 1e9)
        C = C.cpu().numpy()
        costs = [C[i, :, :s] for i, s in enumerate(num_boxes.tolist())]
        return_tgt_indices = (
            do_filtering or torch.any(num_queries < num_boxes * max(repeats, 1)).item()
        )
        if len(costs) == 0:


            indices = []
            tgt_idx = torch.zeros(0).long().to(device) if return_tgt_indices else None
        elif return_tgt_indices:
            indices, tgt_indices = zip(
                *(
                    _do_matching(
                        c,
                        repeats=repeats,
                        return_tgt_indices=return_tgt_indices,
                        do_filtering=do_filtering,
                    )
                    for c in costs
                )
            )
            tgt_indices = list(tgt_indices)
            sizes = torch.cumsum(num_boxes, -1)[:-1]
            for i in range(1, len(tgt_indices)):
                tgt_indices[i] += sizes[i - 1].item()
            tgt_idx = torch.from_numpy(np.concatenate(tgt_indices)).long().to(device)
        else:
            indices = [
                _do_matching(c, repeats=repeats, do_filtering=do_filtering)
                for c in costs
            ]
            tgt_idx = None

        if self.remove_samples_with_0_gt:
            kept_inds = batch_keep.nonzero().squeeze(1)
            batch_idx = torch.as_tensor(
                sum([[kept_inds[i]] * len(src) for i, src in enumerate(indices)], []),
                dtype=torch.long,
                device=device,
            )
        else:
            batch_idx = torch.as_tensor(
                sum([[i] * len(src) for i, src in enumerate(indices)], []),
                dtype=torch.long,
                device=device,
            )


        if len(indices) > 0:
            src_idx = torch.from_numpy(np.concatenate(indices)).long().to(device)
        else:
            src_idx = torch.empty(0, dtype=torch.long, device=device)
        return batch_idx, src_idx, tgt_idx


class BinaryOneToManyMatcher(nn.Module):


    def __init__(
        self,
        alpha: float = 0.3,
        threshold: float = 0.4,
        topk: int = 6,
    ):

        super().__init__()
        self.norm = nn.Sigmoid()
        self.alpha = alpha
        self.threshold = threshold
        self.topk = topk

    @torch.no_grad()
    def forward(
        self,
        outputs,
        batched_targets,
        repeats=1,
        repeat_batch=1,
        out_is_valid=None,
        target_is_valid_padded=None,
    ):

        assert repeats <= 1 and repeat_batch <= 1
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = self.norm(outputs["pred_logits"]).squeeze(-1)  
        out_bbox = outputs["pred_boxes"]  

        num_boxes = batched_targets["num_boxes"]


        tgt_bbox = batched_targets["boxes_padded"]
        assert len(tgt_bbox) == bs
        num_targets = tgt_bbox.shape[1]
        if num_targets == 0:
            return (
                torch.empty(0, dtype=torch.long, device=out_prob.device),
                torch.empty(0, dtype=torch.long, device=out_prob.device),
                torch.empty(0, dtype=torch.long, device=out_prob.device),
            )

        iou, _ = box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        assert iou.shape == (bs, num_queries, num_targets)


        C = self.alpha * out_prob.unsqueeze(-1) + (1 - self.alpha) * iou
        if out_is_valid is not None:
            C = torch.where(out_is_valid[:, :, None], C, -1e9)
        if target_is_valid_padded is not None:
            C = torch.where(target_is_valid_padded[:, None, :], C, -1e9)


        matches = C > torch.quantile(
            C, 1 - self.topk / num_queries, dim=1, keepdim=True
        )


        matches = matches & (C > self.threshold)
        if out_is_valid is not None:
            matches = matches & out_is_valid[:, :, None]
        if target_is_valid_padded is not None:
            matches = matches & target_is_valid_padded[:, None, :]


        matches = matches & (
            torch.arange(0, num_targets, device=num_boxes.device)[None]
            < num_boxes[:, None]
        ).unsqueeze(1)

        batch_idx, src_idx, tgt_idx = torch.nonzero(matches, as_tuple=True)

        cum_num_boxes = torch.cat(
            [
                torch.zeros(1, dtype=num_boxes.dtype, device=num_boxes.device),
                num_boxes.cumsum(-1)[:-1],
            ]
        )
        tgt_idx += cum_num_boxes[batch_idx]

        return batch_idx, src_idx, tgt_idx
