# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import dataclasses
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from sam3.model import box_ops
from sam3.model.data_misc import BatchedInferenceMetadata, interpolate
from sam3.train.masks_ops import rle_encode, robust_rle_encode
from torch import nn


class PostProcessNullOp(nn.Module):
    def __init__(self, **kwargs):
        super(PostProcessNullOp).__init__()
        pass

    def forward(self, input):
        pass

    def process_results(self, **kwargs):
        return kwargs["find_stages"]


class PostProcessImage(nn.Module):


    def __init__(
        self,
        max_dets_per_img: int,
        iou_type="bbox",
        to_cpu: bool = True,
        use_original_ids: bool = False,
        use_original_sizes_box: bool = False,
        use_original_sizes_mask: bool = False,
        convert_mask_to_rle: bool = False,
        always_interpolate_masks_on_gpu: bool = True,
        use_presence: bool = True,
        detection_threshold: float = -1.0,
    ) -> None:
        super().__init__()
        self.max_dets_per_img = max_dets_per_img
        self.iou_type = iou_type
        self.to_cpu = to_cpu
        self.convert_mask_to_rle = convert_mask_to_rle
        self.always_interpolate_masks_on_gpu = always_interpolate_masks_on_gpu

        self.use_presence = use_presence
        self.detection_threshold = detection_threshold
        self.use_original_ids = use_original_ids
        self.use_original_sizes_box = use_original_sizes_box
        self.use_original_sizes_mask = use_original_sizes_mask

    @torch.no_grad()
    def forward(
        self,
        outputs,
        target_sizes_boxes,
        target_sizes_masks,
        forced_labels=None,
        consistent=False,
        ret_tensordict: bool = False,
    ):

        if ret_tensordict:
            assert consistent is True, (
                "We don't support returning TensorDict if the outputs have different shapes"
            )
            assert self.detection_threshold <= 0.0, "TODO: implement?"
            try:
                from tensordict import TensorDict
            except ImportError:
                logging.info(
                    "tensordict is not installed. Install by running `pip install tensordict --no-deps`. Falling back by setting `ret_tensordict=False`"
                )
                ret_tensordict = False

        out_bbox = outputs["pred_boxes"] if "pred_boxes" in outputs else None
        out_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"] if self.iou_type == "segm" else None
        out_probs = out_logits.sigmoid()
        if self.use_presence:
            presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            out_probs = out_probs * presence_score

        assert target_sizes_boxes.shape[1] == 2
        assert target_sizes_masks.shape[1] == 2
        batch_size = target_sizes_boxes.shape[0]

        boxes, scores, labels, keep = self._process_boxes_and_labels(
            target_sizes_boxes, forced_labels, out_bbox, out_probs
        )
        assert boxes is None or len(boxes) == batch_size
        out_masks = self._process_masks(
            target_sizes_masks, pred_masks, consistent=consistent, keep=keep
        )
        del pred_masks

        if boxes is None:
            assert out_masks is not None
            assert not ret_tensordict, (
                "We don't support returning TensorDict if the output does not contain boxes"
            )
            B = len(out_masks)
            boxes = [None] * B
            scores = [None] * B
            labels = [None] * B

        results = {
            "scores": scores,
            "labels": labels,
            "boxes": boxes,
        }
        if out_masks is not None:
            if self.convert_mask_to_rle:
                results.update(masks_rle=out_masks)
            else:
                results.update(masks=out_masks)

        if ret_tensordict:
            results = TensorDict(results).auto_batch_size_()
            if self.to_cpu:
                results = results.cpu()
        else:

            results = [
                dict(zip(results.keys(), res_tuple))
                for res_tuple in zip(*results.values())
            ]

        return results

    def _process_masks(self, target_sizes, pred_masks, consistent=True, keep=None):
        if pred_masks is None:
            return None
        if self.always_interpolate_masks_on_gpu:
            gpu_device = target_sizes.device
            assert gpu_device.type == "cuda"
            pred_masks = pred_masks.to(device=gpu_device)
        if consistent:
            assert keep is None, "TODO: implement?"

            target_size = target_sizes.unique(dim=0)
            assert target_size.size(0) == 1, "Expecting all target sizes to be equal"
            out_masks = (
                interpolate(
                    pred_masks,
                    target_size.squeeze().tolist(),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()
                > 0.5
            )
            if self.convert_mask_to_rle:
                raise RuntimeError("TODO: implement?")
            if self.to_cpu:
                out_masks = out_masks.cpu()
        else:
            out_masks = [[]] * len(pred_masks)

            assert keep is None or len(keep) == len(pred_masks)
            for i, mask in enumerate(pred_masks):
                h, w = target_sizes[i]
                if keep is not None:
                    mask = mask[keep[i]]

                try:
                    interpolated = (
                        interpolate(
                            mask.unsqueeze(1),
                            (h, w),
                            mode="bilinear",
                            align_corners=False,
                        ).sigmoid()
                        > 0.5
                    )
                except Exception as e:
                    logging.info("Issue found, reverting to CPU mode!")
                    mask_device = mask.device
                    mask = mask.cpu()
                    interpolated = (
                        interpolate(
                            mask.unsqueeze(1),
                            (h, w),
                            mode="bilinear",
                            align_corners=False,
                        ).sigmoid()
                        > 0.5
                    )
                    interpolated = interpolated.to(mask_device)

                if self.convert_mask_to_rle:
                    out_masks[i] = robust_rle_encode(interpolated.squeeze(1))
                else:
                    out_masks[i] = interpolated
                    if self.to_cpu:
                        out_masks[i] = out_masks[i].cpu()

        return out_masks

    def _process_boxes_and_labels(
        self, target_sizes, forced_labels, out_bbox, out_probs
    ):
        if out_bbox is None:
            return None, None, None, None
        assert len(out_probs) == len(target_sizes)
        if self.to_cpu:
            out_probs = out_probs.cpu()
        scores, labels = out_probs.max(-1)
        if forced_labels is None:
            labels = torch.ones_like(labels)
        else:
            labels = forced_labels[:, None].expand_as(labels)


        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.to_cpu:
            boxes = boxes.cpu()

        keep = None
        if self.detection_threshold > 0:

            keep = scores > self.detection_threshold
            assert len(keep) == len(boxes) == len(scores) == len(labels)

            boxes = [b[k.to(b.device)] for b, k in zip(boxes, keep)]
            scores = [s[k.to(s.device)] for s, k in zip(scores, keep)]
            labels = [l[k.to(l.device)] for l, k in zip(labels, keep)]

        return boxes, scores, labels, keep

    def process_results(
        self, find_stages, find_metadatas: List[BatchedInferenceMetadata], **kwargs
    ):
        if find_stages.loss_stages is not None:
            find_metadatas = [find_metadatas[i] for i in find_stages.loss_stages]
        assert len(find_stages) == len(find_metadatas)
        results = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            img_size_for_boxes = (
                meta.original_size
                if self.use_original_sizes_box
                else torch.ones_like(meta.original_size)
            )
            img_size_for_masks = (
                meta.original_size
                if self.use_original_sizes_mask
                else torch.ones_like(meta.original_size)
            )
            detection_results = self(
                outputs,
                img_size_for_boxes,
                img_size_for_masks,
                forced_labels=(
                    meta.original_category_id if self.use_original_ids else None
                ),
            )
            ids = (
                meta.original_image_id if self.use_original_ids else meta.coco_image_id
            )
            assert len(detection_results) == len(ids)
            for img_id, result in zip(ids, detection_results):
                if img_id.item() not in results:
                    results[img_id.item()] = result
                else:
                    assert set(results[img_id.item()].keys()) == set(result.keys())
                    for k in result.keys():
                        if isinstance(result[k], torch.Tensor):
                            results[img_id.item()][k] = torch.cat(
                                [results[img_id.item()][k], result[k]], dim=0
                            )
                        elif isinstance(result[k], list):
                            results[img_id.item()][k] += result[k]
                        else:
                            raise NotImplementedError(
                                f"Unexpected type {type(result[k])} in result."
                            )

        for img_id, result in results.items():
            if (
                self.max_dets_per_img > 0
                and len(result["scores"]) > self.max_dets_per_img
            ):
                _, topk_indexes = torch.topk(
                    result["scores"], self.max_dets_per_img, dim=0
                )
                if self.to_cpu:
                    topk_indexes = topk_indexes.cpu()
                for k in result.keys():
                    if isinstance(results[img_id][k], list):
                        results[img_id][k] = [
                            results[img_id][k][i] for i in topk_indexes.tolist()
                        ]
                    else:
                        results[img_id][k] = results[img_id][k].to(topk_indexes.device)[
                            topk_indexes
                        ]

        return results


class PostProcessAPIVideo(PostProcessImage):


    def __init__(
        self,
        *args,
        to_cpu: bool = True,
        convert_mask_to_rle: bool = False,
        always_interpolate_masks_on_gpu: bool = True,
        prob_thresh: float = 0.5,
        use_presence: bool = False,
        **kwargs,
    ):
        super().__init__(
            *args,


            convert_mask_to_rle=False,


            always_interpolate_masks_on_gpu=always_interpolate_masks_on_gpu,
            use_presence=use_presence,
            **kwargs,
        )

        self.EXPECTED_KEYS = [
            "pred_logits",
            "pred_boxes",
            "pred_masks",
        ]

        self.convert_mask_to_rle_for_video = convert_mask_to_rle
        self.to_cpu_for_video = to_cpu
        self.prob_thresh = prob_thresh

    def process_results(
        self, find_stages, find_metadatas: List[BatchedInferenceMetadata], **kwargs
    ):


        try:
            from tensordict import TensorDict
        except ImportError as e:
            logging.error(
                "tensordict is not installed, please install by running `pip install tensordict --no-deps`"
            )
            raise e


        assert len(find_stages) > 0, "There is nothing to postprocess?"
        PROMPT_AXIS, OBJ_QUERY_AXIS = (0, 1)
        NO_OBJ_ID = -1

        tracked_objects_packed_idx = defaultdict(list)

        tracked_objects_frame_idx = defaultdict(list)
        total_num_preds = 0

        vid_preds_packed: List[TensorDict] = []
        vid_masklets_rle_packed: List[Optional[Dict]] = []
        video_id = -1

        for frame_idx, (frame_outs, meta) in enumerate(
            zip(find_stages, find_metadatas)
        ):

            frame_outs_td = TensorDict(
                {k: frame_outs[k] for k in self.EXPECTED_KEYS}
            ).auto_batch_size_()
            meta_td = TensorDict(
                dataclasses.asdict(meta)
            ).auto_batch_size_()
            unique_vid_id = meta.original_image_id.unique()
            assert unique_vid_id.size(0) == 1
            if video_id == -1:
                video_id = unique_vid_id.item()
            else:
                assert video_id == unique_vid_id.item(), (
                    "We can only postprocess one video per datapoint"
                )

            obj_ids_per_frame = frame_outs["pred_object_ids"]
            assert obj_ids_per_frame.size(-1) == frame_outs["pred_logits"].size(-2)
            if self.prob_thresh is not None:


                pred_probs = frame_outs["pred_logits"].sigmoid().squeeze(-1)
                obj_ids_per_frame = torch.where(
                    pred_probs >= self.prob_thresh, obj_ids_per_frame, NO_OBJ_ID
                )
            tracked_obj_ids_idx = torch.where(obj_ids_per_frame != NO_OBJ_ID)

            tracked_obj_ids = [
                (p_id.item(), obj_ids_per_frame[p_id, q_id].item())
                for p_id, q_id in zip(
                    tracked_obj_ids_idx[PROMPT_AXIS],
                    tracked_obj_ids_idx[OBJ_QUERY_AXIS],
                )
            ]
            if len(tracked_obj_ids) == 0:
                continue

            for oid in tracked_obj_ids:
                tracked_objects_packed_idx[oid].append(total_num_preds)
                tracked_objects_frame_idx[oid].append(frame_idx)
                total_num_preds += 1


            tracked_objs_outs_td = frame_outs_td[
                tracked_obj_ids_idx
            ]
            meta_td = meta_td[tracked_obj_ids_idx[PROMPT_AXIS].cpu()]
            if self.always_interpolate_masks_on_gpu:
                gpu_device = meta_td["original_size"].device
                assert gpu_device.type == "cuda"
                tracked_objs_outs_td = tracked_objs_outs_td.to(device=gpu_device)
            frame_results_td = self(
                tracked_objs_outs_td.unsqueeze(1),
                (
                    meta_td["original_size"]
                    if self.use_original_sizes
                    else torch.ones_like(meta_td["original_size"])
                ),
                forced_labels=(
                    meta_td["original_category_id"] if self.use_original_ids else None
                ),
                consistent=True,
                ret_tensordict=True,
            ).squeeze(1)
            del tracked_objs_outs_td


            if self.convert_mask_to_rle_for_video:
                interpolated_binary_masks = frame_results_td.pop("masks")
                rle_list = rle_encode(interpolated_binary_masks, return_areas=True)
                vid_masklets_rle_packed.extend(rle_list)

            if self.to_cpu_for_video:
                frame_results_td = frame_results_td.cpu()
            vid_preds_packed.append(frame_results_td)

        if len(vid_preds_packed) == 0:
            logging.debug(f"Video {video_id} has no predictions")
            return {video_id: []}

        vid_preds_packed = torch.cat(vid_preds_packed, dim=0)

        num_preds = len(tracked_objects_packed_idx)
        num_frames = len(find_stages)


        padded_frames_results = TensorDict(
            {
                k: torch.zeros(
                    num_preds, num_frames, *v.shape[1:], device=v.device, dtype=v.dtype
                )
                for k, v in vid_preds_packed.items()
            },
            batch_size=[
                num_preds,
                num_frames,
            ],
        )
        padded_frames_results["scores"][...] = -1e8  

        tracklet_scores = []
        tracklet_labels = []


        if self.convert_mask_to_rle_for_video:
            vid_masklets_rle_padded = [[None] * num_frames for _ in range(num_preds)]
        for o_idx, oid in enumerate(tracked_objects_packed_idx):
            oid2packed_idx = tracked_objects_packed_idx[oid]
            oid2padded_idx = tracked_objects_frame_idx[oid]
            obj_packed_results = vid_preds_packed[oid2packed_idx]
            padded_frames_results[o_idx][oid2padded_idx] = obj_packed_results
            if self.convert_mask_to_rle_for_video:
                for packed_idx, padded_idx in zip(oid2packed_idx, oid2padded_idx):
                    vid_masklets_rle_padded[o_idx][padded_idx] = (
                        vid_masklets_rle_packed[packed_idx]
                    )


            tracklet_scores.append(obj_packed_results["scores"].mean())


            tracklet_labels.append(obj_packed_results["labels"].mode()[0])

        results = padded_frames_results.to_dict()
        results["scores"] = torch.stack(tracklet_scores, dim=0)
        results["labels"] = torch.stack(tracklet_labels, dim=0)
        if self.convert_mask_to_rle_for_video:
            results["masks_rle"] = vid_masklets_rle_padded

        results["per_frame_scores"] = padded_frames_results["scores"]

        return {video_id: results}


class PostProcessTracking(PostProcessImage):


    def __init__(
        self,
        max_dets_per_img: int,
        iou_type="bbox",
        force_single_mask: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(max_dets_per_img=max_dets_per_img, iou_type=iou_type, **kwargs)
        self.force_single_mask = force_single_mask

    def process_results(
        self, find_stages, find_metadatas: BatchedInferenceMetadata, **kwargs
    ):
        assert len(find_stages) == len(find_metadatas)
        results = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            if self.force_single_mask:
                scores, labels = outputs["pred_logits"].max(-1)
                m = []
                for i in range(len(outputs["pred_masks"])):
                    score, idx = scores[i].max(0)
                    m.append(outputs["pred_masks"][i][idx])
                outputs["pred_masks"] = torch.stack(m, 0).unsqueeze(1)
            detection_results = self(outputs, meta.original_size, consistent=False)
            assert len(detection_results) == len(meta.coco_image_id)
            results.update(
                {
                    (media_id.item(), object_id.item(), frame_index.item()): result
                    for media_id, object_id, frame_index, result in zip(
                        meta.original_image_id,
                        meta.object_id,
                        meta.frame_index,
                        detection_results,
                    )
                }
            )
        return results


class PostProcessCounting(nn.Module):


    def __init__(
        self,
        use_original_ids: bool = False,
        threshold: float = 0.5,
        use_presence: bool = False,
    ) -> None:

        super().__init__()
        self.use_original_ids = use_original_ids
        self.threshold = threshold
        self.use_presence = use_presence

    def forward(self, outputs, target_sizes):


        scores = torch.sigmoid(outputs["pred_logits"]).squeeze(-1)  
        if self.use_presence:
            presence_score = outputs["presence_logit_dec"].sigmoid()
            if presence_score.ndim == 1:
                presence_score = presence_score.unsqueeze(1)
            scores = scores * presence_score


        counts = (scores > self.threshold).float().sum(dim=1)

        assert len(counts) == len(target_sizes)
        results = []
        for count in counts:
            results.append({"count": count.item()})

        return results

    @torch.no_grad()
    def process_results(
        self, find_stages, find_metadatas: List[BatchedInferenceMetadata], **kwargs
    ):
        assert len(find_stages) == len(find_metadatas)
        results = {}
        for outputs, meta in zip(find_stages, find_metadatas):
            detection_results = self(
                outputs,
                meta.original_size,
            )
            ids = (
                meta.original_image_id if self.use_original_ids else meta.coco_image_id
            )
            assert len(detection_results) == len(ids)
            for img_id, result in zip(ids, detection_results):
                results[img_id.item()] = result

        return results
