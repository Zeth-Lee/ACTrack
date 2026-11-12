# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import copy
import gc
import logging
import os
from collections import defaultdict
from operator import xor
from pathlib import Path
from typing import List, Optional

import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.cocoeval import COCOeval
from sam3.eval.cgf1_eval import CGF1Eval
from sam3.eval.coco_eval_offline import convert_to_xywh
from sam3.model.box_ops import box_xywh_inter_union
from sam3.train.masks_ops import rle_encode
from sam3.train.utils import distributed as dist
from typing_extensions import override

try:
    import rapidjson as json
except ModuleNotFoundError:
    import json

from iopath.common.file_io import g_pathmgr


class YTVISevalMixin:


    @override
    def _prepare(self):

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(
                self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
            dts = self.cocoDt.loadAnns(
                self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
            )
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))


        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        self.evalImgs = defaultdict(list)
        self.eval = {}

    def computeIoU(self, imgId, catId):

        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 or len(dt) == 0:
            return []


        assert hasattr(self, "sort_inds_by_scores_in_iou"), (
            "subclasses that inherits YTVISevalMixin should set `self.sort_inds_by_scores_in_iou` "
            "(True for class mAP and phrase AP, False for demo F1)"
        )
        if self.sort_inds_by_scores_in_iou:
            inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
            dt = [dt[i] for i in inds]
            if len(dt) > p.maxDets[-1]:
                dt = dt[0 : p.maxDets[-1]]

        if p.iouType == "segm":
            g = [g["segmentations"] for g in gt]
            d = [d["segmentations"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bboxes"] for g in gt]
            d = [d["bboxes"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")

        def iou_tracklets(preds, gts):
            preds = torch.tensor(preds)
            gts = torch.tensor(gts)
            inter, union = box_xywh_inter_union(
                preds.unsqueeze(1), gts.unsqueeze(0)
            )
            inter = inter.sum(-1)
            union = union.sum(-1)
            assert (union > 0).all(), (
                "There exists a tracklet with zero GTs across time. This is suspicious"
            )
            return inter / union

        def iou_masklets(preds, gts):
            inter = 0
            union = 0
            for p_i, gt_i in zip(preds, gts):
                if p_i and gt_i:

                    inter += mask_util.area(
                        mask_util.merge([p_i, gt_i], intersect=True)
                    )
                    union += mask_util.area(
                        mask_util.merge([p_i, gt_i], intersect=False)
                    )
                elif gt_i:
                    union += mask_util.area(gt_i)
                elif p_i:
                    union += mask_util.area(p_i)
            if union > 0:
                iou = inter / union
                assert iou >= 0 and iou <= 1, "Encountered an error in IoU computation"
            else:
                assert np.isclose(inter, 0) and np.isclose(union, 0), (
                    "Encountered an error in IoU computation"
                )
                iou = 1
            return iou

        if p.iouType == "segm":
            ious = [[iou_masklets(d_i, g_i) for g_i in g] for d_i in d]
        else:
            ious = iou_tracklets(d, g)
        return np.array(ious)


class YTVISeval(YTVISevalMixin, COCOeval):

    sort_inds_by_scores_in_iou = True


class VideoDemoF1Eval(YTVISevalMixin, CGF1Eval):

    sort_inds_by_scores_in_iou = False


class YTVISResultsWriter:


    def __init__(
        self,
        dump_file: str,
        postprocessor,
        gather_pred_via_filesys=False,
        pred_file_evaluators: Optional[List] = None,
        save_per_frame_scores: bool = False,
        write_eval_metrics_file: bool = True,
        eval_metrics_file_suffix: str = ".sam3_eval_metrics",
    ):
        self.dump_file = dump_file
        self.dump = []
        self.postprocessor = postprocessor
        self.gather_pred_via_filesys = gather_pred_via_filesys
        if dist.is_main_process():
            dirname = os.path.dirname(self.dump_file)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
                logging.info(f"Creating folder: {dirname}")


        self.pred_file_evaluators = pred_file_evaluators or []
        self.save_per_frame_scores = save_per_frame_scores


        self.write_eval_metrics_file = write_eval_metrics_file
        if self.write_eval_metrics_file:
            self.eval_metrics_file = self.dump_file + eval_metrics_file_suffix
            os.makedirs(os.path.dirname(self.eval_metrics_file), exist_ok=True)

    def _dump_vid_preds(self, results):
        dumped_results = copy.deepcopy(results)
        self.dump.extend(dumped_results)

    def prepare(self, predictions):
        ytvis_results = []
        for video_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            for k in ["boxes", "scores", "labels"]:
                assert k in prediction, (
                    f"Expected predictions to have `{k}` key, available keys are {prediction.keys()}"
                )
            if self.save_per_frame_scores:
                assert "per_frame_scores" in prediction, (
                    f"Expected predictions to have `per_frame_scores` key, available keys are {prediction.keys()}"
                )
            assert xor("masks" in prediction, "masks_rle" in prediction), (
                f"Expected predictions to have either `masks` key or `masks_rle` key, available keys are {prediction.keys()}"
            )

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            if "masks" in prediction:
                masks = prediction["masks"].squeeze(2)
                assert masks.ndim == 4, (
                    "Expected masks to be of shape(N_preds,T_frames,H,W)"
                )

                areas = [mask.flatten(1).sum(1).tolist() for mask in masks]
                rles = [rle_encode(masklet) for masklet in masks]


                del masks
                del prediction["masks"]
            elif "masks_rle" in prediction:
                rles = prediction.pop("masks_rle")
                areas = [
                    [0 if rle is None else rle.pop("area") for rle in rles_per_obj]
                    for rles_per_obj in rles
                ]
            else:
                raise ValueError(
                    "Expected either `masks` or `masks_rle` key in the predictions."
                )

            new_results = [
                {
                    "video_id": video_id,
                    "category_id": track_label,
                    "bboxes": track_boxes,
                    "score": track_score,
                    "segmentations": track_masks,
                    "areas": track_areas,
                }
                for (
                    track_boxes,
                    track_masks,
                    track_areas,
                    track_score,
                    track_label,
                ) in zip(boxes, rles, areas, scores, labels)
            ]

            if self.save_per_frame_scores:
                per_frame_scores = prediction["per_frame_scores"].tolist()
                for res, track_per_frame_scores in zip(new_results, per_frame_scores):
                    res["per_frame_scores"] = track_per_frame_scores

            ytvis_results.extend(new_results)

        return ytvis_results

    def set_sync_device(self, device: torch.device):
        self._sync_device = device

    def update(self, *args, **kwargs):
        predictions = self.postprocessor.process_results(*args, **kwargs)
        results = self.prepare(predictions)
        self._dump_vid_preds(results)

    def _dump_preds(self):
        if not dist.is_main_process():
            self.dump = []
            gc.collect()
            return
        dumped_file = Path(self.dump_file)
        logging.info(f"YTVIS evaluator: Dumping predictions to {dumped_file}")
        with g_pathmgr.open(str(dumped_file), "w") as f:
            json.dump(self.dump, f)
        self.dump = []
        gc.collect()
        return str(dumped_file)

    def synchronize_between_processes(self):
        logging.info("YT-VIS evaluator: Synchronizing between processes")
        dump_dict = self._dedup_pre_gather(self.dump)
        if self.gather_pred_via_filesys:
            dump_dict_all_gpus = dist.gather_to_rank_0_via_filesys(dump_dict)
        else:
            dump_dict_all_gpus = dist.all_gather(dump_dict, force_cpu=True)
        self.dump = self._dedup_post_gather(dump_dict_all_gpus)
        logging.info(f"Gathered all {len(self.dump)} predictions")

    def _dedup_pre_gather(self, predictions):

        prediction_dict = defaultdict(list)
        for p in predictions:
            prediction_dict[(p["video_id"], p["category_id"])].append(p)
        return prediction_dict

    def _dedup_post_gather(self, list_of_prediction_dict):

        dedup_prediction_dict = {}
        duplication_keys = []
        for prediction_dict in list_of_prediction_dict:
            for k, v in prediction_dict.items():
                if k not in dedup_prediction_dict:
                    dedup_prediction_dict[k] = v
                else:
                    duplication_keys.append(k)

        logging.info(
            f"skipped {len(duplication_keys)} duplicated predictions in YTVISResultsWriter "
            f"with the following (video_id, category_id) tuples: {duplication_keys}"
        )
        dedup_predictions = sum(dedup_prediction_dict.values(), [])
        return dedup_predictions

    def compute_synced(
        self,
    ):
        self.synchronize_between_processes()
        dumped_file = self._dump_preds()
        if not dist.is_main_process():
            return {"": 0.0}


        meters = {}
        all_video_np_level_results = defaultdict(dict)
        for evaluator in self.pred_file_evaluators:
            gc.collect()
            results, video_np_level_results = evaluator.evaluate(dumped_file)
            meters.update(results)
            for (video_id, category_id), res in video_np_level_results.items():
                all_video_np_level_results[(video_id, category_id)].update(res)

        gc.collect()
        if self.write_eval_metrics_file:


            video_np_level_metrics = [
                {"video_id": video_id, "category_id": category_id, **res}
                for (video_id, category_id), res in all_video_np_level_results.items()
            ]
            eval_metrics = {
                "dataset_level_metrics": meters,
                "video_np_level_metrics": video_np_level_metrics,
            }
            with g_pathmgr.open(self.eval_metrics_file, "w") as f:
                json.dump(eval_metrics, f)
            logging.info(
                f"YTVIS evaluator: Dumped evaluation metrics to {self.eval_metrics_file}"
            )

        if len(meters) == 0:
            meters = {"": 0.0}
        return meters

    def compute(self):
        return {"": 0.0}

    def reset(self, *args, **kwargs):
        self.dump = []
