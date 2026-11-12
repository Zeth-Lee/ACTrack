# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import contextlib
import copy
import json
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pycocotools.mask as mask_utils
import torch
from iopath.common.file_io import g_pathmgr
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sam3.train.masks_ops import rle_encode
from sam3.train.utils.distributed import (
    all_gather,
    gather_to_rank_0_via_filesys,
    get_rank,
    is_main_process,
)

RARITY_BUCKETS = {0: "frequent", 1: "common", 2: "medium", 3: "rare"}


class CocoEvaluator:
    def __init__(
        self,
        coco_gt,
        iou_types: List[str],
        useCats: bool,
        dump_dir: Optional[str],
        postprocessor,
        average_by_rarity=False,
        metrics_dump_dir: Optional[str] = None,
        gather_pred_via_filesys=False,
        use_normalized_areas=True,
        maxdets=[1, 10, 100],
        exhaustive_only=False,
        all_exhaustive_only=True,
    ):


        self.coco_gts = [coco_gt] if not isinstance(coco_gt, list) else coco_gt
        assert len(maxdets) == 3, f"expecting 3 detection threshold, got {len(maxdets)}"

        self.use_normalized_areas = use_normalized_areas
        self.iou_types = iou_types
        self.useCats = useCats
        self.maxdets = maxdets
        self.dump = None
        self.dump_dir = dump_dir
        if self.dump_dir is not None:
            self.dump = []
            if is_main_process():
                if not os.path.exists(self.dump_dir):
                    os.makedirs(self.dump_dir, exist_ok=True)
                    logging.info(f"Create the folder: {dump_dir}")

        self.initialized = False


        self.gather_pred_via_filesys = gather_pred_via_filesys
        self.use_self_evaluate = True
        self.postprocessor = postprocessor
        self.average_by_rarity = average_by_rarity
        self.exhaustive_only = exhaustive_only
        self.all_exhaustive_only = all_exhaustive_only
        self.metrics_dump_dir = metrics_dump_dir
        if self.metrics_dump_dir is not None:
            if is_main_process():
                if not os.path.exists(self.metrics_dump_dir):
                    os.makedirs(self.metrics_dump_dir, exist_ok=True)
                    logging.info(f"Create the folder: {metrics_dump_dir}")

    def _lazy_init(self, coco_cls=COCO):
        if self.initialized:
            return

        self.initialized = True

        self.coco_gts = [
            coco_cls(g_pathmgr.get_local_path(gt)) if isinstance(gt, str) else gt
            for gt in self.coco_gts
        ]

        self.reset()

        self.eval_img_ids = None

        if self.exhaustive_only:
            exclude_img_ids = set()

            if self.all_exhaustive_only:
                for coco_gt in self.coco_gts[1:]:
                    exclude_img_ids = exclude_img_ids.union(
                        {
                            img["id"]
                            for img in coco_gt.dataset["images"]
                            if not img["is_instance_exhaustive"]
                        }
                    )

            self.eval_img_ids = [
                img["id"]
                for img in self.coco_gts[0].dataset["images"]
                if (img["is_instance_exhaustive"] and img["id"] not in exclude_img_ids)
            ]

        self.rarity_buckets = None
        if self.average_by_rarity:
            self.rarity_buckets = defaultdict(list)
            eval_img_ids_set = (
                set(self.eval_img_ids) if self.eval_img_ids is not None else None
            )
            for img in self.coco_gts[0].dataset["images"]:
                if self.eval_img_ids is not None and img["id"] not in eval_img_ids_set:
                    continue
                self.rarity_buckets[img["rarity"]].append(img["id"])
            print("Rarity buckets sizes:")
            for k, v in self.rarity_buckets.items():
                print(f"{k}: {len(v)}")

    def set_sync_device(self, device: torch.device) -> Any:
        self._sync_device = device

    def _evaluate(self, *args, **kwargs):
        return evaluate(*args, **kwargs)

    def _loadRes(self, *args, **kwargs):
        return loadRes(*args, **kwargs)

    def update(self, *args, **kwargs):
        self._lazy_init()
        predictions = self.postprocessor.process_results(*args, **kwargs)

        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            self._dump(results)

            assert len(self.coco_gts) == len(self.coco_evals)
            all_scorings = []
            for cur_coco_gt, cur_coco_eval in zip(self.coco_gts, self.coco_evals):

                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull):
                        coco_dt = (
                            self._loadRes(cur_coco_gt, results) if results else COCO()
                        )

                coco_eval = cur_coco_eval[iou_type]

                coco_eval.cocoDt = coco_dt
                coco_eval.params.imgIds = list(img_ids)
                coco_eval.params.useCats = self.useCats
                coco_eval.params.maxDets = self.maxdets
                img_ids, eval_imgs = self._evaluate(coco_eval, self.use_self_evaluate)
                all_scorings.append(eval_imgs)

            selected = self.select_best_scoring(all_scorings)
            self.eval_imgs[iou_type].append(selected)

    def select_best_scoring(self, scorings):


        if len(scorings) == 1:
            return scorings[0]


        raise RuntimeError("Not implemented")

    def _dump(self, results):
        if self.dump is not None:
            dumped_results = copy.deepcopy(results)
            for r in dumped_results:
                if "bbox" not in self.iou_types and "bbox" in r:
                    del r["bbox"]
                elif "bbox" in r:
                    r["bbox"] = [round(coord, 5) for coord in r["bbox"]]
                r["score"] = round(r["score"], 5)
            self.dump.extend(dumped_results)

    def synchronize_between_processes(self):
        self._lazy_init()
        logging.info("Coco evaluator: Synchronizing between processes")
        for iou_type in self.iou_types:
            if len(self.eval_imgs[iou_type]) > 0:
                self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            else:
                num_areas = len(self.coco_evals[0][iou_type].params.areaRng)

                assert not self.useCats
                self.eval_imgs[iou_type] = np.empty((1, num_areas, 0))
            create_common_coco_eval(
                self.coco_evals[0][iou_type],
                self.img_ids,
                self.eval_imgs[iou_type],
                use_self_evaluate=self.use_self_evaluate,
                gather_pred_via_filesys=self.gather_pred_via_filesys,
                metrics_dump_dir=self.metrics_dump_dir,
            )
        if self.dump is not None:
            dumped_file = Path(self.dump_dir) / f"coco_predictions_{get_rank()}.json"
            logging.info(f"COCO evaluator: Dumping local predictions to {dumped_file}")
            with g_pathmgr.open(str(dumped_file), "w") as f:
                json.dump(self.dump, f)


    def accumulate(self, imgIds=None):
        self._lazy_init()
        logging.info(
            f"Coco evaluator: Accumulating on {len(imgIds) if imgIds is not None else 'all'} images"
        )
        if not is_main_process():
            return

        if imgIds is None:
            for coco_eval in self.coco_evals[0].values():
                accumulate(coco_eval, use_self_eval=self.use_self_evaluate)

        if imgIds is not None:
            imgIds = set(imgIds)
            for coco_eval in self.coco_evals[0].values():
                p = coco_eval.params
                id_mask = np.array([(i in imgIds) for i in p.imgIds], dtype=bool)
                old_img_ids = p.imgIds
                coco_eval.params.imgIds = np.asarray(p.imgIds)[id_mask]
                old_img_evals = coco_eval.evalImgs
                catIds = p.catIds if p.useCats else [-1]
                coco_eval.evalImgs = list(
                    np.asarray(coco_eval.evalImgs)
                    .reshape(len(catIds), len(p.areaRng), len(old_img_ids))[
                        ..., id_mask
                    ]
                    .flatten()
                )
                accumulate(coco_eval, use_self_eval=self.use_self_evaluate)
                coco_eval.evalImgs = old_img_evals
                coco_eval.params.imgIds = old_img_ids

    def summarize(self):
        self._lazy_init()
        logging.info("Coco evaluator: Summarizing")
        if not is_main_process():
            return {}

        outs = {}
        if self.rarity_buckets is None:
            self.accumulate(self.eval_img_ids)
            for iou_type, coco_eval in self.coco_evals[0].items():
                print("IoU metric: {}".format(iou_type))
                summarize(coco_eval)

            if "bbox" in self.coco_evals[0]:
                for key, value in zip(*self.coco_evals[0]["bbox"].stats):
                    outs[f"coco_eval_bbox_{key}"] = value
            if "segm" in self.coco_evals[0]:
                for key, value in zip(*self.coco_evals[0]["segm"].stats):
                    outs[f"coco_eval_masks_{key}"] = value
        else:
            total_stats = {}
            all_keys = {}
            for bucket, img_list in self.rarity_buckets.items():
                self.accumulate(imgIds=img_list)
                bucket_name = RARITY_BUCKETS[bucket]
                for iou_type, coco_eval in self.coco_evals[0].items():
                    print(f"IoU metric: {iou_type}. Rarity bucket: {bucket_name}")
                    summarize(coco_eval)

                if "bbox" in self.coco_evals[0]:
                    if "bbox" not in total_stats:
                        total_stats["bbox"] = np.zeros_like(
                            self.coco_evals[0]["bbox"].stats[1]
                        )
                        all_keys["bbox"] = self.coco_evals[0]["bbox"].stats[0]
                    total_stats["bbox"] += self.coco_evals[0]["bbox"].stats[1]
                    for key, value in zip(*self.coco_evals[0]["bbox"].stats):
                        outs[f"coco_eval_bbox_{bucket_name}_{key}"] = value
                if "segm" in self.coco_evals[0]:
                    if "segm" not in total_stats:
                        total_stats["segm"] = np.zeros_like(
                            self.coco_evals[0]["segm"].stats[1]
                        )
                        all_keys["segm"] = self.coco_evals[0]["segm"].stats[0]
                    total_stats["segm"] += self.coco_evals[0]["segm"].stats[1]
                    for key, value in zip(*self.coco_evals[0]["segm"].stats):
                        outs[f"coco_eval_masks_{bucket_name}_{key}"] = value

            if "bbox" in total_stats:
                total_stats["bbox"] /= len(self.rarity_buckets)
                for key, value in zip(all_keys["bbox"], total_stats["bbox"]):
                    outs[f"coco_eval_bbox_{key}"] = value
            if "segm" in total_stats:
                total_stats["segm"] /= len(self.rarity_buckets)
                for key, value in zip(all_keys["segm"], total_stats["segm"]):
                    outs[f"coco_eval_masks_{key}"] = value


        return outs

    def compute_synced(self):
        self._lazy_init()
        self.synchronize_between_processes()
        return self.summarize()

    def compute(self):
        self._lazy_init()
        return {"": 0.0}

    def reset(self, cocoeval_cls=COCOeval):
        self.coco_evals = [{} for _ in range(len(self.coco_gts))]
        for i, coco_gt in enumerate(self.coco_gts):
            for iou_type in self.iou_types:
                self.coco_evals[i][iou_type] = cocoeval_cls(coco_gt, iouType=iou_type)
                self.coco_evals[i][iou_type].params.useCats = self.useCats
                self.coco_evals[i][iou_type].params.maxDets = self.maxdets
                if self.use_normalized_areas:
                    self.coco_evals[i][iou_type].params.areaRng = [
                        [0, 1e5],
                        [0, 0.001],
                        [0.001, 0.01],
                        [0.01, 0.1],
                        [0.1, 0.5],
                        [0.5, 0.95],
                        [0.95, 1e5],
                    ]
                    self.coco_evals[i][iou_type].params.areaRngLbl = [
                        "all",
                        "tiny",
                        "small",
                        "medium",
                        "large",
                        "huge",
                        "whole_image",
                    ]

        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
        if self.dump is not None:
            self.dump = []

    def write(self, stats):
        self._lazy_init()

        if "bbox" in self.coco_evals[0]:
            stats["coco_eval_bbox"] = self.coco_evals[0]["bbox"].stats.tolist()
        if "segm" in self.coco_evals[0]:
            stats["coco_eval_masks"] = self.coco_evals[0]["segm"].stats.tolist()
        return stats

    def prepare(self, predictions, iou_type):
        self._lazy_init()
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        self._lazy_init()
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    @torch.no_grad()
    def prepare_for_coco_segmentation(self, predictions):
        self._lazy_init()
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            boundaries, dilated_boundaries = None, None
            if "boundaries" in prediction:
                boundaries = prediction["boundaries"]
                dilated_boundaries = prediction["dilated_boundaries"]
                assert dilated_boundaries is not None
                assert len(scores) == len(boundaries)

            if "masks_rle" in prediction:
                rles = prediction["masks_rle"]
                areas = []
                for rle in rles:
                    cur_area = mask_utils.area(rle)
                    h, w = rle["size"]
                    areas.append(cur_area / (h * w))
            else:
                masks = prediction["masks"]

                masks = masks > 0.5
                h, w = masks.shape[-2:]

                areas = masks.flatten(1).sum(1) / (h * w)
                areas = areas.tolist()

                rles = rle_encode(masks.squeeze(1))


                del masks
                del prediction["masks"]

            assert len(areas) == len(rles) == len(scores)
            for k, rle in enumerate(rles):
                payload = {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                    "area": areas[k],
                }
                if boundaries is not None:
                    payload["boundary"] = boundaries[k]
                    payload["dilated_boundary"] = dilated_boundaries[k]

                coco_results.append(payload)

        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        self._lazy_init()
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)


def merge(img_ids, eval_imgs, gather_pred_via_filesys=False):
    if gather_pred_via_filesys:


        all_img_ids = gather_to_rank_0_via_filesys(img_ids)
        all_eval_imgs = gather_to_rank_0_via_filesys(eval_imgs)
    else:
        all_img_ids = all_gather(img_ids, force_cpu=True)
        all_eval_imgs = all_gather(eval_imgs, force_cpu=True)
    if not is_main_process():
        return None, None

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)


    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(
    coco_eval,
    img_ids,
    eval_imgs,
    use_self_evaluate,
    gather_pred_via_filesys=False,
    metrics_dump_dir=None,
):
    img_ids, eval_imgs = merge(img_ids, eval_imgs, gather_pred_via_filesys)
    if not is_main_process():
        return
    if metrics_dump_dir is not None:
        dumped_file = (
            Path(metrics_dump_dir) / f"coco_eval_img_metrics_{get_rank()}.json"
        )
        logging.info(f"COCO evaluator: Dumping local predictions to {dumped_file}")
        with g_pathmgr.open(str(dumped_file), "w") as f:
            json.dump(eval_imgs.squeeze(), f, default=lambda x: x.tolist())
    img_ids = list(img_ids)


    missing_img_ids = set(coco_eval.cocoGt.getImgIds()) - set(img_ids)
    if len(missing_img_ids) > 0:
        print(f"WARNING: {len(missing_img_ids)} images were not predicted!")
        coco_eval.cocoDt = COCO()
        coco_eval.params.imgIds = list(missing_img_ids)
        new_img_ids, new_eval_imgs = evaluate(coco_eval, use_self_evaluate)
        img_ids.extend(new_img_ids)
        eval_imgs = np.concatenate((eval_imgs, new_eval_imgs), axis=2)

    eval_imgs = list(eval_imgs.flatten())
    assert len(img_ids) == len(coco_eval.cocoGt.getImgIds())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def segmentation_prepare(self):

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


def evaluate(self, use_self_evaluate):


    p = self.params

    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print(
            "useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType)
        )

    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()

    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds
    }

    maxDet = p.maxDets[-1]
    if use_self_evaluate:
        evalImgs = [
            self.evaluateImg(imgId, catId, areaRng, maxDet)
            for catId in catIds
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]

        evalImgs = np.asarray(evalImgs).reshape(
            len(catIds), len(p.areaRng), len(p.imgIds)
        )
        return p.imgIds, evalImgs


def loadRes(self, resFile):

    res = COCO()
    res.dataset["images"] = [img for img in self.dataset["images"]]

    if type(resFile) == str:
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, "results in not an array of objects"
    annsImgIds = [ann["image_id"] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), (
        "Results do not correspond to current coco set"
    )
    if "caption" in anns[0]:
        imgIds = set([img["id"] for img in res.dataset["images"]]) & set(
            [ann["image_id"] for ann in anns]
        )
        res.dataset["images"] = [
            img for img in res.dataset["images"] if img["id"] in imgIds
        ]
        for id, ann in enumerate(anns):
            ann["id"] = id + 1
    elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            bb = ann["bbox"]
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if "segmentation" not in ann:
                ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann["area"] = bb[2] * bb[3]
            ann["id"] = id + 1
            ann["iscrowd"] = 0
    elif "segmentation" in anns[0]:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):


            ann["id"] = id + 1
            ann["iscrowd"] = 0
    elif "keypoints" in anns[0]:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            s = ann["keypoints"]
            x = s[0::3]
            y = s[1::3]
            x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann["area"] = (x1 - x0) * (y1 - y0)
            ann["id"] = id + 1
            ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]

    res.dataset["annotations"] = anns
    res.createIndex()
    return res


def summarize(self):


    def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
        p = self.params
        iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
        titleStr = "Average Precision" if ap == 1 else "Average Recall"
        typeStr = "(AP)" if ap == 1 else "(AR)"
        iouStr = (
            "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
            if iouThr is None
            else "{:0.2f}".format(iouThr)
        )

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:

            s = self.eval["precision"]

            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:

            s = self.eval["recall"]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s

    def _summarizeDets():
        nb_results = 6 + (len(self.params.areaRng) - 1) * 2
        assert len(self.params.areaRng) == len(self.params.areaRngLbl)
        stats = np.zeros((nb_results,))
        keys = ["AP", "AP_50", "AP_75"]
        stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
        stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
        cur_id = 3
        for area in self.params.areaRngLbl[1:]:
            stats[cur_id] = _summarize(1, areaRng=area, maxDets=self.params.maxDets[2])
            cur_id += 1
            keys.append(f"AP_{area}")
        stats[cur_id] = _summarize(0, maxDets=self.params.maxDets[0])
        cur_id += 1
        stats[cur_id] = _summarize(0, maxDets=self.params.maxDets[1])
        cur_id += 1
        stats[cur_id] = _summarize(0, maxDets=self.params.maxDets[2])
        cur_id += 1
        keys += ["AR", "AR_50", "AR_75"]

        for area in self.params.areaRngLbl[1:]:
            stats[cur_id] = _summarize(0, areaRng=area, maxDets=self.params.maxDets[2])
            cur_id += 1
            keys.append(f"AR_{area}")
        assert len(stats) == len(keys)
        return keys, stats

    if not self.eval:
        raise Exception("Please run accumulate() first")
    self.stats = _summarizeDets()


def accumulate(self, use_self_eval=False):

    if use_self_eval:
        self.accumulate()
        return


