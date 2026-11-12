# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import copy
import gc
import heapq
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pycocotools.mask as mask_utils
import torch
from iopath.common.file_io import g_pathmgr
from sam3.eval.coco_eval_offline import convert_to_xywh
from sam3.train.masks_ops import rle_encode
from sam3.train.utils.distributed import (
    all_gather,
    gather_to_rank_0_via_filesys,
    get_rank,
    is_main_process,
)


class HeapElement:


    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val["score"] < other.val["score"]


class PredictionDumper:


    def __init__(
        self,
        dump_dir: str,
        postprocessor,
        maxdets: int,
        iou_type: str,
        gather_pred_via_filesys: bool = False,
        merge_predictions: bool = False,
        pred_file_evaluators: Optional[Any] = None,
    ):

        self.iou_type = iou_type
        self.maxdets = maxdets
        self.dump_dir = dump_dir
        self.postprocessor = postprocessor
        self.gather_pred_via_filesys = gather_pred_via_filesys
        self.merge_predictions = merge_predictions
        self.pred_file_evaluators = pred_file_evaluators
        if self.pred_file_evaluators is not None:
            assert merge_predictions, (
                "merge_predictions must be True if pred_file_evaluators are provided"
            )
        assert self.dump_dir is not None, "dump_dir must be provided"

        if is_main_process():
            os.makedirs(self.dump_dir, exist_ok=True)
            logging.info(f"Created prediction dump directory: {self.dump_dir}")


        self.reset()

    def update(self, *args, **kwargs):

        predictions = self.postprocessor.process_results(*args, **kwargs)
        results = self.prepare(predictions, self.iou_type)
        self._dump(results)

    def _dump(self, results):

        dumped_results = copy.deepcopy(results)
        for r in dumped_results:
            if "bbox" in r:
                r["bbox"] = [round(coord, 5) for coord in r["bbox"]]
            r["score"] = round(r["score"], 5)
        self.dump.extend(dumped_results)

    def synchronize_between_processes(self):

        logging.info("Prediction Dumper: Synchronizing between processes")

        if not self.merge_predictions:
            dumped_file = (
                Path(self.dump_dir)
                / f"coco_predictions_{self.iou_type}_{get_rank()}.json"
            )
            logging.info(
                f"Prediction Dumper: Dumping local predictions to {dumped_file}"
            )
            with g_pathmgr.open(str(dumped_file), "w") as f:
                json.dump(self.dump, f)
        else:
            self.dump = self.gather_and_merge_predictions()
            dumped_file = Path(self.dump_dir) / f"coco_predictions_{self.iou_type}.json"
            if is_main_process():
                logging.info(
                    f"Prediction Dumper: Dumping merged predictions to {dumped_file}"
                )
                with g_pathmgr.open(str(dumped_file), "w") as f:
                    json.dump(self.dump, f)

        self.reset()
        return dumped_file

    def gather_and_merge_predictions(self):

        logging.info("Prediction Dumper: Gathering predictions from all processes")
        gc.collect()

        if self.gather_pred_via_filesys:
            dump = gather_to_rank_0_via_filesys(self.dump)
        else:
            dump = all_gather(self.dump, force_cpu=True)


        preds_by_image = defaultdict(list)
        seen_img_cat = set()

        for cur_dump in dump:
            cur_seen_img_cat = set()
            for p in cur_dump:
                image_id = p["image_id"]
                cat_id = p["category_id"]


                if (image_id, cat_id) in seen_img_cat:
                    continue

                cur_seen_img_cat.add((image_id, cat_id))


                if len(preds_by_image[image_id]) < self.maxdets:
                    heapq.heappush(preds_by_image[image_id], HeapElement(p))
                else:
                    heapq.heappushpop(preds_by_image[image_id], HeapElement(p))

            seen_img_cat.update(cur_seen_img_cat)


        merged_dump = sum(
            [[h.val for h in cur_preds] for cur_preds in preds_by_image.values()], []
        )

        return merged_dump

    def compute_synced(self):

        dumped_file = self.synchronize_between_processes()
        if not is_main_process():
            return {"": 0.0}

        meters = {}
        if self.pred_file_evaluators is not None:
            for evaluator in self.pred_file_evaluators:
                results = evaluator.evaluate(dumped_file)
                meters.update(results)

        if len(meters) == 0:
            meters = {"": 0.0}
        return meters

    def compute(self):

        return {"": 0.0}

    def reset(self):

        self.dump = []

    def prepare(self, predictions, iou_type):

        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        else:
            raise ValueError(f"Unknown iou type: {iou_type}")

    def prepare_for_coco_detection(self, predictions):

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

        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            boxes = None
            if "boxes" in prediction:
                boxes = prediction["boxes"]
                boxes = convert_to_xywh(boxes).tolist()
                assert len(boxes) == len(scores)

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
                if boxes is not None:
                    payload["bbox"] = boxes[k]

                coco_results.append(payload)

        return coco_results
