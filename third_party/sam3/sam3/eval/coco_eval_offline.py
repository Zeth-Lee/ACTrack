# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import logging
from collections import defaultdict

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sam3.train.utils.distributed import is_main_process

try:
    from tidecv import datasets, TIDE

    HAS_TIDE = True
except ImportError:
    HAS_TIDE = False
    print("WARNING: TIDE not installed. Detailed analysis will not be available.")


COCO_METRICS = [
    "AP",
    "AP_50",
    "AP_75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_maxDets@1",
    "AR_maxDets@10",
    "AR_maxDets@100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def convert_to_xywh(boxes):

    xmin, ymin, xmax, ymax = boxes.unbind(-1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=-1)


class HeapElement:


    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val["score"] < other.val["score"]


class COCOevalCustom(COCOeval):


    def __init__(
        self, cocoGt=None, cocoDt=None, iouType="segm", dt_only_positive=False
    ):
        super().__init__(cocoGt, cocoDt, iouType)
        self.dt_only_positive = dt_only_positive

    def _prepare(self):


        def _toMask(anns, coco):

            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

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


        if p.iouType == "segm":
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)

        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)

        _gts_cat_ids = defaultdict(set)
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
            _gts_cat_ids[gt["image_id"]].add(gt["category_id"])


        for dt in dts:
            if (
                self.dt_only_positive
                and dt["category_id"] not in _gts_cat_ids[dt["image_id"]]
            ):
                continue
            self._dts[dt["image_id"], dt["category_id"]].append(dt)

        self.evalImgs = defaultdict(list)
        self.eval = {}


class CocoEvaluatorOfflineWithPredFileEvaluators:
    def __init__(
        self,
        gt_path,
        tide: bool = True,
        iou_type: str = "bbox",
        positive_split=False,
    ):
        self.gt_path = gt_path
        self.tide_enabled = HAS_TIDE and tide
        self.positive_split = positive_split
        self.iou_type = iou_type

    def evaluate(self, dumped_file):
        if not is_main_process():
            return {}

        logging.info("OfflineCoco evaluator: Loading groundtruth")
        self.gt = COCO(self.gt_path)


        logging.info("Coco evaluator: Creating the result file")
        cocoDt = self.gt.loadRes(str(dumped_file))


        logging.info("Coco evaluator: Running evaluation")
        coco_eval = COCOevalCustom(
            self.gt, cocoDt, iouType=self.iou_type, dt_only_positive=self.positive_split
        )
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        outs = {}
        for i, value in enumerate(coco_eval.stats):
            outs[f"coco_eval_{self.iou_type}_{COCO_METRICS[i]}"] = value

        if self.tide_enabled:
            logging.info("Coco evaluator: Loading TIDE")
            self.tide_gt = datasets.COCO(self.gt_path)
            self.tide = TIDE(mode="mask" if self.iou_type == "segm" else "bbox")


            logging.info("Coco evaluator: Running TIDE")
            self.tide.evaluate(
                self.tide_gt, datasets.COCOResult(str(dumped_file)), name="coco_eval"
            )
            self.tide.summarize()
            for k, v in self.tide.get_main_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

            for k, v in self.tide.get_special_errors()["coco_eval"].items():
                outs[f"coco_eval_{self.iou_type}_TIDE_{k}"] = v

        return outs
