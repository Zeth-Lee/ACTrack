# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import contextlib
import copy
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pycocotools.mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


@dataclass
class Metric:
    name: str


    image_level: bool


    iou_threshold: Union[float, None]


CGF1_METRICS = [
    Metric(name="cgF1", image_level=False, iou_threshold=None),
    Metric(name="precision", image_level=False, iou_threshold=None),
    Metric(name="recall", image_level=False, iou_threshold=None),
    Metric(name="F1", image_level=False, iou_threshold=None),
    Metric(name="positive_macro_F1", image_level=False, iou_threshold=None),
    Metric(name="positive_micro_F1", image_level=False, iou_threshold=None),
    Metric(name="positive_micro_precision", image_level=False, iou_threshold=None),
    Metric(name="IL_precision", image_level=True, iou_threshold=None),
    Metric(name="IL_recall", image_level=True, iou_threshold=None),
    Metric(name="IL_F1", image_level=True, iou_threshold=None),
    Metric(name="IL_FPR", image_level=True, iou_threshold=None),
    Metric(name="IL_MCC", image_level=True, iou_threshold=None),
    Metric(name="cgF1", image_level=False, iou_threshold=0.5),
    Metric(name="precision", image_level=False, iou_threshold=0.5),
    Metric(name="recall", image_level=False, iou_threshold=0.5),
    Metric(name="F1", image_level=False, iou_threshold=0.5),
    Metric(name="positive_macro_F1", image_level=False, iou_threshold=0.5),
    Metric(name="positive_micro_F1", image_level=False, iou_threshold=0.5),
    Metric(name="positive_micro_precision", image_level=False, iou_threshold=0.5),
    Metric(name="cgF1", image_level=False, iou_threshold=0.75),
    Metric(name="precision", image_level=False, iou_threshold=0.75),
    Metric(name="recall", image_level=False, iou_threshold=0.75),
    Metric(name="F1", image_level=False, iou_threshold=0.75),
    Metric(name="positive_macro_F1", image_level=False, iou_threshold=0.75),
    Metric(name="positive_micro_F1", image_level=False, iou_threshold=0.75),
    Metric(name="positive_micro_precision", image_level=False, iou_threshold=0.75),
]


class COCOCustom(COCO):


    def createIndex(self):

        print("creating index...")
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                imgToAnns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:

            if self.imgs:
                imgs = self.imgs
            else:
                for img in self.dataset["images"]:
                    imgs[img["id"]] = img


        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])

        print("index created!")


        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def loadRes(self, resFile):

        res = COCOCustom()
        res.dataset["info"] = copy.deepcopy(self.dataset.get("info", {}))


        res.dataset["images"] = self.dataset["images"]


        print("Loading and preparing results...")
        tic = time.time()
        if type(resFile) == str:
            with open(resFile) as f:
                anns = json.load(f)
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, "results in not an array of objects"
        annsImgIds = [ann["image_id"] for ann in anns]

        if not hasattr(self, "img_id_set"):
            self.img_id_set = set(self.getImgIds())
        assert set(annsImgIds).issubset(self.img_id_set), (
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
                if not "segmentation" in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "segmentation" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):

                ann["area"] = maskUtils.area(ann["segmentation"])
                if not "bbox" in ann:
                    ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
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
        print("DONE (t={:0.2f}s)".format(time.time() - tic))

        res.dataset["annotations"] = anns

        res.imgs = self.imgs

        res.createIndex()
        return res


class CGF1Eval(COCOeval):


    def __init__(
        self,
        coco_gt=None,
        coco_dt=None,
        iouType="segm",
        threshold=0.5,
    ):

        super().__init__(coco_gt, coco_dt, iouType)
        self.threshold = threshold

        self.params.useCats = False
        self.params.areaRng = [[0**2, 1e5**2]]
        self.params.areaRngLbl = ["all"]
        self.params.maxDets = [1000000]

    def computeIoU(self, imgId, catId):

        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []

        if p.iouType == "segm":
            g = [g["segmentation"] for g in gt]
            d = [d["segmentation"] for d in dt]
        elif p.iouType == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise Exception("unknown iouType for iou computation")


        iscrowd = [int(o["iscrowd"]) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):

        p = self.params
        assert not p.useCats, "This evaluator does not support per-category evaluation."
        assert catId == -1
        all_gts = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
        keep_gt = np.array([not g["ignore"] for g in all_gts], dtype=bool)
        gt = [g for g in all_gts if not g["ignore"]]
        all_dts = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        keep_dt = np.array([d["score"] >= self.threshold for d in all_dts], dtype=bool)
        dt = [d for d in all_dts if d["score"] >= self.threshold]
        if len(gt) == 0 and len(dt) == 0:


            return {
                "image_id": imgId,
                "IL_TP": 0,
                "IL_TN": 1,
                "IL_FP": 0,
                "IL_FN": 0,
                "num_dt": len(dt),
            }

        if len(gt) > 0 and len(dt) == 0:

            return {
                "image_id": imgId,
                "IL_TP": 0,
                "IL_TN": 0,
                "IL_FP": 0,
                "IL_FN": 1,
                "TPs": np.zeros((len(p.iouThrs),), dtype=np.int64),
                "FPs": np.zeros((len(p.iouThrs),), dtype=np.int64),
                "FNs": np.ones((len(p.iouThrs),), dtype=np.int64) * len(gt),
                "local_F1s": np.zeros((len(p.iouThrs),), dtype=np.int64),
                "local_positive_F1s": np.zeros((len(p.iouThrs),), dtype=np.int64),
                "num_dt": len(dt),
            }


        ious = self.ious[(imgId, catId)]


        if len(ious) == 0:
            ious = np.zeros((len(dt), len(gt)))
        else:
            ious = ious[keep_dt, :][:, keep_gt]
        assert ious.shape == (len(dt), len(gt))

        matched_dt, matched_gt = linear_sum_assignment(-ious)

        match_scores = ious[matched_dt, matched_gt]

        TPs, FPs, FNs = [], [], []
        IL_perfect = []
        for thresh in p.iouThrs:
            TP = (match_scores >= thresh).sum()
            FP = len(dt) - TP
            FN = len(gt) - TP
            assert FP >= 0 and FN >= 0, (
                f"FP: {FP}, FN: {FN}, TP: {TP}, match_scores: {match_scores}, len(dt): {len(dt)}, len(gt): {len(gt)}, ious: {ious}"
            )
            TPs.append(TP)
            FPs.append(FP)
            FNs.append(FN)

            if FP == FN and FP == 0:
                IL_perfect.append(1)
            else:
                IL_perfect.append(0)

        TPs = np.array(TPs, dtype=np.int64)
        FPs = np.array(FPs, dtype=np.int64)
        FNs = np.array(FNs, dtype=np.int64)
        IL_perfect = np.array(IL_perfect, dtype=np.int64)


        precision = TPs / (TPs + FPs + 1e-4)
        assert np.all(precision <= 1)
        recall = TPs / (TPs + FNs + 1e-4)
        assert np.all(recall <= 1)
        F1 = 2 * precision * recall / (precision + recall + 1e-4)

        result = {
            "image_id": imgId,
            "TPs": TPs,
            "FPs": FPs,
            "FNs": FNs,
            "local_F1s": F1,
            "IL_TP": (len(gt) > 0) and (len(dt) > 0),
            "IL_FP": (len(gt) == 0) and (len(dt) > 0),
            "IL_TN": (len(gt) == 0) and (len(dt) == 0),
            "IL_FN": (len(gt) > 0) and (len(dt) == 0),
            "num_dt": len(dt),
        }
        if len(gt) > 0 and len(dt) > 0:
            result["local_positive_F1s"] = F1
        return result

    def accumulate(self, p=None):

        if self.evalImgs is None or len(self.evalImgs) == 0:
            print("Please run evaluate() first")

        if p is None:
            p = self.params

        setImgIds = set(p.imgIds)


        TPs = np.zeros((len(p.iouThrs),), dtype=np.int64)
        FPs = np.zeros((len(p.iouThrs),), dtype=np.int64)
        pmFPs = np.zeros((len(p.iouThrs),), dtype=np.int64)
        FNs = np.zeros((len(p.iouThrs),), dtype=np.int64)
        local_F1s = np.zeros((len(p.iouThrs),), dtype=np.float64)


        IL_TPs = 0
        IL_FPs = 0
        IL_TNs = 0
        IL_FNs = 0

        valid_img_count = 0
        valid_F1_count = 0
        evaledImgIds = set()
        for res in self.evalImgs:
            if res["image_id"] not in setImgIds:
                continue
            evaledImgIds.add(res["image_id"])
            IL_TPs += res["IL_TP"]
            IL_FPs += res["IL_FP"]
            IL_TNs += res["IL_TN"]
            IL_FNs += res["IL_FN"]

            if "TPs" not in res:
                continue

            TPs += res["TPs"]
            FPs += res["FPs"]
            FNs += res["FNs"]
            valid_img_count += 1

            if "local_positive_F1s" in res:
                local_F1s += res["local_positive_F1s"]
                pmFPs += res["FPs"]
                if res["num_dt"] > 0:
                    valid_F1_count += 1

        assert len(setImgIds - evaledImgIds) == 0, (
            f"{len(setImgIds - evaledImgIds)} images not evaluated. "
            f"Here are the IDs of the first 3: {list(setImgIds - evaledImgIds)[:3]}"
        )


        precision = TPs / (TPs + FPs + 1e-4)
        positive_micro_precision = TPs / (TPs + pmFPs + 1e-4)
        assert np.all(precision <= 1)
        recall = TPs / (TPs + FNs + 1e-4)
        assert np.all(recall <= 1)
        F1 = 2 * precision * recall / (precision + recall + 1e-4)
        positive_micro_F1 = (
            2
            * positive_micro_precision
            * recall
            / (positive_micro_precision + recall + 1e-4)
        )

        IL_rec = IL_TPs / (IL_TPs + IL_FNs + 1e-6)
        IL_prec = IL_TPs / (IL_TPs + IL_FPs + 1e-6)
        IL_F1 = 2 * IL_prec * IL_rec / (IL_prec + IL_rec + 1e-6)
        IL_FPR = IL_FPs / (IL_FPs + IL_TNs + 1e-6)
        IL_MCC = float(IL_TPs * IL_TNs - IL_FPs * IL_FNs) / (
            (
                float(IL_TPs + IL_FPs)
                * float(IL_TPs + IL_FNs)
                * float(IL_TNs + IL_FPs)
                * float(IL_TNs + IL_FNs)
            )
            ** 0.5
            + 1e-6
        )

        self.eval = {
            "params": p,
            "TPs": TPs,
            "FPs": FPs,
            "positive_micro_FPs": pmFPs,
            "FNs": FNs,
            "precision": precision,
            "positive_micro_precision": positive_micro_precision,
            "recall": recall,
            "F1": F1,
            "positive_micro_F1": positive_micro_F1,
            "positive_macro_F1": local_F1s / valid_F1_count,
            "IL_recall": IL_rec,
            "IL_precision": IL_prec,
            "IL_F1": IL_F1,
            "IL_FPR": IL_FPR,
            "IL_MCC": IL_MCC,
        }
        self.eval["cgF1"] = self.eval["positive_micro_F1"] * self.eval["IL_MCC"]

    def summarize(self):

        if not self.eval:
            raise Exception("Please run accumulate() first")

        def _summarize(iouThr=None, metric=""):
            p = self.params
            iStr = " {:<18} @[ IoU={:<9}] = {:0.3f}"
            titleStr = "Average " + metric
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            s = self.eval[metric]

            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, iouStr, mean_s))
            return mean_s

        def _summarize_single(metric=""):
            titleStr = "Average " + metric
            iStr = " {:<35} = {:0.3f}"
            s = self.eval[metric]
            print(iStr.format(titleStr, s))
            return s

        def _summarizeDets():
            stats = []

            for metric in CGF1_METRICS:
                if metric.image_level:
                    stats.append(_summarize_single(metric=metric.name))
                else:
                    stats.append(
                        _summarize(iouThr=metric.iou_threshold, metric=metric.name)
                    )
            return np.asarray(stats)

        summarize = _summarizeDets
        self.stats = summarize()


def _evaluate(self):

    p = self.params

    p.imgIds = list(np.unique(p.imgIds))
    p.useCats = False
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()

    catIds = [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU
    else:
        raise RuntimeError(f"Unsupported iou {p.iouType}")
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds
    }

    maxDet = p.maxDets[-1]
    evalImgs = [
        self.evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]

    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    return p.imgIds, evalImgs


class CGF1Evaluator:


    def __init__(
        self,
        gt_path: Union[str, List[str]],
        iou_type="segm",
        verbose=False,
    ):

        self.gt_paths = gt_path if isinstance(gt_path, list) else [gt_path]
        self.iou_type = iou_type

        self.coco_gts = [COCOCustom(gt) for gt in self.gt_paths]

        self.verbose = verbose

        self.coco_evals = []
        for i, coco_gt in enumerate(self.coco_gts):
            self.coco_evals.append(
                CGF1Eval(
                    coco_gt=coco_gt,
                    iouType=iou_type,
                )
            )
            self.coco_evals[i].useCats = False

        exclude_img_ids = set()

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

    def evaluate(self, pred_file: str):

        assert len(self.coco_gts) > 0, "No ground truth provided for evaluation."
        assert len(self.coco_gts) == len(self.coco_evals), (
            "Mismatch in number of ground truths and evaluators."
        )

        if self.verbose:
            print(f"Loading predictions from {pred_file}")

        with open(pred_file, "r") as f:
            preds = json.load(f)

        if self.verbose:
            print(f"Loaded {len(preds)} predictions")

        img2preds = defaultdict(list)
        for pred in preds:
            img2preds[pred["image_id"]].append(pred)

        all_eval_imgs = []
        for img_id in tqdm(self.eval_img_ids, disable=not self.verbose):
            results = img2preds[img_id]
            all_scorings = []
            for cur_coco_gt, coco_eval in zip(self.coco_gts, self.coco_evals):

                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull):
                        coco_dt = (
                            cur_coco_gt.loadRes(results) if results else COCOCustom()
                        )

                coco_eval.cocoDt = coco_dt
                coco_eval.params.imgIds = [img_id]
                coco_eval.params.useCats = False
                img_ids, eval_imgs = _evaluate(coco_eval)
                all_scorings.append(eval_imgs)
            selected = self._select_best_scoring(all_scorings)
            all_eval_imgs.append(selected)


        self.coco_evals[0].evalImgs = list(
            np.concatenate(all_eval_imgs, axis=2).flatten()
        )
        self.coco_evals[0].params.imgIds = self.eval_img_ids
        self.coco_evals[0]._paramsEval = copy.deepcopy(self.coco_evals[0].params)

        if self.verbose:
            print(f"Accumulating results")
        self.coco_evals[0].accumulate()
        print("cgF1 metric, IoU type={}".format(self.iou_type))
        self.coco_evals[0].summarize()
        print()

        out = {}
        for i, value in enumerate(self.coco_evals[0].stats):
            name = CGF1_METRICS[i].name
            if CGF1_METRICS[i].iou_threshold is not None:
                name = f"{name}@{CGF1_METRICS[i].iou_threshold}"
            out[f"cgF1_eval_{self.iou_type}_{name}"] = float(value)

        return out

    @staticmethod
    def _select_best_scoring(scorings):


        if len(scorings) == 1:
            return scorings[0]

        assert scorings[0].ndim == 3, (
            f"Expecting results in [numCats, numAreas, numImgs] format, got {scorings[0].shape}"
        )
        assert scorings[0].shape[0] == 1, (
            f"Expecting a single category, got {scorings[0].shape[0]}"
        )

        for scoring in scorings:
            assert scoring.shape == scorings[0].shape, (
                f"Shape mismatch: {scoring.shape}, {scorings[0].shape}"
            )

        selected_imgs = []
        for img_id in range(scorings[0].shape[-1]):
            best = scorings[0][:, :, img_id]

            for scoring in scorings[1:]:
                current = scoring[:, :, img_id]
                if "local_F1s" in best[0, 0] and "local_F1s" in current[0, 0]:


                    best_score = best[0, 0]["local_F1s"].mean()
                    current_score = current[0, 0]["local_F1s"].mean()
                    if current_score > best_score:
                        best = current

                else:


                    if "local_F1s" not in current[0, 0]:
                        best = current
            selected_imgs.append(best)
        result = np.stack(selected_imgs, axis=-1)
        assert result.shape == scorings[0].shape
        return result
