# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import json
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from pycocotools import mask as mask_util


def convert_boxlist_to_normalized_tensor(box_list, image_width, image_height):

    boxes = torch.tensor(box_list, dtype=torch.float32)
    boxes[:, [0, 2]] /= image_width
    boxes[:, [1, 3]] /= image_height
    boxes = boxes.clamp(0, 1)
    return boxes


def load_coco_and_group_by_image(json_path: str) -> Tuple[List[Dict], Dict[int, str]]:

    with open(json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    sorted_image_ids = sorted(images.keys())

    grouped = []
    for image_id in sorted_image_ids:
        image_info = images[image_id]
        grouped.append(
            {"image": image_info, "annotations": anns_by_image.get(image_id, [])}
        )

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    return grouped, cat_id_to_name


def ann_to_rle(segm, im_info: Dict) -> Dict:

    h, w = im_info["height"], im_info["width"]

    if isinstance(segm, list):

        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif isinstance(segm["counts"], list):

        rle = mask_util.frPyObjects(segm, h, w)
    else:

        rle = segm

    return rle


class COCO_FROM_JSON:


    def __init__(
        self,
        annotation_file,
        prompts=None,
        include_negatives=True,
        category_chunk_size=None,
    ):

        self._raw_data, self._cat_idx_to_text = load_coco_and_group_by_image(
            annotation_file
        )
        self._sorted_cat_ids = sorted(list(self._cat_idx_to_text.keys()))
        self.prompts = None
        self.include_negatives = include_negatives
        self.category_chunk_size = (
            category_chunk_size
            if category_chunk_size is not None
            else len(self._sorted_cat_ids)
        )
        self.category_chunks = [
            self._sorted_cat_ids[i : i + self.category_chunk_size]
            for i in range(0, len(self._sorted_cat_ids), self.category_chunk_size)
        ]
        if prompts is not None:
            prompts = eval(prompts)
            self.prompts = {}
            for loc_dict in prompts:
                self.prompts[int(loc_dict["id"])] = loc_dict["name"]
            assert len(self.prompts) == len(self._sorted_cat_ids), (
                "Number of prompts must match number of categories"
            )

    def getDatapointIds(self):

        return list(range(len(self._raw_data) * len(self.category_chunks)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):

        img_idx = idx // len(self.category_chunks)
        chunk_idx = idx % len(self.category_chunks)
        cat_chunk = self.category_chunks[chunk_idx]

        queries = []
        annotations = []

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,  
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }

        annot_template = {
            "image_id": 0,
            "bbox": None,  
            "area": None,  
            "segmentation": None,  
            "object_id": None,
            "is_crowd": None,
            "id": None,
        }

        raw_annotations = self._raw_data[img_idx]["annotations"]
        image_info = self._raw_data[img_idx]["image"]
        width, height = image_info["width"], image_info["height"]


        cat_id_to_anns = defaultdict(list)
        for ann in raw_annotations:
            cat_id_to_anns[ann["category_id"]].append(ann)

        annotations_by_cat_sorted = [
            (cat_id, cat_id_to_anns[cat_id]) for cat_id in cat_chunk
        ]

        for cat_id, anns in annotations_by_cat_sorted:
            if len(anns) == 0 and not self.include_negatives:
                continue

            cur_ann_ids = []


            for ann in anns:
                annotation = annot_template.copy()
                annotation["id"] = len(annotations)
                annotation["object_id"] = annotation["id"]
                annotation["is_crowd"] = ann["iscrowd"]

                normalized_boxes = convert_boxlist_to_normalized_tensor(
                    [ann["bbox"]], width, height
                )
                bbox = normalized_boxes[0]

                annotation["area"] = (bbox[2] * bbox[3]).item()
                annotation["bbox"] = bbox

                if (
                    "segmentation" in ann
                    and ann["segmentation"] is not None
                    and ann["segmentation"] != []
                ):
                    annotation["segmentation"] = ann_to_rle(
                        ann["segmentation"], im_info=image_info
                    )

                annotations.append(annotation)
                cur_ann_ids.append(annotation["id"])


            query = query_template.copy()
            query["id"] = len(queries)
            query["original_cat_id"] = cat_id
            query["query_text"] = (
                self._cat_idx_to_text[cat_id]
                if self.prompts is None
                else self.prompts[cat_id]
            )
            query["object_ids_output"] = cur_ann_ids
            queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):

        img_idx = idx // len(self.category_chunks)
        img_data = self._raw_data[img_idx]["image"]
        images = [
            {
                "id": 0,
                "file_name": img_data["file_name"],
                "original_img_id": img_data["id"],
                "coco_img_id": img_data["id"],
            }
        ]
        return images


class SAM3_EVAL_API_FROM_JSON_NP:


    def __init__(self, annotation_file):

        with open(annotation_file, "r") as f:
            data = json.load(f)
        self._image_data = data["images"]

    def getDatapointIds(self):

        return list(range(len(self._image_data)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):

        cur_img_data = self._image_data[idx]
        queries = []
        annotations = []

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }


        query = query_template.copy()
        query["id"] = len(queries)
        query["original_cat_id"] = int(cur_img_data["queried_category"])
        query["query_text"] = cur_img_data["text_input"]
        query["object_ids_output"] = []
        queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):

        img_data = self._image_data[idx]
        images = [
            {
                "id": 0,
                "file_name": img_data["file_name"],
                "original_img_id": img_data["id"],
                "coco_img_id": img_data["id"],
            }
        ]
        return images


class SAM3_VEVAL_API_FROM_JSON_NP:


    def __init__(self, annotation_file):

        with open(annotation_file, "r") as f:
            data = json.load(f)

        assert "video_np_pairs" in data, "Incorrect data format"

        self._video_data = data["videos"]
        self._video_id_to_np_ids = defaultdict(list)
        self._cat_id_to_np = {}

        for cat_dict in data["categories"]:
            self._cat_id_to_np[cat_dict["id"]] = cat_dict["name"]

        for video_np_dict in data["video_np_pairs"]:
            self._video_id_to_np_ids[video_np_dict["video_id"]].append(
                video_np_dict["category_id"]
            )
            assert (
                self._cat_id_to_np[video_np_dict["category_id"]]
                == video_np_dict["noun_phrase"]
            ), "Category name does not match text input"

    def getDatapointIds(self):

        return list(range(len(self._video_data)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):

        cur_vid_data = self._video_data[idx]
        queries = []
        annotations = []

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }

        all_np_ids = self._video_id_to_np_ids[cur_vid_data["id"]]

        for np_id in all_np_ids:
            text_input = self._cat_id_to_np[np_id]

            for i, image_path in enumerate(cur_vid_data["file_names"]):
                query = query_template.copy()
                query["id"] = len(queries)
                query["original_cat_id"] = np_id
                query["query_text"] = text_input
                query["image_id"] = i
                query["query_processing_order"] = i
                query["object_ids_output"] = []
                queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):

        video_data = self._video_data[idx]
        images = [
            {
                "id": i,
                "file_name": file_name,
                "original_img_id": video_data["id"],
                "coco_img_id": video_data["id"],
            }
            for i, file_name in enumerate(video_data["file_names"])
        ]
        return images
