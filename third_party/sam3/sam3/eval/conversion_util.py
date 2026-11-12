# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import json
import os
from collections import defaultdict

from tqdm import tqdm


def convert_ytbvis_to_cocovid_gt(ann_json, save_path=None):


    VIS = {
        "info": {},
        "images": [],
        "videos": [],
        "tracks": [],
        "annotations": [],
        "categories": [],
        "licenses": [],
    }


    official_anns = json.load(open(ann_json))
    VIS["categories"] = official_anns["categories"]  


    records = dict(img_id=1, ann_id=1)


    vid_to_anns = defaultdict(list)
    for ann in official_anns["annotations"]:
        vid_to_anns[ann["video_id"]].append(ann)


    VIS["tracks"] = [
        {
            "id": ann["id"],
            "category_id": ann["category_id"],
            "video_id": ann["video_id"],
        }
        for ann in official_anns["annotations"]
    ]


    for video_info in tqdm(official_anns["videos"]):

        video = {
            "id": video_info["id"],
            "name": os.path.dirname(video_info["file_names"][0]),
            "width": video_info["width"],
            "height": video_info["height"],
            "length": video_info["length"],
            "neg_category_ids": [],
            "not_exhaustive_category_ids": [],
        }
        VIS["videos"].append(video)


        num_frames = len(video_info["file_names"])
        for frame_idx in range(num_frames):

            image = {
                "id": records["img_id"],
                "video_id": video_info["id"],
                "file_name": video_info["file_names"][frame_idx],
                "width": video_info["width"],
                "height": video_info["height"],
                "frame_index": frame_idx,
                "frame_id": frame_idx,
            }
            VIS["images"].append(image)


            if video_info["id"] in vid_to_anns:
                for ann in vid_to_anns[video_info["id"]]:
                    bbox = ann["bboxes"][frame_idx]
                    if bbox is None:
                        continue


                    annotation = {
                        "id": records["ann_id"],
                        "video_id": video_info["id"],
                        "image_id": records["img_id"],
                        "track_id": ann["id"],
                        "category_id": ann["category_id"],
                        "bbox": bbox,
                        "area": ann["areas"][frame_idx],
                        "segmentation": ann["segmentations"][frame_idx],
                        "iscrowd": ann["iscrowd"],
                    }
                    VIS["annotations"].append(annotation)
                    records["ann_id"] += 1

            records["img_id"] += 1


    print(f"Converted {len(VIS['videos'])} videos")
    print(f"Converted {len(VIS['images'])} images")
    print(f"Created {len(VIS['tracks'])} tracks")
    print(f"Created {len(VIS['annotations'])} annotations")

    if save_path is None:
        return VIS


    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    json.dump(VIS, open(save_path, "w"))

    return VIS


def convert_ytbvis_to_cocovid_pred(
    youtubevis_pred_path: str, converted_dataset_path: str, output_path: str
) -> None:


    with open(youtubevis_pred_path) as f:
        ytv_predictions = json.load(f)


    with open(converted_dataset_path) as f:
        coco_dataset = json.load(f)


    image_id_map = {
        (img["video_id"], img["frame_index"]): img["id"]
        for img in coco_dataset["images"]
    }

    coco_annotations = []
    track_id_counter = 1

    for pred in tqdm(ytv_predictions):
        video_id = pred["video_id"]
        category_id = pred["category_id"]
        bboxes = pred["bboxes"]
        segmentations = pred.get("segmentations", [])  
        areas = pred.get("areas", [])  
        score = pred["score"]


        track_id = track_id_counter
        track_id_counter += 1


        if len(segmentations) == 0:
            segmentations = [None] * len(bboxes)
        if len(areas) == 0:
            areas = [None] * len(bboxes)

        for frame_idx, (bbox, segmentation, area_from_pred) in enumerate(
            zip(bboxes, segmentations, areas)
        ):

            if bbox is None or all(x == 0 for x in bbox):
                continue


            image_id = image_id_map.get((video_id, frame_idx))
            if image_id is None:
                raise RuntimeError(
                    f"prediction {video_id=}, {frame_idx=} does not match any images in the converted COCO format"
                )


            x, y, w, h = bbox


            if area_from_pred is not None and area_from_pred > 0:
                area = area_from_pred
            else:
                area = w * h


            coco_annotation = {
                "image_id": int(image_id),
                "video_id": video_id,  
                "track_id": track_id,
                "category_id": category_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(area),
                "iscrowd": 0,
                "score": float(score),
            }


            if segmentation is not None:
                coco_annotation["segmentation"] = segmentation

            coco_annotations.append(coco_annotation)


    with open(output_path, "w") as f:
        json.dump(coco_annotations, f)

    print(f"Converted {len(coco_annotations)} predictions to COCO format with video_id")
