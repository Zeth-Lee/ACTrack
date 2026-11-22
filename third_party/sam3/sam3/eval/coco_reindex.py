# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def reindex_coco_to_temp(input_json_path: str) -> Optional[str]:


    def is_coco_json(data: Dict[str, Any]) -> bool:

        if not isinstance(data, dict):
            return False

        coco_keys = {"images", "annotations", "categories"}
        return any(key in data for key in coco_keys)

    def check_zero_indexed(data: Dict[str, Any]) -> Tuple[bool, bool, bool]:

        annotations_zero = False
        images_zero = False
        categories_zero = False


        annotations = data.get("annotations", [])
        if annotations and any(ann.get("id", -1) == 0 for ann in annotations):
            annotations_zero = True


        images = data.get("images", [])
        if images and any(img.get("id", -1) == 0 for img in images):
            images_zero = True


        categories = data.get("categories", [])
        if categories and any(cat.get("id", -1) == 0 for cat in categories):
            categories_zero = True

        return annotations_zero, images_zero, categories_zero

    def reindex_coco_data(data: Dict[str, Any]) -> Dict[str, Any]:

        modified_data = data.copy()

        annotations_zero, images_zero, categories_zero = check_zero_indexed(data)


        image_id_mapping = {}
        category_id_mapping = {}


        if images_zero and "images" in modified_data:
            for img in modified_data["images"]:
                old_id = img["id"]
                new_id = old_id + 1
                image_id_mapping[old_id] = new_id
                img["id"] = new_id


        if categories_zero and "categories" in modified_data:
            for cat in modified_data["categories"]:
                old_id = cat["id"]
                new_id = old_id + 1
                category_id_mapping[old_id] = new_id
                cat["id"] = new_id


        if "annotations" in modified_data:
            for ann in modified_data["annotations"]:

                if annotations_zero:
                    ann["id"] = ann["id"] + 1


                if images_zero and ann.get("image_id") is not None:
                    old_image_id = ann["image_id"]
                    if old_image_id in image_id_mapping:
                        ann["image_id"] = image_id_mapping[old_image_id]


                if categories_zero and ann.get("category_id") is not None:
                    old_category_id = ann["category_id"]
                    if old_category_id in category_id_mapping:
                        ann["category_id"] = category_id_mapping[old_category_id]

        return modified_data


    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input file not found: {input_json_path}")


    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {input_json_path}: {e}")


    if not is_coco_json(data):
        raise ValueError(
            f"File does not appear to be in COCO format: {input_json_path}"
        )


    annotations_zero, images_zero, categories_zero = check_zero_indexed(data)

    if not (annotations_zero or images_zero or categories_zero):

        input_path = Path(input_json_path)
        temp_dir = tempfile.mkdtemp()
        temp_filename = f"{input_path.stem}_1_indexed{input_path.suffix}"
        temp_path = os.path.join(temp_dir, temp_filename)

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return temp_path


    modified_data = reindex_coco_data(data)


    input_path = Path(input_json_path)
    temp_dir = tempfile.mkdtemp()
    temp_filename = f"{input_path.stem}_1_indexed{input_path.suffix}"
    temp_path = os.path.join(temp_dir, temp_filename)


    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(modified_data, f, indent=2, ensure_ascii=False)

    return temp_path


def test_reindex_function():


    test_data = {
        "info": {"description": "Test COCO dataset", "version": "1.0", "year": 2023},
        "images": [
            {"id": 0, "width": 640, "height": 480, "file_name": "test1.jpg"},
            {"id": 1, "width": 640, "height": 480, "file_name": "test2.jpg"},
        ],
        "categories": [
            {"id": 0, "name": "person", "supercategory": "person"},
            {"id": 1, "name": "car", "supercategory": "vehicle"},
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 0,
                "bbox": [100, 100, 50, 75],
                "area": 3750,
                "iscrowd": 0,
            },
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [200, 150, 120, 80],
                "area": 9600,
                "iscrowd": 0,
            },
        ],
    }


    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_data, f, indent=2)
        test_file_path = f.name

    try:

        result_path = reindex_coco_to_temp(test_file_path)
        print(f"Original file: {test_file_path}")
        print(f"Converted file: {result_path}")


        with open(result_path, "r") as f:
            result_data = json.load(f)

        print("\nConverted data sample:")
        print(f"First image ID: {result_data['images'][0]['id']}")
        print(f"First category ID: {result_data['categories'][0]['id']}")
        print(f"First annotation ID: {result_data['annotations'][0]['id']}")
        print(f"First annotation image_id: {result_data['annotations'][0]['image_id']}")
        print(
            f"First annotation category_id: {result_data['annotations'][0]['category_id']}"
        )


        os.unlink(result_path)
        os.rmdir(os.path.dirname(result_path))

    finally:

        os.unlink(test_file_path)


if __name__ == "__main__":
    test_reindex_function()
