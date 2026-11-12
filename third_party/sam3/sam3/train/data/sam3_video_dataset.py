# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import copy
import io
import json
import logging
import math
import os
import pickle
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torchvision


from iopath.common.file_io import PathManager
from PIL import Image as PILImage

from .sam3_image_dataset import Datapoint, Sam3ImageDataset


SEED = 42


class VideoGroundingDataset(Sam3ImageDataset):
    def __init__(
        self,
        num_stages_sample: int = 4,
        stage_stride_min: int = 1,
        stage_stride_max: int = 5,
        random_reverse_time_axis: bool = True,
        is_tiling_single_image: bool = False,


        tile_img_keep_find_queries_with_geo_inputs: bool = False,
        tile_img_keep_get_queries: bool = False,


        max_query_num: int = -1,


        override_query_is_exhaustive_to_true: bool = False,


        max_masklet_num_in_video: int = 300,
        **kwargs,
    ):

        super().__init__(**kwargs)
        assert num_stages_sample >= 1
        assert stage_stride_min >= 1
        assert stage_stride_max >= stage_stride_min
        self.num_stages_sample = num_stages_sample
        self.stage_stride_min = stage_stride_min
        self.stage_stride_max = stage_stride_max
        self.random_reverse_time_axis = random_reverse_time_axis
        self.is_tiling_single_image = is_tiling_single_image
        self.tile_img_keep_find_queries_with_geo_inputs = (
            tile_img_keep_find_queries_with_geo_inputs
        )
        self.tile_img_keep_get_queries = tile_img_keep_get_queries
        self.max_query_num = max_query_num
        self.override_query_is_exhaustive_to_true = override_query_is_exhaustive_to_true
        self.max_masklet_num_in_video = max_masklet_num_in_video
        self.rng = random.Random()
        self.set_curr_epoch(0)

    def set_curr_epoch(self, epoch: int):
        super().set_curr_epoch(epoch)
        self.rng.seed(SEED + epoch)

    def _load_datapoint(self, index: int) -> Datapoint:
        id = self.ids[index].item()
        queries, annotations = self.coco.loadQueriesAndAnnotationsFromDatapoint(id)


        if self.training and not self.is_tiling_single_image:

            stage_stride = self.rng.randint(
                self.stage_stride_min, self.stage_stride_max
            )
            stage_ids_to_keep = self._sample_stage_ids(
                queries, self.num_stages_sample, stage_stride
            )


            reverse_time_axis = (
                self.rng.random() < 0.5 if self.random_reverse_time_axis else False
            )
            queries, annotations, kept_img_ids = self._filter_query_and_anns(
                queries,
                annotations,
                stage_ids_to_keep,
                remap_stage_id=True,
                reverse_time_axis=reverse_time_axis,
            )
            pil_images, img_metadata = self._load_images(id, kept_img_ids)
            if reverse_time_axis:


                pil_images = pil_images[::-1]
                img_metadata = img_metadata[::-1]
        else:
            pil_images, img_metadata = self._load_images(id)


        assert all(p.size == pil_images[0][1].size for _, p in pil_images)

        queries.sort(key=lambda q: q["query_processing_order"])
        if self.override_query_is_exhaustive_to_true:
            for query in queries:
                query["is_exhaustive"] = True
        datapoint = self.load_queries(pil_images, annotations, queries, img_metadata)


        num_masklets_in_video = len(datapoint.images[0].objects)
        if num_masklets_in_video > self.max_masklet_num_in_video > 0:
            logging.warning(
                f"Datapoint {id} has ({num_masklets_in_video=}), exceeding "
                f"the maximum allowed ({self.max_masklet_num_in_video}). "
                "Skipping this datapoint."
            )
            next_index = (index + 1) % len(self)
            return self._load_datapoint(next_index)

        if self.is_tiling_single_image:
            datapoint = self._tile_single_image_data(datapoint, self.num_stages_sample)
        if self.max_query_num > 0:
            datapoint = self._subsample_queries(datapoint, self.max_query_num)


        for query in datapoint.find_queries:
            assert query.image_id == query.query_processing_order, (
                f"find query has inconsistent image_id and "
                f"query_processing_order: {query.image_id=} vs "
                f"{query.query_processing_order=}"
            )
        return datapoint

    def _sample_stage_ids(self, queries, num_stages_sample, stage_stride):


        all_stage_ids = sorted(set(q["query_processing_order"] for q in queries))
        num_stages_total = len(all_stage_ids)
        if num_stages_total < num_stages_sample:
            raise ValueError("Not enough stages to sample")


        b_e_gap = (num_stages_sample - 1) * stage_stride
        if b_e_gap > num_stages_total - 1:


            prev_stage_stride = stage_stride
            stage_stride = math.floor((num_stages_total - 1) / (num_stages_sample - 1))
            logging.info(
                f"lowering stride from {prev_stage_stride} to {stage_stride} to "
                f"sample {num_stages_sample} stages (from {num_stages_total} total)"
            )
            b_e_gap = (num_stages_sample - 1) * stage_stride


        b_max = len(all_stage_ids) - 1 - b_e_gap
        b = self.rng.randint(0, b_max)
        e = b + b_e_gap
        stage_ids_to_keep = all_stage_ids[b : e + 1 : stage_stride]
        return stage_ids_to_keep

    def _filter_query_and_anns(
        self, queries, annotations, stage_ids_to_keep, remap_stage_id, reverse_time_axis
    ):

        stage_ids_to_keep = set(stage_ids_to_keep)
        kept_img_ids = set()
        kept_stage_ids = set()


        filtered_queries = []
        for query in queries:
            input_box = query.get("input_box", None)
            input_points = query.get("input_points", None)
            has_geo_input = input_box is not None or input_points is not None
            if has_geo_input and not self.tile_img_keep_find_queries_with_geo_inputs:
                continue
            stage_id = query["query_processing_order"]
            if stage_id in stage_ids_to_keep:
                kept_img_ids.add(query["image_id"])
                kept_stage_ids.add(stage_id)
                filtered_queries.append(query)

        all_frame_present = kept_stage_ids == stage_ids_to_keep
        assert all_frame_present, f"{kept_stage_ids=} vs {stage_ids_to_keep=}"
        if remap_stage_id:

            old_stage_ids = sorted(kept_stage_ids, reverse=reverse_time_axis)
            stage_id_old2new = {old: new for new, old in enumerate(old_stage_ids)}
            for query in filtered_queries:
                ptr_x_is_empty = query["ptr_x_query_id"] in [None, -1]
                ptr_y_is_empty = query["ptr_y_query_id"] in [None, -1]
                assert ptr_x_is_empty and ptr_y_is_empty, (
                    "Remapping stage ids is not supported for queries with non-empty ptr_x or ptr_y pointers"
                )
                query["query_processing_order"] = stage_id_old2new[
                    query["query_processing_order"]
                ]


        filtered_annotations = [
            ann for ann in annotations if ann["image_id"] in kept_img_ids
        ]

        return filtered_queries, filtered_annotations, kept_img_ids

    def _tile_single_image_data(self, datapoint: Datapoint, num_stages_sample: int):


        assert len(datapoint.images) == 1, "Expected only one single image"
        tiled_images = [
            copy.deepcopy(datapoint.images[0]) for _ in range(num_stages_sample)
        ]
        for stage_id, img in enumerate(tiled_images):
            for obj in img.objects:
                obj.frame_index = stage_id


        tiled_raw_images = None
        if datapoint.raw_images is not None:
            assert len(datapoint.raw_images) == 1, "Expected only one single image"
            tiled_raw_images = [
                datapoint.raw_images[0].copy() for _ in range(num_stages_sample)
            ]


        tiled_find_queries_per_stage = [[] for _ in range(num_stages_sample)]
        for query in datapoint.find_queries:
            assert query.image_id == 0
            assert query.query_processing_order == 0


            assert query.ptr_x is None and query.ptr_y is None
            assert query.ptr_mem is None


            has_geo_input = (
                query.input_bbox is not None or query.input_points is not None
            )
            if has_geo_input and not self.tile_img_keep_find_queries_with_geo_inputs:
                continue
            for stage_id in range(num_stages_sample):

                new_query = copy.deepcopy(query)
                new_query.image_id = stage_id
                new_query.query_processing_order = stage_id
                if new_query.inference_metadata is not None:
                    new_query.inference_metadata.frame_index = stage_id
                tiled_find_queries_per_stage[stage_id].append(new_query)

        tiled_find_queries = sum(tiled_find_queries_per_stage, [])


        if self.tile_img_keep_get_queries:
            raise NotImplementedError("Tiling get queries is not implemented yet")
        else:
            tiled_get_queries = []

        return Datapoint(
            images=tiled_images,
            raw_images=tiled_raw_images,
            find_queries=tiled_find_queries,
            get_queries=tiled_get_queries,
        )

    def _subsample_queries(self, datapoint: Datapoint, max_query_num: int):


        num_frames = max(q.query_processing_order for q in datapoint.find_queries) + 1
        find_queries_per_stage = [[] for _ in range(num_frames)]
        for query in datapoint.find_queries:
            find_queries_per_stage[query.query_processing_order].append(query)


        num_queries_per_stage = len(find_queries_per_stage[0])
        for queries in find_queries_per_stage:
            assert len(queries) == num_queries_per_stage
        if max_query_num <= 0 or num_queries_per_stage <= max_query_num:
            return datapoint


        sampled_inds = self.rng.sample(range(num_queries_per_stage), max_query_num)
        sampled_find_queries_per_stage = [
            [queries[idx] for idx in sampled_inds] for queries in find_queries_per_stage
        ]
        sampled_find_queries = sum(sampled_find_queries_per_stage, [])
        return Datapoint(
            images=datapoint.images,
            raw_images=datapoint.raw_images,
            find_queries=sampled_find_queries,
            get_queries=datapoint.get_queries,
        )
