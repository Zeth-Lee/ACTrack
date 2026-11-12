import csv
import io
import os
import traceback
import zipfile
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from .. import _timing
from ..utils import TrackEvalException


class _BaseDataset(ABC):
    @abstractmethod
    def __init__(self):
        self.tracker_list = None
        self.seq_list = None
        self.class_list = None
        self.output_fol = None
        self.output_sub_fol = None
        self.should_classes_combine = True
        self.use_super_categories = False


    @abstractmethod
    def _load_raw_file(self, tracker, seq, is_gt):
        ...

    @_timing.time
    @abstractmethod
    def get_preprocessed_seq_data(self, raw_data, cls):
        ...

    @abstractmethod
    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        ...


    @classmethod
    def get_class_name(cls):
        return cls.__name__

    def get_name(self):
        return self.get_class_name()

    def get_output_fol(self, tracker):
        return os.path.join(self.output_fol, tracker, self.output_sub_fol)

    def get_display_name(self, tracker):

        return tracker

    def get_eval_info(self):

        return self.tracker_list, self.seq_list, self.class_list

    @_timing.time
    def get_raw_seq_data(self, tracker, seq):


        raw_gt_data = self._load_raw_file(tracker, seq, is_gt=True)
        raw_tracker_data = self._load_raw_file(tracker, seq, is_gt=False)
        raw_data = {**raw_tracker_data, **raw_gt_data}


        similarity_scores = []
        for _, (gt_dets_t, tracker_dets_t) in enumerate(
            zip(raw_data["gt_dets"], raw_data["tk_dets"])
        ):
            ious = self._calculate_similarities(gt_dets_t, tracker_dets_t)
            similarity_scores.append(ious)
        raw_data["similarity_scores"] = similarity_scores
        return raw_data

    @staticmethod
    def _load_simple_text_file(
        file,
        time_col=0,
        id_col=None,
        remove_negative_ids=False,
        valid_filter=None,
        crowd_ignore_filter=None,
        convert_filter=None,
        is_zipped=False,
        zip_file=None,
        force_delimiters=None,
    ):


        if remove_negative_ids and id_col is None:
            raise TrackEvalException(
                "remove_negative_ids is True, but id_col is not given."
            )
        if crowd_ignore_filter is None:
            crowd_ignore_filter = {}
        if convert_filter is None:
            convert_filter = {}
        try:
            if is_zipped:
                if zip_file is None:
                    raise TrackEvalException(
                        "is_zipped set to True, but no zip_file is given."
                    )
                archive = zipfile.ZipFile(os.path.join(zip_file), "r")
                fp = io.TextIOWrapper(archive.open(file, "r"))
            else:
                fp = open(file)
            read_data = {}
            crowd_ignore_data = {}
            fp.seek(0, os.SEEK_END)

            if fp.tell():
                fp.seek(0)
                dialect = csv.Sniffer().sniff(
                    fp.readline(), delimiters=force_delimiters
                )
                dialect.skipinitialspace = (
                    True
                )
                fp.seek(0)
                reader = csv.reader(fp, dialect)
                for row in reader:
                    try:

                        if row[-1] in "":
                            row = row[:-1]
                        timestep = str(int(float(row[time_col])))

                        is_ignored = False
                        for ignore_key, ignore_value in crowd_ignore_filter.items():
                            if row[ignore_key].lower() in ignore_value:

                                for (
                                    convert_key,
                                    convert_value,
                                ) in convert_filter.items():
                                    row[convert_key] = convert_value[
                                        row[convert_key].lower()
                                    ]

                                if timestep in crowd_ignore_data.keys():
                                    crowd_ignore_data[timestep].append(row)
                                else:
                                    crowd_ignore_data[timestep] = [row]
                                is_ignored = True
                        if (
                            is_ignored
                        ):
                            continue

                        if valid_filter is not None:
                            for key, value in valid_filter.items():
                                if row[key].lower() not in value:
                                    continue
                        if remove_negative_ids:
                            if int(float(row[id_col])) < 0:
                                continue

                        for convert_key, convert_value in convert_filter.items():
                            row[convert_key] = convert_value[row[convert_key].lower()]

                        if timestep in read_data.keys():
                            read_data[timestep].append(row)
                        else:
                            read_data[timestep] = [row]
                    except Exception:
                        exc_str_init = (
                            "In file %s the following line cannot be read correctly: \n"
                            % os.path.basename(file)
                        )
                        exc_str = " ".join([exc_str_init] + row)
                        raise TrackEvalException(exc_str)
            fp.close()
        except Exception:
            print("Error loading file: %s, printing traceback." % file)
            traceback.print_exc()
            raise TrackEvalException(
                "File %s cannot be read because it is either not present or invalidly formatted"
                % os.path.basename(file)
            )
        return read_data, crowd_ignore_data

    @staticmethod
    def _calculate_mask_ious(masks1, masks2, is_encoded=False, do_ioa=False):


        from pycocotools import mask as mask_utils


        if not is_encoded:
            masks1 = mask_utils.encode(
                np.array(np.transpose(masks1, (1, 2, 0)), order="F")
            )
            masks2 = mask_utils.encode(
                np.array(np.transpose(masks2, (1, 2, 0)), order="F")
            )


        ious = mask_utils.iou(masks1, masks2, [do_ioa] * len(masks2))
        if len(masks1) == 0 or len(masks2) == 0:
            ious = np.asarray(ious).reshape(len(masks1), len(masks2))
        assert (ious >= 0 - np.finfo("float").eps).all()
        assert (ious <= 1 + np.finfo("float").eps).all()

        return ious

    @staticmethod
    def _calculate_box_ious(bboxes1, bboxes2, box_format="xywh", do_ioa=False):

        if box_format in "xywh":

            bboxes1 = deepcopy(bboxes1)
            bboxes2 = deepcopy(bboxes2)

            bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
            bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
            bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
            bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]
        elif box_format not in "x0y0x1y1":
            raise (TrackEvalException("box_format %s is not implemented" % box_format))


        min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(
            min_[..., 3] - max_[..., 1], 0
        )
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1]
        )

        if do_ioa:
            ioas = np.zeros_like(intersection)
            valid_mask = area1 > 0 + np.finfo("float").eps
            ioas[valid_mask, :] = (
                intersection[valid_mask, :] / area1[valid_mask][:, np.newaxis]
            )

            return ioas
        else:
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
                bboxes2[..., 3] - bboxes2[..., 1]
            )
            union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
            intersection[area1 <= 0 + np.finfo("float").eps, :] = 0
            intersection[:, area2 <= 0 + np.finfo("float").eps] = 0
            intersection[union <= 0 + np.finfo("float").eps] = 0
            union[union <= 0 + np.finfo("float").eps] = 1
            ious = intersection / union
            return ious

    @staticmethod
    def _calculate_euclidean_similarity(dets1, dets2, zero_distance=2.0):

        dist = np.linalg.norm(dets1[:, np.newaxis] - dets2[np.newaxis, :], axis=2)
        sim = np.maximum(0, 1 - dist / zero_distance)
        return sim

    @staticmethod
    def _check_unique_ids(data, after_preproc=False):

        gt_ids = data["gt_ids"]
        tracker_ids = data["tk_ids"]
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(gt_ids, tracker_ids)):
            if len(tracker_ids_t) > 0:
                unique_ids, counts = np.unique(tracker_ids_t, return_counts=True)
                if np.max(counts) != 1:
                    duplicate_ids = unique_ids[counts > 1]
                    exc_str_init = (
                        "Tracker predicts the same ID more than once in a single timestep "
                        "(seq: %s, frame: %i, ids:" % (data["seq"], t + 1)
                    )
                    exc_str = (
                        " ".join([exc_str_init] + [str(d) for d in duplicate_ids]) + ")"
                    )
                    if after_preproc:
                        exc_str_init += (
                            "\n Note that this error occurred after preprocessing (but not before), "
                            "so ids may not be as in file, and something seems wrong with preproc."
                        )
                    raise TrackEvalException(exc_str)
            if len(gt_ids_t) > 0:
                unique_ids, counts = np.unique(gt_ids_t, return_counts=True)
                if np.max(counts) != 1:
                    duplicate_ids = unique_ids[counts > 1]
                    exc_str_init = (
                        "Ground-truth has the same ID more than once in a single timestep "
                        "(seq: %s, frame: %i, ids:" % (data["seq"], t + 1)
                    )
                    exc_str = (
                        " ".join([exc_str_init] + [str(d) for d in duplicate_ids]) + ")"
                    )
                    if after_preproc:
                        exc_str_init += (
                            "\n Note that this error occurred after preprocessing (but not before), "
                            "so ids may not be as in file, and something seems wrong with preproc."
                        )
                    raise TrackEvalException(exc_str)
