import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame


class LasHeR(BaseVideoDataset):


    def __init__(self, root=None, split='train', dtype='rgbrgb', seq_ids=None, data_fraction=None,
                 multi_modal_vision=False, multi_modal_language=False):

        root = env_settings().lasher_dir if root is None else root
        assert split in ['train', 'val','all'], 'Only support all, train or val split in LasHeR, got {}'.format(split)
        super().__init__('LasHeR', root)
        self.dtype = dtype


        self.sequence_list = self._get_sequence_list(split)


        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language

    def get_name(self):
        return 'lasher'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _get_sequence_list(self, split):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        file_path = os.path.join(ltr_path, 'data_specs', 'lasher_{}.txt'.format(split))
        with open(file_path, 'r') as f:
            dir_list = f.read().splitlines()
        return dir_list

    def _read_bb_anno(self, seq_path):

        rgb_bb_anno_file = os.path.join(seq_path, "visible.txt")

        rgb_gt = pandas.read_csv(rgb_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values

        return torch.tensor(rgb_gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):

        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):

        rgb_frame_path = os.path.join(seq_path, 'visible', '{:06d}.jpg'.format(frame_id))  
        ir_frame_path = os.path.join(seq_path, 'infrared', '{:06d}.jpg'.format(frame_id))
        return (rgb_frame_path, ir_frame_path)

    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, frame_id)
        img = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        return img

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
