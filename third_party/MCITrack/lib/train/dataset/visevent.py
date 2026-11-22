import os
import os.path
import torch
import numpy as np
import pandas
import csv
from glob import glob
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader_w_failsafe
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame


class VisEvent(BaseVideoDataset):


    def __init__(self, root=None, dtype='rgbrgb', split='train', image_loader=jpeg4py_loader_w_failsafe,
                 multi_modal_vision=False, multi_modal_language=False):

        root = env_settings().visevent_dir if root is None else root
        assert split in ['train'], 'Only support train split in VisEvent, got {}'.format(split)
        super().__init__('VisEvent', root, image_loader)

        self.dtype = dtype
        self.split = split
        self.sequence_list = self._build_sequence_list()

        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language

    def _build_sequence_list(self):

        file_path = os.path.join(self.root, '{}list.txt'.format(self.split))

        sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        return sequence_list

    def get_name(self):
        return 'visevent'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=True, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):

        occlusion_file = os.path.join(seq_path, "absent_label.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in list(csv.reader(f))])

        target_visible = occlusion

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)


        valid = (bbox[:, 2] > 5.0) & (bbox[:, 3] > 5.0)
        visible = self._read_target_visible(seq_path) & valid.byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):

        vis_img_files = sorted(glob(os.path.join(seq_path, 'vis_imgs', '*.bmp')))

        try:
            vis_path = vis_img_files[frame_id]
        except Exception as e:
            print(f"visevent bug: {e}")


        event_path = vis_path.replace('vis_imgs', 'event_imgs')


        return vis_path, event_path

    def _get_frame(self, seq_path, frame_id):

        color_path, event_path = self._get_frame_path(seq_path, frame_id)
        img = get_x_frame(color_path, event_path, dtype=self.dtype, depth_clip=False)
        return img

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for ii, f_id in enumerate(frame_ids)]

        frame_list = [self._get_frame(seq_path, f_id) for ii, f_id in enumerate(frame_ids)]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
