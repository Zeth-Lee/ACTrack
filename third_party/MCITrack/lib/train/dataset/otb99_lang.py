import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class Otb99_lang(BaseVideoDataset):


    def __init__(self, root=None, image_loader=jpeg4py_loader, vid_ids=None, split=None, data_fraction=None,
                 multi_modal_vision=False, multi_modal_language=False):

        root = env_settings().lasot_dir if root is None else root
        super().__init__('Otb99_lang', root, image_loader)

        self.class_list = [f for f in os.listdir(self.root)]
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}

        self.sequence_list = self._build_sequence_list(vid_ids, split)
        self.split = split
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.seq_per_class = self._build_class_list()

        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language

    def _build_sequence_list(self, vid_ids=None, split=None):
        if split is not None:
            if vid_ids is not None:
                raise ValueError('Cannot set both split_name and vid_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'otb99_train_split.txt')
            elif split == 'test':
                file_path = os.path.join(ltr_path, 'data_specs', 'otb99_test_split.txt')
            else:
                raise ValueError('Unknown split name.')

            sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        elif vid_ids is not None:
            sequence_list = [c + '-' + str(v) for c in self.class_list for v in vid_ids]
        else:
            raise ValueError('Set either split_name or vid_ids.')
        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name
            if seq_name.isdigit():
                class_name = seq_name.rstrip('0123456789')
            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]
        return seq_per_class

    def get_name(self):
        return 'otb99_lang'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=",|\s+", header=None, dtype=np.float32, na_filter=False,
                             engine='python').values
        return torch.tensor(gt)

    def _read_nlp(self, seq_name):
        nlp_file = os.path.join(self.root, "OTB_query_" + self.split, seq_name + ".txt")
        nlp = pandas.read_csv(nlp_file, dtype=str, header=None, low_memory=False).values
        return nlp[0][0]

    def _read_target_visible(self, seq_path):

        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
        with open(out_of_view_file, 'r') as f:
            out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

        target_visible = ~occlusion & ~out_of_view

        return target_visible

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, "OTB_videos", seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = 1 & valid.byte()
        output = {'bbox': bbox, 'valid': valid, 'visible': visible}
        if self.multi_modal_language:
            nlp = self._read_nlp(self.sequence_list[seq_id])
            output['nlp'] = nlp
        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'nlp': nlp}

    def get_sequence_nlp(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        nlp = self._read_nlp(seq_path)
        return nlp

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:04}.jpg'.format(frame_id + 1))  

    def _get_frame(self, seq_path, frame_id):
        frame = self.image_loader(self._get_frame_path(seq_path, frame_id))
        if self.multi_modal_vision:
            frame = np.concatenate((frame, frame), axis=-1)
        return frame

    def _get_class(self, seq_path):
        raw_class = seq_path.split('/')[-1].rstrip('0123456789')
        return raw_class

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key == 'nlp':
                anno_frames[key] = [value for _ in frame_ids]
            else:
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def get_path(self, seq_id, frame_ids):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame_path(seq_path, f_id) for f_id in frame_ids]
        return frame_list
