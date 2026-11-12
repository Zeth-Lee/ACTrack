import os
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import torch
from collections import OrderedDict
from lib.train.admin import env_settings
from lib.utils.lmdb_utils import decode_img, decode_json


def get_target_to_image_ratio(seq):
    anno = torch.Tensor(seq['anno'])
    img_sz = torch.Tensor(seq['image_size'])
    return (anno[0, 2:4].prod() / (img_sz.prod())).sqrt()


class ImagenetVID_lmdb(BaseVideoDataset):

    def __init__(self, root=None, image_loader=jpeg4py_loader, min_length=0, max_target_area=1):

        root = env_settings().imagenet_dir if root is None else root
        super().__init__("imagenetvid_lmdb", root, image_loader)

        sequence_list_dict = decode_json(root, "cache.json")
        self.sequence_list = sequence_list_dict


        self.sequence_list = [x for x in self.sequence_list if len(x['anno']) >= min_length and
                              get_target_to_image_ratio(x) < max_target_area]

    def get_name(self):
        return 'imagenetvid_lmdb'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        bb_anno = torch.Tensor(self.sequence_list[seq_id]['anno'])
        valid = (bb_anno[:, 2] > 0) & (bb_anno[:, 3] > 0)
        visible = torch.ByteTensor(self.sequence_list[seq_id]['target_visible']) & valid.byte()
        return {'bbox': bb_anno, 'valid': valid, 'visible': visible}

    def _get_frame(self, sequence, frame_id):
        set_name = 'ILSVRC2015_VID_train_{:04d}'.format(sequence['set_id'])
        vid_name = 'ILSVRC2015_train_{:08d}'.format(sequence['vid_id'])
        frame_number = frame_id + sequence['start_frame']
        frame_path = os.path.join('Data', 'VID', 'train', set_name, vid_name,
                                  '{:06d}.JPEG'.format(frame_number))
        return decode_img(self.root, frame_path)

    def get_frames(self, seq_id, frame_ids, anno=None):
        sequence = self.sequence_list[seq_id]

        frame_list = [self._get_frame(sequence, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)


        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]


        object_meta = OrderedDict({'object_class': sequence['class_name'],
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta

    def get_annos(self, seq_id, frame_ids, anno=None):
        if anno is None:
            anno = self.get_sequence_info(seq_id)


        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        return anno_frames
