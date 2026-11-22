import os
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import torch
import random
import torchvision.datasets as datasets
from collections import OrderedDict
from lib.train.admin import env_settings


class Imagenet1k(BaseVideoDataset):


    def __init__(self, root=None, image_loader=jpeg4py_loader, split="train"):

        root = env_settings().imagenet1k_dir if root is None else root
        super().__init__('imagenet1k', root, image_loader)

        self.dataset = datasets.ImageFolder(os.path.join(root, 'train'), loader=image_loader)


    def is_video_sequence(self):
        return False


    def get_name(self):
        return 'imagenet1k'


    def get_num_sequences(self):
        return len(self.dataset.samples)


    def get_sequence_info(self, seq_id):

        valid = torch.tensor([True])
        visible = valid.clone().byte()
        return {'bbox': None, 'mask': None, 'valid': valid, 'visible': visible}


    def _get_frames(self, seq_id):
        img, target = self.dataset.__getitem__(seq_id)
        return img


    def get_frames(self, seq_id=None, frame_ids=None, anno=None):


        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]


        return frame_list, None, None

if __name__ == '__main__':
    data_root = './data/imagenet1k'
    dataset = Imagenet1k(data_root)
