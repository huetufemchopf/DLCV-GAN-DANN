import os
import json
import torch
import scipy.misc
import numpy as np

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
import glob
import random

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 225]
# randomangle = random.rand(0,1)
# print
# exit()


class DataLoaderSegmentation(Dataset):
    def __init__(self, args, mode="train"):

        super(DataLoaderSegmentation, self).__init__()

        self.mode = mode
        self.data_dir = args.data_dir


        self.transform = transforms.Compose([
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD),


        ])

        if mode =="train":


            self.img_files = glob.glob(os.path.join(self.data_dir, 'face', 'train', '*.png'))
            # self.mask_files = glob.glob(os.path.join(self.data_dir, 'train', 'seg', '*.png'))

        elif self.mode == 'val' or self.mode == 'test':

            self.img_files = glob.glob(os.path.join(self.data_dir, 'val', 'img', '*.png'))
            # self.mask_files = glob.glob(os.path.join(self.data_dir, 'val', 'seg', '*.png'))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            filename = os.path.basename(self.img_files[index])
            data = self.transform(np.array(Image.open(img_path).convert('RGB')))

            return data, filename

    def __len__(self):
        return len(self.img_files)

class DataLoaderBirds(Dataset):
    def __init__(self, args, mode="train"):

        super(DataLoaderBirds, self).__init__()

        self.mode = mode
        self.data_dir = args.data_dir


        self.transform = transforms.Compose([
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD),


        ])

        if mode =="train":


            self.img_files = glob.glob(os.path.join(self.data_dir, '*.png'))
            # self.mask_files = glob.glob(os.path.join(self.data_dir, 'train', 'seg', '*.png'))

        elif self.mode == 'val' or self.mode == 'test':

            self.img_files = glob.glob(os.path.join(self.data_dir, 'val', 'img', '*.png'))
            # self.mask_files = glob.glob(os.path.join(self.data_dir, 'val', 'seg', '*.png'))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            filename = os.path.basename(self.img_files[index])
            data = self.transform(np.array(Image.open(img_path).convert('RGB')))

            return data, filename

    def __len__(self):
        return len(self.img_files)