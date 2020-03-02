import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob

from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 225]


class DataLoaderACGAN(Dataset):
    def __init__(self, args, mode="train"):

        super(DataLoaderACGAN, self).__init__()

        self.mode = mode
        self.data_dir = args.data_dir
        self.attr_list = pd.read_csv(os.path.join(self.data_dir, 'face', 'train.csv'))


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
            attrs_img = self.attr_list.loc[self.attr_list['image_name'] == filename]
            attr = attrs_img['Smiling'].to_numpy()
            attr = torch.FloatTensor(attr)
            return data, filename, attr

    def __len__(self):
        return len(self.img_files)

