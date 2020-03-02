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


class DataLoaderDANN(Dataset):
    def __init__(self, args, mode="train"):

        super(DataLoaderDANN, self).__init__()

        self.mode = mode
        self.data_dir = args.data_dir
        self.attr_list = pd.read_csv(os.path.join(self.data_dir, 'digits', 'train.csv'))


        self.transform = transforms.Compose([
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD),


        ])

        if mode =="train":


            self.img_files1 = glob.glob(os.path.join(self.data_dir, 'digits', 'mnistm', 'train', '*.png'))
            self.img_files2 = glob.glob(os.path.join(self.data_dir, 'digits', 'svhn', 'train','*.png'))
            self.attr_list1 = pd.read_csv(os.path.join(self.data_dir, 'digits', 'mnistm', 'train.csv'))
            self.attr_list2 = pd.read_csv(os.path.join(self.data_dir, 'digits', 'svhn', 'train.csv'))

            # self.mask_files = glob.glob(os.path.join(self.data_dir, 'train', 'seg', '*.png'))

        elif self.mode == 'val' or self.mode == 'test':

            self.img_files1 = glob.glob(os.path.join(self.data_dir, 'digits', 'mnistm', 'test', '*.png'))
            self.img_files2 = glob.glob(os.path.join(self.data_dir, 'digits', 'svhn', 'test', '*.png'))
            self.attr_list1 = pd.read_csv(os.path.join(self.data_dir, 'digits', 'mnistm', 'test.csv'))
            self.attr_list2 = pd.read_csv(os.path.join(self.data_dir, 'digits', 'svhn', 'test.csv'))

    def __getitem__(self, index):
            img_path1 = self.img_files[index]
            img_path2 = self.img_files[index]
            filename1 = os.path.basename(self.img_files1[index])
            filename2 = os.path.basename(self.img_files2[index])
            data = self.transform(np.array(Image.open(img_path).convert('RGB')))
            attrs_img = self.attr_list.loc[self.attr_list['image_name'] == filename]
            attr = attrs_img['Smiling'].to_numpy()
            attr = torch.FloatTensor(attr)
            return data, filename, attr

    def __len__(self):
        return len(self.img_files)

