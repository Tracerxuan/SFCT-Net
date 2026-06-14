import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
import cv2

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class Allage_Dataset(Dataset):
    def __init__(self, alone_image_npy_path, alone_mask_npy_path, new_shape=(568, 426)):
        super().__init__()
        self.alone_image_npy_path = alone_image_npy_path
        self.alone_mask_npy_path = alone_mask_npy_path
        self.new_shape = new_shape

    def __len__(self):
        return len(os.listdir(self.alone_mask_npy_path))

    def __getitem__(self, index):
        alone_image_list = os.listdir(self.alone_image_npy_path)

        alone_image_file = alone_image_list[index]

        alone_image_file_order = alone_image_file.split('.')[0].split('-')[-1]
        alone_image_file_split = alone_image_file.split('.')[0].split('-')
        alone_image_file_split_num = len(alone_image_file_split)

        alone_image_file_base = ''
        for each_split in range(alone_image_file_split_num-1):
            alone_image_file_base = alone_image_file_base + alone_image_file_split[each_split] + '-'

        alone_image_file_1 = alone_image_file_base + str(int(alone_image_file_order) - 1) + '.npy'
        alone_image_file1 = alone_image_file_base + str(int(alone_image_file_order) + 1) + '.npy'
        if alone_image_file_1 not in alone_image_list:
            alone_image_file_1 = alone_image_file
        if alone_image_file1 not in alone_image_list:
            alone_image_file1 = alone_image_file

        alone_image_path = os.path.join(self.alone_image_npy_path, alone_image_file)
        alone_image_npy = np.load(alone_image_path)
        alone_image_npy = np.array(alone_image_npy, dtype=np.float32)
        alone_image_npy = alone_image_npy / 255
        alone_image_npy = alone_image_npy[..., np.newaxis]
        alone_image_npy = cv2.resize(alone_image_npy, self.new_shape)
        alone_image_npy = alone_image_npy[np.newaxis, ...]
        alone_image_npy = torch.from_numpy(alone_image_npy)

        alone_image_path_1 = os.path.join(self.alone_image_npy_path, alone_image_file_1)
        alone_image_npy_1 = np.load(alone_image_path_1)
        alone_image_npy_1 = np.array(alone_image_npy_1, dtype=np.float32)
        alone_image_npy_1 = alone_image_npy_1 / 255
        alone_image_npy_1 = alone_image_npy_1[..., np.newaxis]
        alone_image_npy_1 = cv2.resize(alone_image_npy_1, self.new_shape)
        alone_image_npy_1 = alone_image_npy_1[np.newaxis, ...]
        alone_image_npy_1 = torch.from_numpy(alone_image_npy_1)

        alone_image_path1 = os.path.join(self.alone_image_npy_path, alone_image_file1)
        alone_image_npy1 = np.load(alone_image_path1)
        alone_image_npy1 = np.array(alone_image_npy1, dtype=np.float32)
        alone_image_npy1 = alone_image_npy1 / 255
        alone_image_npy1 = alone_image_npy1[..., np.newaxis]
        alone_image_npy1 = cv2.resize(alone_image_npy1, self.new_shape)
        alone_image_npy1 = alone_image_npy1[np.newaxis, ...]
        alone_image_npy1 = torch.from_numpy(alone_image_npy1)

        mask_path = os.path.join(self.alone_mask_npy_path, alone_image_file)
        mask_npy = np.load(mask_path)
        mask_npy = mask_npy / 255
        mask_npy = mask_npy[..., np.newaxis]
        mask_npy = cv2.resize(mask_npy, self.new_shape, interpolation=cv2.INTER_NEAREST)
        mask_npy = np.array(mask_npy, dtype=np.int64)
        mask_npy = torch.from_numpy(mask_npy)
        mask_npy = mask_npy.unsqueeze(0)
        mask_npy = mask_npy.float()

        return [alone_image_npy_1, alone_image_npy, alone_image_npy1], mask_npy, alone_image_file
