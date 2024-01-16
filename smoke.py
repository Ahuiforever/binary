# !/usr/bin/env/binary python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 15:42
# @Author  : Ahuiforever
# @File    : smoke.py.py
# @Software: PyCharm


# import cv2
import os

import cv2
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import numpy as np


class Smoke(Dataset):
    """
    D:/WORK/TRY7/450
    ├─images
    │  ├─test
    │  ├─train
    │  └─val
    └─labels
        ├─test
        ├─train
        └─val
    root_dir = 'D:/WORK/TRY7/450/images/train'
    """

    def __init__(self, root_dir: str, transform: any = None, show: bool = False):
        self.transform = transform
        self.root_dir = root_dir
        self.image_paths = os.listdir(root_dir)
        self.show = show

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        # original_image = Image.open(img_name)
        # original_image = cv2.imread(img_name)
        # image = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
        # image = Image.fromarray(original_image)  # .convert("RGB")
        image = Image.open(img_name)
        image = np.array(image).astype(np.float32)
        image = np.expand_dims(image, axis=-1)
        # image = cv2.imread(img_name)
        image = torch.from_numpy(image).permute(2, 0, 1)/65535  # ? Replace the transforms.ToTensor()
        if self.transform:
            image = self.transform(image)

        label_name = img_name.replace("images", "labels").replace("tif", "txt")
        # % _old:'tif' -> 'jpg'
        message = ''

        try:
            with open(label_name, "r") as label_file:
                try:
                    # label = torch.Tensor(int(label_file.read(1)))
                    label_file.read(1)
                    label = 1.
                except ValueError:
                    label = 0.
                    message = 'Empty in '
                    # label = torch.Tensor(0)
                # label = label_file.readline().split()[0]
        except FileNotFoundError:
            label = 0.
            message = f'Not Found '
            # label = torch.Tensor(0)
            # print(f"Label {label_name} missed.")

        # plt.imshow(np.transpose(image, (1, 2, 0)))
        # plt.title(img_name +
        #           f"\n{message}" +
        #           f"{label_name} -- {label}")
        # plt.show()

        return image, torch.tensor(label)
