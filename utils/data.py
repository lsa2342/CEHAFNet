import os
import random
from typing import Callable, Optional, Tuple

import cv2
import torch
from torch.utils import data
from torchvision.transforms import v2
from PIL import Image
# data loader
import glob
import torch
from skimage import io, transform, color
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torchvision import transforms, utils
from PIL import Image

tensor2pil = v2.ToPILImage()
pil2tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

from .transforms import ColorJitter, Rescale, RandomCrop, RandomRotate

def MaxMinNormalization(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())



class SalObjDataset(Dataset):
    def __init__(self, img_name_list, gt_name_list, transform=None):
        # self.root_dir = root_dir
        # self.image_name_list = glob.glob(image_dir+'*.png')
        # self.label_name_list = glob.glob(label_dir+'*.png')
        self.image_name_list = img_name_list
        self.label_name_list = gt_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):

        # image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
        # label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])

        image = io.imread(self.image_name_list[idx])

        if (0 == len(self.label_name_list)):
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        #print("len of label3")
        #print(len(label_3.shape))
        #print(label_3.shape)

        label = np.zeros(label_3.shape[0:2])
        if (3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif (2 == len(label_3.shape)):
            label = label_3

        if (3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(label.shape)):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        # #vertical flipping
        # # fliph = np.random.randn(1)
        # flipv = np.random.randn(1)
        #
        # if flipv>0:
        # 	image = image[::-1,:,:]
        # 	label = label[::-1,:,:]
        # #vertical flip

        sample = {'image': image, 'label': label}
        # sample = {'image': image, 'label': label, 'idx': idx}

        if self.transform:
            sample = self.transform(sample)

        image = sample['image'].type(torch.FloatTensor)
        label = sample['label'].type(torch.FloatTensor)
        sample = {'image': image, 'label': label}

        return sample


class test_dataset(Dataset):
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        # self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    img_dir = '/home/lsa/Shared/Dataset/ORSSD/train/image/'
    label_dir = '/home/lsa/Shared/Dataset/ORSSD/train/gt/'
    data_dir = "./data/ORSSD"

    # 自定义增强
    custom_transform  = transforms.Compose([
        ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5)),
        RandomRotate(90),
        Rescale((256, 256)),
        RandomCrop(224),
        transforms.ToTensor(),
    ])

    # 初始化数据集
    dataset = OrsiDataset(root_dir=data_dir, split="train", transform=custom_transform)

    # 加载样本
    sample = dataset[0]

    # 检查样本
    print("Image shape:", sample["image"].shape)
    print("Label shape:", sample["label"].shape)
    print("Image name:", sample["name"])
    print("Sample ID:", sample["id"])

