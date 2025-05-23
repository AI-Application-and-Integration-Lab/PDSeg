import torch
import os
import torchvision.transforms as transforms
from transforms import *

from skimage import io
from torch.utils.data import Dataset

import PIL
import numpy as np
import pandas as pd

class Dataset_train(Dataset):
    def __init__(self, dataset_size, device):
        super(Dataset_train, self).__init__()
        file = pd.read_csv("data/glas/Grade.csv")
        file = file[file['name'].str.contains('train')]
        self.root = 'data/glas'
        self.data = file.iloc[:, 0].tolist() # name
        self.label = file.iloc[:, 2].tolist() # grade
        self.label = [1 if ("malignant" in l) else 0 for l in self.label]
        
        self.device = device
        self.size = dataset_size
        # self.transforms = transforms.Compose([
        #     transforms.Resize(self.size),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        #     ], p=0.8),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        #     transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1)
        #     # transforms.Normalize(mean=[164.7261, 129.4018, 176.4253], std=[43.0450, 49.9314, 32.2143])
        #                ])
        # self.transforms_test = transforms.Compose([
        #     transforms.Resize(self.size)
        # ])
        
        self.transforms = Compose([
            Resize(self.size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor()
        ])

    def __getitem__(self, index):

        image, img_show = self.read(self.root, self.data[index])
        label = torch.ones(1) * self.label[index]
            
        return image, label, img_show

    def __len__(self):
        return len(self.data)

    def read(self, path, name):
        img = PIL.Image.open(os.path.join(path, name + ".bmp"))
        # img = torch.from_numpy(img).float().permute(2, 0, 1)
        img, img_show = self.transforms(img, img.copy())
        return img, img_show


class Dataset_valid(Dataset):
    def __init__(self, dataset_size, device):
        super(Dataset_valid, self).__init__()
        file = pd.read_csv("data/glas/Grade.csv")
        file = file[file['name'].str.contains('test')]
        self.root = 'data/glas'
        self.data = file.iloc[:, 0].tolist() # name
        self.label = file.iloc[:, 2].tolist() # grade
        self.label = [1 if ("malignant" in l) else 0 for l in self.label]       
        
        self.device = device
        self.size = dataset_size
        # self.transforms_test = transforms.Compose([
        #     transforms.Resize(self.size),
        #     # transforms.Normalize(mean=[164.7261, 129.4018, 176.4253], std=[43.0450, 49.9314, 32.2143])
        #                 ])
        # self.transforms_grdth = transforms.Compose([
        #     transforms.Resize(self.size)
        #                 ])
        
        self.transforms_test = Compose([
            Resize(self.size),
            ToTensor()
        ])
        self.transforms_grdth = Compose([
            Resize(self.size),
            ToTensor()
        ])

    def __getitem__(self, index):
        
        if self.label[index]:
            image = self.read(self.root, self.data[index], 'test')
            grdth = self.read(self.root, self.data[index] + '_anno', 'grdth')
        else:
            image = self.read(self.root, self.data[index], 'test')
            grdth = torch.zeros(1, self.size[0], self.size[1])

        return image, grdth

    def __len__(self):
        return len(self.data)

    def read(self, path, name, norm=None):
        img = PIL.Image.open(os.path.join(path, name + ".bmp"))

        if norm == 'test':
            # img = torch.from_numpy(img).float().permute(2, 0, 1)
            img = self.transforms_test(img)

        elif norm == 'grdth':
            # if len(img.shape) > 2:
            #     img = img[:, :, 0]
            # img = torch.from_numpy(img).float().unsqueeze(0)
            img = self.transforms_grdth(img)
            img = (img > 0) + 0

        return img


class Dataset_test(Dataset):
    def __init__(self, dataset_size, device):
        super(Dataset_test, self).__init__()
        file = pd.read_csv("data/glas/Grade.csv")
        file = file[file['name'].str.contains('test')]
        self.root = 'data/glas'
        self.data = file.iloc[:, 0].tolist() # name
        self.label = file.iloc[:, 2].tolist() # grade
        self.label = [1 if ("malignant" in l) else 0 for l in self.label]       
        
        self.device = device
        self.size = dataset_size
        # self.transforms_test = transforms.Compose([
        #     transforms.Resize(self.size),
        #     # transforms.Normalize(mean=[164.7261, 129.4018, 176.4253], std=[43.0450, 49.9314, 32.2143])
        #                 ])
        # self.transforms_grdth = transforms.Compose([
        #     transforms.Resize(self.size)
        #                 ])
        
        self.transforms_test = Compose([
            Resize(self.size),
            ToTensor()
        ])
        self.transforms_grdth = Compose([
            Resize(self.size),
            ToTensor()
        ])

    def __getitem__(self, index):

        if self.label[index]:
            image = self.read(self.root, self.data[index], 'test')
            label = self.read(self.root, self.data[index] + '_anno', 'grdth')
            image_show = self.read(self.root, self.data[index])
        else:
            image = self.read(self.root, self.data[index], 'test')
            label = torch.zeros(self.size)
            image_show = self.read(self.root, self.data[index])

        return image, label, image_show

    def __len__(self):
        return len(self.data)

    def read(self, path, name, norm=None):
        img = PIL.Image.open(os.path.join(path, name + ".bmp"))

        if norm == 'test':
            # img = torch.from_numpy(img).float().permute(2, 0, 1)
            img = self.transforms_test(img)

        elif norm == 'grdth':
            # if len(img.shape) > 2:
            #     img = img[:, :, 0]
            # img = torch.from_numpy(img).float().unsqueeze(0)
            img = self.transforms_grdth(img)
            img = (img > 0) + 0
        else:
            img = torch.from_numpy(np.array(img)).float()

        return img
    

