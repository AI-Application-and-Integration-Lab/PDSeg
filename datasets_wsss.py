import torch
import os
import torchvision.transforms as transforms
from transforms import *

from skimage import io
from torch.utils.data import Dataset

import PIL
import numpy as np
import pandas as pd


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=True, fill_value=255):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.fill_value = fill_value
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl=None):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        
        if lbl is None:
            if self.padding > 0:
                img = F.pad(img, self.padding)
            # pad the width if needed
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            # pad the height if needed
            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))

            i, j, h, w = self.get_params(img, self.size)

            return F.crop(img, i, j, h, w)

        else:
            if isinstance(lbl, torch.Tensor):
                lbl_size = lbl.size()[-1], lbl.size()[-2]
                fill_value = 0
            else:
                lbl_size = lbl.size
                fill_value = self.fill_value
            
            assert img.size == lbl_size, 'size of img and lbl should be the same. %s, %s' % (img.size, lbl_size)
            
            
            if self.padding > 0:
                img = F.pad(img, self.padding)
                lbl = F.pad(lbl, self.padding)

            # pad the width if needed
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
                lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl_size[0]) / 2), fill=fill_value)

            # pad the height if needed
            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
                lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl_size[1]) / 2), fill=fill_value)

            i, j, h, w = self.get_params(img, self.size)

            return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class Dataset_train(Dataset):
    def __init__(self, dataset_size, device, dataset, test_on_train, stage):
        super(Dataset_train, self).__init__()
        
        self.root = f'./data/{dataset}/training'
        self.data = sorted(os.listdir(self.root)) # name
        if dataset[:4] == 'LUAD':
            self.label = [file[:-4].split(']')[0].split('[')[-1].split(' ') for file in self.data]
        elif dataset[:4] == 'BCSS':
            self.label = [[x for x in file[:-4].split(']')[0].split('[')[-1]] for file in self.data]
        elif dataset == 'WSSS4LUAD':
            self.label = [file[:-4].split(']')[0].split('[')[-1].split(', ') for file in self.data]
        else:
            raise NotImplementedError
        
        self.device = device
        self.size = dataset_size
        self.stage = stage
        self.dataset = dataset

        if test_on_train:
            self.transforms = Compose([
                ToTensor(),
                Normalize()
            ])
        else:
            self.transforms = Compose([
                RandomCrop(self.size),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomGaussianBlur(),
                ToTensor(),
                Normalize()
            ])
    
        print('Data Distribution: ', np.array([d for d in self.label]).astype(int).sum(axis=0))

    def __getitem__(self, index):
        image, img_show = self.read(self.root, self.data[index])
        if self.stage == 1:
            if self.dataset=='WSSS4LUAD':
                label = torch.Tensor([int(self.label[index][0]),int(self.label[index][1]),int(self.label[index][2])])
            else:
                label = torch.Tensor([int(self.label[index][0]),int(self.label[index][1]),int(self.label[index][2]),int(self.label[index][3])])

        return image, label, {"img_show": img_show, "file_name": self.data[index]}

    def __len__(self):
        return len(self.data)

    def read(self, path, name):
        img = PIL.Image.open(os.path.join(path, name))
        img, img_show = self.transforms(img, img.copy())
        return img, img_show

class Dataset_valid(Dataset):
    def __init__(self, dataset_size, device, dataset):
        super(Dataset_valid, self).__init__()
        self.root = f'./data/{dataset}/test/'
        self.data = sorted(os.listdir(self.root + 'img')) # name   
        
        self.device = device
        self.size = dataset_size
        
        self.transforms_test = Compose([
            ToTensor(),
            Normalize()
        ])
        self.transforms_grdth = Compose([
            ToTensor()
        ])

    def __getitem__(self, index):
        
        image = self.read(self.root, 'img/' + self.data[index], 'test')
        grdth = self.read(self.root, 'mask/' + self.data[index], 'grdth') * 255

        return image, grdth

    def __len__(self):
        return len(self.data)

    def read(self, path, name, norm=None):
        img = PIL.Image.open(os.path.join(path, name))

        if norm == 'test':
            img = self.transforms_test(img)

        elif norm == 'grdth':
            img = self.transforms_grdth(img)

        return img

class Dataset_test(Dataset):
    def __init__(self, dataset_size, device, dataset):
        super(Dataset_test, self).__init__()
        self.root = f'./data/{dataset}/test/'
        self.data = sorted(os.listdir(self.root + 'img')) # name  
        
        self.device = device
        self.size = dataset_size
        
        self.transforms_test = Compose([
            ToTensor(),
            Normalize()
        ])
        self.transforms_grdth = Compose([
            ToTensor()
        ])

    def __getitem__(self, index):

        image = self.read(self.root, 'img/' + self.data[index], 'test')
        label = self.read(self.root, 'mask/' + self.data[index], 'grdth') * 255
        image_show = self.read(self.root, 'img/' + self.data[index])

        return image, label, image_show

    def __len__(self):
        return len(self.data)

    def read(self, path, name, norm=None):
        img = PIL.Image.open(os.path.join(path, name))

        if norm == 'test':
            img = self.transforms_test(img)

        elif norm == 'grdth':
            img = self.transforms_grdth(img)
        else:
            img = torch.from_numpy(np.array(img)).float()

        return img
    
class Dataset_test_inf(Dataset):
    def __init__(self, dataset_size, device, data_path):
        super(Dataset_test_inf, self).__init__()
        self.data_path = data_path
        self.data = sorted(os.listdir(data_path)) # name  
        self.device = device
        self.size = dataset_size
        
        self.transforms_test = Compose([
            ToTensor(),
            Normalize()
        ])
        self.transforms_grdth = Compose([
            ToTensor()
        ])

    def __getitem__(self, index):
        image = self.read(self.data_path, self.data[index], 'test')
        return image
    
    def __len__(self):
        return len(self.data)

    def read(self, path, name, norm=None):
        img = PIL.Image.open(os.path.join(path, name))
        if norm == 'test':
            img = self.transforms_test(img)

        elif norm == 'grdth':
            img = self.transforms_grdth(img)
        else:
            img = torch.from_numpy(np.array(img)).float()

        return img
