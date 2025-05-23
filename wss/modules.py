from matplotlib.pyplot import get
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import torch.nn as nn

from functools import partial
import math
import numpy as np

#
# Helper modules
#
class LocalAffinity(nn.Module):

    def __init__(self, dilations=[1]):
        super(LocalAffinity, self).__init__()
        self.dilations = dilations
        weight = self._init_aff()
        self.register_buffer('kernel', weight)

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3)

        for i in range(weight.size(0)):
            weight[i, 0, 1, 1] = 1

        weight[0, 0, 0, 0] = -1
        weight[1, 0, 0, 1] = -1
        weight[2, 0, 0, 2] = -1

        weight[3, 0, 1, 0] = -1
        weight[4, 0, 1, 2] = -1

        weight[5, 0, 2, 0] = -1
        weight[6, 0, 2, 1] = -1
        weight[7, 0, 2, 2] = -1

        self.weight_check = weight.clone()

        return weight

    def forward(self, x):

        self.weight_check = self.weight_check.type_as(x)
        assert torch.all(self.weight_check.eq(self.kernel))

        B, K, H, W = x.size()
        x = x.view(B * K, 1, H, W)

        x_affs = []
        for d in self.dilations:
            x_pad = F.pad(x, [d] * 4, mode='replicate')
            x_aff = F.conv2d(x_pad, self.kernel, dilation=d)
            x_affs.append(x_aff)

        x_aff = torch.cat(x_affs, 1)
        return x_aff.view(B, K, -1, H, W)


class LocalAffinityCopy(LocalAffinity):

    def _init_aff(self):
        # initialising the shift kernel
        weight = torch.zeros(8, 1, 3, 3)

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 2] = 1

        weight[5, 0, 2, 0] = 1
        weight[6, 0, 2, 1] = 1
        weight[7, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight


class LocalStDev(LocalAffinity):

    def _init_aff(self):
        weight = torch.zeros(9, 1, 3, 3)
        weight.zero_()

        weight[0, 0, 0, 0] = 1
        weight[1, 0, 0, 1] = 1
        weight[2, 0, 0, 2] = 1

        weight[3, 0, 1, 0] = 1
        weight[4, 0, 1, 1] = 1
        weight[5, 0, 1, 2] = 1

        weight[6, 0, 2, 0] = 1
        weight[7, 0, 2, 1] = 1
        weight[8, 0, 2, 2] = 1

        self.weight_check = weight.clone()
        return weight

    def forward(self, x):
        # returns (B,K,P,H,W), where P is the number
        # of locations
        x = super(LocalStDev, self).forward(x)

        return x.std(2, keepdim=True)


class LocalAffinityAbs(LocalAffinity):

    def forward(self, x):
        x = super(LocalAffinityAbs, self).forward(x)
        return torch.abs(x)


# PAMR module
class PAMR(nn.Module):

    def __init__(self, num_iter=10, dilations=[1, 2, 4, 8, 12, 24]):
        super(PAMR, self).__init__()

        self.num_iter = num_iter
        self.aff_x = LocalAffinityAbs(dilations)
        self.aff_m = LocalAffinityCopy(dilations)
        self.aff_std = LocalStDev(dilations)

    def forward(self, x, mask):
        mask = F.interpolate(mask, size=x.size()[-2:], mode="bilinear", align_corners=True)

        # x: [BxKxHxW]
        # mask: [BxCxHxW]
        B, K, H, W = x.size()
        _, C, _, _ = mask.size()

        x_std = self.aff_std(x)

        x = -self.aff_x(x) / (1e-8 + 0.1 * x_std)
        x = x.mean(1, keepdim=True)
        x = F.softmax(x, 2)

        for _ in range(self.num_iter):
            m = self.aff_m(mask)  # [BxCxPxHxW]
            mask = (m * x).sum(2)

        # xvals: [BxCxHxW]
        return mask
    
###
#local pixel refinement
###
def get_kernel():
    
    weight = torch.zeros(8, 1, 3, 3)
    weight[0, 0, 0, 0] = 1
    weight[1, 0, 0, 1] = 1
    weight[2, 0, 0, 2] = 1

    weight[3, 0, 1, 0] = 1
    weight[4, 0, 1, 2] = 1

    weight[5, 0, 2, 0] = 1
    weight[6, 0, 2, 1] = 1
    weight[7, 0, 2, 2] = 1

    return weight

class PAR(nn.Module):

    def __init__(self, dilations, num_iter,):
        super().__init__()
        self.dilations = dilations
        self.num_iter = num_iter
        kernel = get_kernel()
        self.register_buffer('kernel', kernel)
        self.pos = self.get_pos()
        self.dim = 2
        self.w1 = 0.3
        self.w2 = 0.01

    def get_dilated_neighbors(self, x):

        b, c, h, w = x.shape
        x_aff = []
        for d in self.dilations:
            _x_pad = F.pad(x, [d]*4, mode='replicate', value=0)
            _x_pad = _x_pad.reshape(b*c, -1, _x_pad.shape[-2], _x_pad.shape[-1])
            _x = F.conv2d(_x_pad, self.kernel, dilation=d).view(b, c, -1, h, w)
            x_aff.append(_x)
 
        return torch.cat(x_aff, dim=2)

    def get_pos(self):
        pos_xy = []

        ker = torch.ones(1, 1, 8, 1, 1)
        ker[0, 0, 0, 0, 0] = np.sqrt(2)
        ker[0, 0, 2, 0, 0] = np.sqrt(2)
        ker[0, 0, 5, 0, 0] = np.sqrt(2)
        ker[0, 0, 7, 0, 0] = np.sqrt(2)
        
        for d in self.dilations:
            pos_xy.append(ker*d)
        return torch.cat(pos_xy, dim=2)

    def forward(self, imgs, masks):

        masks = F.interpolate(masks, size=imgs.size()[-2:], mode="bilinear", align_corners=True)

        b, c, h, w = imgs.shape
        _imgs = self.get_dilated_neighbors(imgs)
        _pos = self.pos.to(_imgs.device)

        _imgs_rep = imgs.unsqueeze(self.dim).repeat(1,1,_imgs.shape[self.dim],1,1)
        _pos_rep = _pos.repeat(b, 1, 1, h, w)

        _imgs_abs = torch.abs(_imgs - _imgs_rep)
        _imgs_std = torch.std(_imgs, dim=self.dim, keepdim=True)
        _pos_std = torch.std(_pos_rep, dim=self.dim, keepdim=True)

        aff = -(_imgs_abs / (_imgs_std + 1e-8) / self.w1)**2
        aff = aff.mean(dim=1, keepdim=True)

        pos_aff = -(_pos_rep / (_pos_std + 1e-8) / self.w1)**2
        #pos_aff = pos_aff.mean(dim=1, keepdim=True)

        aff = F.softmax(aff, dim=2) + self.w2 * F.softmax(pos_aff, dim=2)

        for _ in range(self.num_iter):
            _masks = self.get_dilated_neighbors(masks)
            masks = (_masks * aff).sum(2)

        return masks
