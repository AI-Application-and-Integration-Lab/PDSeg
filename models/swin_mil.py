import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer import SwinTransformer
import os
# from mmcv.cnn import ConvModule


class Swin_MIL(nn.Module):
    def __init__(self, classes):
        super(Swin_MIL, self).__init__()
        self.backbone = SwinTransformer(depths=[2, 2, 6, 2], out_indices=(0, 1, 2))

        self.decoder1 = nn.Sequential(
            nn.Conv2d(96, classes, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(192, classes, 1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(384, classes, 1),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )
        
        # self.linear_fuse = nn.Sequential(
        #     nn.Conv2d(3, 1, 1),
        # )

        self.w = [0.3, 0.4, 0.3]

    def pretrain(self, model, device):
        for w in model.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.xavier_uniform_(w.weight)

        model.backbone.init_weights()
        model_dict = model.state_dict()
        pretrained_dict = torch.load('pretrained_models/swin_tiny_patch4_window7_224.pth')['model']
        # pretrained_dict = torch.load('pretrained_models/moby_deit_small_300ep_pretrained.pth')['model']
        new_dict = {'backbone.' + k: v for k, v in pretrained_dict.items() if 'backbone.' + k in model_dict.keys()}
        model_dict.update(new_dict)
        model.load_state_dict(model_dict, strict=True)

        return model

    def forward(self, x):
        x1_, x2_, x3_ = self.backbone(x)

        x1 = self.decoder1(x1_)
        x2 = self.decoder2(x2_)
        x3 = self.decoder3(x3_)

        x = self.w[0] * x1 + self.w[1] * x2 + self.w[2] * x3
        # x = self.linear_fuse(torch.cat([x1, x2, x3], dim=1))

        # return x1, x2, x3, x, x3_
        return x1, x2, x3, x, x3_
