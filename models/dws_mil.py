import torch
import torch.nn as nn
import torchvision


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DWS_MIL(nn.Module):
    def __init__(self):
        super(DWS_MIL, self).__init__()
        self.stage1 = nn.Sequential(
            Conv(3, 64),
            Conv(64, 64)
        )
        self.stage2 = nn.Sequential(
            Conv(64, 128),
            Conv(128, 128)
        )
        self.stage3 = nn.Sequential(
            Conv(128, 256),
            Conv(256, 256),
            Conv(256, 256)
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            nn.Sigmoid()
        )

        self.w = [0.2, 0.35, 0.45]

    def pretrain(self, model, device):
        model_pre = torchvision.models.vgg16(pretrained=True).to(device)
        model.stage1[0].conv[0] = model_pre.features[0]
        model.stage1[1].conv[0] = model_pre.features[2]
        model.stage2[0].conv[0] = model_pre.features[5]
        model.stage2[1].conv[0] = model_pre.features[7]
        model.stage3[0].conv[0] = model_pre.features[10]
        model.stage3[1].conv[0] = model_pre.features[12]
        model.stage3[2].conv[0] = model_pre.features[14]

        return model

    def forward(self, x):
        x = self.stage1(x)
        x = self.pool(x)
        x1 = x

        x = self.stage2(x)
        x = self.pool(x)
        x2 = x

        x = self.stage3(x)
        x = self.pool(x)
        x3 = x

        x1 = self.decoder1(x1)
        x2 = self.decoder2(x2)
        x3 = self.decoder3(x3)

        x = self.w[0] * x1 + self.w[1] * x2 + self.w[2] * x3

        return x1, x2, x3, x
