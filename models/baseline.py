import torch
import torch.nn as nn


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


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1_1 = Conv(3, 64)
        self.conv1_2 = Conv(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2_1 = Conv(64, 128)
        self.conv2_2 = Conv(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3_1 = Conv(128, 256)
        self.conv3_2 = Conv(256, 256)
        self.conv3_3 = Conv(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv = nn.Conv2d(256, 1, 1)
        self.upsample = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

    def pretrain(self, model, device):
        model_pre = torchvision.models.vgg16(pretrained=True).to(device)
        model.conv1_1.conv[0] = model_pre.features[0]
        model.conv1_2.conv[0] = model_pre.features[2]
        model.conv2_1.conv[0] = model_pre.features[5]
        model.conv2_2.conv[0] = model_pre.features[7]
        model.conv3_1.conv[0] = model_pre.features[10]
        model.conv3_2.conv[0] = model_pre.features[12]
        model.conv3_3.conv[0] = model_pre.features[14]

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv(x)
        x = self.upsample(x)
        x = torch.sigmoid(x)

        return x
