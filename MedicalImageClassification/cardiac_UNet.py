import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dim=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        batchnorm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d

        self.double_conv = nn.Sequential(
            conv(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            batchnorm(mid_channels),
            nn.ReLU(inplace=True),
            conv(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            batchnorm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dim=2):
        super().__init__()
        pool = nn.MaxPool2d(2) if dim == 2 else nn.MaxPool3d(2)
        self.maxpool_conv = nn.Sequential(
            pool,
            DoubleConv(in_channels, out_channels, dim=dim)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dim=2, bilinear=True):
        super().__init__()
        self.dim = dim

        if bilinear:
            mode = 'bilinear' if dim == 2 else 'trilinear'
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dim=dim)
        else:
            conv_transpose = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
            self.up = conv_transpose(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dim=dim)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 处理可能的尺寸差异
        diff = [x2.size()[i] - x1.size()[i] for i in range(2, len(x2.shape))]
        pad = []
        for d in diff:
            pad.extend([d // 2, d - d // 2])

        # 反转pad顺序以适应F.pad的要求
        pad = pad[::-1]
        x1 = F.pad(x1, pad)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CardiacUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, dim=2, bilinear=True):
        super(CardiacUNet, self).__init__()
        self.dim = dim
        self.bilinear = bilinear
        base_channels = 32

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels, dim=dim)
        self.down1 = Down(base_channels, base_channels * 2, dim=dim)
        self.down2 = Down(base_channels * 2, base_channels * 4, dim=dim)
        self.down3 = Down(base_channels * 4, base_channels * 8, dim=dim)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, dim=dim)

        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, dim=dim, bilinear=bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, dim=dim, bilinear=bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, dim=dim, bilinear=bilinear)
        self.up4 = Up(base_channels * 2, base_channels, dim=dim, bilinear=bilinear)

        # Output
        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        self.outc = conv(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        original_size = x.shape[2:]

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)

        # 确保输出尺寸匹配输入
        if logits.shape[2:] != original_size:
            mode = 'bilinear' if self.dim == 2 else 'trilinear'
            logits = F.interpolate(logits, size=original_size, mode=mode, align_corners=True)

        return logits
