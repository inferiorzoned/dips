import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """(3D convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# class Down3D(nn.Module):
#     """Downscaling with maxpool then double conv"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool3d(2),
#             DoubleConv3D(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)

# class Up3D(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#             self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv3D(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CDHW
#         diffD = x2.size()[2] - x1.size()[2]
#         diffH = x2.size()[3] - x1.size()[3]
#         diffW = x2.size()[4] - x1.size()[4]

#         x1 = F.pad(x1, [
#             diffW // 2, diffW - diffW // 2,  # Pad width
#             diffH // 2, diffH - diffH // 2,  # Pad height
#             diffD // 2, diffD - diffD // 2   # Pad depth
#         ])

#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # Use kernel_size=(1,2,2) to only pool in height and width dimensions
            nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            # Adjust upsampling to match the modified downsampling
            self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            # Adjust ConvTranspose3d to match the modified downsampling
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, 
                                       kernel_size=(1,2,2), stride=(1,2,2))
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CDHW
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [
            diffW // 2, diffW - diffW // 2,  # Pad width
            diffH // 2, diffH - diffH // 2,  # Pad height
            diffD // 2, diffD - diffD // 2   # Pad depth
        ])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ExtraDeepUNet3D_knee(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ExtraDeepUNet3D_knee, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(512, 1024)
        self.down5 = Down3D(1024, 2048 // factor)
        self.up1 = Up3D(2048, 1024 // factor, bilinear)
        self.up2 = Up3D(1024, 512 // factor, bilinear)
        self.up3 = Up3D(512, 256 // factor, bilinear)
        self.up4 = Up3D(256, 128 // factor, bilinear)
        self.up5 = Up3D(128, 64, bilinear)
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits