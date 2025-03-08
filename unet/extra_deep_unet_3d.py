import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
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

class Down3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handling different sizes
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2,
                       diff_z // 2, diff_z - diff_z // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ExtraDeep3DUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ExtraDeep3DUNet, self).__init__()
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

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

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
        x = self.outc(x)
        
        return x

from torch.utils.checkpoint import checkpoint

# Optional: Gradient Checkpointing wrapper
def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for memory efficiency"""
    for module in model.modules():
        if isinstance(module, (DoubleConv3D, Down3D, Up3D)):
            module.forward = checkpoint.checkpoint(module.forward)


######################################################################
# checkpointing
######################################################################

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint

# class DoubleConv3D(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class Down3D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool3d(2),
#             DoubleConv3D(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)

# class Up3D(nn.Module):
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
        
#         diff_z = x2.size()[2] - x1.size()[2]
#         diff_y = x2.size()[3] - x1.size()[3]
#         diff_x = x2.size()[4] - x1.size()[4]

#         x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
#                        diff_y // 2, diff_y - diff_y // 2,
#                        diff_z // 2, diff_z - diff_z // 2])

#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class OutConv3D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv3D, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)

# class ExtraDeep3DUNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True, use_checkpointing=True):
#         super(ExtraDeep3DUNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.use_checkpointing = use_checkpointing

#         self.inc = DoubleConv3D(n_channels, 64)
#         self.down1 = Down3D(64, 128)
#         self.down2 = Down3D(128, 256)
#         self.down3 = Down3D(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down3D(512, 1024)
#         self.down5 = Down3D(1024, 2048 // factor)
        
#         self.up1 = Up3D(2048, 1024 // factor, bilinear)
#         self.up2 = Up3D(1024, 512 // factor, bilinear)
#         self.up3 = Up3D(512, 256 // factor, bilinear)
#         self.up4 = Up3D(256, 128 // factor, bilinear)
#         self.up5 = Up3D(128, 64, bilinear)
#         self.outc = OutConv3D(64, n_classes)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv3d):
#             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm3d):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x1 = self.inc(x)
        
#         if self.use_checkpointing:
#             x2 = checkpoint(self.down1, x1)
#             x3 = checkpoint(self.down2, x2)
#             x4 = checkpoint(self.down3, x3)
#             x5 = checkpoint(self.down4, x4)
#             x6 = checkpoint(self.down5, x5)
            
#             x = checkpoint(lambda x6, x5: self.up1(x6, x5), x6, x5)
#             x = checkpoint(lambda x, x4: self.up2(x, x4), x, x4)
#             x = checkpoint(lambda x, x3: self.up3(x, x3), x, x3)
#             x = checkpoint(lambda x, x2: self.up4(x, x2), x, x2)
#             x = checkpoint(lambda x, x1: self.up5(x, x1), x, x1)
#         else:
#             x2 = self.down1(x1)
#             x3 = self.down2(x2)
#             x4 = self.down3(x3)
#             x5 = self.down4(x4)
#             x6 = self.down5(x5)
            
#             x = self.up1(x6, x5)
#             x = self.up2(x, x4)
#             x = self.up3(x, x3)
#             x = self.up4(x, x2)
#             x = self.up5(x, x1)
            
#         x = self.outc(x)
#         return x

# model = ExtraDeep3DUNet(n_channels=1, n_classes=2, use_checkpointing=True)

######################################################################
# a little simpler model (got rid of 1024 and 2048 channels)
######################################################################

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DoubleConv3D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class Down3D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool3d(2),
#             DoubleConv3D(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)

# class Up3D(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True) if bilinear \
#             else nn.ConvTranspose3d(in_channels//2, in_channels//2, kernel_size=2, stride=2)
#         self.conv = DoubleConv3D(in_channels, out_channels)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
        
#         diff_z = x2.size()[2] - x1.size()[2]
#         diff_y = x2.size()[3] - x1.size()[3]
#         diff_x = x2.size()[4] - x1.size()[4]

#         x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
#                        diff_y // 2, diff_y - diff_y // 2,
#                        diff_z // 2, diff_z - diff_z // 2])

#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class OutConv3D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv3D, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)

# class ExtraDeep3DUNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(ExtraDeep3DUNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         # Encoder path
#         self.inc = DoubleConv3D(n_channels, 32)
#         self.down1 = Down3D(32, 64)
#         self.down2 = Down3D(64, 128)
#         self.down3 = Down3D(128, 256)
        
#         # Decoder path
#         self.up3 = Up3D(384, 128)  # 256 + 128 input channels
#         self.up4 = Up3D(192, 64)   # 128 + 64 input channels
#         self.up5 = Up3D(96, 32)    # 64 + 32 input channels
#         self.outc = OutConv3D(32, n_classes)

#     def forward(self, x):
#         # Encoder
#         x1 = self.inc(x)      # 32
#         x2 = self.down1(x1)   # 64
#         x3 = self.down2(x2)   # 128
#         x4 = self.down3(x3)   # 256
        
#         # Decoder
#         x = self.up3(x4, x3)  # 128
#         x = self.up4(x, x2)   # 64
#         x = self.up5(x, x1)   # 32
#         x = self.outc(x)
        
#         return x