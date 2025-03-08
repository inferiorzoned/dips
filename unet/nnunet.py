import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvDropoutNormNonlin(nn.Module):
    """
    nnUNet's basic building block that implements conv + dropout + norm + nonlinearity
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, nonlin=nn.LeakyReLU,
                 dropout_p=0.1, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.nonlin = nonlin(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p, inplace=True) if dropout_p > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm = norm_layer(out_channels) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x


class StackedConvLayers(nn.Module):
    """
    nnUNet's stack of convolutional layers with residual connections
    """
    def __init__(self, in_channels, out_channels, num_convs=2, dropout_p=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            ConvDropoutNormNonlin(
                in_channels if i == 0 else out_channels,
                out_channels,
                dropout_p=dropout_p if i == 0 else 0
            ) for i in range(num_convs)
        ])
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        for conv in self.convs:
            x = conv(x)
        x = x + residual
        return x


class nnUNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_op=nn.MaxPool2d(2)):
        super().__init__()
        self.pool = pool_op
        self.conv_block = StackedConvLayers(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv_block(x)
        return x


class nnUNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = (nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear 
                  else nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
        self.conv_block = StackedConvLayers(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle potential size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)


class nnUNet(nn.Module):
    def __init__(self, in_channels, num_classes, base_features=32, max_features=512, num_pool=4, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # Initial conv
        self.input_block = StackedConvLayers(in_channels, base_features)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        current_features = base_features
        for i in range(num_pool):
            next_features = min(current_features * 2, max_features)
            self.down_blocks.append(nnUNetDownBlock(current_features, next_features))
            current_features = next_features

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i in range(num_pool):
            next_features = current_features // 2 if bilinear else current_features
            self.up_blocks.append(nnUNetUpBlock(current_features, next_features, bilinear))
            current_features = next_features

        # Final convolution
        self.output_conv = nn.Conv2d(current_features, num_classes, kernel_size=1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial block
        x = self.input_block(x)
        
        # Store intermediate results for skip connections
        skip_connections = [x]
        
        # Downsampling path
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)

        # Remove the last skip connection (bottom of U)
        skip_connections = skip_connections[:-1]
        
        # Upsampling path
        for up_block, skip in zip(self.up_blocks, reversed(skip_connections)):
            x = up_block(x, skip)

        # Final convolution
        return self.output_conv(x)

