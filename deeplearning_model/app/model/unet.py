""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn

from .unet_parts import *
from .utils import *
from .sspp import SwinASPP


"""
Full assembly of the parts to form the complete UNet network with SwinASPP.

This implementation integrates a U-Net model with SwinASPP for enhanced feature extraction.

Attributes:
    - n_channels (int): Number of input channels.
    - n_classes (int): Number of output classes.
    - bilinear (bool): Whether to use bilinear interpolation in upsampling.

Usage:
    model = UNet(n_channels=3, n_classes=1, bilinear=False)
"""
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        """
        Initialize the U-Net model with SwinASPP for semantic segmentation.

        Args:
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes.
            bilinear (bool, optional): Whether to use bilinear upsampling. Defaults to False.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Initial convolution block.
        self.inc = (DoubleConv(n_channels, 64))

        # Downsampling path.
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        # SwinASPP module for multi-scale feature extraction.
        self.aspp = SwinASPP(
            input_size=8,
            input_dim=1024,
            out_dim=512,
            cross_attn='CBAM',
            depth=2,
            num_heads=32,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            aspp_norm=False,
            aspp_activation='relu',
            start_window_size=7,
            aspp_dropout=0.1,
            downsample=None,
            use_checkpoint=True
        )

        # Bottleneck layer to refine extracted features.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Upsampling path.
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        # Final output convolution.
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, n_channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, n_classes, height, width).
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Apply SwinASPP for enhanced feature representation.
        x5 = x5.permute(0, 2, 3, 1)
        x = self.aspp(x5)
        x = x.permute(0, 3, 1, 2)
        x = self.bottleneck(x)

        # Upsampling with skip connections.
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """
        Enable gradient checkpointing to reduce memory usage during training.
        """
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
