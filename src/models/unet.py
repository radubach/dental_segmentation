import torch
import torch.nn as nn



"""
UNet: A Convolutional Neural Network for Semantic Segmentation

This implementation defines a UNet architecture, commonly used for image segmentation tasks, 
with a symmetric encoder-decoder structure. It extracts hierarchical features from the input 
image in the encoder, combines them with upsampled features in the decoder using skip connections, 
and outputs a per-pixel classification map.

Attributes:
    encoder1, encoder2, encoder3, encoder4 (nn.Module): Convolutional blocks in the encoder, 
        progressively extracting features at different spatial resolutions.
    pool (nn.Module): Max pooling layer used for downsampling feature maps.
    bottleneck (nn.Module): A convolutional block at the bottleneck, capturing global context.
    upconv4, upconv3, upconv2, upconv1 (nn.Module): Transposed convolution layers for upsampling.
    decoder4, decoder3, decoder2, decoder1 (nn.Module): Convolutional blocks in the decoder, 
        combining upsampled features with corresponding encoder features via skip connections.
    final_conv (nn.Module): A 1x1 convolution layer for generating the final per-pixel 
        classification map with `out_channels` classes.

Parameters:
    in_channels (int): Number of input channels. Default is 1 (e.g., grayscale images).
    out_channels (int): Number of output channels (classes) for segmentation. Default is 33.

Methods:
    forward(x): Passes an input tensor through the UNet model and returns the segmentation map.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width) 
        containing the per-pixel class probabilities.
"""


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=33):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def up_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = up_block(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = up_block(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = up_block(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_block(128, 64)
        self.decoder1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))

        return self.final_conv(dec1)
