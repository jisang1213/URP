# -*- coding: utf-8 -*-
"""UNet with PConv Working.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oqfO0dSL526MfLEi5UFDotMgDO_2VT20
"""

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv1 = PartialConv2d(in_channels, out_channels, 3, 1, 1, bias=False, multi_channel=True, return_mask=True)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)
        self.conv2 = PartialConv2d(out_channels, out_channels, 3, 1, 1, bias=False, multi_channel=True, return_mask=True)
        self.BN2 = nn.BatchNorm2d(out_channels)
        #nn.ReLU(inplace=True)

    def forward(self, x, mask):
        x, mask = self.conv1(x, mask)
        x = self.BN1(x)
        x = self.ReLU(x)
        out, mask = self.conv2(x, mask)
        x = self.BN2(x)
        x = self.ReLU(x)

        return out, mask

class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[32, 64, 128, 256],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature


        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.ups.append(PartialConv2d(feature*2, feature, 3, 1, 1, bias=False, multi_channel=True, return_mask=True))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = PartialConv2d(features[0], out_channels, kernel_size=1, multi_channel=True, return_mask=True)

    def forward(self, x, mask):
        skip_connections = []
        masks = []

        for down in self.downs:
            x, mask = down(x, mask)
            skip_connections.append(x)
            masks.append(mask)
            x = self.pool(x)
            mask = self.pool(mask)

        x, mask = self.bottleneck(x, mask)

        skip_connections = skip_connections[::-1]
        masks = masks[::-1]

        for idx in range(0, len(self.ups), 3):
            # print(f"layer{idx//3} of up path")
            # Nearest Neighbour Upsampling
            x = self.ups[idx](x)
            mask = self.ups[idx](mask)
            # Conv once to halve number of channels
            x, mask = self.ups[idx+1](x, mask)

            skip_connection = skip_connections[idx//3]
            skip_mask = masks[idx//3]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            concat_mask = torch.cat((skip_mask, mask), dim=1)
            x, mask = self.ups[idx+2](concat_skip, concat_mask)

        return self.final_conv(x, mask)

# Function for weight initialization.
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    x = torch.randn((1, 1, 256, 256))
    mask = torch.randn((1, 1, 256, 256))
    model = UNET(in_channels=1, out_channels=1)
    model.apply(weight_init)
    total_params = count_parameters(model)
    print(f"Total number of parameters: {total_params}")

    preds, outmask = model(x, mask)
    print(f"output shape: {preds.shape}")
    assert preds[:,0:1,:,:].shape == x.shape

    print("FORWARD PASS SUCCESSFULLY COMPLETED")