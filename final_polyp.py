#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pandas as pd
import numpy as np
import cv2
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from torchvision.transforms import ToTensor
from PIL import Image
import os
# from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
from torchvision import transforms
from torchinfo import summary
import timm

trainsize = 384

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomGamma (gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
    A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
    A.OneOf([A.Blur(), A.GaussianBlur(), A.GlassBlur(), A.MotionBlur(), A.GaussNoise(), A.Sharpen(), A.MedianBlur(), A.MultiplicativeNoise()]),
    A.Cutout(p=0.2, max_h_size=35, max_w_size=35, fill_value=255),
    A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.15, brightness_coeff=1.5, p=0.09),
    A.RandomShadow(p=0.1),
    A.ShiftScaleRotate(p=0.45, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.15, scale_limit=0.15),
    A.RandomCrop(384, 384),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bottle_neck = False):
        super(DoubleConv, self).__init__() 
        self.double_conv = nn. Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if bottle_neck == True:
            self.double_conv = nn. Sequential(
                nn.Conv2d(in_channels, out_channels*2, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_channels*2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class DownBlock (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels) 
        self.down_sample = nn.MaxPool2d(2)
        
    def forward(self, x):
        skip_out = self.double_conv(x) 
        down_out = self.down_sample(skip_out) 
        return (down_out, skip_out)
class UpBlock (nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            if out_channels*4 == in_channels:
                self.up_sample= nn.ConvTranspose2d(in_channels-out_channels*2, in_channels-out_channels*2, kernel_size=2, stride=2)
            else:
                self.up_sample= nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2) 
        else:
            self.up_sample= nn.Upsample (scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1) 
        return self.double_conv(x)


class PolypModel(nn.Module):
    def __init__(self, out_classes=3, up_sample_mode='conv_transpose'):
        super().__init__()
        self.out_classes = out_classes
        self.encoder = timm.create_model("resnet152", pretrained=True, features_only=True)
#         self.down_conv1 = DownBlock(3, 64) 
        self.down_conv1 = DownBlock(64, 128) 
        self.down_conv2 = DownBlock(256, 512) 
        self.down_conv3 = DownBlock (512, 1024) 
        self.down_conv4 = DownBlock (1024, 2048) 
        self.up_sample_mode = up_sample_mode
        self.block_neck = DoubleConv(2048, 1024)
        self.block_up1 = UpBlock (1024+1024, 512, self.up_sample_mode) 
        self.block_up2 = UpBlock (512+512, 256, self.up_sample_mode) 
        self.block_up3 = UpBlock (256+256, 128, self.up_sample_mode) 
        self.block_up4 = UpBlock(128+64, 64, self.up_sample_mode) 
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1) 
        self.upsample = nn.Upsample (scale_factor=2, mode="bilinear")
    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.block_neck(x5)
        x = self.block_up1(x, x4)
        x = self.block_up2(x, x3)
        x = self.block_up3(x, x2) 
        x = self.block_up4(x, x1)
        x = self.conv_last(x)
        x = self.upsample(x) 
        return x

