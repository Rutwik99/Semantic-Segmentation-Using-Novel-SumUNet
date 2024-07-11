import torch.nn as nn
import torch
import torchvision.models
import torch.nn.functional as F
from torchsummary import summary


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, is_sum_unet = False):
        super().__init__()

        self.is_sum_unet = is_sum_unet
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        if is_sum_unet:
            self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print("convtrans", x1.shape)

        if self.is_sum_unet:
            x = x2 + x1
        else:
            x = torch.cat([x2, x1], dim=1)
        # print("merged", x.shape)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.up2 = nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.up(x)
        x2 = self.conv(x1)
        x3 = self.up2(x2)
        return self.conv2(x3)


class UNet(nn.Module):
    def __init__(self, n_class, load_pretrained_encoder_layers=False):
        super().__init__()
        self.n_class = n_class
        
        self.base_model = torchvision.models.resnet18(pretrained=load_pretrained_encoder_layers)
        self.base_layers = list(self.base_model.children())

        self.inc = nn.Sequential(self.base_layers[0], self.base_layers[1], self.base_layers[2])
        self.inc_maxpool = self.base_layers[3] 
        self.encoder1 = self.base_model.layer1
        self.encoder2 = self.base_model.layer2
        self.encoder3 = self.base_model.layer3
        self.encoder4 = self.base_model.layer4

        self.connector = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = Up(1024, 512)
        self.decoder2 = Up(512, 256)
        self.decoder3 = Up(256, 128)
        self.decoder4 = Up(128, 64)
        
        self.outc = OutConv(64, 32, self.n_class)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.inc_maxpool(x1)
        x3 = self.encoder1(x2)
        x4 = self.encoder2(x3)
        x5 = self.encoder3(x4)
        x6 = self.encoder4(x5)
        x7 = self.connector(x6)
        # print("x", x.shape)
        # print("x1", x1.shape)
        # print("x2", x2.shape)
        # print("x3", x3.shape)
        # print("x4", x4.shape)
        # print("x5", x5.shape)
        # print("x6", x6.shape)
        # print("x7", x7.shape)

        x8 = self.decoder1(x7, x6)
        x9 = self.decoder2(x8, x5)
        x10 = self.decoder3(x9, x4)
        x11 = self.decoder4(x10, x3)
        # print("x8", x8.shape)
        # print("x9", x9.shape)
        # print("x10", x10.shape)
        # print("x11", x11.shape)
        
        x13 = self.outc(x11)
        # print("x13", x13.shape)     
        return x13


class SumUNet(nn.Module):
    def __init__(self, n_class, load_pretrained_encoder_layers=False):
        super().__init__()
        self.n_class = n_class
        
        self.base_model = torchvision.models.resnet18(pretrained=load_pretrained_encoder_layers)
        self.base_layers = list(self.base_model.children())

        self.inc = nn.Sequential(self.base_layers[0], self.base_layers[1], self.base_layers[2])
        self.inc_maxpool = self.base_layers[3] 
        self.encoder1 = self.base_model.layer1
        self.encoder2 = self.base_model.layer2
        self.encoder3 = self.base_model.layer3
        self.encoder4 = self.base_model.layer4

        self.connector = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.decoder1 = Up(1024, 512, is_sum_unet=True)
        self.decoder2 = Up(512, 256, is_sum_unet=True)
        self.decoder3 = Up(256, 128, is_sum_unet=True)
        self.decoder4 = Up(128, 64, is_sum_unet=True)

        self.outc = OutConv(64, 32, self.n_class)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.inc_maxpool(x1)
        x3 = self.encoder1(x2)
        x4 = self.encoder2(x3)
        x5 = self.encoder3(x4)
        x6 = self.encoder4(x5)
        x7 = self.connector(x6)
        # print("x", x.shape)
        # print("x1", x1.shape)
        # print("x2", x2.shape)
        # print("x3", x3.shape)
        # print("x4", x4.shape)
        # print("x5", x5.shape)
        # print("x6", x6.shape)
        # print("x7", x7.shape)

        x8 = self.decoder1(x7, x6)
        x9 = self.decoder2(x8, x5)
        x10 = self.decoder3(x9, x4)
        x11 = self.decoder4(x10, x3)
        # print("x8", x8.shape)
        # print("x9", x9.shape)
        # print("x10", x10.shape)
        # print("x11", x11.shape)
        
        x13 = self.outc(x11)
        # print("x13", x13.shape)     
        return x13