# !/usr/bin/env python
# encoding: utf-8
"""         
  @Author: Yongheng Sun          
  @Contact: 3304925266@qq.com          
  @Software: PyCharm    
  @Project: transformer_0322
  @File: model_mia1201.py          
  @Time: 2021/10/15 11:24                   
"""
""" Parts of the U-Net model """

import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        final = F.sigmoid(logits)

        return final
        # return logits



class Extractor_DI(nn.Module):
    def __init__(self, n_channels=64):
        super(Extractor_DI, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, n_channels*2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(n_channels*2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(n_channels*2, n_channels*2, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(n_channels*2),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Conv2d(n_channels*2, n_channels, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(n_channels),
            )
    def forward(self, f_all):
        f_di = self.inc(f_all)
        return f_di

class Domain_classifier(nn.Module):
    def __init__(self, n_channels=64):
        super(Domain_classifier, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(n_channels, n_channels * 2, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(n_channels * 2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(n_channels * 4),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(n_channels * 8),
        #     nn.ReLU(inplace=True),
        # )
        self.discriminator = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(n_channels, n_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(n_channels, n_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(n_channels * 2, n_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(n_channels * 4, n_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(n_channels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        # self.pool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(n_channels * 8, 2)

    def forward(self, x):
        # x1 = self.conv(x)
        # x2 = self.pool(x1).squeeze(-1).squeeze(-1)
        # prob = self.fc(x2)
        prob = self.discriminator(x)
        return prob

class Domain_classifier_end(nn.Module):
    def __init__(self, n_channels=64):
        super(self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(n_channels, n_channels * 2, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(n_channels * 2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(n_channels * 4),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=7, stride=2, padding=3),
        #     nn.BatchNorm2d(n_channels * 8),
        #     nn.ReLU(inplace=True),
        # )
        self.discriminator = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(n_channels, n_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(n_channels, n_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(n_channels * 2, n_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(n_channels * 4, n_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(n_channels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        # self.pool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(n_channels * 8, 2)

    def forward(self, x):
        # x1 = self.conv(x)
        # x2 = self.pool(x1).squeeze(-1).squeeze(-1)
        # prob = self.fc(x2)
        prob = self.discriminator(x)
        return prob


from typing import Optional, Any, Tuple
import torch.nn as nn
from torch.autograd import Function
import torch

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
    # def forward(self, ctx, input, coeff = 1.):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
    # def backward(self, ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

# class GradReverse(Function):
#     def __init__(self, lambd):
#         self.lambd = lambd
#     def forward(self, x):
#         return x.view_as(x)
#     def backward(self, grad_output):
#         return (grad_output*-self.lambd)
# def grad_reverse(x,lambd=1.0):
#     return GradReverse(lambd)(x)


class UNet_DA(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.feature_extractor = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.E_DI = Extractor_DI()
        self.grl = GradientReverseLayer()
        self.domain_classifier = Domain_classifier()
        self.domain_classifier2 = Domain_classifier()
        self.avgpool = torch.nn.AvgPool2d(224)

    def forward(self, x):
        f_all = self.feature_extractor(x)

        f_di = self.E_DI(f_all)
        f_ds = f_all - f_di

        f_di_pool = self.avgpool(f_di).squeeze().squeeze()
        f_ds_pool = self.avgpool(f_ds).squeeze().squeeze()
        # loss_orthogonal = torch.mean(f_di_pool.square() * f_ds_pool.square(), dim=1)
        loss_orthogonal = (f_di_pool.square() * f_ds_pool.square()).mean()

        # f_ds = grad_reverse(f_ds, 1.0)
        f_di = self.grl(f_di)
        prob_ds = self.domain_classifier(f_ds)
        prob_di = self.domain_classifier2(f_di)

        x1 = f_di
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        final = F.sigmoid(logits)

        return final, loss_orthogonal, prob_di, prob_ds
        # return logits

class UNet_DA_end(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        self.E_DI = Extractor_DI()

        self.grl = GradientReverseLayer()
        self.domain_classifier = Domain_classifier_end()
        self.domain_classifier2 = Domain_classifier_end()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)


        f_all = x
        f_di = self.E_DI(f_all)
        f_ds = f_all - f_di

        f_di_pool = self.avgpool(f_di).squeeze().squeeze()
        f_ds_pool = self.avgpool(f_ds).squeeze().squeeze()
        # loss_orthogonal = torch.mean(f_di_pool.square() * f_ds_pool.square(), dim=1)
        loss_orthogonal = (f_di_pool.square() * f_ds_pool.square()).mean()

        # f_ds = grad_reverse(f_ds, 1.0)
        f_di = self.grl(f_di)
        prob_ds = self.domain_classifier(f_ds)
        prob_di = self.domain_classifier2(f_di)
        
        logits = self.outc(f_di)
        final = F.sigmoid(logits)

        return final, loss_orthogonal, prob_di, prob_ds
        # return logits


if __name__ == '__main__':
    inp = torch.randn(2, 3, 224, 224)
    # inp = torch.randn(2, 64, 224, 224)

    # model = UNet()
    model = UNet_DA()
    model = nn.DataParallel(model)
    # model = Domain_classifier()
    for name, param in model.named_parameters():
        if "feature_extractor" in name:
            param.requires_grad = False
        if "down" in name:
            param.requires_grad = False
        if "up" in name:
            param.requires_grad = False
        if "outc" in name:
            param.requires_grad = False
        print(name)
    out = model(inp)


