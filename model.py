'''
@author: Mobarakol Islam
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x

# Source: https://github.com/nyoki-mtl/pytorch-segmentation/blob/master/src/models/common.py
class ActivatedBatchNorm(nn.Module):
    def __init__(self, num_features, activation='relu', slope=0.01, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, **kwargs)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)
        elif activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x

# SCSEBlock Source: https://github.com/nyoki-mtl/pytorch-segmentation/blob/master/src/models/scse.py
class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv2d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    def forward(self, x):
        bahs, chs, _, _ = x.size()
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1))
        chn_se = torch.mul(x, chn_se)

        spa_se = torch.sigmoid(self.spatial_se(x))
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)

class DecoderSCSE(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.ABN = ActivatedBatchNorm(middle_channels)
        self.SCSEBlock = SCSEBlock(middle_channels)
        self.deconv = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            ActivatedBatchNorm(middle_channels),
            SCSEBlock(middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, *args):
        x = torch.cat(args, 1)
        x = self.conv1(x)
        x = self.ABN(x)
        x = self.SCSEBlock(x)
        x = self.deconv(x)
        return x


class BR(nn.Module):
    def __init__(self, out_c):
        super(BR, self).__init__()
        # self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)
        x = x + x_res

        return x

#Source: https://github.com/ooooverflow/BiSeNet/blob/5c924e3b09c4c01fb0048e8806629a131b9f3bf9/model/build_BiSeNet.py
class AttentionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.SCSEBlockARM = SCSEBlock(128)

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        x = self.conv(x)
        x = self.sigmoid(x)
        x = torch.mul(input, x)
        return x


class SalSegNet(nn.Module):
    """
    Model Architecture
    """
    def __init__(self, num_classes=8):
        """
        Model initialization
        """
        super(SalSegNet, self).__init__()

        base = resnet.resnet18(pretrained=True)
        self.num_classes = num_classes
        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        #Saliency
        self.decoder1_sal = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2_sal = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3_sal = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4_sal = Decoder(512, 256, 3, 2, 1, 1)

        # Saliency Classifier
        self.tp_conv1_sal = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.conv2_sal = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True), )
        self.tp_conv2_sal = nn.ConvTranspose2d(32, 1, 2, 2, 0)
        self.sig = nn.Sigmoid()

        #Extra
        self.AM1 = AttentionModule(64, 64)

        # Segmentation
        self.pool = nn.MaxPool2d(2, 2)
        self.center = DecoderSCSE(512, 256, 256)
        self.decoder5 = DecoderSCSE(768, 512, 256)
        self.decoder4 = DecoderSCSE(512, 256, 128)
        self.decoder3 = DecoderSCSE(256, 128, 64)
        self.decoder2 = DecoderSCSE(128, 64, 64)
        self.decoder1 = DecoderSCSE(128, 64, 64)

        # Segmentation Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True), )
        self.tp_conv2 = nn.ConvTranspose2d(32, num_classes, 3, 1, 1)
        self.br = BR(num_classes)
        self.lsm = nn.LogSoftmax(dim=1)



    def forward(self, x):
        # Initial block
        x = self.in_block(x) #64

        # Encoder blocks
        e1 = self.encoder1(x) #64
        e2 = self.encoder2(e1) #128
        e3 = self.encoder3(e2) #256
        e4 = self.encoder4(e3) #512

        # Seg Decoder blocks
        c = self.center(self.pool(e4))
        d4 = self.decoder5(c, e4)
        d3 = self.decoder4(d4, e3)
        d2 = self.decoder3(d3, e2)
        d1 = self.decoder2(d2, e1)
        x_new = F.upsample(x, d1.size()[2:], mode='bilinear', align_corners=True)
        d0 = self.decoder1(d1, x_new)

        # Seg Classifier
        y = self.conv2(d0)
        y = self.tp_conv2(y)
        y = self.br(y)
        y_seg = self.lsm(y)


        # Sal Decoder blocks
        d4 = e3 + self.decoder4_sal(e4)
        d3 = e2 + self.decoder3_sal(d4)
        d2 = e1 + self.decoder2_sal(d3)

        d1 = x + self.decoder1_sal(d2)
        d1 = self.AM1(d1)

        # Sal Classifier
        y = self.tp_conv1_sal(d1)
        y = self.conv2_sal(y)
        y = self.tp_conv2_sal(y)
        y_sal = self.sig(y)

        return y_seg, y_sal

