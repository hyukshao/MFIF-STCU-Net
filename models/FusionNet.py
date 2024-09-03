import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_

import depth_loaddata_demo
from . import depth_senet, depth_modules, depth_net
from .util import ResBlock, conv
import torch
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
# from .cross_swin_transformer_unet import SwinTransformerSys

from .swin_transformer_unet_conv import SwinTransformerSys_conv
import depth_loaddata_demo as loaddata
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

__all__ = [
    'fusionnet', 'fusionnet_bn'
]


class CBAfusionNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(CBAfusionNet, self).__init__()
        feats = 64
        kernel_size = 3
        defocus_out = 1
        deblur_out = 3

        self.swintransformer = SwinTransformerSys()
        self.conv01a = conv(deblur_out, feats, kernel_size)#in:3
        self.convRes01a = ResBlock(feats, feats, kernel_size)
        self.convRes02a = ResBlock(feats, feats, kernel_size)
        self.convRes03a = ResBlock(feats, feats, kernel_size)
        self.convRes04a = ResBlock(feats, feats, kernel_size)
        self.conv02a = conv(feats, feats, kernel_size)
        self.conv03a = conv(feats, defocus_out, kernel_size)


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)


    def forward(self, input_image, target_fusion=None, target_Gmap=None):

        out_swin = self.swintransformer(input_image)
        out_conv01a = self.conv01a(out_swin)
        out_convRes01a = out_conv01a + self.convRes01a(out_conv01a)
        out_convRes02a = out_convRes01a + self.convRes02a(out_convRes01a)
        out_convRes03a = out_convRes02a + self.convRes03a(out_convRes02a)
        out_convRes04a = out_convRes03a + self.convRes04a(out_convRes03a)
        out_conv02a = self.conv02a(out_convRes04a)
        out_conv03a = self.conv03a(out_conv02a)
        temp_result = out_conv03a * input_image[:, :3] + (1-out_conv03a) * input_image[:, 3:]
        output = torch.cat((out_conv03a, temp_result), 1)
        return output


    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def fusionnet(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = CBAfusionNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def fusionnet_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = CBAfusionNet(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
