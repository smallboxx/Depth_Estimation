# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The deconvolution code is based on Simple Baseline.
# (https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer)
from mmcv.cnn.utils import constant_init, normal_init
from .swin_transformer_v2 import SwinTransformerV2
from .depthmlp import depth_mpl
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

class GLPDepth(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.depth_list = args.depth_list
        self.in_net_size= args.crop_h
        
        if 'tiny' in args.backbone:
            embed_dim = 96
            num_heads = [3, 6, 12, 24]
        elif 'base' in args.backbone:
            embed_dim = 128
            num_heads = [4, 8, 16, 32]
        elif 'large' in args.backbone:
            embed_dim = 192
            num_heads = [6, 12, 24, 48]
        elif 'huge' in args.backbone:
            embed_dim = 352
            num_heads = [11, 22, 44, 88]
        else:
            raise ValueError(args.backbone+" is not implemented, please add it in the models/model.py.")

        self.encoder = SwinTransformerV2(
            embed_dim=embed_dim,
            depths=args.depths,
            num_heads=num_heads,
            window_size=args.window_size,
            pretrain_window_size=args.pretrain_window_size,
            drop_path_rate=args.drop_path_rate,
            use_checkpoint=args.use_checkpoint,
            use_shift=args.use_shift,
        )

        self.encoder.init_weights(pretrained=args.pretrained)
        
        channels_in = embed_dim*8
        channels_out = embed_dim


        self.depth_mlp = depth_mpl(in_channels=channels_in)
        

        self.decoder = Decoder(channels_in, channels_out, args)
        self.decoder.init_weights()

        self.last_layer_depth= nn.Sequential(
            nn.Conv2d(channels_out*2, channels_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels_out, 1, kernel_size=3, stride=1, padding=1))

        for m in self.last_layer_depth.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)

        if args.finetune:
            for params in self.encoder.parameters():
                params.requires_grad = False
            for params in self.depth_mlp.parameters():
                params.requires_grad = False

    def infer(self, mode: bool = True):
        self.mode = mode
        return self


    def forward(self, input_RGB):  
        N,_,H,W = input_RGB.shape    
        x = F.interpolate(input_RGB, size=(self.in_net_size, self.in_net_size), mode='bilinear', align_corners=True)
        conv_feats    = self.encoder(x)[0]
        domain_logits = self.depth_mlp(conv_feats)

        fea = self.decoder(conv_feats)

        out = []
        for i in range(N):
            refer   = domain_logits[i] * 20.0
            fea_per = torch.cat([fea[i].unsqueeze(0),refer*fea[i].unsqueeze(0)], dim=1)
            out_per = self.last_layer_depth(fea_per)
            out_per = F.interpolate(out_per, size=(H,W), mode='bilinear', align_corners=True)
            if self.mode:
                out_per = dy_infer_depth(out_per, refer, self.depth_list)
            else:
                out_per = torch.sigmoid(out_per) * refer
            out.append(out_per)

        out = torch.cat(out, dim=0)
        return {'pred_d': out, 'domain':domain_logits}


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.deconv = args.num_deconv
        self.in_channels = in_channels
        
        self.deconv_layers = self._make_deconv_layer(
            args.num_deconv,
            args.num_filters,
            args.deconv_kernels,
        )
        
        conv_layers = []
        conv_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=args.num_filters[-1],
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1))
        conv_layers.append(
            build_norm_layer(dict(type='BN'), out_channels)[1])
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, conv_feats):
        out = self.deconv_layers(conv_feats)
        out = self.conv_layers(out)

        out = self.up(out)
        out = self.up(out)

        return out

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        
        layers = []
        in_planes = self.in_channels
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=in_planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            in_planes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        return deconv_kernel, padding, output_padding

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

# def dy_infer_depth(input, refer_depth, depth_list):
#     input = torch.sigmoid(input)
#     if refer_depth <= 10.0:
#         output = input * refer_depth
#     elif refer_depth > 10.0 and refer_depth <= 18.0:
#         output = input * depth_list[0]
#     else:
#         output = input * depth_list[1]

#     return output


def dy_infer_depth(input, refer_depth, depth_list):
    input = torch.sigmoid(input)
    B,C,H,W = input.shape
    count = torch.sum((input > 0.8) & (input < 0.9))

    mask1 = (input < 0.1)
    mask2 = (input >= 0.1) & (input < 0.2)
    mask3 = (input >= 0.2) & (input < 0.3)
    mask4 = (input >= 0.3) & (input < 0.4)
    mask5 = (input >= 0.4) & (input < 0.5)
    mask6 = (input >= 0.6) & (input < 0.7)
    mask7 = (input >= 0.7) & (input < 0.8)
    mask8 = (input >= 0.8) & (input < 0.9)
    mask9 = (input >= 0.9)
    count_true = torch.sum(mask6)

    print("mask6 中的 True 值数量：", count_true.item())
    count_true = torch.sum(mask7)

    print("mask7 中的 True 值数量：", count_true.item())
    if refer_depth <= 10.0:
        output = input * refer_depth
    if refer_depth > 10.0 and refer_depth <= 18.0:
        output = input * depth_list[0]
        output[mask1] = output[mask1]*1.00
        output[mask2] = output[mask2]*1.00
        output[mask3] = output[mask3]*1.00
        output[mask4] = output[mask4]*1.00
        output[mask5] = output[mask5]*1.05
        output[mask6] = output[mask6]*1.25
        output[mask7] = output[mask7]*1.25
        output[mask8] = output[mask8]*1.25
        output[mask9] = output[mask9]*1.25

    elif refer_depth > 18.0 and refer_depth <20.0:
        output = input * depth_list[1]
        output[mask1] = output[mask1]*0.95
        output[mask2] = output[mask2]*0.95
        output[mask3] = output[mask3]*0.98
        output[mask4] = output[mask4]*1.00
        output[mask5] = output[mask5]*1.00
        output[mask6] = output[mask6]*1.15
        output[mask7] = output[mask7]*1.15
        output[mask8] = output[mask8]*1.15
        output[mask9] = output[mask9]*1.15
    else:
        output = input * depth_list[1]
        output[mask1] = output[mask1]*0.93
        output[mask2] = output[mask2]*0.93
        output[mask3] = output[mask3]*1.00
        output[mask4] = output[mask4]*1.00
        output[mask5] = output[mask5]*1.00
        output[mask6] = output[mask6]*1.30
        output[mask7] = output[mask7]*1.40
        output[mask8] = output[mask8]*1.50
        output[mask9] = output[mask9]*1.50

    return output

