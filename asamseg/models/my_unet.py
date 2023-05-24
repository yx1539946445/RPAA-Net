import warnings
import numpy as np
import torch
from torch.nn import functional as F
from monai.networks.layers import Conv, ChannelPad
from torch import nn
from typing import Callable, Tuple, List
import pytorch_lightning as pl
import asamseg.utils as myut

from asamseg.attention_packages import CoordAttention, AxialAttention, EfficientAttention, SpatialAttention, \
    CrissCrossAttention, sa_layer, ACmix, SimAM, CBAM, PsAAttention, NAM, SpatialAttentionModule, selfattention

'''
   BN 就是批量归一化

   RELU 就是激活函数

   lambda x:x 这个函数的意思是输出等于输入

   identity 就是残差

   1个resnet block 包含2个basic block
   1个resnet block 需要添加2个残差

   在resnet block之间残差形式是1*1conv，在resnet block内部残差形式是lambda x:x
   resnet block之间的残差用粗箭头表示，resnet block内部的残差用细箭头表示

   3*3conv s=2，p=1 特征图尺寸会缩小
   3*3conv s=1，p=1 特征图尺寸不变
'''

'''SA '''
import torch
from torch import nn

import math
import cv2

from torch.nn import init


class conv_1X1(nn.Module):
    '''
    使用conv_1X1 改变维度通道数
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 strides: int = 1,
                 num_groups=32
                 ):
        super(conv_1X1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class conv_3X3(nn.Module):
    '''
    使用conv_3X3 改变维度通道数
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 num_groups=32,
                 ):
        super(conv_3X3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout2d(0.1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class conv_7X7(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups=32,
                 ):
        super(conv_7X7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, in_channels):
        super().__init__()
        # self.bn = nn.BatchNorm2d(in_channels)
        self.ca = CoordAttention(in_channels, in_channels)
        self.conv = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(torch.cat([self.bn(x[..., ::2, ::2]), self.bn(x[..., 1::2, ::2]), self.bn(x[..., ::2, 1::2]),
                                    self.bn(x[..., 1::2, 1::2])], 1))


# 残差块
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False,
                 num_groups=4):
        super(CBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                      bias=bias, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, in_channels, out_channels, norm_layer=nn.LayerNorm):
        super().__init__()

        self.reduction = CBR(4 * in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.ca = CoordAttention(4 * in_channels, 4 * in_channels)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.permute([0, 2, 3, 1])
        # x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape((B, H // 2, W // 2, 4 * C))  # B H/2*W/2 4*C
        # x = x.view(B, -1, 4 * C)
        x = x.permute([0, 3, 1, 2])
        # x = self.ca(x)
        # x = self.norm(x)
        x = self.reduction(x)

        return x

    # def extra_repr(self) -> str:
    #     return f"input_resolution={self.input_resolution}, dim={self.dim}"
    #
    # def flops(self):
    #     H, W = self.input_resolution
    #     flops = H * W * self.dim
    #     flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
    #     return flops


def autopad(kernel, padding=None):  # kernel, padding
    # Pad to 'same'
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]  # auto-pad
    return padding


class DWConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, ):
        super().__init__()

        self.DW_Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      groups=min(in_channels, out_channels),
                      padding=dilation,
                      dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.PConv = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   )

    def forward(self, x):
        out = self.DW_Conv(x)
        out = self.PConv(out)

        return out


# class ResBolck(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=[1, 3, 5], groups=1,
#                  bias=False, is_decoder=False):
#         super(ResBolck, self).__init__()
#         inter_channels = out_channels // 4
#         groups = inter_channels
#         self.conv_input = nn.Sequential(
#             CBR(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
#         )
#         self.conv_1 = nn.Sequential(
#             CBR(inter_channels, inter_channels, kernel_size=kernel_size, stride=stride, padding=dilation[0],
#                 dilation=dilation[0], groups=groups),
#         )
#         self.conv_2 = nn.Sequential(
#             CBR(inter_channels, inter_channels, kernel_size=kernel_size, stride=stride, padding=dilation[1],
#                 dilation=dilation[1], groups=groups),
#         )
#         self.conv_3 = nn.Sequential(
#             CBR(inter_channels, inter_channels, kernel_size=kernel_size, stride=stride, padding=dilation[2],
#                 dilation=dilation[2], groups=groups),
#         )
#         #
#         # self.conv_1_1 = nn.Sequential(
#         #     CBR(inter_channels, inter_channels, kernel_size=kernel_size, stride=stride, padding=dilation[0],
#         #         dilation=dilation[0], groups=groups),
#         # )
#         # self.conv_2_1 = nn.Sequential(
#         #     CBR(inter_channels, inter_channels, kernel_size=kernel_size, stride=stride, padding=dilation[1],
#         #         dilation=dilation[1], groups=groups),
#         # )
#         # self.conv_3_1 = nn.Sequential(
#         #     CBR(inter_channels, inter_channels, kernel_size=kernel_size, stride=stride, padding=dilation[2],
#         #         dilation=dilation[2], groups=groups),
#         # )
#
#         self.ca = CoordAttention(out_channels, out_channels)
#
#     def forward(self, x):
#         x = self.conv_input(x)
#         output_1 = self.conv_1(x)
#         output_2 = self.conv_2(output_1)
#         output_3 = self.conv_3(output_2)
#         output = channel_shuffle(torch.cat([x, output_1, output_2, output_3], dim=1))
#
#         # output_1_1 = self.conv_1_1(x)
#         # output_2_1 = self.conv_2_1(output_1_1)
#         # output_3_1 = self.conv_3_1(output_2_1)
#         # output2 = torch.cat([x, output_1_1, output_2_1, output_3_1], dim=1)
#         #
#         # output = output1 + output2
#         output = self.ca(output)
#         return output

class ResCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False,
                 num_groups=4):
        super(ResCBR, self).__init__()
        self.conv = CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.conv(x)
        output = self.sigmoid(output) * output
        return output


# class ResBolck(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=[1, 1], groups=1,
#                  bias=False, is_decoder=False):
#         super(ResBolck, self).__init__()
#         groups = min(in_channels, out_channels)
#         if stride == 1 and in_channels == out_channels:
#             self.short = None
#         else:
#             self.short = CBR(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
#         self.conv_1 = nn.Sequential(
#             CBR(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation[0],
#                 dilation=dilation[0], groups=groups),
#
#             CBR(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=dilation[0],
#                 dilation=dilation[0], groups=groups),
#         )
#
#     def forward(self, x):
#         output_conv_1 = self.conv_1(x)
#         if self.short is None:
#             output = output_conv_1 + x
#         else:
#             output = output_conv_1 + self.short(x)
#         return output


def channel_shuffle(x, groups=2):
    b, c, h, w = x.shape
    x = x.reshape(b, groups, -1, h, w)
    x = x.permute(0, 2, 1, 3, 4)
    # flatten
    x = x.reshape(b, -1, h, w)
    return x


# class ResBolck(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=[1, 1], groups=1,
#                  bias=False, is_decoder=False):
#         super(ResBolck, self).__init__()
#         self.conv_1 = nn.Sequential(
#             CBR(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=dilation[0],
#                 dilation=dilation[0], groups=in_channels),
#             CBR(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=dilation[1],
#                 dilation=dilation[1], groups=in_channels),
#         )
#
#     def forward(self, x):
#         return self.conv_1(x) + x


# class ResCBR(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=[1, 3], ):
#         super(ResCBR, self).__init__()
#         groups = 4
#         self.conv_1 = CBR(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation[0],
#                           dilation=dilation[0], groups=groups)
#         self.conv_2 = CBR(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=dilation[1],
#                           dilation=dilation[1], groups=groups)
#
#     def forward(self, x):
#         output_1 = channel_shuffle(self.conv_1(x))
#         output_2 = channel_shuffle(self.conv_2(output_1))
#         return output_1 + output_2 + x


# class DownSumple(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DownSumple, self).__init__()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.conv_1 = CBR(in_channels , out_channels , kernel_size=3, stride=2, padding=1)
#         self.conv_output = CBR(in_channels * 2  , out_channels , kernel_size=1, stride=1, padding=0)
#     def forward(self, x):
#         output = channel_shuffle(torch.cat([self.maxpool(x), self.conv_1(x)], dim=1))
#         return output

# class AGs(nn.Module):
#     def __init__(self, in_channels_1,in_channels_2,in_channels_3, out_channels):
#         super(AGs, self).__init__()
#         self.maxpool_1 = nn.Sequential(
#             nn.Conv2d(in_channels_1, out_channels, kernel_size=1, stride=1, padding=0),
#             nn.MaxPool2d(kernel_size=8, stride=8)
#         )
#         self.maxpool_2 = nn.Sequential(
#             nn.Conv2d(in_channels_2, out_channels, kernel_size=1, stride=1, padding=0),
#             nn.MaxPool2d(kernel_size=4, stride=4)
#         )
#         self.maxpool_3 = nn.Sequential(
#             nn.Conv2d(in_channels_3, out_channels, kernel_size=1, stride=1, padding=0),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.gate = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x_1, x_2, x_3, x_4):
#         x_1 = self.maxpool_1(x_1)
#         x_2 = self.maxpool_2(x_2)
#         x_3 = self.maxpool_3(x_3)
#         x = x_1 + x_2 + x_3 + x_4
#         output = self.gate(x) * x_4
#         return output
#
#

# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:[n,c,h,w]
        # mean和max会沿着通道进行求平均和求最大值，并且我们保留了通道这个维度，不保留的话就是[n,h,w]了
        avg_out = torch.mean(x, dim=1, keepdim=True)  # avg_out:[n,1,h,w]
        max_out = torch.max(x, dim=1, keepdim=True)[0]  # max_out:[n,1,h,w]
        out = torch.cat((avg_out, max_out), 1)  # out:[n,2,h,w]
        out = self.conv2d(out)
        out = self.sigmoid(out) * x
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(ASPP, self).__init__()
        inter_channels = out_channels // 2
        groups = inter_channels
        self.conv_1 = CBR(in_channels, inter_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=groups)
        self.conv_2 = CBR(in_channels, inter_channels, kernel_size=3, stride=1, padding=6, dilation=6, groups=groups)
        self.ca = CoordAttention(out_channels, out_channels)

    def forward(self, x):
        output_1 = self.conv_1(x)
        output_2 = self.conv_2(x)
        output = channel_shuffle(torch.cat([output_1, output_2], dim=1))
        output = self.ca(output) + x
        return output


class U_encoder(nn.Module):
    def __init__(self):
        super(U_encoder, self).__init__()
        channels = [32, 64, 128, 256]
        self.layer_1 = nn.Sequential(
            conv_3X3(1, channels[0]),
            # ASPP(channels[0], channels[0]),
            conv_3X3(channels[0], channels[0]),
            # ResBolck(1, channels[0])
        )
        self.layer_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_3X3(channels[0], channels[1]),
            # ASPP(channels[1], channels[1]),

            conv_3X3(channels[1], channels[1]),
            # ResBolck(channels[0], channels[1], stride=2),  # down
            # ResBolck(channels[1], channels[1]),
            # ResBolck(channels[0], channels[1])
        )
        self.layer_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_3X3(channels[1], channels[2]),
            # ASPP(channels[2], channels[2]),
            conv_3X3(channels[2], channels[2]),
            # ResBolck(channels[1], channels[2], stride=2),
            # ResBolck(channels[2], channels[2]),  # down
            # ResBolck(channels[1], channels[2])
        )
        self.layer_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_3X3(channels[2], channels[3]),
            # ASPP(channels[3], channels[3]),
            conv_3X3(channels[3], channels[3]),
            CoordAttention(channels[3], channels[3])
            # conv_3X3(channels[3], channels[3]),
            # Tree_Fusion(channels[2],channels[3])
            # ResBolck(channels[2], channels[3])
            # ResBolck(channels[2], channels[3], stride=2),
            # ResBolck(channels[3], channels[3]),
        )

        # self.ags_3 = AGs(channels[0], channels[1], channels[2], channels[3])

    def forward(self, x):
        features = []

        # pre = self.layer_pre(x)
        # features.append(pre)

        x = self.layer_1(x)  # skip...

        features.append(x)  # (256, 256, 64)   16

        x = self.layer_2(x)  # skip....

        features.append(x)  # (128, 128, 128)  32

        x = self.layer_3(x)  # skip...
        features.append(x)
        x = self.layer_4(x)
        # x = self.ags_3(features[0], features[1], features[2], x)
        return x, features


class U_decoder(nn.Module):
    def __init__(self):
        super(U_decoder, self).__init__()
        channels = [32, 64, 128, 256]
        self.trans_1 = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True)
        )
        self.res_1 = nn.Sequential(
            conv_3X3(channels[3], channels[2]),
            conv_3X3(channels[2], channels[2]),
        )
        self.trans_2 = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )
        self.res_2 = nn.Sequential(
            conv_3X3(channels[2], channels[1]),
            conv_3X3(channels[1], channels[1]),
        )
        self.trans_3 = nn.Sequential(
            nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        self.res_3 = nn.Sequential(
            conv_3X3(channels[1], channels[0]),
            conv_3X3(channels[0], channels[0]),
        )

    def forward(self, x, feature):
        # x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x = self.res_1(torch.cat([self.trans_1(x), feature[2]], dim=1))
        # x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x = self.res_2(torch.cat([self.trans_2(x), feature[1]], dim=1))
        # x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x = self.res_3(torch.cat([self.trans_3(x), feature[0]], dim=1))
        # x = self.res_4(self.trans_4(x))

        return x


def instance_to_mask(instance):
    '''
    Args:
        instance: PIL or numpy image
    '''
    instance = np.array(instance, dtype=np.uint32)
    # instance = 256 * (256 * instance[:, :, 0] + instance[:, :, 1]) + instance[:, :, 2]  # 3通道转灰度图
    object_list = np.unique(instance[instance != 0])  # 挑出tensor中的独立不重复元素
    current_num = 1
    for obj in object_list:
        instance[instance == obj] = current_num
        current_num += 1
    return instance.astype(np.uint8)


def get_gap_feature(feature, h, w):
    """
    Convert instance label to binary label, and generate gap label.
    """

    feature = feature.reshape(h, w)
    mask = instance_to_mask(feature)
    _, dilated_instance_list = myut.get_instance_list(mask)
    gap_map = myut.get_gap_map(dilated_instance_list, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
    # kernel =   # 矩形结构
    # gap_map = cv2.dilate(gap_map.astype(np.uint8), kernel=kernel, iterations=1)
    gap_map_dilate = cv2.dilate(gap_map.astype(np.uint8), kernel=kernel, iterations=2)
    gap_map_erode = cv2.erode(gap_map.astype(np.uint8), kernel=kernel, iterations=2)
    gap_map = gap_map_dilate - gap_map_erode  # 膨胀的图像 - 腐蚀的图像 边缘信息
    # mask = myut.instance_mask_to_binary_label(mask)
    # gap_map = myut.clean_gap_map(gap_map, mask)
    feature = gap_map.astype(np.uint8)
    return feature.reshape(1, 1, h, w)


class my_unet(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 extra_gap_weight: float,
                 learning_rate: float = 1.0e-3,
                 loss_func: Callable = nn.CrossEntropyLoss(),
                 total_iterations: int = 1000,
                 ):
        super(my_unet, self).__init__()
        self.u_encoder = U_encoder()
        self.u_decoder = U_decoder()
        # self.kernel = torch.nn.Parameter(1,requires_grad=True)
        self.SegmentationHead_1 = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0),
        )
        # self.sigmoid = nn.Sigmoid()
        # self.SegmentationHead_2 = nn.Conv2d(1, out_channels, 1)
        # self.classify_conv = ChannelPad(2, 64, out_channels,
        #                                 'project')  # number of spatial dimensions of the input image is 2
        self.extra_gap_weight = extra_gap_weight
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.total_iterations = total_iterations

    def forward(self, x):

        encoder_x_first, encoder_skips_first = self.u_encoder(x)
        decoder_x_first = self.u_decoder(encoder_x_first, encoder_skips_first)
        x1 = self.SegmentationHead_1(decoder_x_first)

        # b, c, h, w = x1.shape
        # # print(x1.shape)
        # x2 = x1.cpu().data.numpy()
        # feature = np.zeros((b, c, h, w))
        # for i in range(b):
        #     for j in range(c):
        #         feature[i][j] = get_gap_feature(x2[i][j], h, w)
        # feature = torch.from_numpy(feature).cuda()
        # # feature = 1 - self.sigmoid(feature)
        # # feature = feature.unsqueeze(0).unsqueeze(0)  # 增加维度
        # # feature = feature.reshape(b, c, h, w)
        # x1 = x1 * feature
        return x1

    def training_step(self, batch, batch_idx):
        loss = None
        if self.extra_gap_weight is None:
            loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=False)
        else:
            loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
                                           use_sliding_window=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = None
        if self.extra_gap_weight is None:
            loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=True)
        else:
            loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
                                           use_sliding_window=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
            on_tpu=False, using_native_amp=False, using_lbfgs=False,
    ):
        initial_learning_rate = self.learning_rate
        current_iteration = self.trainer.global_step
        total_iteration = self.total_iterations
        for pg in optimizer.param_groups:
            pg['lr'] = myut.poly_learning_rate(initial_learning_rate, current_iteration, total_iteration)
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        return myut.configure_optimizers(self, self.learning_rate)


# class my_unet(pl.LightningModule):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  extra_gap_weight: float,
#                  learning_rate: float = 1.0e-3,
#                  loss_func: Callable = nn.CrossEntropyLoss(),
#                  total_iterations: int = 1000,
#                  ):
#         super(my_unet, self).__init__()
#         self.u_encoder = U_encoder()
#         self.u_decoder = U_decoder()
#         self.SegmentationHead_1 = nn.Conv2d(32, out_channels, 1, bias=False)
#         self.SegmentationHead_2 = nn.Conv2d(32, out_channels, 1, bias=False)
#         self.SegmentationHead_3 = nn.Conv2d(64, out_channels, 1, bias=False)
#         self.SegmentationHead_4 = nn.Conv2d(128, out_channels, 1, bias=False)
#
#         self.pool = nn.MaxPool2d(2, ceil_mode=False)
#
#         self.extra_gap_weight = extra_gap_weight
#         self.learning_rate = learning_rate
#         self.loss_func = loss_func
#         self.total_iterations = total_iterations
#
#     def forward(self, x):
#
#         print("x", x.shape)
#
#         encoder_x_first, encoder_skips_first = self.u_encoder(x)
#         decoder_x_first, decoder_x_second, decoder_x_third, decoder_x_four = self.u_decoder(encoder_x_first,encoder_skips_first)
#         x1 = self.SegmentationHead_1(decoder_x_first)
#         x2 = self.SegmentationHead_2(decoder_x_second)
#         x3 = self.SegmentationHead_3(decoder_x_third)
#         x4 = self.SegmentationHead_4(decoder_x_four)
#
#         # x = self.SegmentationHead(decoder_x_first)
#         return x1, x2, x3, x4
#
#     def training_step(self, batch, batch_idx):
#         loss = None
#         if self.extra_gap_weight is None:
#             loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=False)
#         else:
#             images, labels, gap_maps = myut.get_batch_data(batch)
#             downsampledx2_labels = self.pool(labels.float())
#             downsampledx4_labels = self.pool(downsampledx2_labels)
#             downsampledx8_labels = self.pool(downsampledx4_labels)
#             images, labels ,gap_maps= images.float(), labels.long(),gap_maps.long()
#             downsampledx2_labels = downsampledx2_labels.long()
#             downsampledx4_labels = downsampledx4_labels.long()
#             downsampledx8_labels = downsampledx8_labels.long()
#
#             x1, x2, x3, x4 = self(images)
#
#             loss1 = self.loss_func(x1, labels) + loss * gap_maps * self.extra_gap_weight
#             loss2 = self.loss_func(x2, downsampledx2_labels) + loss * gap_maps * self.extra_gap_weight
#             loss3 = self.loss_func(x3, downsampledx4_labels) + loss * gap_maps * self.extra_gap_weight
#             loss4 = self.loss_func(x4, downsampledx8_labels) + loss * gap_maps * self.extra_gap_weight
#
#             # loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#             #                                 use_sliding_window=False)
#
#             loss = loss1 + 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss = None
#         if self.extra_gap_weight is None:
#             loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=True)
#         else:
#             images, labels, gap_maps = myut.get_batch_data(batch)
#             downsampledx2_labels = self.pool(labels.float())
#             downsampledx4_labels = self.pool(downsampledx2_labels)
#             downsampledx8_labels = self.pool(downsampledx4_labels)
#             images, labels ,gap_maps= images.float(), labels.long(),gap_maps.long()
#             downsampledx2_labels = downsampledx2_labels.long()
#             downsampledx4_labels = downsampledx4_labels.long()
#             downsampledx8_labels = downsampledx8_labels.long()
#
#             x1, x2, x3, x4 = self(images)
#
#             loss1 = self.loss_func(x1, labels) + loss * gap_maps * self.extra_gap_weight
#             loss2 = self.loss_func(x2, downsampledx2_labels) + loss * gap_maps * self.extra_gap_weight
#             loss3 = self.loss_func(x3, downsampledx4_labels) + loss * gap_maps * self.extra_gap_weight
#             loss4 = self.loss_func(x4, downsampledx8_labels) + loss * gap_maps * self.extra_gap_weight
#
#             # loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#             #                                 use_sliding_window=False)
#
#             loss = loss1 + 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4
#             # loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#             #                                use_sliding_window=True)
#         self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def optimizer_step(
#             self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
#             on_tpu=False, using_native_amp=False, using_lbfgs=False,
#     ):
#         initial_learning_rate = self.learning_rate
#         current_iteration = self.trainer.global_step
#         total_iteration = self.total_iterations
#         for pg in optimizer.param_groups:
#             pg['lr'] = myut.poly_learning_rate(initial_learning_rate, current_iteration, total_iteration)
#         optimizer.step(closure=optimizer_closure)
#
#     def configure_optimizers(self):
#         return myut.configure_optimizers(self, self.learning_rate)


# import warnings
# import numpy as np
# import torch
# from torch.nn import functional as F
# from monai.networks.layers import Conv
# from torch import nn
# from typing import Callable
# import pytorch_lightning as pl
# import asamseg.utils as myut
#
# from asamseg.attention_packages import CoordAttention, AxialAttention, EfficientAttention, SpatialAttention, \
#     CrissCrossAttention, sa_layer, ACmix, SimAM, CBAM
#
# '''
#    BN 就是批量归一化
#
#    RELU 就是激活函数
#
#    lambda x:x 这个函数的意思是输出等于输入
#
#    identity 就是残差
#
#    1个resnet block 包含2个basic block
#    1个resnet block 需要添加2个残差
#
#    在resnet block之间残差形式是1*1conv，在resnet block内部残差形式是lambda x:x
#    resnet block之间的残差用粗箭头表示，resnet block内部的残差用细箭头表示
#
#    3*3conv s=2，p=1 特征图尺寸会缩小
#    3*3conv s=1，p=1 特征图尺寸不变
# '''
#
# '''SA '''
# import torch
# from torch import nn
#
# import math
# import cv2
#
# from torch.nn import init
#
#
#
# class conv_1X1(nn.Module):
#     '''
#     使用conv_1X1 改变维度通道数
#     '''
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  strides: int = 1,
#                  num_groups=32
#                  ):
#         super(conv_1X1, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
#
#             # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
#             # nn.SiLU(inplace=True),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class conv_3X3(nn.Module):
#     '''
#     使用conv_3X3 改变维度通道数
#     '''
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: int = 3,
#                  stride: int = 1,
#                  padding: int = 1,
#                  num_groups=32,
#                  ):
#         super(conv_3X3, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class conv_7X7(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  num_groups=32,
#                  ):
#         super(conv_7X7, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# # 残差块
# class CBR(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False,
#                  num_groups=32):
#         super(CBR, self).__init__()
#         self.conv = nn.Sequential(
#
#             # nn.GroupNorm(num_groups, out_channels),
#             # nn.ReLU(inplace=True),
#
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
#                       bias=bias, dilation=dilation),
#             nn.BatchNorm2d(out_channels),
#             # nn.Dropout2d(0.1),
#             nn.ReLU(inplace=True),
#             # nn.PReLU(num_parameters=1, init=0.25),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# # without BN version
# class ASPP(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(ASPP, self).__init__()
#         self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
#         self.conv = nn.Conv2d(in_channel, out_channel, 1, 1)
#         self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
#         self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
#         self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
#         self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
#         self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)
#
#     def forward(self, x):
#         size = x.shape[2:]
#
#         image_features = self.mean(x)
#         image_features = self.conv(image_features)
#         image_features = F.upsample(image_features, size=size, mode='bilinear')
#
#         atrous_block1 = self.atrous_block1(x)
#         atrous_block6 = self.atrous_block6(x)
#         atrous_block12 = self.atrous_block12(x)
#         atrous_block18 = self.atrous_block18(x)
#
#         net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
#                                               atrous_block12, atrous_block18], dim=1))
#         return net
#
#
# class ResBolck(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=[1, 1], groups=1,
#                  bias=False, is_decoder=False):
#         super(ResBolck, self).__init__()
#
#         if is_decoder is False:
#             inter_channels = in_channels
#         else:
#             inter_channels = out_channels
#         self.conv_1 = nn.Sequential(
#             CBR(in_channels, inter_channels, kernel_size=kernel_size, stride=stride, padding=dilation[0],
#                 dilation=dilation[0]),
#             CBR(inter_channels, out_channels, kernel_size=kernel_size, stride=1, padding=dilation[1],
#                 dilation=dilation[1]),
#         )
#         if stride == 1:
#             self.shortcut = nn.Sequential(
#                 CBR(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
#             )
#         else:
#             self.shortcut = nn.Sequential(
#                 nn.MaxPool2d(kernel_size=2, stride=2),
#                 CBR(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
#             )
#
#     def forward(self, x):
#         print("self.conv_1(x)", self.conv_1(x).shape)
#         print("self.shortcut(x)", self.shortcut(x).shape)
#         return self.conv_1(x) + self.shortcut(x)
#
#
#
#
# class ResCBR(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResCBR, self).__init__()
#         self.conv_1 = CBR(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, )
#         self.shortcut = CBR(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
#
#     def forward(self, x):
#         return self.conv_1(x) + self.shortcut(x)
#
#
# class ResASPP(nn.Module):
#     def __init__(self, in_channels, out_channels, ):
#         super(ResASPP, self).__init__()
#         inter_channels = out_channels // 2
#         self.conv = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
#         self.conv_1 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=3, dilation=3)
#         self.conv_2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=5, dilation=5)
#         self.gate = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         short = x
#         x = self.conv(x)
#         return F.relu(self.gate(torch.cat([self.conv_1(x), self.conv_2(x)], dim=1)) * short + short)
#
#
# class Resatten(nn.Module):
#     def __init__(self, in_channels, out_channels, ):
#         super(Resatten, self).__init__()
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = CBR(in_channels, 1, kernel_size=1, stride=1, padding=0)
#         self.sigmiod = nn.Sigmoid()
#
#         self.shortcut = CBR(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         return self.shortcut(x) + self.sigmiod(self.conv(self.max_pool(x) + self.avg_pool(x))) * x
#
#
# class ResCBRAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, ):
#         super(ResCBRAttention, self).__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#             CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1, ),
#         )
#         self.shortcut_attention = nn.Sequential(
#             CoordAttention(in_channels, in_channels),
#             CBR(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
#         )
#
#     def forward(self, x):
#         return self.maxpool_conv(x) + self.shortcut_attention(x) + x
#
#
#
#
#
# class MSA_1(nn.Module):
#     def __init__(self, in_channels, out_channels, ratio=16):
#         super(MSA_1, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.psi = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, 1, kernel_size=1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, g1, x):
#         x = self.conv(x)
#         m = g1 + x
#         psi = self.psi(m) * x + g1
#         return psi
#
#
# class MSA_2(nn.Module):
#     def __init__(self, in_channels, out_channels, ratio=16):
#         super(MSA_2, self).__init__()
#
#         self.psi_1 = nn.Sequential(
#             # CoordAttention(in_channels, out_channels),
#             # nn.BatchNorm2d(in_channels),
#             # nn.ReLU(inplace=True),
#             # nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
#             # nn.Sigmoid(),
#         )
#         # self.attention = nn.Sequential(
#         #     CoordAttention(in_channels, in_channels),
#         #     # EfficientAttention(in_channels, in_channels, 8, in_channels),
#         #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         # )
#
#     def forward(self, g2, x):
#         # m = torch.cat([g2, x], dim=1)
#         m = g2 + x
#         psi = self.psi_1(m)
#         return psi + x
#
#
# class SPM(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(SPM, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(in_channel),
#             nn.ReLU(inplace=True),
#         )
#
#         self.h_pool = nn.AdaptiveAvgPool2d((1, None))
#         self.w_pool = nn.AdaptiveAvgPool2d((None, 1))
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True),
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_h = self.conv1(x)
#         x_w = self.conv1(x)
#
#         # split
#         x_h = self.h_pool(x_h)
#         x_w = self.w_pool(x_w)
#
#         # expand
#         x_h = F.upsample_bilinear(x_h, (h, w))
#         x_w = F.upsample_bilinear(x_w, (h, w))
#
#         # fusion
#         fusion = x_h + x_w
#         fusion = self.conv2(fusion)
#
#         fusion = self.sigmoid(fusion)
#
#         fusion = fusion * x
#
#         return F.relu(fusion)
#
#
# class PPM(nn.Module):
#     def __init__(self, channels=2048):
#         super(PPM, self).__init__()
#         bins = (1, 2, 3, 6)
#         reduction_dim = int(channels / len(bins))
#         self.features = []
#         for bin in bins:
#             self.features.append(nn.Sequential(
#                 nn.AdaptiveAvgPool2d(bin),
#                 nn.Conv2d(channels, reduction_dim, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(reduction_dim),
#                 nn.ReLU(inplace=True)
#             ))
#         self.features = nn.ModuleList(self.features)
#
#     def forward(self, x):
#         x_size = x.size()
#         out = [x]
#         for f in self.features:
#             out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
#         return torch.cat(out, 1)
#
#
#
#
# class Fusion(nn.Module):
#     def __init__(self):
#         super(Fusion, self).__init__()
#         self.feature_0 = nn.MaxPool2d(4)  # 64
#
#         self.feature_1 = nn.MaxPool2d(2)  # 64
#
#         self.feature_1_out = nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0)
#         self.feature_2_out = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
#         self.feature_3_out = nn.Sequential(
#             nn.MaxPool2d(2),
#             nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
#
#         )
#         # self.atten = eca_layer(256)
#         self.gate = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid(),
#         )
#         # self.gate_1 = nn.Sequential(
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
#         #     nn.Sigmoid(),
#         # )
#         # self.gate_2 = nn.Sequential(
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
#         #     nn.Sigmoid(),
#         # )
#         # self.gate_3 = nn.Sequential(
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
#         #     nn.Sigmoid(),
#         # )
#
#     def forward(self, feature):
#         feature_ = feature
#         feature_0 = self.feature_0(feature[0])  # 16
#         feature_1 = self.feature_1(feature[1])  # 16  32 64
#         # feature_2 = feature[2]  # 128
#         feature_3 = F.interpolate(feature[3], scale_factor=(2, 2), mode='bilinear')  # 256
#         feature = torch.cat([feature_0, feature_1, feature[2], feature_3], dim=1)  # 128
#         feature = self.gate(feature) * feature
#         # feature = self.atten(feature)
#         # m = torch.cat([g3, x], dim=1)
#         # feature_1 = self.gate_1(self.feature_1_out(F.interpolate(feature, scale_factor=(2, 2), mode='bilinear')) +feature[1])
#         # feature_2 = self.gate_2(self.feature_2_out(feature) + feature[2])
#         # feature_3 = self.gate_3(self.feature_3_out(feature)+feature[3])
#         feature_1 = self.feature_1_out(F.interpolate(feature, scale_factor=(2, 2), mode='bilinear')) + feature_[1]
#         feature_2 = self.feature_2_out(feature) + feature_[2]
#         feature_3 = self.feature_3_out(feature) + feature_[3]
#         return feature_1, feature_2, feature_3
#
#
#
# class U_encoder(nn.Module):
#     def __init__(self, ):
#         super(U_encoder, self).__init__()
#
#         self.layer_pre = nn.Sequential(
#
#             conv_3X3(1, 32),
#
#         )
#         self.layer_1 = nn.Sequential(
#
#             conv_7X7(1, 32),
#
#         )
#         self.layer_2 = nn.Sequential(
#             nn.MaxPool2d(2),
#             ResBolck(32, 16),
#             ResBolck(16, 16),
#         )
#
#         self.layer_3 = nn.Sequential(
#             ResBolck(16, 16, stride=2),
#             ResBolck(16, 32, ),
#         )
#         self.layer_4 = nn.Sequential(
#             ResBolck(32, 64, dilation=[1, 1]),
#             ResBolck(64, 64, dilation=[1, 1]),
#             # ResASPP(256, 256),
#         )
#         self.layer_5 = nn.Sequential(
#             ResBolck(64, 64, dilation=[2, 2]),
#             ResBolck(64, 128, dilation=[4, 4]),
#             CBAM(in_channels=128)
#
#         )
#
#     def forward(self, x):
#         features = []
#         pre = self.layer_pre(x)
#
#         features.append(pre)
#
#         x = self.layer_1(x)
#
#         features.append(x)  # (256, 256, 64)   128
#         # x = self.pool1(x)   # skip--->1
#
#         x = self.layer_2(x)  # 1
#
#         features.append(x)  # (128, 128, 128)  64
#
#         x = self.layer_3(x)
#
#
#         x = self.layer_4(x)  # 3  # skip--->3
#
#         x = self.layer_5(x)
#
#         return x, features
#
#
# class U_decoder(nn.Module):
#     def __init__(self):
#         super(U_decoder, self).__init__()
#         self.res_1 = nn.Sequential(
#             ResBolck(144, 64, is_decoder=True),
#         )
#
#         self.res_2 = nn.Sequential(
#             ResBolck(96, 32, is_decoder=True)
#         )
#
#         self.res_3 = nn.Sequential(
#             ResBolck(64, 32, is_decoder=True)
#         )
#
#     def forward(self, x, feature):
#         x = self.res_1(
#             torch.cat((F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True), feature[2]), dim=1))
#         x = self.res_2(
#             torch.cat((F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True), feature[1]), dim=1))
#         x = self.res_3(
#             torch.cat((F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True), feature[0]), dim=1))
#
#         return x
#
#
# class my_unet(pl.LightningModule):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  extra_gap_weight: float,
#                  learning_rate: float = 1.0e-3,
#                  loss_func: Callable = nn.CrossEntropyLoss(),
#                  total_iterations: int = 1000,
#                  ):
#         super(my_unet, self).__init__()
#         self.u_encoder = U_encoder()
#         self.u_decoder = U_decoder()
#         self.SegmentationHead_1 = nn.Conv2d(32, out_channels, 1, bias=False)
#
#         self.extra_gap_weight = extra_gap_weight
#         self.learning_rate = learning_rate
#         self.loss_func = loss_func
#         self.total_iterations = total_iterations
#
#     def forward(self, x):
#         print("x", x.shape)
#         encoder_x_first, encoder_skips_first = self.u_encoder(x)
#         decoder_x_first = self.u_decoder(encoder_x_first, encoder_skips_first)
#         x = self.SegmentationHead_1(decoder_x_first)
#
#         return x
#
#     def training_step(self, batch, batch_idx):
#         loss = None
#
#         if self.extra_gap_weight is None:
#             loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=False)
#
#         else:
#             loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                            use_sliding_window=False)
#
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss = None
#
#         if self.extra_gap_weight is None:
#             loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=False)
#
#         else:
#             loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                            use_sliding_window=False)
#
#         self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def optimizer_step(
#             self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
#             on_tpu=False, using_native_amp=False, using_lbfgs=False,
#     ):
#         initial_learning_rate = self.learning_rate
#         current_iteration = self.trainer.global_step
#         total_iteration = self.total_iterations
#         for pg in optimizer.param_groups:
#             pg['lr'] = myut.poly_learning_rate(initial_learning_rate, current_iteration, total_iteration)
#         optimizer.step(closure=optimizer_closure)
#
#     def configure_optimizers(self):
#         return myut.configure_optimizers(self, self.learning_rate)


#
#
# class U_decoder(nn.Module):
#     def __init__(self):
#         super(U_decoder, self).__init__()
#         # self.trans_1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.res_1 = nn.Sequential(
#             ResBolck(144, 64, is_decoder=True),
#         )
#         # self.trans_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.res_2 = nn.Sequential(
#             ResBolck(96, 32, is_decoder=True)
#         )
#         # self.trans_3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
#         self.res_3 = nn.Sequential(
#             ResBolck(64, 32, is_decoder=True)
#         )
#
#         # self.trans_4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
#         # self.res_4 = nn.Sequential(
#         #     ResBolck(64, 32, is_decoder=True)
#         # )
#
#     #         self.gate_1 = MSA_1(128, 64)
#     #         self.gate_2 = MSA_1(128, 32)
#     #         self.gate_3 = MSA_1(64, 16)
#     #         self.gate_4 = MSA_1(32, 16)
#
#     def forward(self, x, feature):
#         # x = torch.cat((F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True), feature[3]), dim=1)
#         #
#         # x = torch.cat((F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True), feature[2]), dim=1)
#         #
#         # x = torch.cat((F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True), feature[1]), dim=1)
#         #
#         # x = torch.cat((F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True), feature[0]), dim=1)
#         # x = self.res_1(
#         #     torch.cat((F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True), feature[3]), dim=1))
#         x1 = F.interpolate(x, scale_factor=(8, 8), mode='bilinear', align_corners=True)
#         x = self.res_1(
#             torch.cat((F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True), feature[2]), dim=1))
#         x2 = F.interpolate(x, scale_factor=(4, 4), mode='bilinear', align_corners=True)
#         x = self.res_2(
#             torch.cat((F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True), feature[1]), dim=1))
#         x3 = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
#         x = self.res_3(
#             torch.cat((F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True), feature[0]), dim=1))
#
#         # x = self.res_1(torch.cat((self.trans_1(x), feature[3]), dim=1))
#         #
#         # x = self.res_2(torch.cat((self.trans_2(x), feature[2]), dim=1))
#         #
#         # x = self.res_3(torch.cat((self.trans_3(x), feature[1]), dim=1))
#         #
#         # x = self.res_4(torch.cat((self.trans_4(x), feature[0]), dim=1))
#
#         return x, x1, x2, x3
#
#
# class my_unet(pl.LightningModule):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  extra_gap_weight: float,
#                  learning_rate: float = 1.0e-3,
#                  loss_func: Callable = nn.CrossEntropyLoss(),
#                  total_iterations: int = 1000,
#                  ):
#         super(my_unet, self).__init__()
#         self.u_encoder = U_encoder()
#         self.u_decoder = U_decoder()
#         self.SegmentationHead_1 = nn.Conv2d(32, out_channels, 1, bias=False)
#         self.SegmentationHead_2 = nn.Conv2d(32, out_channels, 1, bias=False)
#         self.SegmentationHead_3 = nn.Conv2d(64, out_channels, 1, bias=False)
#         self.SegmentationHead_4 = nn.Conv2d(128, out_channels, 1, bias=False)
#
#         self.extra_gap_weight = extra_gap_weight
#         self.learning_rate = learning_rate
#         self.loss_func = loss_func
#         self.total_iterations = total_iterations
#
#     def forward(self, x):
#
#         encoder_x_first, encoder_skips_first = self.u_encoder(x)
#         decoder_x_first, decoder_x_second, decoder_x_third, decoder_x_four = self.u_decoder(encoder_x_first,
#                                                                                             encoder_skips_first)
#
#         x = self.SegmentationHead_1(decoder_x_first)
#         x1 = self.SegmentationHead_4(decoder_x_second)
#         x2 = self.SegmentationHead_3(decoder_x_third)
#         x3 = self.SegmentationHead_2(decoder_x_four)
#
#         return x, x1, x2, x3
#
#     def training_step(self, x, x1, x2, x3, batch, batch_idx):
#         loss = None
#         if self.extra_gap_weight is None:
#             loss1 = myut.cal_batch_loss(x, batch, self.loss_func, use_sliding_window=False)
#             loss2 = myut.cal_batch_loss(x1, batch, self.loss_func, use_sliding_window=False)
#             loss3 = myut.cal_batch_loss(x2, batch, self.loss_func, use_sliding_window=False)
#             loss4 = myut.cal_batch_loss(x3, batch, self.loss_func, use_sliding_window=False)
#             loss = loss1 + 0.2 * loss2 + 0.3 * loss3 + 0.5 * loss4
#         else:
#             loss1 = myut.cal_batch_loss_gap(x, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                             use_sliding_window=False)
#             loss2 = myut.cal_batch_loss_gap(x1, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                             use_sliding_window=False)
#             loss3 = myut.cal_batch_loss_gap(x2, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                             use_sliding_window=False)
#             loss4 = myut.cal_batch_loss_gap(x3, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                             use_sliding_window=False)
#             loss = loss1 + 0.2 * loss2 + 0.3 * loss3 + 0.5 * loss4
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def validation_step(self, x, x1, x2, x3, batch, batch_idx):
#         loss = None
#         if self.extra_gap_weight is None:
#             loss1 = myut.cal_batch_loss(x, batch, self.loss_func, use_sliding_window=False)
#             loss2 = myut.cal_batch_loss(x1, batch, self.loss_func, use_sliding_window=False)
#             loss3 = myut.cal_batch_loss(x2, batch, self.loss_func, use_sliding_window=False)
#             loss4 = myut.cal_batch_loss(x3, batch, self.loss_func, use_sliding_window=False)
#             loss = loss1 + 0.2 * loss2 + 0.3 * loss3 + 0.5 * loss4
#         else:
#             loss1 = myut.cal_batch_loss_gap(x, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                             use_sliding_window=False)
#             loss2 = myut.cal_batch_loss_gap(x1, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                             use_sliding_window=False)
#             loss3 = myut.cal_batch_loss_gap(x2, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                             use_sliding_window=False)
#             loss4 = myut.cal_batch_loss_gap(x3, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                             use_sliding_window=False)
#             loss = loss1 + 0.2 * loss2 + 0.3 * loss3 + 0.5 * loss4
#         self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
#         return loss

# import warnings
# import numpy as np
# import torch
# from torch.nn import functional as F
# from monai.networks.layers import Conv
# from torch import nn
# from typing import Callable
# import pytorch_lightning as pl
# import asamseg.utils as myut
#
# from asamseg.attention_packages import CoordAttention, AxialAttention, EfficientAttention, SpatialAttention, \
#     CrissCrossAttention
#
# '''
#    BN 就是批量归一化
#
#    RELU 就是激活函数
#
#    lambda x:x 这个函数的意思是输出等于输入
#
#    identity 就是残差
#
#    1个resnet block 包含2个basic block
#    1个resnet block 需要添加2个残差
#
#    在resnet block之间残差形式是1*1conv，在resnet block内部残差形式是lambda x:x
#    resnet block之间的残差用粗箭头表示，resnet block内部的残差用细箭头表示
#
#    3*3conv s=2，p=1 特征图尺寸会缩小
#    3*3conv s=1，p=1 特征图尺寸不变
# '''
#
# '''SA '''
# import torch
# from torch import nn
#
# import math
# import cv2
#
#
# class External_attention(nn.Module):
#     '''
#     Arguments:
#         c (int): The input and output channel number. 官方的代码中设为512
#     '''
#
#     def __init__(self, in_channels, out_channels):
#         super(External_attention, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
#         self.linear_0 = nn.Conv1d(in_channels, in_channels, 1, bias=False)
#
#         self.linear_1 = nn.Conv1d(in_channels, in_channels, 1, bias=False)
#         self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.Conv1d):
#                 n = m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, x):
#         idn = x
#         x = self.conv1(x)
#
#         b, c, h, w = x.size()
#         n = h * w
#         x = x.view(b, c, h * w)  # b * c * n
#
#         attn = self.linear_0(x)  # b, k, n
#         # linear_0是第一个memory unit
#         attn = F.softmax(attn, dim=-1)  # b, k, n
#
#         attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
#
#         x = self.linear_1(attn)  # b, c, n
#         # linear_1是第二个memory unit
#         x = x.view(b, c, h, w)
#         x = self.conv2(x)
#         x = x + idn
#         x = F.relu(x)
#         return x
#
#
# from torch.nn import init
#
#
# class DWConv(nn.Module):
#     def __init__(self, dim=768):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         x = x.transpose(1, 2).view(B, C, H, W)
#         x = self.dwconv(x)
#         x = x.flatten(2).transpose(1, 2)
#
#         return x
#
#
# class conv_1X1(nn.Module):
#     '''
#     使用conv_1X1 改变维度通道数
#     '''
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  strides: int = 1,
#                  num_groups=32
#                  ):
#         super(conv_1X1, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
#
#             # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
#             # nn.SiLU(inplace=True),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class conv_3X3(nn.Module):
#     '''
#     使用conv_3X3 改变维度通道数
#     '''
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: int = 3,
#                  stride: int = 1,
#                  padding: int = 1,
#                  num_groups=32,
#                  ):
#         super(conv_3X3, self).__init__()
#         self.conv = nn.Sequential(
#             # nn.BatchNorm2d(in_channels),
#             # nn.ReLU(inplace=True),  # 49 改动去掉3x3 7x7 的relu  再加上bias = False
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),  # 49 改动去掉3x3 7x7 的relu  再加上bias = False
#
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class conv_7X7(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  num_groups=32,
#                  ):
#         super(conv_7X7, self).__init__()
#         self.conv = nn.Sequential(
#             # nn.BatchNorm2d(in_channels),
#             # nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# # 残差块
# class CBR(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False,
#                  num_groups=32):
#         super(CBR, self).__init__()
#         self.conv = nn.Sequential(
#
#             # nn.GroupNorm(num_groups, out_channels),
#             # nn.ReLU(inplace=True),
#
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
#                       bias=bias, dilation=dilation),
#             nn.BatchNorm2d(out_channels),
#             # nn.Dropout2d(0.1),
#             nn.ReLU(inplace=True),
#             # nn.PReLU(num_parameters=1, init=0.25),
#
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# # without BN version
# class ASPP(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(ASPP, self).__init__()
#         self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
#         self.conv = nn.Conv2d(in_channel, out_channel, 1, 1)
#         self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
#         self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
#         self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
#         self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
#         self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)
#
#     def forward(self, x):
#         size = x.shape[2:]
#
#         image_features = self.mean(x)
#         image_features = self.conv(image_features)
#         image_features = F.upsample(image_features, size=size, mode='bilinear')
#
#         atrous_block1 = self.atrous_block1(x)
#         atrous_block6 = self.atrous_block6(x)
#         atrous_block12 = self.atrous_block12(x)
#         atrous_block18 = self.atrous_block18(x)
#
#         net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
#                                               atrous_block12, atrous_block18], dim=1))
#         return net
#
#
# class ACBR(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
#         super(ACBR, self).__init__()
#         self.conv = nn.Sequential(
#             CBR(in_channels, in_channels, kernel_size=(kernel_size, 1), stride=stride, padding=(padding, 0)),
#             CBR(in_channels, out_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, padding))
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# def channel_shuffle(x, groups=8):
#     b, c, h, w = x.shape
#     x = x.reshape(b, groups, -1, h, w)
#     x = x.permute(0, 2, 1, 3, 4)
#     # flatten
#     x = x.reshape(b, -1, h, w)
#     return x
#
#
# class ResBolck(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=[1, 1], groups=1,
#                  bias=False, is_decoder=False):
#         super(ResBolck, self).__init__()
#         if is_decoder is False:
#             inter_channels = in_channels
#         else:
#             inter_channels = out_channels
#         self.conv_1 = nn.Sequential(
#             CBR(in_channels, inter_channels, kernel_size=kernel_size, stride=stride, padding=dilation[0],
#                 dilation=dilation[0]),
#             CBR(inter_channels, out_channels, kernel_size=kernel_size, stride=1, padding=dilation[1],
#                 dilation=dilation[1]),
#         )
#
#         self.shortcut = nn.Sequential(
#             CBR(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
#         )
#
#     def forward(self, x):
#
#         return self.conv_1(x) + self.shortcut(x)
#
#
# # class ResBolck(nn.Module):
# #     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=[1, 1], groups=1,
# #                  bias=False, is_decoder=False):
# #         super(ResBolck, self).__init__()
# #
# #         self.conv_1 = nn.Sequential(
# #             CBR(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
# #             nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
# #             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
# #             # CoordAttention(out_channels, out_channels),
# #         )
# #
# #         self.shortcut = nn.Sequential(
# #             CBR(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
# #         )
# #
# #     def forward(self, x):
# #         return self.conv_1(x) + self.shortcut(x)
#
#
# class MSA_1(nn.Module):
#     def __init__(self, in_channels, out_channels, ratio=16):
#         super(MSA_1, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.psi = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, 1, kernel_size=1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, g1, x):
#         x = self.conv(x)
#         m = g1 + x
#         psi = self.psi(m) * g1
#         return psi
#
#
# class MSA_2(nn.Module):
#     def __init__(self, in_channels, out_channels, ratio=16):
#         super(MSA_2, self).__init__()
#
#         self.psi_1 = nn.Sequential(
#             # CoordAttention(in_channels, out_channels),
#             # nn.BatchNorm2d(in_channels),
#             # nn.ReLU(inplace=True),
#             # nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
#             # nn.Sigmoid(),
#         )
#         # self.attention = nn.Sequential(
#         #     CoordAttention(in_channels, in_channels),
#         #     # EfficientAttention(in_channels, in_channels, 8, in_channels),
#         #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         # )
#
#     def forward(self, g2, x):
#         # m = torch.cat([g2, x], dim=1)
#         m = g2 + x
#         psi = self.psi_1(m)
#         return psi + x
#
#
# class SPM(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(SPM, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(in_channel),
#             nn.ReLU(inplace=True),
#         )
#
#         self.h_pool = nn.AdaptiveAvgPool2d((1, None))
#         self.w_pool = nn.AdaptiveAvgPool2d((None, 1))
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(inplace=True),
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_h = self.conv1(x)
#         x_w = self.conv1(x)
#
#         # split
#         x_h = self.h_pool(x_h)
#         x_w = self.w_pool(x_w)
#
#         # expand
#         x_h = F.upsample_bilinear(x_h, (h, w))
#         x_w = F.upsample_bilinear(x_w, (h, w))
#
#         # fusion
#         fusion = x_h + x_w
#         fusion = self.conv2(fusion)
#
#         fusion = self.sigmoid(fusion)
#
#         fusion = fusion * x
#
#         return F.relu(fusion)
#
#
# class PPM(nn.Module):
#     def __init__(self, channels=2048):
#         super(PPM, self).__init__()
#         bins = (1, 2, 3, 6)
#         reduction_dim = int(channels / len(bins))
#         self.features = []
#         for bin in bins:
#             self.features.append(nn.Sequential(
#                 nn.AdaptiveAvgPool2d(bin),
#                 nn.Conv2d(channels, reduction_dim, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(reduction_dim),
#                 nn.ReLU(inplace=True)
#             ))
#         self.features = nn.ModuleList(self.features)
#
#     def forward(self, x):
#         x_size = x.size()
#         out = [x]
#         for f in self.features:
#             out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
#         return torch.cat(out, 1)
#
#
# class edge_canny(nn.Module):
#     def __init__(self):
#         super(edge_canny, self).__init__()
#
#     def forward(self, x):
#         im_arr = x.cpu().detach().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
#         canny = np.zeros((x.size()[0], x.size()[1], x.size()[2], x.size()[3]))
#         for i in range(x.size()[0]):
#             canny[i] = cv2.Canny(im_arr[i], 10, 100)
#         canny = torch.from_numpy(canny).cuda().float()
#
#         return canny
#
#
# class CBAM(nn.Module):
#     def __init__(self, in_channels, out_channels, h, w, reduction=4, spatial_kernel=7):
#         super(CBAM, self).__init__()
#         # channel attention 压缩H,W为1
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         self.avg_pool_h, self.max_pool_h = nn.AdaptiveAvgPool2d((h, 1)), nn.AdaptiveMaxPool2d((h, 1))
#         self.avg_pool_w, self.max_pool_w = nn.AdaptiveAvgPool2d((1, w)), nn.AdaptiveMaxPool2d((1, w))
#
#         inter_channels = in_channels + in_channels
#         # shared MLP
#         self.mlp = nn.Sequential(
#             # Conv2d比Linear方便操作
#             # nn.Linear(channel, channel // reduction, bias=False)
#             nn.BatchNorm2d(inter_channels),
#             nn.Conv2d(inter_channels, in_channels // reduction, 1, bias=False),
#             # inplace=True直接替换，节省内存
#             nn.ReLU(inplace=True),
#             # nn.Linear(channel // reduction, channel,bias=False)
#             nn.Conv2d(in_channels // reduction, out_channels, 1, bias=False)
#         )
#         # spatial attention
#         self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
#                               padding=spatial_kernel // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_pool_h, max_pool_h = self.avg_pool_h(x).permute(0, 1, 3, 2), self.max_pool_h(x).permute(0, 1, 3, 2)
#         avg_pool_w, max_pool_w = self.avg_pool_w(x), self.avg_pool_w(x)
#         channel_out = self.sigmoid(self.mlp(torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1))) * x
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#
#         spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1))) * x
#         return spatial_out + channel_out
#
#
# class Fusion(nn.Module):
#     def __init__(self):
#         super(Fusion, self).__init__()
#         self.feature_0 = nn.MaxPool2d(4)  # 64
#
#         self.feature_1 = nn.MaxPool2d(2)  # 64
#
#         self.feature_1_out = nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0)
#         self.feature_2_out = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
#         self.feature_3_out = nn.Sequential(
#             nn.MaxPool2d(2),
#             nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
#
#         )
#         # self.atten = eca_layer(256)
#         self.gate = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid(),
#         )
#         # self.gate_1 = nn.Sequential(
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
#         #     nn.Sigmoid(),
#         # )
#         # self.gate_2 = nn.Sequential(
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
#         #     nn.Sigmoid(),
#         # )
#         # self.gate_3 = nn.Sequential(
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
#         #     nn.Sigmoid(),
#         # )
#
#     def forward(self, feature):
#         feature_ = feature
#         feature_0 = self.feature_0(feature[0])  # 16
#         feature_1 = self.feature_1(feature[1])  # 16  32 64
#         # feature_2 = feature[2]  # 128
#         feature_3 = F.interpolate(feature[3], scale_factor=(2, 2), mode='bilinear')  # 256
#         feature = torch.cat([feature_0, feature_1, feature[2], feature_3], dim=1)  # 128
#         feature = self.gate(feature) * feature
#         # feature = self.atten(feature)
#         # m = torch.cat([g3, x], dim=1)
#         # feature_1 = self.gate_1(self.feature_1_out(F.interpolate(feature, scale_factor=(2, 2), mode='bilinear')) +feature[1])
#         # feature_2 = self.gate_2(self.feature_2_out(feature) + feature[2])
#         # feature_3 = self.gate_3(self.feature_3_out(feature)+feature[3])
#         feature_1 = self.feature_1_out(F.interpolate(feature, scale_factor=(2, 2), mode='bilinear')) + feature_[1]
#         feature_2 = self.feature_2_out(feature) + feature_[2]
#         feature_3 = self.feature_3_out(feature) + feature_[3]
#         return feature_1, feature_2, feature_3
#
#
# class _AtrousSpatialPyramidPoolingModule(nn.Module):
#     '''
#     operations performed:
#       1x1 x depth
#       3x3 x depth dilation 6
#       3x3 x depth dilation 12
#       3x3 x depth dilation 18
#       image pooling
#       concatenate all together
#       Final 1x1 conv
#     '''
#
#     def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
#         super(_AtrousSpatialPyramidPoolingModule, self).__init__()
#
#         # Check if we are using distributed BN and use the nn from encoding.nn
#         # library rather than using standard pytorch.nn
#
#         if output_stride == 8:
#             rates = [2 * r for r in rates]
#         elif output_stride == 16:
#             pass
#         else:
#             raise 'output stride of {} not supported'.format(output_stride)
#
#         self.features = []
#         # 1x1
#         self.features.append(
#             nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
#                           nn.BatchNorm2d(reduction_dim),
#                           nn.ReLU(inplace=True)
#                           )
#         )
#         # other rates
#         for r in rates:
#             self.features.append(nn.Sequential(
#                 nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
#                           dilation=r, padding=r, bias=False),
#                 nn.BatchNorm2d(reduction_dim),
#                 nn.ReLU(inplace=True)
#             ))
#         self.features = torch.nn.ModuleList(self.features)
#
#         # img level features
#         self.img_pooling = nn.AdaptiveAvgPool2d(1)
#         self.img_conv = nn.Sequential(
#             nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(reduction_dim),
#             nn.ReLU(inplace=True)
#         )
#         self.edge_conv = nn.Sequential(
#             nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
#             nn.BatchNorm2d(reduction_dim),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x, edge):
#         x_size = x.size()
#
#         img_features = self.img_pooling(x)
#         img_features = self.img_conv(img_features)
#         img_features = F.interpolate(img_features, x_size[2:],
#                                      mode='bilinear', align_corners=True)
#         out = img_features
#
#         edge_features = F.interpolate(edge, x_size[2:],
#                                       mode='bilinear', align_corners=True)
#         edge_features = self.edge_conv(edge_features)
#         out = torch.cat((out, edge_features), 1)
#
#         for f in self.features:
#             y = f(x)
#             out = torch.cat((out, y), 1)
#         return out
#
#
# class U_encoder(nn.Module):
#     def __init__(self, ):
#         super(U_encoder, self).__init__()
#
#         self.layer_pre = nn.Sequential(
#             conv_3X3(1, 16),
#         )
#         self.layer_1 = nn.Sequential(
#             conv_7X7(1, 16),
#         )
#         self.layer_2 = nn.Sequential(
#             nn.MaxPool2d(2),
#             ResBolck(16, 16),
#             ResBolck(16, 32),
#         )
#         self.layer_3 = nn.Sequential(
#             ResBolck(32, 32, stride=2, dilation=[1, 1]),
#             ResBolck(32, 32),
#             ResBolck(32, 64),
#             # EfficientAttention(64, 64, 4, 64),
#         )
#         self.layer_4 = nn.Sequential(
#             ResBolck(64, 64, stride=2, dilation=[1, 1]),
#             ResBolck(64, 64, dilation=[2, 2]),
#             ResBolck(64, 128, dilation=[1, 1]),
#             # ASPP(128,128),
#             # CrissCrossAttention(128, 128)
#
#         )
#
#         self.canny_layer_pre = Canny(self.layer_pre)
#         self.canny_layer_pre_add = CannyAdd(self.layer_pre, self.canny_layer_pre)
#
#         self.canny_layer_1 = Canny(self.layer_1)
#         self.canny_layer_1_add = CannyAdd(self.layer_1, self.canny_layer_1)
#         self.canny_layer_2 = Canny(self.layer_2)
#         self.canny_layer_2_add = CannyAdd(self.layer_2, self.canny_layer_2)
#
#         self.canny_layer_3 = Canny(self.layer_3)
#         self.canny_layer_3_add = CannyAdd(self.layer_3, self.canny_layer_3)
#
#         self.canny_layer_4 = Canny(self.layer_4)
#         self.canny_layer_4_add = CannyAdd(self.layer_4, self.canny_layer_4)
#         # self.fusion = Fusion()
#         self.multi_scale_input_1 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             CBR(1, 16, kernel_size=1, stride=1, padding=0),
#             CoordAttention(16, 16),
#         )
#         self.multi_scale_input_2 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             CBR(16, 32, kernel_size=1, stride=1, padding=0),
#             CoordAttention(32, 32),
#
#         )
#         self.multi_scale_input_3 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             CBR(32, 64, kernel_size=1, stride=1, padding=0),
#             CoordAttention(64, 64),
#         )
#
#     def forward(self, x):
#         multi_scale_input_1 = self.multi_scale_input_1(x)
#         multi_scale_input_2 = self.multi_scale_input_2(multi_scale_input_1)
#         multi_scale_input_3 = self.multi_scale_input_3(multi_scale_input_2)
#
#         features = []
#         pre = self.layer_pre(x)
#
#         canny_layer_pre = self.canny_layer_pre.process(pre)
#         canny_layer_pre_add = self.canny_layer_pre_add.process(pre, canny_layer_pre)
#
#         features.append(canny_layer_pre_add)
#
#         x = self.layer_1(x)
#
#         canny_layer_1 = self.canny_layer_pre.process(x)
#         canny_layer_1_add = self.canny_layer_1_add.process(x, canny_layer_1)
#
#         features.append(canny_layer_1_add)  # (256, 256, 64)   128
#         # x = self.pool1(x)   # skip--->1
#
#         x = self.layer_2(x + multi_scale_input_1)  # 1
#
#         canny_layer_2 = self.canny_layer_2.process(x)
#         canny_layer_2_add = self.canny_layer_2_add.process(x, canny_layer_2)
#         features.append(canny_layer_2_add)  # (128, 128, 128)  64
#
#         x = self.layer_3(x + multi_scale_input_2)
#
#         canny_layer_3 = self.canny_layer_3.process(x)
#         canny_layer_3_add = self.canny_layer_3_add.process(x, canny_layer_3)
#         features.append(canny_layer_3_add)  # (64, 64, 256)    32
#
#         x = self.layer_4(x + multi_scale_input_3)  # 3  # skip--->3
#
#         canny_layer_4 = self.canny_layer_4.process(x)
#         canny_layer_4_add = self.canny_layer_4_add.process(x, canny_layer_4)
#
#         return canny_layer_4_add, features
#
#
# #
# # canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
# # for i in range(x_size[0]):
# #     canny[i] = cv2.Canny(im_arr[i],10,100)
# # canny = torch.from_numpy(canny).cuda().float()
#
#
# class GradientAdd():
#     def __init__(self, tensor1, tensor2):
#         self.tensor1 = tensor1
#         self.tensor2 = tensor2
#
#     def process(self, tensor1, tensor2):  # 第2次卷积后张量, 第1次卷积后张量
#         return torch.add(tensor1, tensor2)
#
#
# class CannyAdd():
#     def __init__(self, tensor1, tensor2):
#         self.tensor1 = tensor1
#         self.tensor2 = tensor2
#
#     def process(self, tensor1, tensor2):  # 已经相加后张量,canny处理后张量
#         return torch.add(tensor1, tensor2)
#
#
# class Canny:
#     def __init__(self, tensor):
#         self.tensor = tensor
#
#     def process(self, tensor):  # 已经相加后张量, 第1次卷积后张量
#         tensor = tensor.cpu().detach().numpy()  # 张量转换为数组
#         # tensor().detach().cpu().numpy() 使用cuda时使用此句
#         tensor = (tensor * 255).astype(np.uint8)  # float32转换为uint8
#         num1 = tensor.shape[0]  # 类别数
#         num2 = tensor.shape[1]  # 通道数
#
#         image1 = tensor[0, 0, :, :]  # 抽取第1个类别的第1层矩阵,会将矩阵降维成二维矩阵
#         image1 = cv2.Canny(image1, 20, 60)  # 对单一通道的矩阵进行canny处理
#         image1 = image1[None, None, :, :]  # 数组升维
#         for i in range(1, num2):  # tensor.shape[1]为通道数
#             img = tensor[0, i, :, :]  # 抽取第1个类别的第1层矩阵,会将矩阵降维成二维矩阵
#             img = cv2.Canny(img, 20, 60)  # Canny函数要求图像为uint8
#             image1 = np.insert(image1, image1.shape[1], img, axis=1)
#
#         for x in range(1, num1):
#             image2 = tensor[x, 0, :, :]  # 抽取第1个类别的第1层矩阵,会将矩阵降维成二维矩阵
#             image2 = cv2.Canny(image2, 20, 60)  # 对单一通道的矩阵进行canny处理
#             image2 = image2[None, :, :]  # 升维至3维矩阵,方便拼接其余的2维矩阵
#             for i in range(1, num2):  # tensor.shape[1]为通道数
#                 img = tensor[x, i, :, :]  # 抽取第x个bs内的第i个2维矩阵
#                 img = cv2.Canny(img, 20, 60)  # Canny函数要求图像为uint8
#                 image2 = np.insert(image2, image2.shape[0], img, axis=0)  # image2为3维矩阵,需要拼接2维矩阵
#             image1 = np.insert(image1, image1.shape[0], image2, axis=0)  # image1为4维矩阵,需要拼接3维矩阵
#             # 2维矩阵可以拼接到4维矩阵上任意一个维度;4维矩阵也可以拼接到4维矩阵上任意一个维度
#         image1 = image1.astype(np.float32) / 255
#         image1 = torch.from_numpy(image1).cuda().float()
#         # torch.from_numpy(image1).cuda(0) 使用cuda时使用此句
#         return image1
#
#
# class U_decoder(nn.Module):
#     def __init__(self):
#         super(U_decoder, self).__init__()
#         self.res_1 = nn.Sequential(
#             CBR(192, 128, kernel_size=3, stride=1, padding=1),
#         )
#         self.res_2 = nn.Sequential(
#             CBR(160, 64, kernel_size=3, stride=1, padding=1),
#         )
#         self.res_3 = nn.Sequential(
#             CBR(80, 32, kernel_size=3, stride=1, padding=1),
#         )
#         self.res_4 = nn.Sequential(
#             CBR(48, 32, kernel_size=3, stride=1, padding=1),
#
#         )
#
#         self.gate_1 = MSA_1(128, 64)
#         self.gate_2 = MSA_1(128, 32)
#         self.gate_3 = MSA_1(64, 16)
#         self.gate_4 = MSA_1(32, 16)
#
#     def forward(self, x, feature):
#         x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
#         x = self.res_1(torch.cat((x, self.gate_1(feature[3], x)), dim=1))
#         x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
#         x = self.res_2(torch.cat((x, self.gate_2(feature[2], x)), dim=1))
#         x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
#         x = self.res_3(torch.cat((x, self.gate_3(feature[1], x)), dim=1))
#         x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)
#         x = self.res_4(torch.cat((x, self.gate_4(feature[0], x)), dim=1))
#         return x
#
#
# class my_unet(pl.LightningModule):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  extra_gap_weight: float,
#                  learning_rate: float = 1.0e-3,
#                  loss_func: Callable = nn.CrossEntropyLoss(),
#                  total_iterations: int = 1000,
#                  ):
#         super(my_unet, self).__init__()
#         self.u_encoder = U_encoder()
#         self.u_decoder = U_decoder()
#         self.SegmentationHead = nn.Conv2d(32, out_channels, 1, bias=False)
#
#         self.extra_gap_weight = extra_gap_weight
#         self.learning_rate = learning_rate
#         self.loss_func = loss_func
#         self.total_iterations = total_iterations
#
#     def forward(self, x):
#
#         encoder_x_first, encoder_skips_first = self.u_encoder(x)
#         decoder_x_first = self.u_decoder(encoder_x_first, encoder_skips_first)
#
#         x = self.SegmentationHead(decoder_x_first)
#         return x
#
#     def training_step(self, batch, batch_idx):
#         loss = None
#         if self.extra_gap_weight is None:
#             loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=False)
#         else:
#             loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                            use_sliding_window=False)
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss = None
#         if self.extra_gap_weight is None:
#             loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=True)
#         else:
#             loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                            use_sliding_window=True)
#         self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
#         return loss
#
#     def optimizer_step(
#             self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
#             on_tpu=False, using_native_amp=False, using_lbfgs=False,
#     ):
#         initial_learning_rate = self.learning_rate
#         current_iteration = self.trainer.global_step
#         total_iteration = self.total_iterations
#         for pg in optimizer.param_groups:
#             pg['lr'] = myut.poly_learning_rate(initial_learning_rate, current_iteration, total_iteration)
#         optimizer.step(closure=optimizer_closure)
#
#     def configure_optimizers(self):
#         return myut.configure_optimizers(self, self.learning_rate)


#
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
#
# class CoordAttention(nn.Module):
#
#     def __init__(self, in_channels, out_channels, reduction=4):
#         super(CoordAttention, self).__init__()
#         self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))
#         self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))
#         # self.pool_avg = nn.AdaptiveAvgPool2d((1, 1))
#         self.gamma = nn.Parameter(torch.zeros(1))
#         # self.mlp = conv_1X1(in_channels,out_channels)
#         # self.conv_1x1 = conv_1X1(in_channels, out_channels)
#
#         temp_c = max(8, out_channels // reduction)
#         self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=7, stride=1, padding=3, bias=False)
#
#         self.bn1 = nn.BatchNorm2d(temp_c)
#         self.act1 = h_swish()
#
#         self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
#         self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
#         # self.spatial = SpatialAttention()
#         # self.sa = self_attention(out_channels, out_channels)
#
#     def forward(self, x):
#         short = x
#         n, c, H, W = x.shape
#         x_h = self.pool_h_avg(x)
#         x_w = self.pool_w_avg(x)
#         # x_y = self.pool_avg(x)
#         x_w = x_w.permute(0, 1, 3, 2)
#         x_cat = torch.cat([x_h, x_w], dim=2)
#         out = self.act1(self.bn1(self.conv1(x_cat)))
#         x_h, x_w = torch.split(out, [H, W], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#         out_h = torch.sigmoid(self.conv2(x_h))
#         out_w = torch.sigmoid(self.conv3(x_w))
#         out = short * out_w * out_h + short * 0.1
#         # out = self.sa(out) + out
#         return F.relu(out)
#
#
# class eca_layer(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#
#     def __init__(self, in_channels, out_channels, gamma=2, b=1):
#         super(eca_layer, self).__init__()
#         kernel_size = int(abs((math.log(out_channels, 2) + b) / gamma))
#         kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # x: input features with shape [b, c, h, w]
#         b, c, h, w = x.size()
#
#         # feature descriptor on the global spatial information
#         y = self.avg_pool(x)
#
#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
#
#         return x * y.expand_as(x) + x
#
#
# class MSA_1(nn.Module):
#     def __init__(self):
#         super(MSA_1, self).__init__()
#
#         self.Up_conv = nn.Sequential(
#             UNetDecoder(512, 128),
#         )
#         self.psi = nn.Sequential(
#             AxialAttention(128, 128)
#         )
#
#     def forward(self, g, x):
#         g0 = self.Up_conv(g)
#         psi = self.psi(g0 + x)
#         return x * psi
#
#
# class MSA_2(nn.Module):
#     def __init__(self):
#         super(MSA_2, self).__init__()
#
#         self.Up_conv = nn.Sequential(
#             UNetDecoder(128, 128),
#         )
#         self.psi = nn.Sequential(
#             AxialAttention(128, 128)
#         )
#
#     def forward(self, g, x, y):
#         g0 = self.Up_conv(g)
#         psi = self.psi(g0 + x + y)
#         return x * psi
#
#
# class MSA_3(nn.Module):
#     def __init__(self):
#         super(MSA_3, self).__init__()
#
#         self.Up_conv = nn.Sequential(
#             UNetDecoder(128, 64),
#         )
#         self.psi = nn.Sequential(
#             AxialAttention(64, 64)
#         )
#
#     def forward(self, g, x, y):
#         g0 = self.Up_conv(g)
#         g0 = myut.get_adjusted_feature_map_s2b(g0, x)
#         y0 = myut.get_adjusted_feature_map_s2b(y, x)
#         psi = self.psi(g0 + x + y0)
#         return x * psi
#
#
# """
#     Author: Mingle Xu
#     Time: 18 Mar, 2019
#     Time: 2020-04-08 22:04:07 revise something
# """
#
# import torch
# import torch.nn as nn
#
#
# class self_attention(nn.Module):
#     r"""
#         Create global dependence.
#         Source paper: https://arxiv.org/abs/1805.08318
#     """
#
#     def __init__(self, in_channels, out_channels, reduction=8):
#         super(self_attention, self).__init__()
#         self.in_channels = in_channels
#
#         self.q = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // reduction, kernel_size=1)
#         self.k = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // reduction, kernel_size=1)
#         self.v = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
#         self.softmax_ = nn.Softmax(dim=2)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#         self.init_weight(self.q)
#         self.init_weight(self.k)
#         self.init_weight(self.v)
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#
#         assert channels == self.in_channels
#
#         q = self.q(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B * (H * W) * C//8
#         k = self.k(x).view(batch_size, -1, height * width)  # B * C//8 * (H * W)
#
#         attention = torch.bmm(q, k)  # B * (H * W) * (H * W)
#         attention = self.softmax_(attention)
#
#         v = self.v(x).view(batch_size, channels, -1)  # B * C * (H * W)
#
#         self_attention_map = torch.bmm(v, attention).view(batch_size, channels, height, width)  # B * C * H * W
#
#         return self.gamma * self_attention_map + x
#
#     def init_weight(self, conv):
#         nn.init.kaiming_uniform_(conv.weight)
#         if conv.bias is not None:
#             conv.bias.data.zero_()
#
#
# class RowAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction=8):
#         super(RowAttention, self).__init__()
#         inter_channels = in_channels // 2
#         self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
#         self.softmax_ = nn.Softmax(dim=2)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         Q = self.query_conv(x)
#         K = self.key_conv(x)
#         V = self.value_conv(x)
#
#         Q = Q.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
#         K = K.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
#         V = V.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
#
#         row_attention = torch.bmm(Q, K)
#
#         row_attention = self.softmax_(row_attention)
#
#         out = torch.bmm(V, row_attention.permute(0, 2, 1))
#
#         out = out.view(b, h, -1, w).permute(0, 2, 1, 3)
#
#         return self.gamma * out + x
#
#
# class ColAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction=8):
#         super(ColAttention, self).__init__()
#         self.in_channels = in_channels
#         inter_channels = out_channels // 2
#         self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
#         self.softmax_ = nn.Softmax(dim=2)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#
#         Q = self.query_conv(x)
#         K = self.key_conv(x)
#         V = self.value_conv(x)
#
#         Q = Q.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
#         K = K.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
#         V = V.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
#
#         col_attention = torch.bmm(Q, K)
#
#         col_attention = self.softmax_(col_attention)
#
#         out = torch.bmm(V, col_attention.permute(0, 2, 1))
#
#         out = out.view(b, w, -1, h).permute(0, 2, 3, 1)
#
#         return self.gamma * out + x
#
#
# class AxialAttention(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(AxialAttention, self).__init__()
#         self.row_attention = RowAttention(in_channels, out_channels)
#         self.col_attention = ColAttention(out_channels, out_channels)
#
#     def forward(self, x):
#         x1 = self.row_attention(x)
#         return self.col_attention(x1)
#
#
# class DEPTHWISECONV(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DEPTHWISECONV, self).__init__()
#         # 也相当于分组为1的分组卷积
#         self.depth_conv = nn.Conv2d(in_channels=in_channels,
#                                     out_channels=out_channels,
#                                     kernel_size=3,
#                                     stride=1,
#                                     padding=1,
#                                     groups=out_channels)
#         self.point_conv = nn.Conv2d(in_channels=in_channels,
#                                     out_channels=out_channels,
#                                     kernel_size=1,
#                                     stride=1,
#                                     padding=0,
#                                     groups=1)
#
#     def forward(self, input):
#         out = self.depth_conv(input)
#         out = self.point_conv(out)
#         return out
#
#
# class conv_1X1(nn.Module):
#     '''
#     使用conv_1X1 改变维度通道数
#     '''
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  strides: int = 1,
#                  num_groups=32
#                  ):
#         super(conv_1X1, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
#             # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class conv_3X3(nn.Module):
#     '''
#     使用conv_3X3 改变维度通道数
#     '''
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: int = 3,
#                  stride: int = 1,
#                  padding: int = 1,
#                  num_groups=32,
#                  ):
#         super(conv_3X3, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# class conv_7X7(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  num_groups=32,
#                  ):
#         super(conv_7X7, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
#             # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# # 定义残差块ResBlock
# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation=[1, 1], stride=1, groups=1, num_groups=32):
#         super(ResBlock, self).__init__()
#         # 这里定义了残差块内连续的2个卷积层
#         self.left = nn.Sequential(
#             # RepVGGBlock(in_channels, out_channels, stride=stride, groups=groups),
#             # # # CoordAttention(out_channels, out_channels),
#             # RepVGGBlock(out_channels, out_channels, stride=1, groups=groups),
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation[0],
#                       dilation=dilation[0],
#                       groups=groups,
#                       bias=False),
#             # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
#             nn.BatchNorm2d(out_channels),
#             # nn.LeakyReLU(0.1, inplace=True),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.2),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation[1], dilation=dilation[1],
#                       groups=groups,
#                       bias=False),
#             # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
#             nn.BatchNorm2d(out_channels),
#             # nn.LeakyReLU(0.1, inplace=True),
#             nn.ReLU(inplace=True),
#             # nn.Dropout(0.2),
#         )
#
#         self.shortcut = nn.Sequential()
#         # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
#         self.shortcut = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#             nn.BatchNorm2d(out_channels),
#             # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
#             # nn.LeakyReLU(0.1, inplace=True),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         out = self.left(x)
#         # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
#         out = out + self.shortcut(x)
#         return out
#
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         # 共享权重的MLP
#         self.fc1 = nn.Conv2d(in_channels, out_channels // 16, 1, bias=False)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Conv2d(out_channels // 16, out_channels, 1, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
#         max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
#         out = avg_out + max_out
#         return self.sigmoid(out) * x
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, out_channels, kernel_size=7, ):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, out_channels, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x) * x
#
#
# class FPN(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(FPN, self).__init__()
#         self.con_1 = conv_1X1(in_channels, 128)
#         self.con_2 = conv_1X1(out_channels, 128)
#         self.conv_output = conv_1X1(128, in_channels)
#         # self.ca = CoordAttention(in_channels, in_channels)
#         # self.gama = 0.2
#
#     def forward(self, x, y):
#         size = x.shape[2:]
#         x = self.con_1(x)
#         y = self.con_2(y)
#         y = F.interpolate(y, size=size, mode='bilinear', align_corners=True)
#         out = self.conv_output(x + y)
#         return F.relu(out)
#
#
# class PyramidPooling(nn.Module):
#     """Pyramid pooling module"""
#
#     def __init__(self, in_channels, out_channels):
#         super(PyramidPooling, self).__init__()
#         inter_channels = int(in_channels / 4)  # 这里N=4与原文一致
#         self.conv1 = conv_1X1(in_channels, inter_channels)  # 四个1x1卷积用来减小channel为原来的1/N
#         self.conv2 = conv_1X1(in_channels, inter_channels)
#         self.conv3 = conv_1X1(in_channels, inter_channels)
#         self.conv4 = conv_1X1(in_channels, inter_channels)
#         # self.ca_1 = CoordAttention(inter_channels, inter_channels)
#         # self.ca_2 = CoordAttention(inter_channels, inter_channels)
#         # self.ca_3 = CoordAttention(inter_channels, inter_channels)
#         # self.ca_4 = CoordAttention(inter_channels, inter_channels)
#         self.out = conv_1X1(in_channels * 2, out_channels)  # 最后的1x1卷积缩小为原来的channel
#
#     def pool(self, x, size):
#         avgpool = nn.AdaptiveAvgPool2d(size)  # 自适应的平均池化，目标size分别为1x1,2x2,3x3,6x6
#         return avgpool(x)
#
#     def upsample(self, x, size):  # 上采样使用双线性插值
#         return F.interpolate(x, size, mode='bilinear', align_corners=True)
#
#     def forward(self, x):
#         short = x
#         size = x.size()[2:]
#         feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
#         feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
#         feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
#         feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
#         x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)  # concat 四个池化的结果
#         return F.relu(self.out(x))
#
#
# def autopad(kernel, padding=None):  # kernel, padding
#     # Pad to 'same'
#     if padding is None:
#         padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]  # auto-pad
#     return padding
#
#
# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, in_channels, out_channels, kernel=1, stride=1, padding=None, groups=1, act=True,
#                  num_groups=16):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, autopad(kernel, padding), groups=groups,
#                               bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = nn.ReLU(inplace=True) if act else nn.Identity()
#
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))
#
#     def fuseforward(self, x):
#         return self.act(self.conv(x))
#
#
# class GhostConv(nn.Module):
#     # Ghost Convolution https://github.com/huawei-noah/ghostnet
#     def __init__(self, in_channels, out_channels, kernel=1, stride=1, groups=1,
#                  act=True):  # ch_in, ch_out, kernel, stride, groups
#         super(GhostConv, self).__init__()
#         inter_channels = out_channels // 2  # hidden channels
#         self.conv_1 = Conv(in_channels, inter_channels, kernel=3, stride=stride, padding=None, groups=groups, act=act)
#         self.conv_2 = Conv(inter_channels, inter_channels, kernel=3, stride=1, padding=None, groups=inter_channels,
#                            act=act)  # 也可以改成3x3
#
#     def forward(self, x):
#         y = self.conv_1(x)
#         return torch.cat([y, self.conv_2(y)], 1)
#
#
# class RepVGGBlock(nn.Module):  # RepVGG Block
#     def __init__(self, in_channels, out_channels, stride, deploy=False, num_groups=32, groups=1, act=True):
#         super(RepVGGBlock, self).__init__()
#         self.out_channels = out_channels
#         self.conv3x3 = Conv(in_channels, out_channels, kernel=3, stride=stride, groups=groups, padding=None,
#                             act=act)  # 3x3ConvBN分支
#         self.conv1x1 = Conv(in_channels, out_channels, kernel=1, stride=stride, groups=groups, padding=None,
#                             act=act)  # 1x1ConvBN分支
#         if stride == 1 and in_channels == out_channels:
#             self.bn = nn.BatchNorm2d(out_channels)
#         #               self.bn =  nn.BatchNorm2d(out_channels)
#         else:
#             self.bn = None  # BN分支(下采样层没有BN分支)
#
#     def forward(self, x):
#         if self.bn is not None:
#             return self.conv3x3(x) + self.conv1x1(x) + self.bn(x)
#         else:
#             return self.conv3x3(x) + self.conv1x1(x)
#
#
# class UNetEncoder(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, dilation=[1, 1], left: bool = True, is_aspp: bool = False,
#                  groups=1):
#         super(UNetEncoder, self).__init__()
#         # 这里定义了残差块内连续的2个卷积层
#         if left is True:
#             if is_aspp is False:
#                 self.layer_1 = nn.Sequential(
#                     ResBlock(in_channels, out_channels, stride=stride, dilation=dilation),
#                     # AxialAttention(out_channels, out_channels),
#                     # CoordAttention(out_channels, out_channels),
#                     AxialAttention(out_channels, out_channels),
#                     ResBlock(out_channels, out_channels, stride=1, dilation=dilation),
#                     # ResBlock(out_channels, out_channels, stride=1, dilation=dilation),
#
#                 )
#             else:
#                 self.layer_1 = nn.Sequential(
#                     ResBlock(in_channels, out_channels, stride=1, dilation=dilation),
#                     AxialAttention(out_channels, out_channels),
#                     ResBlock(out_channels, out_channels, stride=1, dilation=dilation),
#                     # CoordAttention(out_channels, out_channels),
#                     # AxialAttention(out_channels, out_channels),
#                 )
#         else:
#             self.layer_1 = nn.Sequential(
#                 ResBlock(in_channels, out_channels, stride=1, dilation=dilation),
#                 # AxialAttention(out_channels, out_channels),
#                 # RepVGGBlock(in_channels, out_channels, stride=1, groups=groups),
#             )
#
#     def forward(self, x):
#         return self.layer_1(x)
#
#
# class UNetDecoder(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=2,
#                  strides=2,
#                  num_groups=32):
#         super(UNetDecoder, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, bias=True, ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#             # nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
#             # nn.LeakyReLU(0.1, inplace=True),
#         )
#
#         # self.conv1 = nn.Sequential(
#         #     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
#         #     nn.BatchNorm2d(out_channels),
#         #     nn.ReLU(inplace=True),
#         #     #                  nn.GroupNorm(num_groups,out_channels),
#         #     #                  nn.SiLU(inplace=True),
#         # )
#
#     def forward(self, x):
#         # return self.conv1(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))
#         return self.conv1(x)
#
#
# class my_unet(pl.LightningModule):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  extra_gap_weight: float,
#                  learning_rate: float = 1.0e-3,
#                  loss_func: Callable = nn.CrossEntropyLoss(),
#                  total_iterations: int = 1000,
#                  ):
#         super(my_unet, self).__init__()
#         self.u_encoder = U_encoder()
#         self.u_decoder = U_decoder()
#
#         self.SegmentationHead = nn.Conv2d(64, out_channels, 1)
#
#         self.extra_gap_weight = extra_gap_weight
#         self.learning_rate = learning_rate
#         self.loss_func = loss_func
#         self.total_iterations = total_iterations
#
#     def forward(self, x):
#         identity = x
#         x = myut.get_adjusted_feature_map_b2s_pre(x, x)
#         x, skips = self.u_encoder(x)
#         x = self.u_decoder(x, skips)
#         x = self.SegmentationHead(x)
#         x = myut.get_adjusted_feature_map_s2b(x, identity)
#         return x
#         # self.layer_pre_conv = conv_3X3(in_channels, 64)
#         #
#         # self.multi_scale_input = conv_3X3(in_channels, 128)
#         # self.multi_scale_input_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # self.multi_scale_input_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # self.multi_scale_input_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         #
#         # self.layer1_conv = conv_7X7(in_channels, 128)
#         # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # self.layer2_conv = UNetEncoder(128, 64)
#         #
#         # self.layer3_conv = UNetEncoder(64, 128, stride=2)
#         #
#         # self.layer4_conv = UNetEncoder(128, 256, [1, 2], is_aspp=True)
#         # self.layer5_conv = UNetEncoder(256, 512, [1, 3], is_aspp=True)
#         # # self.layer5_conv_1 = conv_1X1(896, 512)
#         # # self.conv_1x1_1 = conv_1X1(128, 512)
#         # # self.conv_1x1_2 = conv_1X1(512, 128)
#         # # self.msa_1 = MSA_1()  # 512 -->128
#         # # self.msa_2 = MSA_2()  # 128 -->128
#         # # self.msa_3 = MSA_3()  # 128 -->64
#         # # self.layer4_conv = ASPP(128, 256)
#         # # self.layer5_conv = ASPP(256, 512)
#         #
#         # # self.attention_1 = CoordAttention(64, 64)
#         # # self.attention_2 = CoordAttention(128, 128)
#         # # self.attention_3 = CoordAttention(64, 64)
#         # # self.aspp = ASPP(512, 512)
#         # #
#         # # self.fpn_3 = FPN(64, 128)
#         # # self.fpn_2 = FPN(128, 64)
#         # # self.fpn_1 = FPN(64, 128)
#         #
#         # self.layer6_conv = UNetEncoder(320, 256, left=False)
#         # self.layer7_conv = UNetEncoder(256, 128, left=False)
#         # self.layer8_conv = UNetEncoder(128, 64, left=False)
#         # # self.layer8_conv = last_layer(128, 64)
#         #
#         # self.classify_conv = ChannelPad(2, 64, out_channels,
#         #                                 'project')  # number of spatial dimensions of the input image is 2
#         # self.deconv1 = UNetDecoder(512, 256, )  # 64 + 256 = 320
#         # self.deconv2 = UNetDecoder(256, 128, )  # 128 + 128 = 256
#         # self.deconv3 = UNetDecoder(128, 64, )  # 64 + 64 = 128
#
#         # self.extra_gap_weight = extra_gap_weight
#         # self.learning_rate = learning_rate
#         # self.loss_func = loss_func
#         # self.total_iterations = total_iterations
#
#     # def forward(self, x) -> torch.Tensor:
#     #
#     #     #
#     #     identity = myut.get_adjusted_feature_map_b2s_pre(x, x)
#     #     Encoder_layer_pre = self.layer_pre_conv(identity)
#     #     multi_scale_input = self.multi_scale_input(identity)
#     #     multi_scale_inputs_1 = self.multi_scale_input_1(multi_scale_input)
#     #     multi_scale_inputs_2 = self.multi_scale_input_2(multi_scale_inputs_1)
#     #     multi_scale_inputs_3 = self.multi_scale_input_3(multi_scale_inputs_2)
#     #
#     #     Encoder_layer1 = self.layer1_conv(identity) + multi_scale_inputs_1  #
#     #
#     #     Encoder_layer2 = self.layer2_conv(self.maxpool(Encoder_layer1) + multi_scale_inputs_2)  #
#     #     Encoder_layer3 = self.layer3_conv(Encoder_layer2) + multi_scale_inputs_3  #
#     #     Encoder_layer4 = self.layer4_conv(Encoder_layer3)
#     #     Encoder_layer5 = self.layer5_conv(Encoder_layer4)
#     #     # Encoder_layer5_1 = self.conv_1x1_1(Encoder_layer3) + Encoder_layer5
#     #     # Encoder_layer5_aspp = self.aspp(Encoder_layer5)
#     #
#     #     Decoder_layer4 = self.deconv1(Encoder_layer5)
#     #     # msa_1 = self.msa_1(Encoder_layer5, Encoder_layer2)
#     #     # print("msa_1", msa_1.shape)
#     #     # print("Encoder_layer1", Encoder_layer1.shape)
#     #     Decoder_layer4 = torch.cat([Encoder_layer2, Decoder_layer4], dim=1)
#     #     Decoder_layer4 = self.layer6_conv(Decoder_layer4)
#     #
#     #     Decoder_layer3 = self.deconv2(Decoder_layer4)
#     #
#     #     # msa_2 = self.msa_2(msa_1, Encoder_layer1, Decoder_layer3)
#     #
#     #     Decoder_layer3 = torch.cat([Encoder_layer1, Decoder_layer3], dim=1)  # fpn_2
#     #     Decoder_layer3 = self.layer7_conv(Decoder_layer3)
#     #     #
#     #     Decoder_layer2 = self.deconv3(Decoder_layer3)
#     #     # msa_3 = self.msa_3(msa_2, Encoder_layer_pre, Decoder_layer2)
#     #     Decoder_layer2 = torch.cat([Encoder_layer_pre, Decoder_layer2],
#     #                                dim=1)  # self.conv_1x1_2(fpn_1)
#     #     Decoder_layer2 = myut.get_adjusted_feature_map_s2b(Decoder_layer2, x)
#     #     Decoder_layer2 = self.layer8_conv(Decoder_layer2)
#     #
#     #     Decoder_layer1 = self.classify_conv(Decoder_layer2)
#     #
#     #     return Decoder_layer1
#
#     def training_step(self, batch, batch_idx):
#         loss = None
#         if self.extra_gap_weight is None:
#             loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=False)
#         else:
#             loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                            use_sliding_window=False)
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
#         # self.log('train_loss', loss, on_step=True, on_epoch=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss = None
#         if self.extra_gap_weight is None:
#             loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=True)
#         else:
#             loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                            use_sliding_window=True)
#         self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
#         # self.log('val_loss', loss, on_epoch=True)
#         return loss
#
#     def optimizer_step(
#             self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
#             on_tpu=False, using_native_amp=False, using_lbfgs=False,
#     ):
#         initial_learning_rate = self.learning_rate
#         current_iteration = self.trainer.global_step
#         total_iteration = self.total_iterations
#         for pg in optimizer.param_groups:
#             pg['lr'] = myut.poly_learning_rate(initial_learning_rate, current_iteration, total_iteration)
#         optimizer.step(closure=optimizer_closure)
#
#     def configure_optimizers(self):
#         return myut.configure_optimizers(self, self.learning_rate)


# class PyramidPool(nn.Module):
#     def __init__(self, in_channels, out_channels, pool_size):
#         super(PyramidPool, self).__init__()
#         self.features = nn.Sequential(
#             nn.AdaptiveAvgPool2d(pool_size),
#             nn.Conv2d(in_channels, out_channels, 1, bias=True),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         size = x.size()
#         output = F.upsample_bilinear(self.features(x), size[2:])
#         out = output
#         return out
#
#
# class PSPPooling(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(PSPPooling, self).__init__()
#
#         self.layer5a = PyramidPool(in_channels, out_channels, 1)
#         self.layer5b = PyramidPool(in_channels, out_channels, 2)
#         self.layer5c = PyramidPool(in_channels, out_channels, 3)
#         self.layer5d = PyramidPool(in_channels, out_channels, 6)
#
#         self.final = nn.Sequential(
#             nn.Conv2d(in_channels * 5, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         size = x.size()
#         x = self.final(torch.cat([
#             x,
#             self.layer5a(x),
#             self.layer5b(x),
#             self.layer5c(x),
#             self.layer5d(x),
#         ], dim=1))
#
#         return F.upsample_bilinear(x, size[2:])
#
#
# class PyramidPooling(nn.Module):
#     """
#     Reference:
#         Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
#     """
#
#     def __init__(self, in_channels, norm_layer, up_kwargs):
#         super(PyramidPooling, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d(1)
#         self.pool2 = nn.AdaptiveAvgPool2d(2)
#         self.pool3 = nn.AdaptiveAvgPool2d(3)
#         self.pool4 = nn.AdaptiveAvgPool2d(6)
#
#         out_channels = int(in_channels / 4)
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    norm_layer(out_channels),
#                                    nn.ReLU(True))
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    norm_layer(out_channels),
#                                    nn.ReLU(True))
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    norm_layer(out_channels),
#                                    nn.ReLU(True))
#         self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                    norm_layer(out_channels),
#                                    nn.ReLU(True))
#         # bilinear interpolate options
#         self._up_kwargs = up_kwargs
#
#     def forward(self, x):
#         _, _, h, w = x.size()
#         feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
#         feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
#         feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
#         feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
#         return torch.cat((x, feat1, feat2, feat3, feat4), 1)
#
#
# class StripPooling(nn.Module):
#     def __init__(self, in_channels, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
#         super(StripPooling, self).__init__()
#         self.pool1 = nn.AdaptiveAvgPool2d((1, None))  # 1*W
#         self.pool2 = nn.AdaptiveAvgPool2d((None, 1))  # H*1
#         inter_channels = int(in_channels / 4)
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
#                                    nn.BatchNorm2d(inter_channels),
#                                    nn.ReLU(True))
#         self.conv2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
#                                    nn.BatchNorm2d(inter_channels))
#         self.conv3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
#                                    nn.BatchNorm2d(inter_channels))
#         self.conv4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
#                                    nn.BatchNorm2d(inter_channels),
#                                    nn.ReLU(True))
#         self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, in_channels, 1, bias=False),
#                                    nn.BatchNorm2d(in_channels))
#         self._up_kwargs = up_kwargs
#
#     def forward(self, x):
#         _, _, h, w = x.size()
#         x1 = self.conv1(x)
#         x2 = F.interpolate(self.conv2(self.pool1(x1)), (h, w), **self._up_kwargs)  # 结构图的1*W的部分
#         x3 = F.interpolate(self.conv3(self.pool2(x1)), (h, w), **self._up_kwargs)  # 结构图的H*1的部分
#         x4 = self.conv4(F.relu_(x2 + x3))  # 结合1*W和H*1的特征
#         out = self.conv5(x4)
#         return F.relu_(x + out)  # 将输出的特征与原始输入特征结合
#
#
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
#
# class CoordAttention(nn.Module):
#
#     def __init__(self, in_channels, out_channels, reduction=4):
#         super(CoordAttention, self).__init__()
#         self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
#         temp_c = max(8, in_channels // reduction)
#         self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)
#
#         self.bn1 = nn.BatchNorm2d(temp_c)
#         self.act1 = h_swish()
#
#         self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
#         self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         short = x
#         n, c, H, W = x.shape
#         x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
#         x_cat = torch.cat([x_h, x_w], dim=2)
#         out = self.act1(self.bn1(self.conv1(x_cat)))
#         x_h, x_w = torch.split(out, [H, W], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#         out_h = torch.sigmoid(self.conv2(x_h))
#         out_w = torch.sigmoid(self.conv3(x_w))
#         out = short * out_w * out_h + short
#         return out
#
#
# class SE_attention(nn.Module):
#     def __init__(self, channel, ratio=4):
#         super(SE_attention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // ratio, False),
#             nn.ReLu(),
#             nn.Linear(channel // ratio, channel, False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         avg = self.avg_pool(x).view([b, c])  # b,c,h,w->b,c
#         fc = self.fc(avg).view([b, c, 1, 1])  # b,c->b,c//ratio->b,c->b,c,1,1
#         return x * fc
#
#
# class BasicConv(nn.Module):
#
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         if bn:
#             self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                                   dilation=dilation, groups=groups, bias=False)
#             self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
#             self.relu = nn.ReLU(inplace=True) if relu else None
#         else:
#             self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                                   dilation=dilation, groups=groups, bias=True)
#             self.bn = None
#             self.relu = nn.ReLU(inplace=True) if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
#
# class MDRB(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, groups=1):
#         super(MDRB, self).__init__()
#         out_channel = out_channels // 4
#         self.con_1x1 = nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1, bias=False)
#         self.branch0 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False,
#                       groups=groups),
#             nn.BatchNorm2d(out_channel),
#             nn.ReLU(),
#         )
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=3, dilation=3, bias=False,
#                       groups=groups),
#             nn.BatchNorm2d(out_channel),
#         )
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=5, dilation=5, bias=False,
#                       groups=groups),
#             nn.BatchNorm2d(out_channel),
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=7, dilation=7, bias=False,
#                       groups=groups),
#             nn.BatchNorm2d(out_channel),
#         )
#         self.relu = nn.ReLU(inplace=False)
#         self.CoordAttention_layer = CoordAttention(in_channels, out_channels)
#
#     def forward(self, x):
#         identity = self.con_1x1(x)
#         x0 = self.branch0(identity)
#         x1 = self.branch1(identity)
#         x2 = self.branch2(identity)
#         x3 = self.branch3(identity)
#         out = torch.cat((x0, x1, x2, x3), 1)
#         out = myut.get_adjusted_feature_map_small_big(out, x)
#         out = out + x
#         out = self.relu(out)
#         return out
#
#
# class Channel_Attention(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super(Channel_Attention, self).__init__()
#         self.bn = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         residual = x
#         x = self.bn(x)
#         weight_bn = self.bn.weight.data.abs() / torch.sum(self.bn.weight.data.abs())
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = torch.mul(weight_bn, x)
#         x = x.permute(0, 3, 2, 1).contiguous()
#         out = torch.sigmoid(x) * residual + residual
#         return out
#
#
# class NAMAttention(nn.Module):
#
#     def __init__(self, in_channels, out_channels, reduction=4):
#         super(NAMAttention, self).__init__()
#         self.channel_att = Channel_Attention(out_channels, out_channels)
#
#     def forward(self, x):
#         out = self.channel_att(x)
#         return out
#
#
# class CA_Block(nn.Module):
#     def __init__(self, channel, h, w, reduction=16):
#         super(CA_Block, self).__init__()
#
#         self.h = h
#         self.w = w
#
#         self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
#         self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
#
#         self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
#                                   bias=False)
#
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm2d(channel // reduction)
#
#         self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
#                              bias=False)
#         self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
#                              bias=False)
#
#         self.sigmoid_h = nn.Sigmoid()
#         self.sigmoid_w = nn.Sigmoid()
#
#     def forward(self, x):
#         x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
#         x_w = self.avg_pool_y(x)
#
#         x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
#
#         x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
#
#         s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
#         s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
#
#         out = x * s_h.expand_as(x) * s_w.expand_as(x)
#
#         return out
#
#
# # without bn version
# class ASPP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPP, self).__init__()
#         self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
#         self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)
#         self.atrous_block1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1, dilation=1)
#         self.atrous_block3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=3, dilation=3)
#         self.atrous_block5 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=5, dilation=5)
#
#     def forward(self, x):
#         size = x.shape[2:]
#         image_features = self.mean(x)
#         image_features = self.conv(image_features)
#         image_features = F.upsample(image_features, size=size, mode='bilinear')
#         atrous_block1 = self.atrous_block1(x)
#         atrous_block3 = self.atrous_block3(atrous_block1)
#         atrous_block5 = self.atrous_block5(atrous_block3)
#         output = image_features + atrous_block5
#         return output
#
#
# class sa_layer(nn.Module):
#     """Constructs a Channel Spatial Group module.
#     Args:
#         k_size: Adaptive selection of kernel size
#     """
#
#     def __init__(self, in_channels, groups=2):
#         super(sa_layer, self).__init__()
#         self.groups = groups
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化操作
#         self.cweight = Parameter(torch.zeros(1, in_channels // (2 * groups), 1, 1))  # channel w
#         self.cbias = Parameter(torch.ones(1, in_channels // (2 * groups), 1, 1))  # channel b
#         self.sweight = Parameter(torch.zeros(1, in_channels // (2 * groups), 1, 1))  # Spatial w
#         self.sbias = Parameter(torch.ones(1, in_channels // (2 * groups), 1, 1))  # Spatial b
#
#         self.sigmoid = nn.Sigmoid()
#         self.gn = nn.GroupNorm(in_channels // (2 * groups), in_channels // (2 * groups))  # groupnorm
#
#     @staticmethod
#     def channel_shuffle(x, groups):
#         b, c, h, w = x.shape
#         x = x.reshape(b, groups, -1, h, w)
#         x = x.permute(0, 2, 1, 3, 4)
#         # flatten
#         x = x.reshape(b, -1, h, w)
#         return x
#
#     def forward(self, x):
#         short = x
#         b, c, h, w = x.shape
#         x = x.reshape(b * self.groups, -1, h, w)
#         x_0, x_1 = x.chunk(2, dim=1)
#         # channel attention
#         xn = self.avg_pool(x_0)
#         xn = self.cweight * xn + self.cbias
#         xn = x_0 * self.sigmoid(xn)
#         # spatial attention
#         xs = self.gn(x_1)
#         xs = self.sweight * xs + self.sbias
#         xs = x_1 * self.sigmoid(xs)
#
#         # concatenate along channel axis
#         out = torch.cat([xn, xs], dim=1)
#         out = out.reshape(b, -1, h, w)
#
#         out = self.channel_shuffle(out, 2) + short
#         return out
#
#
# def ConvBNReLU(in_channels, out_channels, kernel_size, stride, groups=1):
#     return nn.Sequential(
#         nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                   padding=kernel_size // 2, groups=groups),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     )
#
#
# def Conv1x1BNReLU(in_channels, out_channels, groups=1):
#     return nn.Sequential(
#         nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, groups=groups),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU6(inplace=True)
#     )
#
#
# class PyConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_sizes, groups, stride=1):
#         super(PyConv, self).__init__()
#         if out_channels is None:
#             out_channels = []
#         assert len(out_channels) == len(kernel_sizes) == len(groups)
#
#         self.pyconv_list = nn.ModuleList()
#         for i in range(len(kernel_sizes)):
#             self.pyconv_list.append(
#                 ConvBNReLU(in_channels=in_channels, out_channels=out_channels[i], kernel_size=kernel_sizes[i],
#                            stride=stride, groups=groups[i]))
#
#     def forward(self, x):
#         outputs = []
#         for pyconv in self.pyconv_list:
#             outputs.append(pyconv(x))
#         return torch.cat(outputs, 1)
#
#
# class PyConv4(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 9], groups=[1, 4, 8, 16], stride=1):
#         super(PyConv4, self).__init__()
#         assert len(kernel_sizes) == len(groups)
#         if in_channels == 1:
#             groups = [1, 1, 1, 1]
#         out_channel = [out_channels // 4, out_channels // 4, out_channels // 4, out_channels // 4]
#         self.pyconv_list = nn.ModuleList()
#         for i in range(len(kernel_sizes)):
#             self.pyconv_list.append(
#                 ConvBNReLU(in_channels=in_channels, out_channels=out_channel[i], kernel_size=kernel_sizes[i],
#                            stride=stride, groups=groups[i]))
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         outputs = []
#         for pyconv in self.pyconv_list:
#             outputs.append(pyconv(x))
#         return torch.cat(outputs, 1) + self.shortcut(x)
#
#
# class PyConv3(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], groups=[1, 4, 8], stride=1):
#         super(PyConv3, self).__init__()
#         assert len(kernel_sizes) == len(groups)
#         out_channel = [out_channels // 4, out_channels // 4, out_channels // 2]
#         self.pyconv_list = nn.ModuleList()
#         for i in range(len(kernel_sizes)):
#             self.pyconv_list.append(
#                 ConvBNReLU(in_channels=in_channels, out_channels=out_channel[i], kernel_size=kernel_sizes[i],
#                            stride=stride, groups=groups[i]))
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         outputs = []
#         for pyconv in self.pyconv_list:
#             outputs.append(pyconv(x))
#         return torch.cat(outputs, 1) + self.shortcut(x)
#
#
# class PyConv2(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5], groups=[1, 4], stride=1):
#         super(PyConv2, self).__init__()
#         assert len(kernel_sizes) == len(groups)
#         out_channel = [out_channels // 2, out_channels // 2]
#         self.pyconv_list = nn.ModuleList()
#         for i in range(len(kernel_sizes)):
#             self.pyconv_list.append(
#                 ConvBNReLU(in_channels=in_channels, out_channels=out_channel[i], kernel_size=kernel_sizes[i],
#                            stride=stride, groups=groups[i]))
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         outputs = []
#         for pyconv in self.pyconv_list:
#             outputs.append(pyconv(x))
#         return torch.cat(outputs, 1) + self.shortcut(x)
#
#
# class PyConv1(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_sizes=3, groups=1, stride=1):
#         super(PyConv1, self).__init__()
#         self.pyconv = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_sizes,
#                                  stride=stride, groups=groups)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         return self.pyconv(x) + self.shortcut(x)
#
#
# class LocalPyConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(LocalPyConv, self).__init__()
#         out_channel = out_channels // 4
#         self._reduce = Conv1x1BNReLU(out_channels, out_channels)
#         self._pyConv = PyConv(in_channels=out_channels,
#                               out_channels=[out_channel, out_channel, out_channel, out_channel],
#                               kernel_sizes=[3, 5, 7, 9], groups=[1, 4, 8, 16])
#         self._combine = Conv1x1BNReLU(out_channels, out_channels)
#
#     def forward(self, x):
#         return self._combine(self._pyConv(self._reduce(x)))
#
#
# class GlobalPyConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(GlobalPyConv, self).__init__()
#         out_channel = out_channels // 4
#         self.global_pool = nn.AdaptiveAvgPool2d(output_size=9)
#         self._reduce = Conv1x1BNReLU(out_channels, out_channels)
#         self._pyConv = PyConv(in_channels=out_channels,
#                               out_channels=[out_channel, out_channel, out_channel, out_channel],
#                               kernel_sizes=[3, 5, 7, 9], groups=[1, 4, 8, 16])
#         self._fuse = Conv1x1BNReLU(out_channels, out_channels)
#
#     def forward(self, x):
#         b, c, w, h = x.shape
#         x = self._fuse(self._pyConv(self._reduce(self.global_pool(x))))
#         out = F.interpolate(x, (w, h), align_corners=True, mode='bilinear')
#         return out
#
#
# class MergePyConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MergePyConv, self).__init__()
#         self.conv3 = ConvBNReLU(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=3, stride=1)
#         self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, groups=1)
#
#     def forward(self, x):
#         b, c, w, h = x.shape
#         x = self.conv3(x)
#         x = F.interpolate(x, (w, h), align_corners=True, mode='bilinear')
#         out = self.conv1(x)
#         return out
#
#
# class PyConvParsingHead(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(PyConvParsingHead, self).__init__()
#
#         self.globalPyConv = GlobalPyConv(in_channels, out_channels)
#         self.localPyConv = LocalPyConv(in_channels, out_channels)
#         self.mergePyConv = MergePyConv(in_channels, out_channels)
#
#     def forward(self, x):
#         short = x
#         g_x = self.globalPyConv(x)
#         l_x = self.localPyConv(x)
#         x = torch.cat([g_x, l_x], dim=1)
#         out = self.mergePyConv(x) + short
#         return out
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=8):
#         super(BasicBlock, self).__init__()
#         if in_channels // groups == 0:
#             groups = 1
#         self.left = nn.Sequential(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
#                       groups=groups),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
#                       groups=groups),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )
#
#     def forward(self, x):
#         out = self.left(x)
#         return out
#
#
# class BasicRFB(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
#         super(BasicRFB, self).__init__()
#         self.scale = scale
#         self.out_channels = out_planes
#         inter_planes = in_planes // map_reduce
#
#         self.branch0 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
#             BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1,
#                       dilation=vision + 1, relu=False, groups=groups)
#         )
#         self.branch1 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
#             BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2,
#                       dilation=vision + 2, relu=False, groups=groups)
#         )
#         self.branch2 = nn.Sequential(
#             BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
#             BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
#             BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1,
#                       groups=groups),
#             BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4,
#                       dilation=vision + 4, relu=False, groups=groups)
#         )
#
#         self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
#         self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
#         self.relu = nn.ReLU(inplace=False)
#
#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#
#         out = torch.cat((x0, x1, x2), 1)
#
#         out = self.ConvLinear(out)
#
#         short = self.shortcut(x)
#
#         out = out * self.scale + short
#         out = self.relu(out)
#
#         return out
#
#
# class conv_1X1(nn.Module):
#     '''
#     使用conv_1X1 改变维度通道数
#     '''
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: int = 1,
#                  relu: bool = False,
#                  strides: int = 1,
#                  ):
#         super(conv_1X1, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
#                                stride=strides, padding=0, dilation=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         return out
#
#
# class conv_3X3(nn.Module):
#     '''
#     使用conv_3X3 改变维度通道数
#     '''
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: int = 3,
#                  stride: int = 1,
#                  padding: int = 1,
#                  ):
#         super(conv_3X3, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
#                                stride=stride, padding=padding, dilation=padding)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         return out
#
#
# class conv_7X7(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  ):
#         super(conv_7X7, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         out = F.relu(self.conv(x))
#         return out
#
#
# # 定义残差块ResBlock
# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation=1, stride=1, groups=1):
#         super(ResBlock, self).__init__()
#         # 这里定义了残差块内连续的2个卷积层
#         self.left = nn.Sequential(
#             ACBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
#                     groups=groups,
#                     ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             ACBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
#                     groups=groups,
#                     ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         out = self.left(x)
#         # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
#         out = out + self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# # 定义残差块ResBlock
# class DResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation=1, stride=1, groups=1):
#         super(DResBlock, self).__init__()
#         # 这里定义了残差块内连续的2个卷积层
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
#                       groups=groups,
#                       bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3,
#                       groups=groups,
#                       bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5,
#                       groups=groups,
#                       bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#
#     def forward(self, x):
#         layer1 = self.conv1(x)
#         layer2 = self.conv2(layer1)
#         layer3 = self.conv3(layer2)
#         out = layer1 + layer2 + layer3 + self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class Res2NetBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, scales=4):
#         super(Res2NetBlock, self).__init__()
#
#         if out_channels % scales != 0:  # 输出通道数为4的倍数
#             raise ValueError('Planes must be divisible by scales')
#
#         self.scales = scales
#         out_channel = out_channels // scales
#         # 1*1的卷积层
#         self.inconv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0),
#             nn.BatchNorm2d(out_channels)
#         )
#         # 3*3的卷积层，一共有3个卷积层和3个BN层
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1),
#             nn.BatchNorm2d(out_channel)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1),
#             nn.BatchNorm2d(out_channel)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, 3, 1, 1),
#             nn.BatchNorm2d(out_channel)
#         )
#         # 1*1的卷积层
#         self.outconv = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, 1, 1, 0),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         input = x
#         x = self.inconv(x)
#
#         # scales个部分
#         xs = torch.chunk(x, self.scales, 1)
#         ys = []
#         ys.append(xs[0])
#         ys.append(F.relu(self.conv1(xs[1])))
#         ys.append(F.relu(self.conv2(xs[2]) + ys[1]))
#         ys.append(F.relu(self.conv2(xs[3]) + ys[2]))
#         y = torch.cat(ys, 1)
#
#         y = self.outconv(y)
#
#         output = F.relu(y + x)
#
#         return output
#
#
#
# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super().__init__()
#         self.maxpool = nn.AdaptiveMaxPool2d(1)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.se = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // reduction, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_result = self.maxpool(x)
#         avg_result = self.avgpool(x)
#         max_out = self.se(max_result)
#         avg_out = self.se(avg_result)
#         output = self.sigmoid(max_out + avg_out)
#         return output
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         max_result, _ = torch.max(x, dim=1, keepdim=True)
#         avg_result = torch.mean(x, dim=1, keepdim=True)
#         result = torch.cat([max_result, avg_result], 1)
#         output = self.conv(result)
#         output = self.sigmoid(output)
#         return output
#
#
# class CBAMBlock(nn.Module):
#
#     def __init__(self, in_channels, out_channels, reduction=16, kernel_size=7):
#         super().__init__()
#         self.ca = ChannelAttention(channel=out_channels, reduction=reduction)
#         self.sa = SpatialAttention(kernel_size=kernel_size)
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         residual = x
#         out = x * self.ca(x)
#         out = out * self.sa(out)
#         return out + residual
#
#
# class CropLayer(nn.Module):
#
#     #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
#     def __init__(self, crop_set):
#         super(CropLayer, self).__init__()
#         self.rows_to_crop = - crop_set[0]
#         self.cols_to_crop = - crop_set[1]
#         assert self.rows_to_crop >= 0
#         assert self.cols_to_crop >= 0
#
#     def forward(self, input):
#         return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]
#
#
# # 论文提出的3x3+1x3+3x1
# class ACBlock(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
#                  padding_mode='zeros', deploy=False):
#         super(ACBlock, self).__init__()
#         self.deploy = deploy
#         if deploy:
#             self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                         kernel_size=(kernel_size, kernel_size), stride=stride,
#                                         padding=padding, dilation=dilation, groups=groups, bias=True,
#                                         padding_mode=padding_mode)
#         else:
#             self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                          kernel_size=(kernel_size, kernel_size), stride=stride,
#                                          padding=padding, dilation=dilation, groups=groups, bias=False,
#                                          padding_mode=padding_mode)
#             self.square_bn = nn.BatchNorm2d(num_features=out_channels)
#
#             center_offset_from_origin_border = padding - kernel_size // 2
#             ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
#             hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
#             if center_offset_from_origin_border >= 0:
#                 self.ver_conv_crop_layer = nn.Identity()
#                 ver_conv_padding = ver_pad_or_crop
#                 self.hor_conv_crop_layer = nn.Identity()
#                 hor_conv_padding = hor_pad_or_crop
#             else:
#                 self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
#                 ver_conv_padding = (0, 0)
#                 self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
#                 hor_conv_padding = (0, 0)
#             self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
#                                       stride=stride,
#                                       padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
#                                       padding_mode=padding_mode)
#
#             self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
#                                       stride=stride,
#                                       padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
#                                       padding_mode=padding_mode)
#             self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
#             self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
#
#     # forward函数
#     def forward(self, input):
#         if self.deploy:
#             return self.fused_conv(input)
#         else:
#             square_outputs = self.square_conv(input)
#             square_outputs = self.square_bn(square_outputs)
#             # print(square_outputs.size())
#             # return square_outputs
#             vertical_outputs = self.ver_conv_crop_layer(input)
#             vertical_outputs = self.ver_conv(vertical_outputs)
#             vertical_outputs = self.ver_bn(vertical_outputs)
#             # print(vertical_outputs.size())
#             horizontal_outputs = self.hor_conv_crop_layer(input)
#             horizontal_outputs = self.hor_conv(horizontal_outputs)
#             horizontal_outputs = self.hor_bn(horizontal_outputs)
#             # print(horizontal_outputs.size())
#             return square_outputs + vertical_outputs + horizontal_outputs
#
#
# class UNetEncoder(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, dilation=1, ):
#         super(UNetEncoder, self).__init__()
#         # 这里定义了残差块内连续的2个卷积层
#         self.layer_1 = Res2NetBlock(in_channels, out_channels)
#
#         # self.layer_3 = PyConv4(out_channels, out_channels)
#         # self.attention = sa_layer(out_channels)
#
#     def forward(self, x):
#         out = self.layer_1(x)
#
#         # out = self.layer_3(out)
#         # out = self.attention(out)
#         out = F.relu(out)
#         return out
#
#
# class UNetDecoder(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=2,
#                  strides=2, ):
#         super(UNetDecoder, self).__init__()
#         self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
#                                         kernel_size=kernel_size,
#                                         stride=strides, bias=True, )
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         return out
#
#
# class HSBlock(nn.Module):
#     '''
#     替代3x3卷积
#     '''
#
#     def __init__(self, in_channels, s=8):
#         '''
#         特征大小不改变
#         :param in_ch: 输入通道
#         :param s: 分组数
#         '''
#         super(HSBlock, self).__init__()
#         self.s = s
#         self.module_list = nn.ModuleList()
#
#         in_ch_range = torch.Tensor(in_channels)
#         in_ch_list = list(in_ch_range.chunk(chunks=self.s, dim=0))
#
#         self.module_list.append(nn.Sequential())
#         channel_nums = []
#         for i in range(1, len(in_ch_list)):
#             if i == 1:
#                 channels = len(in_ch_list[i])
#             else:
#                 random_tensor = torch.Tensor(channel_nums[i - 2])
#                 _, pre_ch = random_tensor.chunk(chunks=2, dim=0)
#                 channels = len(pre_ch) + len(in_ch_list[i])
#             channel_nums.append(channels)
#             self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels))
#         self.initialize_weights()
#
#     def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
#         conv_bn_relu = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#         return conv_bn_relu
#
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, x):
#         x = list(x.chunk(chunks=self.s, dim=1))
#         for i in range(1, len(self.module_list)):
#             y = self.module_list[i](x[i])
#             if i == len(self.module_list) - 1:
#                 x[0] = torch.cat((x[0], y), 1)
#             else:
#                 y1, y2 = y.chunk(chunks=2, dim=1)
#                 x[0] = torch.cat((x[0], y1), 1)
#                 x[i + 1] = torch.cat((x[i + 1], y2), 1)
#         return x[0]
#
#
# class my_unet(pl.LightningModule):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  extra_gap_weight: float,
#                  learning_rate: float = 1.0e-3,
#                  loss_func: Callable = nn.CrossEntropyLoss(),
#                  total_iterations: int = 1000,
#                  ):
#         super(my_unet, self).__init__()
#
#         self.layer1_conv = conv_3X3(in_channels, 64)
#
#         self.layer2_conv = UNetEncoder(in_channels, 64)
#
#         self.layer3_conv = UNetEncoder(64, 128)
#
#         self.layer4_conv = UNetEncoder(128, 256)
#         self.layer5_conv = UNetEncoder(256, 512)
#
#         self.layer5_1conv = ASPP(512, 512)
#
#         self.layer6_conv = UNetEncoder(512, 256)
#         self.layer7_conv = UNetEncoder(256, 128)
#         self.layer8_conv = UNetEncoder(128, 64, )
#
#         self.layer9_conv = conv_1X1(64, out_channels)
#
#         self.deconv1 = UNetDecoder(512, 256, )
#         self.deconv2 = UNetDecoder(256, 128, )
#         self.deconv3 = UNetDecoder(128, 64, )
#
#         self.attention_1 = CoordAttention(512, 512)
#         self.attention_2 = CoordAttention(256, 256)
#         self.attention_3 = CoordAttention(128, 128)
#         self.attention_4 = CoordAttention(64, 64)
#
#         self.extra_gap_weight = extra_gap_weight
#         self.learning_rate = learning_rate
#         self.loss_func = loss_func
#         self.total_iterations = total_iterations
#
#     def forward(self, x) -> torch.Tensor:
#
#         Encoder_layer1 = self.attention_4(self.layer1_conv(x))
#
#         Encoder_layer2 = self.layer2_conv(x)
#         Encoder_layer3 = self.layer3_conv(F.max_pool2d(Encoder_layer2, 2))
#         Encoder_layer4 = self.layer4_conv(F.max_pool2d(Encoder_layer3, 2))
#         Encoder_layer5 = self.layer5_conv(F.max_pool2d(Encoder_layer4, 2))
#         Encoder_layer5 = self.layer5_1conv(Encoder_layer5)
#         Decoder_layer4 = self.deconv1(self.attention_1(Encoder_layer5))
#
#         Decoder_layer4 = torch.cat([Encoder_layer4, Decoder_layer4], dim=1)
#         Decoder_layer4 = self.layer6_conv(Decoder_layer4)
#         #
#         Decoder_layer3 = self.deconv2(self.attention_2(Decoder_layer4))
#         Decoder_layer3 = torch.cat([Decoder_layer3, Encoder_layer3], dim=1)
#         Decoder_layer3 = self.layer7_conv(Decoder_layer3)
#         #
#         Decoder_layer2 = self.deconv3(self.attention_3(Decoder_layer3))
#         Decoder_layer2 = torch.cat(
#             [myut.get_adjusted_feature_map_small_big(Decoder_layer2, Encoder_layer2), Encoder_layer2], dim=1)
#         Decoder_layer2 = self.layer8_conv(Decoder_layer2)
#
#         Decoder_layer1 = self.layer9_conv(self.attention_4(Decoder_layer2 + Encoder_layer1))
#
#         return Decoder_layer1
#
#     def training_step(self, batch, batch_idx):
#         loss = None
#         if self.extra_gap_weight is None:
#             loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=False)
#         else:
#             loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                            use_sliding_window=False)
#         self.log('train_loss', loss, on_step=True, on_epoch=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss = None
#         if self.extra_gap_weight is None:
#             loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=True)
#         else:
#             loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                            use_sliding_window=True)
#         self.log('val_loss', loss, on_epoch=True)
#         return loss
#
#     def optimizer_step(
#             self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
#             on_tpu=False, using_native_amp=False, using_lbfgs=False,
#     ):
#         initial_learning_rate = self.learning_rate
#         current_iteration = self.trainer.global_step
#         total_iteration = self.total_iterations
#         for pg in optimizer.param_groups:
#             pg['lr'] = myut.poly_learning_rate(initial_learning_rate, current_iteration, total_iteration)
#         optimizer.step(closure=optimizer_closure)
#
#     def configure_optimizers(self):
#         return myut.configure_optimizers(self, self.learning_rate)


# net = my_unet(in_channels=1)
#
# print(net)


# class conv_1X1(nn.Module):
#     '''
#     使用conv_1X1 改变维度通道数
#     '''
#
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  kernel_size: int = 1,
#                  strides: int = 1,
#                  ):
#         super(conv_1X1, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
#                                stride=strides, padding=0, dilation=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         return out
#
#
# class UNetEncoder(nn.Module):
#     '''
#     带残差的unet
#     '''
#
#     def __init__(self, in_channels: int,
#                  out_channels: int,
#                  use_1x1conv: True,
#                  is_last_layer: False,
#                  padding: Sequence[int],
#                  kernel_size: int = 3,
#                  strides: int = 1,
#                  ):
#         super(UNetEncoder, self).__init__()
#         ''' Res_block'''
#         self.conv1 = nn.Conv2d(in_channels, out_channels,
#                                kernel_size=kernel_size,
#                                stride=strides, padding=padding[0], bias=True, dilation=padding[0])
#         self.conv2 = nn.Conv2d(out_channels, out_channels,
#                                kernel_size=kernel_size,
#                                stride=strides, padding=padding[1], bias=True, dilation=padding[1])
#         if use_1x1conv is True:
#             self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
#                                    stride=strides, padding=0, dilation=1)
#         else:
#             self.conv3 = None
#
#         if is_last_layer is True:
#             self.conv4 = nn.Conv2d(out_channels, out_channels,
#                                    kernel_size=kernel_size,
#                                    stride=strides, padding=padding[2], bias=True, dilation=padding[2])
#         else:
#             self.conv4 = None
#
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#
#         identity = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         if self.conv3:
#             identity = self.conv3(x)
#         if self.conv4:
#             out = F.relu(out)
#             out = self.bn3(self.conv4(out))
#         out += identity
#         out = F.relu(out)
#         return out
#
#
# class UNetDecoder(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=2,
#                  strides=2, ):
#         super(UNetDecoder, self).__init__()
#         self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
#                                         kernel_size=kernel_size,
#                                         stride=strides, bias=True, )
#         self.bn1 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         return out
#
#
# class my_unet(pl.LightningModule):
#     def __init__(self,
#                  in_channels: int,
#                  extra_gap_weight: bool = None,
#                  learning_rate: float = 1.0e-3,
#                  loss_func: Callable = nn.CrossEntropyLoss(),
#                  total_iterations: int = 1000,
#                  ):
#         super(my_unet, self).__init__()
#         self.layer1_conv = UNetEncoder(in_channels, 32, padding=(1, 1), use_1x1conv=True, is_last_layer=False)
#
#         self.layer2_conv = UNetEncoder(32, 64, padding=(1, 1), use_1x1conv=True, is_last_layer=False)
#         self.layer3_conv = UNetEncoder(64, 128, padding=(1, 1), use_1x1conv=True, is_last_layer=False)
#
#         self.layer4_conv = UNetEncoder(128, 256, padding=(1, 2, 4), use_1x1conv=True, is_last_layer=True)
#
#         self.layer4_1_conv = UNetEncoder(256, 256, padding=(1, 2, 4), use_1x1conv=True, is_last_layer=True)
#         self.layer4_2_conv = UNetEncoder(256, 256, padding=(1, 2, 4), use_1x1conv=True, is_last_layer=True)
#         self.layer4_3_conv = UNetEncoder(256, 256, padding=(1, 2, 4), use_1x1conv=True, is_last_layer=True)
#
#         self.layer5_conv = UNetEncoder(256, 128, padding=(1, 1), use_1x1conv=True, is_last_layer=False)
#         self.layer6_conv = UNetEncoder(128, 64, padding=(1, 1), use_1x1conv=True, is_last_layer=False)
#         self.layer7_conv = UNetEncoder(64, 32, padding=(1, 1), use_1x1conv=True, is_last_layer=False)
#
#         self.layer8_conv = nn.Conv2d(32, 1, kernel_size=3,
#                                      stride=1, padding=1, bias=True, dilation=1)
#
#         self.deconv1 = UNetDecoder(256, 128, )
#         self.deconv2 = UNetDecoder(128, 64, )
#         self.deconv3 = UNetDecoder(64, 32, )
#
#         '''残差 多尺度 连接 '''
#         self.connection_1 = UNetEncoder(32, 32, padding=(1, 2, 4), use_1x1conv=True, is_last_layer=True)
#         self.connection_2 = UNetEncoder(64, 64, padding=(1, 2, 4), use_1x1conv=True, is_last_layer=True)
#         self.connection_3 = UNetEncoder(128, 128, padding=(1, 2, 4), use_1x1conv=True, is_last_layer=True)
#
#         self.classify_conv = ChannelPad(2, 32, 2,
#                                         'project')  # number of spatial dimensions of the input image is 2
#
#         self.conv_1X1_1 = conv_1X1(in_channels, 32)
#         self.conv_1X1_2 = conv_1X1(32, 64)
#         self.conv_1X1_3 = conv_1X1(64, 128)
#
#         self.extra_gap_weight = extra_gap_weight
#         self.learning_rate = learning_rate
#         self.loss_func = loss_func
#         self.total_iterations = total_iterations
#
#     def forward(self, x) -> torch.Tensor:
#         conv1 = self.layer1_conv(x)
#
#         pool1 = F.max_pool2d(conv1, 2)
#         feature_map_1 = self.conv_1X1_1(x)
#         feature_map_1_1 = myut.get_adjusted_feature_map_b2s(feature_map_1, pool1)
#         pool1 = feature_map_1_1 + pool1
#
#         conv2 = self.layer2_conv(pool1)
#
#         pool2 = F.max_pool2d(conv2, 2)
#         feature_map_2 = self.conv_1X1_2(feature_map_1_1)
#         feature_map_2_2 = myut.get_adjusted_feature_map_b2s(feature_map_2, pool2)
#         pool2 = feature_map_2_2 + pool2
#
#         conv3 = self.layer3_conv(pool2)
#
#         pool3 = F.max_pool2d(conv3, 2)
#         feature_map_3 = self.conv_1X1_3(feature_map_2_2)
#         feature_map_3_3 = myut.get_adjusted_feature_map_b2s(feature_map_3, pool3)
#         pool3 = feature_map_3_3 + pool3
#
#         '''最后一层'''
#         conv4 = self.layer4_conv(pool3)
#         conv4_1 = self.layer4_1_conv(conv4)
#         conv4_2 = self.layer4_2_conv(conv4_1)
#         conv4_3 = self.layer4_3_conv(conv4_2)
#
#         convt1 = self.deconv1(conv4_3)
#         conv3 = self.connection_3(conv3)
#         convt1 = myut.get_adjusted_feature_map_s2b(convt1, conv3)
#         concat1 = torch.cat([convt1, conv3], dim=1)
#         concat1 = myut.get_adjusted_feature_map_channels(concat1, conv3)
#         feature_map_4 = feature_map_3
#         concat1 = torch.cat([concat1, feature_map_4], dim=1)
#         conv5 = self.layer5_conv(concat1)
#
#         convt2 = self.deconv2(conv5)
#         conv2 = self.connection_2(conv2)
#         convt2 = myut.get_adjusted_feature_map_s2b(convt2, conv2)
#         concat2 = torch.cat([convt2, conv2], dim=1)
#         concat2 = myut.get_adjusted_feature_map_channels(concat2, conv2)
#         feature_map_5 = feature_map_2
#         concat2 = torch.cat([concat2, feature_map_5], dim=1)
#         conv6 = self.layer6_conv(concat2)
#
#         convt3 = self.deconv3(conv6)
#         conv1 = self.connection_1(conv1)
#         convt3 = myut.get_adjusted_feature_map_s2b(convt3, conv1)
#         concat3 = torch.cat([convt3, conv1], dim=1)
#         concat3 = myut.get_adjusted_feature_map_channels(concat3, conv1)
#         feature_map_6 = feature_map_1
#         concat3 = torch.cat([concat3, feature_map_6], dim=1)
#         conv7 = self.layer7_conv(concat3)
#
#         outp = self.classify_conv(conv7)
#
#         return outp
#
#     def training_step(self, batch, batch_idx):
#         loss = None
#         if self.extra_gap_weight is None:
#             loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=False)
#         else:
#             loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                            use_sliding_window=False)
#         self.log('train_loss', loss, on_step=True, on_epoch=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss = None
#         if self.extra_gap_weight is None:
#             loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=True)
#         else:
#             loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
#                                            use_sliding_window=True)
#         self.log('val_loss', loss, on_epoch=True)
#         return loss
#
#     def optimizer_step(
#             self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
#             on_tpu=False, using_native_amp=False, using_lbfgs=False,
#     ):
#         initial_learning_rate = self.learning_rate
#         current_iteration = self.trainer.global_step
#         total_iteration = self.total_iterations
#         for pg in optimizer.param_groups:
#             pg['lr'] = myut.poly_learning_rate(initial_learning_rate, current_iteration, total_iteration)
#         optimizer.step(closure=optimizer_closure)
#
#     def configure_optimizers(self):
#         return myut.configure_optimizers(self, self.learning_rate)


'''
class UNetEncoder(nn.Module):
   

    def __init__(self, in_channels,
                 out_channels,
                 use_1x1conv: False,
                 dilation,
                 padding,
                 kernel_size: int = 3,
                 strides: int = 1,
                 ):
        super(UNetEncoder, self).__init__()
      
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True, dilation=dilation)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True, dilation=dilation)
        if use_1x1conv is True:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=strides, padding=0, dilation=dilation)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            identity = self.conv3(x)
        out += identity
        out = F.relu(out)
        return out


class UNetDecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 kernel_size=2,
                 strides=2, ):
        super(UNetDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class my_unet(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 extra_gap_weight: bool = None,
                 learning_rate: float = 1.0e-3,
                 loss_func: Callable = nn.CrossEntropyLoss(),
                 total_iterations: int = 1000,
                 ):
        super(my_unet, self).__init__()
        self.layer1_conv = UNetEncoder(in_channels, 32, use_1x1conv=True, dilation=1, padding=1)
        self.layer2_conv = UNetEncoder(32, 64, use_1x1conv=True, dilation=1, padding=1)
        self.layer3_conv = UNetEncoder(64, 128, use_1x1conv=True, dilation=1, padding=1)
        self.layer4_conv = UNetEncoder(128, 256, use_1x1conv=True, dilation=1, padding=1)
        self.layer5_conv = UNetEncoder(256, 512, use_1x1conv=True, dilation=1, padding=1)

        self.layer6_conv = UNetEncoder(512, 256, use_1x1conv=True, dilation=1, padding=1)
        self.layer7_conv = UNetEncoder(256, 128, use_1x1conv=True, dilation=1, padding=1)
        self.layer8_conv = UNetEncoder(128, 64, use_1x1conv=True, dilation=1, padding=1)
        self.layer9_conv = UNetEncoder(64, 32, use_1x1conv=True, dilation=1, padding=1)

        self.layer10_conv = nn.Conv2d(32, 1, kernel_size=3,
                                      stride=1, padding=1, bias=True, dilation=1)


        self.deconv1 = UNetDecoder(512, 256, dilation=1)
        self.deconv2 = UNetDecoder(256, 128, dilation=1)
        self.deconv3 = UNetDecoder(128, 64, dilation=1)
        self.deconv4 = UNetDecoder(64, 32, dilation=1)

        self.classify_conv = ChannelPad(2, 32, 2,
                                        'project')  # number of spatial dimensions of the input image is 2

        self.extra_gap_weight = extra_gap_weight
        self.learning_rate = learning_rate
        self.loss_func = loss_func
        self.total_iterations = total_iterations

    def forward(self, x) -> torch.Tensor:
        conv1 = self.layer1_conv(x)

        pool1 = F.max_pool2d(conv1, 2)

        conv2 = self.layer2_conv(pool1)

        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.layer3_conv(pool2)

        pool3 = F.max_pool2d(conv3, 2)

        conv4 = self.layer4_conv(pool3)

        pool4 = F.max_pool2d(conv4, 2)

        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        convt1 = myut.get_adjusted_feature_map_s2b(convt1, conv4)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        convt2 = myut.get_adjusted_feature_map_s2b(convt2, conv3)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        convt3 = myut.get_adjusted_feature_map_s2b(convt3, conv2)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        convt4 = myut.get_adjusted_feature_map_s2b(convt4, conv1)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.classify_conv(conv9)

        return outp

    def training_step(self, batch, batch_idx):
        loss = None
        if self.extra_gap_weight is None:
            loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=False)
        else:
            loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
                                           use_sliding_window=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = None
        if self.extra_gap_weight is None:
            loss = myut.cal_batch_loss(self, batch, self.loss_func, use_sliding_window=True)
        else:
            loss = myut.cal_batch_loss_gap(self, batch, self.loss_func, extra_gap_weight=self.extra_gap_weight,
                                           use_sliding_window=True)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
            on_tpu=False, using_native_amp=False, using_lbfgs=False,
    ):
        initial_learning_rate = self.learning_rate
        current_iteration = self.trainer.global_step
        total_iteration = self.total_iterations
        for pg in optimizer.param_groups:
            pg['lr'] = myut.poly_learning_rate(initial_learning_rate, current_iteration, total_iteration)
        optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        return myut.configure_optimizers(self, self.learning_rate)


        # if self.conv4:
        #     identity = self.conv4(x)
        # out1 = torch.cat([F.relu(self.bn1((self.conv1(x)))), identity], dim=1)
        # out1 = myut.get_adjusted_feature_map_channels(out1, identity)
        #
        # out2 = torch.cat([F.relu(self.bn2((self.conv2(out1)))), out1], dim=1)
        # out2 = myut.get_adjusted_feature_map_channels(out2, identity)
        # out2 = torch.cat([out2, identity], dim=1)
        # out2 = myut.get_adjusted_feature_map_channels(out2, identity)
        #
        # out3 = torch.cat([F.relu(self.bn3((self.conv3(out2)))), out2], dim=1)
        # out3 = myut.get_adjusted_feature_map_channels(out3, identity)
        # out3 = torch.cat([out3, identity], dim=1)
        # out3 = myut.get_adjusted_feature_map_channels(out3, identity)
        #
        # out3 += identity
        # out = F.relu(out3)
'''
