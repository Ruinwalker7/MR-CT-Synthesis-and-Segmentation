from typing import Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor


# 下采样模块
class DownConv(nn.Module):
    def __init__(self, in_size, out_size, stride=2):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1)
        self.act_fn = nn.LeakyReLU(negative_slope=0.2)
        self.norm = nn.InstanceNorm2d(out_size)
    
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.norm(outputs)
        outputs = self.act_fn(outputs)
        outputs = self.conv2(outputs)
        outputs = self.norm(outputs)
        outputs = self.act_fn(outputs)
        return outputs


# 编码器模块，由多个 DownConv 模块组成
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.down_conv1 = DownConv(1, 32, stride=1)
        self.down_conv2 = DownConv(32, 64, stride=2)
        self.down_conv3 = DownConv(64, 128, stride=2)
        self.down_conv4 = DownConv(128, 256, stride=2)
        self.down_conv5 = DownConv(258, 512, stride=2)
    
    def forward(self, x: Tensor):
        feat1 = self.down_conv1(x)
        feat2 = self.down_conv2(feat1)
        feat3 = self.down_conv3(feat2)
        feat4 = self.down_conv4(feat3)
        feat5 = self.down_conv5(feat4)
        
        return [feat1, feat2, feat3, feat4, feat5]


# 上采样模块
class UpConv(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpConv, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.LeakyReLU(negative_slope=0.2,inplace=True)
        self.norm = torch.nn.InstanceNorm2d(num_features=out_size)
    
    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.norm(outputs)
        outputs = self.relu(outputs)
        output1 = self.conv2(outputs)
        output1 = self.norm(output1)
        output1 = self.relu(output1)
        return output1


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_conv1 = UpConv(512, 256)
        self.up_conv2 = UpConv(256, 128)
        self.up_conv3 = UpConv(128, 64)
        self.up_conv3 = UpConv(64, 32)
    
    def forward(self, feat, x: Tensor):
        x = self.up_conv1(feat[3], x)
        x = self.up_conv1(feat[2], x)
        x = self.up_conv1(feat[1], x)
        x = self.up_conv1(feat[0], x)
        return x


class Segmentation(nn.Module):
    def __init__(self, outputs=1):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.conv = nn.Conv2d(32, outputs, kernel_size=3, padding=1)
    
    def forward(self, x: Tensor):
        feat = self.encoder(x)
        x = self.decoder(feat, feat[4])
        outputs = self.conv(x)
        return x, outputs
