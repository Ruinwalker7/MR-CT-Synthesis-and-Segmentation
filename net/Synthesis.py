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
    
    def forward(self, inputs, ctff_outputs=0):
        output1 = self.conv1(inputs)
        output1 = self.norm(output1)
        output1 = self.act_fn(output1)
        output1 += ctff_outputs
        outputs = self.conv2(output1)
        outputs = self.norm(outputs)
        outputs = self.act_fn(outputs)
        return outputs, output1


# 编码器模块，由多个 DownConv 模块组成
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.down_conv1 = DownConv(1, 32, 1)
        self.down_conv2 = DownConv(32, 64, 2)
        self.down_conv3 = DownConv(64, 128, 2)
        self.down_conv4 = DownConv(128, 256, 2)
        self.down_conv5 = DownConv(258, 512, 2)
    
    def forward(self, x: Tensor):
        feat1, CTFF_input1 = self.down_conv1(x)
        feat2, _ = self.down_conv2(feat1)
        feat3, _ = self.down_conv3(feat2)
        feat4, _ = self.down_conv4(feat3)
        feat5, _ = self.down_conv5(feat4)
        
        return [feat1, feat2, feat3, feat4, feat5, CTFF_input1]


# 上采样模块
class UpConv(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpConv, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.norm = torch.nn.InstanceNorm2d()
    
    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.norm(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.norm(outputs)
        outputs = self.relu(outputs)
        return outputs
    
    def get_first_conv_output(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_conv1 = UpConv(512, 256)
        self.up_conv2 = UpConv(256, 128)
        self.up_conv3 = UpConv(128, 64)
        self.up_conv3 = UpConv(64, 32)
    
    def forward(self, feat, x: Tensor) -> Tensor:
        x = self.up_conv1(feat[3], x)
        x = self.up_conv1(feat[2], x)
        x = self.up_conv1(feat[1], x)
        x = self.up_conv1(feat[0], x)
        return x


class Synthesis(nn.Module):
    def __init__(self, outputs=1):
        super().__init__()
        self.encoder = Encoder()
        self.decode = Decoder()
        self.conv = nn.Conv2d(32, outputs, kernel_size=3, padding=1)
    
    def forward(self, x: Tensor):
        feats = self.encoder(x)
        x = self.decoder(feats, feats[4])
        x = self.conv(x)
        return x, feats[5]
