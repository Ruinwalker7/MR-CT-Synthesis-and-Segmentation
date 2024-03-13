import torch
import torch.nn as nn
from torch import Tensor


# 下采样模块
class DownConv(nn.Module):
    def __init__(self, in_size, out_size, stride=2):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1)
        self.act_fn = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.norm = nn.InstanceNorm2d(out_size)

    def forward(self, inputs, ctff_outputs: Tensor = None):
        output1 = self.conv1(inputs)
        output1 = self.norm(output1)
        output1 = self.act_fn(output1)
        if ctff_outputs is None:
            pass
        else:
            output1 = output1 + ctff_outputs
        outputs = self.conv2(output1)
        outputs = self.norm(outputs)
        outputs = self.act_fn(outputs)
        # print(outputs.shape)
        return outputs, output1


# 编码器模块，由多个 DownConv 模块组成
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.down_conv1 = DownConv(1, 32, 1)
        self.down_conv2 = DownConv(32, 64, 2)
        self.down_conv3 = DownConv(64, 128, 2)
        self.down_conv4 = DownConv(128, 256, 2)
        self.down_conv5 = DownConv(256, 512, 2)

    def forward(self, x: Tensor, ctff_output: Tensor = None):
        feat1, CTFF_input1 = self.down_conv1(x, ctff_output)
        feat2, _ = self.down_conv2(feat1)
        feat3, _ = self.down_conv3(feat2)
        feat4, _ = self.down_conv4(feat3)
        feat5, _ = self.down_conv5(feat4)

        return [feat1, feat2, feat3, feat4, feat5, CTFF_input1]


# 上采样模块
class UpConv(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpConv, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.norm = torch.nn.InstanceNorm2d(out_size)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        # print(outputs.shape)
        outputs = self.conv1(outputs)
        outputs = self.norm(outputs)
        outputs = self.relu(outputs)
        # print(outputs.shape)
        outputs = self.conv2(outputs)
        outputs = self.norm(outputs)
        outputs = self.relu(outputs)
        # print(outputs.shape)
        return outputs

    def get_first_conv_output(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_conv1 = UpConv(512 + 256, 256)
        self.up_conv2 = UpConv(256 + 128, 128)
        self.up_conv3 = UpConv(128 + 64, 64)
        self.up_conv4 = UpConv(64 + 32, 32)

    def forward(self, feat, x: Tensor) -> Tensor:
        x = self.up_conv1(feat[3], x)
        x = self.up_conv2(feat[2], x)
        x = self.up_conv3(feat[1], x)
        x = self.up_conv4(feat[0], x)


        return x


class Synthesis(nn.Module):
    def __init__(self, outputs=1):
        super().__init__()
        self.encoder = Encoder()
        self.decode = Decoder()
        self.conv = nn.Conv2d(32, outputs, kernel_size=3, padding=1)

    def forward(self, x: Tensor, ctff_output: Tensor = 0):
        feats = self.encoder(x, ctff_output)
        x = self.decode(feats, feats[4])
        x = self.conv(x)
        return x, feats[5]
