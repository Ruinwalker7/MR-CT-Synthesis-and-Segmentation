import torch
import torch.nn as nn
from torch import Tensor

class CTFF(nn.Module):
    def __init__(self):
        super(CTFF, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.act_fn = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.norm = torch.nn.InstanceNorm2d(32)

    def forward(self, input1, input2):
        # print("CTFF1:", input1.shape)
        # print("CTFF2:", input2.shape)

        merged_tensor = torch.cat((input1, input2), dim=1)
        outputs = self.conv1(merged_tensor)
        outputs = self.norm(outputs)
        outputs = self.act_fn(outputs)
        outputs = self.conv2(outputs)
        outputs = self.norm(outputs)
        outputs = self.act_fn(outputs)
        return outputs