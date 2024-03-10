import torch
import torch.nn as nn
from torch import Tensor

class CTFF(nn.Module):
    def __init__(self):
        super(CTFF, self).__init__()
        self.conv1 = nn.Conv2d(, , kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(, , kernel_size=3, padding=1)
        self.act_fn = nn.LeakyReLU(negative_slope=0.2)
        self.norm = torch.nn.InstanceNorm2d()

    def forward(self, input1, input2):
        
        outputs = self.conv1(inputs)
        outputs = self.norm(outputs)
        outputs = self.act_fn(outputs)
        outputs = self.conv2(outputs)
        outputs = self.norm(outputs)
        outputs = self.act_fn(outputs)
        return outputs