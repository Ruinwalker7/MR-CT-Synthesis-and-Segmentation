import torch
import torch.nn as nn
from net.Synthesis import Synthesis
from net.CTFF import CTFF
from net.Segmentation import Segmentation


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.CTFF = CTFF()
        self.Synthesis = Synthesis()
        self.Segmentation = Segmentation()

    def forward(self, x):
        output1, feature1 = self.Synthesis(x)
        output2 = torch.cat((output1, x), dim=1)
        # print("cat:", output2.shape)
        output2, feature2 = self.Segmentation(output2)
        ctff_output = self.CTFF(feature1, feature2)
        output3, _ = self.Synthesis(x, ctff_output)
        output4 = torch.cat((output3, x), dim=1)
        output4, _ = self.Segmentation(output4)
        return output1, output2, output3, output4
