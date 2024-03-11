import torch
import torch.nn as nn
import torch.nn.functional as F


class MultitaskDiscriminator(nn.Module):
    def __init__(self):
        super(MultitaskDiscriminator, self).__init__()
        # 定义初始卷积层，输入通道取决于你的输入，这里假设为3个通道的MR图像，3个通道的CT图像和1个通道的分割掩码，共7个通道
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # 定义四个阶段的卷积块
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, kernel_size=3, stride=2),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 定义最后的卷积层，输出通道为1
        self.final_conv = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(1 * 31 * 31, 1)  # 假设前面层的输出已经是1x1的特征图
    def forward(self, x):
        # x 是输入的张量，应该包含MR图像、CT图像和分割掩码的合并
        x = self.initial_conv(x)
        x = self.conv_blocks(x)
        x = self.final_conv(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        # 使用Sigmoid激活函数输出概率
        return torch.sigmoid(x)


# 假设输入张量的大小
# input_tensor = torch.randn((1, 3, 512, 512))  # Batch size为1，7个通道，大小为256x256
#
# # 实例化鉴别器
# discriminator = MultitaskDiscriminator()
#
# # 前向传播
# output = discriminator(input_tensor)
#
# print(output.shape)  # 查看输出的大小