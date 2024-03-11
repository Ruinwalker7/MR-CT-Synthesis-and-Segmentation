import torch
import torch.nn as nn
from torchvision.models import vgg19

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        self.slice = torch.nn.Sequential()
        for x in range(21):  # Assuming we are using features up to layer 20 (after ReLU5_2)
            self.slice.add_module(str(x), vgg_pretrained_features[x])

        # Freeze parameters. We don't need to train them.
        for param in self.slice.parameters():
            param.requires_grad = False

    def forward(self, a, b):
        # Normalize the images to fit VGG training
        a = self.normalize_batch(a)
        b = self.normalize_batch(b)

        # Obtain features
        a_features = self.slice(a)
        b_features = self.slice(b)

        # Compute L1 loss between feature representations
        perceptual_loss = torch.nn.functional.l1_loss(a_features, b_features)
        return perceptual_loss

    def normalize_batch(self, batch):
        # Normalize using ImageNet mean and std
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        batch = (batch - mean) / std
        return batch

class SynthesisLoss(nn.Module):
    def __init__(self, lambda_3):
        super(SynthesisLoss, self).__init__()
        self.lambda_3 = lambda_3
        self.l1_loss = torch.nn.L1Loss()
        self.vgg_loss = VGGPerceptualLoss()

    def forward(self, a, b):
        l1_loss = self.l1_loss(a, b)
        vgg_loss = self.vgg_loss(a, b)
        total_loss = self.lambda_3 * l1_loss + vgg_loss
        return total_loss

