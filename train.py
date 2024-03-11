import torch
import torch.nn as nn
from net.vgg import vgg19
from net.Model import Model
from loss.VGGPerceptualLoss import SynthesisLoss
from loss.SegmentationLoss import SegmentationLoss
from torch.utils.data import random_split
from net.Discriminator import MultitaskDiscriminator
import os
from typing import Optional

batch_size = 16
lr = 3e-4
adam_betas = (0.5, 0.999)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

lambda_1 = 0.5  # 控制合成损失和分割损失权重
lambda_2 = 0.5  # 控制初始输出和最终输出的权重
lambda_3 = 0.5  # 控制L1距离损失的权重

vgg = vgg19(True)
discriminator = MultitaskDiscriminator()
synthesisLoss = SynthesisLoss(lambda_3)
segmentationLoss = SegmentationLoss()
d_loss_function = nn.BCELoss()
model = Model()
discriminator.to(device)
vgg.to(device)
synthesisLoss.to(device)
d_loss_function.to(device)

g_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=adam_betas)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=adam_betas)

dataset = ImageDataset("./img_align_celeba")
generator = torch.Generator().manual_seed(123)
train_dataset, test_dataset = random_split(dataset, [0.9, 0.1], generator=generator)

last_epoch = 0
train_loss_hist = []

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_name: Optional[str] = "epoch_10.pt"

if isinstance(checkpoint_name, str):
    path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = torch.load(path)
    last_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    g_optimizer.load_state_dict(checkpoint["g_optim_state_dict"])
    d_optimizer.load_state_dict(checkpoint["d_optim_state_dict"])
    train_loss_hist = checkpoint["train_loss_hist"]

num_epochs = 200
save_interval = 15

for e in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for x_mr, x_ct, x_mask in train_dataset:
        bsz = x_mr.shape[0]
        x_mr = x_mr.to(device)
        x_ct = x_ct.to(device)
        x_mask = x_mask.to(device)
        
        real = torch.cat((x_mr, x_ct, x_mask), dim=1)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        # 训练鉴别器
        d_optimizer.zero_grad()
        outputs = discriminator(real)
        d_loss_real = d_loss_function(outputs, real_labels)
        d_loss_real.backward()
        
        _, _, x2_ct, y2 = model(x_mr)
        fake = torch.cat((x_mr, x2_ct, y2), dim=1)
        outputs = discriminator(fake)
        d_loss_fake = d_loss_function(outputs, fake_labels)
        d_loss_fake.backward()
        d_optimizer.step()
        
        # 训练主模型
        g_optimizer.zero_grad()
        x1_ct, y1, x2_ct, y2 = model(x_mr)
        fake = torch.cat((x_mr, x2_ct, y2), dim=1)
        
        # Gan损失
        outputs = discriminator(fake)
        g_loss = d_loss_function(outputs, real_labels)
        
        # 合成损失
        synthesisLoss1 = synthesisLoss(x1_ct, x_ct)
        synthesisLoss2 = synthesisLoss(x2_ct, x_ct)
        total_syn_loss = lambda_2 * synthesisLoss1 + (1 - lambda_2) * synthesisLoss2
        
        # 分割损失
        segmentationloss1 = segmentationLoss(y1, x_mask)
        segmentationloss2 = segmentationLoss(y2, x_mask)
        total_seg_loss = lambda_2 * segmentationloss1 + (1 - lambda_2) * segmentationloss2
        
        total_loss = g_loss + lambda_1 * total_syn_loss + (1 - lambda_1) * total_seg_loss
        total_loss.backward()
        # 更新生成器权重
        g_optimizer.step()
        
        train_loss += total_loss.item() * bsz
    
    train_loss /= len(train_dataset.dataset)
    train_loss_hist.append(train_loss)
    
    epoch = e + last_epoch + 1
    print(f"Epoch {epoch} loss: {train_loss:.4f}")
    # save checkpoint
    if epoch % save_interval == 0 or e == num_epochs - 1:
        path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "g_optim_state_dict": g_optimizer.state_dict(),
                "d_optim_state_dict": d_optimizer.state_dict(),
                "train_loss_hist": train_loss_hist,
            },
            path,
        )
