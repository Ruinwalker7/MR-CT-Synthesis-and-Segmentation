import torch
import torch.nn as nn
import net.Synthesis
import torch.nn.functional as fn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import os, glob
from dataclasses import dataclass, asdict
from typing import Any, List, Optional, Tuple, Union
import numpy
from PIL import Image


class ImageDataset(Dataset):
    def __init__(
            self,
            data_dir: Union[str, os.PathLike],
            resolution: int = 64,
            center_crop: bool = True,
            ext: str = "jpg",
    ):
        self.images = sorted(
            [f for f in glob.glob(os.path.join(data_dir, f"*.{ext}"))]
        )
        self.pre_proc = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.ToTensor(),
            ]
        )
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tensor:
        img = Image.open(self.images[idx]).convert("RGB")
        return self.pre_proc(img)


batch_size = 512
lr = 3e-4
adam_betas = (0.5, 0.999)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Model()
model_copy = Model().requires_grad_(False)
model.to(device)
model_copy.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=adam_betas)

dataset = ImageDataset("./img_align_celeba")
generator = torch.Generator().manual_seed(123)
train_dataset, test_dataset = random_split(dataset, [0.9, 0.1], generator=generator)

rec_weight = 20
idem_weight = 20
tight_weight = 2.5
idem_weight /= rec_weight
tight_weight /= rec_weight
loss_tight_clamp_ratio = 1.5

last_epoch = 0
train_loss_hist = []

checkpoint_dir = "ign-celeba"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_name: Optional[str] = "epoch_560.pt"
if isinstance(checkpoint_name, str):
    path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = torch.load(path)
    last_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])
    train_loss_hist = checkpoint["train_loss_hist"]

num_epochs = 200
save_interval = 15

for e in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for x in train_dl:
        bsz = x.shape[0]
        x = x.to(device)
        # normalize
        x = 2.0 * x - 1.0
        
        # get noise from input frequency statistics
        #   freq_means_and_stds = get_freq_means_and_stds(x)
        #   z = torch.stack([get_noise(*freq_means_and_stds) for _ in range(bsz)])
        # two lines shown above are the old sampling code I used for training
        freq_means_and_stds = torch.stack(get_freq_means_and_stds(x)).unsqueeze_(0)
        num_dims = len(freq_means_and_stds.shape) - 1
        freq_means_and_stds = freq_means_and_stds.repeat(
            bsz, *(1,) * num_dims
        ).unbind(dim=1)
        z = get_noise(*freq_means_and_stds)
        z = z.to(device, memory_format=torch.contiguous_format)
        
        # compute model outputs
        model_copy.load_state_dict(model.state_dict())
        fx = model(x)
        fz = model(z)
        f_z = fz.detach()
        ff_z = model(f_z)
        f_fz = model_copy(fz)
        
        # compute losses
        loss_rec = fn.l1_loss(fx, x, reduction="none").view(bsz, -1).mean(dim=-1)
        loss_idem = fn.l1_loss(f_fz, fz, reduction="mean")
        loss_tight = -fn.l1_loss(ff_z, f_z, reduction="none").view(bsz, -1).mean(dim=-1)
        loss_tight_clamp = loss_tight_clamp_ratio * loss_rec
        loss_tight = fn.tanh(loss_tight / loss_tight_clamp) * loss_tight_clamp
        loss_rec = loss_rec.mean()
        loss_tight = loss_tight.mean()
        
        loss = loss_rec + idem_weight * loss_idem + tight_weight * loss_tight
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * bsz
    
    train_loss /= len(train_dl.dataset)
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
                "optim_state_dict": optimizer.state_dict(),
                "train_loss_hist": train_loss_hist,
            },
            path,
        )





