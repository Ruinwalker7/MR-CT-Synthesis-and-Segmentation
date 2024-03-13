import os
import torch
from torch.utils.data import Dataset, DataLoader
import elastix
import nrrd
import SimpleITK as sitk


class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 包含所有子文件夹的根目录的路径。
            transform (callable, optional): 可选的转换函数/函数序列。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        # 遍历根目录下的所有子文件夹
        for subdir in sorted(os.listdir(self.root_dir)):
            subdir_path = os.path.join(self.root_dir, subdir)
            if os.path.isdir(subdir_path):
                # 分别存储MR、CT和OAR图像的路径
                mr, ct, oars = None, None, []
                for filename in sorted(os.listdir(subdir_path)):
                    filepath = os.path.join(subdir_path, filename)
                    if 'MR' in filename.upper():
                        mr = filepath
                    elif 'CT' in filename.upper():
                        ct = filepath
                    elif 'OAR' in filename.upper():
                        oars.append(filepath)
                if mr and ct and oars:  # 确保三种类型的图像都存在
                    samples.append((mr, ct, oars))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mr_path, ct_path, oar_paths = self.samples[idx]
        mr_image = sitk.ReadImage(mr_path)
        ct_image = sitk.ReadImage(ct_path)
        oar_images = [sitk.ReadImage(oar_path) for oar_path in oar_paths]

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetInput(mr_image)

        # 执行N4校正
        mr_image = corrector.Execute()
        
        
        if self.transform:
            mr_image = self.transform(mr_image)
            ct_image = self.transform(ct_image)
            oar_images = [self.transform(oar_image) for oar_image in oar_images]

        # 将numpy数组转换为torch张量
        mr_tensor = torch.from_numpy(mr_image).float()
        ct_tensor = torch.from_numpy(ct_image).float()
        oar_tensors = torch.stack([torch.from_numpy(oar_image).float() for oar_image in oar_images])

        return mr_tensor, ct_tensor, oar_tensors
    # def __getitem__(self, idx):
    #     mr_path, ct_path, oar_paths = self.samples[idx]
    #     mr_image, _ = nrrd.read(mr_path)
    #     ct_image, _ = nrrd.read(ct_path)
    #     oar_images = [nrrd.read(oar_path)[0] for oar_path in oar_paths]
    #
    #     if self.transform:
    #         mr_image = self.transform(mr_image)
    #         ct_image = self.transform(ct_image)
    #         oar_images = [self.transform(oar_image) for oar_image in oar_images]
    #
    #     # 将numpy数组转换为torch张量
    #     mr_tensor = torch.from_numpy(mr_image).float()
    #     ct_tensor = torch.from_numpy(ct_image).float()
    #     oar_tensors = torch.stack([torch.from_numpy(oar_image).float() for oar_image in oar_images])
    #
    #     return mr_tensor, ct_tensor, oar_tensors

# 示例用法
if __name__ == "__main__":
    root_dir = 'C:\\Users\\23472\\Desktop\\MR-CT_Synthesis_and_Segmentation\\data\\HaN-Seg\\set_1'  # 替换为你的根目录路径
    dataset = MedicalImageDataset(root_dir=root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for mr, ct, oars in dataloader:
        print(f'MR Shape: {mr.shape}, CT Shape: {ct.shape}, OARs Shape: {oars.shape}')
        # 在这里可以进行进一步处理，例如模型训练等