import torch
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, tensor1, tensor2, tensor_list):
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.tensor_list = tensor_list

    def __len__(self):
        return self.tensor1.shape[0] // 1

    def __getitem__(self, idx):
        start_idx = idx * 1
        end_idx = start_idx + 1
        tensor1_slice = self.tensor1[start_idx:end_idx, :, :].unsqueeze(1)
        tensor2_slice = self.tensor2[start_idx:end_idx, :, :].unsqueeze(1)
        tensor_list_slices = [tensor[start_idx:end_idx, :, :] for tensor in self.tensor_list]
        stacked_tensor = torch.stack(tensor_list_slices, dim=1)  # 在第 0 维上堆叠
        return tensor1_slice, tensor2_slice, stacked_tensor


        # stacked_tensor = torch.stack(tensor_list_slices, dim=0)  # 在第 0 维上堆叠
        # return tensor1_slice, tensor2_slice, stacked_tensor

def getDateset():
    # 假设你已经获取了tensor1, tensor2和tensor_list
    tensor1 = torch.randn(160, 512, 512)
    tensor2 = torch.randn(160, 512, 512)
    tensor_list = [torch.randn(160, 512, 512) for _ in range(5)]  # 假设tensor_list包含5个tensor

    dataset = MyDataset(tensor1, tensor2, tensor_list)
    return dataset

