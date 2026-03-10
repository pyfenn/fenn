import torch
from torch.utils.data import Dataset

class BinaryDataset(Dataset):

    def __init__(self, X, y):
        super().__init__()
        self._X = X
        self._y = y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, index):
        x = torch.tensor(self._X[index], dtype=torch.float)
        y = torch.tensor(self._y[index], dtype=torch.float)
        y = y.unsqueeze(0)  # [B] -> [B,1] - for binary cross entropy
        return x, y