from torch import Tensor
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, X:Tensor, y:Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_sample = self.X[idx]
        y_sample = self.y[idx]
        return X_sample, y_sample

    def shape(self, dim=None):
        if dim:
            return self.X.size(dim=dim)
        else:
            return self.X.size()