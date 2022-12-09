import torch
from anndata import AnnData
import scanpy as sc
from sklearn import preprocessing

from base.base_dataset import BaseDataset

class AnnDataDataset(BaseDataset):
    def __init__(self, dataset:AnnData, fmod_dim, smod_dim) -> None:
        self.X = torch.from_numpy(dataset.X.todense())
        self.y = torch.from_numpy(self._get_nparr_of_batches(dataset.obs["batch"]))
        # Think about some other data that we need to save from AnnData object
        self.fmod_dim = fmod_dim
        self.smod_dim = smod_dim
        assert self.X.size(dim=1) == (fmod_dim + smod_dim)

        super().__init__(self.X, self.y)

    def get_first_modality(self):
        return self.X[:, 0:self.fmod_dim]

    def get_second_modality(self):
        return self.X[:, self.fmod_dim:]
    
    def _get_nparr_of_batches(self, obs_batches):
        labels = obs_batches.values
        le = preprocessing.LabelEncoder()
        print("The order of labels: {}".format(labels))
        targets = le.fit_transform(labels)
        return targets

    def get_batch_counts(self):
        unique_batches = torch.unique(self.y)
        batch_counts = {}
        for i in unique_batches:
            batch_counts[i.item()] = len(self.X[self.y == i])
        return batch_counts

    # Get the indices corresponding to each class in the 
    # given dataset 
    def get_batch_indices(self):
        unique_batches = torch.unique(self.y)
        batch_indices = {}
        for i in unique_batches:
            batch_indices[i.item()] = torch.where(self.y == i)[0]
        return batch_indices
