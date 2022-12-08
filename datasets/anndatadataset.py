import torch
from anndata import AnnData
import scanpy as sc
from sklearn import preprocessing

from base import BaseDataset
#from base_dataset import BaseDataset

class AnnDataDataset(BaseDataset):
    def __init__(self, dataset:AnnData) -> None:
        self.X = torch.from_numpy(dataset.X.todense())
        self.y = torch.from_numpy(self._get_nparr_of_batches(dataset.obs["batch"]))
        # Think about some other data that we need to save from AnnData object

        super().__init__(self.X, self.y)
    
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

    
if __name__ == "__main__":
    cite = sc.read_h5ad("data/multimodal/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")
    ds = AnnDataDataset(cite)