import torch
import random
from torch.utils.data import DataLoader
import scanpy as sc

from datasets.anndatadataset import AnnDataDataset 
#from anndatadataset import AnnDataDataset


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size=1):
        super().__init__(dataset=dataset, batch_size=batch_size)
        if isinstance(dataset, AnnDataDataset):
            self.batch_counts = dataset.get_batch_counts()
            self.batch_indices = dataset.get_batch_indices()
            self.batch_iter = self._create_class_iter()

    def _create_class_iter(self):
        for label, indices in self.batch_indices.items():
            index_len = indices.shape[0]
            index_perm = torch.randperm(index_len)
            self.batch_indices[label] = indices[index_perm]
        
        class_indices_copy = self.batch_indices.copy()
        class_counts_copy = self.batch_counts.copy()
        
        while len(class_counts_copy) >=1:
            label, indices = random.choice(list(class_indices_copy.items()))
            if len(indices) < self.batch_size:
                class_counts_copy.pop(label)
                class_indices_copy.pop(label)
                continue
            yield label, indices[:self.batch_size]
            class_indices_copy[label] = indices[self.batch_size:]

    def __iter__(self):
        labels = list(self.batch_indices.keys())
        random.shuffle(labels)
        self.batch_indices = {label:self.batch_indices[label] for label in labels}
        self.batch_counts = {label:self.batch_counts[label] for label in labels}
        self.batch_iter = self._create_class_iter()
        return self

    def __next__(self):
        try:
            _, indices = next(self.batch_iter)
            return self.dataset[indices]
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return sum(count for _, count in self.batch_counts)

if __name__ == '__main__':
    cite = sc.read_h5ad("data/multimodal/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")
    ds = AnnDataDataset(cite)
    ds_loader = BaseDataLoader(ds, batch_size=10)
    i = 0
    for batch in ds_loader:
        print(batch)
        i += 1
        if i == 5:
            break