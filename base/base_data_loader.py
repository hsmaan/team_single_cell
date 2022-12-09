import torch
import random
from torch.utils.data import DataLoader

from datasets.anndatadataset import AnnDataDataset 


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last)
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
