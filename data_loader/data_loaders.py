import torch
import scanpy as sc
from base import BaseDataLoader
from datasets.anndatadataset import AnnDataDataset
#from base_data_loader import BaseDataLoader
#from anndatadataset import AnnDataDataset

class GexAtacDataLoader(BaseDataLoader):
    def __init__(self, dataset:AnnDataDataset, gex_dim, atac_dim, batch_size=1) -> None:
        expected_dim = gex_dim + atac_dim
        true_dim = dataset.shape(dim=1)  
        try:
            assert expected_dim == true_dim
        except AssertionError:
            print("AssertionError: Dimensions do not match: {} != {}".format((gex_dim+atac_dim), dataset.shape(dim=1)))
            if abs(expected_dim - true_dim) > 2:
                exit()

        self.gex_dim = gex_dim
        self.atac_dim = atac_dim
        
        super().__init__(dataset=dataset, batch_size=batch_size)

class GexAdtDataLoader(BaseDataLoader):
    def __init__(self, dataset:AnnDataDataset, gex_dim, adt_dim, batch_size=1) -> None:
        expected_dim = gex_dim + adt_dim
        true_dim = dataset.shape(dim=1)  
        try:
            assert expected_dim == true_dim
        except AssertionError:
            print("AssertionError: Dimensions do not match: {} != {}".format((gex_dim+adt_dim), dataset.shape(dim=1)))
            if abs(expected_dim - true_dim) > 2:
                exit()

        self.gex_dim = gex_dim
        self.adt_dim = adt_dim
        
        super().__init__(dataset=dataset, batch_size=batch_size)

if __name__ == "__main__":
    assert (1 == 0)