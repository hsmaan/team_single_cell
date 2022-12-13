from base.base_data_loader import BaseDataLoader
from datasets.anndatadataset import AnnDataDataset

class WrapperDataLoader(BaseDataLoader):
    def __init__(self, dataset:AnnDataDataset, batch_size=1, 
        shuffle=True, num_workers=4, drop_last=True) -> None:

        super().__init__(dataset, batch_size, shuffle, num_workers, drop_last)

# Not used
class GexAdtDataLoader(BaseDataLoader):
    def __init__(self, dataset:AnnDataDataset, gex_dim, adt_dim, batch_size=1,
        shuffle=True, num_workers=4, drop_last=True) -> None:
        expected_dim = gex_dim + adt_dim
        true_dim = dataset.shape(dim=1)  
        assert expected_dim == true_dim

        self.gex_dim = gex_dim
        self.adt_dim = adt_dim
        
        super().__init__(dataset, batch_size, shuffle, num_workers, drop_last)
