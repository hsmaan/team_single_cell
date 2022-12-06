import torch
import scanpy as sc
from base import BaseDataLoader


#class MnistDataLoader(BaseDataLoader):
    #"""
    #MNIST data loading demo using BaseDataLoader
    #"""
    #def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        #trsfm = transforms.Compose([
            #transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        #])
        #self.data_dir = data_dir
        #self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        #super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class GexAtacDataLoader(BaseDataLoader):
    def __init__(self, directory, batch_size=1) -> None:
        gex_atac = sc.read_h5ad(directory)
        gex_atac_tensor = torch.from_numpy(gex_atac.X.todense())
        super().__init__(gex_atac_tensor, batch_size)

class GexAdtDataLoader(BaseDataLoader):
    pass