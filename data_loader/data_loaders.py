from torchvision import datasets, transforms
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
    def __init__(self, gex_adata, atac_adata, batch_size, shuffle=False, validation_split=0.0, num_workers=1, training=True) -> None:
        gex_arr = gex_adata.X.todense()
        atac_arr = atac_adata.X.todense()
        gex_data = 0 # Convert numpy array to torch IterableDataSet ??
        atac_data = 0 # Convert numpy array to torch IterableDataSet ??
        super().__init__(gex_adata, atac_adata, batch_size, shuffle, validation_split, num_workers)

class GexAdtDataLoader(BaseDataLoader):
    pass