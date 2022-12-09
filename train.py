import torch
import torch.optim as optim

from models.multimodal_autoencoder import MultiModalAutoencoder
from data_loader.data_loaders import WrapperDataLoader
from datasets.anndatadataset import AnnDataDataset
from models.loss import gex_atac_loss

class GexAtacTrainer:
    def __init__(self, gex_atac_adata, batches, latent_dim, gex_dim, atac_dim) -> None:

        self.gex_dim = gex_dim
        self.atac_dim = atac_dim
        # Initialize autoencoder
        self.autoencoder = MultiModalAutoencoder(
            latent_dim = latent_dim,
            gex_dim = gex_dim,
            atac_dim = atac_dim
        )

        # Init dataset
        print("Initializing dataset and dataloader...")
        self.gex_atac_ds = AnnDataDataset(gex_atac_adata, batches, gex_dim, atac_dim)

        # Create dataloader for concatenated data
        self.gex_atac_loader = WrapperDataLoader(
            dataset = self.gex_atac_ds,
            batch_size = 512,
            shuffle = True,
            num_workers = 4,
            drop_last = True
        )

        # use the GPU if available (it should be)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.autoencoder.double().to(self.device)

        # Adam optimizer 
        self.optimizer = optim.Adam(
            self.autoencoder.parameters(),
            lr = 0.001,
            weight_decay = 0.001
        )


        # Loggers
        print()
        print("The model: {};".format(str(self.autoencoder)))
        print("The dataset: {};".format(self.gex_atac_ds))
        print("The optimizer: {};".format(self.optimizer))
        print()

    def train(self):

        # Train the model for 20 epochs 
        self.autoencoder.train()
        for epoch in range(20):
            running_recon_loss = 0.0
            steps = 0
            for index, batch in enumerate(self.gex_atac_loader):
                data_tensor, batch_tensor = batch
                self.optimizer.zero_grad()
                inputs = data_tensor.double().to(self.device)
                outs = self.autoencoder.forward(inputs)
                loss = gex_atac_loss(outs, inputs, self.gex_dim, self.atac_dim)
                running_recon_loss += loss.cpu().detach().item()
                steps += 1
                loss.backward()
                self.optimizer.step()
            epoch_recon_loss = running_recon_loss/steps
            print(
                "Epoch " + 
                str(epoch) + 
                " recon loss: " +
                str(round(epoch_recon_loss, 10))
            )
