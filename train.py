import torch
import torch.optim as optim
import wandb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

from models.multimodal_autoencoder import GexAtacMultiModalAutoencoder, GexAdtMultiModalAutoencoder, DeepGexAdtMultiModalAutoencoder, DeepGexAtacMultiModalAutoencoder
from data_loader.data_loaders import WrapperDataLoader
from datasets.anndatadataset import AnnDataDataset
from models.loss import gex_atac_loss, gex_adt_loss
from helpers.preprocessing import GexAdtPreprocess, GexAtacPreprocess

class GexAtacTrainer:
    def __init__(self, preprocess_object: GexAtacPreprocess, latent_dim, model=[],
    init="xavier", lr=0.001, weight_decay=0.001) -> None:
        """
        model - list of 4 lists of number of features in HIDDEN layers (the first number is NOT
        the first modality dim and the last number is NOT the latent dim - for encoder).
        Each list for each modality encoder and decoder: [[1stmodality_encoder], [1stmodality_decoder],
        [2ndmodality_encoder], [2ndmodality_decoder]].
        """

        self.gex_dim = preprocess_object.gex_dim
        self.atac_dim = preprocess_object.atac_dim
        gex_atac_adata = preprocess_object.dataset
        batches = preprocess_object.obs["batch"]
        # Initialize autoencoder
        if model == []:
            self.autoencoder = GexAtacMultiModalAutoencoder(
                latent_dim=latent_dim,
                gex_dim=self.gex_dim,
                atac_dim=self.atac_dim
            )
        else:
            self.autoencoder = DeepGexAtacMultiModalAutoencoder(
                latent_dim=latent_dim,
                gex_dim=self.gex_dim,
                atac_dim=self.atac_dim,
                model=model,
                init=init
            )

        # Init dataset
        print("Initializing dataset and dataloader...")
        self.gex_atac_ds = AnnDataDataset(gex_atac_adata, batches, self.gex_dim, self.atac_dim)

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
        print("The device: {};".format(self.device))
        self.autoencoder.double().to(self.device)

        # Adam optimizer 
        self.optimizer = optim.Adam(
            self.autoencoder.parameters(),
            lr = lr,
            weight_decay = weight_decay
        )

        # Loggers
        print()
        print("The model: {};".format(str(self.autoencoder)))
        print("The dataset: {};".format(self.gex_atac_ds))
        print("The optimizer: {};".format(self.optimizer))
        print()

    def train(self, epochs, gex_loss_w=5, atac_loss_w=5, wandb_log=False):
        self.autoencoder.train()
        for epoch in range(epochs):
            running_recon_loss = 0.0
            steps = 0
            for index, batch in enumerate(self.gex_atac_loader):
                data_tensor, batch_tensor = batch
                self.optimizer.zero_grad()
                inputs = data_tensor.double().to(self.device)
                outs = self.autoencoder.forward(inputs)
                loss = gex_atac_loss(outs, inputs, self.gex_dim, self.atac_dim, gex_loss_w, atac_loss_w)
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
            if wandb_log:
                wandb.log({"recon loss": epoch_recon_loss})

class GexAdtTrainer:
    def __init__(self, preprocess_object: GexAdtPreprocess, latent_dim, model=[],
    init="xavier", lr=0.001, weight_decay=0.001) -> None:
        """
        model - list of 4 lists of number of features in HIDDEN layers (the first number is NOT
        the first modality dim and the last number is NOT the latent dim - for encoder).
        Each list for each modality encoder and decoder: [[1stmodality_encoder], [1stmodality_decoder],
        [2ndmodality_encoder], [2ndmodality_decoder]].
        """

        self.gex_dim = preprocess_object.gex_dim
        self.adt_dim = preprocess_object.adt_dim
        gex_adt_adata = preprocess_object.dataset
        batches = preprocess_object.obs["batch"]
        # Initialize autoencoder
        if model == []:
            self.autoencoder = GexAdtMultiModalAutoencoder(
                latent_dim=latent_dim,
                gex_dim=self.gex_dim,
                adt_dim=self.adt_dim
            )
        else:
            self.autoencoder = DeepGexAdtMultiModalAutoencoder(
                latent_dim=latent_dim,
                gex_dim=self.gex_dim,
                adt_dim=self.adt_dim,
                model=model,
                init=init
            )

        # Init dataset
        print("Initializing dataset and dataloader...")
        self.gex_adt_ds = AnnDataDataset(gex_adt_adata, batches, self.gex_dim, self.adt_dim)

        # Create dataloader for concatenated data
        self.gex_atac_loader = WrapperDataLoader(
            dataset = self.gex_adt_ds,
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
        print("The device: {};".format(self.device))
        self.autoencoder.double().to(self.device)

        # Adam optimizer 
        self.optimizer = optim.Adam(
            self.autoencoder.parameters(),
            lr = lr,
            weight_decay = weight_decay
        )

        # Loggers
        print()
        print("The model: {};".format(str(self.autoencoder)))
        print("The dataset: {};".format(self.gex_adt_ds))
        print("The optimizer: {};".format(self.optimizer))
        print()

    def train(self, epochs, gex_loss_w=5, adt_loss_w=5, wandb_log=False):

        self.autoencoder.train()
        for epoch in range(epochs):
            running_recon_loss = 0.0
            steps = 0
            for index, batch in enumerate(self.gex_atac_loader):
                data_tensor, batch_tensor = batch
                self.optimizer.zero_grad()
                inputs = data_tensor.double().to(self.device)
                outs = self.autoencoder.forward(inputs)
                loss = gex_adt_loss(outs, inputs, self.gex_dim, self.adt_dim, gex_loss_w, adt_loss_w)
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
            if wandb_log:
                wandb.log({"recon loss": epoch_recon_loss})
