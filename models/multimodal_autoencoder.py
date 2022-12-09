import torch
import torch.nn as nn
import torch.nn.functional as F

# We're going to define this function to initialize weights 
# To learn more about xavier initialization, see:
# https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf 

# This function will be applied to all the layers in our model
# - it will initialize the weights of each layer to be
# sampled from a uniform distribution
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

# This is the class for our multimodal autoencoder
class MultiModalAutoencoder(nn.Module):
    def __init__(self, latent_dim, gex_dim, atac_dim):
        super(MultiModalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.gex_dim = gex_dim
        self.atac_dim = atac_dim
        # We have two encoders and decoder - for each modality
        # We divide latent dim by two because we are going to 
        # concatenate the two modalities in latent space and
        # then use that concatenated representation to reconstruct
        # each modality 
        self.gex_encoder = nn.Sequential(
            nn.Linear(self.gex_dim, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, int(self.latent_dim/2)),
            nn.BatchNorm1d(int(self.latent_dim/2)),
            nn.ReLU()
        )
        self.atac_encoder = nn.Sequential(
            nn.Linear(self.atac_dim, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, int(self.latent_dim/2)),
            nn.BatchNorm1d(int(self.latent_dim/2)),
            nn.ReLU()
        )
        self.gex_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, self.gex_dim),
            nn.ReLU()
        )
        self.atac_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, self.atac_dim),
            nn.Sigmoid()
        )
        
        # Weight initialization 
        self.gex_encoder.apply(weights_init)
        self.atac_encoder.apply(weights_init)
        self.gex_decoder.apply(weights_init)
        self.atac_decoder.apply(weights_init)
        
    def gex_encode(self, gex_X):
        gex_Z = self.gex_encoder(gex_X)
        return gex_Z
    
    def gex_decode(self, gex_atac_c):
        gex_X_decoded = self.gex_decoder(gex_atac_c)
        return gex_X_decoded
        
    def atac_encode(self, atac_X):
        atac_Z = self.atac_encoder(atac_X)
        return atac_Z
        
    def atac_decode(self, gex_atac_c):
        atac_X_decoded = self.atac_decoder(gex_atac_c)
        return atac_X_decoded
        
    def forward(self, gex_atac_X):
        # Extract the data
        gex_X = gex_atac_X[:, 0:self.gex_dim]
        atac_X = gex_atac_X[:, self.gex_dim:]
        # Encode both the GEX and ATAC data 
        gex_Z = self.gex_encode(gex_X)
        atac_Z = self.atac_encode(atac_X)
        # Concatenate the two encoded modalities 
        gex_atac_c = torch.cat([gex_Z, atac_Z], axis =1) # This is our latent we'll use later
        # Use the concatenated representation to recover both GEx and ATAC
        gex_X_decoded = self.gex_decode(gex_atac_c)
        atac_X_decoded = self.atac_decode(gex_atac_c)
        gex_atac_X_decoded = torch.cat([gex_X_decoded, atac_X_decoded], axis=1)
        return gex_atac_X_decoded 
