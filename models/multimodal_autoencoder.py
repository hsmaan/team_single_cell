import torch
import torch.nn as nn
import torch.nn.functional as F

# We're going to define this function to initialize weights 
# To learn more about xavier initialization, see:
# https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf 

# This function will be applied to all the layers in our model
# - it will initialize the weights of each layer to be
# sampled from a uniform distribution
def xavier_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

def he_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)

# This is the class for our multimodal autoencoder
class DeepGexAtacMultiModalAutoencoder(nn.Module):
    def __init__(self, latent_dim, gex_dim, atac_dim, init="xavier"):
        super(DeepGexAtacMultiModalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.gex_dim = gex_dim
        self.atac_dim = atac_dim
        # We have two encoders and decoder - for each modality
        # We divide latent dim by two because we are going to 
        # concatenate the two modalities in latent space and
        # then use that concatenated representation to reconstruct
        # each modality 
        self.gex_encoder = nn.Sequential(
            nn.Linear(self.gex_dim, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1200, 900),
            nn.BatchNorm1d(900),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(900, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            nn.Linear(600, 480),
            nn.BatchNorm1d(480),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(480, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(120, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Linear(80, int(self.latent_dim/2)),
            nn.BatchNorm1d(int(self.latent_dim/2)),
            nn.ReLU()
        )
        self.atac_encoder = nn.Sequential(
            nn.Linear(self.atac_dim, 3200),
            nn.BatchNorm1d(3200),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(3200, 1600),
            nn.BatchNorm1d(1600),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1600, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(),
            nn.Linear(1280, 920),
            nn.BatchNorm1d(920),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(920, 440),
            nn.BatchNorm1d(440),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(440, 240),
            nn.BatchNorm1d(240),
            nn.ReLU(),
            nn.Linear(240, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Linear(120, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Linear(80, int(self.latent_dim/2)),
            nn.BatchNorm1d(int(self.latent_dim/2)),
            nn.ReLU()
        )
        self.gex_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(80, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(120, 240),
            nn.BatchNorm1d(240),
            nn.ReLU(),
            nn.Linear(240, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(600, 900),
            nn.BatchNorm1d(900),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(900, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Linear(1200, self.gex_dim),
            nn.ReLU()
        )
        self.atac_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(80, 240),
            nn.BatchNorm1d(240),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(240, 480),
            nn.BatchNorm1d(480),
            nn.ReLU(),
            nn.Linear(480, 900),
            nn.BatchNorm1d(900),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(900, 1280),
            nn.BatchNorm1d(1280),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1280, 1600),
            nn.BatchNorm1d(1600),
            nn.ReLU(),
            nn.Linear(1600, 2400),
            nn.BatchNorm1d(2400),
            nn.ReLU(),
            nn.Linear(2400, 3200),
            nn.BatchNorm1d(3200),
            nn.ReLU(),
            nn.Linear(3200, self.atac_dim),
            nn.Sigmoid()
        )
        
        # Weight initialization 
        if init == "xavier":
            self.gex_encoder.apply(xavier_init)
            self.atac_encoder.apply(xavier_init)
            self.gex_decoder.apply(xavier_init)
            self.atac_decoder.apply(xavier_init)
        else:
            self.gex_encoder.apply(he_init)
            self.atac_encoder.apply(he_init)
            self.gex_decoder.apply(he_init)
            self.atac_decoder.apply(he_init)
        
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

class DeepGexAdtMultiModalAutoencoder(nn.Module):
    def __init__(self, latent_dim, gex_dim, adt_dim, init="xavier"):
        super(DeepGexAdtMultiModalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.gex_dim = gex_dim
        self.adt_dim = adt_dim
        self.gex_encoder = nn.Sequential(
            nn.Linear(self.gex_dim, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1200, 900),
            nn.BatchNorm1d(900),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(900, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            nn.Linear(600, 480),
            nn.BatchNorm1d(480),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(480, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(120, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Linear(80, int(self.latent_dim/2)),
            nn.BatchNorm1d(int(self.latent_dim/2)),
            nn.ReLU()
        )
        self.adt_encoder = nn.Sequential(
            nn.Linear(self.adt_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(100, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(80, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, int(self.latent_dim/2)),
            nn.BatchNorm1d(int(self.latent_dim/2)),
            nn.ReLU()
        )
        self.gex_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(80, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(120, 240),
            nn.BatchNorm1d(240),
            nn.ReLU(),
            nn.Linear(240, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(600, 900),
            nn.BatchNorm1d(900),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(900, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(),
            nn.Linear(1200, self.gex_dim),
            nn.ReLU()
        )
        self.adt_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(48, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(96, self.adt_dim),
            nn.ReLU()
        )
        
        # Weight initialization 
        if init == "xavier":
            self.gex_encoder.apply(xavier_init)
            self.atac_encoder.apply(xavier_init)
            self.gex_decoder.apply(xavier_init)
            self.atac_decoder.apply(xavier_init)
        else:
            self.gex_encoder.apply(he_init)
            self.atac_encoder.apply(he_init)
            self.gex_decoder.apply(he_init)
            self.atac_decoder.apply(he_init)
        
    def gex_encode(self, gex_X):
        gex_Z = self.gex_encoder(gex_X)
        return gex_Z
    
    def gex_decode(self, gex_adt_c):
        gex_X_decoded = self.gex_decoder(gex_adt_c)
        return gex_X_decoded
        
    def adt_encode(self, adt_X):
        adt_Z = self.adt_encoder(adt_X)
        return adt_Z
        
    def adt_decode(self, gex_adt_c):
        adt_X_decoded = self.adt_decoder(gex_adt_c)
        return adt_X_decoded
        
    def forward(self, gex_adt_X):
        # Extract the data
        gex_X = gex_adt_X[:, 0:self.gex_dim]
        adt_X = gex_adt_X[:, self.gex_dim:]
        # Encode both the GEX and ATAC data 
        gex_Z = self.gex_encode(gex_X)
        adt_Z = self.adt_encode(adt_X)
        # Concatenate the two encoded modalities 
        gex_adt_c = torch.cat([gex_Z, adt_Z], axis =1) # This is our latent we'll use later
        # Use the concatenated representation to recover both GEx and ATAC
        gex_X_decoded = self.gex_decode(gex_adt_c)
        adt_X_decoded = self.adt_decode(gex_adt_c)
        gex_adt_X_decoded = torch.cat([gex_X_decoded, adt_X_decoded], axis=1)
        return gex_adt_X_decoded 

class GexAtacMultiModalAutoencoder(nn.Module):
    def __init__(self, latent_dim, gex_dim, atac_dim):
        super(GexAtacMultiModalAutoencoder, self).__init__()
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
        self.gex_encoder.apply(xavier_init)
        self.atac_encoder.apply(xavier_init)
        self.gex_decoder.apply(xavier_init)
        self.atac_decoder.apply(xavier_init)
        
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


class GexAdtMultiModalAutoencoder(nn.Module):
    def __init__(self, latent_dim, gex_dim, adt_dim):
        super(GexAdtMultiModalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.gex_dim = gex_dim
        self.adt_dim = adt_dim
        self.gex_encoder = nn.Sequential(
            nn.Linear(self.gex_dim, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, int(self.latent_dim/2)),
            nn.BatchNorm1d(int(self.latent_dim/2)),
            nn.ReLU()
        )
        self.adt_encoder = nn.Sequential(
            nn.Linear(self.adt_dim, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, int(self.latent_dim/2)),
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
        self.adt_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, self.adt_dim),
            nn.ReLU()
        )
        
        # Weight initialization 
        self.gex_encoder.apply(xavier_init)
        self.adt_encoder.apply(xavier_init)
        self.gex_decoder.apply(xavier_init)
        self.adt_decoder.apply(xavier_init)
        
    def gex_encode(self, gex_X):
        gex_Z = self.gex_encoder(gex_X)
        return gex_Z
    
    def gex_decode(self, gex_adt_c):
        gex_X_decoded = self.gex_decoder(gex_adt_c)
        return gex_X_decoded
        
    def adt_encode(self, adt_X):
        adt_Z = self.adt_encoder(adt_X)
        return adt_Z
        
    def adt_decode(self, gex_adt_c):
        adt_X_decoded = self.adt_decoder(gex_adt_c)
        return adt_X_decoded
        
    def forward(self, gex_adt_X):
        # Extract the data
        gex_X = gex_adt_X[:, 0:self.gex_dim]
        adt_X = gex_adt_X[:, self.gex_dim:]
        # Encode both the GEX and ATAC data 
        gex_Z = self.gex_encode(gex_X)
        adt_Z = self.adt_encode(adt_X)
        # Concatenate the two encoded modalities 
        gex_adt_c = torch.cat([gex_Z, adt_Z], axis =1) # This is our latent we'll use later
        # Use the concatenated representation to recover both GEx and ATAC
        gex_X_decoded = self.gex_decode(gex_adt_c)
        adt_X_decoded = self.adt_decode(gex_adt_c)
        gex_adt_X_decoded = torch.cat([gex_X_decoded, adt_X_decoded], axis=1)
        return gex_adt_X_decoded 
