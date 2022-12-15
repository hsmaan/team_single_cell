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
    def __init__(self, latent_dim, gex_dim, atac_dim, model, init="xavier"):
        super(DeepGexAtacMultiModalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.gex_dim = gex_dim
        self.atac_dim = atac_dim
        try:
            gexenc_layers, gexdec_layers, atacenc_layers, atacdec_layers = model
        except:
            print("Uncorrectly specified hidden layers: {}".format(model))
            raise RuntimeError
        # We have two encoders and decoder - for each modality
        # We divide latent dim by two because we are going to 
        # concatenate the two modalities in latent space and
        # then use that concatenated representation to reconstruct
        # each modality 
        gexenc_modules = []
        for index, value in enumerate(gexenc_layers):
            if index == 0:
                gexenc_modules.append(nn.Linear(self.gex_dim, value))
                gexenc_modules.append(nn.BatchNorm1d(value))
                gexenc_modules.append(nn.ReLU())
                gexenc_modules.append(nn.Dropout(p=0.1))
            elif index % 2 == 0:
                gexenc_modules.append(nn.Linear(gexenc_layers[index-1], value))
                gexenc_modules.append(nn.BatchNorm1d(value))
                gexenc_modules.append(nn.ReLU())
                gexenc_modules.append(nn.Dropout(p=0.1))
            else:
                gexenc_modules.append(nn.Linear(gexenc_layers[index-1], value))
                gexenc_modules.append(nn.BatchNorm1d(value))
                gexenc_modules.append(nn.ReLU())
        gexenc_modules.append(nn.Linear(gexenc_layers[-1], int(latent_dim/2)))
        gexenc_modules.append(nn.ReLU())
        self.gex_encoder = nn.Sequential(*gexenc_modules)

        atacenc_modules = []
        for index, value in enumerate(atacenc_layers):
            if index == 0:
                atacenc_modules.append(nn.Linear(self.atac_dim, value))
                atacenc_modules.append(nn.BatchNorm1d(value))
                atacenc_modules.append(nn.ReLU())
                atacenc_modules.append(nn.Dropout(p=0.1))
            elif index % 2 == 0:
                atacenc_modules.append(nn.Linear(atacenc_layers[index-1], value))
                atacenc_modules.append(nn.BatchNorm1d(value))
                atacenc_modules.append(nn.ReLU())
                atacenc_modules.append(nn.Dropout(p=0.1))
            else:
                atacenc_modules.append(nn.Linear(atacenc_layers[index-1], value))
                atacenc_modules.append(nn.BatchNorm1d(value))
                atacenc_modules.append(nn.ReLU())
        atacenc_modules.append(nn.Linear(atacenc_layers[-1], int(latent_dim/2)))
        atacenc_modules.append(nn.ReLU())
        self.atac_encoder = nn.Sequential(*atacenc_modules)

        gexdec_modules = []
        for index, value in enumerate(gexdec_layers):
            if index == 0:
                gexdec_modules.append(nn.Linear(self.latent_dim, value))
                gexdec_modules.append(nn.BatchNorm1d(value))
                gexdec_modules.append(nn.ReLU())
                gexdec_modules.append(nn.Dropout(p=0.1))
            elif index % 2 == 0:
                gexdec_modules.append(nn.Linear(gexdec_layers[index-1], value))
                gexdec_modules.append(nn.BatchNorm1d(value))
                gexdec_modules.append(nn.ReLU())
                gexdec_modules.append(nn.Dropout(p=0.1))
            else:
                gexdec_modules.append(nn.Linear(gexdec_layers[index-1], value))
                gexdec_modules.append(nn.BatchNorm1d(value))
                gexdec_modules.append(nn.ReLU())
        gexdec_modules.append(nn.Linear(gexdec_layers[-1], self.gex_dim))
        gexdec_modules.append(nn.ReLU())
        self.gex_decoder = nn.Sequential(*gexdec_modules)

        atacdec_modules = []
        for index, value in enumerate(atacdec_layers):
            if index == 0:
                atacdec_modules.append(nn.Linear(self.latent_dim, value))
                atacdec_modules.append(nn.BatchNorm1d(value))
                atacdec_modules.append(nn.ReLU())
                atacdec_modules.append(nn.Dropout(p=0.1))
            elif index % 2 == 0:
                atacdec_modules.append(nn.Linear(atacdec_layers[index-1], value))
                atacdec_modules.append(nn.BatchNorm1d(value))
                atacdec_modules.append(nn.ReLU())
                atacdec_modules.append(nn.Dropout(p=0.1))
            else:
                atacdec_modules.append(nn.Linear(atacdec_layers[index-1], value))
                atacdec_modules.append(nn.BatchNorm1d(value))
                atacdec_modules.append(nn.ReLU())
        atacdec_modules.append(nn.Linear(atacdec_layers[-1], self.atac_dim))
        atacdec_modules.append(nn.Sigmoid())
        self.atac_decoder = nn.Sequential(*atacdec_modules)
        
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
    def __init__(self, latent_dim, gex_dim, adt_dim, model, init="xavier"):
        super(DeepGexAdtMultiModalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.gex_dim = gex_dim
        self.adt_dim = adt_dim
        try:
            gexenc_layers, gexdec_layers, adtenc_layers, adtdec_layers = model
        except:
            print("Uncorrectly specified hidden layers: {}".format(model))
            raise RuntimeError
        
        gexenc_modules = []
        for index, value in enumerate(gexenc_layers):
            if index == 0:
                gexenc_modules.append(nn.Linear(self.gex_dim, value))
                gexenc_modules.append(nn.BatchNorm1d(value))
                gexenc_modules.append(nn.ReLU())
                gexenc_modules.append(nn.Dropout(p=0.1))
            elif index % 2 == 0:
                gexenc_modules.append(nn.Linear(gexenc_layers[index-1], value))
                gexenc_modules.append(nn.BatchNorm1d(value))
                gexenc_modules.append(nn.ReLU())
                gexenc_modules.append(nn.Dropout(p=0.1))
            else:
                gexenc_modules.append(nn.Linear(gexenc_layers[index-1], value))
                gexenc_modules.append(nn.BatchNorm1d(value))
                gexenc_modules.append(nn.ReLU())
        gexenc_modules.append(nn.Linear(gexenc_layers[-1], int(latent_dim/2)))
        gexenc_modules.append(nn.ReLU())
        self.gex_encoder = nn.Sequential(*gexenc_modules)

        adtenc_modules = []
        for index, value in enumerate(adtenc_layers):
            if index == 0:
                adtenc_modules.append(nn.Linear(self.adt_dim, value))
                adtenc_modules.append(nn.BatchNorm1d(value))
                adtenc_modules.append(nn.ReLU())
                adtenc_modules.append(nn.Dropout(p=0.1))
            elif index % 2 == 0:
                adtenc_modules.append(nn.Linear(adtenc_layers[index-1], value))
                adtenc_modules.append(nn.BatchNorm1d(value))
                adtenc_modules.append(nn.ReLU())
                adtenc_modules.append(nn.Dropout(p=0.1))
            else:
                adtenc_modules.append(nn.Linear(adtenc_layers[index-1], value))
                adtenc_modules.append(nn.BatchNorm1d(value))
                adtenc_modules.append(nn.ReLU())
        adtenc_modules.append(nn.Linear(adtenc_layers[-1], int(latent_dim/2)))
        adtenc_modules.append(nn.ReLU())
        self.adt_encoder = nn.Sequential(*adtenc_modules)

        gexdec_modules = []
        for index, value in enumerate(gexdec_layers):
            if index == 0:
                gexdec_modules.append(nn.Linear(self.latent_dim, value))
                gexdec_modules.append(nn.BatchNorm1d(value))
                gexdec_modules.append(nn.ReLU())
                gexdec_modules.append(nn.Dropout(p=0.1))
            elif index % 2 == 0:
                gexdec_modules.append(nn.Linear(gexdec_layers[index-1], value))
                gexdec_modules.append(nn.BatchNorm1d(value))
                gexdec_modules.append(nn.ReLU())
                gexdec_modules.append(nn.Dropout(p=0.1))
            else:
                gexdec_modules.append(nn.Linear(gexdec_layers[index-1], value))
                gexdec_modules.append(nn.BatchNorm1d(value))
                gexdec_modules.append(nn.ReLU())
        gexdec_modules.append(nn.Linear(gexdec_layers[-1], self.gex_dim))
        gexdec_modules.append(nn.ReLU())
        self.gex_decoder = nn.Sequential(*gexdec_modules)

        adtdec_modules = []
        for index, value in enumerate(adtdec_layers):
            if index == 0:
                adtdec_modules.append(nn.Linear(self.latent_dim, value))
                adtdec_modules.append(nn.BatchNorm1d(value))
                adtdec_modules.append(nn.ReLU())
                adtdec_modules.append(nn.Dropout(p=0.1))
            elif index % 2 == 0:
                adtdec_modules.append(nn.Linear(adtdec_layers[index-1], value))
                adtdec_modules.append(nn.BatchNorm1d(value))
                adtdec_modules.append(nn.ReLU())
                adtdec_modules.append(nn.Dropout(p=0.1))
            else:
                adtdec_modules.append(nn.Linear(adtdec_layers[index-1], value))
                adtdec_modules.append(nn.BatchNorm1d(value))
                adtdec_modules.append(nn.ReLU())
        adtdec_modules.append(nn.Linear(adtdec_layers[-1], self.adt_dim))
        adtdec_modules.append(nn.ReLU())
        self.adt_decoder = nn.Sequential(*adtdec_modules)
        
        # Weight initialization 
        if init == "xavier":
            self.gex_encoder.apply(xavier_init)
            self.adt_encoder.apply(xavier_init)
            self.gex_decoder.apply(xavier_init)
            self.adt_decoder.apply(xavier_init)
        else:
            self.gex_encoder.apply(he_init)
            self.adt_encoder.apply(he_init)
            self.gex_decoder.apply(he_init)
            self.adt_decoder.apply(he_init)
        
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
