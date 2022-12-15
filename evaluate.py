import torch

from helpers.eval_embeddings import EvalEmbeddings
from datasets.anndatadataset import AnnDataDataset
from helpers.preprocessing import GexAdtPreprocess, GexAtacPreprocess
from train import GexAtacTrainer, GexAdtTrainer

class GexAtacEvaluation:
    def __init__(self, train_obj:GexAtacTrainer, preprocess_obj:GexAtacPreprocess) -> None:
        # Put the model in eval mode and turn off torch gradients to get 
        # the model embeddings 
        autoencoder = train_obj.autoencoder
        dataset = train_obj.gex_atac_ds
        device = train_obj.device
        adata = preprocess_obj.dataset
        autoencoder.eval()
        with torch.no_grad():
            # Have to move tensors to device and put them in the same dtype or else
            # this won't work
            gex_Z = autoencoder.gex_encoder(dataset.get_first_modality().double().to(device))
            atac_Z = autoencoder.atac_encoder(dataset.get_second_modality().double().to(device))
            
        Z_concat = torch.cat([gex_Z, atac_Z], axis=1)
        Z_concat_np = Z_concat.cpu().detach().numpy()
        self.latent_repr = Z_concat_np
        self.adata = adata

        # Loggers
        print()
        print("Latent space type: {};".format(type(self.latent_repr)))
        print("GEX encoded shape: {};".format(gex_Z.shape))
        print("ATAC encoded shape: {};".format(atac_Z.shape))
        print("Latent space shape: {};".format(self.latent_repr.shape))
        print()

    def evaluate(self):
        eval_obj = EvalEmbeddings(adata=self.adata) # Initialize the class
        total_score, res_df = eval_obj.evaluate(self.latent_repr) # Evaluate
        print(total_score)
        print(res_df)
        eval_obj.plot()
        return (total_score, res_df)

class GexAdtEvaluation:
    def __init__(self, train_obj:GexAdtTrainer, preprocess_obj:GexAdtPreprocess) -> None:
        # Put the model in eval mode and turn off torch gradients to get 
        # the model embeddings 
        autoencoder = train_obj.autoencoder
        dataset = train_obj.gex_adt_ds
        device = train_obj.device
        adata = preprocess_obj.dataset
        autoencoder.eval()
        with torch.no_grad():
            # Have to move tensors to device and put them in the same dtype or else
            # this won't work
            gex_Z = autoencoder.gex_encoder(dataset.get_first_modality().double().to(device))
            adt_Z = autoencoder.adt_encoder(dataset.get_second_modality().double().to(device))
            
        Z_concat = torch.cat([gex_Z, adt_Z], axis=1)
        Z_concat_np = Z_concat.cpu().detach().numpy()
        self.latent_repr = Z_concat_np
        self.adata = adata

        # Loggers
        print()
        print("Latent space type: {};".format(type(self.latent_repr)))
        print("GEX encoded shape: {};".format(gex_Z.shape))
        print("ADT encoded shape: {};".format(adt_Z.shape))
        print("Latent space shape: {};".format(self.latent_repr.shape))
        print()

    def evaluate(self):
        eval_obj = EvalEmbeddings(adata=self.adata) # Initialize the class
        total_score, res_df = eval_obj.evaluate(self.latent_repr) # Evaluate
        print(total_score)
        print(res_df)
        eval_obj.plot()
        return (total_score, res_df)
