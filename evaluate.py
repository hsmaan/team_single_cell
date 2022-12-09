import torch

from helpers.eval_embeddings import EvalEmbeddings
from datasets.anndatadataset import AnnDataDataset

class GexAtacEvaluation:
    def __init__(self, autoencoder, dataset:AnnDataDataset, adata, device) -> None:
        # Put the model in eval mode and turn off torch gradients to get 
        # the model embeddings 
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
        print("AnnData object shape: {};".format(self.adata.shape))
        print("Latent space shape: {}".format(self.latent_repr.size))
        print()

    def evaluate(self):
        eval_obj = EvalEmbeddings(adata=self.adata) # Initialize the class
        total_score, res_df = eval_obj.evaluate(self.latent_repr) # Evaluate
        print(total_score)
        print(res_df)
        eval_obj.plot()