import scanpy as sc
import anndata as ann 
import numpy as np
import pandas as pd 
import sklearn 

# Create class to perform evaluation of embeddings for both
# cell-type and batch labels 
class EvalEmbeddings:
    def __init__(self, adata):
        self.adata = adata
        self.celltype = adata.obs['cell_type'].values
        self.batch = adata.obs['batch'].values

    def evaluate(self, embeddings, leiden_res=1):
        # First cluster the embeddings using Leiden clustering
        # Add the embeddings to the adata object
        self.adata.obsm['X_custom'] = embeddings
        
        # Get the neighborhood graph using custom embeddings
        sc.pp.neighbors(self.adata, use_rep='X_custom')
        
        # Cluster the embeddings using Leiden clustering
        sc.tl.leiden(self.adata, resolution=leiden_res, key_added='leiden_custom')
        
        # Get the UMAP representation of the data 
        sc.tl.umap(self.adata)
        
        # Evaluate the clustering compared to the cell-type using 
        # ARI, AMI, Homogeneity, and Completeness
        celltype_ari = sklearn.metrics.adjusted_rand_score(
            self.celltype, self.adata.obs['leiden_custom']
        )
        celltype_ami = sklearn.metrics.adjusted_mutual_info_score(
            self.celltype, self.adata.obs['leiden_custom']
        )
        celltype_homogeneity = sklearn.metrics.homogeneity_score(
            self.celltype, self.adata.obs['leiden_custom']
        )
        celltype_complete = sklearn.metrics.completeness_score(
            self.celltype, self.adata.obs['leiden_custom']
        )
        
        # Evaluate the clustering compared to the batch using the same metrics
        batch_ari = 1 - sklearn.metrics.adjusted_rand_score(
            self.batch, self.adata.obs['leiden_custom']
        )
        batch_ami = 1 - sklearn.metrics.adjusted_mutual_info_score(
            self.batch, self.adata.obs['leiden_custom']
        )
        batch_homogeneity = 1 - sklearn.metrics.homogeneity_score(
            self.batch, self.adata.obs['leiden_custom']
        )
        batch_complete = 1 - sklearn.metrics.completeness_score(
            self.batch, self.adata.obs['leiden_custom']
        )
        
        # Create a dataframe of the results 
        res_df = pd.DataFrame(
            {
                'celltype_ari': celltype_ari,
                'celltype_ami': celltype_ami,
                'celltype_homogeneity': celltype_homogeneity,
                'celltype_complete': celltype_complete,
                'batch_ari': batch_ari,
                'batch_ami': batch_ami,
                'batch_homogeneity': batch_homogeneity,
                'batch_complete': batch_complete
            },
            index=[0]
        )
        
        # Get the average of the cell-type and batch metrics
        celltype_avg = np.mean(
            np.array([celltype_ari, celltype_ami, celltype_homogeneity, celltype_complete])
        )
        batch_avg = np.mean(
            np.array([batch_ari, batch_ami, batch_homogeneity, batch_complete])
        )
        total_score = (0.6 * celltype_avg) + (0.4 * batch_avg)
        
        return total_score, res_df
        
    def plot(self):
        # Plot the embeddings colored by cluster, cell-type, and batch
        sc.pl.umap(self.adata, color = 'cell_type')
        sc.pl.umap(self.adata, color = 'batch')
        sc.pl.umap(self.adata, color = 'leiden_custom')
        
        
        