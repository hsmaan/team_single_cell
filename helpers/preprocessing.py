import os

import scanpy as sc 
import anndata as ann
import episcanpy as esc

class GexAtacPreprocess:
    def __init__(self, gex_dim, atac_dim) -> None:
        """
        gex_dim - Number of GEX features to select and pass to the model
        atac_dim - Number of ATAC features to select and pass to the model
        """
        os.chdir("..")
        multiome = sc.read_h5ad("data/multimodal/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")

        gex = multiome[:, multiome.var["feature_types"] == "GEX"] # Subset all data, not just the counts 
        sc.pp.highly_variable_genes(gex, n_top_genes=gex_dim, flavor="seurat_v3") # Feature selection

        atac = multiome[:, multiome.var["feature_types"] == "ATAC"] # Subset all data, not just the counts 
        esc.pp.select_var_feature(atac, nb_features=atac_dim, show=False) # Feature-selection - most variable features

        gex = gex[:, gex.var["highly_variable"]]

        gex_atac_concat = ann.concat([gex, atac], axis = 1)
        self.dataset = gex_atac_concat
        self.gex_dim = gex_dim
        self.atac_dim = atac_dim

class GexAdtPreprocess:
    def __init__(self, gex_dim) -> None:
        """
        gex_dim - Number of GEX features to select and pass to the model
        """
        os.chdir("..")
        cite = sc.read_h5ad("data/multimodal/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")

        gex = cite[:, cite.var["feature_types"] == "GEX"] # Subset all data, not just the counts 
        sc.pp.highly_variable_genes(gex, n_top_genes=gex_dim, flavor="seurat_v3") # Feature selection

        adt = cite[:, cite.var["feature_types"] == "ADT"] # We are not feature selecting for ADT

        gex = gex[:, gex.var["highly_variable"]]

        gex_adt_concat = ann.concat([gex, adt], axis = 1)
        self.dataset = gex_adt_concat
        self.gex_dim = gex_dim
        self.adt_dim = adt.shape[1]
