import pandas as pd
import scanpy as sc 
import anndata as ann
import episcanpy as esc

class GexAtacPreprocess:
    def __init__(self, absolute_path, gex_dim, atac_dim) -> None:
        """
        gex_dim - Number of GEX features to select and pass to the model
        atac_dim - Number of ATAC features to select and pass to the model
        absolute_path - absolute path to the directory with the GEX-ATAC dataset
        """
        print("Reading dataset...")
        multiome = sc.read_h5ad(absolute_path)
        self.obs= multiome.obs

        print("Feature selecting GEX...")
        gex = multiome[:, multiome.var["feature_types"] == "GEX"] # Subset all data, not just the counts 
        sc.pp.highly_variable_genes(gex, n_top_genes=gex_dim, flavor="seurat_v3") # Feature selection

        print("Feature selecting ATAC...")
        atac = multiome[:, multiome.var["feature_types"] == "ATAC"] # Subset all data, not just the counts 
        esc.pp.select_var_feature(atac, nb_features=atac_dim, show=False) # Feature-selection - most variable features

        gex_subset = gex[:, gex.var["highly_variable"]]


        gex_atac_concat = ann.concat([gex_subset, atac], axis = 1)
        gex_atac_concat.obs = self.obs
        self.dataset = gex_atac_concat

        self.gex_dim = gex_subset.shape[1]
        self.atac_dim = atac.shape[1]

        # Loggers
        print()
        print("New GEX dim: {};\nNew ATAC dim: {};".format(self.gex_dim, self.atac_dim))
        print("AnnData dataset's shape: {}".format(self.dataset.shape))
        print()

class GexAdtPreprocess:
    def __init__(self, absolute_path, gex_dim) -> None:
        """
        gex_dim - Number of GEX features to select and pass to the model
        absolute_path - absolute path to the directory with the GEX-ADT dataset
        """
        cite = sc.read_h5ad(absolute_path)
        self.obs= cite.obs

        gex = cite[:, cite.var["feature_types"] == "GEX"] # Subset all data, not just the counts 
        sc.pp.highly_variable_genes(gex, n_top_genes=gex_dim, flavor="seurat_v3") # Feature selection

        adt = cite[:, cite.var["feature_types"] == "ADT"] # We are not feature selecting for ADT

        gex_subset = gex[:, gex.var["highly_variable"]]

        gex_adt_concat = ann.concat([gex_subset, adt], axis = 1)
        gex_adt_concat.obs = self.obs
        self.dataset = gex_adt_concat

        self.gex_dim = gex_subset.shape[1]
        self.adt_dim = adt.shape[1]

        # Loggers
        print()
        print("New GEX dim: {};\nNew ADT dim: {};".format(self.gex_dim, self.adt_dim))
        print("AnnData dataset's shape: {}".format(self.dataset.shape))
        print()
