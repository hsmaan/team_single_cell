import sys
sys.path.append("../..")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import wandb

from helpers.preprocessing import GexAtacPreprocess
from train import GexAtacTrainer
from evaluate import GexAtacEvaluation

hyperparameter_defaults = dict(
    gex_dim = 2500,
    atac_dim = 7000,
    latent_dim = 20,
    model = [[1000, 800, 600, 400, 200],
      [10, 30, 50, 100, 160],
      [1200, 900, 600, 400, 150],
      [15, 40, 60, 100, 180]],
    init = "he",
    lr = 0.001,
    weight_decay = 0.001,
    epochs = 20,
    gex_weight = 5,
    atac_weight = 5,
)

run = wandb.init(
    entity="team-single-cell",
    config=hyperparameter_defaults,
    project="gex_atac_sweep",
    reinit=True,
)
wargs = wandb.config

# Hyperparameters
gex_dim = wargs.gex_dim
atac_dim = wargs.atac_dim
latent_dim = wargs.latent_dim
model = wargs.model
init = wargs.init
lr = wargs.lr
weight_decay = wargs.weight_decay
epochs = wargs.epochs
gex_weight = wargs.gex_weight
atac_weight = wargs.atac_weight

gex_atac_preprocess = GexAtacPreprocess("../../data/multimodal/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad",
    gex_dim=gex_dim, atac_dim=atac_dim
)


gex_atac_trainer = GexAtacTrainer(
    gex_atac_preprocess, latent_dim, model=model, init=init,
    lr=lr, weight_decay=weight_decay
)

gex_atac_trainer.train(epochs=epochs, gex_loss_w=gex_weight, atac_loss_w=atac_weight, wandb_log=True)

gex_atac_eval = GexAtacEvaluation(
    gex_atac_trainer, gex_atac_preprocess
)

total_score, res_df = gex_atac_eval.evaluate()

wandb_resdf = wandb.Table(dataframe=res_df)
wandb.log({"total_score": total_score, "res_df": wandb_resdf})