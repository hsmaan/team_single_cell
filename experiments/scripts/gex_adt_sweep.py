import os
os.chdir("../..")

import wandb

from helpers.preprocessing import GexAdtPreprocess
from train import GexAdtTrainer
from evaluate import GexAdtEvaluation

hyperparameter_defaults = dict(
    gex_dim = 2500,
    latent_dim = 20,
    model = [[1600, 1200, 800, 400, 200, 100, 50],
    [80, 160, 240, 640, 960, 1200],
    [2400, 1200, 800, 400, 200, 100, 50],
    [80, 160, 240, 640, 960, 1200, 1600]],
    init = "he",
    lr = 0.001,
    weight_decay = 0.001,
    epochs = 20,
    gex_weight = 5,
    adt_weight = 5,
)

run = wandb.init(
    entity="team-single-cell",
    config=hyperparameter_defaults,
    project="test-project",
    reinit=True,
)
wargs = wandb.config

# Hyperparameters
gex_dim = wargs.gex_dim
latent_dim = wargs.latent_dim
model = wargs.model
init = wargs.init
lr = wargs.lr
weight_decay = wargs.weight_decay
epochs = wargs.epochs
gex_weight = wargs.gex_weight
adt_weight = wargs.adt_weight

gex_adt_preprocess = GexAdtPreprocess("data/multimodal/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad",
    gex_dim=gex_dim
)


gex_adt_trainer = GexAdtTrainer(
    gex_adt_preprocess, latent_dim, model=model, init=init,
    lr=lr, weight_decay=weight_decay
)

gex_adt_trainer.train(epochs=epochs, gex_loss_w=gex_weight, adt_loss_w=adt_weight, wandb_log=True)

gex_adt_eval = GexAdtEvaluation(
    gex_adt_trainer, gex_adt_preprocess
)

total_score, res_df = gex_adt_eval.evaluate()

wandb.log({"total_score": total_score, "res_df": res_df})