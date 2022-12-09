import torch.nn as nn

def gex_atac_loss(output, target, gex_dim, atac_dim, gex_weight=10, atac_weight=1):
    # GEX loss 
    mse = nn.MSELoss()
    gex_true = target[:, 0:gex_dim]
    gex_pred = output[:, 0:gex_dim]
    gex_loss = mse(gex_pred, gex_true)
    
    # ATAC loss 
    bce = nn.BCELoss()
    atac_true = target[:, gex_dim:]
    atac_pred = output[:, gex_dim:]
    atac_loss = bce(atac_pred, atac_true)
    
    # Combine both and return
    loss = gex_loss*gex_weight + atac_loss*atac_weight
    return loss 


def gex_adt_loss(output, target, gex_dim, adt_dim, gex_weight=5, adt_weight=5):
    # GEX loss 
    gex_mse = nn.MSELoss()
    gex_true = target[:, 0:gex_dim]
    gex_pred = output[:, 0:gex_dim]
    gex_loss = gex_mse(gex_pred, gex_true)
    
    # ADT loss 
    adt_mse = nn.MSELoss()
    adt_true = target[:, gex_dim:]
    adt_pred = output[:, gex_dim:]
    adt_loss = adt_mse(adt_pred, adt_true)
    
    # Combine both and return
    loss = gex_loss*gex_weight + adt_loss*adt_weight
    return loss 
