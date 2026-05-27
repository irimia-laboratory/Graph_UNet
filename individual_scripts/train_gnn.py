import os
import torch
import sys
sys.path.append('/mnt/md0/tempFolder/samAnderson/unet-gnn/') 
sys.path.append('/mnt/md0/tempFolder/samAnderson/unet-gnn/functions/') 
from config import precursor_config as cfg
from nn_optim_unet import gnn_builder, train_nn

# Create the model
model = gnn_builder(
    feature_sizes=cfg.feature_sizes, # from precursor_config
    dropout_levels=cfg.dropout_levels
)

# Train the model
os.makedirs(os.path.dirname(cfg.model_weights), exist_ok=True)
trained_model = train_nn(model, cfg, robust=False) 

# Save the trained model
torch.save(
    trained_model.state_dict(),
    cfg.model_weights
)