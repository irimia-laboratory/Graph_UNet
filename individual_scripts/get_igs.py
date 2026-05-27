import numpy as np
import torch
import sys
sys.path.append('/mnt/md0/tempFolder/samAnderson/unet-gnn/')
sys.path.append('/mnt/md0/tempFolder/samAnderson/unet-gnn/functions/')
from config import feature_config as cfg
from nn_optim_unet import gnn_builder, compute_igs

# Load model
model = gnn_builder(feature_sizes=cfg.feature_sizes, # from feature_config
                    dropout_levels=cfg.dropout_levels 
                    )
# Load the trained model
#model.load_state_dict(torch.load(cfg.model_weights))

# Compute and save IGs
grad_array = compute_igs(model, cfg)
np.save(f'{cfg.grad_array_path}igs_per_subject.npy', grad_array)

# Create grad dict
avg_grad = np.mean(grad_array, axis=0) # average per-feature
grad_dict = {}
for idx, feature in enumerate(cfg.features):
    grad_dict[feature] = avg_grad[:, idx]

# Save grad dict
np.save(f'{cfg.grad_array_path}mean_ig_dict.npy', grad_dict)

# NOTE: we don't end up using this because it's so slow to run