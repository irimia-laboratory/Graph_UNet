import os
import torch
import numpy as np
from types import SimpleNamespace
import sys
sys.path.append('/mnt/md0/tempFolder/samAnderson/unet-gnn/')
sys.path.append('/mnt/md0/tempFolder/samAnderson/unet-gnn/functions/')
from config import feature_config as cfg
from nn_optim_unet import gnn_builder, compute_ablation
from postprocessing import PostProcessor

# Create the postprocessing object
p = PostProcessor(first=cfg.first)

# Load the trained model
model = gnn_builder(feature_sizes=cfg.feature_sizes, # from feature_config
                    dropout_levels=cfg.dropout_levels
                    )
model.load_state_dict(torch.load(cfg.model_weights))

# Load in correction factors
CN_factors = np.load(cfg.CN_factors)

# Loop over cohorts
for cohort_name, cohort_cfg in cfg.cohort_dict.items():

    print(f'\nProcessing {cohort_name}')

    # Make the ablation folder paths
    os.makedirs(cohort_cfg.array_path, exist_ok=True)
    os.makedirs(cohort_cfg.vis_path, exist_ok=True)

    # Create cumulative dict
    ablation_dict = {}  

    # Get cohort-specific config and compute ablation
    cohort_fused_cfg = SimpleNamespace(**vars(cohort_cfg), **vars(cfg))
    pred_dict, targets_array = compute_ablation(model, cohort_fused_cfg)

    # Get baseline LBAGs
    # Clip + smooth
    raw_lbas, _, _ = p.clip_and_smooth(
        pred_dict['baseline'],
        targets_array,
        medial_present=True
    )

    # Bias correction
    base_bc_lbags, _, _, _ = p.bias_correct(
        y_pred=raw_lbas,
        y_true=targets_array,
        factors=CN_factors
    )

    # Remove baseline from dict
    pred_dict.pop('baseline')

    # Loop over features
    for feature in cfg.features:

        # Clip + smooth
        raw_lbas, _, _ = p.clip_and_smooth(
            pred_dict[feature],
            targets_array,
            medial_present=True
        )

        # Bias correction
        bc_lbags, _, _, _ = p.bias_correct(
            y_pred=raw_lbas,
            y_true=targets_array,
            factors=CN_factors
        )

        # Save bias-corrected mean LBAG differences to dict
        lbag_diff_map = np.mean((base_bc_lbags - bc_lbags), axis=0)
        ablation_dict[feature] = lbag_diff_map  # (nodes)

        # Create a visualization for the feature
        p.generate_cortical_plot(lbag_diff_map, 
            save_to=f'{cohort_cfg.vis_path}ablation_diff_{feature}', 
            medial_present=False
            );

    # Save cohort-level output
    np.save(f'{cohort_cfg.array_path}ablation_dict.npy', ablation_dict)