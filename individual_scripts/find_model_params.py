# =========================
# Imports
# =========================
import os
import sys
import numpy as np
from itertools import product
from types import SimpleNamespace
import hashlib
import json
import torch
sys.path.append('/mnt/md0/tempFolder/samAnderson/unet-gnn/functions/') 
from nn_optim_unet import gnn_builder, train_nn

# =========================
# Helper: deterministic config ID
# =========================
def config_to_id(cfg_dict):
    s = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]

# =========================
# Load data
# =========================
processed_data_path = '/mnt/md0/projects/graph_unet/atlas_projected_surfaces/processed/training/'
subj = np.load(f'{processed_data_path}subj_train_ico6.npy')
X = np.load(f'{processed_data_path}X_train_ico6.npy')
y = np.load(f'{processed_data_path}y_train_ico6.npy')

# =========================
# 80/20 split (by subject)
# =========================
if isinstance(subj[0], bytes): subj = np.array([s.decode() for s in subj])
# extract subject IDs (remove last "_date")
subject_ids = np.array([s.rsplit('_', 1)[0] for s in subj])
unique_subjects = np.unique(subject_ids)
# deterministic shuffle
rng = np.random.default_rng(seed=808)
rng.shuffle(unique_subjects)
# create the split
N_subj = len(unique_subjects)
train_end = int(0.8 * N_subj)
# get the indices
train_subj = unique_subjects[:train_end]
val_subj   = unique_subjects[train_end:]
# map subjects -> indices
train_idx = np.where(np.isin(subject_ids, train_subj))[0]
val_idx   = np.where(np.isin(subject_ids, val_subj))[0]
# final splits
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val     = X[val_idx], y[val_idx]
# sanity check (no overlap)
train_subjects = set(subject_ids[train_idx])
val_subjects   = set(subject_ids[val_idx])
assert train_subjects.isdisjoint(val_subjects)


print("\n--- Target Distribution ---")

print(y.shape)
print(f"FULL      | mean: {np.mean(y):.4f} | std: {np.std(y):.4f}")
print(f"TRAIN 80% | mean: {np.mean(y_train):.4f} | std: {np.std(y_train):.4f}")
print(f"VAL 20%   | mean: {np.mean(y_val):.4f} | std: {np.std(y_val):.4f}")

# =========================
# Base config
# =========================
training_config = SimpleNamespace(
    name='find_approx_params',
    output_path='/mnt/md0/tempFolder/samAnderson/unet-gnn/model_outputs/runs/',
    
    # data
    #batch_size=4, # 16 -> 8 -> 4 -> custom
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    n_vertices=81924,

    # early stoppage
    patience=20, # 20 -> 40 -> 20 -> 20
    min_delta=0.03,

    # LR scheduler (ReduceLROnPlateau)
    use_scheduler=True,
    scheduler_factor=0.2, # 0.1 -> 0.2 -> 0.2 -> 0.2
    scheduler_patience=4, # 3 -> 6 -> 4 -> 4
    scheduler_min_lr=1e-6 # 1e-6 -> 1e-6 -> 1e-6 -> 1e-6
)

# =========================
# Param grid
# =========================
param_grid = {
    'feature_sizes': [
        [256, 512, 512, 256, 256]
    ],
    'dropout_levels': [
        [0, 0, 0, 0, 0]
    ],
    'lr': [5e-4], # 1e-3 -> 1e-3 -> 5e-4 -> 5e-4
    'weight_decay': [3e-4], # 1e-4 -> 1e-4 -> 1e-4 -> 3e-4
}

# =========================
# Grid iteration
# =========================
keys = list(param_grid.keys())

for combo in product(*param_grid.values()):

    cfg_dict = dict(zip(keys, combo))

    # merge configs
    config = SimpleNamespace(**vars(training_config), **cfg_dict)
    
    # CUSTOM: update batch size based on feature_sizes
    if max(config.feature_sizes) == 512:
        config.batch_size = 8
    elif max(config.feature_sizes) == 1024:
        config.batch_size = 4

    # deterministic run folder
    cfg_id = config_to_id(cfg_dict)
    run_name = f"{config.name}_{cfg_id}"
    run_dir = os.path.join(config.output_path, run_name)

    os.makedirs(run_dir, exist_ok=True)

    # attach run_dir to config
    config.run_dir = run_dir

    # check if already completed
    ckpt_path = os.path.join(run_dir, 'last_epoch.pt')

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if ckpt['epochs_since_improve'] >= config.patience:
            print(f'Skipping completed run: {run_name}')
            continue
        else:
            print(f'Resuming run: {run_name}')
    else:
        print(f'Starting run: {run_name}')

    # build model
    model = gnn_builder(
        feature_sizes=config.feature_sizes,
        dropout_levels=config.dropout_levels,
    )

    # train
    train_nn(model, config) 