# Import relevant packages
import os
import torch
import csv
from glob import glob
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.data import Data as Data_pyg
from torch_geometric.loader import DataLoader as DataLoader_pyg
from captum.attr import IntegratedGradients
import numpy as np
import random
import time

### Set torch configs ###
SEED = 808
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#device = torch.device("cpu")
###

### Set seed for reproduceability ###
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)
set_seed(SEED)

# Function for building the GNN dynamically
def gnn_builder(
        feature_sizes = [8, 16, 16, 8, 8], 
        dropout_levels = [0, 0, 0, 0, 0],
        pooling_path = '/mnt/md0/tempFolder/samAnderson/unet-gnn/pooling/', 
        ico_levels = [6, 5, 4]
        ):
    
    # Confirm that feature sizes are matched appropriately
    assert feature_sizes[0] == feature_sizes[3] and feature_sizes[1] == feature_sizes[2], \
        f"Invalid fs: requires c1=c4 and c2=c3, got {feature_sizes}"
    
    # Set seed for reproducability
    set_seed(SEED) 

    # Load in the downsampling indices associating receptive fields across ico levels
    indice_paths = glob(f'{pooling_path}*')
    downsample_indices = {}
    for path in indice_paths:
        target_ico = path[path.find("->")-1]
        if 'downsample' in path:
            # load the np array and convert to tensor, then move to GPU
            downsample_indices[int(target_ico)] = torch.from_numpy(np.load(path)).to(device)

    # Get the edge indices for each ico level
    edge_indices = {}
    for ico in ico_levels:
        # Start by getting the faces
        if ico == 7:
            fsavg_path = '/mnt/md0/softwares/freesurfer/subjects/fsaverage/'
        else:
            fsavg_path = f'/mnt/md0/softwares/freesurfer/subjects/fsaverage{ico}/'
        _, faces = nib.freesurfer.read_geometry(f'{fsavg_path}surf/rh.pial')

        # Stack the faces to account for both hemispheres (same for each hemi)
        faces_both_hemi = np.vstack((faces, faces + (np.max(faces) + 1)))
        # Derive edges from the faces
        edges = np.vstack([
            faces_both_hemi[:, [0, 1]],  # edge 1: v1, v2
            faces_both_hemi[:, [1, 2]],  # edge 2: v2, v3
            faces_both_hemi[:, [2, 0]],  # edge 3: v3, v1
            faces_both_hemi[:, [1, 0]],  # reverse of edge 1
            faces_both_hemi[:, [2, 1]],  # reverse of edge 2
            faces_both_hemi[:, [0, 2]]   # reverse of edge 3
        ])
        # Sort the edges, remove duplicates, and transpose
        sorted_edges = np.sort(edges, axis=1)
        unique_edges = np.unique(sorted_edges, axis=0)
        # Convert to tensor and move to GPU
        edge_indices[ico] = torch.tensor(unique_edges.T, dtype=torch.long, device=device)
        
    # Create the dict holding the upsample indices
    upsampling_toolkit = {}
    for ico in ico_levels[1:]:  # Skip the largest ico
        # Create tensors relating across receptive fields in reverse (i.e. from lower to higher)
        # Since there are only at most two receptive fields that a higher-ico vertex is a part of, we only need two arrays
        first_coor = torch.full((torch.max(downsample_indices[ico + 1]) + 1, 1), -1, dtype=torch.long, device=device)
        second_coor = torch.full((torch.max(downsample_indices[ico + 1]) + 1, 1), -1, dtype=torch.long, device=device)
        # Loop through the rows of the downsampled indices and get the upsample indices
        # Note that rows can also be thought of as receptive fields
        for row_idx, row in enumerate(downsample_indices[ico + 1]):  # Lower ico nodes
            for indice in row:  # Higher ico nodes
                # Set the index, as contained in the tensor, to the receptive field, as conveyed by the row index
                # so if '100' is in the 3rd row you set index 100 = 3
                if first_coor[indice.item()] == -1:
                    first_coor[indice.item()] = row_idx
                else:
                    second_coor[indice.item()] = row_idx  # Some nodes are represented twice in the array, because they are part of 2 receptive fields

        # Save the relevant list of tensors to the ico dict entry
        upsampling_toolkit[ico] = [first_coor.squeeze(), second_coor.squeeze()]
        
    # Define the gnn model
    class gnn_model(torch.nn.Module):
        def __init__(self):
            super().__init__()

            # Save the relevant indices
            self.ico_levels = ico_levels
            #
            self.edge_indices = edge_indices
            self.downsample_indices = downsample_indices
            self.upsampling_toolkit = upsampling_toolkit
            #
            self.backups_saved = False # note that we haven't saved backups for batch processing
            #
            self.batch_processed = True # whether or not the arrays have been modified for batch size
            self.last_batch_size = 1 # helps to account for changes in batch size, relevant for preprocessing; defaults to 1                
            #
            c1, c2, c3, c4, c5 = feature_sizes

            ### first block ###
            self.gcn1 = GCNConv(5, c1, cached=True)
            self.bn1 = BatchNorm(c1)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(p=dropout_levels[0])

            ### second block ###
            self.gcn2 = GCNConv(c1, c2, cached=True)
            self.bn2 = BatchNorm(c2)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(p=dropout_levels[1])

            ### third block ###
            self.gcn3 = GCNConv(c2, c3, cached=True)
            self.bn3 = BatchNorm(c3)
            self.relu3 = nn.ReLU()
            self.dropout3 = nn.Dropout(p=dropout_levels[2])

            ### fourth block ###
            # concat: c3 (upsampled) + c2 (skip) → c3 + c2
            self.gcn4 = GCNConv(c3 + c2, c4, cached=True)
            self.bn4 = BatchNorm(c4)
            self.relu4 = nn.ReLU()
            self.dropout4 = nn.Dropout(p=dropout_levels[3])

            ### fifth block ###
            # concat: c4 (upsampled) + c1 (skip) → c4 + c1
            self.gcn5 = GCNConv(c4 + c1, c5, cached=True)
            self.bn5 = BatchNorm(c5)
            self.relu5 = nn.ReLU()
            self.dropout5 = nn.Dropout(p=dropout_levels[4])

            ### sixth block ###
            self.gcn6 = GCNConv(c5, 1)

            # save gcn layers in a list for clearing cache later
            self.gcn_layers = [self.gcn1, self.gcn2, self.gcn3, self.gcn4, self.gcn5, self.gcn6]
        
        def downsample_block(self, x_in, ico, weights='mean'):
            # Define the lower ico with respect to its higher-ico receptive field
            x_out = x_in[self.downsample_indices[ico]]
            if weights == 'mean': # Mean pool; avg by receptive field
                x_out = torch.mean(x_out, dim=1)
            elif weights == 'max': # Max pool; max of receptive field
                x_out, _ = torch.max(x_out, dim=1)
            else: # Learnable attn weights
                x_out = torch.sum(F.softmax(weights, dim=1).repeat(self.last_batch_size, 1, x_out.shape[2]) * x_out, dim=1) # softmax(weights) -> attn weights -> learnable weighted sum
            return x_out

        def upsample_block(self, x_in, ico):
            # Get the precomputed coordinates
            first_coor, second_coor = self.upsampling_toolkit[ico]  # shapes: (n_vertices,)

            # Initialize the output tensor
            x_out = torch.zeros((torch.max(self.edge_indices[ico+1]) + 1, x_in.shape[1]), device=device)

            # Use first_coor to identify targets
            x_out[:] = x_in[first_coor]

            # Mask for valid indices in second_coor, add these to the output, then average by the number of receptive fields
            valid_second_coor = second_coor >= 0
            x_out[valid_second_coor] += x_in[second_coor[valid_second_coor]] # combine the values derived from the coordinates
            x_out[valid_second_coor] /= 2 # average the values if two
            return x_out
        
        def batch_process(self, n_batches):

            if not self.backups_saved:
                # Save backups for the indices, so that when batch processing you have them; on the GPU
                self.backups = [
                    {k: v.clone().to(device) for k, v in self.edge_indices.items()},
                    {k: v.clone().to(device) for k, v in self.downsample_indices.items()},
                    {k: [v[0].clone().to(device), v[1].clone().to(device)] for k, v in self.upsampling_toolkit.items()}
                ]
                # note that we have backups saved and don't need to save them again
                self.backups_saved = True

            # Use backups so we aren't influenced by prior preprocessing
            self.edge_indices = {k: v.clone().to(device) for k, v in self.backups[0].items()}
            self.downsample_indices = {k: v.clone().to(device) for k, v in self.backups[1].items()}
            self.upsampling_toolkit = {k: [v[0].clone().to(device), v[1].clone().to(device)] for k, v in self.backups[2].items()}

            # Format into dict for batch processing
            absolute_indices = {
                'edge_indices': self.edge_indices,
                'downsample_indices': self.downsample_indices,
                'upsampling_toolkit': self.upsampling_toolkit
            }

            # Extend indices with respect to batch size
            for indice_type, indices_dict in absolute_indices.items():
                if indice_type == 'upsampling_toolkit':
                    # Handle upsampling indices separately
                    for ico, (first_coor, second_coor) in indices_dict.items():
                        # Extend the coordinates with respect to batch
                        shift = torch.max(first_coor) + 1
                        extended_first_coor = [first_coor]
                        extended_second_coor = [second_coor]
                        # Extend both first_coor and second_coor in a single loop
                        for i in range(1, n_batches):
                            extended_first_coor.append(first_coor + (shift * i))
                            shifted_second_coor = second_coor.clone()
                            shifted_second_coor[shifted_second_coor != -1] += (shift * i)
                            extended_second_coor.append(shifted_second_coor)
                        # Concatenate the extended coordinates
                        extended_first_coor = torch.cat(extended_first_coor, dim=0).long()
                        extended_second_coor = torch.cat(extended_second_coor, dim=0).long()
                        # Update the dictionary
                        indices_dict[ico] = [extended_first_coor, extended_second_coor]
                else:
                    # Handle edge_indices and downsample_indices
                    for ico, indices in indices_dict.items():
                        shift = torch.max(indices) + 1
                        extended_indices = [indices]
                        for i in range(1, n_batches):
                            extended_indices.append(indices + (shift * i))
                        extended_indices = torch.cat(extended_indices, dim=1) if indice_type == 'edge_indices' else torch.cat(extended_indices, dim=0)
                        # Update the dictionary
                        indices_dict[ico] = extended_indices.long()

            # Update the indices
            self.edge_indices = absolute_indices['edge_indices']
            self.downsample_indices = absolute_indices['downsample_indices']
            self.upsampling_toolkit = absolute_indices['upsampling_toolkit']

            return 

        def forward(self, gnn_data):

            x, batch_size = gnn_data.x, gnn_data.num_graphs
            #start_time = time.time()

            # Check if batch preprocessing needs to be performed
            try: assert (self.last_batch_size == batch_size)
            except AssertionError:
                self.batch_processed = False

            if not self.batch_processed: # defaults to True for batch size 1
                # Extend the indices with respect to the batch size
                self.batch_process(batch_size)
                # Clear the cache for the edge indices
                for layer in self.gcn_layers: layer._cached_edge_index = None
                # Update the batch processing status
                self.batch_processed = True
                # Update the batch sizes
                self.last_batch_size = batch_size

            # Encoding phase

            ### First block ###
            gnn_x_6 = self.gcn1(x, self.edge_indices[self.ico_levels[0]]) # 5 -> c1
            gnn_x_6 = self.bn1(gnn_x_6)
            gnn_x_6 = self.relu1(gnn_x_6)
            gnn_x_6 = self.dropout1(gnn_x_6)

            ### Second block ###
            gnn_x_5 = self.downsample_block(gnn_x_6, ico=self.ico_levels[0]) # ico 6 -> ico 5
            gnn_x_5 = self.gcn2(gnn_x_5, self.edge_indices[self.ico_levels[1]])  # c1 -> c2
            gnn_x_5 = self.bn2(gnn_x_5)
            gnn_x_5 = self.relu2(gnn_x_5)
            gnn_x_5 = self.dropout2(gnn_x_5)

            # Embedding phase 

            ### Third block ###
            gnn_x_4 = self.downsample_block(gnn_x_5, ico=self.ico_levels[1]) # ico 5 -> ico 4
            gnn_x_4 = self.gcn3(gnn_x_4, self.edge_indices[self.ico_levels[2]]) # c2 -> c3
            gnn_x_4 = self.bn3(gnn_x_4)
            gnn_x_4 = self.relu3(gnn_x_4)
            gnn_x_4 = self.dropout3(gnn_x_4)

            # Decoding phase 

            ### Fourth block ###
            gnn_x1_5 = self.upsample_block(gnn_x_4, ico=self.ico_levels[2]) # ico 4 -> ico 5
            gnn_x1_5 = torch.cat((gnn_x1_5, gnn_x_5), dim=1) # skip connection; (c3 + c2)
            #
            gnn_x1_5 = self.gcn4(gnn_x1_5, self.edge_indices[ico_levels[1]])  # (c3 + c2) -> c4
            gnn_x1_5 = self.bn4(gnn_x1_5)
            gnn_x1_5 = self.relu4(gnn_x1_5)
            gnn_x1_5 = self.dropout4(gnn_x1_5)

            ### Fifth block ###
            gnn_x1_6 = self.upsample_block(gnn_x1_5, ico=ico_levels[1]) # ico 5 -> ico 6
            gnn_x1_6 = torch.cat((gnn_x1_6, gnn_x_6), dim=1) # skip connection; (c4 + c1)
            #
            gnn_x1_6 = self.gcn5(gnn_x1_6, self.edge_indices[ico_levels[0]]) # (c4 + c1) -> c5
            gnn_x1_6 = self.bn5(gnn_x1_6)
            gnn_x1_6 = self.relu5(gnn_x1_6)
            gnn_x1_6 = self.dropout5(gnn_x1_6)

            ### Sixth block ###
            gnn_x1_6 = self.gcn6(gnn_x1_6, self.edge_indices[ico_levels[0]]) # c5 -> 1

            #end_time = time.time()
            #print(f"Elapsed time: {end_time - start_time} seconds")
            return gnn_x1_6.squeeze(-1)
        
    built_nn = gnn_model()
    return built_nn

# Helper function: for loading in datasets
def get_loader(X, y, batch_size, shuffle):
    
    # Account for path, numpy, or torch
    def fix_type(var):
        if isinstance(var, str):
            var = np.load(var)
        if isinstance(var, np.ndarray):
            var = torch.from_numpy(var.astype(np.float32))
        return var
            
    X = fix_type(X)
    y = fix_type(y)
    
    # Create the DataLoader
    built_data = [Data_pyg(x=X[i], y=y[i]) for i in range(len(X))]
    built_loader = DataLoader_pyg(built_data, batch_size=batch_size, shuffle=shuffle)
    
    return built_loader

# Helper function: for running a single epoch
def run_epoch(model, loader, criterion, n_vertices, device, optimizer=None, return_preds=False):
    
    is_train = optimizer is not None
    if is_train: model.train()
    else: model.eval()

    epoch_loss, total_samples = 0.0, 0

    # Only allocate if needed
    if return_preds:
        preds_all, targets_all = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)

            if is_train:
                optimizer.zero_grad()

            output = model(batch)

            target = torch.repeat_interleave(batch.y, repeats=n_vertices)
            loss = criterion(output.view(-1), target.view(-1))

            if is_train:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs

            if return_preds:
                preds_all.append(output.view(batch.num_graphs, n_vertices).detach().cpu()) # (scan, vertices)
                targets_all.append(batch.y.detach().cpu()) # CA

    epoch_loss /= total_samples

    if return_preds:
        preds_all = torch.cat(preds_all)
        targets_all = torch.cat(targets_all)
        return epoch_loss, preds_all, targets_all

    return epoch_loss

# Function for training a graph neural network for local brain age
def train_nn(model, config, robust=True):

    # Set the seed
    set_seed(SEED)

    # Send the model to cuda and prepare to train
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    criterion = nn.L1Loss()

    # Learning rate scheduler reduces LR when validation loss plateaus
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr
        )
    else:
        scheduler = None

    # Data
    train_loader = get_loader(config.X_train, config.y_train, config.batch_size, shuffle=True)
    if robust:
        val_loader = get_loader(config.X_val, config.y_val, config.batch_size, shuffle=False)
        
    # Train based on val loss
    if robust:

        # Set run folder
        run_folder = config.run_dir
        log_path = os.path.join(run_folder, 'log.csv')
        weight_path = os.path.join(run_folder, 'last_epoch.pt')

        # Create CSV if it doesn't exist
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'val_loss',
                    'lr', 'weight_decay',
                    'dropout_levels', 'feature_sizes'
                ])

        # Resume logic
        best_loss = float('inf')
        epochs_since_improve = 0
        ckpt_epoch = 0

        if os.path.exists(weight_path):
            ckpt = torch.load(weight_path, map_location='cpu')

            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])

            # Restore scheduler state so LR progression continues correctly
            if scheduler is not None and ckpt.get('scheduler_state') is not None:
                scheduler.load_state_dict(ckpt['scheduler_state'])

            best_loss = ckpt.get('best_loss', float('inf'))
            epochs_since_improve = ckpt.get('epochs_since_improve', 0)
            ckpt_epoch = ckpt.get('epoch', 0)

            start_epoch = ckpt_epoch + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            start_epoch = 1

        # Training loop
        i = start_epoch
        while True:

            train_loss = run_epoch(model, train_loader, criterion, config.n_vertices, device, optimizer)
            val_loss = run_epoch(model, val_loader, criterion, config.n_vertices, device)

            # Update learning rate based on validation performance
            prev_lr = optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr < prev_lr:
                print(f"LR reduced: {prev_lr:.2e} → {new_lr:.2e}")

            print(f"Epoch {i} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # Early stop update
            if val_loss < best_loss - config.min_delta:
                best_loss = val_loss
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            # Save checkpoint (overwrite)
            torch.save({
                'epoch': i,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                'best_loss': best_loss,
                'epochs_since_improve': epochs_since_improve
            }, weight_path)

            # Log
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    i, train_loss, val_loss,
                    optimizer.param_groups[0]['lr'],
                    config.weight_decay,
                    str(config.dropout_levels),
                    str(config.feature_sizes)
                ])

            # Early stop
            if epochs_since_improve >= config.patience:
                print(f"Early stopping at epoch {i}")
                break

            i += 1

        return model

    # If simply training
    else:

        # train for specified number of epochs, no early stoppage, etc.
        for i in range(1, config.num_epochs + 1):

            train_loss = run_epoch(model, train_loader, criterion, config.n_vertices, device, optimizer)
            print(f"Epoch {i} | Train: {train_loss:.4f}")

        return model

# Function for testing a graph neural network for local brain age
def test_nn(model, config):

    # Set the seed
    set_seed(SEED)

    # Send model to device
    model = model.to(device)

    # Loss + data
    criterion = nn.L1Loss()
    loader = get_loader(config.X_test, config.y_test, config.batch_size, shuffle=False)

    # Test
    test_loss, preds_all, targets_all = run_epoch(
        model, loader, criterion, 
        config.n_vertices, device, return_preds=True
    ) # eval status is implicit through run_epoch

    return {
        'test_loss': test_loss,
        'predictions': preds_all.numpy(),
        'targets': targets_all.numpy()
    }

# Function for computing integrated gradients
# VERY SLOW. Not practical for us to implement, but it works?
def compute_igs(model, config):
    
    # Set the seed
    set_seed(SEED)
    
    # Send the model to device and eval mode
    model = model.to(device)
    model.eval()

    # Build PYG dataset (batch size = 1)
    loader = get_loader(config.X, config.y, 1, shuffle=False)

    # Wrapper for Captum
    def model_forward(x, batch, node_idx):
        new_batch = batch.clone()
        new_batch.x = x

        out = model(new_batch)

        # Ensure shape is (num_nodes,)
        if out.dim() > 1:
            out = out.squeeze(-1)

        # Return scalar with batch dimension
        return out[node_idx].unsqueeze(0)

    ig = IntegratedGradients(model_forward)

    # Run IG for one graph
    def run_IG(batch, baseline):

        # Ensure gradients
        batch.x.requires_grad_(True)

        node_attributions = []

        # Get attributions PER-NODE to identify LOCAL significance
        for node_idx in range(batch.x.shape[0]):
            attributions = ig.attribute(
                inputs=batch.x,
                baselines=baseline,
                additional_forward_args=(batch, node_idx),
                n_steps=config.n_steps,
                internal_batch_size=config.n_steps # compute every step at once
            )

            # Keep attribution for this node only
            node_attributions.append(
                attributions[node_idx].detach().cpu()
            )

        # Stack; (num_nodes, num_features)
        return torch.stack(node_attributions)

    # Store results
    all_attributions = []

    # Baseline handling
    if getattr(config, 'set_baseline', None) is not None:

        set_baseline = torch.as_tensor(
            config.set_baseline,
            dtype=torch.float32,
            device=device
        )

        for i, batch in enumerate(loader):

            # Progress update (every 100 subjects, plus first and last)
            if i % 20 == 0 or i == len(loader) - 1:
                print(f"Processing subject {i+1}/{len(loader)}")

            batch = batch.to(device)

            # Sample baseline per node (GPU-safe)
            idx = torch.randint(
                0, set_baseline.shape[0],
                (batch.x.shape[0],),
                device=device
            )
            baseline = set_baseline[idx]

            # Sanity check
            assert baseline.shape[1] == batch.x.shape[1], \
                'Baseline feature dimension mismatch'

            node_attr = run_IG(batch, baseline)
            all_attributions.append(node_attr)

    else: # if baseline not provided: what we actually use
        for i, batch in enumerate(loader):

            # Progress update (every 100 subjects, plus first and last)
            if i % 100 == 0 or i == len(loader) - 1:
                print(f"Processing subject {i+1}/{len(loader)}")

            batch = batch.to(device)

            baseline = torch.zeros_like(batch.x) # for z-scored distributions

            node_attr = run_IG(batch, baseline)
            all_attributions.append(node_attr)

    # Concatenate and reshape
    all_attributions = torch.cat(all_attributions, dim=0)
    n_subjects = len(loader.dataset)
    n_features = all_attributions.shape[1]

    assert all_attributions.shape[0] == n_subjects * config.n_vertices, \
        'Mismatch between attributions and expected (subjects * vertices)'

    all_attributions = all_attributions.view(
        n_subjects,
        config.n_vertices,
        n_features
    )

    return all_attributions

# Get ablation results (test set) to identify feature influence
@torch.no_grad()
def compute_ablation(model, config):
    
    # Prepare model
    model = model.to(device)
    model.eval()

    # Data loader
    loader = get_loader(config.X_test, config.y_test, batch_size=1, shuffle=False)

    # Store raw predictions for later bias correction
    # Dict: feature -> list of (nodes,)
    pred_dict = {feature: [] for feature in config.features}
    pred_dict['baseline'] = []  # baseline (no ablation)

    all_targets = []

    for batch in loader:
        batch = batch.to(device)

        # Get baseline output (predicted local brain age)
        pred_full = model(batch)  # (num_nodes,)
        pred_dict['baseline'].append(pred_full.cpu())

        # Loop over features
        for f, feature_name in enumerate(config.features):

            # Zero-out feature f (z-scored baseline)
            x_ablated = batch.x.clone()
            x_ablated[:, f] = 0.0

            # Create new batch with ablated feature
            new_batch = batch.clone()
            new_batch.x = x_ablated

            # Get prediction after ablation
            pred_ablated = model(new_batch)

            pred_dict[feature_name].append(pred_ablated.cpu())

        # Store target (chronological age)
        # shape: scalar -> expand later when needed
        all_targets.append(batch.y.cpu())

    # Stack predictions per feature
    # Result: feature -> (subjects, nodes)
    for key in pred_dict:
        pred_dict[key] = torch.stack(pred_dict[key]).numpy()

    # Stack targets
    # Shape: (subjects,)
    targets_array = torch.stack(all_targets).numpy().squeeze()

    # Gives raw predictions
    return pred_dict, targets_array