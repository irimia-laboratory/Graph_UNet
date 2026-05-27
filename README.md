## Repository Overview

This repository contains most of the code used for the manuscript:

> **Graph Neural Network Reveals the Cortical Morphology of Local Brain Aging in Normal Cognition and Alzheimer’s Disease**  
> https://arxiv.org/abs/2601.10912

---

## Repository Structure

### [`./pooling`](./pooling)

Contains the indices used when downsampling or upsampling across ico levels. These indices are computed using the surface-projected atlas meshes (i.e., Euclidean space).

The script used to generate these files is located in:

- [`./individual_scripts`](./individual_scripts)

---

### [`./notebooks`](./notebooks)

Contains the Jupyter notebooks used to generate most experimental results and analyses presented in the manuscript.

Each notebook uses its own configuration namespace defined in [`./config.py`](./config.py). This file:

- defines globally preserved variables (e.g., the number of vertices in the starting ico level)
- propagates shared parameters to notebook-level configs to ensure consistency across processing
- includes small utility functions for:
  - plotting figures
  - cleaning dataframes
  - repeated helper operations

These functions were kept in a single file for convenience rather than split into separate utility modules.

---

### [`./individual_scripts`](./individual_scripts)

Contains one-off scripts used for tasks that were inefficient or unstable to run directly in Jupyter notebooks. These include:

- model training
- ablation experiments
- preprocessing utilities
- other standalone procedures

Scripts were typically executed as:

```bash
python [script_name].py
```

These scripts also reference the namespaces defined in [`./config.py`](./config.py) to maintain consistency with notebook experiments.

For example:

- the ablation notebook references `get_ablation.py`
- correspondingly, `ablation.py` draws from the notebook namespace for consistent parameterization

---

### [`./supplementary_tables`](./supplementary_tables)

Contains the complete supplementary tables associated with the manuscript.

These files correspond to the full supplementary materials referenced throughout the paper.

---

### [`./trained_weights.pth`](./trained_weights.pth)

Contains the trained model weights used in the manuscript experiments.

These weights are intended to be loaded using the provided configuration setup defined in [`./config.py`](./config.py) and the associated training/inference scripts.

## Environment Recreation

The following files provide partial support for recreating the original environment:

- [`./base_env_pip.txt`](./base_env_pip.txt)
- [`./base_env_conda.yml`](./base_env_conda.yml)

While these files may not contain every dependency, they include the critical PyTorch Geometric and related package versions, which are often difficult to configure correctly.

System details used during development:

- CUDA: `12.6`
- Ubuntu: `22.04`

---

## Excluded Files and Directories

Several folders and files referenced throughout the codebase are intentionally excluded from this repository.

### Excluded Directories

#### `./visualization_code`

Contains utilities for generating cortical surface visualizations from array-based outputs.

This codebase was developed separately from the manuscript and is therefore not included here.

---

#### `./model_outputs`

Contains raw model outputs.

These files include subject-level data and are excluded for privacy and data-sharing reasons.

---

#### `./figures`

Contains generated manuscript figures.

All figures are already available in the manuscript, and the raw files were too large to upload.

---

#### `./env_gnn`

Contains the raw development environment.

This environment is only useful within the context of the original system configuration. Recommended package versions are instead provided through the environment specification files.

---

### Excluded Data Files

#### `./GNN_regionalAGs.csv`

Contains subject-level data generated from the GNN pipeline.

---

#### `./ADNI_Complete_regionalAGs.csv`

Contains subject-level data generated from the CNN pipeline.

---

### Additional Exclusions

Any raw or processed subject-level data files are excluded from this repository.
