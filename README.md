## Repository Overview

This repository contains most of the code used for the manuscript:

> **Graph Neural Network Reveals the Cortical Morphology of Local Brain Aging in Normal Cognition and Alzheimer’s Disease**  
> https://arxiv.org/abs/2601.10912

---

## Repository Structure

### [`./pooling`](./pooling)

Contains the indices used when downsampling or upsampling across ico levels. These indices are computed using the surface-projected atlas meshes (i.e., Euclidean space).

The script used to generate these files is:

- [`./individual_scripts/get_pooling.py`](./individual_scripts)

---

### [`./notebooks`](./notebooks)

Contains the Jupyter notebooks used to generate most experimental results and analyses presented in the manuscript.

Each notebook uses its own configuration namespace defined in [`./config.py`](./config.py). This file:

- Defines globally preserved variables (e.g., the number of vertices in the starting ico level)
- Propagates shared parameters to notebook-level configs to ensure consistency across processing
- Includes small utility functions for:
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
