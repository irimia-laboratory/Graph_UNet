# === Standard library imports === #
import os
import glob
import gc
import re
import subprocess
import warnings
import time
import random
from collections import defaultdict
from copy import deepcopy
import string

# === Third-party scientific computing === #
import numpy as np
import pandas as pd
import nibabel as nib
import pickle
import scipy.io
import scipy.sparse as sparse
from scipy.io import loadmat
from scipy.stats import (
    ttest_ind, spearmanr, rankdata, norm, skew
)
from scipy.spatial.distance import squareform, pdist

# === Statistical analysis & ML === #
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
import itertools

# === PyTorch & PyG === #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from captum.attr import IntegratedGradients
from torch_geometric.data import Batch, Data as Data_pyg
from torch_geometric.loader import DataLoader as DataLoader_pyg
from torch_geometric.nn import GCNConv, BatchNorm

# === Plotting === #
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns

# === Suppress warnings === #
warnings.filterwarnings('ignore')  # Explanation: suppress PyTorch slicing warning

# ico lvl
ico_levels=[6, 5, 4]
starting_ico = ico_levels[0] # representing starting ico, i.e. highest ico

# Which hemi is first index wise
first = 'rh'

# dir with data
data_dir = '/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/datasets/processed/'

# Where to put outputs
output_dir = '/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/last_model_outputs/'

# Where the pooling data is
pooling_path='/mnt/md0/tempFolder/samAnderson/gnn_model/unet-gnn/pooling/'

# Where cognitive data is
cog_path = '/mnt/md0/tempFolder/samAnderson/datasets/ADNI_cognitive_scores.csv'

# Paths for training data

X_train = f'{data_dir}X_train.npy'
y_train = f'{data_dir}y_train.npy'
sex_train = f'{data_dir}sex_train.npy'

X_5cv = f'{data_dir}X_train_unique_subjects.npy'
y_5cv = f'{data_dir}y_train_unique_subjects.npy'

X_test_CN = f'{data_dir}X_ADNI_CN.npy'
y_test_CN = f'{data_dir}y_ADNI_CN.npy'

X_test_AD = f'{data_dir}X_ADNI_AD.npy'
y_test_AD = f'{data_dir}y_ADNI_AD.npy'

# Training parameters

batch_size = 128
batch_load = 128
n_train_epochs = 50
lr = 0.01 
weight_decay = 0

intra_w = 0.5 # deprecated, ended up using MAE
global_w = 1 # deprecated, ended up using MAE
feature_scale = 1
dropout_levels = [0, 0, 0, 0, 0]

print_every = 25

# Create a dictionary indicating whether higher scores are associated with better performance

test_relations = {
    'ADAS11' : False,            # Alzheimer's Disease Assessment Scale - Cognitive Subscale (11 items)
    'ADAS13' : False,            # Alzheimer's Disease Assessment Scale - 13-item version
    'ADASQ4' : False,            # Subcomponent of ADAS
    'CDRSB' : False,             # Clinical Dementia Rating - Sum of Boxes
    'DIGITSCOR' : True,          # Digit Span (forward/backward) - higher = better
    'EcogPtDivatt' : False,      # ECog Patient: Divided Attention - higher = more impairment
    'EcogPtLang' : False,        # ECog Patient: Language
    'EcogPtMem' : False,         # ECog Patient: Memory
    'EcogPtOrgan' : False,       # ECog Patient: Organization
    'EcogPtPlan' : False,        # ECog Patient: Planning
    'EcogPtVisspat' : False,     # ECog Patient: Visuospatial
    'EcogPtTotal': False,  # ECog Patient Total Score – higher values indicate greater self-reported cognitive impairment
    'EcogSPDivatt' : False,      # ECog Study Partner: Divided Attention
    'EcogSPLang' : False,        # ECog Study Partner: Language
    'EcogSPMem' : False,         # ECog Study Partner: Memory
    'EcogSPOrgan' : False,       # ECog Study Partner: Organization
    'EcogSPPlan' : False,        # ECog Study Partner: Planning
    'EcogSPTotal' : False,       # ECog Study Partner: Total Score
    'EcogSPVisspat' : False,     # ECog Study Partner: Visuospatial
    'FAQ' : False,               # Functional Activities Questionnaire - higher = more impairment
    'LDELTOTAL' : True,          # Logical Memory Delayed Recall - higher = better
    'MMSE' : True,               # Mini-Mental State Examination
    'MOCA' : True,               # Montreal Cognitive Assessment
    'RAVLT_forgetting' : False,  # Rey Auditory Verbal Learning Test - higher = worse retention
    'RAVLT_immediate' : True,    # RAVLT immediate recall
    'RAVLT_learning' : True,     # RAVLT learning score
    'RAVLT_perc_forgetting' : False,  # Percent forgetting - higher = worse
    'TRABSCOR' : False           # Trail Making Test Part B - Score = time, higher = worse
}